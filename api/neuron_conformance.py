"""Independent bounded neuron contract checks and synthetic fixture oracles.

This is API tooling, not the backend execution adapter or a wire parser.
"""
from copy import deepcopy
import math
import struct


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_json(value):
    if isinstance(value, float):
        require(math.isfinite(value), 'non-finite JSON number')
    elif isinstance(value, dict):
        for item in value.values():
            finite_json(item)
    elif isinstance(value, list):
        for item in value:
            finite_json(item)


def limits(value):
    context, vocabulary = value['context_length'], value['vocabulary_size']
    require(value['max_input_tokens'] == min(512, context), 'input limit')
    require(value['max_new_tokens'] == min(64, context - 1), 'generation limit')
    require(value['max_top_k'] == min(20, vocabulary), 'top-k limit')
    require(value['eos_token_ids'] == sorted(set(value['eos_token_ids'])) and
            all(i < vocabulary for i in value['eos_token_ids']), 'EOS identity')


def module(value):
    width, hidden = value['intermediate_size'], value['hidden_size']
    require(width * hidden * 4 <= 2**53 - 1, 'unsafe binding size')
    ids = []
    for name in ('gate', 'up', 'down'):
        binding = value['weights'][name]
        require(binding['axis'] == ('column' if name == 'down' else 'row') and
                binding['shape'] == ([hidden, width] if name == 'down' else [width, hidden]),
                'weight binding geometry')
        ids.append(binding['tensor_id'])
    require(len(set(ids)) == 3, 'distinct projection weights')


def conditions(value):
    entries = value['conditions']
    require(entries[0] == dict(id='baseline', mode='baseline'), 'baseline first')
    require(all(c['mode'] != 'baseline' for c in entries[1:]), 'unique baseline')
    require(len({c['id'] for c in entries}) == len(entries), 'unique condition IDs')


def request(value, bounds, selected):
    require(value['module_id'] == selected['module_id'], 'unknown module')
    require(len(value['token_ids']) <= bounds['max_input_tokens'], 'unsupported_size')
    require(all(i < bounds['vocabulary_size'] for i in value['token_ids']), 'token range')
    require(all(i < selected['intermediate_size'] for i in value['neuron_indices']), 'neuron range')
    if 'conditions' in value:
        conditions(value)
        require(value['top_k'] <= bounds['max_top_k'] and
                value['max_new_tokens'] <= bounds['max_new_tokens'] and
                len(value['token_ids']) + value['max_new_tokens'] <= bounds['context_length'],
                'unsupported_size')


def validate_neuron(schema, value):
    """Cross-field checks, after authoritative JSON Schema validation."""
    finite_json(value)
    if schema == 'NeuronLimits':
        limits(value)
    elif schema == 'NeuronModule':
        module(value)
    elif schema == 'NeuronComparisonRequest':
        conditions(value)
    elif schema == 'NeuronCapabilities':
        available = value['observation']['status'] == 'available'
        require(available or all(value[k]['status'] == 'unavailable'
                                 for k in ('intervention', 'generation')), 'feature dependency')
        if not available:
            require(value['execution'] is None and value['limits'] is None and
                    value['modules'] == [], 'unavailable descriptor')
            return
        require(value['execution'] is not None and value['limits'] is not None and
                len(value['modules']) > 0, 'available descriptor')
        limits(value['limits'])
        for item in value['modules']:
            module(item)
        for key in ('module_id', 'layer_index'):
            require(len({m[key] for m in value['modules']}) == len(value['modules']), 'unique modules')
        require(len({m['module']['name'] for m in value['modules']}) == len(value['modules']),
                'unique module references')
        layers = [m['layer_index'] for m in value['modules']]
        require(layers == sorted(layers), 'layer order')
    elif schema in ('NeuronActivationsMetadata', 'NeuronComparisonMetadata'):
        bounds, selected = value['limits'], value['module']
        limits(bounds)
        module(selected)
        request(dict(value, module_id=selected['module_id']), bounds, selected)
        if schema == 'NeuronActivationsMetadata':
            name = 'activations'
            shape = [len(value['token_ids']), len(value['neuron_indices'])]
        else:
            name = 'probabilities'
            candidates, tops = value['candidate_token_ids'], value['top_token_ids']
            count, k = len(value['conditions']), value['top_k']
            require(candidates == sorted(set(candidates)) and
                    all(i < bounds['vocabulary_size'] for i in candidates), 'candidate order/range')
            require(len(tops) == count and all(len(row) == k for row in tops), 'top-k shape')
            require(candidates == sorted({i for row in tops for i in row}), 'candidate union')
            require(len(value['continuations']) == count, 'continuation count')
            eos = bounds['eos_token_ids']
            maximum = value['max_new_tokens']
            for condition, continuation in zip(value['conditions'], value['continuations']):
                ids = continuation['token_ids']
                require(continuation['condition_id'] == condition['id'], 'continuation order')
                require(len(ids) <= maximum and all(i < bounds['vocabulary_size'] for i in ids),
                        'continuation range/length')
                reason = continuation['finish_reason']
                if maximum == 0:
                    require(reason == 'not_requested' and not ids and continuation['text'] == '',
                            'no generation')
                elif reason == 'eos':
                    require(bool(ids) and ids[-1] in eos and not any(i in eos for i in ids[:-1]),
                            'EOS stopping')
                else:
                    require(reason == 'length' and len(ids) == maximum and
                            not any(i in eos for i in ids), 'length stopping')
            shape = [count, len(candidates)]
        section = value['sections'][0]
        size = 4 * shape[0] * shape[1]
        require(section == dict(name=name, shape=shape, offset=0, byte_length=size) and
                value['byte_length'] == size, 'neuron section geometry')


def validate_payload(metadata, data):
    require(len(data) == metadata['byte_length'], 'payload length')
    values = [v[0] for v in struct.iter_unpack('<f', data)]
    require(all(math.isfinite(v) for v in values), 'numerical_failure')
    if metadata['kind'] == 'neuron_comparison':
        candidates = metadata['candidate_token_ids']
        width = len(candidates)
        for index, top in enumerate(metadata['top_token_ids']):
            row = values[index * width:(index + 1) * width]
            require(all(0 <= p <= 1 for p in row) and sum(row) <= 1 + 1e-6, 'probability range/sum')
            ranked = sorted(zip(candidates, row), key=lambda pair: (-pair[1], pair[0]))
            require(top == [token for token, _ in ranked[:metadata['top_k']]], 'top-k ranking')


def apply_edits(value, edits):
    value = deepcopy(value)
    for path, replacement in edits:
        target = value
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = deepcopy(replacement)
    return value


def fixtures(frame):
    """Literal small oracles; no model or production implementation is consulted."""
    execution = dict(family='smollm2', adapter='smollm2_native', adapter_version='fixture-1',
                     runtime_version='fixture-1', torch_version='fixture', device='cpu', dtype='float32')
    bounds = dict(max_modules_per_request=1, max_neurons=8, max_input_tokens=512,
                  max_new_tokens=64, max_conditions=4, max_top_k=5, context_length=1024,
                  vocabulary_size=5, eos_token_ids=[4])
    selected = dict(module_id='mlp-7', module=dict(kind='module', name='model.layers.7.mlp'),
                    layer_index=7, intermediate_size=3, hidden_size=2,
                    point='post_activation_product_pre_down_proj', weights={
                        'gate': dict(tensor_id='gate7', axis='row', shape=[3, 2]),
                        'up': dict(tensor_id='up7', axis='row', shape=[3, 2]),
                        'down': dict(tensor_id='down7', axis='column', shape=[2, 3])})
    identity = dict(session_id='12345678-1234-1234-1234-123456789abc',
                    model_id='synthetic-codec-fixture', model_revision='fixture_revision')
    cap = dict(identity, observation=dict(status='available'), intervention=dict(status='available'),
               generation=dict(status='available'), execution=execution, limits=bounds, modules=[selected])
    observation = dict(model_revision='fixture_revision', module_id='mlp-7',
                       neuron_indices=[2, 0], token_ids=[1, 1])
    comparison = dict(observation, conditions=[dict(id='baseline', mode='baseline'),
                      dict(id='zero-prompt', mode='zero', scope='prompt')],
                      top_k=2, max_new_tokens=0, decoding='greedy')
    common = dict(identity, execution=execution, limits=bounds, module=selected,
                  token_ids=[1, 1], neuron_indices=[2, 0], dtype='float32', byte_order='little', layout='c')
    activation_meta = dict(common, kind='neuron_activations', byte_length=16,
                           sections=[dict(name='activations', shape=[2, 2], offset=0, byte_length=16)])
    comparison_meta = dict(common, **{k: comparison[k] for k in ('conditions', 'top_k', 'max_new_tokens', 'decoding')},
                           kind='neuron_comparison', byte_length=24,
                           sections=[dict(name='probabilities', shape=[2, 3], offset=0, byte_length=24)],
                           candidate_token_ids=[0, 1, 2], top_token_ids=[[1, 2], [0, 2]],
                           continuations=[dict(condition_id=c['id'], token_ids=[], text='', finish_reason='not_requested')
                                          for c in comparison['conditions']])
    bases = dict(capabilities=cap, observation=observation, comparison=comparison,
                 activations=activation_meta, probabilities=comparison_meta)
    schemas = dict(capabilities='NeuronCapabilities', observation='NeuronActivationsRequest',
                   comparison='NeuronComparisonRequest', activations='NeuronActivationsMetadata',
                   probabilities='NeuronComparisonMetadata')
    cases = [dict(name=name, base=name, schema=schemas[name], valid=True, edits=[])
             for name in bases]

    def case(name, base, path, replacement, valid=False):
        cases.append(dict(name=name, base=base, schema=schemas[base], valid=valid,
                          edits=[[path, replacement]]))

    case('reordered-neurons-valid', 'observation', ['neuron_indices'], [0, 2], True)
    case('maximum-prompt', 'observation', ['token_ids'], [0]*512, True)
    case('one-token', 'observation', ['token_ids'], [0], True)
    case('maximum-generation', 'comparison', ['max_new_tokens'], 64, True)
    case('vocabulary-top-k', 'comparison', ['top_k'], 5, True)
    case('baseline-only', 'comparison', ['conditions'], [comparison['conditions'][0]], True)
    case('four-conditions', 'comparison', ['conditions'], [*comparison['conditions'],
         dict(id='scale', mode='scale', scope='all', factor=-2),
         dict(id='add', mode='add', scope='continuation', delta=1)], True)
    unavailable_feature = dict(status='unavailable', reason=dict(code='runtime_unavailable',
                               message='Capability unavailable.', recovery='Use an admitted runtime.'))
    for feature, edits, valid in [
        ('intervention', [], False),
        ('intervention', [[['conditions'], [comparison['conditions'][0]]]], True),
        ('generation', [[['max_new_tokens'], 1]], False),
        ('generation', [], True),
    ]:
        cases.append(dict(name=f'{feature}-availability-{valid}', base='comparison', schema=schemas['comparison'],
                          valid=valid, edits=edits, context_edits=[[[feature], unavailable_feature]]))
    cases.append(dict(name='eight-neurons', base='observation', schema=schemas['observation'], valid=True,
                      edits=[[['neuron_indices'], list(range(8))]], context_edits=[
                          [['modules', 0, 'intermediate_size'], 8],
                          [['modules', 0, 'weights', 'gate', 'shape'], [8, 2]],
                          [['modules', 0, 'weights', 'up', 'shape'], [8, 2]],
                          [['modules', 0, 'weights', 'down', 'shape'], [2, 8]]]))
    for name, path, replacement in [
        ('empty-prompt', ['token_ids'], []), ('too-many-tokens', ['token_ids'], [0]*513),
        ('negative-token', ['token_ids'], [-1]), ('token-exclusive-upper', ['token_ids'], [5]),
        ('boolean-token', ['token_ids'], [True]), ('unsafe-token', ['token_ids'], [2**53]),
        ('empty-neurons', ['neuron_indices'], []), ('duplicate-neurons', ['neuron_indices'], [2, 2]),
        ('neuron-exclusive-upper', ['neuron_indices'], [3]),
        ('too-many-neurons', ['neuron_indices'], list(range(9))),
        ('unknown-module', ['module_id'], 'mlp-8'), ('stale-revision', ['model_revision'], 'other')]:
        case(name, 'observation', path, replacement)
    for name, path, replacement in [
        ('baseline-not-first', ['conditions'], list(reversed(comparison['conditions']))),
        ('duplicate-baseline', ['conditions'], [comparison['conditions'][0]]*2),
        ('duplicate-condition-id', ['conditions'], [*comparison['conditions'], comparison['conditions'][1]]),
        ('too-many-conditions', ['conditions'], [comparison['conditions'][0]]*5),
        ('no-baseline', ['conditions'], [comparison['conditions'][1]]),
        ('baseline-mutation', ['conditions', 0], dict(id='baseline', mode='baseline', factor=2)),
        ('mixed-intervention', ['conditions', 1], dict(id='s', mode='scale', scope='all', factor=2, delta=1)),
        ('missing-factor', ['conditions', 1], dict(id='s', mode='scale', scope='all')),
        ('float32-overflow-factor', ['conditions', 1], dict(id='s', mode='scale', scope='all', factor=1e39)),
        ('invalid-scope', ['conditions', 1, 'scope'], 'generated'),
        ('negative-generation', ['max_new_tokens'], -1), ('too-many-new-tokens', ['max_new_tokens'], 65),
        ('zero-top-k', ['top_k'], 0), ('too-large-top-k', ['top_k'], 21),
        ('top-k-vocabulary', ['top_k'], 6), ('sampling', ['decoding'], 'sample')]:
        case(name, 'comparison', path, replacement)
    for mode, param, number in [('scale', 'factor', -2), ('scale', 'factor', 0), ('add', 'delta', -1)]:
        for scope in ('prompt', 'continuation', 'all'):
            case(f'{mode}-{number}-{scope}', 'comparison', ['conditions', 1],
                 dict(id='changed', mode=mode, scope=scope, **{param: number}), True)
    for name, base, path, replacement in [
        ('activation-shape', 'activations', ['sections', 0, 'shape'], [1, 4]),
        ('activation-offset', 'activations', ['sections', 0, 'offset'], 4),
        ('activation-section-bytes', 'activations', ['sections', 0, 'byte_length'], 12),
        ('activation-total-bytes', 'activations', ['byte_length'], 12),
        ('activation-section-name', 'activations', ['sections', 0, 'name'], 'probabilities'),
        ('candidate-order', 'probabilities', ['candidate_token_ids'], [1, 0, 2]),
        ('candidate-range', 'probabilities', ['candidate_token_ids'], [0, 1, 5]),
        ('candidate-union', 'probabilities', ['candidate_token_ids'], [0, 1, 3]),
        ('top-k-count', 'probabilities', ['top_token_ids', 0], [1]),
        ('top-k-duplicate', 'probabilities', ['top_token_ids', 0], [1, 1]),
        ('continuation-order', 'probabilities', ['continuations'], list(reversed(comparison_meta['continuations']))),
        ('continuation-range', 'probabilities', ['continuations', 0, 'token_ids'], [5]),
        ('no-generation-reason', 'probabilities', ['continuations', 0, 'finish_reason'], 'length'),
        ('comparison-shape', 'probabilities', ['sections', 0, 'shape'], [3, 2]),
        ('wrong-weight-axis', 'capabilities', ['modules', 0, 'weights', 'down', 'axis'], 'row'),
        ('wrong-weight-shape', 'capabilities', ['modules', 0, 'weights', 'gate', 'shape'], [2, 3]),
        ('duplicated-module', 'capabilities', ['modules'], [selected, selected]),
        ('wrong-effective-limit', 'capabilities', ['limits', 'max_input_tokens'], 513),
        ('EOS-range', 'capabilities', ['limits', 'eos_token_ids'], [5]),
        ('unavailable-with-modules', 'capabilities', ['observation'],
         dict(status='unavailable', reason=dict(code='unsupported_model', message='Unsupported', recovery='Select native SmolLM2 Base'))),
    ]:
        case(name, base, path, replacement)
    # Source-dependent context boundary with a matching effective descriptor.
    for length, valid in [(4, True), (5, False)]:
        cases.append(dict(name=f'context-total-{length}', base='comparison', schema=schemas['comparison'],
                          valid=valid, edits=[[['max_new_tokens'], length]],
                          context_edits=[[['limits', 'context_length'], 6],
                                         [['limits', 'max_input_tokens'], 6], [['limits', 'max_new_tokens'], 5]]))
    for ids, reason, valid in [([2, 4], 'eos', True), ([2, 3], 'length', True),
                               ([4, 2], 'eos', False), ([2], 'length', False)]:
        cases.append(dict(name=f'stopping-{ids}-{reason}', base='probabilities', schema=schemas['probabilities'],
                          valid=valid, edits=[[['max_new_tokens'], 2], [['continuations'],
                              [dict(condition_id=c['id'], token_ids=ids, text='fixture', finish_reason=reason)
                               for c in comparison['conditions']]]]))
    unavailable = deepcopy(cap)
    for feature in ('observation', 'intervention', 'generation'):
        unavailable[feature] = dict(status='unavailable', reason=dict(code='unsupported_model',
                                    message='Unsupported model.', recovery='Select local native SmolLM2 Base.'))
    unavailable.update(execution=None, limits=None, modules=[])
    bases['unavailable'] = unavailable
    cases.append(dict(name='unavailable', base='unavailable', schema='NeuronCapabilities', valid=True, edits=[]))
    # Deliberately asymmetric signed values and non-normalized selected probabilities.
    numeric = dict(activations=[-1.0, 2.0, 0.0, -3.0], probabilities=[0.125, 0.5, 0.25, 0.5, 0.125, 0.25],
                   full_vocabulary_probabilities=[[0.125, 0.5, 0.25, 0.0625, 0.0625],
                                                  [0.5, 0.125, 0.25, 0.0625, 0.0625]],
                   probability_deltas=[0.375, -0.375, 0.0], percentage_point_deltas=[37.5, -37.5, 0.0])
    payloads = {key: struct.pack('<'+str(len(numeric[key]))+'f', *numeric[key]).hex()
                for key in ('activations', 'probabilities')}
    payload_cases = [dict(name=key, base=key, data_hex=value, valid=True) for key, value in payloads.items()]
    for name, base, data in [('nan-activation', 'activations', bytes.fromhex('0000c07f') + bytes.fromhex(payloads['activations'])[4:]),
                             ('infinite-activation', 'activations', bytes.fromhex('0000807f') + bytes.fromhex(payloads['activations'])[4:]),
                             ('probability-above-one', 'probabilities', struct.pack('<6f', 2, .5, .25, .5, .125, .25)),
                             ('negative-probability', 'probabilities', struct.pack('<6f', -.1, .5, .25, .5, .125, .25)),
                             ('probability-sum', 'probabilities', struct.pack('<6f', .25, .75, .5, .5, .125, .25)),
                             ('top-order', 'probabilities', struct.pack('<6f', .5, .125, .25, .5, .125, .25))]:
        payload_cases.append(dict(name=name, base=base, data_hex=data.hex(), valid=False))
    wire = []
    for key in ('activations', 'probabilities'):
        meta, data = bases[key], bytes.fromhex(payloads[key])
        prefix = frame(1, meta) + frame(2, data)
        wire.append(dict(name='neuron-'+key, wire_hex=(prefix+frame(4)).hex(),
                         expected=dict(outcome='complete', metadata=meta, data_hex=data.hex())))
        wire.append(dict(name='neuron-'+key+'-missing-terminal', wire_hex=prefix.hex(), expected=dict(outcome='reject')))
    # Common parser coverage owns the full framing matrix; only representative new-kind terminals here.
    partial = frame(1, activation_meta) + frame(2, bytes.fromhex(payloads['activations'])[:4])
    wire.append(dict(name='neuron-short-complete', wire_hex=(partial+frame(4)).hex(), expected=dict(outcome='reject')))
    wire.append(dict(name='neuron-partial-cancel', wire_hex=(partial+frame(6)).hex(), expected=dict(outcome='cancelled')))
    # Contextual response association vectors are adopted by the execution clients.
    associations = [dict(field='token_ids', value=[1, 1], valid=True)]
    for field, value in [('session_id', '22345678-1234-1234-1234-123456789abc'),
                         ('model_id', 'different'), ('model_revision', 'other'), ('token_ids', [1, 2]),
                         ('neuron_indices', [0, 2]), ('module', dict(selected, module_id='mlp-8')),
                         ('conditions', list(reversed(comparison['conditions']))),
                         ('top_k', 1), ('max_new_tokens', 1)]:
        associations.append(dict(field=field, value=value, valid=False))
    return dict(bases=bases, cases=cases, numeric=numeric, payload_cases=payload_cases,
                association_cases=associations, wire_cases=wire)
