"""Independent structural/contextual oracles for the published architecture contract.

No application imports, graph execution, checkpoint access, or HTTP implementation.
Consumers must combine generated schema validation with these semantic cases.
"""
from copy import deepcopy
import json

LIMIT = 33_554_432
SAFE = 2**53 - 1


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bounded_size(chunks, limit=LIMIT):
    """Count a stream of UTF-8 chunks, stopping before retaining oversized content."""
    size = 0
    for chunk in chunks:
        size += len(chunk)
        require(size <= limit, 'unsupported_size')
    return size


def serialized_size(value):
    # Schema bounds limit individual strings; iterencode avoids a whole-response copy.
    encoder = json.JSONEncoder(ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    return bounded_size(part.encode('utf-8') for part in encoder.iterencode(value))


def product(shape):
    require(all(type(d) is int and 0 <= d <= SAFE for d in shape), 'unsafe dimension')
    if 0 in shape:
        return 0
    result = 1
    for dimension in shape:
        result *= dimension
        require(result <= SAFE, 'unsafe product')
    return result


def unique(records, field='id'):
    result = {r[field]: r for r in records}
    require(len(result) == len(records), 'duplicate '+field)
    return result


def validate_architecture(value, context=None):
    """Run after generated JSON Schema validation. Context pins model/inventory/tokenizer."""
    serialized_size(value)
    if context is not None:
        require(value['model_id'] == context['session']['model_id'], 'session model mismatch')
    if value['status'] == 'unavailable':
        require(all('node_id' not in d and 'parameter_id' not in d for d in value['diagnostics']),
                'unavailable diagnostic has no graph')
        if value['reason'] in ('restart_required', 'cache_unavailable'):
            require(value['requires_restart'], 'restart required')
        return
    graph = value['graph']
    nodes = unique(graph['nodes'])
    params = unique(graph['parameters'])
    unique(graph['edges'])
    unique(graph['repetitions'])
    unique(graph['nodes'] + graph['edges'] + graph['repetitions'] + graph['parameters'])
    symbols = unique(graph['symbols'], 'name')
    ports = {n['id']: unique(n['ports']) for n in nodes.values()}

    def diagnostic(d):
        require('node_id' not in d or d['node_id'] in nodes, 'diagnostic node closure')
        require('parameter_id' not in d or d['parameter_id'] in params, 'diagnostic parameter closure')

    for d in graph['diagnostics'] + value['diagnostics']:
        diagnostic(d)

    def shape(dims):
        if dims is None:
            return
        for d in dims:
            if d['kind'] == 'symbol':
                require(d['name'] in symbols, 'undeclared symbol')
            if d['kind'] == 'expression':
                require(all(s in symbols for s in d['symbols']), 'undeclared expression symbol')
        if all(d['kind'] == 'constant' for d in dims):
            product([d['value'] for d in dims])

    for n in nodes.values():
        if 'parent_id' in n:
            parent = nodes.get(n['parent_id'])
            require(parent is not None and parent['kind'] == 'group', 'parent must be group')
            require(n['id'] in parent['children'], 'parent/child disagreement')
        if n['kind'] == 'group':
            require(len(set(n['children'])) == len(n['children']), 'duplicate child')
            for child in n['children']:
                require(child in nodes and nodes[child].get('parent_id') == n['id'],
                        'child/parent disagreement')
        require(all(p in params for p in n['parameter_ids']), 'node parameter closure')
        for r in n['references']:
            if r['kind'] == 'parameter':
                require(r['parameter_id'] in params, 'resource parameter closure')
            if r['kind'] == 'tokenizer' and context is not None:
                require(context['tokenizer_available'], 'missing tokenizer capability')
        for p in n['ports']:
            shape(p['shape'])

    # Iterative walks also accept deeply nested but bounded valid graphs.
    def terminate(records, link):
        done = set()
        for key in records:
            path = set()
            while key not in done:
                require(key not in path, 'cyclic '+link)
                path.add(key)
                require(key in records, 'unresolved '+link)
                nxt = records[key].get(link)
                if nxt is None:
                    break
                key = nxt
            done.update(path)

    terminate(nodes, 'parent_id')
    terminate(params, 'alias_of')
    for r in graph['repetitions']:
        parent = nodes.get(r['parent_id'])
        require(parent is not None and parent['kind'] == 'group', 'repetition parent')
        unique(r['instances'], 'node_id')
        indices = [i['index'] for i in r['instances']]
        require(indices == sorted(set(indices)), 'repetition index order')
        positions = []
        for i in r['instances']:
            n = nodes.get(i['node_id'])
            require(n is not None and n['kind'] == 'group' and n.get('parent_id') == parent['id'],
                    'repetition group membership')
            positions.append(parent['children'].index(n['id']))
        require(positions == sorted(positions), 'repetition parent order')

    for e in graph['edges']:
        endpoints = []
        for key in ('source', 'target'):
            ep = e[key]
            require(ep['node_id'] in ports and ep['port_id'] in ports[ep['node_id']],
                    'edge endpoint closure')
            endpoints.append(ports[ep['node_id']][ep['port_id']])
        source, target = endpoints
        sn, tn = nodes[e['source']['node_id']], nodes[e['target']['node_id']]
        # Group input forwards to its children; child output reaches group output.
        compatible = source['direction'] == 'output' and target['direction'] == 'input'
        compatible |= (sn['kind'] == 'group' and tn.get('parent_id') == sn['id']
                       and source['direction'] == target['direction'] == 'input')
        compatible |= (tn['kind'] == 'group' and sn.get('parent_id') == tn['id']
                       and source['direction'] == target['direction'] == 'output')
        require(compatible, 'edge directions')
        # Edges cross containment through explicit group ports, not hidden descendants.
        require(sn.get('parent_id') == tn.get('parent_id') or
                tn.get('parent_id') == sn['id'] or sn.get('parent_id') == tn['id'],
                'edge bypasses group boundary')
        a, b = source['shape'], target['shape']
        mismatch = a is not None and b is not None and (
            len(a) != len(b) or any(x['kind'] == y['kind'] == 'constant' and x['value'] != y['value']
                                   for x, y in zip(a, b)))
        if mismatch:
            require(any(d.get('node_id') in (sn['id'], tn['id'])
                        for d in graph['diagnostics'] + value['diagnostics']),
                    'connected dimension disagreement needs diagnostic')

    inventory = unique(context['inventory']['tensors']) if context is not None else None
    for p in params.values():
        shape(p['logical_shape'])
        for storage in p['storage']:
            product(storage['shape'])
        if p['binding'] == 'fused_region':
            require(p['region']['storage_name'] in {s['name'] for s in p['storage']},
                    'region storage closure')
        inspection = p['inspection']
        if inspection['status'] == 'available':
            require(p['binding'] in ('native', 'alias'), 'non-native inspection')
            dims = p['logical_shape']
            require(dims is not None and len(dims) in (1, 2) and
                    all(d['kind'] == 'constant' for d in dims), 'inspection logical geometry')
            geometry = [d['value'] for d in dims]
            native = p
            while native['binding'] == 'alias':
                native = params[native['alias_of']]
            require(native['binding'] == 'native' and native['logical_shape'] == dims and
                    native['inspection']['status'] == 'available' and
                    native['inspection']['tensor_id'] == inspection['tensor_id'],
                    'alias native identity/geometry')
            require(any(s['name'] == native['name'] and s['shape'] == geometry and
                        s['dtype'] in ('F32', 'F16', 'BF16', 'float32', 'float16', 'bfloat16') and
                        s.get('role') not in ('scales', 'packed', 'packed_data')
                        for s in native['storage']), 'native storage geometry')
            if inventory is not None:
                tensor = inventory.get(inspection['tensor_id'])
                require(tensor is not None and tensor['shape'] == geometry and
                        tensor['name'] == native['name'] and tensor['rank'] == len(geometry) and
                        tensor['numel'] == product(geometry), 'session inventory membership/geometry')
                require(any(s['name'] == tensor['name'] and s['dtype'] == tensor['storage_dtype']
                            for s in native['storage']), 'inventory storage identity')


def fixtures():
    """Compact base documents plus mutation cases; never repeat a full graph per defect."""
    dim = lambda n: {'kind': 'constant', 'value': n}
    port = lambda ident, direction, dims: dict(id=ident, direction=direction, label=ident, shape=dims)
    def node(ident, kind, **extra):
        return dict(id=ident, kind=kind, label=ident, ports=[], parameter_ids=[], references=[],
                    attributes=[], provenance=[], **extra)
    def unavailable(reason):
        return dict(status='unavailable', reason=reason, message='Numeric inspection unavailable.')
    storage = dict(name='linear.weight', dtype='F16', shape=[2, 3])
    native = dict(id='weight', name='linear.weight', logical_shape=[dim(2), dim(3)], binding='native',
                  storage=[storage], inspection=dict(status='available', tensor_id='tensor_native'), provenance=[])
    parameters = [native,
        {**deepcopy(native), 'id': 'alias', 'name': 'tied.weight', 'binding': 'alias', 'alias_of': 'weight'},
        dict(id='quantized', name='packed.weight', logical_shape=[dim(8), dim(8)], binding='quantized',
             storage=[dict(name='packed.qweight', dtype='I32', shape=[1, 8], role='packed_data'),
                      dict(name='packed.scales', dtype='F16', shape=[1, 8], role='scales')],
             inspection=unavailable('unsupported_representation'), provenance=[]),
        dict(id='fused', name='q.weight', logical_shape=[dim(2), dim(3)], binding='fused_region',
             storage=[dict(name='qkv.weight', dtype='F16', shape=[6, 3])],
             region=dict(storage_name='qkv.weight', description='First two rows along output axis.'),
             inspection=unavailable('requires_view'), provenance=[]),
        dict(id='unresolved', name='unknown.weight', logical_shape=None, binding='unresolved', storage=[],
             inspection=unavailable('unresolved_binding'), provenance=[]),
        dict(id='volume', name='patch.weight', logical_shape=[dim(2), dim(3), dim(4)], binding='native',
             storage=[dict(name='patch.weight', dtype='F32', shape=[2, 3, 4])],
             inspection=unavailable('unsupported_rank'), provenance=[])]
    nodes = [node('root', 'group', children=['layer0', 'layer1', 'tokens']),
             node('layer0', 'group', parent_id='root', children=['linear0']),
             node('layer1', 'group', parent_id='root', children=['linear1']),
             node('linear0', 'operation', parent_id='layer0', operation='linear'),
             node('linear1', 'operation', parent_id='layer1', operation='linear'),
             node('tokens', 'context', parent_id='root')]
    nodes[1]['ports'] = [port('in', 'input', [dim(2)]), port('out', 'output', [dim(2)])]
    nodes[2]['ports'] = deepcopy(nodes[1]['ports'])
    for n in nodes[3:5]:
        n['ports'] = deepcopy(nodes[1]['ports'])
        n['parameter_ids'] = ['weight', 'alias']
        n['references'] = [dict(kind='parameter', parameter_id='weight'), dict(kind='module', name='linear')]
    nodes[5]['references'] = [dict(kind='tokenizer')]
    nodes[5]['ports'] = [port('symbols', 'output', [dict(kind='symbol', name='B'),
        dict(kind='expression', text='T + 1', symbols=['T']), dict(kind='unknown', reason='Static rank known only.')]),
        port('scalar', 'output', []), port('unknown_rank', 'output', None)]
    nodes[5]['attributes'] = [dict(name='heads', value=2, provenance=[dict(kind='configuration', source='/num_heads')]),
        dict(name='notes', value=['display only', True, None], provenance=[dict(kind='description', source='reviewed_rule')])]
    parameters[0]['provenance'] = [dict(kind='storage', source='linear.weight')]
    def edge(ident, sn, sp, tn, tp):
        return dict(id=ident, source=dict(node_id=sn, port_id=sp), target=dict(node_id=tn, port_id=tp), kind='data', provenance=[])
    graph = dict(graph_id='graph_synthetic', scope='language_model', coverage='complete',
        symbols=[dict(name='B', meaning='batch'), dict(name='T', meaning='token sequence')], nodes=nodes,
        edges=[edge('between', 'layer0', 'out', 'layer1', 'in'), edge('enter', 'layer0', 'in', 'linear0', 'in'),
               edge('leave', 'linear0', 'out', 'layer0', 'out')],
        repetitions=[dict(id='layers', parent_id='root', label='Layers', instances=[
            dict(node_id='layer0', index=0, variant='attention'), dict(node_id='layer1', index=1, variant='state_space')])],
        parameters=parameters, diagnostics=[])
    response = dict(status='available', model_id='test/architecture', diagnostics=[], graph=graph)
    inventory = dict(coverage='complete', diagnostics=[], tensors=[dict(id='tensor_native', name='linear.weight',
        path=['linear', 'weight'], shape=[2, 3], rank=2, numel=6, storage_dtype='F16', logical_dtype='float32'),
        dict(id='tensor_volume', name='patch.weight', path=['patch', 'weight'], shape=[2, 3, 4], rank=3,
             numel=24, storage_dtype='F32', logical_dtype='float32')])
    context = dict(session=dict(id='12345678-1234-4234-8234-123456789abc', model_id=response['model_id']),
                   tokenizer_available=True, inventory=inventory)
    cases = []
    def case(name, edits=(), valid=False, schema_valid=True, context_edits=()):
        cases.append(dict(name=name, valid=valid, schema_valid=schema_valid, edits=list(edits), context_edits=list(context_edits)))
    def set_(path, value): return dict(path=path.split('/'), value=value)
    def delete(path): return dict(path=path.split('/'), delete=True)
    case('complete-native-alias-fused-quantized-rank-limited-symbolic', valid=True)
    case('url-safe-opaque-id', [set_('graph/graph_id', 'analysis.1~revision')], True)
    case('partial-architecture', [set_('graph/coverage', 'partial')], True)
    case('visual-no-tokenizer', [set_('graph/scope', 'visual_encoder_predictor'), set_('graph/nodes/5/references', [])], True,
         context_edits=[set_('tokenizer_available', False)])
    case('explicit-partial-inventory', valid=True, context_edits=[set_('inventory/coverage', 'partial'),
        set_('inventory/diagnostics', [dict(code='excluded_quantized', message='Packed parameters remain descriptive.')])])
    for reason in ('unsupported_architecture', 'analysis_failed', 'restart_required', 'unsupported_size', 'cache_unavailable'):
        case('unavailable-'+reason, [set_('status','unavailable'), delete('graph'), set_('reason',reason), set_('requires_restart',True)], True)
    for kind in ('input', 'output', 'state'):
        case('node-kind-'+kind, [set_('graph/nodes/5/kind', kind)], True)
    for kind in ('state', 'context'):
        case('edge-kind-'+kind, [set_('graph/edges/0/kind', kind)], True)
    case('rank1-native', [set_('graph/parameters/0/logical_shape', [dim(6)]),
        set_('graph/parameters/1/logical_shape', [dim(6)]), set_('graph/parameters/0/storage/0/shape', [6])], True,
        context_edits=[set_('inventory/tensors/0/shape', [6]), set_('inventory/tensors/0/rank', 1)])
    case('empty-partial-inventory', [set_('graph/parameters', []),
        set_('graph/nodes/3/parameter_ids', []), set_('graph/nodes/4/parameter_ids', []),
        set_('graph/nodes/3/references', []), set_('graph/nodes/4/references', [])], True,
        context_edits=[set_('inventory/tensors', []), set_('inventory/coverage', 'partial')])
    case('display-text-is-inert', [set_('graph/nodes/5/formula', '<script>alert(1)</script>'),
        set_('graph/nodes/5/ports/0/shape/1/text', '__import__("os").system("false")')], True)
    case('zero-product', [set_('graph/parameters/2/logical_shape', [dim(SAFE), dim(SAFE), dim(0)]),
        set_('graph/parameters/2/storage/0/shape', [SAFE, SAFE, 0])], True)
    # Schema-local closed unions and field limits.
    for name, edits in [
        ('unavailable-unknown-reason', [set_('status','unavailable'), delete('graph'), set_('reason','pending'), set_('requires_restart',False)]),
        ('unavailable-missing-restart', [set_('status','unavailable'), delete('graph'), set_('reason','analysis_failed')]),
        ('unknown-binding', [set_('graph/parameters/0/binding','decoder')]),
        ('unknown-node-kind', [set_('graph/nodes/5/kind','execute')]),
        ('unknown-edge-kind', [set_('graph/edges/0/kind','execute')]),
        ('unknown-provenance-kind', [set_('graph/parameters/0/provenance/0/kind','trace')]),
        ('unknown-direction', [set_('graph/nodes/1/ports/0/direction','both')]),
        ('module-without-name', [delete('graph/nodes/3/references/1/name')]),
        ('parameter-reference-without-id', [delete('graph/nodes/3/references/0/parameter_id')]),
        ('unknown-response', [set_('status','pending')]),
        ('available-with-reason', [set_('reason','analysis_failed')]),
        ('available-without-graph', [delete('graph')]),
        ('unavailable-with-graph', [set_('status','unavailable'), set_('reason','analysis_failed'), set_('requires_restart',False)]),
        ('unknown-dimension', [set_('graph/nodes/5/ports/0/shape/0/kind','runtime')]),
        ('constant-missing-value', [delete('graph/parameters/0/logical_shape/0/value')]),
        ('constant-unsafe', [set_('graph/parameters/0/logical_shape/0/value', SAFE+1)]),
        ('constant-negative', [set_('graph/parameters/0/logical_shape/0/value', -1)]),
        ('constant-boolean', [set_('graph/parameters/0/logical_shape/0/value', True)]),
        ('symbol-missing-name', [delete('graph/nodes/5/ports/0/shape/0/name')]),
        ('expression-missing-symbols', [delete('graph/nodes/5/ports/0/shape/1/symbols')]),
        ('unknown-missing-reason', [delete('graph/nodes/5/ports/0/shape/2/reason')]),
        ('dimension-mixed-union', [set_('graph/nodes/5/ports/0/shape/0/value', 2)]),
        ('unknown-inspection', [set_('graph/parameters/0/inspection/status','pending')]),
        ('available-missing-tensor', [delete('graph/parameters/0/inspection/tensor_id')]),
        ('unavailable-actionable-id', [set_('graph/parameters/2/inspection/tensor_id','tensor_native')]),
        ('unknown-inspection-reason', [set_('graph/parameters/2/inspection/reason','decode_now')]),
        ('unknown-reference', [set_('graph/nodes/5/references/0/kind','url')]),
        ('cross-model-reference', [set_('graph/nodes/5/references/0/model_id','other/model')]),
        ('reference-route', [set_('graph/nodes/5/references/0/url','/tokenize')]),
        ('route-as-id', [set_('graph/nodes/0/id','../sessions/other')]),
        ('executable-as-id', [set_('graph/nodes/0/id','javascript:alert(1)')]),
        ('foreign-graph-field', [set_('graph/model_id','other/model')]),
        ('node-runtime-operation', [set_('graph/nodes/0/operation_id','runtime')]),
        ('empty-nodes', [set_('graph/nodes',[])]),
        ('leaf-with-children', [set_('graph/nodes/3/children',[])]),
        ('group-without-children', [delete('graph/nodes/0/children')]),
        ('alias-without-target', [delete('graph/parameters/1/alias_of')]),
        ('fused-without-region', [delete('graph/parameters/3/region')]),
        ('native-with-alias', [set_('graph/parameters/0/alias_of','alias')]),
        ('nested-attribute-payload', [set_('graph/nodes/5/attributes/0/value', [[1,2],[3,4]])]),
        ('object-attribute', [set_('graph/nodes/5/attributes/0/value', {'code':'run()'})]),
        ('edge-duplicate-shape', [set_('graph/edges/0/shape',[2])]),
        ('id-too-long', [set_('graph/graph_id','x'*257)]),
        ('label-too-long', [set_('graph/nodes/0/label','x'*1025)]),
        ('text-too-long', [set_('graph/nodes/0/description','x'*16385)]),
    ]:
        case(name, edits, schema_valid=False)
    for name, edits in [
        ('cache-unavailable-needs-restart', [set_('status','unavailable'), delete('graph'), set_('reason','cache_unavailable'), set_('requires_restart',False)]),
        ('restart-required-needs-restart', [set_('status','unavailable'), delete('graph'), set_('reason','restart_required'), set_('requires_restart',False)]),
        ('unavailable-diagnostic-target', [set_('status','unavailable'), delete('graph'), set_('reason','analysis_failed'), set_('requires_restart',True), set_('diagnostics',[dict(code='failure',message='Analysis failed.',node_id='root')])]),
        ('cross-kind-id-duplicate', [set_('graph/edges/0/id','root')]),
        ('duplicate-repetition', [set_('graph/repetitions', [graph['repetitions'][0], graph['repetitions'][0]])]),
        ('duplicate-node', [set_('graph/nodes/1/id','root')]),
        ('duplicate-port', [set_('graph/nodes/1/ports/1/id','in')]),
        ('duplicate-edge', [set_('graph/edges/1/id','between')]),
        ('duplicate-parameter', [set_('graph/parameters/1/id','weight')]),
        ('duplicate-symbol', [set_('graph/symbols/1/name','B')]),
        ('missing-parent', [set_('graph/nodes/1/parent_id','foreign')]),
        ('parent-not-group', [set_('graph/nodes/1/parent_id','linear0')]),
        ('missing-child', [set_('graph/nodes/0/children/0','foreign')]),
        ('parent-child-disagreement', [set_('graph/nodes/0/children', ['layer1','tokens'])]),
        ('duplicate-child', [set_('graph/nodes/0/children', ['layer0','layer0','layer1','tokens'])]),
        ('containment-cycle', [set_('graph/nodes/0/parent_id','layer0'), set_('graph/nodes/1/children',['linear0','root'])]),
        ('repetition-not-group', [set_('graph/repetitions/0/instances/0/node_id','linear0')]),
        ('repetition-wrong-parent', [set_('graph/repetitions/0/parent_id','layer0')]),
        ('repetition-out-of-order', [set_('graph/repetitions/0/instances/0/node_id','layer1'), set_('graph/repetitions/0/instances/1/node_id','layer0')]),
        ('repetition-index-order', [set_('graph/repetitions/0/instances/0/index',2)]),
        ('repetition-duplicate', [set_('graph/repetitions/0/instances/1/node_id','layer0')]),
        ('alias-cycle', [set_('graph/parameters/1/alias_of','alias')]),
        ('alias-missing-target', [set_('graph/parameters/1/alias_of','foreign')]),
        ('alias-wrong-geometry', [set_('graph/parameters/1/logical_shape/0/value',3)]),
        ('alias-wrong-identity', [set_('graph/parameters/1/inspection/tensor_id','tensor_volume')]),
        ('node-parameter-closure', [set_('graph/nodes/3/parameter_ids/0','foreign')]),
        ('reference-parameter-closure', [set_('graph/nodes/3/references/0/parameter_id','foreign')]),
        ('diagnostic-closure', [set_('diagnostics',[dict(code='unknown',message='Unknown node.',node_id='foreign')])]),
        ('edge-node-closure', [set_('graph/edges/0/source/node_id','foreign')]),
        ('edge-port-closure', [set_('graph/edges/0/source/port_id','foreign')]),
        ('edge-direction', [set_('graph/edges/0/source/port_id','in')]),
        ('edge-skips-group', [set_('graph/edges/0/source/node_id','linear0')]),
        ('connected-dimension-mismatch', [set_('graph/nodes/2/ports/0/shape/0/value',3)]),
        ('unknown-symbol', [set_('graph/nodes/5/ports/0/shape/0/name','X')]),
        ('unknown-expression-symbol', [set_('graph/nodes/5/ports/0/shape/1/symbols',['X'])]),
        ('unsafe-logical-product', [set_('graph/parameters/2/logical_shape',[dim(SAFE),dim(2)])]),
        ('unsafe-physical-product', [set_('graph/parameters/2/storage/0/shape',[SAFE,2])]),
        ('physical-not-logical-geometry', [set_('graph/parameters/0/storage/0/shape',[1,6])]),
        ('packed-cannot-be-inspected', [set_('graph/parameters/2/inspection',dict(status='available',tensor_id='tensor_native'))]),
        ('fused-cannot-be-inspected', [set_('graph/parameters/3/inspection',dict(status='available',tensor_id='tensor_native'))]),
        ('rank3-cannot-be-inspected', [set_('graph/parameters/5/inspection',dict(status='available',tensor_id='tensor_volume'))]),
        ('region-storage-closure', [set_('graph/parameters/3/region/storage_name','foreign')]),
        ('missing-inventory-tensor', [set_('graph/parameters/0/inspection/tensor_id','foreign')]),
    ]:
        case(name, edits)
    case('dimension-mismatch-diagnosed', [set_('graph/nodes/2/ports/0/shape/0/value',3),
        set_('graph/diagnostics',[dict(code='shape_mismatch',message='Known dimensions disagree.',node_id='layer1')])], True)
    case('cross-session-model', context_edits=[set_('session/model_id','other/model')])
    case('no-tokenizer-reference', context_edits=[set_('tokenizer_available',False)])
    case('wrong-inventory-geometry', context_edits=[set_('inventory/tensors/0/shape',[3,2])])
    case('wrong-inventory-identity', context_edits=[set_('inventory/tensors/0/name','other.weight')])
    # Large-size tests use repeated fixed chunks, never a 32 MiB fixture/object allocation.
    bounds = [dict(name='exact-limit', chunk_bytes=4096, repeat=8192, tail_bytes=0, valid=True),
              dict(name='one-byte-over', chunk_bytes=4096, repeat=8192, tail_bytes=1, valid=False)]
    inventory_cases = [
        dict(name='native-complete', value=inventory, valid=True),
        dict(name='partial-empty', value=dict(tensors=[], coverage='partial', diagnostics=[]), valid=True),
        dict(name='coverage-required', value=dict(tensors=[], diagnostics=[]), valid=False),
        dict(name='coverage-closed', value=dict(tensors=[], coverage='unknown', diagnostics=[]), valid=False),
        dict(name='diagnostics-required', value=dict(tensors=[], coverage='complete'), valid=False),
        dict(name='diagnostics-not-graph-local', value=dict(tensors=[], coverage='partial', diagnostics=[
            dict(code='excluded', message='Descriptive only.', node_id='foreign')]), valid=False),
    ]
    return dict(inventory_cases=inventory_cases, response=response, context=context, cases=cases, byte_cases=bounds,
                http_cases=[dict(status=409, value=dict(code='model_content_changed',message='Pinned model content changed.'))])


def apply_edits(value, edits):
    result = deepcopy(value)
    for edit in edits:
        target = result
        path = edit['path']
        for key in path[:-1]:
            target = target[int(key)] if isinstance(target, list) else target[key]
        key = int(path[-1]) if isinstance(target, list) else path[-1]
        if edit.get('delete'):
            del target[key]
        else:
            target[key] = deepcopy(edit['value'])
    return result
