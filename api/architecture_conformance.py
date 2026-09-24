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


def expand_compact_graph(source):
    """Independent expansion of the public repeated-component wire form."""
    if not source.get('compact_components'):
        return source
    graph = deepcopy(source)
    families = graph.pop('compact_components')
    parameters = unique(graph['parameters'])
    repetitions = unique(graph['repetitions'])
    symbols = unique(graph['symbols'], 'name')
    seen = {record['id'] for key in ('nodes', 'edges', 'parameters', 'repetitions') for record in graph[key]}
    prefixes = set()

    def replace(value, ids, before, after):
        if isinstance(value, str):
            return ids.get(value, value.replace(before, after))
        if isinstance(value, list):
            return [replace(item, ids, before, after) for item in value]
        if isinstance(value, dict):
            return {key: replace(item, ids, before, after) for key, item in value.items()}
        return value

    for family in families:
        require(family['id'] not in seen, 'duplicate compact family')
        seen.add(family['id'])
        repetition = repetitions.get(family['repetition_id'])
        require(repetition is not None and family['nodes'][0]['kind'] == 'group', 'compact root')
        require(all(item['variant'] in ('routed_expert', 'routed_swiglu') for item in repetition['instances']),
                'compact routed variant')
        require([(item['node_id'], item['index']) for item in repetition['instances']] ==
                [(item['node_id'], item['index']) for item in family['instances']], 'compact repetition order')
        names = [parameters[pid]['name'] for pid in family['parameter_ids']]
        require(all(name.startswith(family['base_prefix'] + '.') for name in names), 'compact prototype parameter scope')
        require(all(name in symbols for name in family['symbols']), 'compact prototype symbols')
        for instance in family['instances']:
            require(instance['prefix'] not in prefixes, 'duplicate compact prefix')
            prefixes.add(instance['prefix'])
            require(len(instance['node_ids']) == len(family['nodes']) and
                    len(instance['edge_ids']) == len(family['edges']) and
                    len(instance['parameter_ids']) == len(family['parameter_ids']) and
                    len(instance['symbols']) == len(family['symbols']) and
                    instance['node_ids'][0] == instance['node_id'], 'compact mapping length')
            require(all(name in symbols for name in instance['symbols']), 'compact instance symbols')
            for name, pid in zip(names, instance['parameter_ids'], strict=True):
                parameter = parameters.get(pid)
                require(parameter is not None and parameter['name'] == name.replace(family['base_prefix'], instance['prefix']),
                        'compact expert binding is missing or swapped')
            ids = {record['id']: target for record, target in zip(family['nodes'], instance['node_ids'], strict=True)}
            ids.update((record['id'], target) for record, target in zip(family['edges'], instance['edge_ids'], strict=True))
            ids.update(zip(family['parameter_ids'], instance['parameter_ids'], strict=True))
            ids.update(zip(family['symbols'], instance['symbols'], strict=True))
            nodes = replace(family['nodes'], ids, family['base_prefix'], instance['prefix'])
            edges = replace(family['edges'], ids, family['base_prefix'], instance['prefix'])
            nodes[0]['label'] = instance['label']
            for attribute in nodes[0]['attributes']:
                if attribute['name'] == 'expert_index':
                    attribute['value'] = float(instance['index'])
            require(nodes[0]['parent_id'] == repetition['parent_id'], 'compact parent')
            for record in nodes + edges:
                require(record['id'] not in seen, 'duplicate compact record')
                seen.add(record['id'])
            graph['nodes'].extend(nodes)
            graph['edges'].extend(edges)
    return graph


def validate_packed_storage(parameter, geometry, tensor=None):
    """Logical matrices require a complete admitted group, never raw auxiliary IDs."""
    require(parameter['name'].endswith('.weight') and len(geometry) == 2 and min(geometry) > 0,
            'packed logical geometry')
    output, inputs = geometry
    prefix = parameter['name'].removesuffix('.weight')
    representation = tensor.get('storage_format') if tensor is not None else (
        'gptq-int4' if any(s['name'] == prefix + '.qweight' for s in parameter['storage']) else 'nvfp4')
    if representation == 'gptq-int4':
        require(inputs % 128 == 0 and output % 8 == 0, 'GPTQ logical geometry')
        dtype = 'I32'
        expected = {'qweight': ('I32', [inputs // 8, output]),
                    'qzeros': ('I32', [inputs // 128, output // 8]),
                    'scales': ('F16', [inputs // 128, output]), 'g_idx': ('I32', [inputs])}
    elif representation == 'bnb-nf4-dq':
        dtype = 'U8'
        elements = product(geometry)
        absmax_count = (elements + 63) // 64
        nested_count = (absmax_count + 255) // 256
        state_name = prefix + '.weight.quant_state.bitsandbytes__nf4'
        states = [record for record in parameter['storage'] if record['name'] == state_name]
        require(len(states) == 1 and states[0]['dtype'] == 'U8' and len(states[0]['shape']) == 1 and
                0 < states[0]['shape'][0] <= 4096, 'NF4 quantization state metadata')
        expected = {'weight': ('U8', [(elements + 1) // 2, 1]),
                    'weight.absmax': ('U8', [absmax_count]),
                    'weight.quant_map': ('F32', [16]),
                    'weight.nested_absmax': ('F32', [nested_count]),
                    'weight.nested_quant_map': ('F32', [256]),
                    'weight.quant_state.bitsandbytes__nf4': ('U8', states[0]['shape'])}
    elif representation == 'nvfp4':
        require(inputs % 16 == 0, 'NVFP4 logical geometry')
        dtype = 'U8'
        expected = {'weight': ('U8', [output, inputs // 2]),
                    'weight_scale': ('F8_E4M3', [output, inputs // 16]),
                    'weight_scale_2': ('F32', []), 'input_scale': ('F32', [])}
    else:
        raise ValueError('unsupported packed representation')
    require(tensor is None or tensor['storage_dtype'] == dtype, 'packed inventory dtype')
    require(len(parameter['storage']) == len(expected) and
            {s['name']: (s['dtype'], s['shape']) for s in parameter['storage']} ==
            {prefix + '.' + name: record for name, record in expected.items()}, 'packed storage group')


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
    graph = expand_compact_graph(value['graph'])
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
            require(p['binding'] in ('native', 'quantized', 'alias'), 'incomplete logical inspection')
            dims = p['logical_shape']
            require(dims is not None and len(dims) in (1, 2) and
                    all(d['kind'] == 'constant' for d in dims), 'inspection logical geometry')
            geometry = [d['value'] for d in dims]
            native = p
            while native['binding'] == 'alias':
                native = params[native['alias_of']]
            require(native['binding'] in ('native', 'quantized') and native['logical_shape'] == dims and
                    native['inspection']['status'] == 'available' and
                    native['inspection']['tensor_id'] == inspection['tensor_id'],
                    'alias logical identity/geometry')
            if native['binding'] == 'native':
                adapter_factors = [s for s in native['storage'] if s.get('role') == 'adapter_factor']
                if adapter_factors:
                    require(len(native['storage']) == len(adapter_factors) == 1 and
                            adapter_factors[0]['shape'] == geometry and
                            adapter_factors[0]['dtype'] in ('F32', 'F16', 'BF16',
                                                            'float32', 'float16', 'bfloat16'),
                            'adapter factor storage geometry')
                else:
                    require(any(s['name'] == native['name'] and s['shape'] == geometry and
                            s['dtype'] in ('F32', 'F16', 'BF16', 'float32', 'float16', 'bfloat16') and
                            s.get('role') not in ('scales', 'packed', 'packed_data')
                            for s in native['storage']), 'native storage geometry')
            tensor = None
            if inventory is not None:
                tensor = inventory.get(inspection['tensor_id'])
                require(tensor is not None and tensor['shape'] == geometry and
                        tensor['name'] == native['name'] and tensor['rank'] == len(geometry) and
                        tensor['numel'] == product(geometry), 'session inventory membership/geometry')
                if native['binding'] == 'native':
                    require(any((s['name'] == tensor['name'] or
                                 s.get('role') == 'adapter_factor') and
                                s['dtype'] == tensor['storage_dtype'] and s['shape'] == tensor['shape']
                                for s in native['storage']), 'inventory storage identity')
            if native['binding'] == 'quantized':
                validate_packed_storage(native, geometry, tensor)

    validate_templates(graph)


def validate_templates(graph):
    """Independent role-labelled graph comparison; external neighbors remain context."""
    templates = graph.get('templates', [])
    if not templates:
        return
    nodes = {n['id']: n for n in graph['nodes']}
    parameters = {p['id']: p for p in graph['parameters']}
    edges = {e['id']: e for e in graph['edges']}
    unique(graph['nodes'] + graph['parameters'] + graph['edges'] + graph['repetitions'] + templates)
    adjacency = {n: set() for n in nodes}
    for e in edges.values():
        adjacency[e['source']['node_id']].add(e['id'])
        adjacency[e['target']['node_id']].add(e['id'])
    repeated = {i['node_id']: (rep['id'], position) for rep in graph['repetitions']
                for position, i in enumerate(rep['instances'])}
    order = {}
    pending = [(n['id'], None) for n in nodes.values() if 'parent_id' not in n]
    while pending:
        ident, inherited = pending.pop()
        owner = repeated.get(ident, inherited)
        if owner is not None:
            order[ident] = owner
        for position, child in enumerate(nodes[ident].get('children', [])):
            order[child] = owner or (ident, position)
            pending.append((child, owner))
    aliases = {}
    for ident in parameters:
        path, terminal = [], ident
        while terminal not in aliases and parameters[terminal]['binding'] == 'alias':
            path.append(terminal)
            terminal = parameters[terminal]['alias_of']
        terminal = aliases.get(terminal, terminal)
        aliases[ident] = terminal
        for member in path:
            aliases[member] = terminal
    diagnosed = {d.get('node_id') for d in graph['diagnostics']}
    seen = set()
    for template in templates:
        require(any(p['kind'] == 'description' and p.get('revision') for p in template['provenance']),
                'reviewed template provenance')
        previous, scope, baseline = -1, None, None
        for instance in template['instances']:
            root = instance['node_id']
            require(root in nodes and nodes[root]['kind'] == 'group' and root not in seen,
                    'template component identity')
            seen.add(root)
            owner, position = order.get(root, (None, -1))
            require(owner is not None and (scope is None or scope == owner) and position > previous,
                    'template source order/scope')
            scope, previous = owner, position
            members, pending = set(), [root]
            while pending:
                ident = pending.pop()
                require(ident not in members, 'template containment')
                members.add(ident)
                pending.extend(nodes[ident].get('children', []))
            require(not members.intersection(diagnosed), 'unverified template node')
            maps = {}
            for category, field in [('nodes', 'node_id'), ('ports', 'port_id'),
                                    ('edges', 'edge_id'), ('parameters', 'parameter_id')]:
                unique(instance[category], 'role')
                maps[category] = {
                    (m['node_id'], m['port_id']) if category == 'ports' else m[field]: m['role']
                    for m in instance[category]}
                require(len(maps[category]) == len(instance[category]), 'duplicate template target')
            nr, pr, er, wr = (maps[c] for c in ['nodes', 'ports', 'edges', 'parameters'])
            require(set(nr) == members, 'template node closure')
            require(set(pr) == {(ident, p['id']) for ident in members for p in nodes[ident]['ports']},
                    'template port coverage')
            require(set(er) == {eid for ident in members for eid in adjacency[ident]
                    if edges[eid]['source']['node_id'] in members and edges[eid]['target']['node_id'] in members},
                    'template edge coverage')
            require(set(wr) == {pid for ident in members for pid in nodes[ident]['parameter_ids']} |
                    {ref['parameter_id'] for ident in members for ref in nodes[ident]['references']
                     if ref['kind'] == 'parameter'}, 'template parameter coverage')
            def known_shape(shape):
                require(shape is not None and all(d['kind'] != 'unknown' for d in shape),
                        'unknown template shape')
                return shape
            normal = {'root': nr[root], 'nodes': {}, 'ports': {}, 'edges': {}, 'parameters': {}}
            for ident in members:
                n = nodes[ident]
                attributes = unique(n['attributes'], 'name')
                require(all(a['value'] is not None and not (isinstance(a['value'], list) and
                        None in a['value']) for a in attributes.values()), 'unknown template attribute')
                require(n['kind'] == 'group' or n.get('operation'), 'unknown template operation')
                if ident == root:
                    require(attributes.get('semantic_role', {}).get('value') == template['component_role'],
                            'template component role')
                normal['nodes'][nr[ident]] = [n['kind'], n.get('operation'), n.get('formula'), n.get('description'),
                    nr.get(n.get('parent_id')), [nr[c] for c in n.get('children', [])],
                    [pr[(ident, p['id'])] for p in n['ports']], [wr[p] for p in n['parameter_ids']],
                    [wr[r['parameter_id']] for r in n['references'] if r['kind'] == 'parameter'],
                    {name: a['value'] for name, a in attributes.items()}]
                for p in n['ports']:
                    normal['ports'][pr[(ident, p['id'])]] = [nr[ident], p['id'], p['direction'], known_shape(p['shape'])]
            for eid, role in er.items():
                e = edges[eid]
                normal['edges'][role] = [pr[(e['source']['node_id'], e['source']['port_id'])],
                    pr[(e['target']['node_id'], e['target']['port_id'])], e['kind']]
            alias_roles = {}
            for pid, role in wr.items():
                alias_roles.setdefault(aliases[pid], []).append(role)
            for pid, role in wr.items():
                normal['parameters'][role] = [known_shape(parameters[pid]['logical_shape']), sorted(alias_roles[aliases[pid]])]
            require(baseline is None or baseline == normal, 'incompatible template computation')
            baseline = normal


def template_cases():
    """Authored two-instance oracle, independent of all packaged descriptions."""
    provenance = [dict(kind='description', source='independent-attention-fixture', revision='1')]
    dims = [dict(kind='constant', value=2), dict(kind='constant', value=2)]
    def ports(*pairs):
        return [dict(id=ident, label=ident, direction=direction, shape=deepcopy(dims)) for ident, direction in pairs]
    def node(ident, kind, parent=None, **kwargs):
        result = dict(id=ident, kind=kind, label=ident, ports=[], parameter_ids=[], references=[],
                      attributes=[], provenance=provenance, **({'parent_id': parent} if parent else {}))
        result.update(kwargs)
        return result
    def attr(name, value):
        return dict(name=name, value=value, provenance=provenance)
    nodes = [node('root', 'group', children=['layer0', 'layer2'])]
    edges, parameters, instances = [], [], []
    for i in (0, 2):
        layer, component = f'layer{i}', f'attention{i}'
        nodes.append(node(layer, 'group', 'root', children=[component], ports=ports(('x','input'),('out','output'))))
        nodes.append(node(component, 'group', layer, children=[f'q{i}',f'k{i}',f'dot{i}'],
                          ports=ports(('x','input'),('out','output')), attributes=[attr('semantic_role','attention')]))
        for role in ('q', 'k'):
            ident, pid = f'{role}{i}', f'{role}weight{i}'
            nodes.append(node(ident, 'operation', component, operation='linear',
                ports=ports(('x','input'),('out','output')), parameter_ids=[pid],
                references=[dict(kind='parameter',parameter_id=pid)],
                attributes=[attr('semantic_role', 'query_projection' if role == 'q' else 'key_projection'), attr('bias',False)]))
            parameters.append(dict(id=pid,name=f'layer{i}.{role}.weight', logical_shape=deepcopy(dims),
                binding='unresolved',storage=[],inspection=dict(status='unavailable',reason='unresolved_binding',
                message='Independent fixture has no storage.'),provenance=provenance))
        nodes.append(node(f'dot{i}','operation',component,operation='matmul',formula='Q K^T',
            ports=ports(('q','input'),('k','input'),('out','output')),attributes=[attr('causal',True)]))
        routes = [('input',layer,'x',component,'x'),('q_input',component,'x',f'q{i}','x'),
                  ('k_input',component,'x',f'k{i}','x'),('q_dot',f'q{i}','out',f'dot{i}','q'),
                  ('k_dot',f'k{i}','out',f'dot{i}','k'),('result',f'dot{i}','out',component,'out'),
                  ('output',component,'out',layer,'out')]
        for role, source, sp, target, tp in routes:
            edges.append(dict(id=f'{role}{i}',source=dict(node_id=source,port_id=sp),
                target=dict(node_id=target,port_id=tp),kind='data',provenance=provenance))
        mapping = dict(component=component,q=f'q{i}',k=f'k{i}',dot=f'dot{i}')
        instances.append(dict(node_id=component,nodes=[dict(role=role,node_id=ident) for role,ident in mapping.items()],
            ports=[dict(role=role+'.'+p['id'],node_id=ident,port_id=p['id']) for role,ident in mapping.items()
                   for n in nodes if n['id']==ident for p in n['ports']],
            edges=[dict(role=role,edge_id=f'{role}{i}') for role in ('q_input','k_input','q_dot','k_dot','result')],
            parameters=[dict(role=role+'.weight',parameter_id=f'{role}weight{i}') for role in ('q','k')]))
    graph = dict(graph_id='independent-templates',scope='language_model',coverage='complete',symbols=[],nodes=nodes,
        edges=edges,parameters=parameters,diagnostics=[],repetitions=[dict(id='layers',parent_id='root',label='layers',
            instances=[dict(node_id=f'layer{i}',index=i,variant='full_attention') for i in (0,2)])],
        templates=[dict(id='shared_attention',label='Attention',component_role='attention',revision='1',
                        provenance=provenance,instances=instances)])
    cases = []
    def case(name, edits=(), valid=False):
        cases.append(dict(name='template-'+name,base='template_response',edits=list(edits),valid=valid,context_edits=[],
                          schema_valid=name not in {'null','singleton','missing-provenance'}))
    def set_(path, value):
        return dict(path=('graph/'+path).split('/'),value=value)
    case('verified-nonconsecutive',valid=True)
    case('absent', [dict(path=['graph','templates'],delete=True)],True)
    case('empty', [set_('templates',[])],True)
    case('null', [set_('templates',None)])
    case('singleton',[set_('templates/0/instances',instances[:1])])
    case('reversed-order',[set_('templates/0/instances',list(reversed(instances)))])
    case('duplicate-instance',[set_('templates/0/instances/1',instances[0])])
    case('missing-provenance',[set_('templates/0/provenance',[])])
    case('wrong-component-role',[set_('templates/0/component_role','mlp')])
    case('id-collision',[set_('templates/0/id','root')])
    for category, field in [('nodes','node_id'),('ports','node_id'),('edges','edge_id'),('parameters','parameter_id')]:
        first = instances[1][category][0]
        case(category+'-missing',[set_(f'templates/0/instances/1/{category}',instances[1][category][1:])])
        case(category+'-duplicate-role',[set_(f'templates/0/instances/1/{category}/1/role',first['role'])])
        case(category+'-foreign-reference',[set_(f'templates/0/instances/1/{category}/0/{field}','foreign')])
        case(category+'-wrong-subtree',[set_(f'templates/0/instances/1/{category}/0',instances[0][category][0])])
    dot = next(i for i,n in enumerate(nodes) if n['id']=='dot2')
    q = next(i for i,n in enumerate(nodes) if n['id']=='q2')
    case('swapped-same-shape-qk',[set_('edges/10/target/port_id','k'),set_('edges/11/target/port_id','q')])
    case('swapped-port-roles',[set_('templates/0/instances/1/ports/6/role','dot.k'),set_('templates/0/instances/1/ports/7/role','dot.q')])
    case('wrong-instance-parameter',[set_(f'nodes/{q}/parameter_ids',['qweight0'])])
    case('different-operation',[set_(f'nodes/{dot}/operation','linear_attention')])
    case('different-formula',[set_(f'nodes/{dot}/formula','K Q^T')])
    case('different-normalization',[set_(f'nodes/{q}/attributes/1',attr('epsilon',1e-6))])
    case('different-bias',[set_(f'nodes/{q}/attributes/1/value',True)])
    case('different-shape',[set_('parameters/2/logical_shape',[dict(kind='constant',value=3),dict(kind='constant',value=2)])])
    case('different-state-rule',[set_(f'nodes/{dot}/attributes/0',attr('state_rule','recurrent'))])
    case('different-edge-kind',[set_('edges/10/kind','state')])
    case('unknown-attribute',[set_(f'nodes/{dot}/attributes/0/value',None)])
    case('unknown-shape',[set_('parameters/2/logical_shape',None)])
    case('parameter-alias',[set_('parameters/2/binding','alias'),set_('parameters/2/alias_of','qweight0')],True)
    case('internal-alias-change',[set_('parameters/3/binding','alias'),set_('parameters/3/alias_of','qweight2')])
    case('unavailable-storage-difference',[set_('parameters/2/inspection/reason','unsupported_representation')],True)
    return graph, cases


def compact_cases():
    """Small authored routed-expert wire document with distinct exact bindings."""
    provenance = [dict(kind='description', source='independent-compact-fixture', revision='1')]
    prefix = 'model.layers.1.mlp.experts.0'
    root = dict(id='root',kind='group',label='MoE',children=['expert0','expert1'],ports=[],
                parameter_ids=[],references=[],attributes=[],provenance=provenance)
    prototype = dict(id='expert0',kind='group',parent_id='root',label='Routed expert 0',
                     operation='weighted_swiglu_mlp',children=[],ports=[],parameter_ids=['p0'],
                     references=[dict(kind='module',name=prefix),dict(kind='parameter',parameter_id='p0')],
                     attributes=[dict(name='expert_index',value=0.0,provenance=provenance)],
                     provenance=provenance)
    def parameter(index):
        return dict(id=f'p{index}',name=f'model.layers.1.mlp.experts.{index}.w1.weight',
                    logical_shape=[dict(kind='constant',value=2),dict(kind='constant',value=2)],
                    binding='unresolved',storage=[],inspection=dict(status='unavailable',
                    reason='unresolved_binding',message='Independent fixture has no storage.'),provenance=provenance)
    def instance(index):
        return dict(node_id=f'expert{index}',prefix=f'model.layers.1.mlp.experts.{index}',
                    label=f'Routed expert {index}',index=index,node_ids=[f'expert{index}'],
                    edge_ids=[],parameter_ids=[f'p{index}'],symbols=[])
    graph = dict(graph_id='independent-compact',scope='language_model',coverage='complete',symbols=[],
                 nodes=[root],edges=[],parameters=[parameter(0),parameter(1)],diagnostics=[],
                 repetitions=[dict(id='experts',parent_id='root',label='Routed experts',instances=[
                     dict(node_id='expert0',index=0,variant='routed_expert'),
                     dict(node_id='expert1',index=1,variant='routed_expert')])],
                 compact_components=[dict(id='compact_experts',repetition_id='experts',base_prefix=prefix,
                     nodes=[prototype],edges=[],parameter_ids=['p0'],symbols=[],instances=[instance(0),instance(1)])])
    response = dict(status='available',model_id='test/architecture',diagnostics=[],graph=graph)
    def edit(path,value): return dict(path=('graph/'+path).split('/'),value=value)
    cases = [
        dict(name='compact-experts-valid',base='compact_response',edits=[],valid=True,schema_valid=True,context_edits=[]),
        dict(name='compact-experts-missing-binding',base='compact_response',
             edits=[edit('compact_components/0/instances/1/parameter_ids',['missing'])],valid=False,
             schema_valid=True,context_edits=[]),
        dict(name='compact-experts-swapped-binding',base='compact_response',
             edits=[edit('compact_components/0/instances/1/parameter_ids',['p0'])],valid=False,
             schema_valid=True,context_edits=[]),
        dict(name='compact-experts-wrong-index',base='compact_response',
             edits=[edit('compact_components/0/instances/1/index',0)],valid=False,
             schema_valid=True,context_edits=[]),
        dict(name='compact-experts-duplicate-node',base='compact_response',
             edits=[edit('compact_components/0/instances/1/node_ids',['expert0'])],valid=False,
             schema_valid=True,context_edits=[]),
    ]
    return response, cases


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
        dict(id='quantized', name='packed.weight', logical_shape=[dim(8), dim(128)], binding='quantized',
             storage=[dict(name='packed.qweight', dtype='I32', shape=[16, 8], role='packed_data'),
                      dict(name='packed.qzeros', dtype='I32', shape=[1, 1], role='zero_points'),
                      dict(name='packed.scales', dtype='F16', shape=[1, 8], role='scales'),
                      dict(name='packed.g_idx', dtype='I32', shape=[128], role='group_indices')],
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
        ('packed-cannot-use-native-identity', [set_('graph/parameters/2/inspection',dict(status='available',tensor_id='tensor_native'))]),
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
    # These are complete logical matrices backed by exact admitted storage groups.
    # Their descriptors retain physical dtype/format but expose mathematical shape.
    packed = deepcopy(parameters[2])
    packed['inspection'] = dict(status='available', tensor_id='tensor_packed')
    nvfp4 = dict(id='quantized', name='fp4.weight', logical_shape=[dim(2), dim(32)],
        binding='quantized', inspection=dict(status='available', tensor_id='tensor_packed'), provenance=[],
        storage=[dict(name='fp4.weight', dtype='U8', shape=[2,16], role='packed_nvfp4'),
                 dict(name='fp4.weight_scale', dtype='F8_E4M3', shape=[2,2], role='block_scale'),
                 dict(name='fp4.weight_scale_2', dtype='F32', shape=[], role='global_weight_scale'),
                 dict(name='fp4.input_scale', dtype='F32', shape=[], role='input_scale')])
    for label, parameter, dtype, representation in [
        ('gptq', packed, 'I32', 'gptq-int4'), ('nvfp4', nvfp4, 'U8', 'nvfp4')]:
        dims = [d['value'] for d in parameter['logical_shape']]
        descriptor = dict(id='tensor_packed', name=parameter['name'], path=parameter['name'].split('.'),
            shape=dims, rank=2, numel=product(dims), storage_dtype=dtype,
            storage_format=representation, logical_dtype='float32')
        base_edits = [set_('graph/parameters/2', parameter)]
        base_context = [set_('inventory/tensors', inventory['tensors'] + [descriptor])]
        case(label+'-complete-logical-inspection', base_edits, True, context_edits=base_context)
        case(label+'-complete-logical-alias', base_edits + [
            set_('graph/parameters/1/alias_of', 'quantized'),
            set_('graph/parameters/1/logical_shape', parameter['logical_shape']),
            set_('graph/parameters/1/storage', []),
            set_('graph/parameters/1/inspection', parameter['inspection'])], True,
            context_edits=base_context)
        for suffix, changes in [
            ('wrong-primary-geometry', [set_('graph/parameters/2/storage/0/shape', [1,1])]),
            ('wrong-primary-dtype', [set_('graph/parameters/2/storage/0/dtype', 'F32')]),
            ('wrong-scale-dtype', [set_('graph/parameters/2/storage/'+('2' if label == 'gptq' else '1')+'/dtype', 'F32')]),
            ('foreign-companion', [set_('graph/parameters/2/storage/1/name', 'foreign.scale')]),
            ('missing-companion', [set_('graph/parameters/2/storage', parameter['storage'][:-1])]),
            ('duplicate-companion', [set_('graph/parameters/2/storage/3', parameter['storage'][0])]),
            ('wrong-logical-identity', [set_('graph/parameters/2/name', 'other.weight')]),
            ('auxiliary-logical-identity', [set_('graph/parameters/2/name', 'packed.scales')]),
            ('wrong-actionable-id', [set_('graph/parameters/2/inspection/tensor_id', 'tensor_native')]),
        ]:
            case(label+'-'+suffix, base_edits + changes, context_edits=base_context)
        for suffix, changes in [
            ('wrong-inventory-format', [set_('inventory/tensors/2/storage_format', 'unknown')]),
            ('missing-inventory-format', [delete('inventory/tensors/2/storage_format')]),
            ('wrong-inventory-dtype', [set_('inventory/tensors/2/storage_dtype', 'F16')]),
            ('wrong-inventory-name', [set_('inventory/tensors/2/name', 'other.weight')]),
            ('wrong-inventory-shape', [set_('inventory/tensors/2/shape', list(reversed(dims)))]),
        ]:
            case(label+'-'+suffix, base_edits, context_edits=base_context + changes)
    # SmolLM2-135M's q_proj is a 576x576 NF4 matrix in the pinned QLoRA reference.
    # The serialized quantization-state bytes are opaque to this contract fixture.
    nf4_elements = 576 * 576
    nf4_absmax_count = (nf4_elements + 63) // 64
    nf4_nested_count = (nf4_absmax_count + 255) // 256
    nf4_prefix = 'model.layers.0.self_attn.q_proj'
    nf4 = dict(id='quantized', name=nf4_prefix + '.weight',
        logical_shape=[dim(576), dim(576)], binding='quantized',
        inspection=dict(status='available', tensor_id='tensor_packed'), provenance=[],
        storage=[dict(name=nf4_prefix + '.weight', dtype='U8', shape=[(nf4_elements + 1) // 2, 1], role='packed_data'),
                 dict(name=nf4_prefix + '.weight.absmax', dtype='U8', shape=[nf4_absmax_count], role='scales'),
                 dict(name=nf4_prefix + '.weight.quant_map', dtype='F32', shape=[16], role='codebook'),
                 dict(name=nf4_prefix + '.weight.nested_absmax', dtype='F32', shape=[nf4_nested_count], role='scales'),
                 dict(name=nf4_prefix + '.weight.nested_quant_map', dtype='F32', shape=[256], role='codebook'),
                 dict(name=nf4_prefix + '.weight.quant_state.bitsandbytes__nf4', dtype='U8', shape=[1], role='quantization_state')])
    nf4_descriptor = dict(id='tensor_packed', name=nf4['name'], path=nf4['name'].split('.'),
        shape=[576, 576], rank=2, numel=nf4_elements, storage_dtype='U8',
        storage_format='bnb-nf4-dq', logical_dtype='float32')
    nf4_base_edits = [set_('graph/parameters/2', nf4)]
    nf4_context = [set_('inventory/tensors', inventory['tensors'] + [nf4_descriptor])]
    case('bnb-nf4-dq-complete-logical-inspection', nf4_base_edits, True, context_edits=nf4_context)
    template_graph, additional_cases = template_cases()
    cases.extend(additional_cases)
    compact_response, compact_additional = compact_cases()
    cases.extend(compact_additional)
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
    return dict(inventory_cases=inventory_cases, response=response,
                template_response={**response, 'graph': template_graph}, compact_response=compact_response,
                context=context, cases=cases, byte_cases=bounds,
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
