#!/usr/bin/env python3
"""Validate the source contract and generate tiny, language-neutral codec oracles.

This is fixture tooling, not a server/client stream parser. Rejection outcomes
are declarative test inputs for the independent Python and TypeScript codecs.
"""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import struct

from jsonschema import Draft202012Validator, FormatChecker, ValidationError
from openapi_spec_validator import OpenAPIV31SpecValidator
import yaml

from architecture_conformance import (apply_edits, bounded_size, fixtures as architecture_fixtures,
                                      validate_architecture)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / 'docs/spec/api/openapi.yaml'
GOLDEN = ROOT / 'api/fixtures/conformance.json'
EMBEDDINGS_GOLDEN = ROOT / 'api/fixtures/embeddings.json'
EMBEDDING_ANALYSIS_GOLDEN = ROOT / 'api/fixtures/embedding-analysis.json'
ARCHITECTURE_GOLDEN = ROOT / 'api/fixtures/architecture.json'
SAFE = 2**53 - 1
U32 = 2**32 - 1


def require(condition, message):
    if not condition:
        raise ValueError(message)


def product(shape):
    # Determine zero first: e.g. [SAFE, SAFE, 0] has a zero product.
    if 0 in shape:
        return 0
    result = 1
    for dimension in shape:
        result *= dimension
        require(result <= SAFE, 'unsafe product')
    return result


def semantics(schema, value):
    """Only cross-field JSON checks; there is deliberately no wire reader here."""
    if schema in ('StreamMetadata', 'TensorMetadata', 'TensorStatisticsMetadata',
                  'TensorDistributionsMetadata', 'InputEmbeddingsMetadata',
                  'InputEmbeddingsStatisticsMetadata', 'InputEmbeddingsDistributionsMetadata'):
        kind = value['kind']
        if kind.startswith('input_embeddings'):
            require(len(json_bytes(value)) <= 1048576, 'unsupported_size')
        if kind == 'input_embeddings_statistics':
            require(value['shape'][0] == len(value['token_ids']) and
                    value['count'] == product(value['shape']), 'embedding statistics geometry')
        if kind == 'input_embeddings_distributions':
            require(value['rows'] == len(value['token_ids']), 'embedding distribution rows')
            require(value['rows'] != 0 or value['domain_minimum'] is None, 'empty domain')
        if kind in ('tensor', 'input_embeddings'):
            size = 4 * product(value['shape'])
            require(size <= SAFE and value['byte_length'] == size, 'tensor byte length')
            if kind == 'input_embeddings':
                require(value['shape'][0] == len(value['token_ids']), 'embedding row count')
                require(len(json_bytes(value)) <= 1048576, 'unsupported_size')
        elif kind in ('tensor_statistics', 'input_embeddings_statistics'):
            require(value['count'] == value['finite_count'] + value['non_finite_count'],
                    'statistics count sum')
            if value['finite_count']:
                lo, hi = value['minimum'], value['maximum']
                ps = list(value['percentiles'][p] for p in ('p01', 'p05', 'p50', 'p95', 'p99'))
                require(lo <= value['mean'] <= hi and lo <= hi, 'statistics range')
                require([lo, *ps, hi] == sorted([lo, *ps, hi]), 'percentile order')
        elif kind in ('tensor_distributions', 'input_embeddings_distributions'):
            require(product([value['rows'], value['columns']]) <= SAFE, 'source size')
            offset = 0
            for section, shape in zip(value['sections'],
                                      ([value['rows'], 100], [100, value['columns']])):
                length = 4 * product(shape)
                require(length <= SAFE and section['shape'] == shape and
                        section['offset'] == offset and section['byte_length'] == length,
                        'section geometry')
                offset += length
            require(offset <= SAFE and offset == value['byte_length'], 'section total')
            if value['domain_minimum'] is not None:
                require(value['domain_minimum'] <= value['domain_maximum'], 'domain order')
    elif schema == 'ArchitectureResponse':
        validate_architecture(value)
    elif schema == 'TensorInventory':
        for tensor in value['tensors']:
            semantics('TensorDescriptor', tensor)
    elif schema == 'TensorDescriptor':
        require(value['rank'] == len(value['shape']) and value['numel'] == product(value['shape']),
                'descriptor shape/rank/numel')
    elif schema == 'StreamProgress':
        require('total' not in value or value['completed'] <= value['total'], 'progress total')
    elif schema == 'TokenizeResponse':
        for index, token in enumerate(value['tokens']):
            require(token['index'] == index, 'token index')
            if 'start' in token:
                require(token['start'] <= token['end'] <= len(value['text']), 'token span')


def validate_instance(document, schema, instance):
    # Local $refs remain rooted in the one authoritative OpenAPI document.
    validator = Draft202012Validator(
        {'$ref': f'#/components/schemas/{schema}', 'components': document['components']},
        format_checker=FormatChecker())
    validator.validate(instance)
    semantics(schema, instance)


def resolve_references(document):
    count = 0

    def resolve(reference):
        nonlocal count
        require(reference.startswith('#/'), 'external reference not allowed in contract')
        target = document
        for part in reference[2:].split('/'):
            target = target[part.replace('~1', '/').replace('~0', '~')]
        count += 1

    def walk(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key == '$ref':
                    resolve(child)
                elif key == 'discriminator':
                    for target in child.get('mapping', {}).values():
                        resolve(target)
                else:
                    walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
    walk(document)
    return count


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                      separators=(',', ':')).encode('utf-8')


def header(kind, length=0, flags=0, reserved=0):
    return struct.pack('<4sBBHI', b'LMEX', kind, flags, reserved, length)


def frame(kind, payload=b''):
    if isinstance(payload, dict):
        payload = json_bytes(payload)
    return header(kind, len(payload)) + payload


def tensor(shape):
    return dict(kind='tensor', tensor_id='t', name='test.weight', shape=shape,
                dtype='float32', byte_order='little', layout='c', byte_length=4*product(shape))


def statistics(values):
    finite = sorted(v for v in values if math.isfinite(v))
    n = len(finite)
    ps = {}
    for key, q in zip(('p01', 'p05', 'p50', 'p95', 'p99'), (.01, .05, .5, .95, .99)):
        h = (n - 1) * q
        ps[key] = ((1 - (h % 1))*finite[math.floor(h)] +
                   (h % 1)*finite[math.ceil(h)]) if n else None
    mean = math.fsum(finite)/n if n else None
    stddev = math.sqrt(math.fsum((v-mean)**2 for v in finite)/n) if n else None
    return dict(kind='tensor_statistics', tensor_id='t', count=len(values), finite_count=n,
                non_finite_count=len(values)-n, minimum=finite[0] if n else None,
                maximum=finite[-1] if n else None, mean=mean, stddev=stddev,
                percentiles=ps, byte_length=0)


def checked_count(count):
    require(0 <= count <= U32, 'unsupported_size')
    return count


def distributions(rows, columns, values):
    require(len(values) == rows*columns, 'fixture source shape')
    finite = [v for v in values if math.isfinite(v)]
    lo, hi = (min(finite), max(finite)) if finite else (None, None)
    counts = [0] * ((rows + columns)*100)
    bins = []
    for index, value in enumerate(values):
        if not math.isfinite(value):
            bins.append(None)
            continue
        # Python float is IEEE binary64. This reference runs only on tiny fixtures.
        bin_index = 50 if lo == hi else max(0, min(99, math.floor((value-lo)/(hi-lo)*100)))
        bins.append(bin_index)
        row, col = divmod(index, columns)
        for target in (row*100+bin_index, rows*100+bin_index*columns+col):
            counts[target] = checked_count(counts[target]+1)
    section = lambda name, shape, offset: dict(name=name, shape=shape, offset=offset,
                                              byte_length=4*product(shape))
    metadata = dict(kind='tensor_distributions', tensor_id='t', rows=rows, columns=columns,
                    bin_count=100, binning='linear-full-range', domain_minimum=lo,
                    domain_maximum=hi, dtype='uint32', byte_order='little',
                    sections=[section('row_counts', [rows, 100], 0),
                              section('column_counts', [100, columns], rows*400)],
                    byte_length=len(counts)*4)
    return metadata, struct.pack(f'<{len(counts)}I', *counts), bins, counts


def fixtures():
    wire = []
    instances = []

    def instance(name, schema, value, valid=True):
        instances.append(dict(name=name, schema=schema, valid=valid, value=value))

    def success(name, metadata, data=b'', decoded=None, progress=False):
        instance(name, 'StreamMetadata', metadata)
        frames = [frame(1, metadata)]
        if progress:
            p = dict(completed=0, total=1, unit='elements')
            frames.append(frame(3, p))
            instance('progress', 'StreamProgress', p)
        if data:
            frames.append(frame(2, data[:4]))
            if len(data) > 4:
                frames.append(frame(2, data[4:]))
        frames.append(frame(4))
        wire.append(dict(name=name, wire_hex=b''.join(frames).hex(),
                         expected=dict(outcome='complete', metadata=metadata,
                                       data_hex=data.hex(), decoded=decoded)))

    def reject(name, data, reason):
        wire.append(dict(name=name, wire_hex=data.hex(),
                         expected=dict(outcome='reject', reason=reason)))

    words = [0x3F9E0419, 0xC0200000, 0x80000000, 0x7FC00000, 0x7F800000, 0xFF800000]
    raw = struct.pack('<6I', *words)
    success('tensor-special-floats', tensor([2, 3]), raw,
            [struct.unpack('<f', raw[:4])[0], -2.5, '-0', 'NaN', '+Infinity', '-Infinity'], True)
    success('tensor-scalar', tensor([]), struct.pack('<f', 1.0), [1.0])
    success('tensor-zero', tensor([2, 0, 3]), decoded=[])
    def readable(values):
        return [v if math.isfinite(v) else ('NaN' if math.isnan(v) else ('+Infinity' if v > 0 else '-Infinity')) for v in values]

    finite_values = [-2.0, 0.0, 2.0, 4.0, float('nan')]
    stats = statistics(finite_values)
    # Independent hand-calculated anchors keep the reference generator honest.
    require(stats['mean'] == 1 and stats['stddev'] == math.sqrt(5) and
            stats['percentiles']['p50'] == 1 and stats['percentiles']['p05'] == -1.7,
            'statistics oracle')
    success('statistics-finite', stats, decoded=dict(source_values=readable(finite_values)))
    success('statistics-empty', statistics([]), decoded=dict(source_values=[]))
    success('statistics-singleton', statistics([7.]), decoded=dict(source_values=[7.]))
    nonfinite = [float('nan'), float('inf'), -float('inf')]
    success('statistics-all-nonfinite', statistics(nonfinite), decoded=dict(source_values=readable(nonfinite)))
    extreme = struct.unpack('<f', bytes.fromhex('ffff7f7f'))[0]
    success('statistics-extremes', statistics([-extreme, extreme]), decoded=dict(source_values=[-extreme, extreme]))
    for name, rows, cols, values, expected_bins in [
        ('distributions-range', 2, 2, [-2., 0., 2., 4.], [0, 33, 66, 99]),
        ('distributions-constant', 1, 2, [7., 7.], [50, 50]),
        ('distributions-empty-rows', 0, 2, [], []),
        ('distributions-empty-columns', 2, 0, [], []),
        ('distributions-empty-both', 0, 0, [], []),
        ('distributions-all-nonfinite', 1, 3, nonfinite, [None]*3),
        ('distributions-extremes', 1, 3, [-extreme, 0., extreme], [0, 50, 99]),
        ('distributions-mixed', 1, 3, [0., float('nan'), 1.], [0, None, 99]),
    ]:
        meta, data, bins, counts = distributions(rows, cols, values)
        require(bins == expected_bins, f'{name}: bin oracle')
        require(sum(counts[:rows*100]) == sum(counts[rows*100:]) ==
                sum(math.isfinite(v) for v in values), 'histogram sums')
        success(name, meta, data, dict(uint32_length=len(counts),
                                     nonzero={str(i): v for i, v in enumerate(counts) if v},
                                     source_bin_indices=bins, source_values=readable(values)))
    error = dict(code='resource_exhausted', message='Insufficient resources')
    for name, frames, expected in [
        ('early-error', [frame(5, error)], dict(outcome='error', error=error)),
        ('early-cancelled', [frame(6)], dict(outcome='cancelled')),
        ('partial-error', [frame(1, tensor([2])), frame(2, struct.pack('<f', 1)), frame(5, error)],
         dict(outcome='error', error=error, incomplete_data_hex='0000803f')),
        ('partial-cancelled', [frame(1, tensor([2])), frame(2, struct.pack('<f', 1)), frame(6)],
         dict(outcome='cancelled', incomplete_data_hex='0000803f')),
    ]:
        wire.append(dict(name=name, wire_hex=b''.join(frames).hex(), expected=expected))
    meta = frame(1, tensor([]))
    data = frame(2, bytes.fromhex('0000803f'))
    complete = frame(4)
    valid = meta + data + complete
    for name, payload, reason in [
        ('empty-stream', b'', 'missing_terminal'),
        ('bad-magic', b'NOPE'+valid[4:], 'bad_magic'),
        ('unknown-type', header(7), 'unknown_type'),
        ('flags', header(1, flags=1), 'nonzero_flags'),
        ('reserved', header(1, reserved=256), 'nonzero_reserved'),
        ('data-before-meta', data, 'metadata_required'),
        ('progress-before-meta', frame(3, dict(completed=0, unit='bytes')), 'metadata_required'),
        ('complete-before-meta', complete, 'metadata_required'),
        ('duplicate-meta', meta+meta+data+complete, 'duplicate_metadata'),
        ('invalid-utf8', frame(1, b'\xff'), 'invalid_utf8'),
        ('invalid-json', frame(1, b'{'), 'invalid_json'),
        ('json-nan', frame(1, b'{"kind":"tensor","byte_length":NaN}'), 'invalid_json'),
        ('invalid-schema', frame(1, dict(kind='other', byte_length=0)), 'invalid_metadata'),
        ('complete-payload', meta+data+frame(4, b'x'), 'nonempty_terminal'),
        ('cancelled-payload', frame(6, b'x'), 'nonempty_terminal'),
        ('overrun', meta+data+data+complete, 'data_overrun'),
        ('overrun-before-error', meta+data+data+frame(5, error), 'data_overrun'),
        ('short-result', meta+complete, 'data_length_mismatch'),
        ('truncated-header', valid[:-1], 'truncated_header'),
        ('truncated-payload', meta+header(2, 4)+b'\0', 'truncated_payload'),
        ('missing-terminal', meta+data, 'missing_terminal'),
        ('trailing-byte', valid+b'\0', 'trailing_bytes'),
        ('double-terminal', valid+frame(6), 'trailing_bytes'),
        ('statistics-data', frame(1, stats)+frame(2)+complete, 'statistics_data'),
        ('unaligned-tensor-data', meta+frame(2, b'abc')+complete, 'unaligned_data'),
        ('oversized-meta', header(1, 1048577), 'control_frame_too_large'),
        ('oversized-progress', meta+header(3, 1048577), 'control_frame_too_large'),
        ('oversized-error', header(5, 1048577), 'control_frame_too_large'),
        ('large-data-truncated', frame(1, tensor([1073741823]))+header(2, 4294967292),
         'truncated_payload'),
    ]:
        reject(name, payload, reason)
    dm, _, _, _ = distributions(0, 0, [])
    reject('unaligned-distribution-data', frame(1, dm)+frame(2, b'x')+complete, 'unaligned_data')
    token = lambda i, ident, text, special=False, **span: dict(index=i, id=ident, token=text,
                                                             decoded=text, special=special, **span)
    text = 'A😀e\u0301<special>'
    tokens = [token(0, 100, '<bos>', True), token(1, 10, 'A', start=0, end=1),
              token(2, 11, '�', start=1, end=2), token(3, 12, '�', start=1, end=2),
              token(4, 13, 'e', start=2, end=3), token(5, 14, '\u0301', start=3, end=4),
              token(6, 101, '<special>', True, start=4, end=13)]
    response = dict(text=text, add_special_tokens=True, tokens=tokens)
    instance('unicode-overlapping-specials', 'TokenizeResponse', response)
    tokenizer = dict(response=response, code_points=list(text), code_point_length=len(text),
                     utf16_code_unit_length=len(text.encode('utf-16-le'))//2,
                     source_substrings=[text[t['start']:t['end']] if 'start' in t else None for t in tokens])
    instance('token-no-span', 'Token', token(0, 9, 'unknown'))
    instance('progress-unknown-total', 'StreamProgress', dict(completed=0, unit='bytes'))
    instance('zero-product-before-overflow', 'StreamMetadata', dict(tensor([0]), shape=[SAFE, SAFE, 0]))
    instance('safe-integer-boundary', 'SafeInteger', SAFE)
    instance('unsafe-integer', 'SafeInteger', SAFE+1, False)
    instance('negative-integer', 'SafeInteger', -1, False)
    instance('fractional-integer', 'SafeInteger', .5, False)
    instance('scalar-descriptor', 'TensorDescriptor', dict(id='t', name='w', path=['w'],
             shape=[], rank=0, numel=1, storage_dtype='F32', logical_dtype='float32'))
    descriptor = deepcopy(instances[-1]['value']); descriptor['rank'] = 1
    instance('descriptor-rank-mismatch', 'TensorDescriptor', descriptor, False)
    for name, schema, value in [
        ('tensor-length-mismatch', 'StreamMetadata', dict(tensor([2]), byte_length=4)),
        ('tensor-unsafe-product', 'StreamMetadata', dict(tensor([]), shape=[SAFE, 2])),
        ('tensor-unsafe-byte-product', 'StreamMetadata', dict(tensor([]), shape=[SAFE])),
        ('statistics-count-mismatch', 'StreamMetadata', dict(stats, count=100)),
        ('empty-statistics-nonnull', 'StreamMetadata', dict(statistics([]), mean=0)),
        ('finite-statistics-null', 'StreamMetadata', dict(stats, mean=None)),
        ('negative-stddev', 'StreamMetadata', dict(stats, stddev=-1)),
        ('progress-exceeds-total', 'StreamProgress', dict(completed=2, total=1, unit='rows')),
        ('progress-total-zero', 'StreamProgress', dict(completed=0, total=0, unit='rows')),
        ('token-only-start', 'Token', token(0, 1, 'A', start=0)),
        ('token-only-end', 'Token', token(0, 1, 'A', end=1)),
        ('token-span-outside-input', 'TokenizeResponse', dict(response, tokens=[token(0, 1, 'A', start=0, end=14)])),
        ('token-reversed-span', 'TokenizeResponse', dict(response, tokens=[token(0, 1, 'A', start=2, end=1)])),
        ('unknown-error-code', 'Error', dict(error, code='oops')),
        ('error-missing-message', 'Error', dict(code='internal_error')),
    ]:
        instance(name, schema, value, False)
    dm, _, _, _ = distributions(1, 1, [1.])
    for name, mutate in [
        ('section-offset', lambda m: m['sections'][1].update(offset=0)),
        ('section-shape', lambda m: m['sections'][0].update(shape=[100, 1])),
        ('section-length', lambda m: m['sections'][0].update(byte_length=4)),
        ('section-order', lambda m: m['sections'].reverse()),
        ('section-total', lambda m: m.update(byte_length=4)),
        ('domain-pair', lambda m: m.update(domain_minimum=None)),
        ('domain-order', lambda m: m.update(domain_minimum=2)),
    ]:
        value = deepcopy(dm); mutate(value)
        instance(name, 'StreamMetadata', value, False)
    for code in ('malformed_json', 'model_not_found', 'session_not_found', 'tensor_not_found',
                 'model_content_changed', 'validation_error', 'unsupported_representation',
                 'unsupported_rank', 'unsupported_size', 'resource_exhausted', 'internal_error'):
        instance('error-'+code, 'Error', dict(code=code, message='Public explanation'))
    require(checked_count(U32) == U32, 'uint32 boundary')
    try:
        checked_count(U32+1)
    except ValueError:
        pass
    else:
        raise ValueError('uint32 overflow was accepted')
    return dict(wire_cases=wire, schema_cases=instances, tokenizer=tokenizer,
                numeric_cases=dict(uint32=dict(values=[1, 0x01020304, U32],
                                               little_endian_hex='0100000004030201ffffffff'),
                                   histogram_overflow=dict(count=U32+1, error_code='unsupported_size')))


def embedding_request(document, request, vocabulary_size):
    validate_instance(document, 'InputEmbeddingsRequest', request)
    require(all(token < vocabulary_size for token in request['token_ids']), 'validation_error')


def embedding_fixtures(document):
    # Synthetic input table, not claimed output from a downloaded model.
    table = [[1., -2., 3.5], [4.25, 0., -6.], [-7.5, 8., 9.25], [10., -11.5, 12.]]
    wire, instances, requests = [], [], []

    def metadata(ids):
        return dict(kind='input_embeddings', token_ids=ids, shape=[len(ids), 3],
                    dtype='float32', byte_order='little', layout='c', byte_length=len(ids)*12)

    def instance(name, value, valid=True):
        instances.append(dict(name=name, schema='StreamMetadata', value=value, valid=valid))

    # Explicit numeric oracles independently spell out sequence order and duplicates.
    for name, ids, expected_rows in [
        ('embeddings-asymmetric', [2, 0], [[-7.5, 8., 9.25], [1., -2., 3.5]]),
        ('embeddings-duplicates', [3, 1, 3], [[10., -11.5, 12.], [4.25, 0., -6.], [10., -11.5, 12.]]),
        ('embeddings-one-token', [1], [[4.25, 0., -6.]]),
        ('embeddings-empty', [], []),
    ]:
        request = dict(token_ids=ids)
        embedding_request(document, request, len(table))
        rows = [table[token] for token in ids]
        require(rows == expected_rows, 'embedding row order oracle')
        values = [v for row in rows for v in row]
        raw = struct.pack(f'<{len(values)}f', *values)
        if name == 'embeddings-asymmetric':
            require(raw.hex() == '0000f0c000000041000014410000803f000000c000006040',
                    'embedding little-endian oracle')
        require(list(struct.unpack(f'<{len(values)}f', raw)) ==
                [v for row in expected_rows for v in row], 'embedding byte oracle')
        m = metadata(ids)
        require(len(raw) == m['byte_length'] == 4*product(m['shape']), 'embedding geometry')
        instance(name, m)
        # Deliberately split within the first row, so transport frames are not rows.
        body = frame(1, m)
        if raw:
            body += frame(2, raw[:4]) + frame(3, dict(completed=1, total=len(values), unit='elements'))
            body += frame(2, raw[4:])
        body += frame(4)
        wire.append(dict(name=name, request=request, wire_hex=body.hex(),
                         expected=dict(outcome='complete', metadata=m, data_hex=raw.hex(),
                                       decoded=expected_rows)))
        requests.append(dict(name=name, request=request, valid=True))

    for name, request in [
        ('negative-id', dict(token_ids=[0, -1])),
        ('fractional-id', dict(token_ids=[1.5])),
        ('boolean-id', dict(token_ids=[True])),
        ('string-id', dict(token_ids=['1'])),
        ('null-id', dict(token_ids=[None])),
        ('unsafe-id', dict(token_ids=[SAFE+1])),
        ('out-of-range-id', dict(token_ids=[0, len(table)])),
        ('missing-ids', {}), ('null-ids', dict(token_ids=None)),
        ('extra-field', dict(token_ids=[0], tensor_id='guess')),
    ]:
        requests.append(dict(name='embeddings-'+name, request=request, valid=False,
                             expected_http_status=422, error_code='validation_error'))
    for case in requests:
        try:
            embedding_request(document, case['request'], len(table))
        except (ValidationError, ValueError):
            require(not case['valid'], f"valid request rejected: {case['name']}")
        else:
            require(case['valid'], f"invalid request accepted: {case['name']}")
        # Model range cannot be encoded in a standalone request schema.
        instances.append(dict(name=case['name']+'-request', schema='InputEmbeddingsRequest',
                              value=case['request'],
                              valid=case['valid'] or case['name'] == 'embeddings-out-of-range-id'))

    m = metadata([2, 0])
    for name, fields in [
        ('row-count', dict(shape=[1, 3], byte_length=12)),
        ('rank-one', dict(shape=[6])), ('rank-three', dict(shape=[2, 3, 1])),
        ('hidden-size-zero', dict(shape=[2, 0], byte_length=0)),
        ('length', dict(byte_length=20)), ('unsafe-product', dict(shape=[2, SAFE])),
        ('unsafe-byte-product', dict(token_ids=[0], shape=[1, SAFE//4+1], byte_length=0)),
        ('negative-id-meta', dict(token_ids=[2, -1])),
        ('checkpoint-identity', dict(tensor_id='t', name='input.weight')),
        ('wrong-dtype', dict(dtype='float16')),
    ]:
        invalid = dict(m, **fields)
        instance('embeddings-'+name, invalid, False)
        wire.append(dict(name='embeddings-'+name, wire_hex=(frame(1, invalid)+frame(4)).hex(),
                         expected=dict(outcome='reject', reason='invalid_metadata')))

    error = dict(code='unsupported_representation', message='Input embedding source unavailable')
    partial = struct.pack('<f', -7.5)
    for prefix, body, extra in [('early', b'', {}),
                              ('partial', frame(1, m)+frame(2, partial),
                               dict(incomplete_data_hex=partial.hex()))]:
        for outcome, terminal in [('error', frame(5, error)), ('cancelled', frame(6))]:
            wire.append(dict(name=f'embeddings-{prefix}-{outcome}', wire_hex=(body+terminal).hex(),
                             expected=dict(outcome=outcome, **extra,
                                           **(dict(error=error) if outcome == 'error' else {}))))
    raw = struct.pack('<6f', -7.5, 8., 9.25, 1., -2., 3.5)
    meta = frame(1, m)
    valid = meta+frame(2, raw)+frame(4)
    for name, body, reason in [
        ('truncated-header', valid[:-1], 'truncated_header'),
        ('truncated-data', meta+header(2, len(raw))+raw[:-1], 'truncated_payload'),
        ('short-result', meta+frame(2, partial)+frame(4), 'data_length_mismatch'),
        ('overrun', meta+frame(2, raw+partial)+frame(4), 'data_overrun'),
        ('unaligned', meta+frame(2, b'abc')+frame(4), 'unaligned_data'),
        ('duplicate-meta', meta+valid, 'duplicate_metadata'),
        ('missing-terminal', meta+frame(2, raw), 'missing_terminal'),
        ('trailing-byte', valid+b'x', 'trailing_bytes'),
    ]:
        wire.append(dict(name='embeddings-'+name, wire_hex=body.hex(),
                         expected=dict(outcome='reject', reason=reason)))

    # A valid stream can still belong to a different originating request.
    associations = [dict(name='exact-order', request=dict(token_ids=[2, 0]), metadata=m, valid=True),
                    dict(name='reordered-ids', request=dict(token_ids=[0, 2]), metadata=m, valid=False),
                    dict(name='changed-duplicate', request=dict(token_ids=[2, 2]), metadata=m, valid=False)]
    for case in associations:
        require((case['request']['token_ids'] == case['metadata']['token_ids']) == case['valid'],
                'request association oracle')
    # Bound JSON metadata without creating a bulky checked-in fixture.
    oversized = metadata([SAFE]*65536)
    try:
        validate_instance(document, 'InputEmbeddingsMetadata', oversized)
    except ValueError as exc:
        require(str(exc) == 'unsupported_size', 'metadata limit oracle')
    else:
        raise ValueError('oversized embedding metadata accepted')
    return dict(source=dict(vocabulary_size=len(table), hidden_size=3, input_table=table),
                request_cases=requests, schema_cases=instances, wire_cases=wire,
                association_cases=associations)


def embedding_analysis_fixtures():
    """Small independently gathered matrices; never use a backend implementation."""
    cases, wire, schemas = [], [], []
    extreme = struct.unpack('<f', bytes.fromhex('ffff7f7f'))[0]
    table = [-4., 0., 4., -1000., 0., 1000., 1., 2., 3.]
    for name, source, ids in [
        ('duplicates', table, [2, 0, 2]), ('single', table, [2]), ('empty', table, []),
        ('constant', [3.5]*9, [2, 0, 2]),
        ('mixed', [math.nan, math.inf, -math.inf, -1000., 0., 1000., -4., 1., 4.], [2, 0, 2]),
        ('nonfinite', [math.nan, math.inf, -math.inf]*3, [2, 0, 2]),
        ('extremes', [extreme, 1., -extreme]*3, [2, 0, 2]),
    ]:
        values = [value for token in ids for value in source[token*3:token*3+3]]
        stats = statistics(values)
        stats.pop('tensor_id')
        stats.update(kind='input_embeddings_statistics', token_ids=ids, shape=[len(ids), 3])
        dist, payload, _, _ = distributions(len(ids), 3, values)
        dist.pop('tensor_id')
        dist.update(kind='input_embeddings_distributions', token_ids=ids)
        cases.append(dict(name=name, table_shape=[3, 3],
                          table_data_hex=struct.pack('<9f', *source).hex(), token_ids=ids,
                          values_hex=struct.pack(f'<{len(values)}f', *values).hex(),
                          statistics=stats, distributions=dist, counts_hex=payload.hex()))
        for suffix, metadata, data in [('statistics', stats, b''), ('distributions', dist, payload)]:
            wire.append(dict(name=f'embedding-{suffix}-{name}',
                             wire_hex=(frame(1, metadata)+(frame(2, data) if data else b'')+frame(4)).hex(),
                             expected=dict(outcome='complete', metadata=metadata, data_hex=data.hex())))
            schemas.append(dict(name=f'embedding-{suffix}-{name}', schema='StreamMetadata',
                                valid=True, value=metadata))

    base = cases[0]
    for suffix in ['statistics', 'distributions']:
        metadata = base[suffix]
        data = bytes.fromhex(base['counts_hex']) if suffix == 'distributions' else b''
        meta = frame(1, metadata)
        valid = meta+(frame(2, data) if data else b'')+frame(4)
        for name, body in [
            ('missing-terminal', valid[:-12]), ('duplicate-terminal', valid+frame(4)),
            ('trailing-byte', valid+b'x'), ('truncated-header', valid[:-1]),
            ('duplicate-meta', meta+valid),
            ('invalid-data', meta+frame(2, b'')+frame(4) if not data else meta+frame(2, data[:-4])+frame(4)),
            ('truncated-data', meta+header(2, 4)+b'xx'),
            ('overrun', meta+frame(2, data+b'xxxx')+frame(4)),
        ]:
            wire.append(dict(name=f'embedding-{suffix}-{name}', wire_hex=body.hex(),
                             expected=dict(outcome='reject')))
        error = dict(code='resource_exhausted', message='Insufficient resources')
        for name, terminal, outcome in [('cancelled', frame(6), 'cancelled'),
                                         ('error', frame(5, error), 'error')]:
            for stage, prefix in [('early', b''), ('after-meta', meta)]:
                wire.append(dict(name=f'embedding-{suffix}-{stage}-{name}',
                                 wire_hex=(prefix+terminal).hex(),
                                 expected=dict(outcome=outcome)))
            if data:
                wire.append(dict(name=f'embedding-{suffix}-partial-{name}',
                                 wire_hex=(meta+frame(2, data[:4])+terminal).hex(),
                                 expected=dict(outcome=outcome)))

        edits = [('row-count', dict(token_ids=[2, 0])), ('negative-id', dict(token_ids=[2, -1, 2])),
                 ('checkpoint-id', dict(tensor_id='t'))]
        if suffix == 'statistics':
            edits += [('count', dict(count=8)), ('shape', dict(shape=[1, 9])),
                      ('hidden-zero', dict(shape=[3, 0])), ('unsafe-product', dict(shape=[3, SAFE])),
                      ('mean-outside-domain', dict(mean=100)), ('data-length', dict(byte_length=4))]
        else:
            edits += [('hidden-zero', dict(columns=0)), ('unsafe-product', dict(columns=SAFE)),
                      ('section-length', dict(sections=[dict(metadata['sections'][0], byte_length=4),
                                                        metadata['sections'][1]])),
                      ('section-offset', dict(sections=[metadata['sections'][0],
                                                       dict(metadata['sections'][1], offset=0)]))]
        for name, update in edits:
            invalid = dict(metadata, **update)
            schemas.append(dict(name=f'embedding-{suffix}-{name}', schema='StreamMetadata',
                                valid=False, value=invalid))
            wire.append(dict(name=f'embedding-{suffix}-{name}',
                             wire_hex=(frame(1, invalid)+frame(4)).hex(),
                             expected=dict(outcome='reject')))
    # Contextual identity is intentionally stricter than a valid metadata shape.
    associations = [dict(token_ids=ids, valid=ids == [2, 0, 2])
                    for ids in ([2, 0, 2], [0, 2, 2], [2, 0, 0], [2], [])]
    return dict(numerical_cases=cases, association_cases=associations,
                schema_cases=schemas, wire_cases=wire)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true', help='Regenerate committed golden fixture')
    args = parser.parse_args()
    document = yaml.safe_load(CONTRACT.read_text())
    OpenAPIV31SpecValidator(document).validate()
    references = resolve_references(document)
    for schema in document['components']['schemas'].values():
        Draft202012Validator.check_schema(schema)
    expected_operations = {'listModels', 'createSession', 'getSession', 'deleteSession',
                           'listTensors', 'streamTensor', 'streamTensorStatistics',
                           'streamTensorDistributions', 'streamInputEmbeddings',
                           'streamInputEmbeddingsStatistics', 'streamInputEmbeddingsDistributions', 'tokenize', 'cancelOperation', 'getArchitecture'}
    operations = [op for item in document['paths'].values() for method, op in item.items()
                  if method in ('get', 'post', 'delete')]
    require(len(operations) == 14 and {op['operationId'] for op in operations} == expected_operations,
            'operation set changed')
    golden = fixtures()
    embeddings = embedding_fixtures(document)
    embedding_analysis = embedding_analysis_fixtures()
    architecture = architecture_fixtures()
    for case in architecture['cases']:
        value = apply_edits(architecture['response'], case['edits'])
        context = apply_edits(architecture['context'], case['context_edits'])
        validator = Draft202012Validator(
            {'$ref': '#/components/schemas/ArchitectureResponse', 'components': document['components']})
        require(validator.is_valid(value) == case['schema_valid'],
                f"architecture schema expectation: {case['name']}")
        try:
            validator.validate(value)
            validate_instance(document, 'Session', context['session'])
            validate_instance(document, 'TensorInventory', context['inventory'])
            validate_architecture(value, context)
        except (ValidationError, ValueError):
            require(not case['valid'], f"valid architecture rejected: {case['name']}")
        else:
            require(case['valid'], f"invalid architecture accepted: {case['name']}")
    for case in architecture['inventory_cases']:
        try:
            validate_instance(document, 'TensorInventory', case['value'])
        except (ValidationError, ValueError):
            require(not case['valid'], f"valid inventory rejected: {case['name']}")
        else:
            require(case['valid'], f"invalid inventory accepted: {case['name']}")
    for case in architecture['byte_cases']:
        def chunks():
            chunk = b'x' * case['chunk_bytes']
            for _ in range(case['repeat']):
                yield chunk
            yield b'x' * case['tail_bytes']
            if not case['valid']:
                raise AssertionError('oversized input was consumed past rejection')
        try:
            bounded_size(chunks())
        except ValueError:
            require(not case['valid'], 'valid size rejected')
        else:
            require(case['valid'], 'oversize accepted')
    endpoint = document['paths']['/sessions/{session_id}/architecture']['get']
    require(set(endpoint['responses']) == {'200', '404', '409', '422', '503', '500'},
            'architecture HTTP response set')
    require('requestBody' not in endpoint and endpoint['operationId'] == 'getArchitecture',
            'architecture retrieval contract')
    for case in architecture['http_cases']:
        response = document['components']['responses']['ModelChanged']
        Draft202012Validator({**response['content']['application/json']['schema'],
                              'components': document['components']}).validate(case['value'])
    for case in golden['schema_cases'] + embeddings['schema_cases'] + embedding_analysis['schema_cases']:
        try:
            validate_instance(document, case['schema'], case['value'])
        except (ValidationError, ValueError):
            require(not case['valid'], f"valid instance rejected: {case['name']}")
        else:
            require(case['valid'], f"invalid instance accepted: {case['name']}")
    require(struct.pack('<3I', *golden['numeric_cases']['uint32']['values']).hex() ==
            golden['numeric_cases']['uint32']['little_endian_hex'], 'uint32 endian oracle')
    # Literal framing oracle independent from struct.pack arguments.
    require(frame(6).hex() == '4c4d45580600000000000000', 'header oracle')
    for path, fixture in [(GOLDEN, golden), (EMBEDDINGS_GOLDEN, embeddings),
                          (ARCHITECTURE_GOLDEN, architecture), (EMBEDDING_ANALYSIS_GOLDEN, embedding_analysis)]:
        rendered = json.dumps(fixture, ensure_ascii=False, allow_nan=False, indent=2) + '\n'
        if args.write:
            path.write_text(rendered)
        else:
            require(path.exists() and path.read_text() == rendered,
                    f'{path.name} differs; run api/validate_contract.py --write')
    print(f"OpenAPI 3.1 valid; {references} references resolved; "
          f"{len(golden['schema_cases']) + len(embeddings['schema_cases']) + len(embedding_analysis['schema_cases'])} instance cases checked; "
          f"{len(architecture['cases'])} architecture cases checked; "
          f"{len(golden['wire_cases']) + len(embeddings['wire_cases']) + len(embedding_analysis['wire_cases'])} wire fixtures "
          f"{'written' if args.write else 'reproducible'}.")


if __name__ == '__main__':
    main()
