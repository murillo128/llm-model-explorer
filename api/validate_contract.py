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

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / 'docs/spec/api/openapi.yaml'
GOLDEN = ROOT / 'api/fixtures/conformance.json'
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
                  'TensorDistributionsMetadata'):
        kind = value['kind']
        if kind == 'tensor':
            size = 4 * product(value['shape'])
            require(size <= SAFE and value['byte_length'] == size, 'tensor byte length')
        elif kind == 'tensor_statistics':
            require(value['count'] == value['finite_count'] + value['non_finite_count'],
                    'statistics count sum')
            if value['finite_count']:
                lo, hi = value['minimum'], value['maximum']
                ps = list(value['percentiles'][p] for p in ('p01', 'p05', 'p50', 'p95', 'p99'))
                require(lo <= value['mean'] <= hi and lo <= hi, 'statistics range')
                require([lo, *ps, hi] == sorted([lo, *ps, hi]), 'percentile order')
        elif kind == 'tensor_distributions':
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
                           'streamTensorDistributions', 'tokenize', 'cancelOperation'}
    operations = [op for item in document['paths'].values() for method, op in item.items()
                  if method in ('get', 'post', 'delete')]
    require(len(operations) == 10 and {op['operationId'] for op in operations} == expected_operations,
            'operation set changed')
    golden = fixtures()
    for case in golden['schema_cases']:
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
    rendered = json.dumps(golden, ensure_ascii=False, allow_nan=False, indent=2) + '\n'
    if args.write:
        GOLDEN.write_text(rendered)
    else:
        require(GOLDEN.exists() and GOLDEN.read_text() == rendered,
                'golden fixture differs; run api/validate_contract.py --write')
    print(f"OpenAPI 3.1 valid; {references} references resolved; "
          f"{len(golden['schema_cases'])} instance cases checked; "
          f"{len(golden['wire_cases'])} wire fixtures {'written' if args.write else 'reproducible'}.")


if __name__ == '__main__':
    main()
