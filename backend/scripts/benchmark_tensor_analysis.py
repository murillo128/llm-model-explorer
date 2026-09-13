"""One-process synthetic CPU timing/RSS probe; no fixtures or cache files written."""

import argparse
import json
import resource
import time

import torch

from llm_model_explorer.operations import Cancellation
from llm_model_explorer.tensor_analysis import distributions, statistics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["statistics", "distributions"])
    parser.add_argument("rows", type=int)
    parser.add_argument("columns", type=int)
    args = parser.parse_args()
    count = args.rows * args.columns
    values = torch.randn(count, generator=torch.Generator().manual_seed(9), dtype=torch.float32)
    baseline = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    start = time.perf_counter()
    if args.kind == "statistics":
        statistics(values, Cancellation())
    else:
        distributions(values, args.rows, args.columns, Cancellation())
    elapsed = time.perf_counter() - start
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(
        json.dumps(
            dict(
                kind=args.kind,
                shape=[args.rows, args.columns],
                elements=count,
                torch=torch.__version__,
                threads=torch.get_num_threads(),
                seconds=round(elapsed, 4),
                input_bytes=count * 4,
                baseline_rss_kib=baseline,
                peak_rss_kib=peak,
                peak_increase_kib=peak - baseline,
            )
        )
    )


if __name__ == "__main__":
    main()
