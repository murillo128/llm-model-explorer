"""Independent TCP oracles for the production browser's distribution fixtures."""

import math

import numpy as np
import pytest

from acceptance.fixtures import generate
from acceptance.test_network import Service, result


def test_scale_domains_and_exact_counts(tmp_path):
    generate(tmp_path / "models", extended=True)
    service = Service(tmp_path, model_root=tmp_path / "models")
    try:
        session = service.session()
        cases = {
            "concentrated": [-0.0001, 0, 0.0001, 0.0002] * 100,
            "outliers": [-1000] + [0] * 398 + [3000],
            "constant": [2] * 400,
            "nonfinite": [float("nan"), float("inf")] * 200,
        }
        for name, samples in cases.items():
            expected = np.asarray(samples, dtype="<f4").reshape(20, 20)
            path = service.path(session, f"scale.{name}.weight")
            _, payload = result(service, path)
            np.testing.assert_array_equal(np.frombuffer(payload, dtype="<f4"), expected.ravel())
            stats, _ = result(service, service.path(session, f"scale.{name}.weight", "statistics"))
            domain, counts = result(
                service, service.path(session, f"scale.{name}.weight", "distributions")
            )
            finite = expected[np.isfinite(expected)].astype(np.float64)
            low = float(finite.min()) if finite.size else None
            high = float(finite.max()) if finite.size else None
            assert (stats["minimum"], stats["maximum"]) == (low, high)
            assert (domain["domain_minimum"], domain["domain_maximum"]) == (low, high)
            assert stats["finite_count"] == finite.size
            rows, columns = np.zeros((20, 100), dtype="<u4"), np.zeros((100, 20), dtype="<u4")
            for (row, column), scalar in np.ndenumerate(expected):
                if not math.isfinite(scalar):
                    continue
                index = (
                    50
                    if low == high
                    else min(99, math.floor((float(scalar) - low) / (high - low) * 100))
                )
                rows[row, index] += 1
                columns[index, column] += 1
            assert counts == rows.tobytes() + columns.tobytes()
            if name == "outliers":
                assert stats["percentiles"]["p01"] == pytest.approx(0)
                assert stats["percentiles"]["p99"] == pytest.approx(0)
                assert low < 0 < high  # Display domain must not use the collapsed robust anchors.
    finally:
        service.stop()
        service.client.close()
