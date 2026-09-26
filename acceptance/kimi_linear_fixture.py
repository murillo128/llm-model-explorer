"""Small local shell for exercising the complete Kimi graph over production HTTP."""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "backend/tests/fixtures/kimi-linear-reference.json"


def generate(root: Path) -> None:
    """Write a tiny inert checkpoint; the adapter supplies reviewed Kimi metadata."""
    from acceptance.architecture_fixtures import write_checkpoint

    fixture = json.loads(FIXTURE.read_text())
    configuration = dict(fixture["configuration"])
    # This tiny shell has one native tensor, not the reference model's packed
    # inventory. The production description receives its full independent
    # metadata fixture in install_test_adapter below.
    configuration.pop("quantization_config", None)
    write_checkpoint(
        root / "kimi_linear",
        configuration,
        {"model.norm.weight": {"dtype": "BF16", "shape": [2304]}},
    )
    (root / "kimi_linear/modeling_kimi.py").write_text(
        'raise AssertionError("checkpoint modeling code executed")\n'
    )
    (root / "kimi_linear/configuration_kimi.py").write_text(
        'raise AssertionError("checkpoint configuration code executed")\n'
    )


def install_test_adapter() -> None:
    """Use the pinned independent physical-metadata oracle for this test model."""
    tests = str(REPO / "backend/tests")
    if tests not in sys.path:
        sys.path.insert(0, tests)
    from llm_model_explorer.architecture_analysis.core import AnalysisInput
    from test_kimi_linear_architecture import compressed_expert_inputs

    original = AnalysisInput.from_source.__func__

    def from_source(cls, source, *, tokenizer_available: bool):
        if source.model_id != "kimi_linear":
            return original(cls, source, tokenizer_available=tokenizer_available)
        source.check_unchanged()
        configuration = source.configuration()
        expected = compressed_expert_inputs()
        return cls(source.fingerprint, configuration, expected.bindings)

    AnalysisInput.from_source = classmethod(from_source)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    generate(args.root)
