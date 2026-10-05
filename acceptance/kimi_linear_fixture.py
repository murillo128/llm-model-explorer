"""Small local shell for exercising the complete Kimi graph over production HTTP."""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "backend/tests/fixtures/kimi-linear-reference.json"


def generate(root: Path, family: str = "kimi_linear") -> None:
    """Write a tiny inert checkpoint; the adapter supplies reviewed Kimi metadata."""
    from acceptance.architecture_fixtures import write_checkpoint

    fixture = json.loads(FIXTURE.read_text())
    configuration = dict(fixture["configuration"])
    if family == "deepseek_v2":
        sys.path.insert(0, str(REPO / "backend/tests"))
        from test_deepseek_v2_architecture import native_configuration

        configuration = native_configuration()
    # This tiny shell has one native tensor, not the reference model's packed
    # inventory. The production description receives its full independent
    # metadata fixture in install_test_adapter below.
    configuration.pop("quantization_config", None)
    write_checkpoint(
        root / family,
        configuration,
        {
            "model.norm.weight": {
                "dtype": "BF16",
                "shape": [configuration["hidden_size"]],
            }
        },
    )
    (root / family / "modeling_kimi.py").write_text(
        'raise AssertionError("checkpoint modeling code executed")\n'
    )
    (root / family / "configuration_kimi.py").write_text(
        'raise AssertionError("checkpoint configuration code executed")\n'
    )


def install_test_adapter() -> None:
    """Use the pinned independent physical-metadata oracle for this test model."""
    tests = str(REPO / "backend/tests")
    if tests not in sys.path:
        sys.path.insert(0, tests)
    from llm_model_explorer.architecture_analysis.core import AnalysisInput
    from test_deepseek_v2_architecture import inputs as deepseek_inputs
    from test_deepseek_v2_architecture import native_configuration
    from test_kimi_linear_architecture import compressed_expert_inputs

    original = AnalysisInput.from_source.__func__

    def from_source(cls, source, *, tokenizer_available: bool):
        if source.model_id not in {"kimi_linear", "deepseek_v2"}:
            return original(cls, source, tokenizer_available=tokenizer_available)
        source.check_unchanged()
        configuration = source.configuration()
        expected = (
            compressed_expert_inputs()
            if source.model_id == "kimi_linear"
            else deepseek_inputs(native_configuration())
        )
        return cls(
            source.fingerprint,
            configuration,
            replace(expected.bindings, tokenizer_available=tokenizer_available),
        )

    AnalysisInput.from_source = classmethod(from_source)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--family", choices=["kimi_linear", "deepseek_v2"], default="kimi_linear")
    args = parser.parse_args()
    generate(args.root, args.family)
