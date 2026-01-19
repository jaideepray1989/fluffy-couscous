import json
from pathlib import Path

from qd_suite.adapters.example_dummy import ExampleDummyAdapter
from qd_suite.eval.runner import run_suite, synthetic_samples


def test_runner_smoke(tmp_path: Path):
    spec_path = Path("qd_suite/eval/default_spec.json")
    spec = json.loads(spec_path.read_text())
    samples = synthetic_samples()
    adapter = ExampleDummyAdapter()
    metrics, summary, curves = run_suite(samples, adapter, spec)
    assert "clean_acc" in metrics["clean"]
    assert "overall_score" in summary
    assert isinstance(curves, dict)
