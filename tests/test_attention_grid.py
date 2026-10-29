"""Offline, non-executing prospective grid validation."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("attention_grid_tool", ROOT / 'scripts/make_attention_grid.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_grid_contains_all_registered_runs_and_no_test_access():
    grid = module.make_grid()
    assert len(grid["runs"]) == 12
    assert len({r["id"] for r in grid["runs"]}) == 12
    for run in grid["runs"]:
        args = run["esc50_args"]
        assert args[0] == "train"
        assert "--allow-test" not in args
        assert "--specaugment" not in args
        assert args[args.index("--test-fold") + 1] == "5"
        assert args[args.index("--val-fold") + 1] == "4"
        assert args[args.index("--train-folds") + 1] == "1,2,3"


def test_grid_does_not_actually_run_and_fails_closed_on_device():
    try:
        module.make_grid(device="invalid")
    except ValueError as exc:
        assert "device" in str(exc)
    else:
        assert False, "must reject unknown device"
