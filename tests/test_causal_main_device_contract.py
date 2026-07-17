from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_causal_main():
    module_name = "_clinicause_causal_main_test"
    spec = importlib.util.spec_from_file_location(module_name, ROOT / "main.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_mortality_command_forwards_effective_device() -> None:
    causal_main = _load_causal_main()
    context = SimpleNamespace(
        dataset="physionet",
        dataset_config_csv=Path("config.csv"),
        dataset_pkl_path=Path("dataset.pkl"),
        mortality_device="cpu",
        stages={
            "majority_vote": SimpleNamespace(
                output_paths={"majority_vote_csv": "latents.csv"}
            ),
            "mortality_prediction": SimpleNamespace(
                script_path="mortality.py",
                output_paths={"results_txt": "results.txt"},
            ),
        },
    )

    command = causal_main.build_mortality_command(context)

    assert command[command.index("--device") + 1] == "cpu"
