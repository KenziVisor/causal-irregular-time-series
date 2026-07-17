from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset_config import (  # noqa: E402
    get_config_bool,
    get_config_float,
    get_config_int,
    get_config_list,
    load_dataset_config,
    resolve_config_seed,
    resolve_with_precedence,
)


def test_load_physionet_config() -> None:
    config = load_dataset_config("physionet")
    assert isinstance(config["TREATMENTS"], list)
    assert isinstance(config["BACKGROUND_FEATURE_COLUMNS"], list)
    assert "DEFAULT_THRESHOLDS" not in config
    assert "LAT_GLOBAL_SEVERITY" in config["TREATMENTS"]
    assert "Age" in config["BACKGROUND_FEATURE_COLUMNS"]


def test_load_mimic_config() -> None:
    config = load_dataset_config("mimic")
    assert isinstance(config["TREATMENTS"], list)
    assert isinstance(config["BACKGROUND_FEATURE_COLUMNS"], list)
    assert "DEFAULT_THRESHOLDS" not in config
    assert "PICKLE_TS_SUMMARY_SPECS" not in config
    assert "LAT_GLOBAL_SEVERITY" in config["TREATMENTS"]


def test_typed_helpers_and_precedence() -> None:
    config = {
        "__dataset__": "test",
        "__config_csv_path__": "/tmp/test.csv",
        "FLAG": True,
        "COUNT": 10,
        "RATE": 0.25,
        "ITEM": "one",
        "CSV_VALUE": "from_csv",
    }

    assert get_config_bool(config, "FLAG") is True
    assert get_config_int(config, "COUNT") == 10
    assert get_config_float(config, "RATE") == 0.25
    assert get_config_list(config, "ITEM") == ["one"]
    assert resolve_with_precedence("from_cli", config, "CSV_VALUE", "fallback") == "from_cli"
    assert resolve_with_precedence(None, config, "CSV_VALUE", "fallback") == "from_csv"
    assert resolve_with_precedence(None, config, "MISSING", "fallback") == "fallback"


@pytest.mark.parametrize(
    ("requested_dataset", "config_filename"),
    [
        ("mimic", "physionet-global-variables.csv"),
        ("physionet", "mimic-global-variables.csv"),
    ],
)
def test_config_identity_rejects_swapped_dataset_files(
    requested_dataset: str,
    config_filename: str,
) -> None:
    config_path = ROOT / "configs" / config_filename
    with pytest.raises(ValueError, match="DATASET_NAME.*requested dataset"):
        load_dataset_config(requested_dataset, config_path)


def test_resolve_config_seed_preserves_zero_and_defaults_only_on_none() -> None:
    base = {
        "__dataset__": "test",
        "__config_csv_path__": "/tmp/test.csv",
    }
    assert resolve_config_seed({**base, "SEED": 0}, 42) == 0
    assert resolve_config_seed({**base, "SEED": None}, 42) == 42


def main() -> None:
    test_load_physionet_config()
    test_load_mimic_config()
    test_typed_helpers_and_precedence()
    print("dataset_config checks passed.")


if __name__ == "__main__":
    main()
