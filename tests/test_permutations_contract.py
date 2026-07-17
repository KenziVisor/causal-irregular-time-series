from __future__ import annotations

from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import permutations_test as permutations  # noqa: E402


def _write_payload(
    path: Path,
    *,
    oc: pd.DataFrame | None = None,
) -> Path:
    ts = pd.DataFrame(
        {
            "ts_id": ["10", "11.0"],
            "minute": [0, 0],
            "variable": ["HR", "HR"],
            "value": [80.0, 90.0],
        }
    )
    outcomes = oc if oc is not None else pd.DataFrame(
        {
            "ts_id": [10, "11.0"],
            "in_hospital_mortality": [0, 1],
        }
    )
    with path.open("wb") as handle:
        pickle.dump((ts, outcomes, ["10", 11]), handle)
    return path


def test_latent_preflight_preserves_text_and_requires_exact_cohort(
    tmp_path: Path,
) -> None:
    payload_path = _write_payload(tmp_path / "processed.pkl")
    latent_path = tmp_path / "latents.csv"
    latent_path.write_text(
        "ts_id,LAT_A\n10.0,1\n11,0\n",
        encoding="utf-8",
    )

    latent_df = permutations.load_latent_tags_dataframe(
        str(latent_path),
        str(payload_path),
    )
    assert latent_df["ts_id"].tolist() == ["10", "11"]

    latent_path.write_text(
        "ts_id,LAT_A\n10,1\n12,0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing_count=1; extra_count=1"):
        permutations.load_latent_tags_dataframe(
            str(latent_path),
            str(payload_path),
        )


def test_outcome_shuffle_rejects_conflicts_before_destination_write(
    tmp_path: Path,
) -> None:
    conflicting_oc = pd.DataFrame(
        {
            "ts_id": ["10", "10.0"],
            "in_hospital_mortality": [0, 1],
        }
    )
    payload_path = _write_payload(
        tmp_path / "conflicting.pkl",
        oc=conflicting_oc,
    )
    destination = tmp_path / "shuffled.pkl"

    with pytest.raises(ValueError, match="conflicting duplicate ts_id"):
        permutations.shuffle_outcome_column(
            str(payload_path),
            str(destination),
            np.random.default_rng(0),
        )

    assert not destination.exists()
