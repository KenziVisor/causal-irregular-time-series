from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from majority_vote_latents import (  # noqa: E402
    align_voters_on_shared_ts_ids,
    build_majority_vote_dataframe,
    load_latent_csv,
)


def test_equal_voter_cohorts_reorder_only_after_equality() -> None:
    reference = pd.DataFrame(
        {"ts_id": ["10", "20"], "LAT_A": [0, 1]}
    )
    reordered = pd.DataFrame(
        {"ts_id": ["20.0", 10.0], "LAT_A": [1, 0]}
    )

    aligned, final_ids, dropped = align_voters_on_shared_ts_ids(
        [reference, reordered],
        [Path("reference.csv"), Path("reordered.csv")],
        ["LAT_A"],
    )

    assert final_ids == ["10", "20"]
    assert dropped == [0, 0]
    assert aligned[0]["ts_id"].tolist() == ["10", "20"]
    assert aligned[1].to_dict("records") == [
        {"ts_id": "10", "LAT_A": 0},
        {"ts_id": "20", "LAT_A": 1},
    ]


def test_partial_overlap_fails_with_safe_counts_and_no_intersection() -> None:
    reference = pd.DataFrame(
        {"ts_id": ["111111", "222222"], "LAT_A": [0, 1]}
    )
    mismatch = pd.DataFrame(
        {"ts_id": ["222222", "333333"], "LAT_A": [1, 0]}
    )

    with pytest.raises(ValueError) as exc_info:
        align_voters_on_shared_ts_ids(
            [reference, mismatch],
            [Path("reference.csv"), Path("mismatch.csv")],
            ["LAT_A"],
        )

    message = str(exc_info.value)
    assert "missing_count=1" in message
    assert "extra_count=1" in message
    assert "111111" not in message
    assert "222222" not in message
    assert "333333" not in message


def test_csv_load_rejects_duplicates_after_id_canonicalization(
    tmp_path: Path,
) -> None:
    path = tmp_path / "duplicate.csv"
    path.write_text("ts_id,LAT_A\n10,0\n10.0,1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate ts_id values after canonicalization"):
        load_latent_csv(path)


@pytest.mark.parametrize("bad_id", ["010", "1e1", "10.5", "inf", "abc", ""])
def test_csv_load_preserves_and_rejects_invalid_id_lexemes(
    tmp_path: Path,
    bad_id: str,
) -> None:
    path = tmp_path / "invalid.csv"
    path.write_text(f"ts_id,LAT_A\n{bad_id},1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid ts_id"):
        load_latent_csv(path)


def test_empty_voter_cohort_fails() -> None:
    empty = pd.DataFrame(columns=["ts_id", "LAT_A"])
    with pytest.raises(ValueError, match="must not be empty"):
        align_voters_on_shared_ts_ids(
            [empty],
            [Path("empty.csv")],
            ["LAT_A"],
        )


def test_even_voter_tie_remains_positive() -> None:
    reference = pd.DataFrame({"ts_id": ["10"], "LAT_A": [0]})
    second = pd.DataFrame({"ts_id": ["10"], "LAT_A": [1]})

    output = build_majority_vote_dataframe(
        reference,
        [reference, second],
        ["LAT_A"],
    )

    assert output.to_dict("records") == [{"ts_id": "10", "LAT_A": 1}]
