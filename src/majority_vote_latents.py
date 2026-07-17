from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd

from preprocess_mimic_iii_large_contract import canonicalize_mimic_id_series


DEFAULT_OUTPUT_PATH = Path("./latent_tags_majority_vote.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build one majority-vote latent CSV from a folder of latent-tag voter CSV files."
        )
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing the latent-tag voter CSV files.",
    )
    parser.add_argument(
        "--output-path",
        default=str(DEFAULT_OUTPUT_PATH),
        help=f"Output CSV path. Default: {DEFAULT_OUTPUT_PATH}",
    )
    return parser.parse_args()


def resolve_input_dir(path_like: str) -> Path:
    input_dir = Path(path_like).expanduser().resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")
    return input_dir


def resolve_output_path(path_like: str) -> Path:
    output_path = Path(path_like).expanduser().resolve()
    if output_path.exists() and output_path.is_dir():
        raise IsADirectoryError(f"Output path points to a directory, not a CSV file: {output_path}")
    return output_path


def discover_csv_files(input_dir: Path, output_path: Path) -> list[Path]:
    candidates: list[Path] = []
    for child in input_dir.iterdir():
        if not child.is_file():
            continue
        if child.name.startswith("."):
            continue
        if child.suffix.lower() != ".csv":
            continue
        child_resolved = child.resolve()
        if child_resolved == output_path:
            continue
        candidates.append(child_resolved)

    candidates = sorted(candidates, key=lambda path: str(path))
    if not candidates:
        raise ValueError(
            f"No candidate CSV files were found in input directory: {input_dir}"
        )
    return candidates


def load_latent_csv(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, dtype=str)
    except Exception as exc:
        raise ValueError(f"Failed to read CSV file: {path} ({exc})") from exc

    stripped_columns = [str(column).strip() for column in df.columns]
    if any(column == "" for column in stripped_columns):
        raise ValueError(
            f"CSV file has an empty column name after stripping whitespace: {path}"
        )
    if len(set(stripped_columns)) != len(stripped_columns):
        raise ValueError(
            f"CSV file has duplicate column names after stripping whitespace: {path}. "
            f"Columns={stripped_columns}"
        )

    df = df.copy()
    df.columns = stripped_columns

    if "ts_id" not in df.columns:
        raise ValueError(f"CSV file is missing required 'ts_id' column: {path}")

    try:
        df["ts_id"] = canonicalize_mimic_id_series(
            df["ts_id"],
            field_name=f"{path}.ts_id",
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"CSV file contains an invalid ts_id value: {path}. {error}") from error

    duplicate_mask = df["ts_id"].duplicated(keep=False)
    if duplicate_mask.any():
        raise ValueError(
            f"CSV file contains duplicate ts_id values after canonicalization: {path}. "
            f"Duplicate row count={int(duplicate_mask.sum())}."
        )

    latent_columns = [column for column in df.columns if column != "ts_id"]
    if not latent_columns:
        raise ValueError(
            f"CSV file does not contain any latent columns beyond 'ts_id': {path}"
        )

    return df


def coerce_binary_latent_columns(
    df: pd.DataFrame,
    file_path: Path,
    latent_columns: Sequence[str],
) -> pd.DataFrame:
    out = df.copy()

    for column in latent_columns:
        numeric_values = pd.to_numeric(out[column], errors="coerce")
        invalid_mask = numeric_values.isna() | ~numeric_values.isin([0, 1])
        if invalid_mask.any():
            first_bad_position = next(
                position for position, is_bad in enumerate(invalid_mask.tolist(), start=1)
                if is_bad
            )
            offending_values = out.loc[invalid_mask, column].head(5).tolist()
            raise ValueError(
                f"Non-binary latent values found in file {file_path}, column {column!r}. "
                f"First offending data row={first_bad_position}. "
                f"Example offending values={offending_values}"
            )
        out[column] = numeric_values.astype(int)

    return out


def validate_reference_dataframe(
    df: pd.DataFrame,
    file_path: Path,
) -> tuple[pd.DataFrame, list[str]]:
    latent_columns = [column for column in df.columns if column != "ts_id"]
    validated_df = coerce_binary_latent_columns(df, file_path, latent_columns)
    return validated_df, latent_columns


def validate_against_reference(
    df: pd.DataFrame,
    file_path: Path,
    reference_file_path: Path,
    reference_latent_columns: Sequence[str],
) -> pd.DataFrame:
    current_latent_columns = [column for column in df.columns if column != "ts_id"]
    current_latent_set = set(current_latent_columns)
    reference_latent_set = set(reference_latent_columns)

    if current_latent_set != reference_latent_set:
        missing_columns = [
            column for column in reference_latent_columns if column not in current_latent_set
        ]
        extra_columns = sorted(current_latent_set - reference_latent_set)
        raise ValueError(
            f"Latent column set mismatch in file {file_path}. "
            f"Reference file={reference_file_path}. "
            f"Missing columns={missing_columns}. Extra columns={extra_columns}."
        )

    validated_df = coerce_binary_latent_columns(df, file_path, current_latent_columns)
    return validated_df.loc[:, ["ts_id", *reference_latent_columns]].copy()


def align_voters_on_shared_ts_ids(
    voter_dfs: Sequence[pd.DataFrame],
    voter_file_paths: Sequence[Path],
    latent_columns: Sequence[str],
) -> tuple[list[pd.DataFrame], list[str], list[int]]:
    """Prove exact cohort equality, then reorder every voter to the reference."""
    if not voter_dfs:
        raise ValueError("No validated voter dataframes were provided for alignment.")
    if len(voter_dfs) != len(voter_file_paths):
        raise ValueError("Each voter dataframe must have one corresponding file path.")

    latent_columns = list(latent_columns)
    canonical_voters: list[pd.DataFrame] = []
    for voter_df, voter_file_path in zip(voter_dfs, voter_file_paths):
        if "ts_id" not in voter_df.columns:
            raise ValueError(f"Voter dataframe is missing ts_id: {voter_file_path}")
        missing_latents = [
            column for column in latent_columns if column not in voter_df.columns
        ]
        if missing_latents:
            raise ValueError(
                f"Voter dataframe is missing latent columns: {voter_file_path}. "
                f"Missing columns={missing_latents}."
            )
        canonical_voter = voter_df.loc[:, ["ts_id", *latent_columns]].copy()
        canonical_voter["ts_id"] = canonicalize_mimic_id_series(
            canonical_voter["ts_id"],
            field_name=f"{voter_file_path}.ts_id",
        )
        duplicate_mask = canonical_voter["ts_id"].duplicated(keep=False)
        if duplicate_mask.any():
            raise ValueError(
                f"Voter dataframe contains duplicate ts_id values: {voter_file_path}. "
                f"Duplicate row count={int(duplicate_mask.sum())}."
            )
        canonical_voters.append(canonical_voter)

    reference_ts_ids = canonical_voters[0]["ts_id"].tolist()
    if not reference_ts_ids:
        raise ValueError("Voter cohorts must not be empty.")
    reference_set = set(reference_ts_ids)

    for voter_index, (voter_df, voter_file_path) in enumerate(
        zip(canonical_voters[1:], voter_file_paths[1:]),
        start=2,
    ):
        current_set = set(voter_df["ts_id"])
        missing_count = len(reference_set - current_set)
        extra_count = len(current_set - reference_set)
        if missing_count or extra_count:
            raise ValueError(
                f"Voter cohort mismatch for voter {voter_index}: {voter_file_path}. "
                f"missing_count={missing_count}; extra_count={extra_count}."
            )

    aligned_voter_dfs: list[pd.DataFrame] = []
    for voter_df in canonical_voters:
        aligned_voter_df = (
            voter_df.set_index("ts_id")
            .loc[reference_ts_ids, latent_columns]
            .reset_index()
        )
        aligned_voter_dfs.append(
            aligned_voter_df.loc[:, ["ts_id", *latent_columns]].copy()
        )

    return aligned_voter_dfs, reference_ts_ids, [0] * len(aligned_voter_dfs)


def build_majority_vote_dataframe(
    reference_df: pd.DataFrame,
    voter_dfs: Sequence[pd.DataFrame],
    latent_columns: Sequence[str],
) -> pd.DataFrame:
    n_voters = len(voter_dfs)
    ones_count = pd.DataFrame(
        0,
        index=reference_df.index,
        columns=list(latent_columns),
        dtype=int,
    )

    for voter_df in voter_dfs:
        ones_count = ones_count.add(voter_df.loc[:, latent_columns], fill_value=0)

    majority_values = (2 * ones_count >= n_voters).astype(int)
    output_df = pd.concat([reference_df.loc[:, ["ts_id"]].copy(), majority_values], axis=1)
    return output_df


def main() -> None:
    args = parse_args()
    input_dir = resolve_input_dir(args.input_dir)
    output_path = resolve_output_path(args.output_path)

    print("=== Building majority-vote latent CSV ===")
    print(f"[1/5] Discovering voter CSV files in: {input_dir}")
    csv_files = discover_csv_files(input_dir, output_path)
    print(f"Found {len(csv_files)} candidate CSV files")

    reference_file = csv_files[0]
    print(f"[2/5] Loading and validating reference file: {reference_file}")
    reference_df = load_latent_csv(reference_file)
    reference_df, reference_latent_columns = validate_reference_dataframe(
        reference_df,
        reference_file,
    )
    print(
        f"Reference rows before alignment: {len(reference_df):,} | "
        f"latent columns: {len(reference_latent_columns)}"
    )

    voter_dfs: list[pd.DataFrame] = [
        reference_df.loc[:, ["ts_id", *reference_latent_columns]].copy()
    ]
    voter_files: list[Path] = [reference_file]

    print("[3/5] Loading and validating remaining voter files")
    if len(csv_files) == 1:
        print(
            f"Only one voter CSV found: {reference_file} | "
            f"rows before alignment: {len(reference_df):,}"
        )
    else:
        for voter_index, voter_file in enumerate(csv_files[1:], start=2):
            voter_df = load_latent_csv(voter_file)
            validated_voter_df = validate_against_reference(
                voter_df,
                voter_file,
                reference_file,
                reference_latent_columns,
            )
            voter_dfs.append(validated_voter_df)
            voter_files.append(voter_file)
            print(
                f"Validated voter {voter_index}/{len(csv_files)}: {voter_file} | "
                f"rows before alignment: {len(validated_voter_df):,}"
            )

    print("[4/5] Proving exact voter cohort equality")
    aligned_voter_dfs, final_ts_ids, _ = align_voters_on_shared_ts_ids(
        voter_dfs=voter_dfs,
        voter_file_paths=voter_files,
        latent_columns=reference_latent_columns,
    )
    print(
        f"Identical ts_id cohort across {len(aligned_voter_dfs)} voters: "
        f"{len(final_ts_ids):,}"
    )

    reference_df = aligned_voter_dfs[0]
    voter_dfs = aligned_voter_dfs

    print("[5/5] Computing and saving majority vote")
    print(
        f"Voting across {len(voter_dfs)} files, {len(reference_df):,} cohort members, "
        f"{len(reference_latent_columns)} latent columns"
    )
    output_df = build_majority_vote_dataframe(
        reference_df=reference_df,
        voter_dfs=voter_dfs,
        latent_columns=reference_latent_columns,
    )

    print(f"Saving output to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    print(f"Saved majority-vote latent CSV: {output_path}")
    print("Done")


if __name__ == "__main__":
    main()
