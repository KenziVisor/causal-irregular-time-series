# Runnable Scripts

Run examples from the repository root unless the entry says otherwise. Project Python commands that import the analysis stack should normally run in the WSL conda environment `econml310`. Relative defaults are fragile in this repository, so prefer explicit input and output paths for real runs.

## Active Entry Points

### `main.py`

- Purpose: Top-level post-preprocessing orchestrator for graph build, majority-vote latents, mortality, matching, CATE, saved-model analysis, and permutations.
- Inputs: `--dataset`, `--dataset-config-csv`, latent-tag voter CSV directory, and processed `[ts, oc, ts_ids]` pickle.
- Outputs: Stage subdirectories, `logs/`, and `run_summary.json` under `--output-dir`.
- Example: `conda run -n econml310 python main.py --dataset physionet --dataset-config-csv configs/physionet-global-variables.csv --latent-tags-dir /path/to/voter_csvs --dataset-pkl-path /path/to/physionet2012_ts_oc_ids.pkl --output-dir /path/to/main_run`

### `scripts/run_main.sh`

- Purpose: Shell wrapper around `main.py` that resolves required paths before launching the orchestrated run.
- Inputs: `--dataset`, `--latent-tags-dir`, `--dataset-pkl-path`, `--output-dir`, optional `--dataset-config-csv`.
- Outputs: Same as `main.py`.
- Example: `./scripts/run_main.sh --dataset physionet --latent-tags-dir /path/to/voter_csvs --dataset-pkl-path /path/to/physionet2012_ts_oc_ids.pkl --output-dir /path/to/main_run`

### `scripts/validate_global_variables_config.py`

- Purpose: Validate compact dataset config CSVs and each script's required config fields.
- Inputs: `configs/physionet-global-variables.csv`, `configs/mimic-global-variables.csv`, and `docs/global-variables-parameters.txt`.
- Outputs: PASS/FAIL table printed to stdout; nonzero exit on validation errors.
- Example: `python3 scripts/validate_global_variables_config.py`

### `scripts/validate_global_variables_config.sh`

- Purpose: Run config validation plus each active script's `--validate-config-only` check through the `econml310` conda env.
- Inputs: Dataset config CSVs and active script config contracts.
- Outputs: PASS/FAIL lines printed to stdout; nonzero exit on any failure.
- Example: `./scripts/validate_global_variables_config.sh`

### `src/preprocess_physionet_2012.py`

- Purpose: Convert raw PhysioNet 2012 set-a/set-b/set-c files and outcomes into the processed pickle contract.
- Inputs: Raw PhysioNet root via `--raw-data-path`, optional `--processed-dir` or `--output-path`.
- Outputs: Processed `[ts, oc, ts_ids]` pickle.
- Example: `conda run -n econml310 python src/preprocess_physionet_2012.py --raw-data-path /path/to/physionet2012 --output-path /path/to/physionet2012_ts_oc_ids.pkl`

### `src/preprocess_mimic_iii_large.py`

- Purpose: Convert raw MIMIC-III ICU tables into the PhysioNet-compatible processed pickle contract.
- Inputs: Raw MIMIC-III CSV root via `--raw-data-path`, optional `--output-path`.
- Outputs: Processed `[ts, oc, ts_ids]` pickle.
- Example: `conda run -n econml310 python src/preprocess_mimic_iii_large.py --raw-data-path /path/to/mimiciii --output-path /path/to/mimic_iii_ts_oc_ids.pkl`

### `src/physionet2012_causal_graph.py`

- Purpose: Build and render the hand-authored PhysioNet 2012 causal DAG.
- Inputs: Optional dataset config CSV plus graph pickle/PNG output paths.
- Outputs: Graph pickle and rendered PNG.
- Example: `conda run -n econml310 python src/physionet2012_causal_graph.py --graph-pkl-path /path/to/causal_graph.pkl --graph-png-path /path/to/physionet_dag.png`

### `src/mimiciii_causal_graph.py`

- Purpose: Build and render the hand-authored MIMIC-III causal DAG.
- Inputs: Optional dataset config CSV plus graph pickle/PNG output paths.
- Outputs: Graph pickle and rendered PNG.
- Example: `conda run -n econml310 python src/mimiciii_causal_graph.py --graph-pkl-path /path/to/mimiciii_causal_graph.pkl --graph-png-path /path/to/mimiciii_dag.png`

### `src/tagging_latent_variables_physionet.py`

- Purpose: Apply rule-based PhysioNet latent-state tags from a processed pickle.
- Inputs: `--pkl-path`, optional `--output-csv-path`, `--optimized`, and `--thresholds-path`.
- Outputs: Latent tag CSV and matching `_trees.pkl` decision-tree pickle.
- Example: `conda run -n econml310 python src/tagging_latent_variables_physionet.py --pkl-path /path/to/physionet2012_ts_oc_ids.pkl --output-csv-path /path/to/latent_tags_clinical.csv`

### `src/tagging_latent_variables_mimiciii.py`

- Purpose: Apply rule-based MIMIC-III latent-state tags from one input mode: summary CSV, canonical pickle, or raw concept CSVs.
- Inputs: `--summary_csv` or `--pkl_path` or raw concept CSV flags; optional `--output_dir`.
- Outputs: `latent_tags.csv`, `latent_tags_with_features.csv`, validation and prevalence summaries, co-occurrence CSV, and decision-tree pickle.
- Example: `conda run -n econml310 python src/tagging_latent_variables_mimiciii.py --pkl_path /path/to/mimic_iii_ts_oc_ids.pkl --output_dir /path/to/mimiciii_latents`

### `src/majority_vote_latents.py`

- Purpose: Combine multiple binary latent-tag voter CSVs into one majority-vote latent CSV.
- Inputs: `--input-dir` containing CSVs with `ts_id` plus binary latent columns.
- Outputs: Majority-vote latent CSV.
- Example: `conda run -n econml310 python src/majority_vote_latents.py --input-dir /path/to/voter_csvs --output-path /path/to/latent_tags_majority_vote.csv`

### `src/split_predicted_latent_tags.py`

- Purpose: Split a combined predicted-latents CSV into separate probability and absolute-tag CSVs.
- Inputs: `--input-csv` whose first column is `ts_id` and whose remaining columns are probability columns followed by tag columns.
- Outputs: `<input_stem>_probabilities.csv` and `<input_stem>_absolute_tags.csv` beside the input.
- Example: `conda run -n econml310 python src/split_predicted_latent_tags.py --input-csv /path/to/predicted_latent_tags.csv`

### `src/mortality_prediction_using_latents.py`

- Purpose: Train logistic regression and MLP mortality predictors from latent tags.
- Inputs: Latent tags CSV, processed dataset pickle, optional dataset config CSV.
- Outputs: Mortality prediction text report.
- Example: `conda run -n econml310 python src/mortality_prediction_using_latents.py --model physionet --latent-tags-path /path/to/latent_tags.csv --dataset-pkl-path /path/to/physionet2012_ts_oc_ids.pkl --results-txt-path /path/to/mortality_prediction_results.txt`

### `src/matching_causal_effect.py`

- Purpose: Estimate DAG-guided matched-pair causal effect summaries for latent treatments.
- Inputs: Latent tags CSV, processed dataset pickle, graph pickle, optional dataset config CSV.
- Outputs: Per-treatment matching folders plus `global_summary.csv`.
- Example: `conda run -n econml310 python src/matching_causal_effect.py --model physionet --latent-tags-path /path/to/latent_tags.csv --dataset-pkl-path /path/to/physionet2012_ts_oc_ids.pkl --graph-pkl-path /path/to/causal_graph.pkl --output-dir /path/to/matching_outputs`

### `src/cate_estimation.py`

- Purpose: Estimate treatment effects with `CausalForestDML` or `LinearDML` using DAG-guided confounders.
- Inputs: Latent tags CSV, processed dataset pickle, graph pickle, optional model/config flags.
- Outputs: Per-treatment CATE outputs, saved model pickles, and run-level summary/control CSVs.
- Example: `conda run -n econml310 python src/cate_estimation.py --model physionet --latent-tags-path /path/to/latent_tags.csv --dataset-pkl-path /path/to/physionet2012_ts_oc_ids.pkl --graph-pkl-path /path/to/causal_graph.pkl --output-dir /path/to/cate_outputs --model-type CausalForest`

### `src/analyze_cate_results.py`

- Purpose: Analyze saved CATE model artifacts and recompute sensitivity/benchmark outputs.
- Inputs: CATE results directory, latent tags CSV, processed dataset pickle, optional output directory.
- Outputs: Per-treatment benchmark reports/scores/contours plus run-level analysis summary CSVs.
- Example: `conda run -n econml310 python src/analyze_cate_results.py --model physionet --results-dir /path/to/cate_outputs --latent-tags-path /path/to/latent_tags.csv --dataset-pkl-path /path/to/physionet2012_ts_oc_ids.pkl --output-dir /path/to/cate_analysis`

### `src/permutations_test.py`

- Purpose: Run treatment-column and outcome permutation sanity checks by repeatedly calling `cate_estimation.py`.
- Inputs: Latent tags CSV, processed dataset pickle, graph pickle, trial count, estimator type.
- Outputs: `treatment_permutation_results.csv` and `outcome_permutation_results.csv`.
- Example: `conda run -n econml310 python src/permutations_test.py --model physionet --trials 10 --latent-tags-path /path/to/latent_tags.csv --dataset-pkl-path /path/to/physionet2012_ts_oc_ids.pkl --graph-pkl-path /path/to/causal_graph.pkl --experiment-dir /path/to/permutation_outputs`

### `src/decision_trees_plot.py`

- Purpose: Render latent decision-tree pickles into one figure per latent rule.
- Inputs: Dataset name, saved latent decision-tree pickle, optional output/format flags.
- Outputs: Rule diagrams under the output directory.
- Example: `conda run -n econml310 python src/decision_trees_plot.py --dataset physionet --pickle-path /path/to/latent_tags_clinical_trees.pkl --output-dir /path/to/tree_figures --overwrite`

## Legacy Draft Entry Points

These scripts are runnable but live under `src/draft/` and preserve older path assumptions. Check their hard-coded constants before using them.

### `src/draft/clinically_sufficient_tagging_latent_variables.py`

- Purpose: Older clinical/windowed PhysioNet latent tagger.
- Inputs: Hard-coded processed PhysioNet pickle path.
- Outputs: `latent_tags_clinical.csv`, `_trees.pkl`, and optional `_stages.csv`.
- Example: `cd src/draft && conda run -n econml310 python clinically_sufficient_tagging_latent_variables.py`

### `src/draft/optimize_latent_thresholds.py`

- Purpose: Optuna threshold search for the older PhysioNet summary-statistics latent tagger.
- Inputs: Hard-coded processed PhysioNet pickle path and `N_TRIALS`.
- Outputs: `mortality_prediction_results_optimized.txt` and `optimal_thresholds.txt`.
- Example: `cd src/draft && conda run -n econml310 python optimize_latent_thresholds.py`

### `src/draft/causal_inference_on_latent_variables.py`

- Purpose: Legacy stratification estimator labeled as CATE for PhysioNet latent variables.
- Inputs: Hard-coded latent tags CSV, processed PhysioNet pickle, and graph pickle.
- Outputs: `cate_results.txt`.
- Example: `cd src/draft && conda run -n econml310 python causal_inference_on_latent_variables.py`

### `src/draft/causal_inference_on_latent_variables_updated.py`

- Purpose: Conservative legacy ATE stratification estimator for PhysioNet latent variables.
- Inputs: Hard-coded latent tags CSV, processed PhysioNet pickle, and graph pickle.
- Outputs: `ate_results_fixed.txt`.
- Example: `cd src/draft && conda run -n econml310 python causal_inference_on_latent_variables_updated.py`

### `src/draft/physionet2012_causal_graph_old.py`

- Purpose: Older PhysioNet DAG builder retained for comparison.
- Inputs: Optional dataset config CSV plus graph pickle/PNG output paths.
- Outputs: Graph pickle and rendered PNG.
- Example: `conda run -n econml310 python src/draft/physionet2012_causal_graph_old.py --graph-pkl-path /path/to/old_causal_graph.pkl --graph-png-path /path/to/old_physionet_dag.png`

### `src/draft/mimiciii_causal_graph_old.py`

- Purpose: Older MIMIC-III DAG builder retained for comparison.
- Inputs: Optional dataset config CSV plus graph pickle/PNG output paths.
- Outputs: Graph pickle and rendered PNG.
- Example: `conda run -n econml310 python src/draft/mimiciii_causal_graph_old.py --graph-pkl-path /path/to/old_mimiciii_causal_graph.pkl --graph-png-path /path/to/old_mimiciii_dag.png`

### `src/draft/tagging_latent_variables_physionet_old.py`

- Purpose: Older rule-based PhysioNet latent tagger.
- Inputs: Processed pickle, optional output CSV, optimization flag, and thresholds path.
- Outputs: Latent tag CSV and `_trees.pkl`.
- Example: `conda run -n econml310 python src/draft/tagging_latent_variables_physionet_old.py --pkl-path /path/to/physionet2012_ts_oc_ids.pkl --output-csv-path /path/to/latent_tags_optimized.csv`

### `src/draft/tagging_latent_variables_mimiciii_old.py`

- Purpose: Older MIMIC-III latent tagger using summary CSV, canonical pickle, or raw concept CSV inputs.
- Inputs: `--summary_csv` or `--pkl_path` or raw concept CSV flags; optional `--output_dir`.
- Outputs: Latent tags, feature-merged tags, validation summaries, prevalence summaries, co-occurrence CSV, and decision-tree pickle.
- Example: `conda run -n econml310 python src/draft/tagging_latent_variables_mimiciii_old.py --pkl_path /path/to/mimic_iii_ts_oc_ids.pkl --output_dir /path/to/old_mimiciii_latents`

### `src/draft/treatment_split.py`

- Purpose: Demonstrate treatment/control splits from PhysioNet time-series measurement value and spacing rules.
- Inputs: Hard-coded processed PhysioNet pickle path.
- Outputs: Split-size summaries printed to stdout.
- Example: `cd src/draft && conda run -n econml310 python treatment_split.py`
