import argparse
import pickle
import sys
from pathlib import Path

if "--validate-config-only" in sys.argv:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from dataset_config import maybe_run_validate_config_only

    maybe_run_validate_config_only(
        "src/tagging_latent_variables_physionet.py",
        fixed_dataset="physionet",
    )

import pandas as pd
import numpy as np
from tqdm import tqdm
from functools import partial

from dataset_config import (
    get_config_list,
    load_dataset_config,
)


pkl_path = "../../data/processed/physionet2012_ts_oc_ids.pkl"
output_csv_path = "latent_tags_optimized.csv"

OPTIMIZED = True
THRESHOLDS_PATH = "../../data/optimal_thresholds.txt"
DEFAULT_THRESHOLDS = {
    # LAT_CHRONIC_BASELINE_RISK
    "chronic_age": 75,
    "chronic_bmi_low": 18.5,
    "chronic_bmi_high": 40,
    "chronic_albumin": 3.0,
    "chronic_min_count": 2,

    # LAT_GLOBAL_SEVERITY
    "global_map": 70,
    "global_sysabp": 100,
    "global_nimap": 70,
    "global_nisysabp": 100,
    "global_pf": 300,
    "global_sao2": 92,
    "global_resprate": 22,
    "global_gcs": 15,
    "global_creatinine": 2.0,
    "global_bun": 40,
    "global_urine_24h": 500,
    "global_bilirubin": 2.0,
    "global_platelets": 100,
    "global_lactate": 2.0,
    "global_ph": 7.30,
    "global_hco3": 18,
    "global_tropi": 0.1,
    "global_tropt": 0.01,
    "global_hr": 130,
    "global_min_count": 3,
    "global_critical_lactate": 4.0,
    "global_critical_ph": 7.20,
    "global_critical_gcs": 8,
    "global_critical_map": 60,

    # LAT_SHOCK
    "shock_map": 70,
    "shock_nimap": 70,
    "shock_sysabp": 90,
    "shock_nisysabp": 90,
    "shock_lactate": 2.0,
    "shock_hr": 110,
    "shock_urine_24h": 500,
    "shock_ph": 7.30,
    "shock_hco3": 18,
    "shock_critical_lactate": 4.0,
    "shock_min_count": 2,

    # LAT_RESPIRATORY_FAILURE
    "resp_pf": 300,
    "resp_sao2": 92,
    "resp_pao2": 60,
    "resp_rate_high": 22,
    "resp_rate_low": 8,
    "resp_paco2": 50,
    "resp_paco2_acidotic": 45,
    "resp_ph": 7.30,
    "resp_min_count": 2,

    # LAT_RENAL_DYSFUNCTION
    "renal_creatinine": 2.0,
    "renal_creatinine_rise": 0.3,
    "renal_bun": 40,
    "renal_urine_24h": 500,
    "renal_k": 5.5,
    "renal_hco3": 18,
    "renal_critical_creatinine": 3.5,
    "renal_min_count": 2,

    # LAT_HEPATIC_DYSFUNCTION
    "hepatic_bilirubin": 2.0,
    "hepatic_ast": 200,
    "hepatic_alt": 200,
    "hepatic_alp": 250,
    "hepatic_albumin": 2.5,
    "hepatic_platelets": 100,
    "hepatic_min_count": 2,

    # LAT_COAG_HEME_DYSFUNCTION
    "coag_platelets_severe": 100,
    "coag_platelets_mild": 150,
    "coag_hct_low": 25,
    "coag_hct_high": 55,
    "coag_wbc_low": 4,
    "coag_wbc_high": 20,
    "coag_platelet_drop": 50,
    "coag_min_count": 2,

    # LAT_INFLAMMATION_SEPSIS_BURDEN
    "inflam_temp_high": 38.3,
    "inflam_temp_low": 36,
    "inflam_wbc_high": 12,
    "inflam_wbc_low": 4,
    "inflam_hr": 90,
    "inflam_resprate": 20,
    "inflam_paco2": 32,
    "inflam_lactate": 2.0,
    "inflam_platelets": 150,
    "inflam_min_count": 3,

    # LAT_NEUROLOGIC_DYSFUNCTION
    "neuro_gcs_mild": 15,
    "neuro_gcs_severe": 12,
    "neuro_na_low": 130,
    "neuro_na_high": 150,
    "neuro_glucose_low": 70,
    "neuro_glucose_high": 300,
    "neuro_paco2": 50,
    "neuro_sao2": 90,
    "neuro_ph": 7.25,
    "neuro_min_count": 2,

    # LAT_CARDIAC_INJURY_STRAIN
    "card_tropi": 0.1,
    "card_tropt": 0.01,
    "card_hr_high": 130,
    "card_hr_low": 50,
    "card_map": 70,
    "card_sysabp": 90,
    "card_lactate": 2.0,
    "card_min_count": 2,

    # LAT_METABOLIC_DERANGEMENT
    "metab_ph_low": 7.30,
    "metab_ph_high": 7.50,
    "metab_hco3_low": 18,
    "metab_hco3_high": 32,
    "metab_lactate": 2.0,
    "metab_na_low": 130,
    "metab_na_high": 150,
    "metab_k_low": 3.0,
    "metab_k_high": 5.5,
    "metab_mg_low": 0.6,
    "metab_mg_high": 1.2,
    "metab_glucose_low": 70,
    "metab_glucose_high": 250,
    "metab_paco2_low": 32,
    "metab_paco2_high": 50,
    "metab_critical_ph": 7.20,
    "metab_critical_lactate": 4.0,
    "metab_critical_k": 6.0,
    "metab_min_count": 2,
}
LATENT_ORDER = [
    "LAT_CHRONIC_BASELINE_RISK",
    "LAT_GLOBAL_SEVERITY",
    "LAT_SHOCK",
    "LAT_RESPIRATORY_FAILURE",
    "LAT_RENAL_DYSFUNCTION",
    "LAT_HEPATIC_DYSFUNCTION",
    "LAT_COAG_HEME_DYSFUNCTION",
    "LAT_INFLAMMATION_SEPSIS_BURDEN",
    "LAT_NEUROLOGIC_DYSFUNCTION",
    "LAT_CARDIAC_INJURY_STRAIN",
    "LAT_METABOLIC_DERANGEMENT",
]

def str_to_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply rule-based PhysioNet latent variable tags."
    )
    parser.add_argument("--dataset-config-csv", default=None)
    parser.add_argument("--pkl-path", default=None)
    parser.add_argument("--output-csv-path", default=None)
    parser.add_argument("--optimized", type=str_to_bool, default=None)
    parser.add_argument("--thresholds-path", default=None)
    parser.add_argument(
        "--validate-config-only",
        action="store_true",
        help="Resolve dataset config values and exit without loading data.",
    )
    return parser.parse_args()

# ============================================================
# 1. LOAD DATA
# ============================================================


def load_physionet_pickle(pkl_path):
    print("[1/5] Loading PhysioNet pickle...")

    with open(pkl_path, 'rb') as f:
        ts, oc, ts_ids = pickle.load(f)

    print(f"      Loaded ts: {len(ts):,} rows")
    print(f"      Loaded oc: {len(oc):,} rows")
    print(f"      Loaded ts_ids: {len(ts_ids):,} patients")
    return ts, oc, ts_ids


def pivot_ts_to_wide(ts):
    print("[2/5] Pivoting long time-series to wide format...")

    df_wide = ts.pivot_table(
        index=["ts_id", "minute"],
        columns="variable",
        values="value"
    ).reset_index()

    print(f"      Wide dataframe shape: {df_wide.shape}")
    return df_wide


# ============================================================
# 2. SUMMARY STATISTICS
# ============================================================

def build_patient_summaries(df_wide):
    print("[3/5] Building per-patient summary statistics...")

    summary_funcs = {
        "min": np.min,
        "max": np.max,
        "mean": np.mean,
        "first": lambda x: x.iloc[0],
        "last": lambda x: x.iloc[-1]
    }

    variable_cols = [c for c in df_wide.columns if c not in ["ts_id", "minute"]]

    summaries = []

    for pid, g in tqdm(df_wide.groupby("ts_id"),
                       desc="      Processing patients",
                       unit="patient"):

        row = {"ts_id": pid}
        g = g.sort_values("minute")

        for var in variable_cols:
            series = g[var].dropna()
            if len(series) == 0:
                for stat in summary_funcs:
                    row[f"{var}_{stat}"] = np.nan
            else:
                for stat, func in summary_funcs.items():
                    row[f"{var}_{stat}"] = func(series)

        summaries.append(row)

    summary_df = pd.DataFrame(summaries)
    print(f"      Created summary table: {summary_df.shape}")
    return summary_df


def load_optimal_thresholds(path):
    """
    Parses optimal_thresholds.txt into a dict.
    """
    thresholds = {}

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            if ":" not in line:
                continue

            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip()

            try:
                thresholds[key] = float(val)
            except ValueError:
                continue

    print(f"[INFO] Loaded {len(thresholds)} optimized thresholds")
    return thresholds


def get_thr(thr_dict, key, default):
    if thr_dict is None:
        return default
    return thr_dict.get(key, default)

# ============================================================
# 3. DECISION TREES
# ============================================================


def _value(row, names, default=np.nan):
    """
    Return the first non-missing value among possible summary-column names.
    Missing values remain missing and therefore do not count as abnormal.
    """
    if isinstance(names, str):
        names = [names]
    for name in names:
        val = row.get(name, np.nan)
        if pd.notna(val):
            return val
    return default


def _is_present(row, names) -> bool:
    return pd.notna(_value(row, names, np.nan))


def _lt(row, names, threshold):
    val = _value(row, names, np.nan)
    return bool(pd.notna(val) and val < threshold)


def _le(row, names, threshold):
    val = _value(row, names, np.nan)
    return bool(pd.notna(val) and val <= threshold)


def _gt(row, names, threshold):
    val = _value(row, names, np.nan)
    return bool(pd.notna(val) and val > threshold)


def _ge(row, names, threshold):
    val = _value(row, names, np.nan)
    return bool(pd.notna(val) and val >= threshold)


def _eq(row, names, target):
    val = _value(row, names, np.nan)
    return bool(pd.notna(val) and val == target)


def _in(row, names, targets):
    val = _value(row, names, np.nan)
    return bool(pd.notna(val) and val in targets)


def _pf_min(row):
    """
    Use a precomputed PF_min when available. Otherwise approximate from
    PaO2_min and FiO2_max/min summaries if those are the only available inputs.
    """
    pf = _value(row, "PF_min", np.nan)
    if pd.notna(pf):
        return pf

    pao2 = _value(row, "PaO2_min", np.nan)
    fio2 = _value(row, ["FiO2_max", "FiO2_min", "FiO2_first"], np.nan)
    if pd.isna(pao2) or pd.isna(fio2) or fio2 <= 0:
        return np.nan
    if fio2 > 1 and fio2 <= 100:
        fio2 = fio2 / 100.0
    if fio2 <= 0:
        return np.nan
    return pao2 / fio2


def _bmi(row):
    height = _value(row, ["Height_first", "Height"], np.nan)
    weight = _value(row, ["Weight_first", "Weight"], np.nan)
    if pd.isna(height) or pd.isna(weight) or height <= 0:
        return np.nan
    return weight / ((height / 100.0) ** 2)


def _urine_24h_min(row):
    return _value(row, ["Urine_24h_min", "Urine_24h_sum", "Urine_sum"], np.nan)


def tag_lat_chronic_baseline_risk(row, thr=None):
    """
    LAT_CHRONIC_BASELINE_RISK = 1 means evidence of high baseline vulnerability
    or limited physiologic reserve before or at ICU admission.
    """
    bmi = _bmi(row)

    age_risk = _ge(row, ["Age_first", "Age"], get_thr(thr, "chronic_age", 75))
    body_risk = bool(
        pd.notna(bmi) and (
            bmi < get_thr(thr, "chronic_bmi_low", 18.5) or
            bmi >= get_thr(thr, "chronic_bmi_high", 40)
        )
    )
    low_albumin = _lt(row, "Albumin_min", get_thr(thr, "chronic_albumin", 3.0))
    casemix_risk = _in(row, ["ICUType_first", "ICUType"], [1, 3])

    score = sum([age_risk, body_risk, low_albumin, casemix_risk])
    return int(score >= int(round(get_thr(thr, "chronic_min_count", 2))))


def tag_lat_global_severity(row, thr=None):
    """
    LAT_GLOBAL_SEVERITY = 1 means evidence of multi-domain acute physiologic
    severity during the first 48 hours.
    """
    pf_min = _pf_min(row)
    low_pf = bool(pd.notna(pf_min) and pf_min < get_thr(thr, "global_pf", 300))

    hemo = any([
        _lt(row, "MAP_min", get_thr(thr, "global_map", 70)),
        _lt(row, "NIMAP_min", get_thr(thr, "global_nimap", 70)),
        _le(row, "SysABP_min", get_thr(thr, "global_sysabp", 100)),
        _le(row, "NISysABP_min", get_thr(thr, "global_nisysabp", 100)),
    ])

    resp = any([
        _eq(row, "MechVent_max", 1),
        low_pf,
        _lt(row, "SaO2_min", get_thr(thr, "global_sao2", 92)),
        _ge(row, "RespRate_max", get_thr(thr, "global_resprate", 22)),
    ])

    neuro = _lt(row, "GCS_min", get_thr(thr, "global_gcs", 15))

    renal = any([
        _ge(row, "Creatinine_max", get_thr(thr, "global_creatinine", 2.0)),
        _ge(row, "BUN_max", get_thr(thr, "global_bun", 40)),
        bool(pd.notna(_urine_24h_min(row)) and _urine_24h_min(row) < get_thr(thr, "global_urine_24h", 500)),
    ])

    hep_coag = any([
        _ge(row, "Bilirubin_max", get_thr(thr, "global_bilirubin", 2.0)),
        _lt(row, "Platelets_min", get_thr(thr, "global_platelets", 100)),
    ])

    metab = any([
        _gt(row, "Lactate_max", get_thr(thr, "global_lactate", 2.0)),
        _lt(row, "pH_min", get_thr(thr, "global_ph", 7.30)),
        _lt(row, "HCO3_min", get_thr(thr, "global_hco3", 18)),
    ])

    cardiac = any([
        _gt(row, "TropI_max", get_thr(thr, "global_tropi", 0.1)),
        _gt(row, "TropT_max", get_thr(thr, "global_tropt", 0.01)),
        _ge(row, "HR_max", get_thr(thr, "global_hr", 130)),
    ])

    score = sum([hemo, resp, neuro, renal, hep_coag, metab, cardiac])
    critical = any([
        _ge(row, "Lactate_max", get_thr(thr, "global_critical_lactate", 4.0)),
        _lt(row, "pH_min", get_thr(thr, "global_critical_ph", 7.20)),
        _eq(row, "MechVent_max", 1),
        _le(row, "GCS_min", get_thr(thr, "global_critical_gcs", 8)),
        _lt(row, "MAP_min", get_thr(thr, "global_critical_map", 60)),
    ])

    min_count = int(round(get_thr(thr, "global_min_count", 3)))
    return int(score >= min_count or (score >= 2 and critical))


def tag_lat_shock(row, thr=None):
    """
    LAT_SHOCK = 1 means evidence of circulatory shock or clinically meaningful
    hemodynamic instability.
    """
    urine_24h_min = _urine_24h_min(row)

    low_map = any([
        _lt(row, "MAP_min", get_thr(thr, "shock_map", 70)),
        _lt(row, "NIMAP_min", get_thr(thr, "shock_nimap", 70)),
    ])
    low_sbp = any([
        _le(row, "SysABP_min", get_thr(thr, "shock_sysabp", 90)),
        _le(row, "NISysABP_min", get_thr(thr, "shock_nisysabp", 90)),
    ])
    hypoperfusion = _gt(row, "Lactate_max", get_thr(thr, "shock_lactate", 2.0))
    tachy = _ge(row, "HR_max", get_thr(thr, "shock_hr", 110))
    oliguria = bool(pd.notna(urine_24h_min) and urine_24h_min < get_thr(thr, "shock_urine_24h", 500))
    acidosis = any([
        _lt(row, "pH_min", get_thr(thr, "shock_ph", 7.30)),
        _lt(row, "HCO3_min", get_thr(thr, "shock_hco3", 18)),
    ])

    score = sum([low_map, low_sbp, hypoperfusion, tachy, oliguria, acidosis])
    severe_pair = low_map and _ge(row, "Lactate_max", get_thr(thr, "shock_critical_lactate", 4.0))
    return int(score >= int(round(get_thr(thr, "shock_min_count", 2))) or severe_pair)


def tag_lat_respiratory_failure(row, thr=None):
    """
    LAT_RESPIRATORY_FAILURE = 1 means evidence of hypoxemia, ventilatory
    failure, or mechanical ventilatory support.
    """
    pf_min = _pf_min(row)
    low_pf = bool(pd.notna(pf_min) and pf_min < get_thr(thr, "resp_pf", 300))

    vent = _eq(row, "MechVent_max", 1)
    hypoxemia = any([
        _lt(row, "SaO2_min", get_thr(thr, "resp_sao2", 92)),
        _lt(row, "PaO2_min", get_thr(thr, "resp_pao2", 60)),
    ])
    tachypnea = any([
        _ge(row, "RespRate_max", get_thr(thr, "resp_rate_high", 22)),
        _lt(row, "RespRate_min", get_thr(thr, "resp_rate_low", 8)),
    ])
    ventilatory_failure = any([
        _ge(row, "PaCO2_max", get_thr(thr, "resp_paco2", 50)),
        (
            _ge(row, "PaCO2_max", get_thr(thr, "resp_paco2_acidotic", 45)) and
            _lt(row, "pH_min", get_thr(thr, "resp_ph", 7.30))
        ),
    ])

    score = sum([vent, low_pf, hypoxemia, tachypnea, ventilatory_failure])
    return int(score >= int(round(get_thr(thr, "resp_min_count", 2))) or (vent and low_pf))


def tag_lat_renal_dysfunction(row, thr=None):
    """
    LAT_RENAL_DYSFUNCTION = 1 means evidence of acute or clinically meaningful
    renal dysfunction.
    """
    urine_24h_min = _urine_24h_min(row)
    creat_first = _value(row, "Creatinine_first", np.nan)
    creat_max = _value(row, "Creatinine_max", np.nan)

    creat_high = _ge(row, "Creatinine_max", get_thr(thr, "renal_creatinine", 2.0))
    creat_rise = bool(
        pd.notna(creat_first) and pd.notna(creat_max) and
        (creat_max - creat_first >= get_thr(thr, "renal_creatinine_rise", 0.3))
    )
    bun_high = _ge(row, "BUN_max", get_thr(thr, "renal_bun", 40))
    oliguria = bool(pd.notna(urine_24h_min) and urine_24h_min < get_thr(thr, "renal_urine_24h", 500))
    renal_metab = any([
        _ge(row, "K_max", get_thr(thr, "renal_k", 5.5)),
        _lt(row, "HCO3_min", get_thr(thr, "renal_hco3", 18)),
    ])

    score = sum([creat_high, creat_rise, bun_high, oliguria, renal_metab])
    critical = _ge(row, "Creatinine_max", get_thr(thr, "renal_critical_creatinine", 3.5))
    return int(score >= int(round(get_thr(thr, "renal_min_count", 2))) or critical)


def tag_lat_hepatic_dysfunction(row, thr=None):
    """
    LAT_HEPATIC_DYSFUNCTION = 1 means evidence of cholestatic or hepatocellular
    injury or poor synthetic/nutritional hepatic reserve.
    """
    bili = _ge(row, "Bilirubin_max", get_thr(thr, "hepatic_bilirubin", 2.0))
    transam = any([
        _ge(row, "AST_max", get_thr(thr, "hepatic_ast", 200)),
        _ge(row, "ALT_max", get_thr(thr, "hepatic_alt", 200)),
    ])
    chol = _ge(row, "ALP_max", get_thr(thr, "hepatic_alp", 250))
    low_albumin = _lt(row, "Albumin_min", get_thr(thr, "hepatic_albumin", 2.5))
    portal_or_severe = _lt(row, "Platelets_min", get_thr(thr, "hepatic_platelets", 100))

    score = 2 * int(bili) + sum([transam, chol, low_albumin, portal_or_severe])
    return int(score >= int(round(get_thr(thr, "hepatic_min_count", 2))))


def tag_lat_coag_heme_dysfunction(row, thr=None):
    """
    LAT_COAG_HEME_DYSFUNCTION = 1 means evidence of thrombocytopenia, anemia,
    or hematologic stress.
    """
    platelet_first = _value(row, "Platelets_first", np.nan)
    platelet_min = _value(row, "Platelets_min", np.nan)

    severe_plt = _lt(row, "Platelets_min", get_thr(thr, "coag_platelets_severe", 100))
    mild_plt = _lt(row, "Platelets_min", get_thr(thr, "coag_platelets_mild", 150))
    hct_abn = any([
        _lt(row, "HCT_min", get_thr(thr, "coag_hct_low", 25)),
        _gt(row, "HCT_max", get_thr(thr, "coag_hct_high", 55)),
    ])
    wbc_extreme = any([
        _lt(row, "WBC_min", get_thr(thr, "coag_wbc_low", 4)),
        _gt(row, "WBC_max", get_thr(thr, "coag_wbc_high", 20)),
    ])
    platelet_drop = bool(
        pd.notna(platelet_first) and pd.notna(platelet_min) and
        (platelet_first - platelet_min >= get_thr(thr, "coag_platelet_drop", 50))
    )

    score = 2 * int(severe_plt) + sum([mild_plt, hct_abn, wbc_extreme, platelet_drop])
    return int(score >= int(round(get_thr(thr, "coag_min_count", 2))))


def tag_lat_inflammation_sepsis_burden(row, thr=None):
    """
    LAT_INFLAMMATION_SEPSIS_BURDEN = 1 means systemic inflammatory or
    sepsis-like host response. This is not confirmed sepsis.
    """
    temp_abn = any([
        _gt(row, "Temp_max", get_thr(thr, "inflam_temp_high", 38.3)),
        _lt(row, "Temp_min", get_thr(thr, "inflam_temp_low", 36)),
    ])
    wbc_abn = any([
        _gt(row, "WBC_max", get_thr(thr, "inflam_wbc_high", 12)),
        _lt(row, "WBC_min", get_thr(thr, "inflam_wbc_low", 4)),
    ])
    tachy = _gt(row, "HR_max", get_thr(thr, "inflam_hr", 90))
    resp_stress = any([
        _gt(row, "RespRate_max", get_thr(thr, "inflam_resprate", 20)),
        _lt(row, "PaCO2_min", get_thr(thr, "inflam_paco2", 32)),
    ])
    lactate_stress = _gt(row, "Lactate_max", get_thr(thr, "inflam_lactate", 2.0))
    plt_low = _lt(row, "Platelets_min", get_thr(thr, "inflam_platelets", 150))

    score = sum([temp_abn, wbc_abn, tachy, resp_stress, lactate_stress, plt_low])
    anchored = (temp_abn or wbc_abn) and lactate_stress and score >= 2
    return int(score >= int(round(get_thr(thr, "inflam_min_count", 3))) or anchored)


def tag_lat_neurologic_dysfunction(row, thr=None):
    """
    LAT_NEUROLOGIC_DYSFUNCTION = 1 means depressed consciousness or neurologic
    dysfunction, acknowledging sedation/ventilation ambiguity.
    """
    gcs_mild = _lt(row, "GCS_min", get_thr(thr, "neuro_gcs_mild", 15))
    gcs_severe = _le(row, "GCS_min", get_thr(thr, "neuro_gcs_severe", 12))
    metab_neuro = any([
        _lt(row, "Na_min", get_thr(thr, "neuro_na_low", 130)),
        _gt(row, "Na_max", get_thr(thr, "neuro_na_high", 150)),
        _lt(row, "Glucose_min", get_thr(thr, "neuro_glucose_low", 70)),
        _gt(row, "Glucose_max", get_thr(thr, "neuro_glucose_high", 300)),
    ])
    gas_neuro = any([
        _ge(row, "PaCO2_max", get_thr(thr, "neuro_paco2", 50)),
        _lt(row, "SaO2_min", get_thr(thr, "neuro_sao2", 90)),
        _lt(row, "pH_min", get_thr(thr, "neuro_ph", 7.25)),
    ])

    score = 2 * int(gcs_severe) + sum([gcs_mild, metab_neuro, gas_neuro])
    return int(score >= int(round(get_thr(thr, "neuro_min_count", 2))))


def tag_lat_cardiac_injury_strain(row, thr=None):
    """
    LAT_CARDIAC_INJURY_STRAIN = 1 means evidence of myocardial injury,
    ischemia, or severe cardiac strain.
    """
    trop_abn = any([
        _gt(row, "TropI_max", get_thr(thr, "card_tropi", 0.1)),
        _gt(row, "TropT_max", get_thr(thr, "card_tropt", 0.01)),
    ])
    cardiac_unit = _in(row, ["ICUType_first", "ICUType"], [1, 2])
    arrhythmic_stress = any([
        _ge(row, "HR_max", get_thr(thr, "card_hr_high", 130)),
        _lt(row, "HR_min", get_thr(thr, "card_hr_low", 50)),
    ])
    hemo_strain = any([
        _lt(row, "MAP_min", get_thr(thr, "card_map", 70)),
        _le(row, "SysABP_min", get_thr(thr, "card_sysabp", 90)),
    ])
    perfusion_stress = _gt(row, "Lactate_max", get_thr(thr, "card_lactate", 2.0))

    score = 2 * int(trop_abn) + sum([cardiac_unit, arrhythmic_stress, hemo_strain, perfusion_stress])
    return int(score >= int(round(get_thr(thr, "card_min_count", 2))))


def tag_lat_metabolic_derangement(row, thr=None):
    """
    LAT_METABOLIC_DERANGEMENT = 1 means clinically meaningful acid-base,
    lactate, electrolyte, or glucose derangement.
    """
    ph_abn = any([
        _lt(row, "pH_min", get_thr(thr, "metab_ph_low", 7.30)),
        _gt(row, "pH_max", get_thr(thr, "metab_ph_high", 7.50)),
    ])
    bicarb_abn = any([
        _lt(row, "HCO3_min", get_thr(thr, "metab_hco3_low", 18)),
        _gt(row, "HCO3_max", get_thr(thr, "metab_hco3_high", 32)),
    ])
    lactate_abn = _gt(row, "Lactate_max", get_thr(thr, "metab_lactate", 2.0))
    electrolyte_abn = any([
        _lt(row, "Na_min", get_thr(thr, "metab_na_low", 130)),
        _gt(row, "Na_max", get_thr(thr, "metab_na_high", 150)),
        _lt(row, "K_min", get_thr(thr, "metab_k_low", 3.0)),
        _gt(row, "K_max", get_thr(thr, "metab_k_high", 5.5)),
        _lt(row, "Mg_min", get_thr(thr, "metab_mg_low", 0.6)),
        _gt(row, "Mg_max", get_thr(thr, "metab_mg_high", 1.2)),
    ])
    glucose_abn = any([
        _lt(row, "Glucose_min", get_thr(thr, "metab_glucose_low", 70)),
        _gt(row, "Glucose_max", get_thr(thr, "metab_glucose_high", 250)),
    ])
    vent_comp = any([
        _lt(row, "PaCO2_min", get_thr(thr, "metab_paco2_low", 32)),
        _gt(row, "PaCO2_max", get_thr(thr, "metab_paco2_high", 50)),
    ])

    score = sum([ph_abn, bicarb_abn, lactate_abn, electrolyte_abn, glucose_abn, vent_comp])
    critical = any([
        _lt(row, "pH_min", get_thr(thr, "metab_critical_ph", 7.20)),
        _ge(row, "Lactate_max", get_thr(thr, "metab_critical_lactate", 4.0)),
        _ge(row, "K_max", get_thr(thr, "metab_critical_k", 6.0)),
    ])

    return int(score >= int(round(get_thr(thr, "metab_min_count", 2))) or critical)


def get_latent_decision_trees(thr=None):
    """
    Returns dict: latent_variable_name -> decision_function(row)->0/1
    Uses functools.partial so functions remain pickle-safe.
    """
    print("[4/5] Initializing latent decision trees...")

    all_trees = {
        "LAT_CHRONIC_BASELINE_RISK": partial(tag_lat_chronic_baseline_risk, thr=thr),
        "LAT_GLOBAL_SEVERITY": partial(tag_lat_global_severity, thr=thr),
        "LAT_SHOCK": partial(tag_lat_shock, thr=thr),
        "LAT_RESPIRATORY_FAILURE": partial(tag_lat_respiratory_failure, thr=thr),
        "LAT_RENAL_DYSFUNCTION": partial(tag_lat_renal_dysfunction, thr=thr),
        "LAT_HEPATIC_DYSFUNCTION": partial(tag_lat_hepatic_dysfunction, thr=thr),
        "LAT_COAG_HEME_DYSFUNCTION": partial(tag_lat_coag_heme_dysfunction, thr=thr),
        "LAT_INFLAMMATION_SEPSIS_BURDEN": partial(tag_lat_inflammation_sepsis_burden, thr=thr),
        "LAT_NEUROLOGIC_DYSFUNCTION": partial(tag_lat_neurologic_dysfunction, thr=thr),
        "LAT_CARDIAC_INJURY_STRAIN": partial(tag_lat_cardiac_injury_strain, thr=thr),
        "LAT_METABOLIC_DERANGEMENT": partial(tag_lat_metabolic_derangement, thr=thr),
    }
    trees = {latent: all_trees[latent] for latent in LATENT_ORDER if latent in all_trees}

    print(f"      Loaded {len(trees)} latent definitions")
    return trees


# ============================================================
# 4. TAGGING
# ============================================================

def tag_all_patients(summary_df, decision_trees):
    print("[5/5] Applying latent tags to all patients...")

    tag_rows = []

    for _, row in tqdm(summary_df.iterrows(),
                       total=len(summary_df),
                       desc="      Tagging patients",
                       unit="patient"):
        pid = row["ts_id"]
        tag_row = {"ts_id": pid}

        for latent, func in decision_trees.items():
            tag_row[latent] = func(row)

        tag_rows.append(tag_row)

    latent_df = pd.DataFrame(tag_rows)
    print(f"      Tag table shape: {latent_df.shape}")
    return latent_df


# ============================================================
# 5. MAIN PIPELINE
# ============================================================

def run_latent_tagging_pipeline(pkl_path, output_csv_path):

    print("\n=== Starting PhysioNet Latent Tagging Pipeline ===\n")

    ts, oc, ts_ids = load_physionet_pickle(pkl_path)

    df_wide = pivot_ts_to_wide(ts)

    summary_df = build_patient_summaries(df_wide)

    if OPTIMIZED:
        print("[INFO] Using optimized thresholds")
        thr = load_optimal_thresholds(THRESHOLDS_PATH)
    else:
        print("[INFO] Using default thresholds")
        thr = dict(DEFAULT_THRESHOLDS)

    decision_trees = get_latent_decision_trees(thr)

    latent_tags_df = tag_all_patients(summary_df, decision_trees)

    print("\nSaving results to CSV...")
    latent_tags_df.to_csv(output_csv_path, index=False)

    print("\n=== Pipeline completed successfully ✅ ===")
    print(f"Output saved to: {output_csv_path}\n")

    return latent_tags_df, decision_trees


def main():
    global pkl_path
    global output_csv_path
    global OPTIMIZED
    global THRESHOLDS_PATH
    global DEFAULT_THRESHOLDS
    global LATENT_ORDER

    args = parse_args()
    config = load_dataset_config("physionet", args.dataset_config_csv)
    LATENT_ORDER = list(get_config_list(config, "LATENT_ORDER", LATENT_ORDER) or [])

    if args.pkl_path is not None:
        pkl_path = args.pkl_path
    if args.output_csv_path is not None:
        output_csv_path = args.output_csv_path
    if args.thresholds_path is not None:
        THRESHOLDS_PATH = args.thresholds_path
    if args.optimized is not None:
        OPTIMIZED = bool(args.optimized)

    latent_tags_df, decision_trees = run_latent_tagging_pipeline(pkl_path, output_csv_path)
    print(latent_tags_df.head())

    trees_path = output_csv_path.replace(".csv", "_trees.pkl")
    with open(trees_path, "wb") as f:
        pickle.dump(decision_trees, f)

    print(f"Decision trees dictionary saved to: {trees_path}")


if __name__ == "__main__":
    main()
