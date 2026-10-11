"""Generate a SYNTHETIC cohort for demos and tests.

No real patient record is read, sampled or perturbed here. Every column is drawn from a
distribution chosen to loosely match the marginals published in Table 1 of the manuscript,
and the correlations between variables are invented. The file is meant to let someone run
the inference notebooks without access to the real cohorts, which cannot be shared.

Diagnostic performance measured on this file is meaningless. Do not report it.
"""
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 20260928
N = 200
PREVALENCE = 0.08
OUT = Path(__file__).with_name("toy_cohort.csv")

FEATURES = [
    "gender", "age", "ant_chestpain", "post_chestpain", "abd_pain", "lumb_pain", "syncope",
    "sympt_time", "sev_pain", "sud_pain", "rip_pain", "ed_pain", "marfan", "family",
    "aort_valve", "aort_manip", "taa", "pulse", "focal_neuro_def", "murmur", "hyper",
    "diab", "smoke", "drug", "cad", "aaa", "cancer", "ddimer_value",
]

# Marginal rate in the derivation cohort (Table 1), for controls and for AAS cases.
RATES = {                  # column:        (controls, cases)
    "gender":            (0.49, 0.46),
    "ant_chestpain":     (0.80, 0.69),
    "post_chestpain":    (0.31, 0.51),
    "abd_pain":          (0.15, 0.27),
    "lumb_pain":         (0.05, 0.15),
    "syncope":           (0.09, 0.10),
    "sev_pain":          (0.35, 0.74),
    "sud_pain":          (0.35, 0.71),
    "rip_pain":          (0.14, 0.33),
    "ed_pain":           (0.50, 0.55),
    "marfan":            (0.007, 0.009),
    "family":            (0.019, 0.045),
    "aort_valve":        (0.039, 0.090),
    "aort_manip":        (0.008, 0.012),
    "taa":               (0.056, 0.246),
    "pulse":             (0.027, 0.182),
    "focal_neuro_def":   (0.030, 0.089),
    "murmur":            (0.007, 0.074),
    "hyper":             (0.46, 0.74),
    "diab":              (0.12, 0.069),
    "smoke":             (0.30, 0.30),
    "drug":              (0.008, 0.012),
    "cad":               (0.155, 0.096),
    "aaa":               (0.030, 0.108),
    "cancer":            (0.044, 0.027),
}


def main() -> None:
    rng = np.random.default_rng(SEED)
    y = (rng.random(N) < PREVALENCE).astype(int)
    df = pd.DataFrame(index=range(N))

    for col, (p_ctrl, p_case) in RATES.items():
        p = np.where(y == 1, p_case, p_ctrl)
        df[col] = (rng.random(N) < p).astype(int)

    df["age"] = np.clip(rng.normal(np.where(y == 1, 68, 58), 15, N), 18, 98).round().astype(int)
    df["sympt_time"] = np.clip(rng.lognormal(np.where(y == 1, 1.4, 2.2), 1.2, N), 0.5, 720).round(1)

    # D-dimer in ng/mL, as measured: the inference notebooks take the log themselves.
    df["ddimer_value"] = np.clip(
        rng.lognormal(np.where(y == 1, 8.5, 6.0), 1.0, N), 50, 40000
    ).round().astype(int)

    # A realistic sprinkle of missingness for the imputer to handle.
    for col in ["sympt_time", "smoke", "ed_pain", "sev_pain"]:
        df.loc[rng.random(N) < 0.02, col] = np.nan

    df = df.reindex(columns=FEATURES)
    df["final_dgn"] = y
    df.insert(0, "patient_id", [f"TOY{i:04d}" for i in range(N)])
    df.to_csv(OUT, index=False)

    print(f"SYNTHETIC cohort written to {OUT}")
    print(f"  {len(df)} patients, {int(y.sum())} simulated AAS ({y.mean():.1%})")
    print(f"  D-dimer ng/mL median {df.ddimer_value.median():.0f} "
          f"[{df.ddimer_value.quantile(.25):.0f}-{df.ddimer_value.quantile(.75):.0f}]")
    print(f"  missing values: {int(df.isna().sum().sum())}")
    print("\n  NOT real patients. Performance measured here is meaningless.")


if __name__ == "__main__":
    main()
