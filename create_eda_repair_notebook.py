import nbformat as nbf
from textwrap import dedent

nb = nbf.v4.new_notebook()
cells=[]

cells.append(nbf.v4.new_markdown_cell(dedent("""
# EDA-Guided ML Repair Notebook

Notebook ini memperbaiki pipeline forecasting dengan langsung mengadopsi insight dari `rangkuman_deep_eda_klaim.ipynb`:

- Heavy-tail: 11% klaim menyumbang 63.2% nominal (outlier-sensitive)
- Trifecta cost driver: `Usia 60+ -> Kanker -> Singapore`
- Repeat chronic pattern (genitourinari/hemodialisis) vs high-cost volatile (kanker)
- Plan `M-002` berkontribusi besar pada variabilitas

Perbaikan utama:
1. Tambah fitur EDA-driver (`trifecta_idx`, `outlier_rate`, `pct_plan_m002`, `repeat_avgfreq_idx`)
2. Hindari model tidak stabil untuk data pendek (drop SARIMAX exog kompleks)
3. Optimisasi ensemble dengan objective gabungan:
   - `strict` (7-fold) untuk robust
   - `operational` (3-fold) untuk near-term relevance
4. Koherensi matematis: `Total_Claim` dikonsolidasikan dengan `Freq x Severity`
""")))

cells.append(nbf.v4.new_code_cell(dedent("""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from IPython.display import display
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")

BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "dataset"
OUT_SUBMISSION = BASE_DIR / "submission_mepa_eda_repair.csv"
OUT_DETAILS = BASE_DIR / "forecast_details_mepa_eda_repair.json"

FORECAST_HORIZON = 5
FORECAST_MONTHS = pd.date_range("2025-08-01", periods=FORECAST_HORIZON, freq="MS")

print("BASE_DIR:", BASE_DIR)
""")))

cells.append(nbf.v4.new_markdown_cell("## 1) Metrics and Helpers"))

cells.append(nbf.v4.new_code_cell(dedent("""
def mepa(y_true, y_pred, eps=1e-9):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    den = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / den)) * 100)


def mdape(y_true, y_pred, eps=1e-9):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    den = np.maximum(np.abs(y_true), eps)
    return float(np.median(np.abs((y_true - y_pred) / den)) * 100)


def directional_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2:
        return np.nan
    return float(np.mean((np.diff(y_true) > 0) == (np.diff(y_pred) > 0)) * 100)


def evaluate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "MEPA": mepa(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MdAPE": mdape(y_true, y_pred),
        "Directional_Accuracy": directional_accuracy(y_true, y_pred),
    }


def parse_indo_rate_csv(path, value_col):
    fx = pd.read_csv(path)
    fx["Tanggal"] = pd.to_datetime(fx["Tanggal"], format="%d/%m/%Y", errors="coerce")
    fx[value_col] = (
        fx["Terakhir"].astype(str)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )
    fx = fx[["Tanggal", value_col]].dropna().sort_values("Tanggal")
    return fx.set_index("Tanggal").resample("MS")[value_col].mean().to_frame()


def slug_icd(text):
    return (
        str(text).lower()
        .replace("/", "_")
        .replace(" ", "_")
        .replace("-", "_")
        .replace("__", "_")
    )


def fit_ets_1step(series, use_log):
    if use_log:
        m = ExponentialSmoothing(np.log1p(series), trend="add", damped_trend=True, seasonal=None).fit(optimized=True)
        return float(np.expm1(m.forecast(1).iloc[0]))
    m = ExponentialSmoothing(series, trend="add", damped_trend=True, seasonal=None).fit(optimized=True)
    return float(m.forecast(1).iloc[0])
""")))

cells.append(nbf.v4.new_markdown_cell("## 2) Build EDA-Driven Monthly Features"))

cells.append(nbf.v4.new_code_cell(dedent("""
# dari analisis repeat (data user)
repeat_avg_freq = {
    "Genitourinari": 10.950820,
    "Neoplasma/Kanker": 10.686047,
    "Mata/Telinga": 3.664122,
    "Muskuloskeletal": 3.539474,
    "Gejala Umum": 3.405405,
    "Neoplasma/Darah": 3.187500,
    "Lainnya": 3.000000,
    "Cedera": 2.933333,
    "Respiratori": 2.866667,
    "Kulit": 2.789474,
}

OUTLIER_UPPER = 124_375_517


def load_clean_claims(path):
    df = pd.read_csv(path)
    df = df.drop_duplicates(
        subset=["Nomor Polis", "Tanggal Pasien Masuk RS", "Nominal Klaim Yang Disetujui"],
        keep="first",
    )
    df["Inpatient/Outpatient"] = df["Inpatient/Outpatient"].fillna("UNKNOWN")
    df["Lokasi RS"] = df["Lokasi RS"].fillna("Indonesia")
    df = df.dropna(subset=["ICD Diagnosis"])
    df["Bulan"] = pd.to_datetime(df["Bulan"], errors="coerce")
    df = df.dropna(subset=["Bulan"])
    return df


def build_monthly(df):
    df = df.copy()
    df["is_outlier"] = df["Nominal Klaim Yang Disetujui"] > OUTLIER_UPPER
    df["is_singapore"] = df["Lokasi RS"] == "Singapore"
    df["is_overseas"] = df["Lokasi RS"] != "Indonesia"
    df["is_cancer"] = df["ICD_Group"] == "Neoplasma/Kanker"

    base = (
        df.groupby("Bulan")
        .agg(
            Claim_Frequency=("Claim ID", "count"),
            Claim_Severity=("Nominal Klaim Yang Disetujui", "mean"),
            Total_Claim=("Nominal Klaim Yang Disetujui", "sum"),
            pct_age_60plus=("Usia", lambda x: (x >= 60).mean()),
            pct_overseas=("Lokasi RS", lambda x: (x != "Indonesia").mean()),
            pct_singapore=("Lokasi RS", lambda x: (x == "Singapore").mean()),
            pct_cancer=("ICD_Group", lambda x: (x == "Neoplasma/Kanker").mean()),
            pct_genitourinari=("ICD_Group", lambda x: (x == "Genitourinari").mean()),
            pct_cardio=("ICD_Group", lambda x: (x == "Kardiovaskular").mean()),
            pct_plan_m002=("Plan Code", lambda x: (x == "M-002").mean()),
            outlier_rate=("is_outlier", "mean"),
            outlier_total=("Nominal Klaim Yang Disetujui", lambda x: x[df.loc[x.index, "is_outlier"]].sum()),
        )
        .sort_index()
        .asfreq("MS")
    )

    base["outlier_total"] = base["outlier_total"].fillna(0.0)
    base["outlier_total_share"] = base["outlier_total"] / np.maximum(base["Total_Claim"], 1.0)

    # ICD share utk repeat index
    icd_counts = (
        df.pivot_table(index="Bulan", columns="ICD_Group", values="Claim ID", aggfunc="count", fill_value=0)
        .sort_index()
        .asfreq("MS")
        .fillna(0)
    )
    icd_pct = icd_counts.div(icd_counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

    for grp in repeat_avg_freq:
        col = f"pct_{slug_icd(grp)}"
        base[col] = icd_pct[grp] if grp in icd_pct.columns else 0.0

    total_w = float(sum(repeat_avg_freq.values()))
    weighted = 0.0
    for grp, w in repeat_avg_freq.items():
        weighted += base[f"pct_{slug_icd(grp)}"] * float(w)
    base["repeat_avgfreq_idx"] = weighted / max(total_w, 1e-9)

    # trifecta dari EDA: usia60+ x kanker x singapore
    base["trifecta_idx"] = base["pct_age_60plus"] * base["pct_cancer"] * base["pct_singapore"]

    # FX
    fx_myr = parse_indo_rate_csv(DATA_DIR / "Data Historis MYR_IDR.csv", "Avg_Kurs_MYR")
    fx_sgd = parse_indo_rate_csv(DATA_DIR / "Data Historis SGD_IDR.csv", "Avg_Kurs_SGD")
    base = base.join(fx_myr, how="left").join(fx_sgd, how="left")
    base[["Avg_Kurs_MYR", "Avg_Kurs_SGD"]] = base[["Avg_Kurs_MYR", "Avg_Kurs_SGD"]].ffill().bfill()

    # lag features
    lag_cols = [
        "Claim_Frequency", "Claim_Severity", "Total_Claim",
        "repeat_avgfreq_idx", "trifecta_idx", "pct_overseas", "pct_singapore",
        "pct_age_60plus", "pct_cancer", "pct_genitourinari", "pct_cardio",
        "pct_plan_m002", "outlier_rate", "outlier_total_share",
        "Avg_Kurs_SGD", "Avg_Kurs_MYR",
    ]

    for c in lag_cols:
        for l in [1, 2, 3]:
            base[f"{c}_lag{l}"] = base[c].shift(l)

    base["month_num"] = base.index.month
    base["mon_sin"] = np.sin(2 * np.pi * base["month_num"] / 12)
    base["mon_cos"] = np.cos(2 * np.pi * base["month_num"] / 12)

    return base


claims = load_clean_claims(DATA_DIR / "Data_Klaim_Enriched.csv")
monthly = build_monthly(claims)

print("Monthly shape:", monthly.shape)
print("Date range:", monthly.index.min().date(), "to", monthly.index.max().date())

display(monthly[[
    "Claim_Frequency", "Claim_Severity", "Total_Claim",
    "trifecta_idx", "repeat_avgfreq_idx", "outlier_rate", "pct_plan_m002"
]].head(8))
""")))

cells.append(nbf.v4.new_markdown_cell("## 3) Backtest Engine (EDA Features)"))

cells.append(nbf.v4.new_code_cell(dedent("""
TARGETS = {
    "Claim_Frequency": {"use_log": False},
    "Claim_Severity": {"use_log": True},
    "Total_Claim": {"use_log": True},
}

MODELS = ["naive", "ets", "xgb_eda"]

FEATURES = {
    "Claim_Frequency": [
        "Claim_Frequency_lag1", "Claim_Frequency_lag2", "Claim_Frequency_lag3",
        "repeat_avgfreq_idx_lag1", "trifecta_idx_lag1",
        "pct_overseas_lag1", "pct_singapore_lag1", "pct_plan_m002_lag1",
        "outlier_rate_lag1", "Avg_Kurs_SGD_lag1", "Avg_Kurs_MYR_lag1",
        "mon_sin", "mon_cos",
    ],
    "Claim_Severity": [
        "Claim_Severity_lag1", "Claim_Severity_lag2", "Claim_Severity_lag3",
        "trifecta_idx_lag1", "pct_cancer_lag1", "pct_singapore_lag1",
        "outlier_rate_lag1", "outlier_total_share_lag1",
        "Avg_Kurs_SGD_lag1", "Avg_Kurs_MYR_lag1",
        "mon_sin", "mon_cos",
    ],
    "Total_Claim": [
        "Total_Claim_lag1", "Total_Claim_lag2", "Total_Claim_lag3",
        "trifecta_idx_lag1", "repeat_avgfreq_idx_lag1",
        "pct_overseas_lag1", "pct_plan_m002_lag1",
        "outlier_rate_lag1", "outlier_total_share_lag1",
        "Avg_Kurs_SGD_lag1", "Avg_Kurs_MYR_lag1",
        "mon_sin", "mon_cos",
    ],
}


def predict_1step(model_name, target, train_df, test_row, use_log):
    y = train_df[target].astype(float)

    if model_name == "naive":
        return max(float(y.iloc[-1]), 0.0)

    if model_name == "ets":
        p = fit_ets_1step(y, use_log=use_log)
        return max(float(p), 0.0)

    if model_name == "xgb_eda":
        feats = FEATURES[target]
        tr = train_df[[target] + feats].dropna().copy()
        if len(tr) < 8:
            return max(float(y.iloc[-1]), 0.0)

        Xtr = tr[feats].astype(float)
        ytr = np.log1p(tr[target].astype(float)) if use_log else tr[target].astype(float)
        xte = test_row[feats].astype(float).to_frame().T

        mdl = XGBRegressor(
            n_estimators=400,
            max_depth=3,
            learning_rate=0.04,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=1.0,
            reg_lambda=2.0,
            random_state=42,
            n_jobs=1,
            verbosity=0,
        )
        mdl.fit(Xtr, ytr)
        p = float(mdl.predict(xte)[0])
        p = float(np.expm1(p)) if use_log else p
        return max(p, 0.0)

    raise ValueError(model_name)


def backtest_target(target, n_test, min_train=8):
    use_log = TARGETS[target]["use_log"]
    df = monthly.copy()
    feat = FEATURES[target]

    work = df[[target] + feat].dropna().copy()
    start = max(min_train, len(work) - n_test)

    y_true = []
    pred = {m: [] for m in MODELS}
    dates = []

    for i in range(start, len(work)):
        tr = work.iloc[:i]
        te = work.iloc[i]
        y_true.append(float(te[target]))
        dates.append(work.index[i])

        for m in MODELS:
            try:
                p = predict_1step(m, target, tr, te, use_log)
            except Exception:
                p = float(tr[target].iloc[-1])
            pred[m].append(max(0.0, float(p)))

    return {
        "dates": dates,
        "y_true": np.array(y_true, dtype=float),
        "preds": {k: np.array(v, dtype=float) for k, v in pred.items()},
    }
""")))

cells.append(nbf.v4.new_markdown_cell("## 4) Ensemble Optimization (Strict + Operational Balance)"))

cells.append(nbf.v4.new_code_cell(dedent("""
WINDOWS = {"strict": 7, "operational": 3}
LAMBDA_STRICT = 0.70

opt = {}

for t in TARGETS:
    bt = {w: backtest_target(t, n_test=n) for w, n in WINDOWS.items()}

    # ranking per model
    rank = []
    for m in MODELS:
        row = {"model": m}
        for w in WINDOWS:
            y = bt[w]["y_true"]
            p = bt[w]["preds"][m]
            row[f"MEPA_{w}"] = mepa(y, p)
            row[f"MAE_{w}"] = mean_absolute_error(y, p)
        rank.append(row)
    rank_df = pd.DataFrame(rank).sort_values("MEPA_operational").reset_index(drop=True)

    Ys = bt["strict"]["y_true"]
    Yo = bt["operational"]["y_true"]
    Ps = np.column_stack([bt["strict"]["preds"][m] for m in MODELS])
    Po = np.column_stack([bt["operational"]["preds"][m] for m in MODELS])

    def obj_w(w):
        ps = Ps.dot(w)
        po = Po.dot(w)
        return LAMBDA_STRICT * mepa(Ys, ps) + (1 - LAMBDA_STRICT) * mepa(Yo, po)

    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bnds = [(0.0, 1.0)] * len(MODELS)
    x0 = np.ones(len(MODELS)) / len(MODELS)
    res_w = minimize(obj_w, x0=x0, method="SLSQP", bounds=bnds, constraints=cons)
    w_opt = res_w.x if res_w.success else x0

    # affine calibration over combined objective
    def obj_ab(v):
        a, b = v
        ps = np.maximum(0.0, a * Ps.dot(w_opt) + b)
        po = np.maximum(0.0, a * Po.dot(w_opt) + b)
        return LAMBDA_STRICT * mepa(Ys, ps) + (1 - LAMBDA_STRICT) * mepa(Yo, po)

    mean_ref = float(np.mean(Yo))
    res_ab = minimize(
        obj_ab,
        x0=np.array([1.0, 0.0]),
        bounds=[(0.7, 1.3), (-0.2 * mean_ref, 0.2 * mean_ref)],
        method="L-BFGS-B",
    )
    a_opt, b_opt = (res_ab.x if res_ab.success else [1.0, 0.0])

    metrics = {}
    for w in WINDOWS:
        y = bt[w]["y_true"]
        P = np.column_stack([bt[w]["preds"][m] for m in MODELS])
        raw = P.dot(w_opt)
        cal = np.maximum(0.0, a_opt * raw + b_opt)
        metrics[w] = {"raw": evaluate_metrics(y, raw), "cal": evaluate_metrics(y, cal)}

    opt[t] = {
        "bt": bt,
        "rank_df": rank_df,
        "weights": w_opt,
        "a": float(a_opt),
        "b": float(b_opt),
        "metrics": metrics,
    }

    print(f"\\n=== {t} ===")
    display(rank_df)
    print("weights:", dict(zip(MODELS, np.round(w_opt, 4))))
    print("a,b:", round(float(a_opt), 6), round(float(b_opt), 6))
    print("strict MEPA cal:", round(metrics['strict']['cal']['MEPA'], 4))
    print("operational MEPA cal:", round(metrics['operational']['cal']['MEPA'], 4))

summary = pd.DataFrame([
    {
        "target": t,
        "strict_mepa": opt[t]["metrics"]["strict"]["cal"]["MEPA"],
        "operational_mepa": opt[t]["metrics"]["operational"]["cal"]["MEPA"],
    }
    for t in TARGETS
])

print("\\n=== Summary ===")
display(summary)
""")))

cells.append(nbf.v4.new_markdown_cell("## 5) Forecast Future (EDA-consistent)"))

cells.append(nbf.v4.new_code_cell(dedent("""
RAW_DRIVERS = [
    "repeat_avgfreq_idx", "trifecta_idx", "pct_overseas", "pct_singapore", "pct_age_60plus",
    "pct_cancer", "pct_plan_m002", "outlier_rate", "outlier_total_share", "Avg_Kurs_SGD", "Avg_Kurs_MYR",
]


def forecast_raw_drivers(monthly_df):
    fut = pd.DataFrame(index=FORECAST_MONTHS)
    for col in RAW_DRIVERS:
        s = monthly_df[col].astype(float).dropna()
        pred = np.repeat(float(s.iloc[-1]), FORECAST_HORIZON)
        if len(s) >= 6:
            try:
                m = ExponentialSmoothing(s, trend="add", damped_trend=True, seasonal=None).fit(optimized=True)
                pred = m.forecast(FORECAST_HORIZON).values
            except Exception:
                pred = np.repeat(float(s.iloc[-1]), FORECAST_HORIZON)

        if col.startswith("pct_") or col in ["repeat_avgfreq_idx", "trifecta_idx", "outlier_rate", "outlier_total_share"]:
            pred = np.clip(pred, 0.0, 1.0)

        fut[col] = pred
    return fut


def build_lag_frame_with_future(monthly_df, future_raw):
    all_raw = pd.concat([monthly_df[RAW_DRIVERS], future_raw], axis=0)

    for c in RAW_DRIVERS:
        for l in [1, 2, 3]:
            all_raw[f"{c}_lag{l}"] = all_raw[c].shift(l)

    all_raw["month_num"] = all_raw.index.month
    all_raw["mon_sin"] = np.sin(2 * np.pi * all_raw["month_num"] / 12)
    all_raw["mon_cos"] = np.cos(2 * np.pi * all_raw["month_num"] / 12)

    return all_raw


def forecast_model_full(target, model_name, monthly_df, lag_frame):
    use_log = TARGETS[target]["use_log"]
    y = monthly_df[target].astype(float)

    if model_name == "naive":
        return np.repeat(float(y.iloc[-1]), FORECAST_HORIZON)

    if model_name == "ets":
        if use_log:
            m = ExponentialSmoothing(np.log1p(y), trend="add", damped_trend=True, seasonal=None).fit(optimized=True)
            p = np.expm1(m.forecast(FORECAST_HORIZON).values)
        else:
            m = ExponentialSmoothing(y, trend="add", damped_trend=True, seasonal=None).fit(optimized=True)
            p = m.forecast(FORECAST_HORIZON).values
        return np.maximum(p, 0.0)

    if model_name == "xgb_eda":
        feats = FEATURES[target]
        train = monthly_df[[target] + feats].dropna().copy()
        if len(train) < 8:
            return np.repeat(float(y.iloc[-1]), FORECAST_HORIZON)

        Xtr = train[feats].astype(float)
        ytr = np.log1p(train[target].astype(float)) if use_log else train[target].astype(float)

        mdl = XGBRegressor(
            n_estimators=400,
            max_depth=3,
            learning_rate=0.04,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=1.0,
            reg_lambda=2.0,
            random_state=42,
            n_jobs=1,
            verbosity=0,
        )
        mdl.fit(Xtr, ytr)

        # recursive for target lags
        hist = list(monthly_df[target].astype(float).values)
        preds = []

        for d in FORECAST_MONTHS:
            lag1 = hist[-1]
            lag2 = hist[-2] if len(hist) >= 2 else hist[-1]
            lag3 = hist[-3] if len(hist) >= 3 else hist[-1]

            row = {
                f"{target}_lag1": lag1,
                f"{target}_lag2": lag2,
                f"{target}_lag3": lag3,
                "repeat_avgfreq_idx_lag1": float(lag_frame.loc[d, "repeat_avgfreq_idx_lag1"]),
                "trifecta_idx_lag1": float(lag_frame.loc[d, "trifecta_idx_lag1"]),
                "pct_overseas_lag1": float(lag_frame.loc[d, "pct_overseas_lag1"]),
                "pct_singapore_lag1": float(lag_frame.loc[d, "pct_singapore_lag1"]),
                "pct_plan_m002_lag1": float(lag_frame.loc[d, "pct_plan_m002_lag1"]),
                "outlier_rate_lag1": float(lag_frame.loc[d, "outlier_rate_lag1"]),
                "outlier_total_share_lag1": float(lag_frame.loc[d, "outlier_total_share_lag1"]),
                "Avg_Kurs_SGD_lag1": float(lag_frame.loc[d, "Avg_Kurs_SGD_lag1"]),
                "Avg_Kurs_MYR_lag1": float(lag_frame.loc[d, "Avg_Kurs_MYR_lag1"]),
                "mon_sin": float(lag_frame.loc[d, "mon_sin"]),
                "mon_cos": float(lag_frame.loc[d, "mon_cos"]),
            }

            x = pd.DataFrame([row])[feats]
            p = float(mdl.predict(x)[0])
            p = float(np.expm1(p)) if use_log else p
            p = max(0.0, p)
            preds.append(p)
            hist.append(p)

        return np.array(preds)

    raise ValueError(model_name)


future_raw = forecast_raw_drivers(monthly)
lag_frame = build_lag_frame_with_future(monthly, future_raw)

# add lag for target & some drivers from history for recursive rows
for t in TARGETS:
    hist = pd.concat([monthly[[t]], pd.DataFrame(index=FORECAST_MONTHS, columns=[t], dtype=float)], axis=0)
    for l in [1, 2, 3]:
        lag_frame[f"{t}_lag{l}"] = hist[t].shift(l)

# forecast each target from optimized ensemble
fc = {}
for t in TARGETS:
    each = {}
    for m in MODELS:
        each[m] = forecast_model_full(t, m, monthly, lag_frame)

    M = np.column_stack([each[m] for m in MODELS])
    w = opt[t]["weights"]
    raw = M.dot(w)
    cal = np.maximum(0.0, opt[t]["a"] * raw + opt[t]["b"])

    if t == "Claim_Frequency":
        cal = np.round(cal).astype(int)

    fc[t] = {"per_model": each, "raw": raw, "cal": cal}

# EDA coherence: blend direct total with derived total (Freq x Severity)
derived_total = fc["Claim_Frequency"]["cal"] * fc["Claim_Severity"]["cal"]
direct_total = fc["Total_Claim"]["cal"]

# optimize alpha from strict backtest
y_strict = opt["Total_Claim"]["bt"]["strict"]["y_true"]
P_strict = np.column_stack([opt["Total_Claim"]["bt"]["strict"]["preds"][m] for m in MODELS])
p_raw_strict = P_strict.dot(opt["Total_Claim"]["weights"])
p_dir_strict = np.maximum(0.0, opt["Total_Claim"]["a"] * p_raw_strict + opt["Total_Claim"]["b"])

# proxy derived strict from frequency*severity calibrated strict
P_f = np.column_stack([opt["Claim_Frequency"]["bt"]["strict"]["preds"][m] for m in MODELS])
P_s = np.column_stack([opt["Claim_Severity"]["bt"]["strict"]["preds"][m] for m in MODELS])

p_f = np.maximum(0.0, opt["Claim_Frequency"]["a"] * P_f.dot(opt["Claim_Frequency"]["weights"]) + opt["Claim_Frequency"]["b"])
p_s = np.maximum(0.0, opt["Claim_Severity"]["a"] * P_s.dot(opt["Claim_Severity"]["weights"]) + opt["Claim_Severity"]["b"])
p_der_strict = p_f * p_s

alphas = np.linspace(0, 1, 41)
alpha_scores = [(a, mepa(y_strict, a * p_dir_strict + (1 - a) * p_der_strict)) for a in alphas]
alpha_opt, alpha_mepa = sorted(alpha_scores, key=lambda x: x[1])[0]

total_final = alpha_opt * direct_total + (1 - alpha_opt) * derived_total

forecast_df = pd.DataFrame({
    "Bulan": FORECAST_MONTHS,
    "Claim_Frequency": fc["Claim_Frequency"]["cal"],
    "Claim_Severity": fc["Claim_Severity"]["cal"],
    "Total_Claim_Direct": direct_total,
    "Total_Claim_Derived": derived_total,
    "Total_Claim_Final": total_final,
})

print(f"alpha_opt (Total direct vs derived): {alpha_opt:.2f}, strict_mepa={alpha_mepa:.4f}")
display(forecast_df)
""")))

cells.append(nbf.v4.new_markdown_cell("## 6) Save Submission and Details"))

cells.append(nbf.v4.new_code_cell(dedent("""
# build submission
rows = []
month_labels = [d.strftime("%Y_%m") for d in FORECAST_MONTHS]
for i, m in enumerate(month_labels):
    rows.append({"id": f"{m}_Claim_Frequency", "value": float(forecast_df.loc[i, "Claim_Frequency"])})
    rows.append({"id": f"{m}_Claim_Severity", "value": float(forecast_df.loc[i, "Claim_Severity"])})
    rows.append({"id": f"{m}_Total_Claim", "value": float(forecast_df.loc[i, "Total_Claim_Final"])})

submission = pd.DataFrame(rows)
submission.to_csv(OUT_SUBMISSION, index=False)

# CI from operational APE
ci_pack = {}
for t in TARGETS:
    y = opt[t]["bt"]["operational"]["y_true"]
    P = np.column_stack([opt[t]["bt"]["operational"]["preds"][m] for m in MODELS])
    p_raw = P.dot(opt[t]["weights"])
    p_cal = np.maximum(0.0, opt[t]["a"] * p_raw + opt[t]["b"])

    ape = np.abs((y - p_cal) / np.maximum(np.abs(y), 1e-9))
    q80 = float(np.quantile(ape, 0.80))
    q95 = float(np.quantile(ape, 0.95))

    fut = np.asarray(fc[t]["cal"], dtype=float)
    ci80 = np.column_stack([np.maximum(fut * (1 - q80), 0.0), fut * (1 + q80)])
    ci95 = np.column_stack([np.maximum(fut * (1 - q95), 0.0), fut * (1 + q95)])

    ci_pack[t] = {"ci80": ci80.tolist(), "ci95": ci95.tolist(), "q80_ape": q80, "q95_ape": q95}

summary_check = pd.DataFrame([
    {
        "target": t,
        "strict_mepa_cal": opt[t]["metrics"]["strict"]["cal"]["MEPA"],
        "operational_mepa_cal": opt[t]["metrics"]["operational"]["cal"]["MEPA"],
    }
    for t in TARGETS
])

# details json
details = {
    "forecast_months": [d.strftime("%Y-%m") for d in FORECAST_MONTHS],
    "repeat_avg_freq_input": repeat_avg_freq,
    "eda_drivers": [
        "trifecta_idx (age60+ * cancer * singapore)",
        "outlier_rate",
        "outlier_total_share",
        "pct_plan_m002",
        "repeat_avgfreq_idx",
    ],
    "models": MODELS,
    "window_weights": {"strict_weight": 0.70, "operational_weight": 0.30},
    "targets": {},
    "total_blend": {"alpha_opt": float(alpha_opt), "strict_mepa": float(alpha_mepa)},
    "forecast": forecast_df.assign(Bulan=forecast_df["Bulan"].dt.strftime("%Y-%m")).to_dict(orient="records"),
    "confidence_intervals": ci_pack,
}

for t in TARGETS:
    details["targets"][t] = {
        "weights": {m: float(w) for m, w in zip(MODELS, opt[t]["weights"])},
        "calibration": {"a": float(opt[t]["a"]), "b": float(opt[t]["b"] )},
        "strict": opt[t]["metrics"]["strict"],
        "operational": opt[t]["metrics"]["operational"],
        "ranking": opt[t]["rank_df"].to_dict(orient="records"),
    }

with open(OUT_DETAILS, "w", encoding="utf-8") as f:
    json.dump(details, f, indent=2)

print("Saved:", OUT_SUBMISSION)
print("Saved:", OUT_DETAILS)

display(submission)
display(summary_check)
""")))

cells.append(nbf.v4.new_markdown_cell("## 7) Plot"))

cells.append(nbf.v4.new_code_cell(dedent("""
fig, axes = plt.subplots(3, 1, figsize=(14, 14))

series_map = {
    "Claim_Frequency": "Claim_Frequency",
    "Claim_Severity": "Claim_Severity",
    "Total_Claim": "Total_Claim_Final",
}

for ax, (raw_name, fc_name) in zip(axes, series_map.items()):
    ax.plot(monthly.index, monthly[raw_name], marker="o", label="Actual")
    ax.plot(FORECAST_MONTHS, forecast_df[fc_name], marker="s", linestyle="--", label="Forecast")

    ci80 = np.array(ci_pack[raw_name]["ci80"])
    ax.fill_between(FORECAST_MONTHS, ci80[:, 0], ci80[:, 1], alpha=0.2, label="80% CI")

    ax.set_title(raw_name)
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
""")))

nb['cells']=cells
nb['metadata']={
    'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
    'language_info':{'name':'python','version':'3.12'}
}

out='ml_claims_forecast_eda_repair.ipynb'
with open(out,'w',encoding='utf-8') as f:
    nbf.write(nb,f)
print(f'Notebook written: {out}')
