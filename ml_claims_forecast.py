#!/usr/bin/env python3
"""
"""
# %% [markdown]
# # Machine Learning Pipeline — Prediksi Klaim Asuransi Kesehatan Individu
# 
# Forecasts 15 values (Claim Frequency, Severity, Total) for Aug-Dec 2025
# based on historical data Jan 2024 - Jul 2025.
# 
# Models used:
#   - ARIMA / auto-ARIMA (pmdarima)
#   - Exponential Smoothing (Holt-Winters)
#   - SARIMAX with exogenous variables (currency rates + portfolio composition)
#   - XGBoost (factor analysis / feature importance via SHAP)
# 
# Outputs:
#   - `submission.csv`          : 15 predicted values
#   - `ml_forecast.log`         : detailed log of entire pipeline
#   - `models/`                 : saved model objects (joblib)
#   - `charts/`                 : evaluation and forecast plots

# %%
import os

import os
import sys
import logging
import warnings
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pmdarima as pm

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import xgboost as xgb
import shap
import joblib

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
try:
    BASE_DIR = Path(__file__).parent
except NameError:
    BASE_DIR = Path.cwd()
DATASET_DIR = BASE_DIR / 'dataset'
MODELS_DIR = BASE_DIR / 'models'
CHARTS_DIR = BASE_DIR / 'charts'
LOG_FILE = BASE_DIR / 'ml_forecast.log'
SUBMISSION_FILE = BASE_DIR / 'submission.csv'

FORECAST_HORIZON = 5  # Aug-Dec 2025
FORECAST_MONTHS = pd.date_range('2025-08-01', periods=5, freq='MS')

MODELS_DIR.mkdir(exist_ok=True)
CHARTS_DIR.mkdir(exist_ok=True)

# ============================================================================
# LOGGING SETUP
# ============================================================================
logger = logging.getLogger('MLForecast')
logger.setLevel(logging.DEBUG)

# File handler — detailed
fh = logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8')
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter(
    '%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
))

# Console handler — info+
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter('%(levelname)-8s | %(message)s'))

logger.addHandler(fh)
logger.addHandler(ch)

def log_separator(title):
    logger.info('=' * 70)
    logger.info(f'  {title}')
    logger.info('=' * 70)


# ============================================================================
# %% [markdown]
# ## PHASE 1: DATA LOADING & CLEANING
# ============================================================================
# %%
def load_and_clean_data():
    """Load Data_Klaim_Enriched.csv and perform cleaning."""
    log_separator('PHASE 1: DATA LOADING & CLEANING')

    df = pd.read_csv(DATASET_DIR / 'Data_Klaim_Enriched.csv')
    logger.info(f'Loaded Data_Klaim_Enriched.csv: {df.shape[0]} rows × {df.shape[1]} cols')

    # --- 1a. Drop duplicates ---
    n_before = len(df)
    df = df.drop_duplicates(
        subset=['Nomor Polis', 'Tanggal Pasien Masuk RS', 'Nominal Klaim Yang Disetujui'],
        keep='first'
    )
    n_dropped = n_before - len(df)
    logger.info(f'Dropped {n_dropped} duplicate rows → {len(df)} rows remaining')

    # --- 1b. Handle missing values ---
    df['Inpatient/Outpatient'] = df['Inpatient/Outpatient'].fillna('UNKNOWN')
    df['Lokasi RS'] = df['Lokasi RS'].fillna('Indonesia')
    n_missing_icd = df['ICD Diagnosis'].isna().sum()
    df = df.dropna(subset=['ICD Diagnosis'])
    logger.info(f'Imputed Inpatient/Outpatient & Lokasi RS missing values')
    logger.info(f'Dropped {n_missing_icd} rows with missing ICD Diagnosis')

    # --- 1c. Parse dates ---
    date_cols = ['Tanggal Pembayaran Klaim', 'Tanggal Pasien Masuk RS',
                 'Tanggal Pasien Keluar RS', 'Tanggal Lahir', 'Tanggal Efektif Polis']
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    # --- 1d. Ensure Bulan is datetime ---
    df['Bulan_dt'] = pd.to_datetime(df['Bulan'])

    logger.info(f'Data date range: {df["Bulan_dt"].min()} to {df["Bulan_dt"].max()}')
    logger.info(f'Final cleaned dataset: {df.shape[0]} rows × {df.shape[1]} cols')

    return df


# ============================================================================
# %% [markdown]
# ## PHASE 1b: LOAD EXCHANGE RATE DATA
# ============================================================================
# %%
def load_exchange_rates():
    """Load and parse MYR/IDR and SGD/IDR exchange rate data."""
    log_separator('PHASE 1b: LOADING EXCHANGE RATE DATA')

    def parse_indo_csv(filepath, col_name):
        """Parse Indonesian-formatted CSV (titik=ribuan, koma=desimal)."""
        raw = pd.read_csv(filepath)
        raw['Tanggal'] = pd.to_datetime(raw['Tanggal'], format='%d/%m/%Y')
        raw[col_name] = (
            raw['Terakhir']
            .str.replace('.', '', regex=False)
            .str.replace(',', '.', regex=False)
            .astype(float)
        )
        return raw[['Tanggal', col_name]].sort_values('Tanggal').reset_index(drop=True)

    kurs_myr = parse_indo_csv(DATASET_DIR / 'Data Historis MYR_IDR.csv', 'Kurs_MYR')
    kurs_sgd = parse_indo_csv(DATASET_DIR / 'Data Historis SGD_IDR.csv', 'Kurs_SGD')

    logger.info(f'MYR/IDR: {len(kurs_myr)} daily records, '
                f'{kurs_myr["Tanggal"].min().date()} to {kurs_myr["Tanggal"].max().date()}')
    logger.info(f'SGD/IDR: {len(kurs_sgd)} daily records, '
                f'{kurs_sgd["Tanggal"].min().date()} to {kurs_sgd["Tanggal"].max().date()}')

    # Aggregate to monthly average
    kurs_myr_monthly = (
        kurs_myr.set_index('Tanggal')
        .resample('MS')['Kurs_MYR']
        .mean()
        .reset_index()
    )
    kurs_myr_monthly.columns = ['Bulan', 'Avg_Kurs_MYR']

    kurs_sgd_monthly = (
        kurs_sgd.set_index('Tanggal')
        .resample('MS')['Kurs_SGD']
        .mean()
        .reset_index()
    )
    kurs_sgd_monthly.columns = ['Bulan', 'Avg_Kurs_SGD']

    logger.info(f'Monthly MYR: {len(kurs_myr_monthly)} months')
    logger.info(f'Monthly SGD: {len(kurs_sgd_monthly)} months')

    return kurs_myr_monthly, kurs_sgd_monthly


# ============================================================================
# %% [markdown]
# ## PHASE 2: MONTHLY AGGREGATION & FEATURE ENGINEERING
# ============================================================================
# %%
def build_monthly_timeseries(df, kurs_myr_monthly, kurs_sgd_monthly):
    """Aggregate per-claim data to monthly time series."""
    log_separator('PHASE 2: MONTHLY AGGREGATION')

    # --- 2a. Core aggregation ---
    monthly = df.groupby('Bulan_dt').agg(
        Claim_Frequency=('Claim ID', 'count'),
        Claim_Severity=('Nominal Klaim Yang Disetujui', 'mean'),
        Total_Claim=('Nominal Klaim Yang Disetujui', 'sum')
    ).reset_index()
    monthly = monthly.rename(columns={'Bulan_dt': 'Bulan'})
    monthly = monthly.sort_values('Bulan').reset_index(drop=True)

    logger.info(f'Monthly time series: {len(monthly)} data points')
    for _, row in monthly.iterrows():
        logger.debug(f'  {row["Bulan"].strftime("%Y-%m")}: '
                      f'Freq={row["Claim_Frequency"]:.0f}, '
                      f'Severity=Rp {row["Claim_Severity"]:,.0f}, '
                      f'Total=Rp {row["Total_Claim"]:,.0f}')

    # --- 2b. Exogenous features from claim composition ---
    monthly_features = df.groupby('Bulan_dt').agg(
        pct_age_60plus=('Usia', lambda x: (x >= 60).mean()),
        avg_age=('Usia', 'mean'),
        pct_male=('Gender', lambda x: (x == 'M').mean()),
        pct_cancer=('ICD_Group', lambda x: (x == 'Neoplasma/Kanker').mean()),
        pct_cardio=('ICD_Group', lambda x: (x == 'Kardiovaskular').mean()),
        pct_renal=('ICD_Group', lambda x: (x == 'Genitourinari').mean()),
        pct_overseas=('Lokasi RS', lambda x: (x != 'Indonesia').mean()),
        pct_singapore=('Lokasi RS', lambda x: (x == 'Singapore').mean()),
        pct_inpatient=('Inpatient/Outpatient', lambda x: (x == 'IP').mean()),
        pct_plan_m002=('Plan Code', lambda x: (x == 'M-002').mean()),
        median_nominal=('Nominal Klaim Yang Disetujui', 'median'),
        std_nominal=('Nominal Klaim Yang Disetujui', 'std'),
        pct_outlier=('Nominal Klaim Yang Disetujui', lambda x: (x > 124_375_517).mean()),
        unique_polis=('Nomor Polis', 'nunique'),
        avg_claims_per_polis=('Nomor Polis', lambda x: len(x) / x.nunique()),
    ).reset_index()
    monthly_features = monthly_features.rename(columns={'Bulan_dt': 'Bulan'})

    # --- 2c. Merge features into monthly ---
    monthly = monthly.merge(monthly_features, on='Bulan', how='left')

    # --- 2d. Merge exchange rates ---
    monthly = monthly.merge(kurs_myr_monthly, on='Bulan', how='left')
    monthly = monthly.merge(kurs_sgd_monthly, on='Bulan', how='left')

    # Forward-fill any missing rates
    monthly['Avg_Kurs_MYR'] = monthly['Avg_Kurs_MYR'].ffill().bfill()
    monthly['Avg_Kurs_SGD'] = monthly['Avg_Kurs_SGD'].ffill().bfill()

    # Currency change percentage
    monthly['MYR_change_pct'] = monthly['Avg_Kurs_MYR'].pct_change() * 100
    monthly['SGD_change_pct'] = monthly['Avg_Kurs_SGD'].pct_change() * 100
    monthly['MYR_change_pct'] = monthly['MYR_change_pct'].fillna(0)
    monthly['SGD_change_pct'] = monthly['SGD_change_pct'].fillna(0)

    # --- 2e. Log-transformations ---
    monthly['log_Severity'] = np.log1p(monthly['Claim_Severity'])
    monthly['log_Total'] = np.log1p(monthly['Total_Claim'])

    # Set datetime index
    monthly = monthly.set_index('Bulan')
    monthly.index.freq = 'MS'

    logger.info(f'Features added: {list(monthly_features.columns[1:])}')
    logger.info(f'Exchange rates merged. Final monthly shape: {monthly.shape}')

    return monthly


# ============================================================================
# %% [markdown]
# ## PHASE 3: TIME SERIES ANALYSIS & STATIONARITY
# ============================================================================
# %%
def analyze_stationarity(monthly):
    """Run ADF test and log results for each target."""
    log_separator('PHASE 3: STATIONARITY ANALYSIS')

    targets = {
        'Claim_Frequency': monthly['Claim_Frequency'],
        'Claim_Severity': monthly['Claim_Severity'],
        'log_Severity': monthly['log_Severity'],
        'Total_Claim': monthly['Total_Claim'],
        'log_Total': monthly['log_Total'],
    }

    adf_results = {}
    for name, series in targets.items():
        result = adfuller(series.dropna(), autolag='AIC')
        adf_results[name] = {
            'ADF Statistic': result[0],
            'p-value': result[1],
            'Critical Values': result[4],
            'Stationary': result[1] < 0.05
        }
        status = '✓ STATIONARY' if result[1] < 0.05 else '✗ NON-STATIONARY'
        logger.info(f'{name}: ADF={result[0]:.4f}, p={result[1]:.4f} → {status}')

    # --- Plot ACF/PACF ---
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    for i, (name, series) in enumerate([
        ('Claim_Frequency', monthly['Claim_Frequency']),
        ('log_Severity', monthly['log_Severity']),
        ('log_Total', monthly['log_Total'])
    ]):
        plot_acf(series, ax=axes[i, 0], lags=8, title=f'ACF — {name}')
        plot_pacf(series, ax=axes[i, 1], lags=8, title=f'PACF — {name}')
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / 'acf_pacf_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f'Saved ACF/PACF plots to {CHARTS_DIR / "acf_pacf_analysis.png"}')

    return adf_results


# ============================================================================
# %% [markdown]
# ## PHASE 4: MODEL BUILDING
# ============================================================================
# %%
def evaluate_forecast(actual, predicted, model_name):
    """Compute MAE, RMSE, MAPE for a forecast."""
    actual = np.array(actual)
    predicted = np.array(predicted)

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    # MAPE — avoid division by zero
    mask = actual != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    else:
        mape = np.inf

    # Directional accuracy (for > 1 data point)
    if len(actual) > 1:
        actual_dir = np.diff(actual) > 0
        pred_dir = np.diff(predicted) > 0
        dir_acc = np.mean(actual_dir == pred_dir) * 100
    else:
        dir_acc = np.nan

    logger.info(f'  {model_name}: MAE={mae:,.0f}, RMSE={rmse:,.0f}, '
                f'MAPE={mape:.1f}%, DirAcc={dir_acc:.0f}%')

    return {'model': model_name, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape,
            'DirAcc': dir_acc}


def expanding_window_cv(series, model_builder, n_test=7, series_name=''):
    """
    Expanding window cross-validation.
    Train on [0:train_end], test on [train_end].
    model_builder(train_series) → should return forecast (single value).
    """
    results = []
    n = len(series)
    min_train = n - n_test  # Start with enough training data (12 months)

    actuals = []
    predictions = []

    for i in range(n_test):
        train_end = min_train + i
        train = series.iloc[:train_end]
        actual = series.iloc[train_end]

        try:
            pred = model_builder(train)
            actuals.append(actual)
            predictions.append(pred)
        except Exception as e:
            logger.warning(f'  CV fold {i+1} failed for {series_name}: {e}')
            continue

    if len(actuals) > 0:
        metrics = evaluate_forecast(actuals, predictions, f'{series_name} CV')
        return metrics, actuals, predictions
    return None, [], []


def fit_auto_arima(series, series_name='', seasonal=False, m=12):
    """Fit auto-ARIMA using pmdarima."""
    logger.info(f'  Fitting auto-ARIMA for {series_name}...')
    model = pm.auto_arima(
        series,
        start_p=0, start_q=0,
        max_p=3, max_q=3,
        d=None,  # auto-detect
        seasonal=seasonal,
        m=m if seasonal else 1,
        start_P=0, start_Q=0,
        max_P=2, max_Q=2,
        D=None if seasonal else 0,
        trace=False,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True,
        information_criterion='aic',
        n_fits=50
    )
    logger.info(f'  {series_name} auto-ARIMA order: {model.order}, '
                f'seasonal_order: {model.seasonal_order}, AIC: {model.aic():.2f}')
    return model


def fit_exponential_smoothing(series, series_name=''):
    """Fit Exponential Smoothing (Holt method — no seasonal for short data)."""
    logger.info(f'  Fitting Exponential Smoothing for {series_name}...')
    # Use additive trend, no seasonal (only 19 data points)
    model = ExponentialSmoothing(
        series,
        trend='add',
        seasonal=None,
        damped_trend=True
    ).fit(optimized=True)
    logger.info(f'  {series_name} ETS: AIC={model.aic:.2f}, '
                f'alpha={model.params["smoothing_level"]:.4f}, '
                f'beta={model.params["smoothing_trend"]:.4f}')
    return model


def fit_sarimax_with_exog(series, exog, series_name='', order=None):
    """Fit SARIMAX with exogenous variables."""
    logger.info(f'  Fitting SARIMAX with exogenous vars for {series_name}...')

    if order is None:
        # Use auto_arima to find best order
        auto = pm.auto_arima(
            series, exogenous=exog,
            start_p=0, start_q=0, max_p=3, max_q=3,
            d=None, seasonal=False,
            trace=False, error_action='ignore',
            suppress_warnings=True, stepwise=True
        )
        order = auto.order
        logger.info(f'  {series_name} SARIMAX auto order: {order}')

    model = SARIMAX(
        series,
        exog=exog,
        order=order,
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)
    logger.info(f'  {series_name} SARIMAX: AIC={model.aic:.2f}')
    return model, order


def build_all_models(monthly):
    """Build models for all three targets."""
    log_separator('PHASE 4: MODEL BUILDING')

    all_models = {}
    all_cv_results = []

    # --- Exogenous variables selection ---
    exog_cols = ['pct_cancer', 'pct_overseas', 'pct_age_60plus',
                 'Avg_Kurs_SGD', 'Avg_Kurs_MYR']

    exog = monthly[exog_cols].copy()

    # =====================================================================
    # MODEL A: Claim Frequency
    # =====================================================================
    logger.info('')
    logger.info('--- Model A: Claim Frequency ---')
    freq_series = monthly['Claim_Frequency']

    # A1: Auto-ARIMA
    arima_freq = fit_auto_arima(freq_series, 'Frequency')
    all_models['arima_freq'] = arima_freq

    # A2: Exponential Smoothing
    ets_freq = fit_exponential_smoothing(freq_series, 'Frequency')
    all_models['ets_freq'] = ets_freq

    # A3: SARIMAX with exog
    sarimax_freq, sarimax_freq_order = fit_sarimax_with_exog(
        freq_series, exog, 'Frequency'
    )
    all_models['sarimax_freq'] = sarimax_freq
    all_models['sarimax_freq_order'] = sarimax_freq_order

    # CV for frequency models
    logger.info('  Cross-validating Frequency models...')

    def arima_builder_freq(train):
        m = pm.auto_arima(train, start_p=0, start_q=0, max_p=3, max_q=3,
                          seasonal=False, trace=False, error_action='ignore',
                          suppress_warnings=True, stepwise=True)
        pred = m.predict(n_periods=1)
        return pred.iloc[0] if hasattr(pred, 'iloc') else pred[0]

    def ets_builder_freq(train):
        m = ExponentialSmoothing(train, trend='add', seasonal=None,
                                damped_trend=True).fit(optimized=True)
        return m.forecast(1).values[0]

    cv_arima_freq, _, _ = expanding_window_cv(freq_series, arima_builder_freq,
                                               n_test=7, series_name='ARIMA_Freq')
    cv_ets_freq, _, _ = expanding_window_cv(freq_series, ets_builder_freq,
                                             n_test=7, series_name='ETS_Freq')

    if cv_arima_freq:
        all_cv_results.append(cv_arima_freq)
    if cv_ets_freq:
        all_cv_results.append(cv_ets_freq)

    # =====================================================================
    # MODEL B: Claim Severity (log-transformed)
    # =====================================================================
    logger.info('')
    logger.info('--- Model B: Claim Severity (log-transformed) ---')
    sev_series = monthly['log_Severity']

    # B1: Auto-ARIMA on log severity
    arima_sev = fit_auto_arima(sev_series, 'log_Severity')
    all_models['arima_sev'] = arima_sev

    # B2: Exponential Smoothing on log severity
    ets_sev = fit_exponential_smoothing(sev_series, 'log_Severity')
    all_models['ets_sev'] = ets_sev

    # B3: SARIMAX with exog on log severity
    sarimax_sev, sarimax_sev_order = fit_sarimax_with_exog(
        sev_series, exog, 'log_Severity'
    )
    all_models['sarimax_sev'] = sarimax_sev
    all_models['sarimax_sev_order'] = sarimax_sev_order

    # CV for severity models
    logger.info('  Cross-validating Severity models...')

    def arima_builder_sev(train):
        m = pm.auto_arima(train, start_p=0, start_q=0, max_p=3, max_q=3,
                          seasonal=False, trace=False, error_action='ignore',
                          suppress_warnings=True, stepwise=True)
        pred = m.predict(n_periods=1)
        val = pred.iloc[0] if hasattr(pred, 'iloc') else pred[0]
        return np.expm1(val)

    def ets_builder_sev(train):
        m = ExponentialSmoothing(train, trend='add', seasonal=None,
                                damped_trend=True).fit(optimized=True)
        return np.expm1(m.forecast(1).values[0])

    # CV on original scale
    sev_original = monthly['Claim_Severity']

    cv_arima_sev, _, _ = expanding_window_cv(sev_original,
        lambda train: arima_builder_sev(np.log1p(train)),
        n_test=7, series_name='ARIMA_Sev')
    cv_ets_sev, _, _ = expanding_window_cv(sev_original,
        lambda train: ets_builder_sev(np.log1p(train)),
        n_test=7, series_name='ETS_Sev')

    if cv_arima_sev:
        all_cv_results.append(cv_arima_sev)
    if cv_ets_sev:
        all_cv_results.append(cv_ets_sev)

    # =====================================================================
    # MODEL C: Total Claim (log-transformed) — Direct forecast
    # =====================================================================
    logger.info('')
    logger.info('--- Model C: Total Claim (log-transformed, direct) ---')
    total_series = monthly['log_Total']

    # C1: Auto-ARIMA on log total
    arima_total = fit_auto_arima(total_series, 'log_Total')
    all_models['arima_total'] = arima_total

    # C2: Exponential Smoothing on log total
    ets_total = fit_exponential_smoothing(total_series, 'log_Total')
    all_models['ets_total'] = ets_total

    # C3: SARIMAX with exog on log total
    sarimax_total, sarimax_total_order = fit_sarimax_with_exog(
        total_series, exog, 'log_Total'
    )
    all_models['sarimax_total'] = sarimax_total
    all_models['sarimax_total_order'] = sarimax_total_order

    # CV for total models
    logger.info('  Cross-validating Total Claim models...')
    total_original = monthly['Total_Claim']

    def arima_builder_total(train):
        m = pm.auto_arima(np.log1p(train), start_p=0, start_q=0, max_p=3, max_q=3,
                          seasonal=False, trace=False, error_action='ignore',
                          suppress_warnings=True, stepwise=True)
        pred = m.predict(n_periods=1)
        val = pred.iloc[0] if hasattr(pred, 'iloc') else pred[0]
        return np.expm1(val)

    def ets_builder_total(train):
        m = ExponentialSmoothing(np.log1p(train), trend='add', seasonal=None,
                                damped_trend=True).fit(optimized=True)
        return np.expm1(m.forecast(1).values[0])

    cv_arima_total, _, _ = expanding_window_cv(total_original, arima_builder_total,
                                                n_test=7, series_name='ARIMA_Total')
    cv_ets_total, _, _ = expanding_window_cv(total_original, ets_builder_total,
                                              n_test=7, series_name='ETS_Total')

    if cv_arima_total:
        all_cv_results.append(cv_arima_total)
    if cv_ets_total:
        all_cv_results.append(cv_ets_total)

    # --- Log CV summary ---
    logger.info('')
    logger.info('--- Cross-Validation Summary ---')
    cv_df = pd.DataFrame(all_cv_results)
    logger.info(f'\n{cv_df.to_string(index=False)}')

    return all_models, exog, all_cv_results


# ============================================================================
# %% [markdown]
# ## PHASE 5: GENERATE FORECASTS
# ============================================================================
# %%
def generate_forecasts(monthly, all_models, exog):
    """Generate 5-month forecasts for all targets."""
    log_separator('PHASE 5: GENERATING FORECASTS')

    # --- Prepare future exogenous variables ---
    # Use last 3 months average as proxy for future
    exog_cols = exog.columns.tolist()
    future_exog_values = exog.tail(3).mean()
    future_exog = pd.DataFrame(
        [future_exog_values] * FORECAST_HORIZON,
        columns=exog_cols,
        index=FORECAST_MONTHS
    )

    # For currency rates: use the latest available rates data for Aug-Dec 2025
    # The exchange rate data goes up to Jan 2026, so we have actual data
    logger.info('Future exogenous variable assumptions (3-month trailing average):')
    for col in exog_cols:
        logger.info(f'  {col}: {future_exog_values[col]:.4f}')

    results = {}

    # --- A: Claim Frequency forecasts ---
    logger.info('')
    logger.info('--- Frequency Forecasts ---')

    freq_arima = all_models['arima_freq'].predict(n_periods=FORECAST_HORIZON)
    if hasattr(freq_arima, 'values'): freq_arima = freq_arima.values
    freq_ets = all_models['ets_freq'].forecast(FORECAST_HORIZON).values
    freq_sarimax = all_models['sarimax_freq'].forecast(
        steps=FORECAST_HORIZON, exog=future_exog[exog_cols]
    ).values

    # Ensure non-negative and round (counts)
    freq_arima = np.maximum(freq_arima, 0).round().astype(int)
    freq_ets = np.maximum(freq_ets, 0).round().astype(int)
    freq_sarimax = np.maximum(freq_sarimax, 0).round().astype(int)

    # Ensemble: weighted average
    freq_ensemble = np.round(
        0.40 * freq_arima + 0.30 * freq_ets + 0.30 * freq_sarimax
    ).astype(int)

    for i, month in enumerate(FORECAST_MONTHS):
        logger.info(f'  {month.strftime("%Y-%m")}: '
                     f'ARIMA={freq_arima[i]}, ETS={freq_ets[i]}, '
                     f'SARIMAX={freq_sarimax[i]}, Ensemble={freq_ensemble[i]}')

    results['freq_arima'] = freq_arima
    results['freq_ets'] = freq_ets
    results['freq_sarimax'] = freq_sarimax
    results['freq_ensemble'] = freq_ensemble

    # --- B: Claim Severity forecasts ---
    logger.info('')
    logger.info('--- Severity Forecasts (back-transformed from log) ---')

    sev_arima_log = all_models['arima_sev'].predict(n_periods=FORECAST_HORIZON)
    if hasattr(sev_arima_log, 'values'): sev_arima_log = sev_arima_log.values
    sev_ets_log = all_models['ets_sev'].forecast(FORECAST_HORIZON).values
    sev_sarimax_log = all_models['sarimax_sev'].forecast(
        steps=FORECAST_HORIZON, exog=future_exog[exog_cols]
    ).values

    sev_arima = np.expm1(sev_arima_log)
    sev_ets = np.expm1(sev_ets_log)
    sev_sarimax = np.expm1(sev_sarimax_log)

    # Ensure non-negative
    sev_arima = np.maximum(sev_arima, 0)
    sev_ets = np.maximum(sev_ets, 0)
    sev_sarimax = np.maximum(sev_sarimax, 0)

    sev_ensemble = 0.40 * sev_arima + 0.30 * sev_ets + 0.30 * sev_sarimax

    for i, month in enumerate(FORECAST_MONTHS):
        logger.info(f'  {month.strftime("%Y-%m")}: '
                     f'ARIMA=Rp {sev_arima[i]:,.0f}, ETS=Rp {sev_ets[i]:,.0f}, '
                     f'SARIMAX=Rp {sev_sarimax[i]:,.0f}, Ensemble=Rp {sev_ensemble[i]:,.0f}')

    results['sev_arima'] = sev_arima
    results['sev_ets'] = sev_ets
    results['sev_sarimax'] = sev_sarimax
    results['sev_ensemble'] = sev_ensemble

    # --- C: Total Claim forecasts ---
    logger.info('')
    logger.info('--- Total Claim Forecasts ---')

    # C1: Direct forecast
    total_arima_log = all_models['arima_total'].predict(n_periods=FORECAST_HORIZON)
    if hasattr(total_arima_log, 'values'): total_arima_log = total_arima_log.values
    total_ets_log = all_models['ets_total'].forecast(FORECAST_HORIZON).values
    total_sarimax_log = all_models['sarimax_total'].forecast(
        steps=FORECAST_HORIZON, exog=future_exog[exog_cols]
    ).values

    total_arima = np.expm1(total_arima_log)
    total_ets = np.expm1(total_ets_log)
    total_sarimax = np.expm1(total_sarimax_log)

    total_arima = np.maximum(total_arima, 0)
    total_ets = np.maximum(total_ets, 0)
    total_sarimax = np.maximum(total_sarimax, 0)

    # C2: Derived forecast (Freq × Severity)
    total_derived = freq_ensemble * sev_ensemble

    # C3: Final ensemble: weighted average of direct and derived
    alpha = 0.6  # Weight for direct model
    total_direct_ensemble = 0.40 * total_arima + 0.30 * total_ets + 0.30 * total_sarimax
    total_final = alpha * total_direct_ensemble + (1 - alpha) * total_derived

    for i, month in enumerate(FORECAST_MONTHS):
        logger.info(f'  {month.strftime("%Y-%m")}: '
                     f'Direct=Rp {total_direct_ensemble[i]:,.0f}, '
                     f'Derived=Rp {total_derived[i]:,.0f}, '
                     f'Final=Rp {total_final[i]:,.0f}')

    results['total_arima'] = total_arima
    results['total_ets'] = total_ets
    results['total_sarimax'] = total_sarimax
    results['total_direct_ensemble'] = total_direct_ensemble
    results['total_derived'] = total_derived
    results['total_final'] = total_final

    # --- Confidence intervals from ARIMA ---
    logger.info('')
    logger.info('--- Confidence Intervals (from ARIMA models) ---')

    freq_ci = all_models['arima_freq'].predict(
        n_periods=FORECAST_HORIZON, return_conf_int=True, alpha=0.20
    )
    sev_ci = all_models['arima_sev'].predict(
        n_periods=FORECAST_HORIZON, return_conf_int=True, alpha=0.20
    )
    total_ci = all_models['arima_total'].predict(
        n_periods=FORECAST_HORIZON, return_conf_int=True, alpha=0.20
    )

    results['freq_ci_80'] = freq_ci[1]
    results['sev_ci_80'] = np.expm1(sev_ci[1])
    results['total_ci_80'] = np.expm1(total_ci[1])

    freq_ci_95 = all_models['arima_freq'].predict(
        n_periods=FORECAST_HORIZON, return_conf_int=True, alpha=0.05
    )
    sev_ci_95 = all_models['arima_sev'].predict(
        n_periods=FORECAST_HORIZON, return_conf_int=True, alpha=0.05
    )
    total_ci_95 = all_models['arima_total'].predict(
        n_periods=FORECAST_HORIZON, return_conf_int=True, alpha=0.05
    )

    results['freq_ci_95'] = freq_ci_95[1]
    results['sev_ci_95'] = np.expm1(sev_ci_95[1])
    results['total_ci_95'] = np.expm1(total_ci_95[1])

    for i, month in enumerate(FORECAST_MONTHS):
        logger.info(
            f'  {month.strftime("%Y-%m")} CI80: '
            f'Freq=[{results["freq_ci_80"][i, 0]:.0f}, {results["freq_ci_80"][i, 1]:.0f}], '
            f'Sev=[Rp {results["sev_ci_80"][i, 0]:,.0f}, Rp {results["sev_ci_80"][i, 1]:,.0f}], '
            f'Total=[Rp {results["total_ci_80"][i, 0]:,.0f}, Rp {results["total_ci_80"][i, 1]:,.0f}]'
        )

    return results


# ============================================================================
# %% [markdown]
# ## PHASE 6: FACTOR ANALYSIS (XGBoost + SHAP)
# ============================================================================
# %%
def run_factor_analysis(df):
    """XGBoost feature importance on per-claim data + SHAP analysis."""
    log_separator('PHASE 6: FACTOR ANALYSIS (XGBoost + SHAP)')

    # --- Prepare per-claim features ---
    df_model = df.copy()

    # Exclude zero-claims
    df_model = df_model[df_model['Nominal Klaim Yang Disetujui'] > 0].copy()

    # Encode categoricals
    label_encoders = {}
    cat_cols = {
        'Gender': 'Gender_enc',
        'Plan Code': 'Plan_enc',
        'Lokasi RS': 'Lokasi_enc',
        'ICD_Group': 'ICD_Group_enc',
        'Inpatient/Outpatient': 'IP_OP_enc',
        'Reimburse/Cashless': 'RC_enc',
    }
    for col, enc_col in cat_cols.items():
        le = LabelEncoder()
        df_model[enc_col] = le.fit_transform(df_model[col].astype(str))
        label_encoders[col] = le

    # Extract month and quarter numbers
    df_model['month_num'] = df_model['Bulan_dt'].dt.month
    df_model['quarter_num'] = df_model['Bulan_dt'].dt.quarter

    feature_cols = ['Usia', 'Gender_enc', 'Plan_enc', 'Lokasi_enc',
                    'ICD_Group_enc', 'IP_OP_enc', 'LOS', 'RC_enc',
                    'month_num', 'quarter_num']

    X = df_model[feature_cols].copy()
    y = np.log1p(df_model['Nominal Klaim Yang Disetujui'])

    # Original feature names for display
    display_names = ['Usia', 'Gender', 'Plan Code', 'Lokasi RS',
                     'ICD Group', 'Inpatient/Outpatient', 'LOS',
                     'Reimburse/Cashless', 'Month', 'Quarter']

    # Train-test split (time-based: last 20% as test)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    logger.info(f'XGBoost training set: {len(X_train)} samples')
    logger.info(f'XGBoost test set: {len(X_test)} samples')

    # --- Fit XGBoost ---
    model_xgb = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )
    model_xgb.fit(X_train, y_train)

    # Evaluate
    y_pred_train = model_xgb.predict(X_train)
    y_pred_test = model_xgb.predict(X_test)

    train_metrics = evaluate_forecast(np.expm1(y_train), np.expm1(y_pred_train),
                                       'XGBoost Train')
    test_metrics = evaluate_forecast(np.expm1(y_test), np.expm1(y_pred_test),
                                      'XGBoost Test')

    from sklearn.metrics import r2_score
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    logger.info(f'  R² Train: {r2_train:.4f}')
    logger.info(f'  R² Test: {r2_test:.4f}')

    # --- SHAP Analysis ---
    logger.info('  Computing SHAP values...')
    explainer = shap.TreeExplainer(model_xgb)
    shap_values = explainer.shap_values(X_test)

    # Mean absolute SHAP values (feature importance)
    shap_importance = pd.DataFrame({
        'Feature': display_names,
        'Mean_Abs_SHAP': np.abs(shap_values).mean(axis=0)
    }).sort_values('Mean_Abs_SHAP', ascending=False)

    logger.info('')
    logger.info('--- Feature Importance (SHAP) ---')
    for _, row in shap_importance.iterrows():
        bar = '█' * int(row['Mean_Abs_SHAP'] / shap_importance['Mean_Abs_SHAP'].max() * 20)
        logger.info(f'  {row["Feature"]:25s} {bar:20s} {row["Mean_Abs_SHAP"]:.4f}')

    # --- Plot SHAP summary ---
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, feature_names=display_names, show=False)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / 'shap_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f'Saved SHAP summary plot to {CHARTS_DIR / "shap_summary.png"}')

    # --- Plot SHAP bar chart ---
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.bar(shap.Explanation(
        values=shap_values,
        feature_names=display_names,
        data=X_test.values
    ), show=False)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / 'shap_bar.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f'Saved SHAP bar plot to {CHARTS_DIR / "shap_bar.png"}')

    # --- Native XGBoost feature importance ---
    fig, ax = plt.subplots(figsize=(10, 6))
    xgb_imp = pd.DataFrame({
        'Feature': display_names,
        'Importance': model_xgb.feature_importances_
    }).sort_values('Importance', ascending=True)
    ax.barh(xgb_imp['Feature'], xgb_imp['Importance'], color='steelblue')
    ax.set_xlabel('Feature Importance (Gain)')
    ax.set_title('XGBoost Feature Importance — Claim Amount Prediction')
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / 'xgboost_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()

    return model_xgb, shap_importance, label_encoders


# ============================================================================
# %% [markdown]
# ## PHASE 7: VISUALIZATION
# ============================================================================
# %%
def plot_forecasts(monthly, results):
    """Plot actual vs forecast for all three targets."""
    log_separator('PHASE 7: VISUALIZATION')

    fig, axes = plt.subplots(3, 1, figsize=(14, 16))

    targets = [
        ('Claim_Frequency', 'freq_ensemble', 'freq_ci_80', 'freq_ci_95',
         'Claim Frequency (Count)', False),
        ('Claim_Severity', 'sev_ensemble', 'sev_ci_80', 'sev_ci_95',
         'Claim Severity (Rp)', True),
        ('Total_Claim', 'total_final', 'total_ci_80', 'total_ci_95',
         'Total Claim (Rp)', True),
    ]

    for ax, (col, fc_key, ci80_key, ci95_key, title, fmt_rp) in zip(axes, targets):
        # Actual
        ax.plot(monthly.index, monthly[col], 'b-o', label='Actual', linewidth=2, markersize=5)

        # Forecast
        ax.plot(FORECAST_MONTHS, results[fc_key], 'r--s', label='Forecast (Ensemble)',
                linewidth=2, markersize=6)

        # Fill CI
        ci80 = results[ci80_key]
        ci95 = results[ci95_key]
        ax.fill_between(FORECAST_MONTHS, ci80[:, 0], ci80[:, 1],
                        alpha=0.3, color='red', label='80% CI')
        ax.fill_between(FORECAST_MONTHS, ci95[:, 0], ci95[:, 1],
                        alpha=0.15, color='red', label='95% CI')

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        if fmt_rp:
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, _: f'Rp {x/1e9:.1f}B' if x >= 1e9
                                      else f'Rp {x/1e6:.0f}M')
            )

    plt.tight_layout()
    plt.savefig(CHARTS_DIR / 'forecast_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f'Saved forecast plot to {CHARTS_DIR / "forecast_results.png"}')

    # --- Plot individual model comparison ---
    fig, axes = plt.subplots(3, 1, figsize=(14, 16))

    model_names = ['ARIMA', 'ETS', 'SARIMAX']
    colors = ['#e74c3c', '#2ecc71', '#9b59b6']

    for ax, (col, fc_keys, title, fmt_rp) in zip(axes, [
        ('Claim_Frequency',
         ['freq_arima', 'freq_ets', 'freq_sarimax'],
         'Claim Frequency — Model Comparison', False),
        ('Claim_Severity',
         ['sev_arima', 'sev_ets', 'sev_sarimax'],
         'Claim Severity — Model Comparison', True),
        ('Total_Claim',
         ['total_arima', 'total_ets', 'total_sarimax'],
         'Total Claim — Model Comparison', True),
    ]):
        ax.plot(monthly.index, monthly[col], 'b-o', label='Actual', linewidth=2, markersize=5)
        for fc_key, name, color in zip(fc_keys, model_names, colors):
            ax.plot(FORECAST_MONTHS, results[fc_key], '--', color=color,
                    label=f'{name} Forecast', linewidth=1.5, marker='s', markersize=5)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        if fmt_rp:
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, _: f'Rp {x/1e9:.1f}B' if x >= 1e9
                                      else f'Rp {x/1e6:.0f}M')
            )

    plt.tight_layout()
    plt.savefig(CHARTS_DIR / 'model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f'Saved model comparison plot to {CHARTS_DIR / "model_comparison.png"}')


# ============================================================================
# %% [markdown]
# ## PHASE 8: SAVE MODELS, SUBMISSION & LOGGING
# ============================================================================
# %%
def save_outputs(all_models, results, model_xgb, label_encoders, monthly):
    """Save models, submission CSV, and final log."""
    log_separator('PHASE 8: SAVING OUTPUTS')

    # --- Save models ---
    model_save_items = {
        'arima_freq': all_models['arima_freq'],
        'arima_sev': all_models['arima_sev'],
        'arima_total': all_models['arima_total'],
        'ets_freq': all_models['ets_freq'],
        'ets_sev': all_models['ets_sev'],
        'ets_total': all_models['ets_total'],
        'xgboost_claims': model_xgb,
    }

    for name, model in model_save_items.items():
        path = MODELS_DIR / f'{name}.joblib'
        joblib.dump(model, path)
        logger.info(f'Saved model: {path}')

    # Save SARIMAX results (these are results objects, not fit objects)
    for name in ['sarimax_freq', 'sarimax_sev', 'sarimax_total']:
        if name in all_models:
            path = MODELS_DIR / f'{name}.joblib'
            joblib.dump(all_models[name], path)
            logger.info(f'Saved model: {path}')

    # Save label encoders
    joblib.dump(label_encoders, MODELS_DIR / 'label_encoders.joblib')
    logger.info(f'Saved label encoders to {MODELS_DIR / "label_encoders.joblib"}')

    # --- Build submission CSV ---
    logger.info('')
    logger.info('--- Building Submission CSV ---')

    submission_rows = []
    month_labels = ['2025_08', '2025_09', '2025_10', '2025_11', '2025_12']

    for i, ml in enumerate(month_labels):
        submission_rows.append({
            'id': f'{ml}_Claim_Frequency',
            'value': float(results['freq_ensemble'][i])
        })
        submission_rows.append({
            'id': f'{ml}_Claim_Severity',
            'value': float(results['sev_ensemble'][i])
        })
        submission_rows.append({
            'id': f'{ml}_Total_Claim',
            'value': float(results['total_final'][i])
        })

    submission_df = pd.DataFrame(submission_rows)
    submission_df.to_csv(SUBMISSION_FILE, index=False)
    logger.info(f'Saved submission to {SUBMISSION_FILE}')
    logger.info(f'Submission shape: {submission_df.shape}')

    # Print submission table
    logger.info('')
    logger.info('=' * 60)
    logger.info('  FINAL SUBMISSION — 15 PREDICTED VALUES')
    logger.info('=' * 60)
    for _, row in submission_df.iterrows():
        val = row['value']
        # Log raw value as requested
        logger.info(f'  {row["id"]:35s} │ {val}')
    logger.info('=' * 60)

    # --- Save detailed results JSON ---
    detail = {
        'forecast_months': [m.strftime('%Y-%m') for m in FORECAST_MONTHS],
        'frequency': {
            'arima': results['freq_arima'].tolist(),
            'ets': results['freq_ets'].tolist(),
            'sarimax': results['freq_sarimax'].tolist(),
            'ensemble': results['freq_ensemble'].tolist(),
        },
        'severity': {
            'arima': results['sev_arima'].tolist(),
            'ets': results['sev_ets'].tolist(),
            'sarimax': results['sev_sarimax'].tolist(),
            'ensemble': results['sev_ensemble'].tolist(),
        },
        'total_claim': {
            'arima': results['total_arima'].tolist(),
            'ets': results['total_ets'].tolist(),
            'sarimax': results['total_sarimax'].tolist(),
            'direct_ensemble': results['total_direct_ensemble'].tolist(),
            'derived': results['total_derived'].tolist(),
            'final': results['total_final'].tolist(),
        },
        'confidence_intervals': {
            'freq_ci_80': results['freq_ci_80'].tolist(),
            'freq_ci_95': results['freq_ci_95'].tolist(),
            'sev_ci_80': results['sev_ci_80'].tolist(),
            'sev_ci_95': results['sev_ci_95'].tolist(),
            'total_ci_80': results['total_ci_80'].tolist(),
            'total_ci_95': results['total_ci_95'].tolist(),
        }
    }

    with open(BASE_DIR / 'forecast_details.json', 'w') as f:
        json.dump(detail, f, indent=2)
    logger.info(f'Saved detailed results to {BASE_DIR / "forecast_details.json"}')

    # --- Save monthly time series for reference ---
    monthly.to_csv(BASE_DIR / 'monthly_timeseries.csv')
    logger.info(f'Saved monthly time series to {BASE_DIR / "monthly_timeseries.csv"}')

    return submission_df


# ============================================================================
# %% [markdown]
# ## MAIN PIPELINE
# ============================================================================
# %%
def main():
    start_time = datetime.now()
    logger.info('=' * 70)
    logger.info('  ML CLAIMS FORECASTING PIPELINE')
    logger.info(f'  Started: {start_time.strftime("%Y-%m-%d %H:%M:%S")}')
    logger.info('=' * 70)

    # Phase 1: Load & Clean
    df = load_and_clean_data()

    # Phase 1b: Exchange Rates
    kurs_myr, kurs_sgd = load_exchange_rates()

    # Phase 2: Monthly Aggregation
    monthly = build_monthly_timeseries(df, kurs_myr, kurs_sgd)

    # Phase 3: Stationarity Analysis
    adf_results = analyze_stationarity(monthly)

    # Phase 4: Build Models
    all_models, exog, cv_results = build_all_models(monthly)

    # Phase 5: Generate Forecasts
    results = generate_forecasts(monthly, all_models, exog)

    # Phase 6: Factor Analysis
    model_xgb, shap_imp, label_encoders = run_factor_analysis(df)

    # Phase 7: Visualization
    plot_forecasts(monthly, results)

    # Phase 8: Save Everything
    submission_df = save_outputs(all_models, results, model_xgb, label_encoders, monthly)

    # --- Final Summary ---
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info('')
    log_separator('PIPELINE COMPLETE')
    logger.info(f'  Total time: {elapsed:.1f} seconds')
    logger.info(f'  Log file: {LOG_FILE}')
    logger.info(f'  Submission: {SUBMISSION_FILE}')
    logger.info(f'  Models saved: {MODELS_DIR}')
    logger.info(f'  Charts saved: {CHARTS_DIR}')
    logger.info(f'  Detailed results: {BASE_DIR / "forecast_details.json"}')
    logger.info(f'  Monthly series: {BASE_DIR / "monthly_timeseries.csv"}')
    logger.info('')
    logger.info('  Predicted values (Ensemble):')
    for _, row in submission_df.iterrows():
        val = row['Predicted_Value']
        if 'Frequency' in row['Variable']:
            logger.info(f'    {row["Variable"]:35s} = {int(val):,}')
        else:
            logger.info(f'    {row["Variable"]:35s} = Rp {val:,.0f}')


if __name__ == '__main__':
    main()


if __name__ == '__main__':
    main()
