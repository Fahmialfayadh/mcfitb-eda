# Machine Learning Plan — Prediksi Klaim Asuransi Kesehatan Individu

---

## 1. Case Overview

### 1.1 Latar Belakang Masalah

Salah satu isu yang terjadi beberapa tahun terakhir ini adalah terkait **peningkatan klaim asuransi kesehatan individu**, dimana terjadi **kenaikan sebesar 25,5%** antara periode Januari – Juni 2025 jika dibandingkan periode yang sama di tahun 2024.

Peningkatan klaim tersebut akan berdampak ke **penyesuaian premi** yang berpotensi mengakibatkan harga premi produk asuransi kesehatan individu menjadi **lebih tidak terjangkau**.

### 1.2 Tujuan

Diperlukan **analisa berbasis data** untuk:
1. **Memprediksi** faktor-faktor yang paling berpengaruh terhadap nilai klaim
2. Melakukan inisiatif dari segi **seleksi risiko** (underwriting)
3. Melakukan **pencegahan** (preventive care programs)
4. Melakukan **deteksi dini** untuk meminimalisir dampak peningkatan klaim
5. Menjaga **harga premi yang terjangkau**

### 1.3 Deliverable

**Prediksi agregat bulanan** untuk periode **Agustus — Desember 2025** (5 bulan ke depan), masing-masing dengan 3 metrik:

| # | Output Variable | Deskripsi |
|---|---|---|
| 1 | `2025_08_Claim_Frequency` | Prediksi jumlah klaim bulan Agustus 2025 |
| 2 | `2025_08_Claim_Severity` | Prediksi rata-rata nominal per klaim bulan Agustus 2025 |
| 3 | `2025_08_Total_Claim` | Prediksi total nominal klaim bulan Agustus 2025 |
| 4 | `2025_09_Claim_Frequency` | Prediksi jumlah klaim bulan September 2025 |
| 5 | `2025_09_Claim_Severity` | Prediksi rata-rata nominal per klaim bulan September 2025 |
| 6 | `2025_09_Total_Claim` | Prediksi total nominal klaim bulan September 2025 |
| 7 | `2025_10_Claim_Frequency` | Prediksi jumlah klaim bulan Oktober 2025 |
| 8 | `2025_10_Claim_Severity` | Prediksi rata-rata nominal per klaim bulan Oktober 2025 |
| 9 | `2025_10_Total_Claim` | Prediksi total nominal klaim bulan Oktober 2025 |
| 10 | `2025_11_Claim_Frequency` | Prediksi jumlah klaim bulan November 2025 |
| 11 | `2025_11_Claim_Severity` | Prediksi rata-rata nominal per klaim bulan November 2025 |
| 12 | `2025_11_Total_Claim` | Prediksi total nominal klaim bulan November 2025 |
| 13 | `2025_12_Claim_Frequency` | Prediksi jumlah klaim bulan Desember 2025 |
| 14 | `2025_12_Claim_Severity` | Prediksi rata-rata nominal per klaim bulan Desember 2025 |
| 15 | `2025_12_Total_Claim` | Prediksi total nominal klaim bulan Desember 2025 |

**Total: 15 unique values, 15 total values**

---

## 2. Dataset Overview

### 2.1 Sumber Data

| Item | Detail |
|---|---|
| **File** | `dataset/Data_Klaim_Enriched.csv` |
| **Sumber Asli** | Merge dari `Data_Klaim.csv` (4.627 records) + `Data_Polis.csv` (4.096 records) |
| **Dimensi** | 4.625 rows × 28 columns |
| **Periode Data** | Januari 2024 — Juli 2025 (19 bulan) |
| **Granularity** | Per-transaksi klaim |

### 2.1.1 Data External: Kurs Valuta Asing

Tersedia data kurs harian yang akan digunakan sebagai **exogenous variable** untuk menangkap efek fluktuasi mata uang terhadap biaya klaim luar negeri (~33% total klaim).

| File | Mata Uang | Periode | Records | Kolom |
|---|---|---|---|---|
| `dataset/Data Historis MYR_IDR.csv` | Ringgit Malaysia → Rupiah | Jan 2024 — Jan 2026 | 514 baris (harian) | Tanggal, Terakhir, Pembukaan, Tertinggi, Terendah, Vol., Perubahan% |
| `dataset/Data Historis SGD_IDR.csv` | Dollar Singapura → Rupiah | Jan 2024 — Feb 2026 | 514 baris (harian) | Tanggal, Terakhir, Pembukaan, Tertinggi, Terendah, Vol., Perubahan% |

**Format data**: Angka menggunakan format Indonesia (titik = ribuan, koma = desimal). Contoh: `13.332,82` = Rp 13.332,82 per 1 SGD.


#### Cara Penggunaan Kurs dalam Model

**Step 1: Preprocessing Kurs**

```python
import pandas as pd

# Load & parse kurs MYR
kurs_myr = pd.read_csv('dataset/Data Historis MYR_IDR.csv')
# Konversi format Indonesia ke float
kurs_myr['Kurs_MYR'] = kurs_myr['Terakhir'].str.replace('.', '', regex=False).str.replace(',', '.', regex=False).astype(float)
kurs_myr['Tanggal'] = pd.to_datetime(kurs_myr['Tanggal'], format='%d/%m/%Y')

# Load & parse kurs SGD (perlu dilengkapi data historis!)
kurs_sgd = pd.read_csv('dataset/Data Historis SGD_IDR.csv')
kurs_sgd['Kurs_SGD'] = kurs_sgd['Terakhir'].str.replace('.', '', regex=False).str.replace(',', '.', regex=False).astype(float)
kurs_sgd['Tanggal'] = pd.to_datetime(kurs_sgd['Tanggal'], format='%d/%m/%Y')

# Agregasi ke bulanan (rata-rata kurs per bulan)
kurs_myr_monthly = kurs_myr.set_index('Tanggal').resample('MS')['Kurs_MYR'].mean().reset_index()
kurs_myr_monthly.columns = ['Bulan', 'Avg_Kurs_MYR']

kurs_sgd_monthly = kurs_sgd.set_index('Tanggal').resample('MS')['Kurs_SGD'].mean().reset_index()
kurs_sgd_monthly.columns = ['Bulan', 'Avg_Kurs_SGD']
```

**Step 2: Merge dengan Data Klaim Bulanan**

```python
# Merge kurs ke time series bulanan
monthly = monthly.merge(kurs_myr_monthly, on='Bulan', how='left')
monthly = monthly.merge(kurs_sgd_monthly, on='Bulan', how='left')

# Hitung perubahan kurs bulan-ke-bulan (%)
monthly['MYR_change_pct'] = monthly['Avg_Kurs_MYR'].pct_change() * 100
monthly['SGD_change_pct'] = monthly['Avg_Kurs_SGD'].pct_change() * 100
```

**Step 3: Gunakan sebagai Exogenous Variable di SARIMAX**

```python
exog_vars = ['pct_cancer', 'pct_overseas', 'pct_age_60plus',
             'Avg_Kurs_SGD', 'Avg_Kurs_MYR']  # Tambahkan kurs!

model_sev_x = SARIMAX(
    monthly['log_Severity'],
    exog=monthly[exog_vars],
    order=(p, d, q)
)

# Untuk forecast, perlu asumsi kurs ke depan:
# Opsi 1: Gunakan kurs terakhir (flat assumption)
# Opsi 2: Gunakan trend linear kurs
# Opsi 3: Gunakan forward rate dari pasar
```

#### Rasionalisasi Penggunaan Kurs

```
Depresiasi Rupiah vs SGD/MYR
    ↓
Biaya RS luar negeri (dalam Rupiah) naik
    ↓
Claim Severity naik untuk klaim overseas (33% total)
    ↓
Total Claim portfolio naik
```

| Skenario Kurs | Dampak ke Severity | Contoh |
|---|---|---|
| Rupiah melemah 5% vs SGD | Severity klaim Singapore naik ~5% | SGD 12.000 → 12.600 per SGD |
| Rupiah menguat 3% vs MYR | Severity klaim Malaysia turun ~3% | MYR 3.500 → 3.395 per MYR |
| Stabil | Tidak ada efek kurs | Baseline assumption |

**Expected impact**: Karena ~25% klaim di Singapore dan ~8% di Malaysia, efek kurs SGD lebih dominan. Perubahan 1% kurs SGD ≈ perubahan ~0,25% pada total severity portfolio.

### 2.2 Kolom Dataset

#### Kolom Asli (dari Data Klaim)
| Kolom | Tipe | Deskripsi | Missing |
|---|---|---|---|
| `Claim ID` | String | ID unik klaim | 0 |
| `Nomor Polis` | String | ID polis asuransi | 0 |
| `Reimburse/Cashless` | Categorical | R = Reimburse, C = Cashless | 0 |
| `Inpatient/Outpatient` | Categorical | IP, OP, ODC, ODS | 35 |
| `ICD Diagnosis` | String | Kode ICD-10 (753 unique) | 4 |
| `ICD Description` | String | Deskripsi diagnosis (946 unique) | 4 |
| `Tanggal Pembayaran Klaim` | Date | Tanggal pembayaran | 35 |
| `Tanggal Pasien Masuk RS` | Date | Tanggal masuk RS | 0 |
| `Tanggal Pasien Keluar RS` | Date | Tanggal keluar RS | 0 |
| `Nominal Klaim Yang Disetujui` | Float | Nominal yang disetujui (Rp) | 0 |
| `Nominal Biaya RS Yang Terjadi` | Float | Biaya RS aktual (Rp) | 0 |
| `Lokasi RS` | Categorical | 10 lokasi (Indonesia, Singapore, dll) | 7 |

#### Kolom dari Data Polis (hasil merge)
| Kolom | Tipe | Deskripsi | Missing |
|---|---|---|---|
| `Plan Code` | Categorical | M-001, M-002, M-003 | 0 |
| `Gender` | Categorical | M, F | 0 |
| `Tanggal Lahir` | Date | Tanggal lahir pemegang polis | 0 |
| `Tanggal Efektif Polis` | Date | Tanggal polis efektif | 0 |
| `Domisili` | Categorical | 20 kota di Indonesia | 0 |

#### Kolom Feature Engineering
| Kolom | Tipe | Formula | Missing |
|---|---|---|---|
| `Usia` | Integer | (Tgl Masuk RS − Tgl Lahir) / 365.25 | 0 |
| `LOS` | Integer | Tgl Keluar RS − Tgl Masuk RS (hari) | 0 |
| `Processing_Days` | Float | Tgl Bayar − Tgl Keluar (hari) | 35 |
| `Selisih_Klaim` | Float | Nominal Disetujui − Biaya RS | 0 |
| `Rasio_Coverage` | Float | Nominal Disetujui / Biaya RS | 2 |
| `Bulan` | String | Period YYYY-MM | 0 |
| `Kuartal` | String | Period YYYY-Q# | 0 |
| `DayOfWeek` | String | Nama hari (Monday-Sunday) | 0 |
| `Kategori_Nominal` | Categorical | <5Jt, 5-25Jt, 25-100Jt, >100Jt | 0 |
| `ICD_Letter` | String | Huruf pertama kode ICD | 4 |
| `ICD_Group` | Categorical | 18 grup penyakit (mapping ICD) | 0 |

### 2.3 Statistik Deskriptif Variabel Kunci

| Variabel | Mean | Median | Std Dev | Min | Max |
|---|---|---|---|---|---|
| Nominal Disetujui (Rp) | 55.044.929 | 14.467.899 | 131.978.462 | 0 | 2.197.500.000 |
| Biaya RS (Rp) | 59.966.430 | 15.871.000 | 159.815.878 | 0 | 3.892.809.996 |
| Usia (tahun) | 58,8 | 59 | 12,6 | 6 | 90 |
| LOS (hari) | 1,26 | 0 | 2,93 | 0 | 54 |
| Processing_Days | 65,6 | 61 | 33,4 | 8 | 606 |

---

## 3. Key Insights dari EDA (Relevan untuk ML)

### 3.1 Fakta Kunci Portfolio

| Metrik | Nilai | Implikasi untuk ML |
|---|---|---|
| Total Klaim | 4.625 transaksi | Ukuran dataset cukup untuk modeling |
| Polis Aktif | 1.210 dari 4.096 (29,5%) | 70,5% polis tidak klaim — zero-inflated |
| Total Nominal | Rp 254,6 Miliar | Target prediksi bulanan |
| Reimburse vs Cashless | 59% vs 41% | Feature penting |
| Indonesia vs LN | 67% vs 33% | Location multiplier effect |

### 3.2 Cost Driver Utama (dari EDA)

**Ranking fitur berdasarkan pengaruh terhadap nominal klaim:**

| Rank | Feature | Pengaruh | Temuan EDA |
|---|---|---|---|
| 1 | **ICD_Group (Diagnosis)** | Sangat Tinggi | Neoplasma/Kanker = cost driver #1, menyumbang ~30-40% total nominal |
| 2 | **Lokasi RS** | Sangat Tinggi | Singapore = 3-5× biaya Indonesia; 33% volume tapi ~60% nominal |
| 3 | **Usia** | Tinggi | 60+ = 46,2% klaim, 60,3% nominal; avg Rp 71,8 Juta vs Rp 37,5 Juta (46-60) |
| 4 | **Jenis Rawat** | Tinggi | IP >> OP dalam nominal; distribusi sangat berbeda |
| 5 | **LOS** | Tinggi | Proxy intensitas perawatan |
| 6 | **Plan Code** | Sedang | M-002 paling banyak digunakan, variabilitas tertinggi |
| 7 | **Repeat Claimer** | Sedang | Pola prediktif: hemodialisis rutin vs kanker berulang |
| 8 | **Coverage Ratio** | Sedang | 71,8% undercovered, 27,4% full, 0,8% over |
| 9 | **Gender** | Rendah | Distribusi relatif seimbang |
| 10 | **Domisili** | Rendah | Proxy kelas ekonomi/risiko regional |

### 3.3 Pola Temporal yang Terdeteksi

- **Tren naik** pada klaim Jan-Jun 2025 vs Jan-Jun 2024 (+25,5%)
- **Fluktuasi bulanan** pada volume dan nominal klaim
- **Korelasi positif** antara volume klaim dan processing time (capacity constraint)
- **Hari kerja** mendominasi (~80%+ admisi Senin-Jumat)
- **Seasonality**: Perlu divalidasi lebih lanjut (data baru 19 bulan)

### 3.4 Pola Chronic Disease (Repeat Claimers)

**Dua klaster utama:**

| Pattern | Penyakit | Freq/Polis | Cost/Klaim | Prediktabilitas |
|---|---|---|---|---|
| High-Freq Low-Cost | Gagal Ginjal (N18.x) | 64-222 | Rp 1,5-7,8 Juta | **Sangat tinggi** (hemodialisis rutin) |
| Moderate-Freq High-Cost | Kanker (C-codes) | 30-89 | Rp 21-133 Juta | **Sedang** (tergantung stage/protokol) |

- **Top repeat claimer**: POL-2078 (222 klaim hemodialisis dalam 18 bulan)
- **Kasus termahal**: POL-0856 (Rp 5,97 Miliar dari 45 klaim kanker paru di Singapore)

### 3.5 Anomali & Data Quality

| Isu | Jumlah | % | Handling untuk ML |
|---|---|---|---|
| Outlier nominal (IQR method) | 507 | 11,0% | **11% klaim = 63,2% nominal** — log-transform target |
| Inpatient dengan LOS=0 | 762 | 16,5% | Reclassify atau buat flag variable |
| Missing IP/OP | 35 | 0,76% | Impute mode atau drop |
| Missing ICD | 4 | 0,09% | Drop (negligible) |
| Klaim Rp 0 | 13 | 0,28% | Exclude dari severity calculation |
| Potential duplicates | 16 | 0,35% | Validasi & deduplicate |
| Over-coverage (klaim > biaya RS) | 36 | 0,8% | Flag sebagai anomali |

### 3.6 Relasi Kausal Utama

```
Usia 60+ → Risiko Kanker/Kardiovaskular ↑ → Referral ke Singapore → Nominal 3-5× lebih tinggi
```

```
Plan M-002 (Regional Asia) → Akses Singapore/Malaysia → Nominal tinggi → Repeat claims tinggi
```

```
Chronic Disease (CKD/Cancer) → Repeat claimers → Frekuensi tinggi & prediktif → Kontribusi signifikan ke total
```

---

## 4. Problem Formulation

### 4.1 Definisi Formal

**Task**: Time series forecasting pada level agregat bulanan

**Input**: Data historis klaim bulanan (19 bulan: Jan 2024 — Jul 2025)

**Output**: Prediksi 5 bulan ke depan (Aug — Dec 2025) untuk 3 metrik

### 4.2 Target Variables

Dari data transaksional, perlu diagregasi menjadi time series bulanan:

```python
monthly = df.groupby('Bulan').agg(
    Claim_Frequency=('Claim ID', 'count'),                          # Jumlah klaim
    Claim_Severity=('Nominal Klaim Yang Disetujui', 'mean'),        # Rata-rata nominal
    Total_Claim=('Nominal Klaim Yang Disetujui', 'sum')             # Total nominal
).reset_index()
```

**Relasi matematis**: `Total_Claim = Claim_Frequency × Claim_Severity`

### 4.3 Data Historis yang Tersedia

| Bulan | Periode | Keterangan |
|---|---|---|
| 1-12 | Jan 2024 — Dec 2024 | Data historis penuh (12 bulan) |
| 13-19 | Jan 2025 — Jul 2025 | Data historis parsial (7 bulan) |
| 20-24 | Aug 2025 — Dec 2025 | **TARGET PREDIKSI** (5 bulan) |

**Catatan**: Hanya 19 titik data bulanan — ini adalah **keterbatasan utama** untuk time series modeling. Pendekatan hybrid direkomendasikan.

### 4.4 Horizon & Granularity

| Parameter | Nilai |
|---|---|
| **Forecast Horizon** | 5 bulan (Aug-Dec 2025) |
| **Granularity** | Bulanan (agregat seluruh portfolio) |
| **Level** | Portfolio-level (bukan per-polis) |
| **Update Frequency** | One-time forecast (bisa di-retrain bulanan) |

---

## 5. Data Preprocessing Pipeline

### 5.1 Step 1: Data Cleaning

```python
# 1. Drop duplicates (16 potential)
df = df.drop_duplicates(subset=['Nomor Polis', 'Tanggal Pasien Masuk RS',
                                 'Nominal Klaim Yang Disetujui'], keep='first')

# 2. Exclude klaim Rp 0 dari severity calculation
df_severity = df[df['Nominal Klaim Yang Disetujui'] > 0]

# 3. Handle missing values
df['Inpatient/Outpatient'].fillna('UNKNOWN', inplace=True)
df['Lokasi RS'].fillna('Indonesia', inplace=True)  # mode imputation
df = df.dropna(subset=['ICD Diagnosis'])  # hanya 4 record
```

### 5.2 Step 2: Agregasi ke Time Series Bulanan

```python
# Agregasi utama
monthly = df.groupby('Bulan').agg(
    Claim_Frequency=('Claim ID', 'count'),
    Claim_Severity=('Nominal Klaim Yang Disetujui', 'mean'),
    Total_Claim=('Nominal Klaim Yang Disetujui', 'sum')
).reset_index()

# Konversi Bulan ke datetime index
monthly['Bulan'] = pd.to_datetime(monthly['Bulan'])
monthly = monthly.set_index('Bulan').asfreq('MS')  # Monthly Start frequency
```

### 5.3 Step 3: Feature Engineering untuk Time Series

Untuk memperkaya model time series dengan informasi exogenous:

```python
# Agregasi fitur tambahan per bulan (exogenous variables)
monthly_features = df.groupby('Bulan').agg(
    # Komposisi demografis
    pct_age_60plus=('Usia', lambda x: (x >= 60).mean()),
    avg_age=('Usia', 'mean'),
    pct_male=('Gender', lambda x: (x == 'M').mean()),

    # Komposisi diagnosis
    pct_cancer=('ICD_Group', lambda x: (x == 'Neoplasma/Kanker').mean()),
    pct_cardio=('ICD_Group', lambda x: (x == 'Kardiovaskular').mean()),
    pct_renal=('ICD_Group', lambda x: (x == 'Genitourinari').mean()),

    # Komposisi lokasi
    pct_overseas=('Lokasi RS', lambda x: (x != 'Indonesia').mean()),
    pct_singapore=('Lokasi RS', lambda x: (x == 'Singapore').mean()),

    # Komposisi jenis rawat
    pct_inpatient=('Inpatient/Outpatient', lambda x: (x == 'IP').mean()),

    # Komposisi plan
    pct_plan_m002=('Plan Code', lambda x: (x == 'M-002').mean()),

    # Statistik nominal
    median_nominal=('Nominal Klaim Yang Disetujui', 'median'),
    std_nominal=('Nominal Klaim Yang Disetujui', 'std'),
    pct_outlier=('Nominal Klaim Yang Disetujui',
                 lambda x: (x > 124_375_517).mean()),  # IQR upper bound

    # Repeat claimers
    unique_polis=('Nomor Polis', 'nunique'),
    avg_claims_per_polis=('Nomor Polis', lambda x: len(x) / x.nunique()),
).reset_index()
```

### 5.4 Step 4: Transformasi

```python
# Log-transform untuk severity (right-skewed)
monthly['log_Severity'] = np.log1p(monthly['Claim_Severity'])
monthly['log_Total'] = np.log1p(monthly['Total_Claim'])

# Trend & seasonal decomposition
from statsmodels.tsa.seasonal import seasonal_decompose
decomp = seasonal_decompose(monthly['Total_Claim'], model='additive', period=12)
```

---

## 6. Modeling Approach: Time Series Forecasting

### 6.1 Strategi Utama

Karena data hanya 19 titik bulanan, digunakan **multi-model time series** approach:

```
┌─────────────────────────────────────────────────────┐
│                  MODELING PIPELINE                    │
│                                                       │
│  Model A: SARIMA/ARIMA ──► Claim_Frequency (count)   │
│  Model B: SARIMA/ARIMA ──► Claim_Severity (avg Rp)   │
│  Derivasi: Total_Claim = Frequency × Severity        │
│                                                       │
│  Alternatif: Model C: Direct forecast Total_Claim    │
│  Validasi : Cross-check A×B ≈ C                      │
└─────────────────────────────────────────────────────┘
```

### 6.2 Model A: Claim Frequency Forecast

**Target**: Jumlah klaim per bulan (count data)

**Pendekatan utama**: SARIMA (Seasonal ARIMA)

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Karena data < 2 tahun, seasonal period mungkin tidak cukup
# Mulai dengan non-seasonal ARIMA, test seasonal jika memungkinkan

# Step 1: Stationarity test
from statsmodels.tsa.stattools import adfuller
adf_result = adfuller(monthly['Claim_Frequency'])

# Step 2: Determine (p,d,q) via ACF/PACF
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Step 3: Fit model
model_freq = SARIMAX(
    monthly['Claim_Frequency'],
    order=(p, d, q),           # Ditentukan dari ACF/PACF
    seasonal_order=(P, D, Q, 12),  # Seasonal jika cukup data
    enforce_stationarity=False,
    enforce_invertibility=False
)
results_freq = model_freq.fit()

# Step 4: Forecast 5 bulan
forecast_freq = results_freq.forecast(steps=5)
```

**Alternatif jika SARIMA tidak fit (data terlalu pendek)**:
- **Prophet** (Facebook/Meta) — lebih robust untuk data pendek
- **Exponential Smoothing** (Holt-Winters) — simple tapi efektif
- **Linear trend + seasonal dummy** — paling simpel, baseline

```python
from prophet import Prophet

df_prophet = monthly[['Bulan', 'Claim_Frequency']].rename(
    columns={'Bulan': 'ds', 'Claim_Frequency': 'y'}
)
model_prophet = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
    changepoint_prior_scale=0.05  # Regularisasi untuk data pendek
)
model_prophet.fit(df_prophet)
future = model_prophet.make_future_dataframe(periods=5, freq='MS')
forecast = model_prophet.predict(future)
```

### 6.3 Model B: Claim Severity Forecast

**Target**: Rata-rata nominal klaim per bulan (continuous, right-skewed)

**Pendekatan**: SARIMA pada log-transformed severity

```python
# Log-transform untuk menormalisasi distribusi
monthly['log_Severity'] = np.log1p(monthly['Claim_Severity'])

model_sev = SARIMAX(
    monthly['log_Severity'],
    order=(p, d, q),
    enforce_stationarity=False,
    enforce_invertibility=False
)
results_sev = model_sev.fit()
forecast_log_sev = results_sev.forecast(steps=5)

# Back-transform
forecast_severity = np.expm1(forecast_log_sev)
```

**SARIMAX dengan exogenous variables** (jika komposisi berubah):

```python
# Jika ada prediksi/asumsi perubahan komposisi penyakit/lokasi
exog_vars = ['pct_cancer', 'pct_overseas', 'pct_age_60plus']

model_sev_x = SARIMAX(
    monthly['log_Severity'],
    exog=monthly_features[exog_vars],
    order=(p, d, q)
)
results_sev_x = model_sev_x.fit()

# Forecast memerlukan asumsi exog untuk 5 bulan ke depan
# Gunakan rata-rata 3 bulan terakhir sebagai proxy
future_exog = monthly_features[exog_vars].tail(3).mean().to_frame().T
future_exog = pd.concat([future_exog] * 5, ignore_index=True)
forecast_sev_x = results_sev_x.forecast(steps=5, exog=future_exog)
```

### 6.4 Model C: Total Claim (Derivasi + Direct)

```python
# Metode 1: Derivasi dari Model A × Model B
forecast_total_derived = forecast_freq.values * forecast_severity.values

# Metode 2: Direct forecast
model_total = SARIMAX(
    monthly['log_Total'],  # Log-transformed total
    order=(p, d, q),
    enforce_stationarity=False,
)
results_total = model_total.fit()
forecast_total_direct = np.expm1(results_total.forecast(steps=5))

# Cross-validation: bandingkan kedua metode
# Pilih yang MAPE-nya lebih rendah pada validation set
```

### 6.5 Ensemble & Final Prediction

```python
# Weighted average jika kedua metode reasonable
alpha = 0.6  # Weight untuk direct model (lebih stabil)
forecast_total_final = alpha * forecast_total_direct + (1 - alpha) * forecast_total_derived
```

---

## 7. Model Evaluation & Validation

### 7.1 Train-Test Split (Temporal)

Karena data hanya 19 bulan, gunakan **expanding window** cross-validation:

```
Split 1: Train [1-12]  → Test [13]  (Jan 2024-Dec 2024 → Jan 2025)
Split 2: Train [1-13]  → Test [14]  (→ Feb 2025)
Split 3: Train [1-14]  → Test [15]  (→ Mar 2025)
Split 4: Train [1-15]  → Test [16]  (→ Apr 2025)
Split 5: Train [1-16]  → Test [17]  (→ May 2025)
Split 6: Train [1-17]  → Test [18]  (→ Jun 2025)
Split 7: Train [1-18]  → Test [19]  (→ Jul 2025)

Final:   Train [1-19]  → Forecast [20-24] (→ Aug-Dec 2025)
```

### 7.2 Metrik Evaluasi

| Metrik | Formula | Target | Keterangan |
|---|---|---|---|
| **MAPE** | Mean Absolute Percentage Error | < 20% | Metrik utama — interpretable |
| **MAE** | Mean Absolute Error | Minimize | Robust terhadap outlier |
| **RMSE** | Root Mean Squared Error | Minimize | Sensitif terhadap error besar |
| **MdAPE** | Median APE | < 15% | Lebih robust dari MAPE |
| **Directional Accuracy** | % prediksi arah benar (naik/turun) | > 60% | Penting untuk trend detection |

### 7.3 Evaluasi Per Model

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_forecast(actual, predicted, model_name):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100

    print(f"=== {model_name} ===")
    print(f"  MAE  : {mae:,.0f}")
    print(f"  RMSE : {rmse:,.0f}")
    print(f"  MAPE : {mape:.1f}%")
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
```

### 7.4 Model Selection Criteria

```
1. MAPE < 20% pada expanding window CV
2. Forecast tidak menunjukkan pattern yang unreasonable (negatif, spike ekstrem)
3. Confidence interval mencakup actual values pada validation
4. Directional accuracy > 60%
5. Cross-check: Frequency × Severity ≈ Total (consistency check)
```

---

## 8. Output Format

### 8.1 Tabel Prediksi Final

```
15 unique values | 15 total values
─────────────────┬──────────────────
Variable         │ Predicted Value
─────────────────┼──────────────────
2025_08_Claim_Frequency  │  0
2025_08_Claim_Severity   │  0
2025_08_Total_Claim      │  0
2025_09_Claim_Frequency  │  0
2025_09_Claim_Severity   │  0
2025_09_Total_Claim      │  0
2025_10_Claim_Frequency  │  0
2025_10_Claim_Severity   │  0
2025_10_Total_Claim      │  0
2025_11_Claim_Frequency  │  0
2025_11_Claim_Severity   │  0
2025_11_Total_Claim      │  0
2025_12_Claim_Frequency  │  0
2025_12_Claim_Severity   │  0
2025_12_Total_Claim      │  0
─────────────────┴──────────────────
```

> **Catatan**: Nilai `0` akan diganti dengan hasil prediksi model setelah training.

### 8.2 Output Tambahan (untuk bisnis)

Selain 15 values, model juga akan menghasilkan:

| Output Tambahan | Deskripsi |
|---|---|
| **Confidence Interval** | 80% dan 95% CI untuk setiap prediksi |
| **Trend Direction** | Apakah klaim diprediksi naik/turun vs periode sebelumnya |
| **YoY Comparison** | Prediksi Aug-Dec 2025 vs aktual Aug-Dec 2024 (jika tersedia) |
| **Feature Importance** | Ranking faktor yang paling berpengaruh terhadap prediksi |
| **Decomposition** | Breakdown prediksi ke komponen: trend + seasonal + residual |

---

## 9. Feature Importance & Factor Analysis

### 9.1 Identifikasi Faktor Berpengaruh

Selain forecasting, perlu dilakukan **factor analysis** untuk menjawab pertanyaan bisnis: "Faktor apa yang paling berpengaruh terhadap nilai klaim?"

**Pendekatan**: Supplementary XGBoost/LightGBM model pada level per-klaim

```python
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit

# Features
feature_cols = ['Usia', 'Gender_encoded', 'Plan_encoded', 'Lokasi_encoded',
                'ICD_Group_encoded', 'IP_OP_encoded', 'LOS',
                'Reimburse_encoded', 'month', 'quarter']

X = df[feature_cols]
y = np.log1p(df['Nominal Klaim Yang Disetujui'])  # Log-transform target

# Train model
model_xgb = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model_xgb.fit(X_train, y_train)

# Feature importance (SHAP)
import shap
explainer = shap.TreeExplainer(model_xgb)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, feature_names=feature_cols)
```

### 9.2 Expected Feature Importance Ranking

Berdasarkan EDA, ranking yang diharapkan:

```
1. ICD_Group (Diagnosis)     ████████████████████  ~25-30%
2. Lokasi RS                 ████████████████      ~20-25%
3. Usia                      ██████████████        ~15-20%
4. Jenis Rawat (IP/OP)       ██████████            ~10-15%
5. LOS (Length of Stay)      ████████              ~8-12%
6. Plan Code                 ██████                ~5-8%
7. Temporal (bulan/kuartal)  ████                  ~3-5%
8. Gender                    ██                    ~2-3%
9. Domisili                  ██                    ~2-3%
10. Reimburse/Cashless       █                     ~1-2%
```

---

## 10. Risk, Keterbatasan & Mitigasi

### 10.1 Keterbatasan Data

| Risiko | Detail | Mitigasi |
|---|---|---|
| **Data pendek** | Hanya 19 bulan — kurang untuk seasonal decomposition (idealnya 3+ tahun) | Gunakan Prophet yang lebih robust untuk data pendek; tambahkan domain knowledge sebagai prior |
| **Seasonality tidak tervalidasi** | Belum ada data satu siklus tahunan penuh untuk membandingkan | Asumsi minimal seasonal effect; fokus pada trend |
| **Outlier dominasi** | 11% klaim = 63,2% nominal — satu klaim besar bisa menggeser prediksi bulanan | Log-transform; robustify dengan median-based metrics; sensitivity analysis |
| **Chronic claimers** | POL-2078 (222 klaim) sangat mempengaruhi trend — jika berhenti klaim, prediksi meleset | Pisahkan chronic vs incident claims; buat model terpisah |
| **External factors** | Inflasi medis, perubahan regulasi, pandemi — tidak tercermin dalam data historis | Skenario analysis; confidence interval wide |
| **Data SGD tidak lengkap** | Kurs SGD/IDR hanya tersedia Jan-Feb 2026 (24 baris), tidak mencakup periode klaim Jan 2024-Jul 2025 | **Harus dilengkapi** dari investing.com atau Bank Indonesia sebelum digunakan sebagai exogenous variable |

### 10.2 Asumsi Model

| No | Asumsi | Validitas |
|---|---|---|
| 1 | Komposisi portfolio (demografi, plan, lokasi) tidak berubah signifikan dalam 5 bulan ke depan | Reasonable untuk short-term |
| 2 | Tidak ada event katastrofik (pandemi, regulasi baru) | Cannot be modeled |
| 3 | Pola chronic disease tetap konsisten | High confidence (hemodialisis, kemo rutin) |
| 4 | Tren kenaikan 25,5% YoY berlanjut tapi mungkin moderating | Perlu monitoring aktual |
| 5 | Inflasi medis stabil | Reasonable untuk 5 bulan |

### 10.3 Sensitivity Analysis

Model akan diuji dengan skenario:

| Skenario | Asumsi | Impact pada Total Claim |
|---|---|---|
| **Base Case** | Tren berlanjut sesuai historis | Baseline prediction |
| **Optimistic** | Klaim turun 10% dari trend (efek preventif) | −10% dari baseline |
| **Pessimistic** | Klaim naik 15% dari trend (eskalasi kanker) | +15% dari baseline |
| **Catastrophic** | 1 klaim besar baru (>Rp 500 Juta/bulan) | +Rp 500 Juta/bulan |

---

## 11. Business Recommendations & Action Plan

### 11.1 Berdasarkan Temuan EDA → Inisiatif Bisnis

#### A. Seleksi Risiko (Underwriting)

| Faktor Risiko | Temuan EDA | Inisiatif |
|---|---|---|
| Usia 60+ | 46,2% klaim, 60,3% nominal | **Tiered premium** berdasarkan kelompok usia; review plafon untuk 60+ |
| Plan M-002 | Variabilitas tertinggi (max 222 klaim) | **Sub-segmentasi** premi M-002 berdasarkan risk profile |
| Diagnosis Kanker | Cost driver #1, outlier dominan | **Pre-existing condition screening** yang lebih ketat; limit khusus kanker |
| Lokasi Singapore | 3-5× biaya Indonesia | **Second opinion mandate** sebelum approval perawatan luar negeri |

#### B. Pencegahan (Preventive Care)

| Target | Program | Expected Impact |
|---|---|---|
| Gagal Ginjal (N18.x) | Early CKD screening untuk usia 45+ | Reduce hemodialysis frequency via early intervention |
| Kanker (C-codes) | Cancer screening program (mammogram, colonoscopy) | Early detection → lower treatment cost |
| Kardiovaskular (I-codes) | Wellness program (kolesterol, tekanan darah) | Reduce cardiac event probability |
| All chronic | Disease management program terpadu | Optimasi frekuensi dan biaya perawatan rutin |

#### C. Deteksi Dini (Early Warning)

| Signal | Threshold | Action |
|---|---|---|
| Repeat claimer baru | > 5 klaim dalam 3 bulan | Flag untuk case management review |
| Eskalasi ke luar negeri | Klaim pertama ke Singapore/Malaysia | Assign care coordinator |
| Nominal outlier | > Rp 124 Juta (IQR upper) | Mandatory medical review |
| Zero-claim klaim | Nominal = Rp 0 | Investigasi otomatis |
| Duplicate detection | Same polis + date + nominal | Block payment, trigger audit |

### 11.2 Penggunaan Output Model

```
                    ┌──────────────────────┐
                    │  15 Predicted Values │
                    │  (Aug-Dec 2025)      │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
     ┌────────────────┐ ┌──────────────┐ ┌──────────────┐
     │ CLAIM FREQUENCY│ │CLAIM SEVERITY│ │ TOTAL CLAIM  │
     │ Forecast       │ │ Forecast     │ │ Forecast     │
     └───────┬────────┘ └──────┬───────┘ └──────┬───────┘
             │                 │                │
             ▼                 ▼                ▼
     ┌────────────────┐ ┌──────────────┐ ┌──────────────┐
     │ Staffing &     │ │ Reserve      │ │ Premium      │
     │ Capacity Plan  │ │ Adequacy     │ │ Adjustment   │
     │ (tim klaim)    │ │ (aktuarial)  │ │ Decision     │
     └────────────────┘ └──────────────┘ └──────────────┘
```

- **Claim Frequency** → Input untuk **capacity planning** tim klaim dan RS jaringan
- **Claim Severity** → Input untuk **reserve calculation** dan **reinsurance** adequacy
- **Total Claim** → Input untuk **premium pricing** dan **loss ratio** projection

---

## 12. Implementation Roadmap

### Phase 1: Data Preparation & Baseline (Minggu 1-2)

```
□ Agregasi data ke level bulanan (19 titik data)
□ EDA pada time series: stationarity, ACF/PACF, decomposition
□ Baseline model: Simple Moving Average, Naive forecast
□ Establish evaluation framework (expanding window CV)
```

### Phase 2: Model Development (Minggu 3-4)

```
□ ARIMA/SARIMA untuk Claim Frequency
□ ARIMA/SARIMA untuk Claim Severity (log-transformed)
□ Prophet sebagai alternatif (jika SARIMA underperforms)
□ SARIMAX dengan exogenous variables
□ Direct vs derived Total Claim comparison
□ Hyperparameter tuning via grid search
```

### Phase 3: Evaluation & Selection (Minggu 5)

```
□ Expanding window cross-validation (7 folds)
□ Compare models: MAPE, MAE, RMSE, directional accuracy
□ Sensitivity analysis (3 skenario)
□ Consistency check: Freq × Severity ≈ Total
□ Select final model per metrik
```

### Phase 4: Factor Analysis (Minggu 5-6)

```
□ XGBoost supplementary model pada level per-klaim
□ SHAP values untuk feature importance
□ Partial dependence plots untuk top factors
□ Business interpretation & recommendations
```

### Phase 5: Final Output & Reporting (Minggu 6)

```
□ Generate 15 predicted values
□ Confidence intervals (80%, 95%)
□ Visualisasi: actual vs forecast plot
□ Business recommendation report
□ Handover ke tim aktuarial/underwriting
```

---

## 13. Technical Stack

| Component | Tool | Version |
|---|---|---|
| External Data | Kurs MYR/IDR (514 baris), Kurs SGD/IDR (perlu dilengkapi) | investing.com |
| Language | Python | 3.12+ |
| Data Processing | pandas, numpy | Latest |
| Time Series | statsmodels (SARIMAX), prophet | Latest |
| ML (Factor Analysis) | xgboost, lightgbm, scikit-learn | Latest |
| Explainability | shap | Latest |
| Visualization | matplotlib, seaborn, plotly | Latest |
| Notebook | Jupyter Lab / VS Code | Latest |

---

*Dokumen ini merupakan perencanaan komprehensif untuk project Machine Learning prediksi klaim asuransi kesehatan. Semua insight didasarkan pada hasil Deep EDA terhadap 4.625 transaksi klaim dari periode Januari 2024 — Juli 2025.*
