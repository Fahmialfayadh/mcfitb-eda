# Plan Lite - Trend Only Forecast

Dokumen ini adalah versi ringkas dari plan besar, khusus untuk kebutuhan:
- melihat **trend 5 bulan ke depan**,
- dengan model sederhana,
- dan risiko overfitting minimal pada data bulanan yang pendek (19 bulan).

## 1. Scope

1. Target tetap:
- `Claim_Frequency`
- `Claim_Severity`
- `Total_Claim`

2. Horizon:
- Agustus 2025 - Desember 2025.

3. Prinsip:
- trend-oriented, bukan precision micro-segmentation.

## 2. Data Yang Dipakai

1. Sumber utama:
- `monthly_timeseries.csv`

2. Periode historis:
- Jan 2024 - Jul 2025 (19 titik bulanan).

3. Tidak menambah fitur mikro polis pada fase ini.

## 3. Cleansing Minimal

1. Validasi tanggal bulanan berurutan.
2. Cek nilai kosong/negatif pada 3 target utama.
3. Cek konsistensi kasar `Total` vs `Frequency * Severity`.
4. Tanpa imputasi kompleks.

## 4. Modeling Lite

1. Model pool sederhana per target:
- `naive`
- `seasonal_naive`
- `ma3`
- `ets` (damped trend)

2. Ensemble sederhana:
- optimize weight dengan kombinasi 2 window:
  - strict: 7 bulan
  - operational: 3 bulan
- bobot evaluasi condong ke near-term (`operational`) karena target utama trend beberapa bulan ke depan.

3. Total consistency:
- blend `Total_direct` dan `Total_derived = Frequency * Severity`.

## 5. Evaluation

1. Metrik utama: `MEPA`
2. Metrik pendukung: `MAE`, `RMSE`, `MdAPE`
3. Fokus interpretasi:
- arah trend,
- stabilitas forecast,
- range ketidakpastian (CI).

## 6. Output

1. Notebook:
- `ml_claims_forecast_lite_trend.ipynb`

2. Prediksi:
- `submission_mepa_lite_trend.csv`

3. Detail evaluasi:
- `forecast_details_mepa_lite_trend.json`

## 7. Kapan Naik ke Pipeline Kompleks

Upgrade ke pipeline advanced dilakukan hanya jika:
1. strict MEPA tetap jauh di atas target setelah lite tuning.
2. perlu insight underwriting/fraud berbasis polis.
3. data historis bertambah signifikan (misal >30-36 bulan).

Untuk kondisi saat ini, pendekatan lite adalah default yang paling rasional.
