# Rangkuman & Plan Komprehensif
## Data Cleansing, Feature Engineering, Modeling, dan ML

Dokumen ini merangkum hasil terbaru dari:
- `eda_repeat_policy_country_feature_engineering.ipynb`
- dan menghubungkannya dengan insight sebelumnya (`plan_machinelearning.md`, `rangkuman_deep_eda_klaim.ipynb`, serta hasil eksperimen MEPA terakhir).

---

## 1. Objective Utama

Tujuan akhir tetap sama: forecasting agregat bulanan (Agustus-Desember 2025) untuk:
1. `Claim_Frequency`
2. `Claim_Severity`
3. `Total_Claim`

Dengan fokus evaluasi pada metrik **MEPA** (target bisnis: sedekat mungkin ke `< 5`).

---

## 2. Ringkasan Koneksi Insight Lama vs Insight Baru

## 2.1 Insight Lama (Deep EDA/Plan Sebelumnya)

1. Portofolio bersifat **heavy-tail**:
- sekitar **11% klaim outlier menyerap 63.2% total nominal**.

2. Cost driver utama:
- **Neoplasma/Kanker** dominan dari sisi frekuensi dan nominal.

3. Relasi kausal penting:
- `Usia 60+ -> risiko penyakit berat (terutama kanker/kardiovaskular) -> perawatan luar negeri (terutama Singapore) -> biaya tinggi`.

4. Repeat claimers membentuk pola berbeda:
- cluster high-frequency low-cost (contoh pola hemodialisis),
- cluster moderate-frequency high-cost (contoh kanker).

5. Keterbatasan struktural:
- data bulanan hanya **19 titik** (Jan 2024 - Jul 2025), sehingga model kompleks mudah overfit/tidak stabil.

## 2.2 Insight Baru (EDA Repeat Polis x Negara RS)

Sumber: `feature_analysis_outputs/*`

1. Setelah dedup + cleaning di notebook terbaru:
- **4,617** klaim,
- **1,210** polis,
- periode **2024-01-01 s.d. 2025-07-31**.

2. Distribusi negara RS:
- Indonesia: **67.06%**
- Singapore: **22.01%**
- Malaysia: **9.85%**
- Other Overseas: **0.93%**

3. Kandidat penyakit chronic-repeat (hasil data-driven):
- `Mata/Telinga`
- `Neoplasma/Kanker`
- `Muskuloskeletal`
- `Genitourinari`
- `Gejala Umum`
- `Neoplasma/Darah`

4. Intensitas repeat yang sangat kuat pada beberapa grup:
- `Neoplasma/Kanker`: repeat rate tinggi, avg claims repeater sangat tinggi.
- `Genitourinari`: avg claims repeater sangat tinggi.

5. Bukti kuat bahwa biaya sangat dipengaruhi kombinasi penyakit x negara:
- `Genitourinari` di Singapore ~ **8.93x** median Indonesia.
- `Neoplasma/Kanker` di Singapore ~ **4.48x** Indonesia.
- `Mata/Telinga` di Singapore ~ **4.10x** Indonesia.
- (contoh lain: `Endokrin/Metabolik` Singapore ~ **11.94x**, walau volume lebih kecil).

6. Sinyal bulanan baru (leakage-safe, history based):
- `repeat_claim_share_hist` rata-rata tinggi (mean ~0.437).
- `chronic_claim_share` stabil tinggi (mean ~0.637).
- `country_cost_idx` dan `repeat_country_cost_idx` fluktuatif (mencerminkan pressure biaya lintas negara).

7. Risk ranking polis berhasil diekstrak (contoh top score: `POL-2078`, `POL-2878`, `POL-2833`, `POL-0856`), cocok untuk future underwriting/fraud triage + feature risiko.

---

## 3. Gap ke Target MEPA (Status Saat Ini)

Dari model terbaru (`forecast_details_mepa_eda_enhanced.json`):

1. `Claim_Frequency`
- strict: **6.78**
- operational: **2.86**

2. `Claim_Severity`
- strict: **10.36**
- operational: **3.66**

3. `Total_Claim`
- strict: **14.10**
- operational: **2.29**

Kesimpulan gap:
- bottleneck utama ada di **Severity (strict)** dan dampaknya ke **Total (strict)**.
- untuk menurunkan strict MEPA, perlu model yang secara eksplisit menangkap **regime biaya** (normal vs tail) berbasis repeat-disease-country signals.

---

## 4. Plan Komprehensif Data Cleansing

## 4.1 Prinsip Umum

1. Pisahkan cleansing untuk 2 level:
- level transaksi klaim,
- level time series bulanan.

2. Semua transform harus reproducible dan leakage-safe.

## 4.2 Checklist Cleansing Transaksi

1. Dedup utama:
- key: `Nomor Polis + Tanggal Pasien Masuk RS + Nominal Klaim Yang Disetujui`.

2. Standardisasi tanggal:
- `visit_date = Tanggal Pasien Masuk RS` fallback `Tanggal Pembayaran Klaim`.

3. Standardisasi lokasi RS:
- map ke `country_group` konsisten: `Indonesia`, `Singapore`, `Malaysia`, `Other_Overseas`, `Unknown`.

4. Numerik nominal:
- coercion to float,
- cek nilai negatif/aneh,
- flag zero-claim untuk analisis khusus severity.

5. Missing handling minimal-risk:
- `ICD_Group` missing -> `Unknown` (jangan drop masif),
- audit proporsi missing per kolom penting.

6. Outlier tagging (bukan langsung buang):
- maintain flag outlier (upper bound EDA),
- simpan juga share outlier per bulan.

## 4.3 QA Data (wajib sebelum modeling)

1. Cek consistency per bulan:
- `Total_Claim` vs `Claim_Frequency * Claim_Severity`.

2. Cek anomali distribusi negara:
- lonjakan share Singapore/Malaysia harus tervalidasi event-driven.

3. Cek drift feature:
- PSI/KS sederhana per kuartal untuk fitur utama (`pct_cancer`, `repeat_share`, `country_cost_idx`).

---

## 5. Plan Komprehensif Feature Engineering

## 5.1 Layer Fitur yang Akan Dipakai

## Layer A: Policy-Disease Features (micro risk)

Sumber: `policy_disease_repeat_profile.csv`

1. `n_claims` per `(polis, ICD_Group)`
2. `avg_interval_days`, `med_interval_days`, `std_interval_days`
3. `claims_per_active_month`
4. `dominant_country`, `dominant_country_share`
5. `share_indonesia/singapore/malaysia/other`

## Layer B: Disease-Country Cost Features (tarif risk)

Sumber: `disease_country_cost_profile.csv`

1. `median_nominal` per `(ICD_Group, country_group)`
2. `cost_multiplier_vs_id` (median-based)
3. robust quantile stats (`p90_nominal`)

## Layer C: Policy-Level Aggregates (risk scoring)

Sumber: `policy_repeat_country_features.csv`

1. `repeat_intensity`
2. `avg_repeat_country_risk_score`
3. `max_repeat_country_risk_score`
4. `n_repeat_disease`, `n_chronic_disease`
5. chronic subset metrics: `chronic_claims`, `chronic_nominal`, `chronic_avg_interval_days`

## Layer D: Monthly Leakage-Safe Signals (forecast-ready)

Sumber: `monthly_repeat_country_signals.csv`

1. `repeat_claim_share_hist_lag1`
2. `chronic_claim_share_lag1`
3. `repeat_and_chronic_share_lag1`
4. `country_cost_idx_lag1`
5. `repeat_country_cost_idx_lag1`
6. `singapore_share_lag1`, `malaysia_share_lag1`

## Layer E: Existing Macro Portfolio Features

Dari pipeline sebelumnya:
1. `pct_age_60plus`, `pct_cancer`, `pct_plan_m002`, `pct_overseas`
2. `outlier_rate`, `outlier_total_share`
3. FX features `Avg_Kurs_SGD`, `Avg_Kurs_MYR`, change pct
4. calendar harmonics (`mon_sin`, `mon_cos`)

## 5.2 Fitur Turunan Baru yang Direkomendasikan

1. `repeat_tail_pressure_idx`
- formula awal: `repeat_claim_share_hist_lag1 * repeat_country_cost_idx_lag1 * outlier_rate_lag1`

2. `cancer_overseas_pressure_idx`
- formula awal: `pct_cancer_lag1 * singapore_share_lag1 * cost_multiplier_cancer_sg`

3. `chronic_stability_idx`
- fungsi dari kebalikan variabilitas interval chronic (`1 / (1 + std_interval_days_chronic)`)

4. `policy_risk_concentration_idx`
- agregat top-k policy risk score concentration per bulan (contoh top 20 polis).

---

## 6. Plan Modeling & Machine Learning

## 6.1 Prinsip Arsitektur Model

Gunakan pendekatan **hybrid + hierarchical**:

1. Model A: `Claim_Frequency`
- model stabil data pendek + exog lag (`ETS`, seasonal naive, regularized regression).

2. Model B: `Claim_Severity` (bottleneck utama)
- harus dipisah regime:
  - regime normal,
  - regime tail/high-cost month.

3. Model C: `Total_Claim`
- direct forecast + derived forecast (`Freq * Severity`) dengan blend terkalibrasi.

## 6.2 Severity Regime Strategy (Prioritas Utama)

1. Bangun `tail_flag_month`:
- threshold berbasis `outlier_total_share`, `repeat_country_cost_idx`, `cancer_overseas_pressure_idx`.

2. Dua model severity:
- `Model_normal`: fokus bulan low-tail (smooth dynamics).
- `Model_tail`: fokus bulan high-tail (lebih responsif ke disease-country pressure).

3. Gating/blending:
- bobot model tail meningkat saat `tail_flag_month`/`tail_score` naik.

4. Optional target transform:
- selain `log1p`, uji `Huber/Quantile` style objective untuk robust ke ekstrem.

## 6.3 Candidate Model Set

1. Baseline robust:
- `ETS`, `seasonal naive`, `MA rolling`.

2. Regularized linear:
- `Ridge`, `ElasticNet` (fit untuk data kecil).

3. Tree model terbatas (opsional, disiplin regularisasi):
- `XGBoost/LightGBM` shallow depth, fitur sedikit, hanya jika validasi konsisten.

4. Ensemble:
- optimize weights via combined objective strict + operational.

---

## 7. Validation & Experiment Design

## 7.1 Backtest Protocol

1. Expanding window:
- strict window: 7 bulan terakhir,
- operational window: 3 bulan terakhir.

2. Metrik utama:
- MEPA per target,
- plus MAE/RMSE/MdAPE.

3. Constraint evaluasi:
- penalti jika `Total_Claim` tidak konsisten dengan `Freq * Severity` di luar toleransi tertentu.

## 7.2 Ablation Wajib

1. Baseline lama vs baseline + fitur repeat-country.
2. Tanpa policy-risk features vs dengan policy-risk features.
3. Tanpa regime severity vs dengan regime severity.
4. Tanpa FX vs dengan FX.

Deliverable ablation:
- tabel delta MEPA per target dan per window.

---

## 8. Roadmap Eksekusi Praktis

## Phase 0 - Data Foundation

1. Freeze dataset snapshot + cleansing pipeline versi final.
2. Generate ulang semua file di `feature_analysis_outputs/`.
3. QA checklist lulus.

## Phase 1 - Feature Mart

1. Bangun dataset modeling bulanan yang menggabungkan:
- fitur lama,
- monthly repeat-country signals,
- agregat policy risk concentration.

2. Pastikan semua fitur forecasting tersedia dalam bentuk lag (no future leakage).

## Phase 2 - Modeling Iteratif

1. Iterasi cepat Frequency dan Total (stabil model).
2. Fokus mendalam Severity regime.
3. Ensemble + calibration + coherence blend.

## Phase 3 - Model Selection

1. Pilih model berdasarkan objective gabungan strict-operational.
2. Final sanity check forecast curve (business plausibility).
3. Ekspor submission + detail diagnostics.

---

## 9. Prioritas Tinggi (Next Action)

1. Implement **severity regime model** berbasis `repeat_country_cost_idx_lag1` + `outlier_total_share_lag1` + `pct_cancer_lag1`.
2. Tambahkan **policy concentration monthly features** (top-N risk score share) ke pipeline forecasting.
3. Jalankan ablation terstruktur untuk membuktikan kontribusi fitur baru terhadap MEPA.

---

## 10. Nama File Data & Dokumen Acuan

1. Notebook EDA terbaru:
- `eda_repeat_policy_country_feature_engineering.ipynb`

2. Output fitur:
- `feature_analysis_outputs/policy_disease_repeat_profile.csv`
- `feature_analysis_outputs/disease_repeat_profile.csv`
- `feature_analysis_outputs/chronic_icd_candidates.csv`
- `feature_analysis_outputs/disease_country_cost_profile.csv`
- `feature_analysis_outputs/policy_repeat_country_features.csv`
- `feature_analysis_outputs/monthly_repeat_country_signals.csv`
- `feature_analysis_outputs/feature_candidates_catalog.csv`

3. Dokumen lama:
- `plan_machinelearning.md`
- `rangkuman_deep_eda_klaim.ipynb`

4. Hasil model terakhir:
- `forecast_details_mepa_eda_enhanced.json`

