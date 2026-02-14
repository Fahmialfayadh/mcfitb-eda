# Rangkuman Komprehensif Temuan EDA — Data Klaim Asuransi

## A. Gambaran Umum Dataset

| Metrik | Nilai |
|---|---|
| Periode data | Jan 2024 — Jul 2025 (~19 bulan) |
| Total klaim | 4,625 transaksi |
| Total polis | 4,096 polis, hanya **1,210 (29.5%)** pernah klaim |
| Total nominal klaim | **Rp 254.6 Miliar** |
| Rata-rata per klaim | Rp 55.0 Juta |
| Median per klaim | Rp 14.5 Juta |
| Status klaim | **100% PAID** (tidak ada rejected — kolom tidak informatif) |

---

## B. Temuan Fakta per Dimensi

### 1. Tipe Klaim & Rawat
- **Reimburse 59% vs Cashless 41%**
- Mayoritas klaim adalah **Outpatient** (lebih banyak dari Inpatient)
- Terdapat juga One Day Care (ODC) dan One Day Surgery (ODS) dalam jumlah kecil
- **762 record Inpatient dengan LOS = 0 hari** — kemungkinan data quality issue atau prosedur one-day
- Rata-rata LOS inpatient ada di sekitar beberapa hari, distribusi right-skewed

### 2. Lokasi Rumah Sakit & Medical Tourism
- **Indonesia 67% vs Luar Negeri 33%** dari jumlah klaim
- Namun secara nominal: **Luar negeri = Rp 149.8 Miliar (58.8%)** dari total
- Destinasi LN utama: **Singapore (1,019 klaim)** dan **Malaysia (456 klaim)**
- Biaya berobat LN bisa **2x sampai 24x lebih mahal** dibanding Indonesia untuk diagnosis yang sama (contoh: D25.9 = 24.4x, C61 = 15.3x, C18 = 11.5x)
- **Klaim catastrophic (>500 Juta): 76 klaim (1.6%)** tapi menyumbang **24.9% (Rp 63.5M) total nominal**
- Domisili pengirim pasien LN terbanyak: **Surabaya (560)**, Jakarta (315), Balikpapan (141), Makassar (139), Yogyakarta (114)
- Balikpapan paling tinggi rasio ke LN: **77.5%** klaim-nya ke luar negeri

### 3. Diagnosis & Cost Driver
- **946 ICD unik** dalam dataset
- **Kanker (Neoplasma, ICD C\*)** = cost driver terbesar:
  - 934 klaim (23.3% dari total)
  - Total **Rp 84.8 Miliar (33.3%)** dari seluruh nominal
  - **46 polis kanker (29%) menguasai 80% total biaya kanker** — sangat long-tail
- Klaim kanker di LN: rata-rata **Rp 110 Juta** vs Indonesia **Rp 42 Juta** (2.6x)
- Diagnosis yang hampir selalu ke LN: **C25 (pankreas) 100%, C56 (ovarium) 100%, C90.0 (multiple myeloma) 100%, C34.9 (paru) 93%**
- Diagnosis yang umumnya di Indonesia tapi tetap ada ke LN: **I10 (hipertensi), C18.9 (kolon), N20 (urolithiasis)**

### 4. Nominal & Coverage
- Distribusi nominal **sangat right-skewed** — Q1: Rp 2.3 Juta, Median: Rp 14.5 Juta, Q3: Rp 51 Juta, Max: Rp 2.2 Miliar
- Korelasi biaya RS vs nominal klaim: **r = 0.99** (hampir linier)
- **13 klaim dengan nominal disetujui = Rp 0** (tapi status PAID)
- **36 klaim over-coverage** (nominal disetujui > biaya RS)
- **15.4% klaim under-covered** (rasio coverage < 0.8) — total gap **Rp 15.8 Miliar** yang ditanggung nasabah
- Under-covered per lokasi: **Taiwan 40%**, Singapore 22.1%, Malaysia 18%, Indonesia 13%
- Under-covered per plan: **M-001 = 21.3%**, M-002 = 16.4%, M-003 = 7.5%
- Tren rasio coverage menurun seiring waktu — indikasi **inflasi medis**

### 5. Demografis
- Gender relatif seimbang dalam jumlah klaim
- Kelompok usia **60+ tahun** = klaim terbanyak (2,137) dengan rata-rata tertinggi (Rp 71.8 Juta)
- Kelompok usia **46-60** = terbanyak kedua (1,897 klaim)
- Usia muda (0-17 dan 18-30) masing-masing hanya 38 dan 71 klaim
- **3 plan code**: M-001 (176 polis), M-002 (730 polis, terbanyak), M-003 (304 polis)

### 6. Pola Temporal
- Processing time: **Mean 66 hari, Median 61 hari, Max 606 hari**
- Submission lag: Indonesia **median 62 hari** vs LN **median 63 hari** (tapi std LN lebih tinggi: 44 vs 27)
- Cashless processing lebih lama: **median 70 hari** vs Reimburse **median 51 hari**
- **1,478 klaim kecil (<5 Juta) butuh >30 hari diproses** — inefisiensi
- Klaim >100 Juta: processing **median 73 hari** (terlama)

### 7. Repeat Claimers
- Median klaim per polis = **2 klaim**, tapi distribusi sangat skewed
- Top repeat claimers: **POL-2078 (222 klaim, Rp 1.73M)**, POL-2200 (167 klaim), POL-2111 (91), POL-2329 (89)
- Mayoritas repeat claimers berusia 45-62 tahun, dominan Plan M-002

### 8. Fraud Flags
- **859 klaim** terdeteksi potensi split bill (polis sama, RS sama, <3 hari, diagnosis mirip)
- **14 polis** dengan repeat anomaly (>10 klaim, avg gap <14 hari)
- **58 klaim** outlier per diagnosis (z-score > 3)
- **333 polis ter-flag** secara total, 5 polis mendapat **3 flag sekaligus**: POL-2200, POL-2111, POL-2878, POL-1255, POL-0856
- 16 record terdeteksi sebagai **potential duplicates** (polis, tanggal, nominal sama persis)

---

## C. Implikasi untuk Research Data External

| Area | Data External yang Dibutuhkan |
|---|---|
| **Medical tourism pricing** | Tarif benchmark RS di Singapore & Malaysia untuk top ICD codes (C50, C34, C18, C61) — untuk validasi apakah biaya klaim wajar |
| **Inflasi medis** | Data inflasi medis tahunan Indonesia, Singapore, Malaysia — untuk memahami tren coverage gap |
| **Epidemiologi kanker** | Prevalensi & insidensi kanker di Indonesia per usia/gender — untuk benchmarking claim rate |
| **RS network & tarif** | Data tarif INA-CBGs atau tarif RS mitra untuk perbandingan biaya domestik |
| **Demografi regional** | Data penduduk & tingkat ekonomi per kota domisili (Surabaya, Jakarta, Balikpapan, Makassar) — untuk kontekstualisasi pola medical tourism |
| **Fraud benchmark** | Industry benchmark untuk claim frequency, split bill pattern, dan acceptable processing time |

---

## D. Implikasi untuk Persiapan Machine Learning

| Aspek | Status & Catatan |
|---|---|
| **Target variable** | Perlu didefinisikan — opsi: (1) prediksi nominal klaim, (2) prediksi fraud flag, (3) prediksi apakah akan klaim ke LN, (4) prediksi repeat claimer |
| **Feature engineering sudah ada** | Usia, LOS, Processing_Days, Selisih_Klaim, Rasio_Coverage, Bulan, Kuartal, DayOfWeek, Kategori_Nominal, ICD_Group |
| **Data quality issues** | 13 klaim nominal 0, 36 over-coverage, 762 IP dengan LOS=0, 16 potential duplicates — **harus diputuskan: drop, impute, atau flag** |
| **Class imbalance** | 70.5% polis tidak pernah klaim; distribusi nominal sangat skewed; fraud flags hanya ~7% polis |
| **Feature tambahan potensial** | Riwayat klaim sebelumnya (cumulative), jarak domisili ke RS, interaksi gender×usia×plan, seasonal encoding |
| **Missing values** | Inpatient/Outpatient (35), ICD (4), Tgl Bayar (35), Lokasi RS (7) — minor tapi perlu strategi handling |
| **Catatan penting** | Status Klaim = 100% PAID, artinya **tidak ada label rejected** — jika target adalah fraud, label harus dibuat dari proxy (flag rules) atau data tambahan |
