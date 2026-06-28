# 🏦 LoanIQ — Loan Approval Intelligence System

> Prediksi kelayakan pinjaman secara instan menggunakan machine learning.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-EC6C00?logo=xgboost&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22C55E)

---

## 📌 Tentang Proyek

**LoanIQ** adalah aplikasi web berbasis machine learning yang membantu menganalisis kelayakan pengajuan pinjaman nasabah secara cepat dan objektif. Dengan memasukkan data profil nasabah, kondisi finansial, dan nilai aset, sistem akan menghasilkan prediksi keputusan kredit beserta tingkat keyakinan model secara real-time.

Proyek ini dibangun sebagai implementasi end-to-end dari model klasifikasi binary — mulai dari pelatihan model hingga deployment sebagai aplikasi web yang dapat diakses publik.

---

## ✨ Fitur Utama

- **Prediksi Real-time** — Hasil analisis kelayakan pinjaman dalam hitungan detik
- **Confidence Score** — Visualisasi tingkat keyakinan model dalam bentuk persentase dan progress bar
- **11 Fitur Input** — Mencakup profil nasabah, data finansial, dan portofolio aset
- **Dark Fintech UI** — Tampilan profesional yang dirancang untuk kemudahan penggunaan
- **Powered by XGBoost** — Model gradient boosting dengan performa tinggi untuk klasifikasi kredit

---

## 🧠 Model Machine Learning

| Detail | Keterangan |
|---|---|
| **Algoritma** | XGBoost (Extreme Gradient Boosting) |
| **Tipe** | Binary Classification |
| **Output** | Approved / Rejected |
| **Preprocessing** | StandardScaler untuk fitur numerik |
| **Fitur** | 11 variabel (demografi, finansial, aset) |

### Fitur yang Digunakan

| Fitur | Deskripsi |
|---|---|
| `no_of_dependents` | Jumlah tanggungan keluarga |
| `education` | Tingkat pendidikan (Graduate / Not Graduate) |
| `self_employed` | Status pekerjaan |
| `income_annum` | Pendapatan tahunan |
| `loan_amount` | Jumlah pinjaman yang diajukan |
| `loan_term` | Tenor pinjaman (tahun) |
| `cibil_score` | Skor kredit nasabah (300–900) |
| `residential_assets_value` | Nilai aset properti |
| `commercial_assets_value` | Nilai aset komersial |
| `luxury_assets_value` | Nilai aset mewah |
| `bank_asset_value` | Nilai tabungan / deposito |

---

## 🗂️ Struktur Proyek

```
loan-approval-app/
├── app.py                  # Aplikasi Streamlit utama
├── requirements.txt        # Dependensi Python
├── loan_model_final.pkl    # Model XGBoost yang sudah dilatih
├── loan_scaler.pkl         # StandardScaler untuk preprocessing
└── README.md               # Dokumentasi ini
```

---

## 🚀 Cara Menjalankan Secara Lokal

**1. Clone repository ini**
```bash
git clone https://github.com/username/loan-approval-app.git
cd loan-approval-app
```

**2. Install dependensi**
```bash
pip install -r requirements.txt
```

**3. Jalankan aplikasi**
```bash
streamlit run app.py
```

**4. Buka di browser**
```
http://localhost:8501
```

---

## ☁️ Demo Online

Aplikasi ini telah di-deploy dan dapat diakses secara publik melalui Streamlit Community Cloud:

🔗 **[Buka Aplikasi →](https://share.streamlit.io)**

> Ganti link di atas dengan URL deployment kamu setelah berhasil deploy.

---

## 🛠️ Tech Stack

- **Frontend / UI** — [Streamlit](https://streamlit.io)
- **Model** — [XGBoost](https://xgboost.readthedocs.io)
- **Data Processing** — [Pandas](https://pandas.pydata.org), [NumPy](https://numpy.org)
- **Preprocessing** — [Scikit-learn](https://scikit-learn.org) (StandardScaler)
- **Serialisasi Model** — [Joblib](https://joblib.readthedocs.io)
- **Deployment** — [Streamlit Community Cloud](https://share.streamlit.io)

---

## 📄 Lisensi

Proyek ini menggunakan lisensi [MIT](LICENSE). Bebas digunakan, dimodifikasi, dan didistribusikan dengan tetap mencantumkan atribusi.

---

<p align="center">Dibuat dengan ☕ dan Python</p>
