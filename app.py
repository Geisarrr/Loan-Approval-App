import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================
# KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="LoanIQ — Loan Approval Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# CUSTOM CSS — DARK FINTECH STYLE
# ==========================================
st.markdown("""
<style>
  /* Import font */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');

  /* ─── Global Reset ─── */
  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0A0D14;
    color: #E2E8F0;
  }

  /* ─── Hide default Streamlit chrome ─── */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container {
    padding: 2rem 3rem 4rem 3rem;
    max-width: 1200px;
  }

  /* ─── Hero Header ─── */
  .hero {
    background: linear-gradient(135deg, #0F1724 0%, #111827 60%, #0A1628 100%);
    border: 1px solid #1E293B;
    border-radius: 16px;
    padding: 2.5rem 2.5rem 2rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
  }
  .hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(56, 189, 248, 0.08) 0%, transparent 70%);
    border-radius: 50%;
  }
  .hero-badge {
    display: inline-block;
    background: rgba(56, 189, 248, 0.1);
    border: 1px solid rgba(56, 189, 248, 0.3);
    color: #38BDF8;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 1rem;
  }
  .hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: #F1F5F9;
    line-height: 1.2;
    margin: 0 0 0.5rem 0;
  }
  .hero-title span { color: #38BDF8; }
  .hero-sub {
    color: #64748B;
    font-size: 0.9rem;
    font-weight: 400;
    margin: 0;
  }
  .hero-stats {
    display: flex;
    gap: 2rem;
    margin-top: 1.5rem;
  }
  .stat-item { line-height: 1.3; }
  .stat-val {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: #38BDF8;
  }
  .stat-label {
    font-size: 0.72rem;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }

  /* ─── Section Cards ─── */
  .section-card {
    background: #111827;
    border: 1px solid #1E293B;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    height: 100%;
  }
  .section-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #38BDF8;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #1E293B;
  }

  /* ─── Input overrides ─── */
  div[data-baseweb="input"] input,
  div[data-baseweb="select"] div {
    background-color: #0F172A !important;
    border-color: #1E293B !important;
    color: #E2E8F0 !important;
    border-radius: 8px !important;
  }
  div[data-baseweb="input"] input:focus {
    border-color: #38BDF8 !important;
    box-shadow: 0 0 0 2px rgba(56,189,248,0.15) !important;
  }
  label { color: #94A3B8 !important; font-size: 0.82rem !important; }

  /* ─── Predict Button ─── */
  .stButton > button {
    background: linear-gradient(135deg, #0EA5E9, #2563EB) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
    padding: 0.75rem 1.5rem !important;
    transition: opacity 0.2s ease !important;
  }
  .stButton > button:hover { opacity: 0.88 !important; }

  /* ─── Result Cards ─── */
  .result-approved {
    background: linear-gradient(135deg, #052E16, #064E3B);
    border: 1px solid #059669;
    border-radius: 14px;
    padding: 2rem 2.5rem;
    text-align: center;
  }
  .result-rejected {
    background: linear-gradient(135deg, #1C0A0A, #450A0A);
    border: 1px solid #DC2626;
    border-radius: 14px;
    padding: 2rem 2.5rem;
    text-align: center;
  }
  .result-verdict {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    margin: 0.5rem 0;
  }
  .verdict-approved { color: #34D399; }
  .verdict-rejected { color: #F87171; }
  .result-sub { color: #94A3B8; font-size: 0.88rem; margin-top: 0.5rem; }

  /* ─── Confidence bar ─── */
  .conf-bar-wrap {
    background: #1E293B;
    border-radius: 99px;
    height: 8px;
    margin: 1.2rem auto 0 auto;
    max-width: 360px;
    overflow: hidden;
  }
  .conf-bar-fill-green {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, #059669, #34D399);
    transition: width 1s ease;
  }
  .conf-bar-fill-red {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, #DC2626, #F87171);
    transition: width 1s ease;
  }
  .conf-pct {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.4rem;
    font-weight: 700;
    margin-top: 1rem;
  }

  /* ─── Divider ─── */
  hr { border-color: #1E293B !important; margin: 1.5rem 0 !important; }

  /* ─── Metric row ─── */
  .meta-row {
    display: flex;
    gap: 1rem;
    justify-content: center;
    margin-top: 0.8rem;
    flex-wrap: wrap;
  }
  .meta-chip {
    background: rgba(255,255,255,0.04);
    border: 1px solid #1E293B;
    border-radius: 8px;
    padding: 6px 14px;
    font-size: 0.78rem;
    color: #94A3B8;
  }
  .meta-chip b { color: #CBD5E1; }
</style>
""", unsafe_allow_html=True)


# ==========================================
# LOAD MODEL & SCALER
# ==========================================
@st.cache_resource
def load_assets():
    model = joblib.load('loan_model_final.pkl')
    scaler = joblib.load('loan_scaler.pkl')
    return model, scaler

model, scaler = load_assets()


# ==========================================
# HERO HEADER
# ==========================================
st.markdown("""
<div class="hero">
  <div class="hero-badge">⚡ Powered by XGBoost</div>
  <p class="hero-title">Loan<span>IQ</span> — Approval Intelligence</p>
  <p class="hero-sub">Masukkan data nasabah untuk mendapatkan prediksi kelayakan pinjaman secara instan.</p>
  <div class="hero-stats">
    <div class="stat-item">
      <div class="stat-val">XGBoost</div>
      <div class="stat-label">Model Engine</div>
    </div>
    <div class="stat-item">
      <div class="stat-val">11</div>
      <div class="stat-label">Fitur Input</div>
    </div>
    <div class="stat-item">
      <div class="stat-val">Real-time</div>
      <div class="stat-label">Prediksi</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ==========================================
# INPUT FORM
# ==========================================
col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    st.markdown('<div class="section-card"><div class="section-label">👤 Profil Nasabah</div>', unsafe_allow_html=True)
    education = st.selectbox("Tingkat Edukasi", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Status Pekerjaan", ["No — Karyawan", "Yes — Wiraswasta"])
    no_of_dependents = st.number_input("Jumlah Tanggungan", min_value=0, max_value=10, value=0, help="Anggota keluarga yang ditanggung")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-card"><div class="section-label">💳 Data Finansial</div>', unsafe_allow_html=True)
    income_annum = st.number_input("Pendapatan Tahunan (Rp)", min_value=0, value=5_000_000, step=500_000)
    loan_amount = st.number_input("Jumlah Pinjaman (Rp)", min_value=0, value=15_000_000, step=1_000_000)
    loan_term = st.number_input("Tenor Pinjaman (Tahun)", min_value=1, max_value=20, value=5)
    cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, value=600, help="Skor kredit: 300 (buruk) → 900 (sangat baik)")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="section-card"><div class="section-label">🏠 Nilai Aset (Rp)</div>', unsafe_allow_html=True)
    residential_assets_value = st.number_input("Aset Properti", min_value=0, value=12_000_000, step=1_000_000)
    commercial_assets_value = st.number_input("Aset Komersial", min_value=0, value=2_000_000, step=500_000)
    luxury_assets_value = st.number_input("Aset Mewah", min_value=0, value=4_000_000, step=500_000)
    bank_asset_value = st.number_input("Tabungan / Deposito", min_value=0, value=3_000_000, step=500_000)
    st.markdown('</div>', unsafe_allow_html=True)


# ==========================================
# PREDIKSI
# ==========================================
st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button("🔍 Analisis Kelayakan Pinjaman", use_container_width=True)

if predict_btn:
    # Encode categorical
    edu_encoded = 1 if education == "Graduate" else 0
    emp_encoded = 1 if "Yes" in self_employed else 0

    input_data = pd.DataFrame({
        'no_of_dependents': [no_of_dependents],
        'education': [edu_encoded],
        'self_employed': [emp_encoded],
        'income_annum': [income_annum],
        'loan_amount': [loan_amount],
        'loan_term': [loan_term],
        'cibil_score': [cibil_score],
        'residential_assets_value': [residential_assets_value],
        'commercial_assets_value': [commercial_assets_value],
        'luxury_assets_value': [luxury_assets_value],
        'bank_asset_value': [bank_asset_value]
    })

    kolom_skala = [
        'no_of_dependents', 'income_annum', 'loan_amount', 'loan_term', 'cibil_score',
        'residential_assets_value', 'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value'
    ]

    input_scaled = input_data.copy()
    input_scaled[kolom_skala] = scaler.transform(input_data[kolom_skala])

    prediction = model.predict(input_scaled)
    prob = model.predict_proba(input_scaled)[0]

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── APPROVED ───
    if prediction[0] == 1:
        confidence = prob[1] * 100
        bar_width = f"{confidence:.0f}%"
        st.markdown(f"""
        <div class="result-approved">
          <div style="font-size:2.5rem">✅</div>
          <div class="result-verdict verdict-approved">DISETUJUI</div>
          <div class="result-sub">Nasabah dinilai layak dan berisiko rendah untuk mendapatkan pinjaman.</div>
          <div class="conf-pct" style="color:#34D399">{confidence:.1f}%</div>
          <div style="color:#64748B; font-size:0.75rem; letter-spacing:0.08em; text-transform:uppercase; margin-top:2px;">Tingkat keyakinan model</div>
          <div class="conf-bar-wrap">
            <div class="conf-bar-fill-green" style="width:{bar_width}"></div>
          </div>
          <div class="meta-row" style="margin-top:1.5rem;">
            <div class="meta-chip">CIBIL: <b>{cibil_score}</b></div>
            <div class="meta-chip">Tenor: <b>{loan_term} Thn</b></div>
            <div class="meta-chip">Tanggungan: <b>{no_of_dependents}</b></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ─── REJECTED ───
    else:
        risk = prob[0] * 100
        bar_width = f"{risk:.0f}%"
        st.markdown(f"""
        <div class="result-rejected">
          <div style="font-size:2.5rem">❌</div>
          <div class="result-verdict verdict-rejected">DITOLAK</div>
          <div class="result-sub">Nasabah teridentifikasi memiliki risiko gagal bayar yang tinggi.</div>
          <div class="conf-pct" style="color:#F87171">{risk:.1f}%</div>
          <div style="color:#64748B; font-size:0.75rem; letter-spacing:0.08em; text-transform:uppercase; margin-top:2px;">Tingkat risiko gagal bayar</div>
          <div class="conf-bar-wrap">
            <div class="conf-bar-fill-red" style="width:{bar_width}"></div>
          </div>
          <div class="meta-row" style="margin-top:1.5rem;">
            <div class="meta-chip">CIBIL: <b>{cibil_score}</b></div>
            <div class="meta-chip">Tenor: <b>{loan_term} Thn</b></div>
            <div class="meta-chip">Tanggungan: <b>{no_of_dependents}</b></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ─── Disclaimer ───
    st.markdown("""
    <p style="text-align:center; color:#334155; font-size:0.75rem; margin-top:1.5rem;">
      Hasil ini bersifat prediktif dan tidak menggantikan keputusan analis kredit.
    </p>
    """, unsafe_allow_html=True)