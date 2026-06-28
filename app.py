import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

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
# CUSTOM CSS
# ==========================================
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #070B14;
    color: #E2E8F0;
  }

  #MainMenu, footer, header { visibility: hidden; }

  .block-container {
    padding: 1.5rem 2.5rem 4rem 2.5rem;
    max-width: 1280px;
  }

  /* ── HERO ── */
  @keyframes fadeSlideDown {
    from { opacity: 0; transform: translateY(-18px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  @keyframes pulseGlow {
    0%, 100% { box-shadow: 0 0 0px rgba(56,189,248,0); }
    50%       { box-shadow: 0 0 40px rgba(56,189,248,0.08); }
  }

  .hero {
    background: linear-gradient(135deg, #0D1521 0%, #0F1B2D 50%, #0A1220 100%);
    border: 1px solid #1A2744;
    border-radius: 20px;
    padding: 2.8rem 3rem 2.2rem 3rem;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
    animation: fadeSlideDown 0.6s ease both, pulseGlow 4s ease-in-out infinite;
  }
  .hero::before {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(56,189,248,0.07) 0%, transparent 65%);
    border-radius: 50%;
    pointer-events: none;
  }
  .hero::after {
    content: '';
    position: absolute;
    bottom: -60px; left: 30%;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(99,102,241,0.05) 0%, transparent 65%);
    border-radius: 50%;
    pointer-events: none;
  }
  .hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(56,189,248,0.08);
    border: 1px solid rgba(56,189,248,0.25);
    color: #38BDF8;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    padding: 5px 14px;
    border-radius: 99px;
    margin-bottom: 1.1rem;
  }
  .hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.6rem;
    font-weight: 700;
    color: #F8FAFC;
    line-height: 1.15;
    margin: 0 0 0.6rem 0;
    letter-spacing: -0.02em;
  }
  .hero-title .accent { color: #38BDF8; }
  .hero-title .dim { color: #475569; font-weight: 500; font-size: 1.8rem; }
  .hero-sub {
    color: #64748B;
    font-size: 0.92rem;
    margin: 0;
    max-width: 560px;
    line-height: 1.6;
  }
  .hero-stats {
    display: flex;
    gap: 2.5rem;
    margin-top: 2rem;
    padding-top: 1.5rem;
    border-top: 1px solid #1A2744;
  }
  .stat-item { line-height: 1.2; }
  .stat-val {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #38BDF8;
  }
  .stat-label {
    font-size: 0.68rem;
    color: #334155;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 2px;
  }
  .stat-divider {
    width: 1px;
    background: #1A2744;
    align-self: stretch;
  }

  /* ── SECTION LABELS ── */
  .sec-label {
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #38BDF8;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .sec-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, #1A2744, transparent);
  }

  /* ── INPUT CARDS ── */
  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  .input-card {
    background: #0D1521;
    border: 1px solid #1A2744;
    border-radius: 16px;
    padding: 1.6rem 1.6rem 0.8rem 1.6rem;
    transition: border-color 0.25s ease, box-shadow 0.25s ease;
    animation: fadeUp 0.5s ease both;
  }
  .input-card:hover {
    border-color: rgba(56,189,248,0.25);
    box-shadow: 0 4px 24px rgba(56,189,248,0.05);
  }

  /* ── STREAMLIT WIDGET OVERRIDES ── */
  div[data-baseweb="input"] > div,
  div[data-baseweb="select"] > div:first-child {
    background-color: #070B14 !important;
    border-color: #1A2744 !important;
    border-radius: 10px !important;
  }
  div[data-baseweb="input"] input { color: #E2E8F0 !important; }
  div[data-baseweb="input"] > div:focus-within {
    border-color: #38BDF8 !important;
    box-shadow: 0 0 0 3px rgba(56,189,248,0.12) !important;
  }
  label, .stSelectbox label, .stNumberInput label {
    color: #64748B !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.01em !important;
  }
  .stSelectbox [data-baseweb="select"] span { color: #CBD5E1 !important; }

  /* ── SUMMARY CARD ── */
  .summary-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 0.75rem;
    margin-top: 0.5rem;
  }
  .summary-item {
    background: #070B14;
    border: 1px solid #1A2744;
    border-radius: 10px;
    padding: 0.7rem 1rem;
  }
  .summary-key {
    font-size: 0.68rem;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 3px;
  }
  .summary-val {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.95rem;
    font-weight: 600;
    color: #CBD5E1;
  }

  /* ── PREDICT BUTTON ── */
  .stButton > button {
    background: linear-gradient(135deg, #0EA5E9 0%, #6366F1 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    padding: 0.85rem 2rem !important;
    transition: opacity 0.2s ease, transform 0.15s ease !important;
    box-shadow: 0 4px 20px rgba(14,165,233,0.25) !important;
  }
  .stButton > button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 28px rgba(14,165,233,0.35) !important;
  }
  .stButton > button:active { transform: translateY(0) !important; }

  /* ── RESULT ── */
  @keyframes resultReveal {
    from { opacity: 0; transform: scale(0.97); }
    to   { opacity: 1; transform: scale(1); }
  }
  .result-wrap {
    animation: resultReveal 0.5s cubic-bezier(0.34,1.56,0.64,1) both;
  }
  .result-approved {
    background: linear-gradient(145deg, #041F0F 0%, #052E16 60%, #053220 100%);
    border: 1px solid #065F46;
    border-radius: 20px;
    padding: 2.5rem 3rem;
    text-align: center;
    box-shadow: 0 0 60px rgba(16,185,129,0.08), inset 0 1px 0 rgba(52,211,153,0.1);
  }
  .result-rejected {
    background: linear-gradient(145deg, #120404 0%, #1C0707 60%, #200808 100%);
    border: 1px solid #7F1D1D;
    border-radius: 20px;
    padding: 2.5rem 3rem;
    text-align: center;
    box-shadow: 0 0 60px rgba(239,68,68,0.07), inset 0 1px 0 rgba(248,113,113,0.08);
  }
  .result-icon { font-size: 3rem; margin-bottom: 0.5rem; display: block; }
  .result-status {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
  }
  .status-approved { color: #34D399; }
  .status-rejected { color: #F87171; }
  .result-verdict {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.6rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin: 0 0 0.5rem 0;
    line-height: 1;
  }
  .verdict-approved { color: #ECFDF5; }
  .verdict-rejected { color: #FEF2F2; }
  .result-desc { color: #94A3B8; font-size: 0.88rem; line-height: 1.6; max-width: 400px; margin: 0 auto; }

  .conf-number {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3.2rem;
    font-weight: 700;
    margin: 1.5rem 0 0 0;
    line-height: 1;
  }
  .conf-label {
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #475569;
    margin-top: 4px;
  }
  .conf-bar-track {
    background: rgba(255,255,255,0.06);
    border-radius: 99px;
    height: 6px;
    max-width: 320px;
    margin: 1rem auto 0;
    overflow: hidden;
  }
  .conf-bar-fill-g {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, #059669, #34D399, #6EE7B7);
  }
  .conf-bar-fill-r {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, #991B1B, #DC2626, #F87171);
  }

  .chip-row {
    display: flex;
    gap: 0.75rem;
    justify-content: center;
    flex-wrap: wrap;
    margin-top: 1.8rem;
    padding-top: 1.5rem;
    border-top: 1px solid rgba(255,255,255,0.05);
  }
  .chip {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 8px;
    padding: 6px 16px;
    font-size: 0.78rem;
    color: #94A3B8;
  }
  .chip b { color: #E2E8F0; font-weight: 600; }

  .disclaimer {
    text-align: center;
    color: #1E293B;
    font-size: 0.73rem;
    margin-top: 1.2rem;
  }

  /* ── DIVIDER ── */
  hr { border-color: #1A2744 !important; margin: 1.8rem 0 !important; }
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
# HELPER: GAUGE CHART
# ==========================================
def make_gauge(score, min_val=300, max_val=900):
    pct = (score - min_val) / (max_val - min_val)
    if pct < 0.33:
        bar_color = "#EF4444"
        label = "Buruk"
    elif pct < 0.66:
        bar_color = "#F59E0B"
        label = "Cukup"
    else:
        bar_color = "#10B981"
        label = "Sangat Baik"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={
            'font': {'size': 36, 'color': '#E2E8F0', 'family': 'Space Grotesk'},
        },
        title={
            'text': label,
            'font': {'size': 13, 'color': bar_color, 'family': 'Inter'}
        },
        gauge={
            'axis': {
                'range': [min_val, max_val],
                'tickwidth': 1,
                'tickcolor': 'rgba(71,85,105,0.4)',
                'tickfont': {'color': '#334155', 'size': 9},
                'nticks': 5
            },
            'bar': {'color': bar_color, 'thickness': 0.22},
            'bgcolor': 'rgba(0,0,0,0)',
            'borderwidth': 0,
            'steps': [
                {'range': [300, 500], 'color': 'rgba(239,68,68,0.08)'},
                {'range': [500, 700], 'color': 'rgba(245,158,11,0.08)'},
                {'range': [700, 900], 'color': 'rgba(16,185,129,0.08)'},
            ],
            'threshold': {
                'line': {'color': bar_color, 'width': 2},
                'thickness': 0.75,
                'value': score
            }
        }
    ))

    fig.update_layout(
        height=210,
        margin=dict(t=20, b=10, l=20, r=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#64748B', 'family': 'Inter'},
    )
    return fig


# ==========================================
# HERO HEADER
# ==========================================
st.markdown("""
<div class="hero">
  <div class="hero-badge">⚡ Powered by XGBoost</div>
  <div class="hero-title">
    Loan<span class="accent">IQ</span>
    <span class="dim"> — Credit Intelligence</span>
  </div>
  <p class="hero-sub">
    Platform analisis kelayakan kredit berbasis machine learning. Masukkan data nasabah untuk mendapatkan keputusan dan skor risiko secara instan.
  </p>
  <div class="hero-stats">
    <div class="stat-item">
      <div class="stat-val">XGBoost</div>
      <div class="stat-label">Model Engine</div>
    </div>
    <div class="stat-divider"></div>
    <div class="stat-item">
      <div class="stat-val">11</div>
      <div class="stat-label">Fitur Analisis</div>
    </div>
    <div class="stat-divider"></div>
    <div class="stat-item">
      <div class="stat-val">Real-time</div>
      <div class="stat-label">Prediksi Instan</div>
    </div>
    <div class="stat-divider"></div>
    <div class="stat-item">
      <div class="stat-val">Binary</div>
      <div class="stat-label">Klasifikasi</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ==========================================
# INPUT FORM — 3 KOLOM
# ==========================================
col1, col2, col3 = st.columns([1, 1, 1], gap="medium")

with col1:
    st.markdown('<div class="input-card"><div class="sec-label">👤 Profil Nasabah</div>', unsafe_allow_html=True)
    education      = st.selectbox("Tingkat Edukasi", ["Graduate", "Not Graduate"])
    self_employed  = st.selectbox("Status Pekerjaan", ["No — Karyawan", "Yes — Wiraswasta"])
    no_of_dependents = st.number_input("Jumlah Tanggungan", min_value=0, max_value=10, value=0,
                                        help="Anggota keluarga yang ditanggung secara finansial")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="input-card"><div class="sec-label">💳 Data Finansial</div>', unsafe_allow_html=True)
    income_annum = st.number_input("Pendapatan Tahunan (Rp)", min_value=0, value=5_000_000, step=500_000)
    loan_amount  = st.number_input("Jumlah Pinjaman (Rp)",    min_value=0, value=15_000_000, step=1_000_000)
    loan_term    = st.number_input("Tenor Pinjaman (Tahun)",  min_value=1, max_value=20, value=5)
    cibil_score  = st.number_input("CIBIL Score",             min_value=300, max_value=900, value=600,
                                    help="Skor kredit: 300 (buruk) → 900 (sangat baik)")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="input-card"><div class="sec-label">🏠 Portofolio Aset (Rp)</div>', unsafe_allow_html=True)
    residential_assets_value = st.number_input("Properti / Hunian",      min_value=0, value=12_000_000, step=1_000_000)
    commercial_assets_value  = st.number_input("Aset Komersial",          min_value=0, value=2_000_000,  step=500_000)
    luxury_assets_value      = st.number_input("Aset Mewah",              min_value=0, value=4_000_000,  step=500_000)
    bank_asset_value         = st.number_input("Tabungan / Deposito",     min_value=0, value=3_000_000,  step=500_000)
    st.markdown('</div>', unsafe_allow_html=True)


# ==========================================
# CIBIL GAUGE + RINGKASAN INPUT
# ==========================================
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="sec-label">📋 Ringkasan & Analisis Input</div>', unsafe_allow_html=True)

gauge_col, summary_col = st.columns([1, 2], gap="large")

with gauge_col:
    st.markdown("**CIBIL Score**", help="Visualisasi posisi skor kredit nasabah")
    fig_gauge = make_gauge(cibil_score)
    st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})

with summary_col:
    total_assets = residential_assets_value + commercial_assets_value + luxury_assets_value + bank_asset_value
    dti_ratio = (loan_amount / income_annum * 100) if income_annum > 0 else 0
    edu_label = education
    emp_label = "Wiraswasta" if "Yes" in self_employed else "Karyawan"

    st.markdown(f"""
    <div class="summary-grid">
      <div class="summary-item">
        <div class="summary-key">Pendapatan / Tahun</div>
        <div class="summary-val">Rp {income_annum:,.0f}</div>
      </div>
      <div class="summary-item">
        <div class="summary-key">Jumlah Pinjaman</div>
        <div class="summary-val">Rp {loan_amount:,.0f}</div>
      </div>
      <div class="summary-item">
        <div class="summary-key">Tenor</div>
        <div class="summary-val">{loan_term} Tahun</div>
      </div>
      <div class="summary-item">
        <div class="summary-key">Total Aset</div>
        <div class="summary-val">Rp {total_assets:,.0f}</div>
      </div>
      <div class="summary-item">
        <div class="summary-key">Rasio Pinjaman / Income</div>
        <div class="summary-val">{dti_ratio:.1f}%</div>
      </div>
      <div class="summary-item">
        <div class="summary-key">Edukasi</div>
        <div class="summary-val">{edu_label}</div>
      </div>
      <div class="summary-item">
        <div class="summary-key">Status Kerja</div>
        <div class="summary-val">{emp_label}</div>
      </div>
      <div class="summary-item">
        <div class="summary-key">Tanggungan</div>
        <div class="summary-val">{no_of_dependents} orang</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# TOMBOL PREDIKSI
# ==========================================
st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button("🔍  Analisis Kelayakan Pinjaman", use_container_width=True)


# ==========================================
# PREDIKSI & HASIL
# ==========================================
if predict_btn:
    edu_encoded = 1 if education == "Graduate" else 0
    emp_encoded = 1 if "Yes" in self_employed else 0

    input_data = pd.DataFrame({
        'no_of_dependents':          [no_of_dependents],
        'education':                 [edu_encoded],
        'self_employed':             [emp_encoded],
        'income_annum':              [income_annum],
        'loan_amount':               [loan_amount],
        'loan_term':                 [loan_term],
        'cibil_score':               [cibil_score],
        'residential_assets_value':  [residential_assets_value],
        'commercial_assets_value':   [commercial_assets_value],
        'luxury_assets_value':       [luxury_assets_value],
        'bank_asset_value':          [bank_asset_value]
    })

    kolom_skala = [
        'no_of_dependents', 'income_annum', 'loan_amount', 'loan_term', 'cibil_score',
        'residential_assets_value', 'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value'
    ]
    input_scaled = input_data.copy()
    input_scaled[kolom_skala] = scaler.transform(input_data[kolom_skala])

    prediction = model.predict(input_scaled)
    prob       = model.predict_proba(input_scaled)[0]

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-label">📊 Hasil Analisis Kredit</div>', unsafe_allow_html=True)

    res_col, chart_col = st.columns([1.1, 1], gap="large")

    with res_col:
        if prediction[0] == 1:
            confidence = prob[1] * 100
            bar_pct    = f"{confidence:.0f}%"
            st.markdown(f"""
            <div class="result-wrap">
              <div class="result-approved">
                <span class="result-icon">✅</span>
                <div class="result-status status-approved">Keputusan Kredit</div>
                <div class="result-verdict verdict-approved">Disetujui</div>
                <p class="result-desc">Nasabah dinilai layak dan memiliki profil risiko yang rendah berdasarkan analisis 11 variabel.</p>
                <div class="conf-number" style="color:#34D399">{confidence:.1f}%</div>
                <div class="conf-label">Tingkat Keyakinan Model</div>
                <div class="conf-bar-track">
                  <div class="conf-bar-fill-g" style="width:{bar_pct}"></div>
                </div>
                <div class="chip-row">
                  <div class="chip">CIBIL: <b>{cibil_score}</b></div>
                  <div class="chip">Tenor: <b>{loan_term} Thn</b></div>
                  <div class="chip">DTI: <b>{dti_ratio:.1f}%</b></div>
                  <div class="chip">Tanggungan: <b>{no_of_dependents}</b></div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            risk    = prob[0] * 100
            bar_pct = f"{risk:.0f}%"
            st.markdown(f"""
            <div class="result-wrap">
              <div class="result-rejected">
                <span class="result-icon">❌</span>
                <div class="result-status status-rejected">Keputusan Kredit</div>
                <div class="result-verdict verdict-rejected">Ditolak</div>
                <p class="result-desc">Nasabah teridentifikasi memiliki profil risiko gagal bayar yang tinggi berdasarkan analisis model.</p>
                <div class="conf-number" style="color:#F87171">{risk:.1f}%</div>
                <div class="conf-label">Tingkat Risiko Gagal Bayar</div>
                <div class="conf-bar-track">
                  <div class="conf-bar-fill-r" style="width:{bar_pct}"></div>
                </div>
                <div class="chip-row">
                  <div class="chip">CIBIL: <b>{cibil_score}</b></div>
                  <div class="chip">Tenor: <b>{loan_term} Thn</b></div>
                  <div class="chip">DTI: <b>{dti_ratio:.1f}%</b></div>
                  <div class="chip">Tanggungan: <b>{no_of_dependents}</b></div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    with chart_col:
        # ── Donut Probability Chart ──
        approved_pct = prob[1] * 100
        rejected_pct = prob[0] * 100

        fig_donut = go.Figure(data=[go.Pie(
            labels=['Disetujui', 'Ditolak'],
            values=[approved_pct, rejected_pct],
            hole=0.68,
            marker=dict(
                colors=['#10B981', '#EF4444'],
                line=dict(color='#070B14', width=3)
            ),
            textinfo='none',
            hovertemplate='%{label}: %{value:.1f}%<extra></extra>'
        )])

        center_text = f"{approved_pct:.0f}%" if prediction[0] == 1 else f"{rejected_pct:.0f}%"
        center_color = "#10B981" if prediction[0] == 1 else "#EF4444"
        center_label = "Approved" if prediction[0] == 1 else "Risk Score"

        fig_donut.add_annotation(
            text=f'<span style="font-family:Space Grotesk;font-size:28px;font-weight:700;fill:{center_color}">{center_text}</span>',
            x=0.5, y=0.55, showarrow=False,
            font=dict(size=28, color=center_color, family="Space Grotesk"),
        )
        fig_donut.add_annotation(
            text=center_label,
            x=0.5, y=0.38, showarrow=False,
            font=dict(size=11, color='#475569', family="Inter"),
        )

        fig_donut.update_layout(
            height=260,
            margin=dict(t=20, b=10, l=20, r=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=True,
            legend=dict(
                orientation='h',
                yanchor='bottom', y=-0.08,
                xanchor='center', x=0.5,
                font=dict(color='#64748B', size=11, family='Inter'),
            )
        )
        st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})

        # ── Horizontal bar: breakdown aset ──
        aset_labels = ['Properti', 'Komersial', 'Mewah', 'Tabungan']
        aset_values = [residential_assets_value, commercial_assets_value,
                       luxury_assets_value, bank_asset_value]
        aset_colors = ['#0EA5E9', '#6366F1', '#8B5CF6', '#EC4899']

        fig_bar = go.Figure()
        for lbl, val, clr in zip(aset_labels, aset_values, aset_colors):
            fig_bar.add_trace(go.Bar(
                name=lbl,
                x=[val],
                y=[lbl],
                orientation='h',
                marker=dict(color=clr, line=dict(width=0)),
                hovertemplate=f'{lbl}: Rp %{{x:,.0f}}<extra></extra>'
            ))

        fig_bar.update_layout(
            title=dict(text='Komposisi Aset Nasabah', font=dict(size=11, color='#475569', family='Inter'), x=0),
            barmode='group',
            height=210,
            margin=dict(t=32, b=10, l=10, r=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(tickfont=dict(color='#64748B', size=10, family='Inter'), tickcolor='transparent'),
            showlegend=False,
            bargap=0.3,
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<p class="disclaimer">Hasil bersifat prediktif dan tidak menggantikan keputusan analis kredit yang berwenang.</p>', unsafe_allow_html=True)
