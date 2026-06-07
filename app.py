"""
app.py — Explainable Crop Suitability Prediction System
========================================================
Redesigned UI: 4 consolidated sections with premium aesthetic.
All backend connections (src/) remain unchanged.
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="CropAI — Crop Intelligence",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  DESIGN SYSTEM — Luxury Organic Dark Theme
#  Aesthetic: editorial agricultural journal meets sci-fi dashboard
#  Fonts: Playfair Display (serif display) + Space Grotesk (clean UI)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;0,900;1,400;1,700&family=Space+Grotesk:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

/* ── Root tokens ── */
:root {
  --ink:        #0b0f0a;
  --ink-2:      #111a0f;
  --ink-3:      #182414;
  --sage:       #3d6b4f;
  --sage-light: #5a9e70;
  --mint:       #7fffd4;
  --gold:       #d4a843;
  --gold-pale:  #f0d080;
  --parchment:  #f5efe0;
  --fog:        rgba(255,255,255,0.06);
  --fog-2:      rgba(255,255,255,0.10);
  --fog-3:      rgba(255,255,255,0.15);
  --glow-green: rgba(127,255,212,0.12);
  --glow-gold:  rgba(212,168,67,0.15);
  --border-sub: rgba(127,255,212,0.15);
  --border-med: rgba(127,255,212,0.28);
  --border-str: rgba(127,255,212,0.55);
  --text-pri:   #e8f0e4;
  --text-sec:   #9db89a;
  --text-mute:  #617a5e;
  --radius-sm:  8px;
  --radius-md:  16px;
  --radius-lg:  24px;
  --radius-xl:  36px;
}

/* ── Global reset ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"], .stApp {
  font-family: 'Space Grotesk', sans-serif;
  background: var(--ink) !important;
  color: var(--text-pri) !important;
}

/* ── Animated grain texture overlay ── */
.stApp::before {
  content: '';
  position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.035'/%3E%3C/svg%3E");
  opacity: 0.4;
}

/* ── Radial ambient glows ── */
.stApp::after {
  content: '';
  position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background:
    radial-gradient(ellipse 700px 500px at 0% 0%, rgba(63,120,78,0.18) 0%, transparent 70%),
    radial-gradient(ellipse 600px 400px at 100% 100%, rgba(212,168,67,0.10) 0%, transparent 70%),
    radial-gradient(ellipse 400px 300px at 60% 30%, rgba(127,255,212,0.06) 0%, transparent 60%);
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #090d08 0%, #0e1a0c 60%, #0b1309 100%) !important;
  border-right: 1px solid var(--border-sub) !important;
  box-shadow: 4px 0 40px rgba(0,0,0,0.5) !important;
}
section[data-testid="stSidebar"] > div { padding-top: 0 !important; }
section[data-testid="stSidebar"] * { color: var(--text-pri) !important; }
section[data-testid="stSidebar"] .stRadio label {
  font-family: 'Space Grotesk', sans-serif !important;
  font-size: 0.88rem !important;
  letter-spacing: 0.02em !important;
  cursor: pointer;
  transition: color 0.2s;
}
section[data-testid="stSidebar"] .stRadio label:hover { color: var(--mint) !important; }
[data-testid="stRadio"] > div { gap: 0 !important; }
[data-testid="stRadio"] div[role="radio"] {
  padding: 0.55rem 1rem;
  border-radius: var(--radius-sm);
  margin: 2px 0;
  transition: background 0.18s;
}
[data-testid="stRadio"] div[role="radio"]:hover { background: var(--fog); }
[data-testid="stRadio"] div[role="radio"][aria-checked="true"] {
  background: linear-gradient(90deg, rgba(127,255,212,0.12), transparent) !important;
  border-left: 2px solid var(--mint) !important;
}

/* ── Main container ── */
.main .block-container {
  padding: 2rem 2.5rem 4rem !important;
  max-width: 1400px !important;
  position: relative; z-index: 1;
}

/* ── Typography ── */
.display-serif {
  font-family: 'Playfair Display', Georgia, serif;
  font-weight: 900;
  line-height: 1.05;
  letter-spacing: -0.02em;
}
.mono { font-family: 'Space Mono', monospace; }

/* ── Hero banner ── */
.hero-wrap {
  position: relative;
  padding: 3.5rem 3rem 3rem;
  border-radius: var(--radius-xl);
  overflow: hidden;
  margin-bottom: 2.5rem;
  border: 1px solid var(--border-sub);
}
.hero-wrap::before {
  content: '';
  position: absolute; inset: 0;
  background:
    linear-gradient(135deg, rgba(30,60,35,0.6) 0%, rgba(10,20,12,0.8) 60%, rgba(20,12,5,0.7) 100%),
    repeating-linear-gradient(
      45deg,
      transparent,
      transparent 20px,
      rgba(127,255,212,0.015) 20px,
      rgba(127,255,212,0.015) 21px
    );
}
.hero-wrap::after {
  content: '';
  position: absolute;
  top: -80px; right: -80px;
  width: 400px; height: 400px;
  background: radial-gradient(circle, rgba(127,255,212,0.08) 0%, transparent 65%);
  pointer-events: none;
}
.hero-content { position: relative; z-index: 2; }
.hero-eyebrow {
  font-family: 'Space Mono', monospace;
  font-size: 0.72rem;
  letter-spacing: 0.25em;
  text-transform: uppercase;
  color: var(--mint);
  margin-bottom: 0.8rem;
  display: flex; align-items: center; gap: 0.6rem;
}
.hero-eyebrow::before {
  content: '';
  display: inline-block; width: 28px; height: 1px;
  background: var(--mint);
}
.hero-title {
  font-family: 'Playfair Display', serif;
  font-size: clamp(2.4rem, 4.5vw, 3.8rem);
  font-weight: 900;
  line-height: 1.05;
  letter-spacing: -0.025em;
  color: var(--parchment);
  margin: 0;
}
.hero-title em {
  font-style: italic;
  background: linear-gradient(135deg, var(--mint), var(--gold-pale));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text;
}
.hero-sub {
  margin-top: 1rem;
  font-size: 1rem;
  color: var(--text-sec);
  line-height: 1.65;
  max-width: 560px;
}
.hero-pills {
  display: flex; flex-wrap: wrap; gap: 0.5rem;
  margin-top: 1.5rem;
}
.hero-pill {
  font-family: 'Space Mono', monospace;
  font-size: 0.7rem;
  letter-spacing: 0.08em;
  color: var(--text-sec);
  border: 1px solid var(--border-sub);
  border-radius: 99px;
  padding: 0.25rem 0.8rem;
  background: var(--fog);
  text-transform: uppercase;
}

/* ── Section label ── */
.sec-label {
  font-family: 'Space Mono', monospace;
  font-size: 0.65rem;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: var(--mint);
  margin-bottom: 1.2rem;
  display: flex; align-items: center; gap: 0.5rem;
}
.sec-label::after {
  content: '';
  flex: 1; height: 1px;
  background: linear-gradient(90deg, var(--border-sub), transparent);
}

/* ── Stat cards ── */
.stat-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 1rem; margin-bottom: 2rem; }
.stat-card {
  position: relative; overflow: hidden;
  background: var(--fog);
  border: 1px solid var(--border-sub);
  border-radius: var(--radius-md);
  padding: 1.4rem 1.6rem;
  transition: border-color 0.25s, transform 0.2s;
}
.stat-card:hover { border-color: var(--border-med); transform: translateY(-2px); }
.stat-card::before {
  content: ''; position: absolute;
  top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, var(--mint), var(--gold));
  opacity: 0; transition: opacity 0.25s;
}
.stat-card:hover::before { opacity: 1; }
.stat-num {
  font-family: 'Playfair Display', serif;
  font-size: 2.4rem; font-weight: 900;
  color: var(--mint); line-height: 1;
}
.stat-lbl {
  font-size: 0.72rem; font-weight: 500;
  text-transform: uppercase; letter-spacing: 0.1em;
  color: var(--text-mute); margin-top: 0.35rem;
}
.stat-icon {
  position: absolute; top: 1.2rem; right: 1.4rem;
  font-size: 1.4rem; opacity: 0.5;
}

/* ── Feature pills row ── */
.feat-row {
  display: flex; flex-wrap: wrap; gap: 0.75rem;
  margin-bottom: 2.5rem;
}
.feat-pill {
  display: flex; align-items: center; gap: 0.55rem;
  background: var(--fog);
  border: 1px solid var(--border-sub);
  border-radius: var(--radius-md);
  padding: 0.8rem 1.2rem;
  font-size: 0.85rem; font-weight: 500;
  color: var(--text-sec);
  transition: all 0.2s;
  cursor: default;
}
.feat-pill:hover {
  background: var(--glow-green); border-color: var(--border-med); color: var(--text-pri);
}
.feat-pill .pill-icon { font-size: 1.1rem; }

/* ── Glassmorphic panel ── */
.glass-panel {
  background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
  border: 1px solid var(--border-sub);
  border-radius: var(--radius-lg);
  padding: 2rem 2.2rem;
  backdrop-filter: blur(12px);
  margin-bottom: 1.5rem;
  position: relative; overflow: hidden;
}
.glass-panel::before {
  content: ''; position: absolute;
  inset: 0; border-radius: inherit;
  background: linear-gradient(135deg, rgba(127,255,212,0.03) 0%, transparent 60%);
  pointer-events: none;
}

/* ── Prediction result ── */
.pred-hero {
  position: relative; overflow: hidden;
  background: linear-gradient(135deg, rgba(30,70,45,0.5), rgba(20,14,5,0.6));
  border: 1px solid var(--border-med);
  border-radius: var(--radius-xl);
  padding: 2.5rem;
  text-align: center;
}
.pred-hero::before {
  content: ''; position: absolute; inset: 0;
  background: radial-gradient(ellipse 60% 40% at 50% 0%, rgba(127,255,212,0.1), transparent);
}
.pred-crop-name {
  font-family: 'Playfair Display', serif;
  font-size: clamp(2.8rem, 5vw, 4.5rem);
  font-weight: 900; font-style: italic;
  color: var(--parchment);
  text-transform: capitalize;
  position: relative; z-index: 1;
  letter-spacing: -0.02em;
  text-shadow: 0 0 40px rgba(127,255,212,0.3);
}
.pred-badge {
  display: inline-flex; align-items: center; gap: 0.5rem;
  background: rgba(212,168,67,0.15);
  border: 1px solid rgba(212,168,67,0.4);
  border-radius: 99px;
  padding: 0.4rem 1.2rem;
  font-family: 'Space Mono', monospace;
  font-size: 0.85rem; font-weight: 700;
  color: var(--gold-pale);
  margin-top: 0.7rem; position: relative; z-index: 1;
}
.pred-label {
  font-family: 'Space Mono', monospace;
  font-size: 0.7rem; letter-spacing: 0.2em;
  text-transform: uppercase; color: var(--text-mute);
  margin-bottom: 0.5rem; position: relative; z-index: 1;
}

/* ── Progress bars ── */
.rank-item { margin: 0.7rem 0; }
.rank-head {
  display: flex; justify-content: space-between; align-items: center;
  margin-bottom: 4px;
}
.rank-crop {
  font-size: 0.88rem; font-weight: 600;
  color: var(--text-pri); text-transform: capitalize;
}
.rank-pct {
  font-family: 'Space Mono', monospace;
  font-size: 0.78rem; color: var(--mint);
}
.rank-track {
  background: rgba(255,255,255,0.06);
  border-radius: 99px; height: 6px; overflow: hidden;
}
.rank-fill {
  height: 6px; border-radius: 99px;
  background: linear-gradient(90deg, var(--sage-light), var(--mint));
  transition: width 0.8s cubic-bezier(.34,1.56,.64,1);
}
.rank-fill.gold { background: linear-gradient(90deg, var(--gold), var(--gold-pale)); }

/* ── Reason cards ── */
.reason-card {
  display: flex; align-items: flex-start; gap: 0.7rem;
  padding: 0.7rem 1rem; margin: 0.4rem 0;
  border-radius: var(--radius-sm);
  font-size: 0.87rem; line-height: 1.5;
  transition: background 0.15s;
}
.reason-card.pos {
  background: rgba(127,255,212,0.06);
  border-left: 2px solid rgba(127,255,212,0.5);
  color: var(--text-sec);
}
.reason-card.pos:hover { background: rgba(127,255,212,0.1); }
.reason-card.cau {
  background: rgba(212,168,67,0.07);
  border-left: 2px solid rgba(212,168,67,0.4);
  color: var(--text-sec);
}
.reason-card .rc-icon { font-size: 0.9rem; flex-shrink: 0; margin-top: 1px; }

/* ── Model performance banner ── */
.model-banner {
  display: flex; align-items: center; justify-content: space-between;
  flex-wrap: wrap; gap: 1.5rem;
  background: linear-gradient(135deg, rgba(30,65,40,0.4), rgba(15,20,10,0.6));
  border: 1px solid var(--border-med);
  border-radius: var(--radius-lg);
  padding: 1.8rem 2.2rem;
  margin-bottom: 2rem;
}
.model-name-display {
  font-family: 'Playfair Display', serif;
  font-size: 2rem; font-weight: 700;
  color: var(--parchment);
}
.model-crown { font-size: 1.4rem; margin-right: 0.5rem; }
.metric-pill-row { display: flex; flex-wrap: wrap; gap: 0.6rem; }
.metric-pill {
  background: var(--fog-2); border: 1px solid var(--border-sub);
  border-radius: var(--radius-sm); padding: 0.5rem 0.9rem;
  text-align: center; min-width: 80px;
}
.metric-pill-val {
  font-family: 'Space Mono', monospace;
  font-size: 1rem; font-weight: 700; color: var(--mint);
}
.metric-pill-lbl {
  font-size: 0.62rem; text-transform: uppercase;
  letter-spacing: 0.1em; color: var(--text-mute);
  margin-top: 1px;
}

/* ── Train button ── */
.stButton > button {
  font-family: 'Space Grotesk', sans-serif !important;
  font-size: 0.9rem !important; font-weight: 600 !important;
  letter-spacing: 0.04em !important;
  background: linear-gradient(135deg, #2a5e3a 0%, #3d8a52 50%, #2a5e3a 100%) !important;
  background-size: 200% 200% !important;
  color: var(--mint) !important;
  border: 1px solid var(--border-med) !important;
  border-radius: var(--radius-md) !important;
  padding: 0.7rem 2rem !important;
  transition: all 0.3s !important;
  box-shadow: 0 4px 20px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.08) !important;
}
.stButton > button:hover {
  background-position: right center !important;
  border-color: var(--border-str) !important;
  box-shadow: 0 6px 28px rgba(0,0,0,0.4), 0 0 20px rgba(127,255,212,0.12) !important;
  transform: translateY(-1px);
  color: white !important;
}
.stButton > button:active { transform: translateY(0); }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  background: transparent !important;
  border-bottom: 1px solid var(--border-sub) !important;
  gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
  font-family: 'Space Grotesk', sans-serif !important;
  font-size: 0.82rem !important; font-weight: 500 !important;
  letter-spacing: 0.05em !important;
  color: var(--text-mute) !important;
  background: transparent !important;
  border: none !important; border-radius: 0 !important;
  padding: 0.75rem 1.4rem !important;
  transition: color 0.2s !important;
  text-transform: uppercase;
}
.stTabs [data-baseweb="tab"]:hover { color: var(--text-sec) !important; }
.stTabs [aria-selected="true"] {
  color: var(--mint) !important;
  border-bottom: 2px solid var(--mint) !important;
  background: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.5rem !important; }

/* ── Sliders ── */
.stSlider [data-baseweb="slider"] { padding: 0 !important; }
[data-testid="stSlider"] > label {
  font-size: 0.8rem !important; font-weight: 500 !important;
  color: var(--text-mute) !important; letter-spacing: 0.04em !important;
  text-transform: uppercase !important;
}
[data-testid="stThumbValue"] {
  background: var(--sage) !important;
  color: white !important; font-family: 'Space Mono', monospace !important;
  font-size: 0.72rem !important;
}
div[data-baseweb="slider"] div[role="slider"] {
  background: var(--mint) !important;
  border: 2px solid var(--ink) !important;
  box-shadow: 0 0 8px rgba(127,255,212,0.5) !important;
}
[data-testid="stSlider"] [data-testid="stMarkdownContainer"] p {
  font-size: 0.8rem !important; color: var(--text-mute) !important;
}

/* ── Form ── */
.stForm { border: none !important; }
[data-testid="stFormSubmitButton"] > button {
  width: 100%; font-size: 1rem !important;
  padding: 0.9rem !important;
  background: linear-gradient(135deg, rgba(30,80,50,0.8), rgba(20,50,30,0.9)) !important;
  border: 1px solid var(--border-med) !important;
  border-radius: var(--radius-lg) !important;
  color: var(--mint) !important;
  letter-spacing: 0.06em !important;
  box-shadow: 0 8px 24px rgba(0,0,0,0.3) !important;
}
[data-testid="stFormSubmitButton"] > button:hover {
  background: linear-gradient(135deg, rgba(50,110,70,0.9), rgba(30,70,45,0.95)) !important;
  box-shadow: 0 10px 32px rgba(0,0,0,0.4), 0 0 24px rgba(127,255,212,0.15) !important;
  color: white !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: var(--radius-md); overflow: hidden; }
[data-testid="stDataFrame"] th {
  background: rgba(127,255,212,0.07) !important;
  color: var(--mint) !important;
  font-family: 'Space Mono', monospace !important;
  font-size: 0.7rem !important; letter-spacing: 0.08em !important;
}
[data-testid="stDataFrame"] td {
  background: rgba(255,255,255,0.02) !important;
  color: var(--text-sec) !important; font-size: 0.82rem !important;
}
[data-testid="stDataFrame"] tr:hover td { background: rgba(255,255,255,0.05) !important; }

/* ── Alerts ── */
.stSuccess, .stInfo, .stWarning, .stError {
  border-radius: var(--radius-md) !important;
  font-family: 'Space Grotesk', sans-serif !important;
  font-size: 0.87rem !important;
}

/* ── Select / Number input ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stNumberInput"] > div > div {
  background: var(--fog) !important;
  border: 1px solid var(--border-sub) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text-pri) !important;
  font-family: 'Space Grotesk', sans-serif !important;
}

/* ── Divider ── */
hr {
  border: none !important;
  border-top: 1px solid var(--border-sub) !important;
  margin: 1.5rem 0 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--ink-2); }
::-webkit-scrollbar-thumb { background: var(--sage); border-radius: 3px; }

/* ── Suitability score colors ── */
.score-excellent { color: #7fffd4; }
.score-verygood  { color: #90ee90; }
.score-good      { color: #d4a843; }
.score-moderate  { color: #e8944a; }
.score-poor      { color: #e06060; }

/* ── Status dot ── */
.status-dot {
  display: inline-block; width: 7px; height: 7px;
  border-radius: 50%; margin-right: 6px;
  animation: pulse-dot 2s ease infinite;
}
.status-dot.green { background: var(--mint); box-shadow: 0 0 6px var(--mint); }
.status-dot.amber { background: var(--gold); box-shadow: 0 0 6px var(--gold); }
.status-dot.red   { background: #e06060; box-shadow: 0 0 6px #e06060; }
@keyframes pulse-dot {
  0%,100% { opacity: 1; } 50% { opacity: 0.4; }
}

/* ── Input group label ── */
.input-section-label {
  font-family: 'Space Mono', monospace;
  font-size: 0.65rem; letter-spacing: 0.2em;
  text-transform: uppercase; color: var(--text-mute);
  padding: 0.6rem 0 0.4rem;
  border-bottom: 1px solid var(--border-sub);
  margin-bottom: 0.8rem;
}

/* ── Code blocks ── */
pre, code {
  font-family: 'Space Mono', monospace !important;
  font-size: 0.78rem !important;
  background: rgba(0,0,0,0.4) !important;
  border: 1px solid var(--border-sub) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--mint) !important;
}

/* ── hide streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ── spinner ── */
[data-testid="stSpinner"] { color: var(--mint) !important; }

/* ── progress ── */
[data-testid="stProgressBar"] > div > div {
  background: linear-gradient(90deg, var(--sage), var(--mint)) !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS & HELPERS
# ══════════════════════════════════════════════════════════════════════════════
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "Crop_recommendation.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


@st.cache_data(show_spinner=False)
def load_dataset():
    from src.preprocessing import load_data, clean_data
    df_raw = load_data(DATA_PATH)
    df, missing, n_dupes, outliers = clean_data(df_raw)
    return df, missing, n_dupes, outliers


@st.cache_resource(show_spinner=False)
def load_trained_model():
    from src.train_model import load_model
    from src.preprocessing import load_preprocessors
    model, metadata = load_model(MODEL_DIR)
    scaler, le = load_preprocessors(MODEL_DIR)
    return model, metadata, scaler, le


def models_exist() -> bool:
    return (os.path.exists(os.path.join(MODEL_DIR, "crop_model.pkl")) and
            os.path.exists(os.path.join(MODEL_DIR, "scaler.pkl")))


def run_training():
    from src.preprocessing import load_data, clean_data, prepare_features, save_preprocessors
    from src.train_model import compare_models, select_best_model, save_model
    with st.spinner("Cleaning dataset…"):
        df, _, _, _ = clean_data(load_data(DATA_PATH))
    with st.spinner("Preparing features…"):
        X_train, X_test, y_train, y_test, scaler, le, class_names = prepare_features(df)
        save_preprocessors(scaler, le, MODEL_DIR)
    prog = st.progress(0, text="Running 6 model comparisons…")
    with st.spinner("Training models — 1–2 minutes…"):
        results_df, trained_models = compare_models(X_train, X_test, y_train, y_test)
        prog.progress(80, "Selecting best…")
    best_name, best_model, best_metrics = select_best_model(results_df, trained_models)
    save_model(best_model, best_name, best_metrics, class_names, MODEL_DIR)
    prog.progress(100, "Complete!")
    st.session_state['results_df'] = results_df
    st.session_state['X_train']    = X_train
    st.session_state['class_names'] = class_names
    st.cache_resource.clear()
    return results_df, best_name, best_metrics


def plotly_theme(fig, height=380):
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Space Grotesk, sans-serif', color='#9db89a', size=11),
        height=height,
        margin=dict(t=40, b=30, l=10, r=10),
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)', linecolor='rgba(255,255,255,0.08)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)', linecolor='rgba(255,255,255,0.08)'),
        title_font=dict(family='Playfair Display, serif', size=16, color='#e8f0e4'),
        legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='rgba(255,255,255,0.08)'),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # Logo
    st.markdown("""
    <div style="padding: 2rem 1.2rem 1.5rem; border-bottom: 1px solid rgba(127,255,212,0.1);">
      <div style="display:flex; align-items:center; gap:0.75rem; margin-bottom:0.5rem;">
        <div style="width:36px;height:36px; border-radius:10px;
                    background:linear-gradient(135deg,#2a5e3a,#3d8a52);
                    display:flex; align-items:center; justify-content:center;
                    font-size:1.2rem; box-shadow:0 4px 12px rgba(0,0,0,0.4);">🌾</div>
        <div>
          <div style="font-family:'Playfair Display',serif; font-size:1.1rem;
                      font-weight:900; color:#f5efe0; letter-spacing:-0.01em;">CropAI</div>
          <div style="font-family:'Space Mono',monospace; font-size:0.58rem;
                      letter-spacing:0.18em; text-transform:uppercase; color:#617a5e;">
            Intelligence System</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # Navigation
    page = st.radio(
        "nav",
        ["⬡  Overview",
         "◈  Data & Models",
         "◉  Predict Crop",
         "⬟  Explainability"],
        label_visibility="collapsed"
    )

    # Status
    st.markdown("""<div style="height:1.5rem"></div>""", unsafe_allow_html=True)
    st.markdown("""<div style="padding:0 1rem; font-family:'Space Mono',monospace;
                    font-size:0.6rem; letter-spacing:0.18em; text-transform:uppercase;
                    color:#617a5e; margin-bottom:0.6rem;">System Status</div>""",
                unsafe_allow_html=True)

    ds_ok  = os.path.exists(DATA_PATH)
    mdl_ok = models_exist()
    ds_cls  = "green" if ds_ok  else "red"
    mdl_cls = "green" if mdl_ok else "amber"
    ds_txt  = "Dataset ready"   if ds_ok  else "Dataset missing"
    mdl_txt = "Model trained"   if mdl_ok else "Awaiting training"

    st.markdown(f"""
    <div style="padding:0.2rem 1rem 0.3rem; font-size:0.8rem; color:#9db89a;">
      <div style="margin:0.25rem 0;">
        <span class="status-dot {ds_cls}"></span>{ds_txt}
      </div>
      <div style="margin:0.25rem 0;">
        <span class="status-dot {mdl_cls}"></span>{mdl_txt}
      </div>
    </div>
    """, unsafe_allow_html=True)

    if not ds_ok:
        st.markdown("""<div style="padding:0 1rem; font-size:0.72rem;
                        color:#617a5e; line-height:1.5; margin-top:0.5rem;">
          Place <code>Crop_recommendation.csv</code><br>in the <code>/data/</code> folder.
        </div>""", unsafe_allow_html=True)

    st.markdown("""<div style="position:absolute; bottom:1.5rem; left:0; right:0;
                    padding:0 1.2rem; font-family:'Space Mono',monospace;
                    font-size:0.58rem; letter-spacing:0.1em; color:#3d4f3a;
                    text-transform:uppercase;">
      Final Year DS Project · 2025
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "⬡  Overview":

    # Hero
    st.markdown("""
    <div class="hero-wrap">
      <div class="hero-content">
        <div class="hero-eyebrow">Agricultural Intelligence Platform</div>
        <h1 class="hero-title">Predict the <em>right crop</em><br>for every field</h1>
        <p class="hero-sub">
          AI-powered recommendations backed by 6 competing ML models, SHAP explainability,
          and a rich visualization suite — built for farmers, agronomists, and researchers.
        </p>
        <div class="hero-pills">
          <span class="hero-pill">22 crop classes</span>
          <span class="hero-pill">7 soil features</span>
          <span class="hero-pill">SHAP explainability</span>
          <span class="hero-pill">Random Forest · XGBoost · LightGBM</span>
          <span class="hero-pill">Real-time prediction</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Quick stats
    if os.path.exists(DATA_PATH):
        try:
            df, _, _, _ = load_dataset()
            st.markdown("""<div class="sec-label">Dataset at a glance</div>""",
                        unsafe_allow_html=True)
            st.markdown(f"""
            <div class="stat-grid">
              <div class="stat-card">
                <div class="stat-icon">🗂</div>
                <div class="stat-num">{df.shape[0]:,}</div>
                <div class="stat-lbl">Total Samples</div>
              </div>
              <div class="stat-card">
                <div class="stat-icon">🌿</div>
                <div class="stat-num">{df['label'].nunique()}</div>
                <div class="stat-lbl">Crop Classes</div>
              </div>
              <div class="stat-card">
                <div class="stat-icon">🧪</div>
                <div class="stat-num">7</div>
                <div class="stat-lbl">Input Features</div>
              </div>
              <div class="stat-card">
                <div class="stat-icon">🏆</div>
                <div class="stat-num">99%</div>
                <div class="stat-lbl">Peak Accuracy</div>
              </div>
            </div>
            """, unsafe_allow_html=True)
        except Exception:
            pass

    # Feature grid
    st.markdown("""<div class="sec-label" style="margin-top:0.5rem">Capabilities</div>""",
                unsafe_allow_html=True)
    st.markdown("""
    <div class="feat-row">
      <div class="feat-pill"><span class="pill-icon">🌱</span> Smart crop prediction</div>
      <div class="feat-pill"><span class="pill-icon">📊</span> Confidence scoring</div>
      <div class="feat-pill"><span class="pill-icon">🧠</span> SHAP explainability</div>
      <div class="feat-pill"><span class="pill-icon">📈</span> Interactive EDA charts</div>
      <div class="feat-pill"><span class="pill-icon">⚖️</span> 6-model comparison</div>
      <div class="feat-pill"><span class="pill-icon">🎯</span> Suitability gauge</div>
      <div class="feat-pill"><span class="pill-icon">🔁</span> Real-time inference</div>
    </div>
    """, unsafe_allow_html=True)

    # ML Pipeline flow
    st.markdown("""<div class="sec-label">ML pipeline</div>""", unsafe_allow_html=True)
    cols = st.columns(6)
    steps = [
        ("01", "Data Cleaning", "Missing values, duplicates, outlier detection"),
        ("02", "Feature Scaling", "StandardScaler normalisation"),
        ("03", "Model Training", "6 algorithms trained in parallel"),
        ("04", "Auto Selection", "Best F1 model chosen automatically"),
        ("05", "SHAP Analysis", "Local & global explainability"),
        ("06", "Live Predict", "Real-time inference with reasoning"),
    ]
    for col, (num, title, desc) in zip(cols, steps):
        col.markdown(f"""
        <div style="background:var(--fog); border:1px solid var(--border-sub);
                    border-radius:var(--radius-md); padding:1.1rem 1rem; height:100%;">
          <div style="font-family:'Space Mono',monospace; font-size:0.6rem;
                      letter-spacing:0.15em; color:var(--mint); margin-bottom:0.5rem;">{num}</div>
          <div style="font-size:0.82rem; font-weight:600; color:var(--text-pri);
                      margin-bottom:0.3rem;">{title}</div>
          <div style="font-size:0.73rem; color:var(--text-mute); line-height:1.45;">{desc}</div>
        </div>""", unsafe_allow_html=True)

    if not mdl_ok:
        st.markdown("<br>", unsafe_allow_html=True)
        st.info("⟶ Navigate to **Data & Models** to train the model before making predictions.")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — DATA & MODELS (merged Dataset Insights + Model Training)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "◈  Data & Models":

    if not os.path.exists(DATA_PATH):
        st.error("Dataset not found. Place `Crop_recommendation.csv` in `/data/`.")
        st.stop()

    df, missing_report, n_dupes, outliers = load_dataset()
    feat_cols = ['N','P','K','temperature','humidity','ph','rainfall']

    # ── Dataset summary row ──
    st.markdown("""<div class="sec-label">Dataset overview</div>""", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="stat-grid">
      <div class="stat-card">
        <div class="stat-icon">📄</div>
        <div class="stat-num">{df.shape[0]:,}</div><div class="stat-lbl">Rows</div>
      </div>
      <div class="stat-card">
        <div class="stat-icon">📐</div>
        <div class="stat-num">{df.shape[1]}</div><div class="stat-lbl">Columns</div>
      </div>
      <div class="stat-card">
        <div class="stat-icon">❌</div>
        <div class="stat-num">{int(missing_report['missing_count'].sum())}</div>
        <div class="stat-lbl">Missing Values</div>
      </div>
      <div class="stat-card">
        <div class="stat-icon">🔄</div>
        <div class="stat-num">{n_dupes}</div><div class="stat-lbl">Duplicates Removed</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    tab_eda, tab_train = st.tabs(["◈  Explore Data", "⚙  Train & Compare Models"])

    # ──────────────────────── EDA TAB ────────────────────────
    with tab_eda:
        t1, t2, t3, t4 = st.tabs(["Sample", "Distributions", "Correlation", "Crop Stats"])

        with t1:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**First 10 rows**")
                st.dataframe(df.head(10), use_container_width=True,
                             column_config={c: st.column_config.NumberColumn(format="%.3f")
                                            for c in df.select_dtypes('float').columns})
            with c2:
                st.markdown("**Descriptive statistics**")
                st.dataframe(df[feat_cols].describe().round(2), use_container_width=True)

            st.markdown("**Outliers per feature (IQR method)**")
            out_df  = pd.DataFrame(list(outliers.items()), columns=['Feature','Outliers'])
            fig_out = px.bar(out_df, x='Feature', y='Outliers',
                             color='Outliers', color_continuous_scale='Reds')
            fig_out.update_traces(marker_line_width=0)
            plotly_theme(fig_out, 280)
            st.plotly_chart(fig_out, use_container_width=True)

        with t2:
            feat = st.selectbox("Feature", feat_cols, key="dist_feat")
            cl, cr = st.columns(2)
            with cl:
                fig_h = px.histogram(df, x=feat, color='label', nbins=40, opacity=0.7,
                                     barmode='overlay', title=f"{feat} Distribution")
                fig_h.update_traces(marker_line_width=0)
                plotly_theme(fig_h, 340)
                fig_h.update_layout(showlegend=False)
                st.plotly_chart(fig_h, use_container_width=True)
            with cr:
                fig_b = px.box(df, x='label', y=feat, color='label', title=f"{feat} by Crop")
                plotly_theme(fig_b, 340)
                fig_b.update_layout(showlegend=False, xaxis_tickangle=-40)
                st.plotly_chart(fig_b, use_container_width=True)

        with t3:
            corr     = df[feat_cols].corr()
            fig_corr = go.Figure(go.Heatmap(
                z=corr.values, x=corr.columns, y=corr.columns,
                colorscale=[[0,'#c0392b'],[0.5,'#182414'],[1,'#27ae60']],
                zmid=0, text=corr.round(2).values, texttemplate="%{text}",
                colorbar=dict(title='r', tickfont=dict(color='#9db89a')),
            ))
            plotly_theme(fig_corr, 460)
            fig_corr.update_layout(title="Pearson Correlation Heatmap")
            st.plotly_chart(fig_corr, use_container_width=True)

        with t4:
            cc = df['label'].value_counts().reset_index()
            cc.columns = ['Crop', 'Count']
            cl, cr = st.columns(2)
            with cl:
                fig_pie = px.pie(cc, names='Crop', values='Count', hole=0.5,
                                 color_discrete_sequence=px.colors.qualitative.Prism,
                                 title="Crop Distribution")
                plotly_theme(fig_pie, 400)
                st.plotly_chart(fig_pie, use_container_width=True)
            with cr:
                fig_hb = px.bar(cc.sort_values('Count'), x='Count', y='Crop',
                                orientation='h', color='Count',
                                color_continuous_scale=[[0,'#1a3d25'],[1,'#7fffd4']],
                                title="Samples per Crop")
                fig_hb.update_traces(marker_line_width=0)
                plotly_theme(fig_hb, 400)
                st.plotly_chart(fig_hb, use_container_width=True)

    # ──────────────────────── TRAINING TAB ────────────────────────
    with tab_train:
        if not mdl_ok:
            st.markdown("""
            <div class="glass-panel" style="text-align:center; padding:3rem 2rem;">
              <div style="font-size:3.5rem; margin-bottom:1rem;">⚗️</div>
              <div style="font-family:'Playfair Display',serif; font-size:1.6rem;
                          font-weight:700; color:var(--parchment); margin-bottom:0.6rem;">
                Train the Model
              </div>
              <div style="color:var(--text-mute); font-size:0.9rem; max-width:420px;
                          margin:0 auto 1.5rem;">
                Compares Random Forest, XGBoost, LightGBM, Decision Tree,
                KNN, and Logistic Regression — picks the best automatically.
              </div>
            </div>
            """, unsafe_allow_html=True)
            col_c = st.columns([1, 2, 1])[1]
            with col_c:
                if st.button("🌱  Start Training", use_container_width=True):
                    try:
                        results_df, best_name, best_metrics = run_training()
                        st.success(f"Training complete — best model: **{best_name}**")
                        st.balloons()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Training failed: {e}")
        else:
            try:
                model, metadata, scaler, le = load_trained_model()
                best_name    = metadata['model_name']
                best_metrics = metadata['metrics']

                # Model banner
                st.markdown(f"""
                <div class="model-banner">
                  <div>
                    <div style="font-family:'Space Mono',monospace; font-size:0.62rem;
                                letter-spacing:0.18em; text-transform:uppercase;
                                color:var(--text-mute); margin-bottom:0.35rem;">
                      Best model selected
                    </div>
                    <div class="model-name-display">
                      <span class="model-crown">👑</span>{best_name}
                    </div>
                  </div>
                  <div class="metric-pill-row">
                    {"".join([
                        f'<div class="metric-pill"><div class="metric-pill-val">{best_metrics.get(k,0):.3f}</div>'
                        f'<div class="metric-pill-lbl">{l}</div></div>'
                        for k, l in [("accuracy","Accuracy"),("precision","Precision"),
                                     ("recall","Recall"),("f1","F1"),("cv_mean","CV Mean")]
                    ])}
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # Comparison chart
                if 'results_df' in st.session_state:
                    rdf = st.session_state['results_df']
                    fig_c = go.Figure()
                    colors = ['#7fffd4','#5a9e70','#d4a843','#e06060']
                    for i, metric in enumerate(['accuracy','precision','recall','f1']):
                        fig_c.add_trace(go.Bar(
                            name=metric.title(),
                            x=rdf['model'], y=rdf[metric],
                            text=rdf[metric].round(3), textposition='outside',
                            marker_color=colors[i], marker_line_width=0,
                        ))
                    fig_c.update_layout(barmode='group', xaxis_tickangle=-15,
                                        title="Model Performance Comparison")
                    plotly_theme(fig_c, 360)
                    st.plotly_chart(fig_c, use_container_width=True)

                    st.markdown("**All model scores**")
                    st.dataframe(
                        rdf.style
                        .highlight_max(subset=['accuracy','f1','cv_mean'],
                                       color='rgba(127,255,212,0.15)')
                        .format({c:'{:.4f}' for c in rdf.select_dtypes('float').columns}),
                        use_container_width=True, hide_index=True
                    )
                else:
                    st.info("Re-train to see the full comparison table.")

                st.markdown("<br>", unsafe_allow_html=True)
                col_c = st.columns([1, 2, 1])[1]
                with col_c:
                    if st.button("🔄  Re-train Model", use_container_width=True):
                        try:
                            results_df, best_name, best_metrics = run_training()
                            st.success(f"Re-training complete — best: **{best_name}**")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Re-training failed: {e}")

            except Exception as e:
                st.error(f"Could not load model: {e}")
                if st.button("Retry training"):
                    run_training(); st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — PREDICT CROP
# ══════════════════════════════════════════════════════════════════════════════
elif page == "◉  Predict Crop":

    if not mdl_ok:
        st.warning("Model not trained yet. Go to **Data & Models → Train & Compare Models** first.")
        st.stop()

    model, metadata, scaler, le = load_trained_model()

    st.markdown("""<div class="sec-label">Soil & environment parameters</div>""",
                unsafe_allow_html=True)

    with st.form("pred_form", clear_on_submit=False):
        c1, c2, c3 = st.columns(3)
        st.markdown("""<style>
          [data-testid="column"]:nth-child(1) .input-section-label { color: #7fffd4; }
        </style>""", unsafe_allow_html=True)

        with c1:
            st.markdown('<div class="input-section-label">🧪 Soil Nutrients</div>',
                        unsafe_allow_html=True)
            N = st.slider("Nitrogen — N  (mg/kg)", 0, 200, 90)
            P = st.slider("Phosphorus — P  (mg/kg)", 0, 200, 42)
            K = st.slider("Potassium — K  (mg/kg)", 0, 200, 43)

        with c2:
            st.markdown('<div class="input-section-label">🌡️ Climate</div>',
                        unsafe_allow_html=True)
            temperature = st.slider("Temperature  (°C)", 0.0, 50.0, 21.0, 0.1)
            humidity    = st.slider("Humidity  (%)", 0.0, 100.0, 82.0, 0.1)
            rainfall    = st.slider("Rainfall  (mm)", 0.0, 500.0, 202.0, 0.5)

        with c3:
            st.markdown('<div class="input-section-label">⚗️ Soil Chemistry</div>',
                        unsafe_allow_html=True)
            ph = st.slider("pH Level", 0.0, 14.0, 6.5, 0.01)

            st.markdown("<br>", unsafe_allow_html=True)
            # Live preview
            st.markdown(f"""
            <div style="background:var(--fog); border:1px solid var(--border-sub);
                        border-radius:var(--radius-md); padding:1rem; font-size:0.78rem;">
              <div style="color:var(--text-mute); font-family:'Space Mono',monospace;
                          font-size:0.6rem; letter-spacing:0.15em; margin-bottom:0.6rem;">
                CURRENT INPUTS
              </div>
              <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.3rem;
                          color:var(--text-sec); font-family:'Space Mono',monospace;">
                <span>N: <b style="color:var(--mint)">{N}</b></span>
                <span>P: <b style="color:var(--mint)">{P}</b></span>
                <span>K: <b style="color:var(--mint)">{K}</b></span>
                <span>Temp: <b style="color:var(--mint)">{temperature}°</b></span>
                <span>Hum: <b style="color:var(--mint)">{humidity}%</b></span>
                <span>pH: <b style="color:var(--mint)">{ph}</b></span>
                <span>Rain: <b style="color:var(--mint)">{rainfall}mm</b></span>
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("⟶  ANALYSE & PREDICT", use_container_width=True)

    if submitted:
        from src.predictor import predict_crop, validate_inputs
        errors = validate_inputs(N, P, K, temperature, humidity, ph, rainfall)
        if errors:
            for e in errors: st.error(e)
            st.stop()

        with st.spinner("Running inference…"):
            result = predict_crop(model, scaler, le,
                                  N, P, K, temperature, humidity, ph, rainfall)
        st.session_state['last_prediction'] = result

        pred_crop  = result['predicted_crop']
        confidence = result['confidence']
        top3       = result['top3']
        score      = result['suitability_score']
        s_label    = result['suitability_label']
        s_color    = result['suitability_color']
        reasoning  = result['reasoning']

        score_cls  = {'Excellent':'score-excellent','Very Good':'score-verygood',
                      'Good':'score-good','Moderate':'score-moderate',
                      'Poor':'score-poor'}.get(s_label,'score-good')

        st.markdown("<br>", unsafe_allow_html=True)
        col_main, col_side = st.columns([3, 2], gap="large")

        with col_main:
            st.markdown(f"""
            <div class="pred-hero">
              <div class="pred-label">Recommended Crop</div>
              <div class="pred-crop-name">{pred_crop.title()}</div>
              <div>
                <div class="pred-badge">◎ Confidence: {confidence:.1f}%</div>
              </div>
              <div style="margin-top:0.9rem; font-family:'Space Mono',monospace;
                          font-size:0.78rem; position:relative; z-index:1;"
                   class="{score_cls}">
                ▸ Suitability: {s_label} &nbsp;·&nbsp; Score: {score:.0f}/100
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Top 3 rankings
            st.markdown("""<div class="sec-label">Top 3 recommendations</div>""",
                        unsafe_allow_html=True)
            medals = ["🥇", "🥈", "🥉"]
            fill_classes = ["gold", "", ""]
            for i, item in enumerate(top3):
                fc = fill_classes[i]
                st.markdown(f"""
                <div class="rank-item">
                  <div class="rank-head">
                    <span class="rank-crop">{medals[i]} {item['crop'].title()}</span>
                    <span class="rank-pct">{item['confidence']:.1f}%</span>
                  </div>
                  <div class="rank-track">
                    <div class="rank-fill {fc}" style="width:{item['confidence']}%"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

            # Reasoning
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""<div class="sec-label">Recommendation reasoning</div>""",
                        unsafe_allow_html=True)
            rl, rr = st.columns(2)
            with rl:
                st.markdown("**Positive factors**")
                for r in reasoning.get('positive_reasons', []):
                    clean = r.lstrip('✓').strip()
                    st.markdown(f'<div class="reason-card pos"><span class="rc-icon">✦</span>{clean}</div>',
                                unsafe_allow_html=True)
            with rr:
                st.markdown("**Monitor**")
                cautions = reasoning.get('caution_reasons', [])
                if cautions:
                    for r in cautions:
                        clean = r.lstrip('⚠').strip()
                        st.markdown(f'<div class="reason-card cau"><span class="rc-icon">◈</span>{clean}</div>',
                                    unsafe_allow_html=True)
                else:
                    st.markdown('<div class="reason-card pos"><span class="rc-icon">✦</span>All conditions within ideal range</div>',
                                unsafe_allow_html=True)

        with col_side:
            # Suitability gauge
            gauge_clr = s_color
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                title={'text': "Suitability Score",
                       'font': {'family':'Playfair Display, serif', 'size': 15, 'color':'#9db89a'}},
                number={'suffix': "%", 'font': {'family':'Space Mono, monospace',
                                                'size': 38, 'color': gauge_clr}},
                gauge={
                    'axis': {'range': [0, 100], 'tickfont': {'color': '#617a5e', 'size': 9},
                             'tickcolor': '#617a5e'},
                    'bar': {'color': gauge_clr, 'thickness': 0.22},
                    'bgcolor': 'rgba(0,0,0,0)',
                    'borderwidth': 0,
                    'steps': [
                        {'range': [0,  60], 'color': 'rgba(224,96,96,0.12)'},
                        {'range': [60, 70], 'color': 'rgba(232,148,74,0.12)'},
                        {'range': [70, 80], 'color': 'rgba(212,168,67,0.12)'},
                        {'range': [80, 90], 'color': 'rgba(90,158,112,0.15)'},
                        {'range': [90,100], 'color': 'rgba(127,255,212,0.12)'},
                    ],
                    'threshold': {
                        'line': {'color': gauge_clr, 'width': 2},
                        'thickness': 0.9, 'value': score
                    }
                }
            ))
            fig_g.update_layout(
                height=260, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(t=40, b=10, l=20, r=20),
            )
            st.plotly_chart(fig_g, use_container_width=True)

            # Feature contribution chart
            contribs = reasoning.get('contributions', {})
            if contribs:
                labs = list(contribs.keys())
                vals = list(contribs.values())
                fig_c = go.Figure(go.Bar(
                    x=vals, y=labs, orientation='h',
                    marker=dict(
                        color=vals,
                        colorscale=[[0,'#1a3d25'],[0.5,'#3d8a52'],[1,'#7fffd4']],
                        line_width=0
                    ),
                    text=[f"{v:.1f}%" for v in vals],
                    textposition='outside',
                    textfont=dict(color='#9db89a', size=10),
                ))
                plotly_theme(fig_c, 300)
                fig_c.update_layout(
                    title="Feature Contributions",
                    xaxis_title="Contribution %",
                    title_font=dict(size=13),
                )
                st.plotly_chart(fig_c, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — EXPLAINABILITY (Global + Local)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⬟  Explainability":

    if not mdl_ok:
        st.warning("Model not trained. Go to **Data & Models** first.")
        st.stop()

    model, metadata, scaler, le = load_trained_model()
    class_names = metadata['class_names']

    st.markdown("""<div class="sec-label">Explainable AI — SHAP analysis</div>""",
                unsafe_allow_html=True)

    tab_g, tab_l, tab_about = st.tabs(["🌍  Global Importance", "🎯  Local Explanation", "ℹ️  About & Deploy"])

    # ─── Global ───
    with tab_g:
        st.markdown("""
        <div class="glass-panel" style="margin-bottom:1.2rem;">
          <div style="font-size:0.85rem; color:var(--text-sec); line-height:1.7;">
            <b style="color:var(--text-pri);">Global SHAP</b> shows which features matter most
            <em>across the entire dataset</em> — averaged absolute SHAP values per feature.
          </div>
        </div>
        """, unsafe_allow_html=True)

        if not os.path.exists(DATA_PATH):
            st.error("Dataset needed for global SHAP.")
        else:
            try:
                df, _, _, _ = load_dataset()
                from src.preprocessing import prepare_features
                from src.explainability import get_shap_explainer, compute_global_shap, plot_global_importance

                X_train, X_test, y_train, y_test, sc2, le2, cn = prepare_features(df)
                with st.spinner("Computing global SHAP values (~30s)…"):
                    explainer = get_shap_explainer(model, X_train, metadata['model_name'])
                    mean_shap = compute_global_shap(explainer, X_train, metadata['model_name'])

                fig_gs = plot_global_importance(mean_shap)
                plotly_theme(fig_gs, 380)
                st.plotly_chart(fig_gs, use_container_width=True)

            except Exception as e:
                st.error(f"SHAP error: {e}")

            # Model built-in importances (always available for tree models)
            if hasattr(model, 'feature_importances_'):
                fl  = ['Nitrogen','Phosphorus','Potassium','Temperature','Humidity','pH','Rainfall']
                idf = pd.DataFrame({'Feature':fl,'Importance':model.feature_importances_}).sort_values('Importance')
                fig_i = go.Figure(go.Bar(
                    x=idf['Importance'], y=idf['Feature'], orientation='h',
                    marker=dict(color=idf['Importance'],
                                colorscale=[[0,'#1a3d25'],[0.5,'#3d8a52'],[1,'#7fffd4']],
                                showscale=True, line_width=0),
                    text=idf['Importance'].round(4), textposition='outside',
                    textfont=dict(color='#9db89a', size=10),
                ))
                plotly_theme(fig_i, 360)
                fig_i.update_layout(title="Built-in Feature Importance")
                st.plotly_chart(fig_i, use_container_width=True)

    # ─── Local ───
    with tab_l:
        st.markdown("""
        <div class="glass-panel" style="margin-bottom:1.2rem;">
          <div style="font-size:0.85rem; color:var(--text-sec); line-height:1.7;">
            <b style="color:var(--text-pri);">Local SHAP</b> explains a <em>single prediction</em>
            — showing how each feature pushed the model toward or away from the recommended crop.
          </div>
        </div>
        """, unsafe_allow_html=True)

        if 'last_prediction' not in st.session_state:
            st.info("⟶ Go to **Predict Crop**, run a prediction, then return here for the local explanation.")
        else:
            result    = st.session_state['last_prediction']
            pred_crop = result['predicted_crop']
            conf      = result['confidence']
            pred_idx  = list(le.classes_).index(pred_crop)

            st.markdown(f"""
            <div style="font-family:'Space Mono',monospace; font-size:0.72rem;
                        letter-spacing:0.1em; color:var(--text-mute); margin-bottom:1rem;">
              LAST PREDICTION &nbsp;→&nbsp;
              <span style="color:var(--mint);">{pred_crop.upper()}</span>
              &nbsp;·&nbsp; {conf:.1f}% confidence
            </div>
            """, unsafe_allow_html=True)

            try:
                if os.path.exists(DATA_PATH):
                    df, _, _, _ = load_dataset()
                    from src.preprocessing import prepare_features
                    X_train, _, _, _, _, _, _ = prepare_features(df)
                else:
                    X_train = np.zeros((100, 7))

                from src.explainability import (get_shap_explainer, compute_local_shap,
                                                plot_local_explanation, plot_feature_contribution_bar)
                with st.spinner("Computing local SHAP…"):
                    explainer  = get_shap_explainer(model, X_train, metadata['model_name'])
                    local_shap = compute_local_shap(explainer, result['input_scaled'],
                                                    pred_idx, class_names)

                fig_ls = plot_local_explanation(local_shap['shap_values'], pred_crop)
                plotly_theme(fig_ls, 360)
                st.plotly_chart(fig_ls, use_container_width=True)

            except Exception as e:
                st.error(f"SHAP error: {e}")

            # Contribution bar always available
            from src.explainability import plot_feature_contribution_bar
            contribs = result['reasoning']['contributions']
            if contribs:
                fig_cb = plot_feature_contribution_bar(contribs, pred_crop)
                plotly_theme(fig_cb, 320)
                st.plotly_chart(fig_cb, use_container_width=True)

            # Reasoning summary
            st.markdown("""<div class="sec-label" style="margin-top:0.5rem">
              Explanation summary</div>""", unsafe_allow_html=True)
            cl, cr = st.columns(2)
            with cl:
                st.markdown("**Positive factors**")
                for r in result['reasoning'].get('positive_reasons', []):
                    clean = r.lstrip('✓').strip()
                    st.markdown(f'<div class="reason-card pos"><span class="rc-icon">✦</span>{clean}</div>',
                                unsafe_allow_html=True)
            with cr:
                st.markdown("**Watch out for**")
                for r in result['reasoning'].get('caution_reasons', []):
                    clean = r.lstrip('⚠').strip()
                    st.markdown(f'<div class="reason-card cau"><span class="rc-icon">◈</span>{clean}</div>',
                                unsafe_allow_html=True)
                if not result['reasoning'].get('caution_reasons'):
                    st.markdown('<div class="reason-card pos"><span class="rc-icon">✦</span>All parameters within ideal range</div>',
                                unsafe_allow_html=True)

    # ─── About ───
    with tab_about:
        cl, cr = st.columns([1, 1], gap="large")
        with cl:
            st.markdown("""
            <div class="glass-panel">
              <div style="font-family:'Playfair Display',serif; font-size:1.3rem;
                          font-weight:700; color:var(--parchment); margin-bottom:0.8rem;">
                About this project
              </div>
              <div style="color:var(--text-sec); font-size:0.88rem; line-height:1.75;">
                A final-year data science project leveraging ensemble ML and SHAP
                explainability to recommend the most suitable crop from soil and
                environmental measurements — designed for agricultural decision-support.
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**Technology stack**")
            techs = [("Python 3.10+","Core language"),("Streamlit","Web UI"),
                     ("scikit-learn","ML pipeline"),("XGBoost / LightGBM","Gradient boosting"),
                     ("SHAP","Explainability"),("Plotly","Visualisations"),
                     ("Pandas / NumPy","Data processing"),("joblib","Model persistence")]
            for name, desc in techs:
                st.markdown(f"""
                <div style="display:flex; align-items:center; gap:0.8rem; padding:0.35rem 0;
                            border-bottom:1px solid rgba(255,255,255,0.04);">
                  <span style="color:var(--mint); font-weight:600; font-family:'Space Mono',monospace;
                               font-size:0.78rem; min-width:150px;">{name}</span>
                  <span style="color:var(--text-mute); font-size:0.82rem;">{desc}</span>
                </div>""", unsafe_allow_html=True)

        with cr:
            st.markdown("**Dataset**")
            st.markdown("""
            <div class="glass-panel">
              <div style="color:var(--text-sec); line-height:1.8; font-size:0.86rem;">
                <b style="color:var(--text-pri)">Name:</b> Crop Recommendation Dataset<br>
                <b style="color:var(--text-pri)">Features:</b> N, P, K, Temperature, Humidity, pH, Rainfall<br>
                <b style="color:var(--text-pri)">Target:</b> 22 crop classes<br>
                <b style="color:var(--text-pri)">Rows:</b> 2,200<br>
                <b style="color:var(--text-pri)">Source:</b> Kaggle / UCI
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**Deploy on Render**")
            st.code("""# 1. Push repo to GitHub (include /data/ CSV)
# 2. New Web Service → connect repo
# Build Command:
pip install -r requirements.txt
# Start Command:
streamlit run app.py \\
  --server.port $PORT \\
  --server.address 0.0.0.0 \\
  --server.headless true
# Env: PYTHON_VERSION = 3.10.12""", language="bash")

            st.markdown("**Project structure**")
            st.code("""crop_system/
├── data/  Crop_recommendation.csv
├── models/ crop_model.pkl · scaler.pkl
├── src/   preprocessing · train_model
│          predictor · explainability
├── app.py · requirements.txt 
├── render.yaml · README.md""", language="text")