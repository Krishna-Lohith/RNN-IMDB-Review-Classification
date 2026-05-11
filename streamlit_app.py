import re
import numpy as np
import streamlit as st
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence


# ── Keras 3.x compatibility fix ───────────────────────────────────────────────
# Models saved with TF 2.15 / Keras 2.x store 'time_major' in the layer config.
# Keras 3.x removed that argument and raises an error when loading the .h5 file.
# This subclass silently drops the unknown kwarg so the model loads cleanly.
class _CompatSimpleRNN(SimpleRNN):
    def __init__(self, *args, **kwargs):
        kwargs.pop("time_major", None)
        super().__init__(*args, **kwargs)

# ── Page config (must be the very first Streamlit call) ───────────────────────
st.set_page_config(
    page_title="CineScope · Movie Sentiment",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Netflix-inspired dark-red CSS theme ───────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;500;600;700&display=swap');

/* ---------- Global reset ---------- */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background-color: #141414 !important;
    font-family: 'Inter', sans-serif;
    color: #FFFFFF;
}
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* Radial red glow at the top of the page */
[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse 80% 40% at 50% 0%, #2a0000 0%, #141414 60%) !important;
}

/* ---------- Navbar ---------- */
.navbar {
    background: linear-gradient(180deg, #0d0000 0%, #141414 100%);
    border-bottom: 2px solid #E50914;
    padding: 18px 32px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 48px;
    border-radius: 0 0 10px 10px;
}
.brand-name {
    font-family: 'Bebas Neue', cursive;
    font-size: 2rem;
    color: #E50914;
    letter-spacing: 5px;
    text-shadow: 0 0 24px rgba(229,9,20,0.75);
    line-height: 1;
}
.brand-tagline {
    font-size: 0.78rem;
    color: #666;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 2px;
}
.user-block { text-align: right; }
.user-name {
    font-size: 1.1rem;
    font-weight: 600;
    color: #FFFFFF;
    letter-spacing: 0.5px;
}
.user-role {
    font-size: 0.88rem;
    color: #E50914;
    letter-spacing: 2px;
    text-transform: uppercase;
    font-weight: 500;
    margin-top: 2px;
}

/* ---------- Hero section ---------- */
.hero {
    text-align: center;
    padding: 4px 0 36px;
}
.hero-title {
    font-family: 'Bebas Neue', cursive;
    font-size: 4.8rem;
    letter-spacing: 6px;
    color: #FFFFFF;
    line-height: 1.05;
    text-shadow: 0 0 60px rgba(229,9,20,0.3);
    margin-bottom: 14px;
}
.hero-title .accent { color: #E50914; }
.hero-sub {
    color: #777;
    font-size: 1rem;
    letter-spacing: 2.5px;
    text-transform: uppercase;
}

/* ---------- Thin red divider ---------- */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent 0%, #E50914 25%, #E50914 75%, transparent 100%);
    margin: 28px 0;
    border: none;
    opacity: 0.55;
}

/* ---------- Small section labels ---------- */
.section-label {
    font-size: 0.92rem;
    font-weight: 600;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 2.5px;
    margin-bottom: 10px;
}

/* ---------- Textarea ---------- */
[data-testid="stTextArea"] label { display: none !important; }
[data-testid="stTextArea"] textarea {
    background-color: #1C1C1C !important;
    color: #EFEFEF !important;
    border: 1px solid #333 !important;
    border-radius: 7px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 1.08rem !important;
    line-height: 1.7 !important;
    padding: 16px !important;
    resize: vertical !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    caret-color: #E50914 !important;
}
[data-testid="stTextArea"] textarea:focus {
    border-color: #E50914 !important;
    box-shadow: 0 0 0 3px rgba(229,9,20,0.18) !important;
    outline: none !important;
}
[data-testid="stTextArea"] textarea::placeholder { color: #484848 !important; }

/* ---------- Example pick buttons ---------- */
[data-testid="stButton"] > button {
    background: #1C1C1C !important;
    color: #B3B3B3 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.5px !important;
    padding: 12px 14px !important;
    border: 1px solid #333 !important;
    border-radius: 20px !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    margin-bottom: 0 !important;
}
[data-testid="stButton"] > button:hover {
    border-color: #E50914 !important;
    color: #FFFFFF !important;
    background: #2a0000 !important;
    transform: translateY(-1px) !important;
}

/* ---------- Analyze button (last button = full-width red) ---------- */
[data-testid="stButton"]:last-of-type > button {
    background: linear-gradient(135deg, #E50914 0%, #8B0000 100%) !important;
    color: #FFFFFF !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    letter-spacing: 3.5px !important;
    text-transform: uppercase !important;
    padding: 16px !important;
    border: none !important;
    border-radius: 5px !important;
    box-shadow: 0 4px 28px rgba(229,9,20,0.45) !important;
    margin-top: 6px !important;
}
[data-testid="stButton"]:last-of-type > button:hover {
    background: linear-gradient(135deg, #FF1A27 0%, #A50000 100%) !important;
    box-shadow: 0 8px 36px rgba(229,9,20,0.65) !important;
    transform: translateY(-2px) !important;
    color: #FFFFFF !important;
    border-color: transparent !important;
}

/* ---------- Result cards ---------- */
.result-positive {
    background: linear-gradient(135deg, #0C1C0C 0%, #142014 100%);
    border-left: 4px solid #27AE60;
    border-radius: 9px;
    padding: 30px 34px;
    margin-top: 28px;
    box-shadow: 0 6px 50px rgba(39,174,96,0.1);
    animation: rise 0.55s cubic-bezier(0.22, 1, 0.36, 1);
}
.result-negative {
    background: linear-gradient(135deg, #1C0C0C 0%, #201414 100%);
    border-left: 4px solid #E50914;
    border-radius: 9px;
    padding: 30px 34px;
    margin-top: 28px;
    box-shadow: 0 6px 50px rgba(229,9,20,0.1);
    animation: rise 0.55s cubic-bezier(0.22, 1, 0.36, 1);
}
@keyframes rise {
    from { opacity: 0; transform: translateY(22px); }
    to   { opacity: 1; transform: translateY(0px); }
}
.result-icon  { font-size: 3rem; margin-bottom: 10px; }
.result-label {
    font-family: 'Bebas Neue', cursive;
    font-size: 2.8rem;
    letter-spacing: 4px;
    margin-bottom: 8px;
    line-height: 1;
}
.color-green { color: #2ECC71; }
.color-red   { color: #E50914; }
.result-desc {
    color: #888;
    font-size: 1rem;
    line-height: 1.6;
    margin-bottom: 24px;
}

/* Scores row: two stat boxes side by side */
.scores-row {
    display: flex;
    gap: 16px;
    margin-bottom: 20px;
}
.score-box {
    flex: 1;
    background: rgba(255,255,255,0.04);
    border: 1px solid #2a2a2a;
    border-radius: 8px;
    padding: 14px 18px;
}
.score-box-label {
    font-size: 0.72rem;
    color: #555;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.score-box-value {
    font-size: 1.7rem;
    font-weight: 700;
    line-height: 1;
}
.score-box-sub {
    font-size: 0.75rem;
    color: #555;
    margin-top: 4px;
}

/* Low-confidence warning badge */
.low-conf-badge {
    display: inline-block;
    background: rgba(255, 165, 0, 0.12);
    border: 1px solid rgba(255, 165, 0, 0.4);
    color: #FFA500;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 1px;
    padding: 5px 12px;
    border-radius: 4px;
    margin-bottom: 18px;
}

/* Custom progress bar inside result card */
.meter-label {
    font-size: 0.88rem;
    color: #555;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.meter-track {
    background: #2A2A2A;
    border-radius: 6px;
    height: 10px;
    width: 100%;
    overflow: hidden;
}
.meter-fill-green {
    background: linear-gradient(90deg, #27AE60, #2ECC71);
    height: 100%;
    border-radius: 6px;
    transition: width 0.8s cubic-bezier(0.22, 1, 0.36, 1);
}
.meter-fill-red {
    background: linear-gradient(90deg, #8B0000, #E50914);
    height: 100%;
    border-radius: 6px;
    transition: width 0.8s cubic-bezier(0.22, 1, 0.36, 1);
}

/* ---------- Footer ---------- */
.footer {
    text-align: center;
    margin-top: 64px;
    padding: 22px;
    border-top: 1px solid #222;
    color: #FFFFFF;
    font-size: 0.88rem;
    letter-spacing: 1px;
}
.footer em { color: #E50914; font-style: normal; }
</style>
""", unsafe_allow_html=True)


# ── Load model + word index once (cached across all users/sessions) ────────────
@st.cache_resource
def load_resources():
    """Load the trained SimpleRNN model and the IMDB word-to-index mapping."""
    model      = load_model("imdb_rnn_model.h5",
                            custom_objects={"SimpleRNN": _CompatSimpleRNN})
    word_index = imdb.get_word_index()
    return model, word_index

model, word_index = load_resources()


# ── Preprocessing helper ───────────────────────────────────────────────────────
def preprocess(text: str) -> np.ndarray:
    """
    Convert a raw review string into a padded integer sequence.
    Steps: strip punctuation → lowercase → split → encode → pad to length 500.
    Punctuation must be removed first — "good." won't match "good" in the
    word index and would be treated as unknown, hurting accuracy.
    Returns shape (1, 500) — the exact input the model expects.
    """
    text    = re.sub(r"[^a-zA-Z0-9\s]", " ", text).lower()  # remove punctuation
    words   = text.split()
    encoded = [word_index.get(w, 2) + 3 for w in words]     # +3 for reserved tokens
    padded  = sequence.pad_sequences([encoded], maxlen=500)
    return padded


# ── Prediction helper ──────────────────────────────────────────────────────────
def predict(review: str):
    """
    Run inference on a review string.
    Returns (label: str, raw_score: float).
      raw_score ≥ 0.5  →  Positive
      raw_score <  0.5  →  Negative
    """
    padded    = preprocess(review)
    raw_score = model.predict(padded)
    label     = "Positive" if raw_score[0][0] >= 0.5 else "Negative"
    return label, raw_score[0][0]


# ── Session state: lets example buttons pre-fill the text area ────────────────
if "review" not in st.session_state:
    st.session_state.review = ""

# Quick-try example reviews
EXAMPLES = {
    "👍 Great film":    "An interesting attempt but ultimately falls flat. Some scenes shine while others drag on without any real purpose.",
    "👎 Terrible film": "One of the worst films I have ever seen. The plot made no sense and the acting was painfully wooden throughout.",
    "🤔 Mixed feelings": "This movie was absolutely brilliant. The storytelling was captivating and the performances needs to be better.",
}


# ═══════════════════════════════════════════════════════════════════════════════
#  UI LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════

# ── Navbar ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="navbar">
    <div>
        <div class="brand-name">🎬 CineScope</div>
        <div class="brand-tagline">AI Sentiment Engine</div>
    </div>
    <div class="user-block">
        <div class="user-name">Lohith M</div>
        <div class="user-role">AI &amp; ML Engineer</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">DECODE THE <span class="accent">FEELING</span></div>
    <div class="hero-sub">Powered by a Simple RNN trained on 50,000 IMDB reviews</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ── Example pick chips ─────────────────────────────────────────────────────────
st.markdown('<div class="section-label">⚡ Quick examples — click to load</div>', unsafe_allow_html=True)
cols = st.columns(len(EXAMPLES))
for col, (label, text) in zip(cols, EXAMPLES.items()):
    if col.button(label, use_container_width=True):
        st.session_state.review = text

# ── Review text area ───────────────────────────────────────────────────────────
st.markdown('<div class="section-label" style="margin-top:22px;">✍️ Your movie review</div>',
            unsafe_allow_html=True)

review = st.text_area(
    label="review",                       # hidden via CSS
    value=st.session_state.review,
    placeholder='"A gripping thriller with an unexpected twist that left me speechless…"',
    height=160,
)

# ── Analyze button ─────────────────────────────────────────────────────────────
analyze = st.button("🎬  Analyze Sentiment", use_container_width=True)

# ── Result ─────────────────────────────────────────────────────────────────────
if analyze:
    if not review.strip():
        st.warning("Please enter a movie review before analyzing.")
    else:
        with st.spinner("Running the model…"):
            label, raw_score = predict(review)

        is_positive = label == "Positive"
        confidence  = raw_score if is_positive else 1.0 - raw_score
        conf_pct    = confidence * 100

        card_cls  = "result-positive" if is_positive else "result-negative"
        label_cls = "color-green"     if is_positive else "color-red"
        icon      = "🎉" if is_positive else "💀"

        html = (
            f'<div class="{card_cls}">'
            f'<div class="result-icon">{icon}</div>'
            f'<div class="result-label {label_cls}">{label} Review</div>'
            f'<div class="score-box-label" style="margin-top:12px;">Prediction Score</div>'
            f'<div class="score-box-value {label_cls}">{conf_pct:.1f}%</div>'
            f'</div>'
        )
        st.markdown(html, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Built with <em>❤</em> using TensorFlow &amp; Streamlit &nbsp;·&nbsp;
    Stanford IMDB Dataset &nbsp;·&nbsp; SimpleRNN Architecture
</div>
""", unsafe_allow_html=True)