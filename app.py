import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix
import time

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="Heart Disease ML Predictor", layout="wide", page_icon="🫀")

st.markdown("""
<style>
/* Base */
.main { background-color: #f4faf4; }
h1, h2, h3, h4 { color: #1b5e20 !important; }

/* Buttons */
.stButton>button {
    background-color: #2e7d32; color: white; border: none;
    font-weight: bold; border-radius: 8px; padding: 0.55rem 2rem;
    font-size: 15px; width: 100%;
}
.stButton>button:hover { background-color: #1b5e20; color: #a5d6a7; }

/* Metric cards */
.metric-card {
    background: #ffffff; border-radius: 12px; padding: 16px 20px;
    border-left: 5px solid #4caf50;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06); margin-bottom: 12px;
}
.metric-title { font-size: 12px; color: #555; margin: 0 0 4px; text-transform: uppercase; letter-spacing: 0.05em; }
.metric-value { font-size: 28px; font-weight: 700; margin: 0; color: #1b5e20; }
.metric-sub   { font-size: 11px; color: #888; margin: 4px 0 0; }

/* Info banner */
.info-banner {
    background: linear-gradient(135deg, #e8f5e9 0%, #f1f8e9 100%);
    border: 1px solid #a5d6a7; border-radius: 14px;
    padding: 18px 24px; margin-bottom: 18px;
}
.info-banner h4 { margin: 0 0 6px; font-size: 15px; color: #1b5e20 !important; }
.info-banner p  { margin: 0; font-size: 13px; color: #33691e; line-height: 1.6; }

/* Pipeline steps */
.pipeline-wrap { display: flex; gap: 0; margin: 10px 0 20px; }
.pipe-step {
    flex: 1; text-align: center; padding: 12px 6px;
    background: #e8f5e9; border: 1px solid #a5d6a7;
    font-size: 12px; font-weight: 600; color: #1b5e20;
    position: relative;
}
.pipe-step:first-child { border-radius: 10px 0 0 10px; }
.pipe-step:last-child  { border-radius: 0 10px 10px 0; }
.pipe-step .icon { font-size: 20px; display: block; margin-bottom: 4px; }
.pipe-highlight {
    background: #2e7d32 !important; color: white !important; border-color: #1b5e20 !important;
}
.pipe-highlight .step-label { color: #a5d6a7 !important; }
.step-label { font-size: 10px; font-weight: 400; color: #558b2f; display: block; margin-top: 2px; }

/* Dataset info table */
.ds-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.ds-table td { padding: 7px 10px; border-bottom: 1px solid #e8f5e9; }
.ds-table td:first-child { color: #558b2f; font-weight: 600; width: 45%; }
.ds-table td:last-child  { color: #1b5e20; }

/* Feature badges */
.feat-badge {
    display: inline-block; background: #e8f5e9; color: #1b5e20;
    border: 1px solid #a5d6a7; border-radius: 20px;
    padding: 3px 12px; font-size: 12px; margin: 3px 3px;
}

/* Result boxes */
.result-high {
    background: #ffebee; border-left: 6px solid #c62828;
    border-radius: 12px; padding: 18px 22px;
    color: #b71c1c; font-size: 20px; font-weight: 700;
}
.result-low {
    background: #e8f5e9; border-left: 6px solid #2e7d32;
    border-radius: 12px; padding: 18px 22px;
    color: #1b5e20; font-size: 20px; font-weight: 700;
}
.result-sub { font-size: 13px; font-weight: 400; margin-top: 6px; opacity: 0.85; }

/* Method highlight boxes */
.method-box {
    border-radius: 12px; padding: 16px 18px; margin-bottom: 12px;
    border: 1px solid;
}
.method-pca  { background: #e3f2fd; border-color: #90caf9; }
.method-rf   { background: #fff8e1; border-color: #ffe082; }
.method-box h4 { margin: 0 0 6px; font-size: 14px; }
.method-pca h4  { color: #0d47a1 !important; }
.method-rf  h4  { color: #e65100 !important; }
.method-box p  { margin: 0; font-size: 12px; line-height: 1.6; color: #333; }

/* Section divider label */
.section-tag {
    display: inline-block; background: #2e7d32; color: white;
    font-size: 11px; font-weight: 600; padding: 3px 12px;
    border-radius: 20px; margin-bottom: 10px; letter-spacing: 0.05em;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#1b5e20 0%,#2e7d32 60%,#388e3c 100%);
     border-radius:16px; padding:28px 32px; margin-bottom:24px; color:white;">
  <h1 style="color:white !important; margin:0 0 8px; font-size:30px;">
    🫀 Heart Disease ML Predictor
  </h1>
  <p style="margin:0; opacity:0.88; font-size:15px; line-height:1.7;">
    Powered by <b>PCA Dimensionality Reduction</b> + <b>Random Forest Classifier</b><br>
    Trained on the <b>UCI Heart Disease Dataset</b> sourced from
    <a href="https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data"
       style="color:#a5d6a7;">Kaggle (redwankarimsony)</a> —
    920+ patient records · 13 clinical features · Multi-source (Cleveland + Hungarian + Swiss + VA)
  </p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# ML PIPELINE VISUAL
# ─────────────────────────────────────────────
st.markdown('<div class="section-tag">ML PIPELINE</div>', unsafe_allow_html=True)
st.markdown("""
<div class="pipeline-wrap">
  <div class="pipe-step">
    <span class="icon">📦</span>Kaggle Dataset
    <span class="step-label">920+ rows · 13 features</span>
  </div>
  <div class="pipe-step">
    <span class="icon">🧹</span>Preprocessing
    <span class="step-label">Impute · Encode · Scale</span>
  </div>
  <div class="pipe-step pipe-highlight">
    <span class="icon">🔻</span>PCA
    <span class="step-label" style="color:#a5d6a7;">Dimensionality Reduction</span>
  </div>
  <div class="pipe-step pipe-highlight">
    <span class="icon">🌲</span>Random Forest
    <span class="step-label" style="color:#a5d6a7;">100 Trees · depth=5</span>
  </div>
  <div class="pipe-step">
    <span class="icon">🎯</span>Prediction
    <span class="step-label">Risk Probability</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATASET + METHOD INFO CARDS
# ─────────────────────────────────────────────
d1, d2, d3 = st.columns(3)

with d1:
    st.markdown("""
    <div class="info-banner">
      <h4>📦 Dataset — Kaggle</h4>
      <table class="ds-table">
        <tr><td>Source</td><td>redwankarimsony/heart-disease-data</td></tr>
        <tr><td>Origin</td><td>UCI ML Repository</td></tr>
        <tr><td>Sources</td><td>Cleveland · Hungarian · Swiss · VA</td></tr>
        <tr><td>Rows</td><td>920+ patient records</td></tr>
        <tr><td>Features</td><td>13 clinical indicators</td></tr>
        <tr><td>Target</td><td>num (0 = healthy, 1–4 = disease)</td></tr>
        <tr><td>Licence</td><td>CC BY 4.0</td></tr>
      </table>
    </div>
    """, unsafe_allow_html=True)

with d2:
    st.markdown("""
    <div class="method-box method-pca">
      <h4>🔻 Method 1 — PCA (Dimensionality Reduction)</h4>
      <p>
        Principal Component Analysis reduces 13 correlated clinical features
        into <b>2 principal components</b> that capture maximum variance.<br><br>
        <b>Why PCA?</b> Many features overlap (e.g. cp, exang, thal all measure
        cardiac stress). PCA removes redundancy, reduces noise, and enables
        2D visualisation of patient clusters.<br><br>
        <b>Key output:</b> Explained Variance Ratio · Reconstruction Error · Scree plot
      </p>
    </div>
    """, unsafe_allow_html=True)

with d3:
    st.markdown("""
    <div class="method-box method-rf">
      <h4>🌲 Method 2 — Random Forest Classifier</h4>
      <p>
        An ensemble of <b>100 decision trees</b> trained on PCA-reduced features
        to predict heart disease probability.<br><br>
        <b>Why Random Forest?</b> Handles mixed data types, resistant to
        overfitting via bagging, provides feature importance scores, and
        outperforms single decision trees on medical datasets.<br><br>
        <b>Hyperparameters:</b> n_estimators=100 · max_depth=5 · random_state=42
      </p>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD & PREPARE DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_and_prepare():
    url = "https://raw.githubusercontent.com/kb22/Heart-Disease-Prediction/master/dataset.csv"
    df = pd.read_csv(url)

    col_map = {'target': 'num', 'thalach': 'thalch'}
    df.rename(columns=col_map, inplace=True)

    y = (df['num'] > 0).astype(int) if 'num' in df.columns else (df.iloc[:, -1] > 0).astype(int)

    thalch_col = 'thalch' if 'thalch' in df.columns else 'thalach'
    X = df[['cp', 'chol', thalch_col, 'oldpeak', 'age']].copy()
    X.columns = ['cp', 'chol', 'thalch', 'oldpeak', 'age']

    X['cp'] = X['cp'].fillna(X['cp'].mode()[0])
    X[['chol','thalch','oldpeak','age']] = X[['chol','thalch','oldpeak','age']].fillna(
        X[['chol','thalch','oldpeak','age']].median()
    )

    le_cp = LabelEncoder()
    X['cp'] = le_cp.fit_transform(X['cp'].astype(str))
    return X, y, le_cp, df

with st.spinner("Loading Kaggle dataset & training model..."):
    X, y, le_cp, raw_df = load_and_prepare()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca_full = PCA()
    pca_full.fit(X_scaled)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    evr     = float(np.sum(pca.explained_variance_ratio_))
    recon_err = float(mean_squared_error(X_scaled, pca.inverse_transform(X_pca)))

    # Random Forest
    t0 = time.time()
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X, y)
    train_time = time.time() - t0

    acc = float(accuracy_score(y, model.predict(X)))
    cv_scores = cross_val_score(model, X, y, cv=5)


# ─────────────────────────────────────────────
# METRICS ROW
# ─────────────────────────────────────────────
st.divider()
st.markdown('<div class="section-tag">MODEL PERFORMANCE</div>', unsafe_allow_html=True)

m1, m2, m3, m4, m5 = st.columns(5)
metrics = [
    ("Model Accuracy",        f"{acc:.2%}",           "Random Forest on training set"),
    ("CV Score (5-Fold)",     f"{cv_scores.mean():.2%}", f"±{cv_scores.std():.3f} std"),
    ("PCA Variance Captured", f"{evr:.1%}",            "By 2 principal components"),
    ("Reconstruction Error",  f"{recon_err:.4f}",      "PCA inverse transform MSE"),
    ("Training Time",         f"{train_time:.2f}s",    "Random Forest fit time"),
]
for col, (title, val, sub) in zip([m1,m2,m3,m4,m5], metrics):
    with col:
        st.markdown(f"""
        <div class="metric-card">
          <p class="metric-title">{title}</p>
          <p class="metric-value">{val}</p>
          <p class="metric-sub">{sub}</p>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CHARTS ROW
# ─────────────────────────────────────────────
st.markdown('<div class="section-tag">VISUALISATIONS</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)

# Chart 1 — PCA Scatter
with c1:
    st.markdown("**2D PCA Patient Projection**")
    fig, ax = plt.subplots(figsize=(5, 4))
    colors = ['#43a047' if v == 0 else '#e53935' for v in y]
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.6, edgecolors='white', linewidths=0.4, s=45)
    ax.set_xlabel("PC 1", fontsize=10); ax.set_ylabel("PC 2", fontsize=10)
    ax.set_title("PCA — 2 Components", fontsize=11, fontweight='bold')
    ax.set_facecolor("#f4faf4"); fig.patch.set_facecolor("#f4faf4")
    healthy_p = mpatches.Patch(color='#43a047', label='Healthy')
    disease_p = mpatches.Patch(color='#e53935', label='Disease')
    ax.legend(handles=[healthy_p, disease_p], fontsize=9)
    st.pyplot(fig)

# Chart 2 — Scree Plot
with c2:
    st.markdown("**Scree Plot — Explained Variance**")
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    evr_all  = pca_full.explained_variance_ratio_
    cum_evr  = np.cumsum(evr_all)
    comps    = range(1, len(evr_all) + 1)
    ax2.bar(comps, evr_all * 100, color='#81c784', edgecolor='#2e7d32', linewidth=0.6, label='Individual')
    ax2.plot(comps, cum_evr * 100, 'o-', color='#e53935', linewidth=1.8, markersize=4, label='Cumulative')
    ax2.axhline(90, color='#1b5e20', linestyle='--', linewidth=1, alpha=0.6)
    ax2.set_xlabel("Principal Component", fontsize=10)
    ax2.set_ylabel("Variance Explained (%)", fontsize=10)
    ax2.set_title("PCA Scree Plot", fontsize=11, fontweight='bold')
    ax2.set_facecolor("#f4faf4"); fig2.patch.set_facecolor("#f4faf4")
    ax2.legend(fontsize=9)
    st.pyplot(fig2)

# Chart 3 — Feature Importance
with c3:
    st.markdown("**Random Forest — Feature Importance**")
    feat_names = ['Chest Pain (cp)', 'Cholesterol', 'Max HR (thalch)', 'ST Depression', 'Age']
    importances = model.feature_importances_
    idx = np.argsort(importances)
    fig3, ax3 = plt.subplots(figsize=(5, 4))
    bars = ax3.barh([feat_names[i] for i in idx], importances[idx],
                    color='#a5d6a7', edgecolor='#2e7d32', linewidth=0.6)
    bars[-1].set_color('#2e7d32')
    ax3.set_xlabel("Importance Score", fontsize=10)
    ax3.set_title("Feature Importance", fontsize=11, fontweight='bold')
    ax3.set_facecolor("#f4faf4"); fig3.patch.set_facecolor("#f4faf4")
    st.pyplot(fig3)


# ─────────────────────────────────────────────
# FEATURES USED
# ─────────────────────────────────────────────
st.markdown('<div class="section-tag">5 KEY FEATURES USED</div>', unsafe_allow_html=True)
st.markdown("""
<div style="background:#f1f8e9; border-radius:12px; padding:14px 18px; margin-bottom:8px;">
  <span class="feat-badge">🎂 age — Patient age (years)</span>
  <span class="feat-badge">💢 cp — Chest pain type (0–3)</span>
  <span class="feat-badge">🩸 chol — Serum cholesterol (mg/dl)</span>
  <span class="feat-badge">❤️ thalch — Max heart rate achieved</span>
  <span class="feat-badge">📉 oldpeak — ST depression (exercise vs rest)</span>
  <p style="margin:10px 0 0; font-size:12px; color:#558b2f;">
    These 5 features were selected from the original 13 based on highest correlation with heart disease
    diagnosis in clinical literature and EDA analysis of the Kaggle dataset.
  </p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PREDICTION SECTION
# ─────────────────────────────────────────────
st.divider()
st.markdown('<div class="section-tag">LIVE PREDICTION</div>', unsafe_allow_html=True)
st.markdown("### 🔍 Predict Heart Disease Risk for a New Patient")
st.write("Enter the patient's 5 clinical values and click **Predict** to get the probability.")

p1, p2, p3 = st.columns(3)

with p1:
    u_age = st.slider("🎂 Age (years)", min_value=1, max_value=100, value=50)
    u_cp  = st.selectbox(
        "💢 Chest Pain Type (cp)",
        options=["0 — Typical Angina (Highest Risk)",
                 "1 — Atypical Angina",
                 "2 — Non-Anginal Pain",
                 "3 — Asymptomatic (No Pain)"],
        help="0 = classic heart-related pain · 3 = no chest pain at all"
    )

with p2:
    u_chol   = st.number_input("🩸 Cholesterol (mg/dl)", min_value=100, max_value=600, value=200,
                                help="Normal < 200 · Borderline 200–239 · High ≥ 240")
    u_thalch = st.number_input("❤️ Max Heart Rate (thalch)", min_value=60, max_value=220, value=150,
                                help="Max HR during exercise test. Higher = healthier response.")

with p3:
    u_oldpeak = st.number_input("📉 ST Depression (oldpeak)", min_value=0.0, max_value=10.0,
                                 value=1.0, step=0.1,
                                 help="ST segment depression during exercise. 0 = normal · >2 = abnormal")
    st.markdown("""
    <div style="background:#e8f5e9; border-radius:8px; padding:10px 12px; margin-top:8px; font-size:11px; color:#33691e; line-height:1.6;">
      <b>How this works:</b><br>
      Your inputs → StandardScaler → PCA (2 components) →
      Random Forest (100 trees) → Disease probability
    </div>
    """, unsafe_allow_html=True)

st.write("")
u_cp_num = int(u_cp.split(" ")[0])

if st.button("🔬 Calculate Heart Disease Probability"):
    try:
        u_cp_enc = le_cp.transform([str(u_cp_num)])[0]
    except Exception:
        u_cp_enc = u_cp_num

    input_row = pd.DataFrame(
        [[u_cp_enc, u_chol, u_thalch, u_oldpeak, u_age]],
        columns=['cp', 'chol', 'thalch', 'oldpeak', 'age']
    )

    chance = model.predict_proba(input_row)[0][1]

    st.write("")
    if chance > 0.5:
        st.markdown(f"""
        <div class="result-high">
          ⚠️ High Risk — {chance:.1%} Probability of Heart Disease
          <div class="result-sub">Recommend further clinical investigation and specialist referral.</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-low">
          ✅ Low Risk — {chance:.1%} Probability of Heart Disease
          <div class="result-sub">No immediate concern based on these 5 indicators. Regular check-ups advised.</div>
        </div>""", unsafe_allow_html=True)

    # Probability bar
    st.write("")
    st.markdown("**Risk Probability Breakdown**")
    prob_df = pd.DataFrame({
        'Outcome': ['No Heart Disease', 'Heart Disease'],
        'Probability': [f"{(1-chance):.1%}", f"{chance:.1%}"],
        'Score': [round((1-chance)*100, 1), round(chance*100, 1)]
    })
    st.dataframe(prob_df[['Outcome','Probability']], use_container_width=True, hide_index=True)

    # Input summary
    st.write("")
    st.markdown("**📋 Patient Input Summary**")
    summary = pd.DataFrame({
        "Feature":     ["Age", "Chest Pain Type", "Cholesterol", "Max Heart Rate", "ST Depression (oldpeak)"],
        "Value":       [u_age, u_cp, u_chol, u_thalch, u_oldpeak],
        "Normal Range":["29–77 yrs", "0–3", "< 200 mg/dl", "60–202 bpm", "0–0.5 (normal)"]
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.info("⚠️ This tool is for educational purposes only. Not a substitute for professional medical diagnosis.")


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="display:flex; justify-content:space-between; flex-wrap:wrap; gap:8px;
     font-size:12px; color:#558b2f; padding:4px 0;">
  <span>📦 <b>Dataset:</b> UCI Heart Disease · Kaggle (redwankarimsony) · CC BY 4.0</span>
  <span>🔻 <b>DR Method:</b> PCA — 2 principal components</span>
  <span>🌲 <b>Classifier:</b> Random Forest · 100 estimators · depth 5</span>
  <span>🎓 <b>Built for:</b> ML Practical Assignment — Dept. of CSE</span>
</div>
""", unsafe_allow_html=True)