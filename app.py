"""
🌾 Smart Crop Advisor v3 — Full-Featured Streamlit App
Run:  streamlit run app.py

New extensions over v2:
  • Soil Type & Texture panel
  • Organic Matter (OM) Content grader
  • Pest / Disease Risk Index
  • Water Requirement vs Availability planner
  • Multi-language UI (10 languages)
  • In-app Smart Notification / Alert centre
"""

import os, pickle, json, datetime, math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from data import (
    CROP_EMOJI, CROP_INFO, CROP_IDEAL,
    SOIL_TYPES, OM_LEVELS, OM_VALUE_MAP, WATER_SOURCES,
    LANGUAGES, t,
)

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="🌾 Smart Crop Advisor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
# SESSION STATE DEFAULTS
# ══════════════════════════════════════════════════════════════
for key, default in {
    "lang":         "en",
    "alerts":       [],
    "last_pred":    None,
    "notif_count":  0,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ══════════════════════════════════════════════════════════════
# HISTORY HELPERS
# ══════════════════════════════════════════════════════════════
HISTORY_FILE = os.path.join(os.path.dirname(__file__), "prediction_history.json")

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE) as f:
            return json.load(f)
    return []

def save_history(record):
    h = load_history()
    h.append(record)
    with open(HISTORY_FILE, "w") as f:
        json.dump(h[-60:], f)

# ══════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "crop_model.pkl")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        from generate_and_train import train_model
        with st.spinner("Training model for the first time…"):
            train_model()
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

model = load_model()

# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════
def field_match_score(crop, user_vals):
    ideal = CROP_IDEAL.get(crop, {})
    rngs  = dict(N=(0,140), P=(5,145), K=(5,205),
                 temperature=(0,50), humidity=(10,100),
                 ph=(3.5,9.5), rainfall=(20,300))
    scores = {k: max(0, 100 - abs(user_vals.get(k,0) - ideal.get(k,0))
                          / (rngs[k][1]-rngs[k][0]) * 200)
              for k in rngs if k in ideal}
    return scores, float(np.mean(list(scores.values()))) if scores else 0.0

def pest_risk_score(crop, humidity, temperature, rainfall, om_val):
    """0-100 composite pest/disease risk index."""
    base = {"Very High":85,"High":65,"Medium":45,"Low":25,"Very Low":10}
    info  = CROP_INFO.get(crop, {})
    risk  = base.get(info.get("pest_risk","Medium"), 45)
    # humidity drives fungal risk
    if humidity > 80:  risk += 10
    elif humidity < 40: risk -= 8
    # high rainfall
    if rainfall > 200: risk += 8
    # high OM can harbour pests
    if om_val > 3.0:   risk += 5
    # hot temperatures accelerate insect cycles
    if temperature > 32: risk += 7
    return min(100, max(0, risk))

def water_status(crop, water_source, rainfall_mm, field_area_ha=1.0):
    """Return (required_mm, available_mm, gap_mm, status_str)."""
    info     = CROP_INFO.get(crop, {})
    required = info.get("water_mm", 800)
    source_yield = {   # rough mm/season from each source
        "Rainfed only":         rainfall_mm,
        "Borewell / Groundwater":  600,
        "Canal / River":        900,
        "Drip Irrigation":      required * 0.75,   # 25 % saving
        "Sprinkler Irrigation": required * 0.85,
        "Tank / Pond":          400,
    }
    avail  = source_yield.get(water_source, rainfall_mm)
    avail += rainfall_mm * 0.4 if water_source != "Rainfed only" else 0
    avail  = min(avail, required * 1.5)
    gap    = required - avail
    if gap <= 0:
        status = "✅ Sufficient"
    elif gap < 150:
        status = "🟡 Marginal"
    else:
        status = "❌ Deficit"
    return required, avail, gap, status

def push_alert(msg, level="info"):
    """Add to session alert queue and bump badge count."""
    st.session_state.alerts.append({
        "msg":   msg,
        "level": level,
        "time":  datetime.datetime.now().strftime("%H:%M"),
    })
    st.session_state.notif_count += 1

def generate_alerts(crop, confidence, overall_score, pest_idx,
                    water_status_str, om_key, soil_type, lang):
    """Auto-generate contextual alerts after prediction."""
    alerts = []
    info   = CROP_INFO.get(crop, {})
    em     = CROP_EMOJI.get(crop, "🌿")

    if confidence >= 80:
        alerts.append((f"🎯 {em} {crop} is an excellent match ({confidence:.0f}% confidence)!", "success"))
    elif confidence < 55:
        alerts.append((f"⚠️ Low model confidence ({confidence:.0f}%). Consider adjusting inputs.", "warning"))

    if overall_score < 50:
        alerts.append(("🧪 Soil conditions are significantly off from ideal. Review nutrient levels.", "warning"))

    if pest_idx >= 70:
        alerts.append((f"🐛 HIGH pest/disease risk for {crop}. Prepare preventive measures.", "error"))
    elif pest_idx >= 50:
        alerts.append((f"🐛 Moderate pest risk. Monitor for: {', '.join(info.get('diseases',['—'])[:2])}.", "warning"))

    if water_status_str == "❌ Deficit":
        alerts.append(("💧 Water DEFICIT detected. Consider drip/sprinkler irrigation or a drought-tolerant crop.", "error"))
    elif water_status_str == "🟡 Marginal":
        alerts.append(("💧 Water supply is marginal. Conserve with mulching and reduced irrigation frequency.", "warning"))

    om_score = OM_LEVELS.get(om_key, {}).get("score", 3)
    min_om   = info.get("min_om", 1.0)
    if OM_VALUE_MAP.get(om_key, 1.5) < min_om:
        alerts.append((f"🌱 Organic matter too low for {crop} (needs ≥{min_om}%). Add compost/FYM.", "warning"))

    ideal_soils = info.get("soil_types", [])
    if soil_type and soil_type not in ideal_soils:
        alerts.append((f"🪨 '{soil_type}' soil is not ideal for {crop}. Best: {', '.join(ideal_soils)}.", "info"))

    for msg, level in alerts:
        push_alert(msg, level)

# ══════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}

.hero{background:linear-gradient(135deg,#1b5e20 0%,#2e7d32 55%,#388e3c 100%);
      border-radius:18px;padding:1.8rem 2.5rem 1.5rem;margin-bottom:1.2rem;color:white;}
.hero h1{font-size:2.2rem;font-weight:800;margin:0;}
.hero p{opacity:.85;margin:.3rem 0 0;font-size:1rem;}

.result-card{background:linear-gradient(135deg,#e8f5e9,#f1f8e9);border:2.5px solid #66bb6a;
             border-radius:18px;padding:1.6rem 2rem;text-align:center;
             box-shadow:0 4px 20px rgba(102,187,106,.25);margin-bottom:.8rem;}
.crop-name{font-size:1.8rem;font-weight:800;color:#1b5e20;margin:.25rem 0;}
.conf-badge{display:inline-block;background:#2e7d32;color:white;border-radius:20px;
            padding:.2rem .9rem;font-size:.88rem;font-weight:700;margin-bottom:.4rem;}
.meta-pill{display:inline-block;background:#f1f8e9;border:1px solid #a5d6a7;
           border-radius:20px;padding:.18rem .75rem;font-size:.8rem;margin:.18rem;
           color:#2e7d32;font-weight:600;}

.notif-badge{background:#e53935;color:white;border-radius:50%;padding:.1rem .45rem;
             font-size:.75rem;font-weight:800;margin-left:.3rem;vertical-align:top;}
.alert-success{background:#e8f5e9;border-left:4px solid #2e7d32;
               border-radius:8px;padding:.7rem 1rem;margin:.35rem 0;font-size:.9rem;}
.alert-warning{background:#fff8e1;border-left:4px solid #f9a825;
               border-radius:8px;padding:.7rem 1rem;margin:.35rem 0;font-size:.9rem;}
.alert-error  {background:#ffebee;border-left:4px solid #c62828;
               border-radius:8px;padding:.7rem 1rem;margin:.35rem 0;font-size:.9rem;}
.alert-info   {background:#e3f2fd;border-left:4px solid #1565c0;
               border-radius:8px;padding:.7rem 1rem;margin:.35rem 0;font-size:.9rem;}

.soil-card{border-radius:12px;padding:1rem 1.2rem;margin:.4rem 0;
           border:1.5px solid #ddd;background:white;}
.om-bar-wrap{background:#f5f5f5;border-radius:8px;height:18px;width:100%;overflow:hidden;}
.om-bar{height:18px;border-radius:8px;transition:width .4s;}

.section-head{font-size:1rem;font-weight:700;color:#2e7d32;
              border-left:4px solid #66bb6a;padding-left:.6rem;margin:1rem 0 .5rem;}
.stButton>button{background:linear-gradient(135deg,#2e7d32,#43a047)!important;
                 color:white!important;border:none!important;border-radius:10px!important;
                 font-weight:700!important;font-size:.97rem!important;padding:.6rem 1.4rem!important;}
.placeholder-box{text-align:center;padding:3rem 1rem;color:#9e9e9e;
                 border:2px dashed #c8e6c9;border-radius:16px;}
.risk-pill{display:inline-block;border-radius:20px;padding:.25rem .9rem;
           font-size:.83rem;font-weight:700;margin:.2rem;}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# SIDEBAR — Language + Presets + Notification Centre
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    # ── Language selector ────────────────────────────────────
    st.markdown("### 🌐 Language / भाषा")
    lang_label = st.selectbox("", list(LANGUAGES.keys()), label_visibility="collapsed")
    lang = LANGUAGES[lang_label]
    st.session_state.lang = lang

    st.divider()

    # ── Quick presets ────────────────────────────────────────
    st.markdown("### ⚙️ Quick Presets")
    preset = st.selectbox("Load soil profile", [
        "— Custom —","Fertile Plains","Dry Arid Zone",
        "Tropical Humid","Cool Highland","Coastal Sandy",
    ])
    presets = {
        "Fertile Plains": dict(N=110,P=60,K=55,temperature=24,humidity=68,ph=6.5,rainfall=110),
        "Dry Arid Zone":  dict(N=30, P=40,K=25,temperature=33,humidity=30,ph=7.2,rainfall=35),
        "Tropical Humid": dict(N=75, P=55,K=40,temperature=29,humidity=85,ph=6.0,rainfall=220),
        "Cool Highland":  dict(N=20, P=30,K=40,temperature=12,humidity=85,ph=6.2,rainfall=160),
        "Coastal Sandy":  dict(N=15, P=10,K=40,temperature=30,humidity=78,ph=6.8,rainfall=175),
    }
    pv = presets.get(preset, {})

    st.divider()

    # ── Notification centre ──────────────────────────────────
    nc = st.session_state.notif_count
    badge = f'<span class="notif-badge">{nc}</span>' if nc > 0 else ""
    st.markdown(f'### {t("alerts_title", lang)}{badge}', unsafe_allow_html=True)

    alerts = st.session_state.alerts[-8:]  # show last 8
    if not alerts:
        st.caption("No alerts yet. Run a prediction to see smart notifications.")
    else:
        for a in reversed(alerts):
            css = f"alert-{a['level']}"
            st.markdown(f'<div class="{css}"><b>{a["time"]}</b> — {a["msg"]}</div>',
                        unsafe_allow_html=True)
        if st.button("Clear Alerts", key="clr_alerts"):
            st.session_state.alerts = []
            st.session_state.notif_count = 0
            st.rerun()

    st.divider()
    st.markdown("### 🌿 Crop Directory")
    cats = {}
    for crop, info in CROP_INFO.items():
        cats.setdefault(info["category"], []).append(crop)
    for cat, crops in cats.items():
        with st.expander(cat):
            for c in crops:
                st.write(f"{CROP_EMOJI.get(c,'🌿')} {c}")

    st.divider()
    st.caption("🤖 Random Forest · ~92% accuracy · 4 400 training samples")

# ══════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════
notif_badge = (f'<span class="notif-badge">{st.session_state.notif_count}</span>'
               if st.session_state.notif_count > 0 else "")
st.markdown(f"""
<div class="hero">
  <h1>{t('app_title',lang)}{notif_badge}</h1>
  <p>{t('subtitle',lang)}</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════
tabs = st.tabs([
    t("predict_tab",  lang),
    t("soil_tab",     lang),
    t("pest_tab",     lang),
    t("water_tab",    lang),
    t("analysis_tab", lang),
    t("explorer_tab", lang),
    t("history_tab",  lang),
])
tab_predict, tab_soil, tab_pest, tab_water, tab_analysis, tab_explorer, tab_history = tabs

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ════════════════════════════════════════════════════════════════════════════
with tab_predict:
    col_left, col_right = st.columns([1, 1.1], gap="large")

    with col_left:
        # ── Soil inputs ──────────────────────────────────────
        st.markdown(f'<div class="section-head">{t("soil_nutrients",lang)}</div>',
                    unsafe_allow_html=True)
        N = st.slider("Nitrogen (N)",   0,   140, int(pv.get("N",  50)))
        P = st.slider("Phosphorus (P)", 5,   145, int(pv.get("P",  50)))
        K = st.slider("Potassium (K)",  5,   205, int(pv.get("K",  50)))

        st.markdown(f'<div class="section-head">{t("environment",lang)}</div>',
                    unsafe_allow_html=True)
        temperature = st.slider("Temperature (°C)", 0.0,  50.0,
                                float(pv.get("temperature",25.0)), 0.5)
        humidity    = st.slider("Humidity (%)",     10.0, 100.0,
                                float(pv.get("humidity",   65.0)), 0.5)
        ph          = st.slider("Soil pH",          3.5,  9.5,
                                float(pv.get("ph",         6.5)),  0.1)
        rainfall    = st.slider("Rainfall (mm)",    20.0, 300.0,
                                float(pv.get("rainfall",  100.0)), 1.0)

        st.markdown('<div class="section-head">⚙️ Extra Context</div>',
                    unsafe_allow_html=True)
        soil_type   = st.selectbox("Soil Type", list(SOIL_TYPES.keys()))
        om_key      = st.selectbox("Organic Matter", list(OM_LEVELS.keys()))
        water_src   = st.selectbox("Water Source", WATER_SOURCES)
        field_area  = st.number_input("Field Area (hectares)", 0.1, 500.0, 1.0, 0.1)

        predict_btn = st.button(t("recommend_btn", lang), use_container_width=True)

    with col_right:
        if predict_btn:
            features   = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            prediction = model.predict(features)[0]
            proba      = model.predict_proba(features)[0]
            confidence = proba.max() * 100
            info       = CROP_INFO.get(prediction, {})
            emoji_c    = CROP_EMOJI.get(prediction, "🌿")
            user_vals  = dict(N=N,P=P,K=K,temperature=temperature,
                              humidity=humidity,ph=ph,rainfall=rainfall)
            scores, overall = field_match_score(prediction, user_vals)
            om_val     = OM_VALUE_MAP.get(om_key, 1.5)
            pest_idx   = pest_risk_score(prediction, humidity, temperature, rainfall, om_val)
            req_mm, avail_mm, gap_mm, w_status = water_status(
                prediction, water_src, rainfall, field_area)

            # push smart alerts
            st.session_state.alerts = []
            st.session_state.notif_count = 0
            generate_alerts(prediction, confidence, overall, pest_idx,
                            w_status, om_key, soil_type, lang)

            # save
            st.session_state.last_pred = {
                "crop": prediction, "confidence": round(confidence,1),
                "overall": round(overall,1), "pest_idx": round(pest_idx,1),
                "water_status": w_status,
            }

            # ── Result card ──────────────────────────────────
            st.markdown(f"""
            <div class="result-card">
              <div style="font-size:3.2rem">{emoji_c}</div>
              <div class="crop-name">{t('recommended_crop',lang)}: {prediction}</div>
              <div class="conf-badge">✅ {t('confidence',lang)}: {confidence:.1f}%</div><br>
              <span class="meta-pill">📅 {info.get('season','—')}</span>
              <span class="meta-pill">💧 {info.get('water_mm',0)} mm/season</span>
              <span class="meta-pill">⏱ {info.get('duration','—')}</span>
              <span class="meta-pill">💰 {info.get('profit','—')} profit</span>
              <span class="meta-pill">🏷 ₹{info.get('market_price','—')}/kg</span>
              <p style="color:#4e4e4e;font-size:.9rem;margin-top:.7rem">
                💡 {info.get('tip','')}
              </p>
            </div>
            """, unsafe_allow_html=True)

            # ── 4 metric pills ───────────────────────────────
            m1,m2,m3,m4 = st.columns(4)
            m1.metric("🩺 Soil Match",  f"{overall:.0f}%")
            m2.metric("🐛 Pest Risk",   f"{pest_idx:.0f}%")
            m3.metric("💧 Water",       w_status.split()[0])
            m4.metric("🌱 OM Score",
                      OM_LEVELS.get(om_key,{}).get("label","—"))

            # ── Soil Health gauge ────────────────────────────
            col_g = "#2e7d32" if overall>=70 else "#f57c00" if overall>=45 else "#c62828"
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=overall,
                title={"text":"Field–Crop Match","font":{"size":13}},
                gauge={"axis":{"range":[0,100]},"bar":{"color":col_g},
                       "steps":[{"range":[0,40],"color":"#ffebee"},
                                 {"range":[40,60],"color":"#fff8e1"},
                                 {"range":[60,80],"color":"#e8f5e9"},
                                 {"range":[80,100],"color":"#c8e6c9"}]},
                number={"suffix":"%","font":{"size":24}},
            ))
            gauge.update_layout(height=190, margin=dict(t=40,b=0,l=20,r=20),
                                paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(gauge, use_container_width=True)

            # ── Top-5 bar ────────────────────────────────────
            classes  = model.classes_
            top5_idx = proba.argsort()[::-1][:5]
            top5     = [(classes[i], proba[i]*100) for i in top5_idx]
            fig_bar  = go.Figure(go.Bar(
                x=[p for _,p in top5],
                y=[f"{CROP_EMOJI.get(c,'🌿')} {c}" for c,_ in top5],
                orientation="h",
                marker_color=["#2e7d32" if i==0 else "#81c784" for i in range(5)],
                text=[f"{p:.1f}%" for _,p in top5],
                textposition="inside",
            ))
            fig_bar.update_layout(title="Top 5 Matches",height=220,
                                  margin=dict(t=35,b=0,l=5,r=5),
                                  xaxis_title="Confidence %",
                                  yaxis={"autorange":"reversed"},
                                  paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_bar, use_container_width=True)

            # ── Param scores ─────────────────────────────────
            p_keys = ["N","P","K","temperature","humidity","ph","rainfall"]
            p_lbls = ["N","P","K","Temp","RH","pH","Rain"]
            p_vals = [scores.get(k,0) for k in p_keys]
            fig_p  = go.Figure(go.Bar(
                x=p_lbls, y=p_vals,
                marker_color=["#2e7d32" if v>=70 else "#f57c00" if v>=45
                              else "#e53935" for v in p_vals],
                text=[f"{v:.0f}%" for v in p_vals],
                textposition="outside",
            ))
            fig_p.update_layout(yaxis=dict(range=[0,115],title="Match %"),
                                height=240,margin=dict(t=10,b=10,l=5,r=5),
                                paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_p, use_container_width=True)

            # save history
            save_history({
                "time":prediction, "crop":prediction,
                "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                "confidence": round(confidence,1),
                "health_score": round(overall,1),
                "pest_idx": round(pest_idx,1),
                "water_status": w_status,
                "soil_type": soil_type,
                "om": om_key,
                "inputs": dict(N=N,P=P,K=K,temp=temperature,
                               humidity=humidity,ph=ph,rainfall=rainfall),
            })

            # ── Inline alerts banner ─────────────────────────
            st.markdown(f'<div class="section-head">{t("alerts_title",lang)}</div>',
                        unsafe_allow_html=True)
            for a in st.session_state.alerts:
                css = f"alert-{a['level']}"
                st.markdown(f'<div class="{css}">{a["msg"]}</div>',
                            unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="placeholder-box">
                <div style="font-size:3.5rem">🌾</div>
                <div style="font-size:1rem;font-weight:600;margin-top:.7rem;color:#555">
                    Fill in the sliders &amp; options, then click<br>
                    <strong style="color:#2e7d32">Get Recommendation</strong>
                </div>
            </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — SOIL TYPE & ORGANIC MATTER
# ════════════════════════════════════════════════════════════════════════════
with tab_soil:
    st.markdown(f"### {t('soil_tab',lang)} — Soil Type, Texture & Organic Matter")
    sc1, sc2 = st.columns([1,1], gap="large")

    with sc1:
        st.markdown('<div class="section-head">🪨 Soil Type Reference Card</div>',
                    unsafe_allow_html=True)
        sel_soil = st.selectbox("Select Soil Type to Explore", list(SOIL_TYPES.keys()),
                                key="soil_exp")
        sp = SOIL_TYPES[sel_soil]

        st.markdown(f"""
        <div class="soil-card" style="border-left:5px solid {sp['color']}">
          <div style="display:flex;align-items:center;gap:.7rem;margin-bottom:.5rem">
            <div style="width:28px;height:28px;border-radius:50%;
                        background:{sp['color']};border:2px solid #ddd"></div>
            <b style="font-size:1.1rem">{sel_soil}</b>
          </div>
          <table style="width:100%;font-size:.88rem;border-collapse:collapse">
            <tr><td style="color:#666;padding:.25rem 0">Texture</td>
                <td><b>{sp['texture']}</b></td></tr>
            <tr><td style="color:#666;padding:.25rem 0">Drainage</td>
                <td><b>{sp['drainage']}</b></td></tr>
            <tr><td style="color:#666;padding:.25rem 0">Water Hold</td>
                <td><b>{sp['water_hold']}</b></td></tr>
            <tr><td style="color:#666;padding:.25rem 0">Fertility</td>
                <td><b>{sp['fertility']}</b></td></tr>
          </table>
          <p style="color:#555;font-size:.85rem;margin-top:.5rem">📝 {sp['desc']}</p>
        </div>
        """, unsafe_allow_html=True)

        # Soil suitability matrix
        st.markdown('<div class="section-head">✅ Crops Suited to This Soil</div>',
                    unsafe_allow_html=True)
        suited = [f"{CROP_EMOJI.get(c,'🌿')} {c}" for c, info in CROP_INFO.items()
                  if sel_soil in info.get("soil_types", [])]
        if suited:
            for i in range(0, len(suited), 3):
                cols = st.columns(3)
                for j, col in enumerate(cols):
                    if i+j < len(suited):
                        col.markdown(f"<span class='meta-pill'>{suited[i+j]}</span>",
                                     unsafe_allow_html=True)
        else:
            st.info("No exact matches. This soil may still support crops with amendments.")

    with sc2:
        st.markdown('<div class="section-head">🌱 Organic Matter (OM) Analyser</div>',
                    unsafe_allow_html=True)
        om_sel = st.selectbox("Your OM Level", list(OM_LEVELS.keys()), key="om_sel")
        omp    = OM_LEVELS[om_sel]
        om_val_now = OM_VALUE_MAP[om_sel]

        # OM progress bar
        pct    = min(100, om_val_now / 5.0 * 100)
        st.markdown(f"""
        <div style="margin:.5rem 0">
          <div style="display:flex;justify-content:space-between;margin-bottom:.3rem">
            <b>{om_sel}</b>
            <span style="color:{omp['color']};font-weight:700">{omp['label']}</span>
          </div>
          <div class="om-bar-wrap">
            <div class="om-bar" style="width:{pct:.0f}%;background:{omp['color']}"></div>
          </div>
          <div style="color:#666;font-size:.85rem;margin-top:.4rem">
            💡 {omp['advice']}
          </div>
        </div>
        """, unsafe_allow_html=True)

        # OM vs crop requirements radar bar
        st.markdown('<div class="section-head">📊 OM vs Crop Requirements</div>',
                    unsafe_allow_html=True)
        om_crop_rows = [(c, CROP_INFO[c].get("min_om",1.0)) for c in CROP_INFO]
        om_crop_rows.sort(key=lambda x: x[1])
        om_df = pd.DataFrame(om_crop_rows, columns=["Crop","Min OM %"])
        om_df["Your OM"] = om_val_now
        om_df["Status"]  = om_df["Min OM %"].apply(
            lambda m: "✅ OK" if om_val_now >= m else "❌ Low")
        om_df["Crop"]    = om_df["Crop"].apply(lambda c: f"{CROP_EMOJI.get(c,'🌿')} {c}")

        fig_om = go.Figure()
        fig_om.add_trace(go.Bar(name="Min Required",
                                x=om_df["Min OM %"], y=om_df["Crop"],
                                orientation="h", marker_color="#81c784",
                                text=om_df["Min OM %"].apply(lambda v: f"{v}%"),
                                textposition="inside"))
        fig_om.add_vline(x=om_val_now, line_color="#e53935", line_width=2,
                         annotation_text=f"Your OM: {om_val_now}%",
                         annotation_position="top right")
        fig_om.update_layout(height=600, margin=dict(t=10,b=10,l=5,r=5),
                             xaxis_title="Organic Matter %",
                             yaxis={"autorange":"reversed"},
                             paper_bgcolor="rgba(0,0,0,0)",
                             showlegend=False)
        st.plotly_chart(fig_om, use_container_width=True)

    # ── Soil texture triangle (simplified) ────────────────────
    st.markdown('<div class="section-head">🔺 Soil Texture Comparison</div>',
                unsafe_allow_html=True)
    texture_data = [
        {"Soil": k,
         "Drainage Score":  {"Excellent":5,"Good":4,"Moderate":3,"Poor":2,"Very Poor":1}.get(v["drainage"],3),
         "Water Hold":      {"Very High":5,"High":4,"Medium":3,"Low":2,"Very Low":1}.get(v["water_hold"],3),
         "Fertility Score": {"Very High":5,"High":4,"Medium":3,"Low":2,"Very Low":1}.get(v["fertility"],3),
        } for k,v in SOIL_TYPES.items()
    ]
    tx_df = pd.DataFrame(texture_data)
    fig_tx = go.Figure()
    colors = ["#f4d03f","#e67e22","#784212","#922b21","#641e16","#1a1a1a","#c0392b","#aab7b8"]
    for i,row in tx_df.iterrows():
        fig_tx.add_trace(go.Scatterpolar(
            r=[row["Drainage Score"],row["Water Hold"],row["Fertility Score"],row["Drainage Score"]],
            theta=["Drainage","Water Hold","Fertility","Drainage"],
            fill="toself", name=row["Soil"],
            line_color=colors[i % len(colors)], opacity=0.6,
        ))
    fig_tx.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,6])),
                         height=420, paper_bgcolor="rgba(0,0,0,0)",
                         margin=dict(t=20,b=20,l=20,r=20))
    st.plotly_chart(fig_tx, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — PEST & DISEASE RISK INDEX
# ════════════════════════════════════════════════════════════════════════════
with tab_pest:
    st.markdown(f"### {t('pest_tab',lang)} — Risk Index")

    pc1, pc2 = st.columns([1,1], gap="large")
    with pc1:
        st.markdown('<div class="section-head">🎛️ Risk Calculator Inputs</div>',
                    unsafe_allow_html=True)
        pest_crop    = st.selectbox("Select Crop", list(CROP_INFO.keys()), key="pest_crop")
        pest_hum     = st.slider("Humidity (%)", 10.0, 100.0, 65.0, 1.0, key="pest_hum")
        pest_temp    = st.slider("Temperature (°C)", 0.0, 50.0, 25.0, 0.5, key="pest_temp")
        pest_rain    = st.slider("Rainfall (mm)", 20.0, 300.0, 100.0, 1.0, key="pest_rain")
        pest_om      = st.selectbox("Organic Matter", list(OM_LEVELS.keys()), key="pest_om")
        pest_om_val  = OM_VALUE_MAP.get(pest_om, 1.5)

        risk_idx = pest_risk_score(pest_crop, pest_hum, pest_temp, pest_rain, pest_om_val)
        risk_label = ("🟢 LOW" if risk_idx < 35 else "🟡 MODERATE" if risk_idx < 60
                      else "🔴 HIGH" if risk_idx < 80 else "🚨 VERY HIGH")
        risk_col   = ("#2e7d32" if risk_idx < 35 else "#f9a825" if risk_idx < 60
                      else "#e53935" if risk_idx < 80 else "#7b0000")

        fig_risk = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk_idx,
            delta={"reference":50,"increasing":{"color":"#e53935"},
                   "decreasing":{"color":"#2e7d32"}},
            title={"text":f"Pest/Disease Risk — {risk_label}","font":{"size":13}},
            gauge={"axis":{"range":[0,100]},
                   "bar":{"color":risk_col},
                   "steps":[{"range":[0,35],"color":"#e8f5e9"},
                             {"range":[35,60],"color":"#fff8e1"},
                             {"range":[60,80],"color":"#ffebee"},
                             {"range":[80,100],"color":"#b71c1c"}]},
            number={"suffix":"/100","font":{"size":26}},
        ))
        fig_risk.update_layout(height=260, margin=dict(t=40,b=0,l=20,r=20),
                               paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_risk, use_container_width=True)

    with pc2:
        st.markdown('<div class="section-head">🦠 Known Threats</div>',
                    unsafe_allow_html=True)
        info_p = CROP_INFO.get(pest_crop, {})
        diseases = info_p.get("diseases", [])
        base_risk = info_p.get("pest_risk","Medium")

        col_map = {"Very High":"#b71c1c","High":"#e53935","Medium":"#f9a825",
                   "Low":"#388e3c","Very Low":"#1b5e20"}
        pill_col = col_map.get(base_risk,"#888")
        st.markdown(f"""
        <div style="margin-bottom:.8rem">
          <b>Base Pest Risk Level:</b>
          <span class="risk-pill" style="background:{pill_col};color:white">{base_risk}</span>
        </div>
        """, unsafe_allow_html=True)

        for d in diseases:
            sev = ("🔴 High" if "Borer" in d or "Wilt" in d or "Armyworm" in d
                   else "🟡 Medium" if "Mildew" in d or "Blight" in d
                   else "🟢 Low")
            st.markdown(f"""
            <div class="soil-card" style="display:flex;justify-content:space-between;
                         align-items:center;padding:.6rem 1rem">
              <span>🦠 <b>{d}</b></span>
              <span style="font-size:.82rem">{sev}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-head">🛡️ Prevention Tips</div>',
                    unsafe_allow_html=True)
        tips = [
            "🌀 Rotate crops every season to break pest cycles.",
            "🌿 Use neem-based sprays as organic pest deterrents.",
            "💧 Avoid over-irrigation — wet foliage promotes fungal growth.",
            "🌡️ Monitor regularly during high humidity periods (>75%).",
            "🐝 Encourage beneficial insects by planting border crops.",
            "🧪 Apply fungicides preventively before monsoon onset.",
        ]
        for tip in tips:
            st.markdown(f"<div class='alert-info'>{tip}</div>", unsafe_allow_html=True)

    # ── Risk comparison across all crops ─────────────────────
    st.markdown('<div class="section-head">📊 Pest Risk Comparison — All Crops</div>',
                unsafe_allow_html=True)
    all_risks = {c: pest_risk_score(c, pest_hum, pest_temp, pest_rain, pest_om_val)
                 for c in CROP_INFO}
    risk_df   = (pd.DataFrame(list(all_risks.items()), columns=["Crop","Risk"])
                   .sort_values("Risk"))
    risk_df["Label"] = risk_df["Crop"].apply(lambda c: f"{CROP_EMOJI.get(c,'🌿')} {c}")
    risk_df["Color"] = risk_df["Risk"].apply(
        lambda r: "#2e7d32" if r<35 else "#f9a825" if r<60 else "#e53935")

    fig_all_risk = go.Figure(go.Bar(
        x=risk_df["Risk"], y=risk_df["Label"],
        orientation="h",
        marker_color=risk_df["Color"].tolist(),
        text=risk_df["Risk"].apply(lambda r: f"{r:.0f}"),
        textposition="outside",
    ))
    fig_all_risk.update_layout(height=620,
                               margin=dict(t=10,b=10,l=5,r=40),
                               xaxis=dict(title="Risk Index (0–100)", range=[0,115]),
                               yaxis={"autorange":"reversed"},
                               paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_all_risk, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — WATER REQUIREMENT vs AVAILABILITY
# ════════════════════════════════════════════════════════════════════════════
with tab_water:
    st.markdown(f"### {t('water_tab',lang)} — Requirement vs Availability")

    wc1, wc2 = st.columns([1,1], gap="large")
    with wc1:
        st.markdown('<div class="section-head">⚙️ Water Planner Inputs</div>',
                    unsafe_allow_html=True)
        w_crop    = st.selectbox("Crop", list(CROP_INFO.keys()), key="w_crop")
        w_source  = st.selectbox("Primary Water Source", WATER_SOURCES, key="w_src")
        w_rain    = st.slider("Expected Seasonal Rainfall (mm)", 0.0, 500.0, 120.0, 5.0)
        w_area    = st.number_input("Field Area (ha)", 0.1, 1000.0, 2.0, 0.1)
        w_seasons = st.number_input("Growing Seasons per Year", 1, 3, 1)

        req_mm, avail_mm, gap_mm, w_stat = water_status(w_crop, w_source, w_rain, w_area)

        total_req   = req_mm * w_area * w_seasons
        total_avail = avail_mm * w_area * w_seasons
        total_gap   = max(0, total_req - total_avail)

        col_stat = ("#2e7d32" if "✅" in w_stat else
                    "#f9a825" if "🟡" in w_stat else "#e53935")

        st.markdown(f"""
        <div class="soil-card" style="border-left:5px solid {col_stat};margin-top:.5rem">
          <div style="font-size:1.3rem;font-weight:800;color:{col_stat}">{w_stat}</div>
          <table style="width:100%;font-size:.88rem;margin-top:.5rem;border-collapse:collapse">
            <tr><td style="color:#666;padding:.2rem 0">Crop Water Need</td>
                <td><b>{req_mm} mm/season</b></td></tr>
            <tr><td style="color:#666;padding:.2rem 0">Estimated Available</td>
                <td><b>{avail_mm:.0f} mm/season</b></td></tr>
            <tr><td style="color:#666;padding:.2rem 0">Gap</td>
                <td><b style="color:{col_stat}">{max(0,gap_mm):.0f} mm</b></td></tr>
            <tr><td style="color:#666;padding:.2rem 0">Total for {w_area:.1f} ha</td>
                <td><b>{total_req/1000:.1f} ML required</b></td></tr>
            <tr><td style="color:#666;padding:.2rem 0">Annual gap ({w_seasons} season)</td>
                <td><b>{total_gap/1000:.1f} ML deficit</b></td></tr>
          </table>
        </div>
        """, unsafe_allow_html=True)

        # Saving tips
        st.markdown('<div class="section-head">💡 Water Saving Tips</div>',
                    unsafe_allow_html=True)
        saving_tips = {
            "Drip Irrigation":      ["Saves 25–40% vs flood irrigation.",
                                     "Delivers water directly to root zone.",
                                     "Reduces weed growth and disease pressure."],
            "Sprinkler Irrigation": ["Saves 15–25% vs flood irrigation.",
                                     "Good for irregular terrain fields.",
                                     "Best applied early morning to reduce evaporation."],
            "Rainfed only":         ["Adopt mulching to reduce evaporation by 20%.",
                                     "Consider rainwater harvesting pits.",
                                     "Choose drought-tolerant varieties."],
        }
        tips_w = saving_tips.get(w_source, ["Use mulching to retain soil moisture.",
                                            "Schedule irrigation early morning or evening.",
                                            "Soil moisture sensors can cut water use by 15%."])
        for tip in tips_w:
            st.markdown(f"<div class='alert-info'>💧 {tip}</div>", unsafe_allow_html=True)

    with wc2:
        # Req vs Avail gauge
        fig_w = go.Figure(go.Indicator(
            mode="gauge+number",
            value=min(100, avail_mm/req_mm*100) if req_mm>0 else 100,
            title={"text":"Water Supply Adequacy","font":{"size":13}},
            gauge={"axis":{"range":[0,130]},
                   "bar":{"color":col_stat},
                   "steps":[{"range":[0,50],"color":"#ffebee"},
                             {"range":[50,80],"color":"#fff8e1"},
                             {"range":[80,100],"color":"#e8f5e9"},
                             {"range":[100,130],"color":"#c8e6c9"}],
                   "threshold":{"line":{"color":"red","width":2},"thickness":.75,"value":80}},
            number={"suffix":"%"},
        ))
        fig_w.update_layout(height=250, margin=dict(t=40,b=0,l=20,r=20),
                            paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_w, use_container_width=True)

        # Source comparison bar
        st.markdown('<div class="section-head">🔄 Water Source Comparison</div>',
                    unsafe_allow_html=True)
        src_rows = []
        for src in WATER_SOURCES:
            r, a, g, s = water_status(w_crop, src, w_rain, 1)
            src_rows.append({"Source":src, "Available mm":round(a), "Required mm":r, "Status":s})
        src_df = pd.DataFrame(src_rows)
        fig_src = go.Figure()
        fig_src.add_trace(go.Bar(name="Available", x=src_df["Source"],
                                 y=src_df["Available mm"], marker_color="#66bb6a"))
        fig_src.add_hline(y=req_mm, line_dash="dash", line_color="#e53935",
                          annotation_text=f"Required: {req_mm} mm")
        fig_src.update_layout(height=300, margin=dict(t=30,b=10,l=5,r=5),
                              yaxis_title="mm / season",
                              paper_bgcolor="rgba(0,0,0,0)",
                              legend=dict(orientation="h",y=-0.2))
        st.plotly_chart(fig_src, use_container_width=True)

    # ── Water need ranking all crops ──────────────────────────
    st.markdown('<div class="section-head">💧 Water Requirement — All Crops</div>',
                unsafe_allow_html=True)
    wn_df = pd.DataFrame([
        {"Crop": f"{CROP_EMOJI.get(c,'🌿')} {c}",
         "Water mm": CROP_INFO[c].get("water_mm",0),
         "Category": CROP_INFO[c].get("category","")}
        for c in CROP_INFO
    ]).sort_values("Water mm")

    fig_wn = go.Figure(go.Bar(
        x=wn_df["Water mm"], y=wn_df["Crop"],
        orientation="h",
        marker=dict(color=wn_df["Water mm"], colorscale="Blues"),
        text=wn_df["Water mm"].apply(lambda v: f"{v} mm"),
        textposition="outside",
    ))
    fig_wn.add_vline(x=avail_mm, line_dash="dot", line_color="#e53935",
                     annotation_text=f"Your supply: {avail_mm:.0f} mm",
                     annotation_position="top right")
    fig_wn.update_layout(height=620, margin=dict(t=10,b=10,l=5,r=90),
                         xaxis_title="Seasonal Water Requirement (mm)",
                         yaxis={"autorange":"reversed"},
                         paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_wn, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
with tab_analysis:
    st.markdown(f"### {t('analysis_tab',lang)}")
    rows_a = []
    for crop, info in CROP_INFO.items():
        ideal = CROP_IDEAL.get(crop, {})
        rows_a.append({
            "Crop":     f"{CROP_EMOJI.get(crop,'🌿')} {crop}",
            "Category": info["category"],
            "Season":   info["season"],
            "Water mm": info.get("water_mm",0),
            "Profit":   info["profit"],
            "₹/kg":     info["market_price"],
            "Min OM %": info.get("min_om",1.0),
            "Pest Risk":info.get("pest_risk","Medium"),
        })
    df_a = pd.DataFrame(rows_a)

    ac1, ac2 = st.columns(2)
    with ac1:
        cat_c = df_a["Category"].value_counts()
        fig_d = go.Figure(go.Pie(labels=cat_c.index, values=cat_c.values,
                                 hole=.48, marker_colors=px.colors.qualitative.Set2))
        fig_d.update_layout(title="By Category", height=300,
                            margin=dict(t=40,b=0,l=0,r=0),
                            paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_d, use_container_width=True)

    with ac2:
        risk_c = df_a["Pest Risk"].value_counts()
        fig_r  = go.Figure(go.Bar(x=risk_c.index, y=risk_c.values,
                                  marker_color=["#e53935","#f9a825","#66bb6a","#2e7d32","#1b5e20"],
                                  text=risk_c.values, textposition="outside"))
        fig_r.update_layout(title="By Pest Risk Level", height=300,
                            margin=dict(t=40,b=30,l=5,r=5),
                            paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_r, use_container_width=True)

    # Radar crop compare
    st.markdown('<div class="section-head">🕸️ Radar Comparison</div>',
                unsafe_allow_html=True)
    ca2, cb2 = st.columns(2)
    with ca2: crop_a2 = st.selectbox("Crop A", list(CROP_IDEAL.keys()), index=0,  key="cra2")
    with cb2: crop_b2 = st.selectbox("Crop B", list(CROP_IDEAL.keys()), index=5, key="crb2")

    r_keys = ["N","P","K","temperature","humidity","ph","rainfall"]
    r_max  = [140,145,205,50,100,9.5,300]
    def norm2(c):
        d = CROP_IDEAL.get(c,{})
        v = [d.get(k,0)/m*100 for k,m in zip(r_keys,r_max)]
        return v + [v[0]]

    fig_rd = go.Figure()
    for c, col in [(crop_a2,"#2e7d32"),(crop_b2,"#fb8c00")]:
        fig_rd.add_trace(go.Scatterpolar(
            r=norm2(c), theta=r_keys+[r_keys[0]],
            fill="toself", name=f"{CROP_EMOJI.get(c,'')} {c}",
            line_color=col, fillcolor=col, opacity=0.35,
        ))
    fig_rd.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,100])),
                         height=400, paper_bgcolor="rgba(0,0,0,0)",
                         margin=dict(t=20,b=20,l=20,r=20))
    st.plotly_chart(fig_rd, use_container_width=True)

    # Market price
    st.markdown('<div class="section-head">💰 Market Price (₹/kg)</div>',
                unsafe_allow_html=True)
    p_df = df_a.sort_values("₹/kg")
    fig_mp = go.Figure(go.Bar(x=p_df["₹/kg"], y=p_df["Crop"],
                               orientation="h",
                               marker=dict(color=p_df["₹/kg"], colorscale="Greens"),
                               text=p_df["₹/kg"].apply(lambda v: f"₹{v}"),
                               textposition="outside"))
    fig_mp.update_layout(height=620, margin=dict(t=10,b=10,l=5,r=60),
                         xaxis_title="₹ per kg",
                         yaxis={"autorange":"reversed"},
                         paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_mp, use_container_width=True)

    st.markdown('<div class="section-head">📋 Full Reference Table</div>',
                unsafe_allow_html=True)
    st.dataframe(df_a, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 6 — CROP EXPLORER
# ════════════════════════════════════════════════════════════════════════════
with tab_explorer:
    st.markdown(f"### {t('explorer_tab',lang)}")
    sel_e    = st.selectbox("Choose a crop",
                 [f"{CROP_EMOJI.get(c,'🌿')} {c}" for c in CROP_IDEAL], key="exp_crop")
    crop_e   = sel_e.split(" ",1)[1]
    info_e   = CROP_INFO.get(crop_e, {})
    ideal_e  = CROP_IDEAL.get(crop_e, {})
    emoji_e  = CROP_EMOJI.get(crop_e, "🌿")

    st.markdown(f"""
    <div class="result-card" style="text-align:left;padding:1.3rem 1.8rem;">
      <div style="display:flex;align-items:center;gap:1rem;margin-bottom:.6rem">
        <span style="font-size:2.5rem">{emoji_e}</span>
        <div>
          <div class="crop-name" style="font-size:1.6rem">{crop_e}</div>
          <span class="meta-pill">{info_e.get('category','')}</span>
          <span class="meta-pill">{info_e.get('season','')} season</span>
          <span class="meta-pill">Pest: {info_e.get('pest_risk','—')}</span>
        </div>
      </div>
      <p style="color:#444;font-size:.91rem;margin:.3rem 0">💡 {info_e.get('tip','')}</p>
      <span class="meta-pill">💧 {info_e.get('water_mm',0)} mm/season</span>
      <span class="meta-pill">⏱ {info_e.get('duration','')}</span>
      <span class="meta-pill">💰 {info_e.get('profit','')} profit</span>
      <span class="meta-pill">🏷 ₹{info_e.get('market_price','')}/kg</span>
      <span class="meta-pill">🌱 Min OM: {info_e.get('min_om','')}%</span>
      <span class="meta-pill">🪨 {', '.join(info_e.get('soil_types',[]))}</span>
    </div>
    """, unsafe_allow_html=True)

    ek1, ek2 = st.columns([1,1], gap="large")
    with ek1:
        st.markdown('<div class="section-head">🎯 Ideal Conditions</div>',
                    unsafe_allow_html=True)
        ck   = ["N","P","K","temperature","humidity","ph","rainfall"]
        cl   = ["N","P","K","Temp","RH","pH","Rain"]
        cv   = [ideal_e.get(k,0) for k in ck]
        cmx  = [140,145,205,50,100,9.5,300]
        cpct = [v/m*100 for v,m in zip(cv,cmx)]
        fig_ec = go.Figure(go.Bar(x=cl, y=cpct, marker_color="#43a047",
                                  text=[str(v) for v in cv],
                                  textposition="outside"))
        fig_ec.update_layout(yaxis=dict(title="% of scale",range=[0,120]),
                             height=280, margin=dict(t=5,b=5,l=5,r=5),
                             paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_ec, use_container_width=True)

        st.markdown('<div class="section-head">🦠 Disease & Pest Threats</div>',
                    unsafe_allow_html=True)
        for d in info_e.get("diseases",[]):
            st.markdown(f"<div class='alert-warning'>🦠 {d}</div>",
                        unsafe_allow_html=True)

    with ek2:
        st.markdown('<div class="section-head">🔬 Field Checker</div>',
                    unsafe_allow_html=True)
        with st.expander("Enter your field values"):
            fc1,fc2,fc3,fc4 = st.columns(4)
            fn = fc1.number_input("N",  0,  140, 50, key="fc_n")
            fp = fc2.number_input("P",  5,  145, 50, key="fc_p")
            fk = fc3.number_input("K",  5,  205, 50, key="fc_k")
            ft = fc4.number_input("°C", 0., 50., 25.,key="fc_t")
            fg1,fg2,fg3 = st.columns(3)
            fh  = fg1.number_input("RH %",  10.,100.,65.,key="fc_h")
            fph = fg2.number_input("pH",    3.5,9.5, 6.5,key="fc_ph")
            fr  = fg3.number_input("Rain",  20.,300.,100.,key="fc_r")

        fu = dict(N=fn,P=fp,K=fk,temperature=ft,humidity=fh,ph=fph,rainfall=fr)
        fr2 = dict(N=(0,140),P=(5,145),K=(5,205),temperature=(0,50),
                   humidity=(10,100),ph=(3.5,9.5),rainfall=(20,300))
        comp = []
        for k,lbl in zip(ck,cl):
            iv = ideal_e.get(k,0); uv = fu.get(k,0)
            lo,hi = fr2[k]
            m = max(0, 100 - abs(uv-iv)/((hi-lo))*200)
            s = ("✅ Excellent" if m>=80 else "🟡 Good"
                 if m>=55 else "⚠️ Fair" if m>=30 else "❌ Poor")
            comp.append({"Parameter":lbl,"Your":uv,"Ideal":iv,
                          "Match %":round(m,1),"Status":s})
        comp_df2 = pd.DataFrame(comp)
        st.dataframe(comp_df2, use_container_width=True, hide_index=True)
        ov2 = comp_df2["Match %"].mean()
        st.metric("Overall Field–Crop Match", f"{ov2:.1f}%",
                  delta="Good fit ✅" if ov2>=60 else "Needs improvement ⚠️")

# ════════════════════════════════════════════════════════════════════════════
# TAB 7 — HISTORY
# ════════════════════════════════════════════════════════════════════════════
with tab_history:
    st.markdown(f"### {t('history_tab',lang)}")
    history = load_history()
    if not history:
        st.info("No predictions yet. Go to **🔍 Predict** to run your first analysis.")
    else:
        h_df = pd.DataFrame([{
            "Time":         h.get("time",""),
            "Crop":         f"{CROP_EMOJI.get(h.get('crop',''),'🌿')} {h.get('crop','')}",
            "Confidence":   f"{h.get('confidence',0)}%",
            "Soil Match":   f"{h.get('health_score',0)}%",
            "Pest Risk":    f"{h.get('pest_idx','—')}",
            "Water":        h.get("water_status","—"),
            "Soil Type":    h.get("soil_type","—"),
            "OM Level":     h.get("om","—"),
            "N": h["inputs"].get("N",""), "P": h["inputs"].get("P",""),
            "K": h["inputs"].get("K",""),
        } for h in reversed(history)])
        st.dataframe(h_df, use_container_width=True, hide_index=True)

        if len(history) >= 3:
            hc1, hc2 = st.columns(2)
            with hc1:
                freq = {}
                for h in history:
                    freq[h.get("crop","")] = freq.get(h.get("crop",""),0)+1
                f_df = (pd.DataFrame(list(freq.items()),columns=["Crop","Count"])
                          .sort_values("Count", ascending=True))
                f_df["Label"] = f_df["Crop"].apply(lambda c: f"{CROP_EMOJI.get(c,'🌿')} {c}")
                fig_f = go.Figure(go.Bar(x=f_df["Count"], y=f_df["Label"],
                                         orientation="h", marker_color="#43a047",
                                         text=f_df["Count"], textposition="outside"))
                fig_f.update_layout(title="Recommendation Frequency",
                                    height=max(200,len(freq)*42),
                                    margin=dict(t=35,b=10,l=5,r=40),
                                    paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_f, use_container_width=True)

            with hc2:
                tr_df = pd.DataFrame([{
                    "Run": i+1,
                    "Conf": h.get("confidence",0),
                    "Crop": h.get("crop",""),
                } for i,h in enumerate(history)])
                fig_tr = go.Figure(go.Scatter(
                    x=tr_df["Run"], y=tr_df["Conf"],
                    mode="lines+markers+text",
                    text=tr_df["Crop"],
                    textposition="top center",
                    line_color="#2e7d32",
                    marker=dict(size=8),
                ))
                fig_tr.update_layout(title="Confidence Trend",
                                     yaxis=dict(range=[0,105],title="Confidence %"),
                                     xaxis_title="Run #",
                                     height=max(200,len(history)*18+100),
                                     margin=dict(t=35,b=30,l=5,r=5),
                                     paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_tr, use_container_width=True)

        if st.button("🗑️ Clear History"):
            if os.path.exists(HISTORY_FILE):
                os.remove(HISTORY_FILE)
            st.rerun()
