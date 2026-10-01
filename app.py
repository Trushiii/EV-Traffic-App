import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="EV Traffic Grid Predictor", page_icon="⚡", layout="wide")

# ---------------------------------------------------------------------------
# Palette & type — pulled from the simulation's own visual language:
# asphalt roads, white lane paint, yellow charging-station patches, and
# red/green traffic signals doing double duty as the confidence indicator.
# ---------------------------------------------------------------------------
ASPHALT = "#12141A"
ASPHALT_LIGHT = "#1B1E27"
ASPHALT_LIGHTER = "#262A36"
LANE_WHITE = "#EDEBE3"
CHARGE_AMBER = "#FFB74D"
SIGNAL_GREEN = "#3ED598"
CAUTION_YELLOW = "#F2C230"
SIGNAL_RED = "#FF5C6C"
MUTED = "#8B92A3"

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

html, body, [data-testid="stAppViewContainer"], .stApp {{
  background: {ASPHALT} !important;
  color: {LANE_WHITE};
}}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stToolbar"] {{ right: 1rem; }}

body, p, span, div, label, li, [data-testid="stMarkdownContainer"] {{
  font-family: 'Inter', sans-serif;
}}
h1, h2, h3 {{
  font-family: 'Rajdhani', sans-serif !important;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  font-weight: 600 !important;
  border-left: 4px solid {CHARGE_AMBER};
  padding-left: 12px;
  color: {LANE_WHITE} !important;
}}

/* Group any row of columns (sliders, selectors) into a dashboard panel */
[data-testid="stHorizontalBlock"] {{
  background: {ASPHALT_LIGHT};
  border: 1px solid {ASPHALT_LIGHTER};
  border-radius: 14px;
  padding: 18px 20px 4px 20px;
  margin-bottom: 8px;
}}
[data-testid="stExpander"] {{
  background: {ASPHALT_LIGHT};
  border: 1px solid {ASPHALT_LIGHTER};
  border-radius: 14px;
}}

/* --- Hero --- */
.topbar {{ display:flex; align-items:center; gap:14px; margin-bottom: 6px; }}
.badge {{
  font-family:'IBM Plex Mono', monospace; font-size:12px; letter-spacing:0.05em;
  color:{SIGNAL_GREEN}; background: rgba(62,213,152,0.08);
  border:1px solid rgba(62,213,152,0.35); border-radius:20px; padding:4px 12px;
  display:inline-flex; align-items:center;
}}
.badge-alt {{
  font-family:'IBM Plex Mono', monospace; font-size:12px; color:{MUTED};
}}
.dot {{
  width:7px; height:7px; border-radius:50%; background:{SIGNAL_GREEN};
  display:inline-block; margin-right:7px;
}}
@media (prefers-reduced-motion: no-preference) {{
  .dot {{ animation: pulse-glow 2s ease-in-out infinite; }}
}}
@keyframes pulse-glow {{
  0% {{ box-shadow: 0 0 0 0 rgba(62,213,152,0.55); }}
  70% {{ box-shadow: 0 0 0 8px rgba(62,213,152,0); }}
  100% {{ box-shadow: 0 0 0 0 rgba(62,213,152,0); }}
}}
.hero-title {{
  font-family:'Rajdhani', sans-serif; font-weight:700; font-size:44px;
  color:{LANE_WHITE}; border-left:none !important; padding-left:0 !important;
  margin: 6px 0 0 0; letter-spacing:0.01em;
}}
.hero-tag {{
  font-family:'IBM Plex Mono', monospace; color:{CHARGE_AMBER}; font-size:15px;
  margin: 2px 0 10px 0;
}}
.hero-sub {{ color:{MUTED}; font-size:14.5px; max-width:800px; line-height:1.5; }}

/* --- Road divider: the signature motif --- */
.road {{
  height:5px; margin: 26px 0; border-radius:3px;
  background-image: repeating-linear-gradient(90deg, {LANE_WHITE} 0 16px, transparent 16px 32px);
  background-size: 200% 100%; opacity:0.28;
}}
@media (prefers-reduced-motion: no-preference) {{
  .road {{ animation: drive 5s linear infinite; }}
}}
@keyframes drive {{ from {{ background-position: 0 0; }} to {{ background-position: -160px 0; }} }}

/* --- Dashboard prediction cards --- */
.dash-row {{ display:flex; gap:14px; flex-wrap:wrap; margin: 4px 0 8px; }}
.dash-card {{
  flex:1; min-width:160px; background:{ASPHALT_LIGHT};
  border:1px solid {ASPHALT_LIGHTER}; border-left:4px solid {CAUTION_YELLOW};
  border-radius:12px; padding:16px 16px 14px; opacity:0;
}}
@media (prefers-reduced-motion: no-preference) {{
  .dash-card {{ animation: fade-up 0.5s ease forwards; }}
  .dash-card:nth-child(1) {{ animation-delay: 0.00s; }}
  .dash-card:nth-child(2) {{ animation-delay: 0.06s; }}
  .dash-card:nth-child(3) {{ animation-delay: 0.12s; }}
  .dash-card:nth-child(4) {{ animation-delay: 0.18s; }}
  .dash-card:nth-child(5) {{ animation-delay: 0.24s; }}
}}
@media (prefers-reduced-motion: reduce) {{ .dash-card {{ opacity:1; }} }}
@keyframes fade-up {{ from {{ opacity:0; transform:translateY(8px); }} to {{ opacity:1; transform:translateY(0); }} }}
.dash-card.conf-green {{ border-left-color:{SIGNAL_GREEN}; }}
.dash-card.conf-red {{ border-left-color:{SIGNAL_RED}; }}
@media (prefers-reduced-motion: no-preference) {{
  .dash-card.conf-green {{ animation: fade-up 0.5s ease forwards, glow-soft 3s ease-in-out 0.6s infinite; }}
}}
@keyframes glow-soft {{
  0%,100% {{ box-shadow:0 0 0 0 rgba(62,213,152,0); }}
  50% {{ box-shadow:0 0 14px 0 rgba(62,213,152,0.22); }}
}}
.dash-icon {{ font-size:19px; }}
.dash-label {{
  font-family:'Rajdhani', sans-serif; font-size:12px; letter-spacing:0.09em;
  color:{MUTED}; text-transform:uppercase; margin-top:4px;
}}
.dash-value {{ font-family:'IBM Plex Mono', monospace; font-size:25px; font-weight:600; color:{LANE_WHITE}; margin:3px 0 2px; }}
.dash-value .unit {{ font-size:11px; color:{MUTED}; margin-left:3px; font-family:'Inter',sans-serif; }}
.dash-conf {{ font-size:11px; color:{MUTED}; }}
.battery-bar {{ height:12px; border-radius:3px; background:{ASPHALT}; margin-top:9px; overflow:hidden;
  background-image: repeating-linear-gradient(90deg, {ASPHALT_LIGHTER} 0 2px, transparent 2px 20%); }}
.battery-fill {{ height:100%; border-radius:3px; background: linear-gradient(90deg, {CAUTION_YELLOW}, {SIGNAL_GREEN}); }}

.spec-footer {{ color:{MUTED}; font-family:'IBM Plex Mono', monospace; font-size:12px; line-height:1.6; }}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("ev_model.pkl")

bundle = load_model()
model = bundle["model"]
feature_cols = bundle["feature_cols"]
target_cols = bundle["target_cols"]
diagnostics = bundle["diagnostics"]
feature_ranges = bundle["feature_ranges"]

TARGET_LABELS = {
    "mean_wait_time": "Wait time",
    "mean_speed": "Speed",
    "count_charging": "Charging now",
    "num_cars_stopped": "Cars stopped",
    "mean_ev_battery": "EV battery",
}
TARGET_UNITS = {
    "mean_wait_time": "ticks",
    "mean_speed": "patches/tick",
    "count_charging": "cars",
    "num_cars_stopped": "cars",
    "mean_ev_battery": "%",
}
TARGET_ICONS = {
    "mean_wait_time": "⏱️",
    "mean_speed": "💨",
    "count_charging": "🔌",
    "num_cars_stopped": "🛑",
    "mean_ev_battery": "🔋",
}
CONFIDENCE_EMOJI = {
    "Well predicted": "🟢",
    "Weak relationship": "🟡",
    "Not meaningfully predictable from these inputs": "🔴",
}
CONFIDENCE_CLASS = {
    "Well predicted": "conf-green",
    "Weak relationship": "conf-yellow",
    "Not meaningfully predictable from these inputs": "conf-red",
}

def local_confidence(input_point, target_idx, k=None):
    """Per-prediction reliability: R² of the linear model computed only over the
    k nearest training combos in standardized feature space. Moves as the sliders
    move, so all three badges become reachable."""
    X     = bundle["train_X"]
    y     = bundle["train_y"]
    res   = bundle["loo_residuals"]

    if k is None:
        k = max(8, int(round(len(X) * 0.15)))   # ~15% of the training set

    scale = X.std(axis=0)
    scale[scale == 0] = 1.0
    d = np.sqrt((((X - input_point) / scale) ** 2).sum(axis=1))
    idx = np.argsort(d)[:k]

    y_nb = y[idx, target_idx]
    r_nb = res[idx, target_idx]
    ss_tot = np.sum((y_nb - y_nb.mean()) ** 2)
    r2 = 1.0 if ss_tot <= 1e-12 else 1.0 - np.sum(r_nb ** 2) / ss_tot

    if r2 >= 0.4:
        return "Well predicted"
    elif r2 >= 0.15:
        return "Weak relationship"
    return "Not meaningfully predictable from these inputs"

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown(
    """
<div class="topbar">
  <span class="badge"><span class="dot"></span>MODEL LIVE</span>
  <span class="badge-alt">linear regression · 64 simulated scenarios</span>
</div>
<div class="hero-title">⚡ EV Traffic Grid Predictor</div>
<div class="hero-tag">Skip the simulation. Ask the model.</div>
<p class="hero-sub">Trained on 192 NetLogo BehaviorSpace runs of the EV-enabled Traffic Grid
model. Set an EV share and a charging setup below, and get the expected traffic and
charging outcome instantly, no simulation run required.</p>
""",
    unsafe_allow_html=True,
)
st.markdown('<div class="road"></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
st.subheader("Set your scenario")

def _step_for(feature):
    vals = bundle.get("unique_values", {}).get(feature)
    if not vals or len(vals) < 2:
        return 1
    diffs = np.diff(sorted(vals))
    return int(diffs[diffs > 0].min()) if len(diffs) else 1

STEP_EV       = _step_for("ev_percentage")
STEP_STATIONS = _step_for("num_charging_stations")
STEP_BAYS     = _step_for("charging_bays")

col1, col2, col3 = st.columns(3)

with col1:
    ev_percentage = st.slider(
        "⚡ EV percentage — share of the 200 cars that are EVs",
        min_value=int(feature_ranges["ev_percentage"][0]),
        max_value=int(feature_ranges["ev_percentage"][1]),
        value=20,
        step=STEP_EV,
    )
with col2:
    num_charging_stations = st.slider(
        "🔌 Charging stations",
        min_value=int(feature_ranges["num_charging_stations"][0]),
        max_value=int(feature_ranges["num_charging_stations"][1]),
        value=8,
        step=STEP_STATIONS,
    )
with col3:
    charging_bays = st.slider(
        "🅿️ Bays per station",
        min_value=int(feature_ranges["charging_bays"][0]),
        max_value=int(feature_ranges["charging_bays"][1]),
        value=2,
        step=STEP_BAYS,
    )

X_input = pd.DataFrame(
    [[ev_percentage, num_charging_stations, charging_bays]], columns=feature_cols
)
prediction = model.predict(X_input)[0]
pred_dict = dict(zip(target_cols, prediction))

st.markdown('<div class="road"></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Predictions — custom dashboard cards
# ---------------------------------------------------------------------------
st.subheader("Predicted outcome after 500 ticks")

cards_html = '<div class="dash-row">'
for t in target_cols:
   conf = local_confidence(X_input.values[0], target_cols.index(t))
   css_class = CONFIDENCE_CLASS[conf]
   emoji = CONFIDENCE_EMOJI[conf]
   value = pred_dict[t]
   value_str = f"{value:.2f}" if abs(value) < 1000 else f"{value:,.0f}"

extra = ""
if t == "mean_ev_battery":
        pct = max(0.0, min(100.0, value))
        extra = f'<div class="battery-bar"><div class="battery-fill" style="width:{pct:.0f}%"></div></div>'

cards_html += f"""
    <div class="dash-card {css_class}">
      <div class="dash-icon">{TARGET_ICONS[t]}</div>
      <div class="dash-label">{TARGET_LABELS[t]}</div>
      <div class="dash-value">{value_str}<span class="unit">{TARGET_UNITS[t]}</span></div>
      <div class="dash-conf">{emoji} {conf}</div>
      {extra}
    </div>
    """
cards_html += "</div>"
st.markdown(cards_html, unsafe_allow_html=True)

with st.expander("What do the confidence badges mean?"):
    st.markdown(
        """
The simulation has randomness built in (random starting battery, random car
placement), so three repeated runs of the *same* settings still land in
different places. This model was validated with leave-one-out cross-validation
across the 64 unique parameter combinations, and the R² below reflects how much
of that variation is genuinely explained by your three inputs versus simulation noise.

- 🟢 **Well predicted** — R² ≥ 0.4, the input parameters meaningfully drive this outcome
- 🟡 **Weak relationship** — R² between 0.15 and 0.4, some signal but mostly noise
- 🔴 **Not meaningfully predictable** — R² below 0.15, this outcome is dominated by
  factors held constant across the experiment (200 cars, 5×5 grid, fixed traffic
  light timing) rather than by EV share, station count, or bays
        """
    )
    diag_df = pd.DataFrame(diagnostics).T[["r2", "mae", "confidence"]]
    diag_df.index = [TARGET_LABELS[i] for i in diag_df.index]
    st.dataframe(diag_df, use_container_width=True)

st.markdown('<div class="road"></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sensitivity chart: sweep one parameter, hold the others fixed
# ---------------------------------------------------------------------------
st.subheader("How does one parameter change the outcome?")

col_a, col_b = st.columns(2)
with col_a:
    sweep_var = st.selectbox(
        "Sweep this parameter",
        options=feature_cols,
        format_func=lambda x: {
            "ev_percentage": "EV percentage",
            "num_charging_stations": "Number of charging stations",
            "charging_bays": "Charging bays per station",
        }[x],
    )
with col_b:
    metric_to_plot = st.selectbox(
        "Plot this outcome", options=target_cols, format_func=lambda x: TARGET_LABELS[x]
    )

lo, hi = feature_ranges[sweep_var]
step = 1 if sweep_var == "charging_bays" else 4 if sweep_var == "num_charging_stations" else 10
sweep_values = list(range(int(lo), int(hi) + 1, step))

rows = []
for v in sweep_values:
    row = {"ev_percentage": ev_percentage, "num_charging_stations": num_charging_stations, "charging_bays": charging_bays}
    row[sweep_var] = v
    rows.append(row)
sweep_df = pd.DataFrame(rows)[feature_cols]
sweep_preds = model.predict(sweep_df)
metric_idx = target_cols.index(metric_to_plot)

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=sweep_values,
        y=sweep_preds[:, metric_idx],
        mode="lines+markers",
        line=dict(width=3, color=CHARGE_AMBER),
        marker=dict(size=9, color=SIGNAL_GREEN, line=dict(width=2, color=ASPHALT)),
        fill="tozeroy",
        fillcolor="rgba(255,183,77,0.08)",
    )
)
fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Mono, monospace", color=LANE_WHITE, size=12),
    xaxis=dict(title=sweep_var.replace("_", " ").title(), gridcolor=ASPHALT_LIGHTER, zerolinecolor=ASPHALT_LIGHTER),
    yaxis=dict(title=f"{TARGET_LABELS[metric_to_plot]} ({TARGET_UNITS[metric_to_plot]})", gridcolor=ASPHALT_LIGHTER, zerolinecolor=ASPHALT_LIGHTER),
    height=400,
    margin=dict(l=20, r=20, t=20, b=20),
)
st.plotly_chart(fig, use_container_width=True)

st.markdown('<div class="road"></div>', unsafe_allow_html=True)
st.markdown(
    """<p class="spec-footer">MODEL — Linear Regression · trained on outcomes averaged across
    3 repetitions per parameter combination · 64 unique combos · ev-experiment BehaviorSpace
    sweep · NetLogo 7.0.4 Traffic Grid</p>""",
    unsafe_allow_html=True,
)
