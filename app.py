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

/* --- Scenario coverage banner: reacts live to the sliders --- */
.coverage-badge {{
  display:flex; align-items:center; gap:10px; border-radius:10px;
  padding:11px 16px; margin: 2px 0 4px; font-size:13.5px;
}}
.coverage-ok {{
  background: rgba(62,213,152,0.08); border:1px solid rgba(62,213,152,0.35); color:{LANE_WHITE};
}}
.coverage-warn {{
  background: rgba(255,92,108,0.08); border:1px solid rgba(255,92,108,0.4); color:{LANE_WHITE};
}}
.coverage-badge .cov-icon {{ font-size:16px; }}
.coverage-badge .cov-meta {{ color:{MUTED}; font-family:'IBM Plex Mono', monospace; font-size:11.5px; margin-left:auto; }}
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
training_points_normalized = bundle.get("training_points_normalized")
coverage_threshold = bundle.get("coverage_threshold")
has_coverage_data = training_points_normalized is not None and coverage_threshold is not None

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

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown(
    """
<div class="topbar">
  <span class="badge"><span class="dot"></span>MODEL LIVE</span>
  <span class="badge-alt">linear regression · 79 simulated scenarios</span>
</div>
<div class="hero-title">⚡ EV Traffic Grid Predictor</div>
<div class="hero-tag">Skip the simulation. Ask the model.</div>
<p class="hero-sub">Trained on 237 NetLogo BehaviorSpace runs across two experiments on the
EV-enabled Traffic Grid model: EV share and charging setup, plus car count and traffic
light timing. Set a scenario below and get the expected traffic and charging outcome
instantly, no simulation run required.</p>
""",
    unsafe_allow_html=True,
)
st.markdown('<div class="road"></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
st.subheader("Set your scenario")

col1, col2, col3 = st.columns(3)

with col1:
    ev_percentage = st.slider(
        "⚡ EV percentage — share of cars that are EVs",
        min_value=int(feature_ranges["ev_percentage"][0]),
        max_value=int(feature_ranges["ev_percentage"][1]),
        value=20,
        step=10,
    )
with col2:
    num_charging_stations = st.slider(
        "🔌 Charging stations",
        min_value=int(feature_ranges["num_charging_stations"][0]),
        max_value=int(feature_ranges["num_charging_stations"][1]),
        value=8,
        step=4,
    )
with col3:
    charging_bays = st.slider(
        "🅿️ Bays per station",
        min_value=int(feature_ranges["charging_bays"][0]),
        max_value=int(feature_ranges["charging_bays"][1]),
        value=2,
        step=1,
    )

col4, col5 = st.columns(2)
with col4:
    num_cars = st.slider(
        "🚗 Number of cars on the grid",
        min_value=int(feature_ranges["num_cars"][0]),
        max_value=int(feature_ranges["num_cars"][1]),
        value=200,
        step=50,
    )
with col5:
    ticks_per_cycle = st.slider(
        "🚦 Ticks per traffic light cycle",
        min_value=int(feature_ranges["ticks_per_cycle"][0]),
        max_value=int(feature_ranges["ticks_per_cycle"][1]),
        value=20,
        step=10,
    )

X_input = pd.DataFrame(
    [[ev_percentage, num_charging_stations, charging_bays, num_cars, ticks_per_cycle]],
    columns=feature_cols,
)
prediction = model.predict(X_input)[0]
pred_dict = dict(zip(target_cols, prediction))

# The model is a linear fit pooled from two experiments that were never
# jointly swept (a "star" design), so extreme combinations can extrapolate
# past what's physically possible. Clip to valid ranges for display.
pred_dict["mean_wait_time"] = max(0.0, pred_dict["mean_wait_time"])
pred_dict["mean_speed"] = max(0.0, pred_dict["mean_speed"])
pred_dict["count_charging"] = max(0.0, min(pred_dict["count_charging"], num_charging_stations * charging_bays))
pred_dict["num_cars_stopped"] = max(0.0, min(pred_dict["num_cars_stopped"], num_cars))
pred_dict["mean_ev_battery"] = max(0.0, min(100.0, pred_dict["mean_ev_battery"]))

# ---------------------------------------------------------------------------
# Scenario coverage — THIS is the thing that actually responds to your
# sliders. The badges on each card below are fixed properties of the model
# (how well it predicts that metric overall); this banner instead checks how
# close your specific chosen combination is to a combination the simulation
# actually ran, versus the model extrapolating into untested territory.
# ---------------------------------------------------------------------------
if has_coverage_data:
    query_norm = np.array(
        [
            (X_input[c].iloc[0] - feature_ranges[c][0])
            / (feature_ranges[c][1] - feature_ranges[c][0] or 1)
            for c in feature_cols
        ]
    )
    nearest_dist = float(np.linalg.norm(training_points_normalized - query_norm, axis=1).min())
    is_covered = nearest_dist <= coverage_threshold

    if is_covered:
        st.markdown(
            f"""<div class="coverage-badge coverage-ok">
            <span class="cov-icon">✅</span>
            <span>This combination is close to a scenario the simulation actually ran, predictions below are reasonably trustworthy.</span>
            <span class="cov-meta">distance {nearest_dist:.2f} / threshold {coverage_threshold:.2f}</span>
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""<div class="coverage-badge coverage-warn">
            <span class="cov-icon">⚠️</span>
            <span>No simulation run closely matches this combination, the model is extrapolating here. Treat the numbers below as a rough estimate.</span>
            <span class="cov-meta">distance {nearest_dist:.2f} / threshold {coverage_threshold:.2f}</span>
            </div>""",
            unsafe_allow_html=True,
        )
else:
    st.markdown(
        """<div class="coverage-badge coverage-warn">
        <span class="cov-icon">⚠️</span>
        <span>Scenario coverage check unavailable — this model file predates that feature. Retrain with the current train_model.py to enable it.</span>
        </div>""",
        unsafe_allow_html=True,
    )

st.markdown('<div class="road"></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Predictions — custom dashboard cards
# ---------------------------------------------------------------------------
st.subheader("Predicted outcome after 500 ticks")

cards_html = '<div class="dash-row">'
for t in target_cols:
    diag = diagnostics[t]
    css_class = CONFIDENCE_CLASS[diag["confidence"]]
    emoji = CONFIDENCE_EMOJI[diag["confidence"]]
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
      <div class="dash-conf">{emoji} {diag['confidence']}</div>
      {extra}
    </div>
    """
cards_html += "</div>"
st.markdown(cards_html, unsafe_allow_html=True)

with st.expander("What do the confidence badges mean?"):
    st.markdown(
        """
There are two different trust signals on this page, and they answer
different questions:

- The **green/yellow/red badge banner above the cards** changes as you move
  the sliders. It checks whether *this specific combination* is close to a
  scenario the simulation actually ran.
- The **🟢🟡🔴 badge on each card below** does *not* change with the sliders.
  It's a fixed property of the model: how well it predicts that particular
  metric overall, found once during training and the same for every scenario.

The simulation has randomness built in (random starting battery, random car
placement), so three repeated runs of the *same* settings still land in
different places. This model was validated with leave-one-out cross-validation
across the 79 unique parameter combinations pooled from both experiments, and
the R² below reflects how much of that variation is genuinely explained by
your five inputs versus simulation noise.

- 🟢 **Well predicted** — R² ≥ 0.4, the input parameters meaningfully drive this outcome
- 🟡 **Weak relationship** — R² between 0.15 and 0.4, some signal but mostly noise
- 🔴 **Not meaningfully predictable** — R² below 0.15, this outcome is dominated by
  randomness rather than by any of the five inputs
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
            "num_cars": "Number of cars",
            "ticks_per_cycle": "Ticks per traffic light cycle",
        }[x],
    )
with col_b:
    metric_to_plot = st.selectbox(
        "Plot this outcome", options=target_cols, format_func=lambda x: TARGET_LABELS[x]
    )

lo, hi = feature_ranges[sweep_var]
step_map = {
    "charging_bays": 1,
    "num_charging_stations": 4,
    "ev_percentage": 10,
    "num_cars": 50,
    "ticks_per_cycle": 10,
}
step = step_map[sweep_var]
sweep_values = list(range(int(lo), int(hi) + 1, step))

rows = []
for v in sweep_values:
    row = {
        "ev_percentage": ev_percentage,
        "num_charging_stations": num_charging_stations,
        "charging_bays": charging_bays,
        "num_cars": num_cars,
        "ticks_per_cycle": ticks_per_cycle,
    }
    row[sweep_var] = v
    rows.append(row)
sweep_df = pd.DataFrame(rows)[feature_cols]
sweep_preds = model.predict(sweep_df)
metric_idx = target_cols.index(metric_to_plot)

raw_y = sweep_preds[:, metric_idx]
if metric_to_plot == "mean_ev_battery":
    plot_y = np.clip(raw_y, 0.0, 100.0)
elif metric_to_plot == "count_charging":
    caps = (sweep_df["num_charging_stations"] * sweep_df["charging_bays"]).values
    plot_y = np.clip(raw_y, 0.0, caps)
elif metric_to_plot == "num_cars_stopped":
    plot_y = np.clip(raw_y, 0.0, sweep_df["num_cars"].values)
else:  # mean_wait_time, mean_speed
    plot_y = np.clip(raw_y, 0.0, None)

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=sweep_values,
        y=plot_y,
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
    3 repetitions per parameter combination · 79 unique combos pooled from ev-experiment and
    congestion-experiment · NetLogo 7.0.4 Traffic Grid</p>""",
    unsafe_allow_html=True,
)
