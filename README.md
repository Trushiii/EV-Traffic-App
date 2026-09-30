# EV Traffic Grid Predictor

A Streamlit app that predicts the outcome of the NetLogo EV Traffic Grid
simulation (wait time, speed, charging activity, congestion, EV battery
level) for any combination of 5 parameters, without re-running the
simulation: EV percentage, charging stations, bays per station, number of
cars, and traffic light cycle length.

It's trained on two BehaviorSpace exports pooled together: `trafficEV.csv`
(192 runs, sweeps EV share and charging setup) and
`congestion_45runs.csv` (45 runs, sweeps car count and light timing).

## What's in here

| File | Purpose |
|---|---|
| `app.py` | The Streamlit app |
| `ev_model.pkl` | Trained model, saved with feature names, target names, and validation diagnostics |
| `train_model.py` | Reproducible training script, retrains from one or more BehaviorSpace exports |
| `requirements.txt` | Python dependencies |
| `.streamlit/config.toml` | Dark dashboard theme (colors pulled from the sim's own asphalt/road/charging-station palette) |
| `ev_tidy_192runs.csv`, `ev_aggregated_64combos.csv`, `congestion_45runs.csv` | The underlying run data, useful for your thesis appendix |

**Important:** `.streamlit` is a folder whose name starts with a dot. Make sure
it lands in the same folder as `app.py`, not renamed or flattened, or the
custom colors won't apply and Streamlit will fall back to its default theme.

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

It opens at `http://localhost:8501`.

## Retrain on new data

```bash
python train_model.py trafficEV.csv congestion_45runs.csv
```

Pass as many BehaviorSpace CSVs as you have; the script pools all of them.
Any of the 5 parameters an experiment didn't sweep is filled with its
interface slider default (see `DEFAULTS` in `train_model.py`). This
overwrites `ev_model.pkl`, the app picks it up next time it starts.

## Deploy it (free, public URL)

The easiest option for a student project is **Streamlit Community Cloud**:

1. Push this folder to a public (or private, if you connect your GitHub account) GitHub repo. Include `app.py`, `ev_model.pkl`, `.streamlit/config.toml`, and `requirements.txt` at minimum.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click "New app", pick your repo, branch, and set the main file path to `app.py`.
4. Deploy. You'll get a public `*.streamlit.app` URL you can put in your thesis or demo.

Alternatives if you outgrow the free tier or need it private: Hugging Face
Spaces (also free, supports Streamlit natively) or Render's free web service
tier (needs a small `Procfile`, ask if you want that set up).

## A note on model quality

The underlying simulation is stochastic, cars spawn with random starting
battery and random positions, so repeated runs of the exact same settings
still land in different places. The model is trained on the average outcome
across the 3 repetitions of each of the 79 unique parameter combinations
(pooled across both experiments), validated with leave-one-out cross-validation.

That validation showed something worth knowing for your writeup:

- **Cars stopped** (R² 0.88), **speed** (0.66), **EV battery** (0.70), and
  **charging activity** (0.47) are well predicted once car count and light
  timing are in the model, those turned out to be the real congestion
  drivers, confirmed by a dedicated follow-up experiment.
- **Wait time** improved (R² from ~0 to 0.32) but is still only weakly
  explained. Some of its variation looks to come from randomness that isn't
  fully captured by any of the 5 inputs.

One limitation worth stating explicitly: the two experiments were never
**jointly** swept, e.g. no run tested 40% EVs *and* 300 cars *and* a short
light cycle at once (a "star" design, each experiment varies its own
parameters while holding the others at their default). The model assumes
these effects combine additively. It's a reasonable assumption and the
validation numbers above back it up, but interaction effects between the
EV setup and the traffic setup haven't been directly tested. The app
clips predictions to physically valid ranges (no negative speeds, battery
capped at 100%, etc.) since a linear model can otherwise extrapolate past
what's physically possible at extreme, untested combinations.

The app surfaces per-metric confidence with a badge (🟢🟡🔴) rather than
presenting every prediction as equally trustworthy.
