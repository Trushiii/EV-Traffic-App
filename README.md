# EV Traffic Grid Predictor

A Streamlit app that predicts the outcome of the NetLogo EV Traffic Grid
simulation (wait time, speed, charging activity, congestion, EV battery
level) for any combination of EV percentage, number of charging stations,
and charging bays per station, without re-running the simulation.

It's trained on your `trafficEV.csv` BehaviorSpace export (192 runs, full
factorial sweep, 3 repetitions each).

## What's in here

| File | Purpose |
|---|---|
| `app.py` | The Streamlit app |
| `ev_model.pkl` | Trained model, saved with feature names, target names, and validation diagnostics |
| `train_model.py` | Reproducible training script, run it again on a new BehaviorSpace export to retrain |
| `requirements.txt` | Python dependencies |
| `.streamlit/config.toml` | Dark dashboard theme (colors pulled from the sim's own asphalt/road/charging-station palette) |

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

If you re-run the NetLogo experiment and export a new CSV:

```bash
python train_model.py path/to/new_trafficEV.csv
```

This overwrites `ev_model.pkl`. The app will pick it up next time it starts.

## Deploy it (free, public URL)

The easiest option for a student project is **Streamlit Community Cloud**:

1. Push this folder to a public (or private, if you connect your GitHub account) GitHub repo. Include `app.py`, `ev_model.pkl`, and `requirements.txt` at minimum.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click "New app", pick your repo, branch, and set the main file path to `app.py`.
4. Deploy. You'll get a public `*.streamlit.app` URL you can put in your thesis or demo.

Alternatives if you outgrow the free tier or need it private: Hugging Face
Spaces (also free, supports Streamlit natively) or Render's free web service
tier (needs a small `Procfile`, ask if you want that set up).

## A note on model quality

The underlying simulation is stochastic, cars spawn with random starting
battery and random positions, so repeated runs of the exact same settings
still land in different places. To get an honest signal out of this, the
model is trained on the average outcome across the 3 repetitions of each
of the 64 unique parameter combinations, and validated with leave-one-out
cross-validation (appropriate given how few unique combinations there are).

That validation showed something worth knowing for your writeup:

- **EV battery level** and **charging activity** are well predicted by
  these three parameters (R² of 0.64 and 0.45), which makes sense, they're
  directly downstream of how many EVs there are and how much charging
  capacity is available.
- **Wait time** and **cars stopped** are barely explained by these
  parameters at all (R² near 0). That's because the experiment held car
  count (200), grid size (5×5), and traffic light timing fixed throughout,
  those are what actually drive congestion in this model. EV share and
  charging infrastructure have a real but secondary effect on system-wide
  congestion in this particular setup.

The app surfaces this per-metric with a confidence badge (🟢🟡🔴) rather than
presenting every prediction as equally trustworthy.
