# Cepheid MESA Grid Viewer

Interactive browser for the Cepheid evolutionary-track grid from
Smolec et al. (2026), *Toward a Comprehensive Grid of Cepheid Models with MESA. III.*
Zenodo: [10.5281/zenodo.17987357](https://doi.org/10.5281/zenodo.17987357)

Pick any two columns to plot against each other, colour by mass / Z / set,
overlay the instability strip, compare physics variants side by side in
multi-panel mode, export filtered subsets as CSV.

## Folder layout

    cepheid-viewer/
    ├── app.py              # Streamlit app
    ├── preprocess.py       # tgz → parquet (run once)
    ├── requirements.txt
    ├── models/             # raw Zenodo downloads (kept, not committed)
    └── data/               # generated Parquet cache

## Setup

If you already have a working astro Python env (pandas, numpy, plotly),
just add the missing ones:

    pip install streamlit pyarrow watchdog

Otherwise:

    pip install -r requirements.txt

`watchdog` silences the *"For better performance, install the Watchdog module"*
notice and makes file-change auto-reload instant. On macOS you may also need
the command-line tools once per machine: `xcode-select --install`.

## Build the Parquet cache

    python preprocess.py            # full resolution
    python preprocess.py --stride 5 # ~5× smaller, useful for later deploy

Safe to re-run — sets already converted are skipped unless `--force`.

## Run

    streamlit run app.py

Opens at <http://localhost:8501>. Edit `app.py`, save, the browser reloads.

## UI

**Sidebar**

- *Model set(s)* — pick up to 6 (default: `O24`)
- *Compare mass range* toggle — on = range slider, off = single-mass dropdown
- *Compare compositions* toggle — on = multi-select, off = single-composition dropdown
- *Display options* (expander) — points-per-track slider for render speed

**Single plot**

- Preset pills along the top (HRD, Kiel, CMD V–I, CMD V–K, L-vs-age, …)
- Two dropdowns for any column on X and any on Y
- Segmented control to colour by `set / mass / Z / Y`
- ⚙ popover: flip / log axes + instability-strip overlay

**Multi-panel** — 2, 3, or 4 synced panels, each with its own preset.

**Data** — filtered subset table + CSV download.

## Performance knobs

- *Points per track* (sidebar → Display options) — each track is downsampled
  to this many points before being handed to Plotly. 400 is the default and
  usually visually indistinguishable from the raw ~3000-point tracks. Lower
  this for snappier panning on huge selections; raise it for maximum detail.
- Evolution tracks are rendered with Plotly's WebGL backend (`Scattergl`),
  so even 100+ traces pan/zoom smoothly.
- When the selection exceeds ~80 tracks the app shows a soft warning — the
  plot still renders, but interaction gets laggy above that threshold.

## Columns

The pre-processed `history.dat_*` files in the Zenodo archives have 26
columns, tab-separated with a single header row:

`model_number, star_age, star_mass, log_Teff, log_L, log_R, log_g,
log_cntr_P, log_cntr_Rho, log_cntr_T, center_mu,
center_{h1,he4,c12,n14,o16}, surface_{h1,he4,c12,n14,o16},
abs_mag_{V,I,J,H,K}`.

The app adds these derived columns on load: `log_age, V_minus_I, V_minus_K,
J_minus_K, J_minus_H, H_minus_K`. Per-track identity columns `set, mass, Z, Y`
come from the filename.

## Deploying (later)

1. `git init`, commit, push to a public GitHub repo.
2. <https://share.streamlit.io> → sign in with GitHub → point at repo,
   entry-point `app.py`. Done.
3. Push = redeploy.

**Don't commit `models/`** — the raw tgz archives are ~800 MB. Commit `data/`
(the Parquet cache). Suggested `.gitignore`:

    .venv/
    models/
    __pycache__/
    *.pyc

Full-resolution Parquet may total ~400 MB. Streamlit Cloud's free tier has a
~1 GB repo ceiling, so either downsample at preprocess time (`--stride 3`) or
host the Parquet files on GitHub Releases / Zenodo and fetch them on first run.
