"""
Cepheid MESA Grid Viewer — v2.1
===============================
Smolec et al. — Toward a Comprehensive Grid of Cepheid Models with MESA. III
Zenodo: 10.5281/zenodo.17987357

New in v2.1:
  - [Fe/H] surfaced in track hover tooltips
  - CMD (V−K) preset added alongside CMD (V−I)
  - Plot modebar download set to publication-quality 1600×1100 @ 2× scale,
    with self-describing filenames (preset + sets / panel index + axes)
  - Instability strip shows in the figure legend with a clear identifier
  - Header metadata strip — version, Zenodo DOI, live track / mass counts
  - Sidebar "About" expander with paper / dataset links + cache reset

v2 baseline:
  - Axis presets (HRD / Kiel / Nuclear / Luminosity / CMD / Custom)
  - Upload observations CSV → overlay as markers
  - Age animation (play/pause slider, Plotly frames)
  - Click-to-inspect — tap any track to see details
  - Citation box with BibTeX
  - Proper progress indicators on data load

Run:
    streamlit run app.py
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go
import streamlit as st

# streamlit-elements brings Material-UI v5 into Streamlit. We use it for
# the parts that need real component-grade styling (elevation, transitions,
# proper card semantics) — KPI strip and click-inspect detail panel.
# Everything else stays as native Streamlit / Plotly.
try:
    from streamlit_elements import elements, mui
    _HAS_ELEMENTS = True
except ImportError:
    _HAS_ELEMENTS = False

__version__ = "2.1"

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"


# ---------------------------------------------------------------------------
# Set metadata (Table 3)
# ---------------------------------------------------------------------------
SET_META: dict[str, dict] = {
    "O00":    {"fcor": 0.00, "fenv": 0.00, "eta": 0.0, "flavor": "std", "note": "no overshooting"},
    "O02":    {"fcor": 0.00, "fenv": 0.02, "eta": 0.0, "flavor": "std", "note": ""},
    "O04":    {"fcor": 0.00, "fenv": 0.04, "eta": 0.0, "flavor": "std", "note": ""},
    "O06":    {"fcor": 0.00, "fenv": 0.06, "eta": 0.0, "flavor": "std", "note": ""},
    "O10":    {"fcor": 0.01, "fenv": 0.00, "eta": 0.0, "flavor": "std", "note": ""},
    "O12":    {"fcor": 0.01, "fenv": 0.02, "eta": 0.0, "flavor": "std", "note": ""},
    "O14":    {"fcor": 0.01, "fenv": 0.04, "eta": 0.0, "flavor": "std", "note": ""},
    "O16":    {"fcor": 0.01, "fenv": 0.06, "eta": 0.0, "flavor": "std", "note": ""},
    "O20":    {"fcor": 0.02, "fenv": 0.00, "eta": 0.0, "flavor": "std", "note": ""},
    "O22":    {"fcor": 0.02, "fenv": 0.02, "eta": 0.0, "flavor": "std", "note": ""},
    "O24":    {"fcor": 0.02, "fenv": 0.04, "eta": 0.0, "flavor": "std", "note": "reference set"},
    "O26":    {"fcor": 0.02, "fenv": 0.06, "eta": 0.0, "flavor": "std", "note": ""},
    "O30":    {"fcor": 0.03, "fenv": 0.00, "eta": 0.0, "flavor": "std", "note": ""},
    "O32":    {"fcor": 0.03, "fenv": 0.02, "eta": 0.0, "flavor": "std", "note": ""},
    "O34":    {"fcor": 0.03, "fenv": 0.04, "eta": 0.0, "flavor": "std", "note": ""},
    "O36":    {"fcor": 0.03, "fenv": 0.06, "eta": 0.0, "flavor": "std", "note": ""},
    "O24_ML2":{"fcor": 0.02, "fenv": 0.04, "eta": 0.2, "flavor": "Reimers ML", "note": ""},
    "O24_ML4":{"fcor": 0.02, "fenv": 0.04, "eta": 0.4, "flavor": "Reimers ML", "note": ""},
    "O24_ML6":{"fcor": 0.02, "fenv": 0.04, "eta": 0.6, "flavor": "Reimers ML", "note": ""},
    "O00_AB": {"fcor": 0.00, "fenv": 0.00, "eta": 0.0, "flavor": "NACRE ¹⁴N(p,γ)", "note": ""},
    "O24_AB": {"fcor": 0.02, "fenv": 0.04, "eta": 0.0, "flavor": "NACRE ¹⁴N(p,γ)", "note": ""},
    "O00_AC": {"fcor": 0.00, "fenv": 0.00, "eta": 0.0, "flavor": "GS98 solar mix", "note": ""},
    "O24_AC": {"fcor": 0.02, "fenv": 0.04, "eta": 0.0, "flavor": "GS98 solar mix", "note": ""},
    "O00_AE": {"fcor": 0.00, "fenv": 0.00, "eta": 0.0, "flavor": "ΔY/ΔZ = 2.0", "note": ""},
    "O24_AE": {"fcor": 0.02, "fenv": 0.04, "eta": 0.0, "flavor": "ΔY/ΔZ = 2.0", "note": ""},
}

Z_TO_FEH: dict[float, float] = {
    0.0200:  0.197, 0.0160:  0.094, 0.0140:  0.033, 0.0120: -0.037,
    0.0100: -0.119, 0.0080: -0.219, 0.0060: -0.347, 0.0040: -0.526,
    0.0030: -0.652, 0.0020: -0.830, 0.0014: -0.985,
}

def feh_for(z: float) -> float | None:
    for zk, feh in Z_TO_FEH.items():
        if abs(z - zk) < 1e-5:
            return feh
    return None

def set_description(s: str) -> str:
    m = SET_META.get(s)
    if not m:
        return s
    bits = [f"f<sub>core</sub>={m['fcor']:.2f}", f"f<sub>env</sub>={m['fenv']:.2f}"]
    if m["eta"] > 0:
        bits.append(f"η={m['eta']:.1f}")
    if m["flavor"] != "std":
        bits.append(m["flavor"])
    if m["note"]:
        bits.append(m["note"])
    return ", ".join(bits)


def set_long_name(s: str) -> str:
    """
    Human-friendly label for a model set, with the bare ID in parentheses.
    Used as format_func in the sidebar multiselect and in plot titles, so
    a viewer who doesn't know the Smolec et al. set codes can still read
    what each one means at a glance. Labels are kept concise so they
    don't truncate inside Streamlit's dropdown popover.

    Examples:
        'O00'      → 'No overshoot (O00)'
        'O24'      → 'Reference (O24)'
        'O24_ML4'  → 'Reference + Reimers η=0.4 (O24_ML4)'
        'O00_AB'   → 'No overshoot + NACRE ¹⁴N rates (O00_AB)'

    Plain text only — Streamlit selectboxes don't render HTML.
    """
    m = SET_META.get(s)
    if not m:
        return s

    fcor = m.get("fcor", 0.0)
    fenv = m.get("fenv", 0.0)
    flavor = m.get("flavor", "std")
    eta = m.get("eta", 0.0)

    # Base label encodes (f_core, f_env). Names kept short for dropdown fit.
    if abs(fcor) < 1e-9 and abs(fenv) < 1e-9:
        base = "No overshoot"
    elif abs(fcor - 0.02) < 1e-9 and abs(fenv - 0.04) < 1e-9:
        base = "Reference"
    else:
        bits = []
        if fcor > 0:
            bits.append(f"f_core={fcor:.2f}")
        if fenv > 0:
            bits.append(f"f_env={fenv:.2f}")
        base = ", ".join(bits) if bits else "Custom"

    # Compact flavor names for variants
    flavor_short = {
        "NACRE ¹⁴N(p,γ)": "NACRE ¹⁴N rates",
        "GS98 solar mix": "GS98 mix",
        "ΔY/ΔZ = 2.0":    "ΔY/ΔZ = 2",
    }

    # Variant suffixes
    if flavor == "Reimers ML":
        return f"{base} + Reimers η={eta:.1f} ({s})"
    if flavor and flavor != "std":
        f_short = flavor_short.get(flavor, flavor)
        return f"{base} + {f_short} ({s})"
    return f"{base} ({s})"


# Long-form names for the IS-edge identifier dropdown (Tab16 codes).
# Same pattern: descriptive label + raw code in parentheses.
IS_ID_LONG_NAME = {
    "b":   "Blue edge (b)",
    "r":   "Red edge (r)",
    "m":   "Midline (m)",
    "all": "All edges combined (all)",
}

def is_id_long_name(k: str) -> str:
    return IS_ID_LONG_NAME.get(k, k)


# ---------------------------------------------------------------------------
# Pretty axis labels
# ---------------------------------------------------------------------------
AXIS_LABELS: dict[str, str] = {
    "log_Teff":     "log T<sub>eff</sub>  [K]",
    "log_L":        "log L/L<sub>⊙</sub>",
    "log_g":        "log g  [cgs]",
    "log_R":        "log R/R<sub>⊙</sub>",
    "log_cntr_T":   "log T<sub>c</sub>  [K]",
    "log_cntr_Rho": "log ρ<sub>c</sub>  [g cm<sup>−3</sup>]",
    "log_cntr_P":   "log P<sub>c</sub>",
    "center_mu":    "μ<sub>c</sub>",
    "center_h1":    "X<sub>c</sub>(¹H)",
    "center_he4":   "Y<sub>c</sub>(⁴He)",
    "center_c12":   "X<sub>c</sub>(¹²C)",
    "center_n14":   "X<sub>c</sub>(¹⁴N)",
    "center_o16":   "X<sub>c</sub>(¹⁶O)",
    "surface_h1":   "X<sub>s</sub>(¹H)",
    "surface_he4":  "Y<sub>s</sub>(⁴He)",
    "star_age":     "Age  [yr]",
    "log_age":      "log(Age / yr)",
    "star_mass":    "M  [M<sub>⊙</sub>]",
    "mass":         "Initial M  [M<sub>⊙</sub>]",
    "abs_mag_V":    "M<sub>V</sub>",
    "abs_mag_I":    "M<sub>I</sub>",
    "abs_mag_J":    "M<sub>J</sub>",
    "abs_mag_H":    "M<sub>H</sub>",
    "abs_mag_K":    "M<sub>K</sub>",
    "V_minus_I":    "V − I",
    "V_minus_K":    "V − K",
    "J_minus_K":    "J − K",
    "J_minus_H":    "J − H",
    "H_minus_K":    "H − K",
    "W_VI":         "W<sub>VI</sub>  =  M<sub>V</sub> − 2.55 (V − I)",
    "W_VK":         "W<sub>VK</sub>  =  M<sub>V</sub> − 0.13 (V − K)",
    "model_number": "model #",
    "Z":            "Z",
    "Y":            "Y",
    # Tab16 (instability strip) columns used in the P–L tab
    "P_F":          "P<sub>F</sub>  [days]",
    "P_1O":         "P<sub>1O</sub>  [days]",
    "log_P_F":      "log P<sub>F</sub>  [days]",
    "log_P_1O":     "log P<sub>1O</sub>  [days]",
    "M_V":          "M<sub>V</sub>",
    "M_I":          "M<sub>I</sub>",
    "M_J":          "M<sub>J</sub>",
    "M_H":          "M<sub>H</sub>",
    "M_K":          "M<sub>K</sub>",
    # Tab6 (crossing-time) columns used in the Crossings tab
    "log_tcross":   "log t<sub>cross</sub>  [yr]",
    "Pdot_over_P":  "Ṗ / P  [yr<sup>−1</sup>]",
    "crossing":     "crossing #",
}

def ax_label(col: str) -> str:
    return AXIS_LABELS.get(col, col)


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
THEME = {
    # ----------------------------------------------------------------------
    # "Nocturne" — v2.1 palette
    # Deep ink-blue-black canvas with antique brass primary and pale
    # celadon accent. Mood: late-night observatory, leather-and-brass,
    # vintage scientific instrument. The brass + celadon pairing is a
    # classic instrument-and-oxidized-copper combination — warm and
    # sophisticated without feeling literal "warm theme". The IS rose
    # stays close to its prior colour because it is a physical (not
    # stylistic) marker on the HRD.
    # ----------------------------------------------------------------------
    "bg":            "#13141f",  # ink blue-black (page)
    "bg_raised":     "#1f2030",  # lifted ink (cards, popovers)
    "bg_sidebar":    "#0c0d16",  # deepest ink (sidebar)
    "bg_plot":       "#161724",  # plot canvas
    "border":        "#33344a",  # muted indigo-grey
    "border_strong": "#494a64",
    "text":          "#f1ede0",  # soft parchment (warm white)
    "text_muted":    "#b5ad97",  # aged-paper warm grey
    "text_subtle":   "#7a7464",  # dim warm grey
    "primary":       "#d6a45c",  # antique brass — main interactive
    "primary_hover": "#e0b577",  # lighter brass
    "accent":        "#8fc4b0",  # pale celadon — cool counterpoint
    "is_red":        "#d4716e",  # vintage rose (IS marker, retained)
}

FONT_SANS = ('Inter, -apple-system, BlinkMacSystemFont, '
             '"Helvetica Neue", "Segoe UI", Roboto, sans-serif')
FONT_SERIF = ('"Source Serif Pro", "Source Serif 4", Charter, '
              'Georgia, "Times New Roman", serif')
FONT_MONO = ('"JetBrains Mono", "SF Mono", ui-monospace, '
             'Menlo, Consolas, monospace')


def _theme_css() -> str:
    T = THEME
    return f"""
<style>
@import url('https://rsms.me/inter/inter.css');
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,wght@0,400;0,500;0,600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {{
    --bg: {T["bg"]};
    --bg-raised: {T["bg_raised"]};
    --bg-sidebar: {T["bg_sidebar"]};
    --bg-plot: {T["bg_plot"]};
    --border: {T["border"]};
    --border-strong: {T["border_strong"]};
    --text: {T["text"]};
    --text-muted: {T["text_muted"]};
    --text-subtle: {T["text_subtle"]};
    --primary: {T["primary"]};
    --primary-hover: {T["primary_hover"]};
    --accent: {T["accent"]};
}}

html, body, .stApp, [data-testid="stAppViewContainer"] {{
    background-color: var(--bg);
    /* Atmospheric depth on the deep-indigo canvas: a soft copper "horizon
       glow" near the top and a faint cyan nebula in the lower-right. Both
       sit ≈5–6% above the base — present, not loud. */
    background-image:
        radial-gradient(ellipse 1200px 640px at 50% -8%,
                        rgba(214,164,92,0.055) 0%,
                        rgba(214,164,92,0) 60%),
        radial-gradient(ellipse 950px 540px at 82% 110%,
                        rgba(143,196,176,0.045) 0%,
                        rgba(143,196,176,0) 60%);
    background-attachment: fixed;
    font-family: {FONT_SANS};
    color: var(--text);
    font-size: 15.5px;
    font-feature-settings: "ss01", "cv11", "tnum";
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}}

/* Horizon bar — a 2px filament at the very top of the page: copper at
   the edges, cyan at the meridian. Reads as a sky-line above the page
   and gives the chrome a clear, instrument-like upper edge. */
.stApp::before {{
    content: "";
    position: fixed;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg,
        rgba(214,164,92,0) 0%,
        rgba(214,164,92,0.65) 18%,
        rgba(143,196,176,0.9) 50%,
        rgba(214,164,92,0.65) 82%,
        rgba(214,164,92,0) 100%);
    z-index: 9999;
    pointer-events: none;
}}

.block-container {{
    padding-top: 1.4rem;
    padding-bottom: 3rem;
    max-width: 1340px;
}}

h1, h2, h3 {{
    color: var(--text) !important;
    font-family: {FONT_SERIF};
    font-weight: 500;
    letter-spacing: -0.012em;
    line-height: 1.25;
}}
h1 {{
    font-size: 1.7rem;
    font-weight: 500;
    margin-top: 0;
    margin-bottom: 0.35rem;
    padding-bottom: 0.65rem;
    /* Gradient underline — copper "ignition" at the start fading through
       cyan into transparent. Reads like sunlight sweeping across a
       horizon. Replaces the flat solid border. */
    border-bottom: none;
    background-image: linear-gradient(90deg,
        rgba(214,164,92,0.95) 0%,
        rgba(143,196,176,0.7) 12%,
        rgba(143,196,176,0.32) 38%,
        rgba(143,196,176,0) 72%);
    background-position: 0 100%;
    background-size: 100% 1.5px;
    background-repeat: no-repeat;
}}
h2 {{ font-size: 1.2rem; }}
h3 {{ font-size: 1.05rem; }}
h4, h5, h6 {{
    color: var(--text) !important;
    font-family: {FONT_SANS};
    font-weight: 500;
    letter-spacing: 0;
}}
h1 + div p {{
    color: var(--text-muted) !important;
    font-size: 0.95rem;
    line-height: 1.55;
    font-weight: 400;
}}

p, span, label, li {{ color: var(--text); font-size: 1.0rem; }}
[data-testid="stCaptionContainer"] p,
[data-testid="stCaptionContainer"] {{
    color: var(--text-muted) !important;
    font-size: 0.98rem;
    line-height: 1.6;
    font-weight: 400;
}}
hr {{
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.1rem 0;
    opacity: 0.7;
}}

[data-testid="stSidebar"] {{
    background-color: var(--bg-sidebar) !important;
    border-right: 1px solid var(--border);
}}
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] h2 {{
    font-family: {FONT_SANS} !important;
    font-size: 0.82rem !important;
    font-weight: 600;
    color: var(--text-muted) !important;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    margin-top: 0.4rem;
    margin-bottom: 0.75rem;
    padding: 0.1rem 0 0.4rem 0.7rem;
    border-bottom: 1px solid var(--border);
    /* Small accent bar to the left of each section header — pulls the
       eye through the sidebar without adding visual weight. */
    border-left: 2px solid var(--accent);
}}
[data-testid="stSidebar"] hr {{
    border: none;
    border-top: 1px solid var(--border);
    margin: 0.95rem 0;
    opacity: 0.55;
}}
[data-testid="stSidebar"] label {{
    color: var(--text-muted) !important;
    font-size: 0.93rem;
    font-weight: 400;
    letter-spacing: 0.005em;
}}

[data-baseweb="select"] > div,
.stTextInput input, .stNumberInput input {{
    background-color: var(--bg-raised) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 4px !important;
}}
[data-baseweb="select"] > div:hover {{ border-color: var(--border-strong) !important; }}
[data-baseweb="popover"] [data-baseweb="menu"],
[data-baseweb="popover"] {{
    background-color: var(--bg-raised) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: 4px !important;
    /* Keep the popover at its natural (trigger-anchored) width to
       avoid overflowing the viewport edge. Long labels wrap to a
       second line via `white-space: normal` below. */
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.45),
                0 0 0 1px rgba(143, 196, 176, 0.06) !important;
}}
[data-baseweb="popover"] [role="option"],
[data-baseweb="popover"] [role="listbox"] li {{
    /* Allow long descriptive labels to wrap to a second line within
       the trigger's natural width. Padding and line-height tuned so
       both 1- and 2-line items read as deliberate. */
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: clip !important;
    word-break: normal;
    overflow-wrap: anywhere;
    padding: 0.6rem 0.95rem !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.005em;
    font-variant-numeric: tabular-nums;
    line-height: 1.4 !important;
}}
[data-baseweb="popover"] [role="option"][aria-selected="true"] {{
    background: rgba(214, 164, 92, 0.10) !important;
    color: var(--primary) !important;
    font-weight: 500;
}}
/* Multiselect chip (the selected pill in the input) — same wrap rules
   so a chosen "Reference + Reimers η=0.4 (O24_ML4)" doesn't push the
   input wider than the sidebar. */
[data-baseweb="select"] [data-baseweb="tag"] {{
    max-width: 100% !important;
}}
[data-baseweb="select"] [data-baseweb="tag"] > span {{
    white-space: normal !important;
    overflow-wrap: anywhere;
}}
[data-baseweb="menu"] [role="option"] {{ color: var(--text) !important; }}
[data-baseweb="menu"] [role="option"]:hover {{
    background-color: rgba(214, 164, 92, 0.13) !important;
    color: var(--primary) !important;
}}
[data-baseweb="tag"] {{
    background-color: rgba(214, 164, 92, 0.18) !important;
    color: var(--primary) !important;
    border: 1px solid rgba(214, 164, 92, 0.42) !important;
    border-radius: 3px !important;
}}

[data-testid="stButtonGroup"] button {{
    background-color: transparent !important;
    color: var(--text-muted) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    font-size: 1.7rem;
    font-weight: 400;
    letter-spacing: -0.005em;
    padding: 1.0rem 1.85rem !important;
    transition: color 0.12s ease, border-color 0.12s ease, background-color 0.12s ease;
}}
[data-testid="stButtonGroup"] button:hover {{
    color: var(--text) !important;
    border-color: var(--border-strong) !important;
}}
[data-testid="stButtonGroup"] button[aria-pressed="true"] {{
    background: linear-gradient(180deg,
        rgba(214, 164, 92, 0.18) 0%,
        rgba(214, 164, 92, 0.10) 100%) !important;
    color: var(--primary) !important;
    border-color: var(--primary) !important;
    font-weight: 500;
    box-shadow: inset 0 -2px 0 rgba(143, 196, 176, 0.7);  /* cyan underline */
}}

[data-testid="stMetric"] {{
    background-color: transparent;
    padding: 0.5rem 0 0.3rem 0.7rem;
    border: none;
    border-left: 2px solid rgba(143, 196, 176, 0.65);  /* cyan accent rule */
    border-top: 1px solid var(--border);
    margin: 0.15rem 0;
}}
[data-testid="stMetricLabel"] {{
    color: var(--text-subtle) !important;
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    font-weight: 500;
}}
[data-testid="stMetricValue"] {{
    color: var(--text) !important;
    font-weight: 500;
    font-size: 1.45rem !important;
    font-family: {FONT_SERIF} !important;
    font-variant-numeric: tabular-nums;
    letter-spacing: -0.01em;
}}

.stButton > button, .stDownloadButton > button {{
    background-color: transparent;
    color: var(--primary);
    border: 1px solid var(--primary);
    border-radius: 4px;
    font-weight: 500;
    font-size: 1.0rem;
    letter-spacing: 0.005em;
    transition: background-color 0.15s ease, color 0.15s ease,
                border-color 0.15s ease, box-shadow 0.15s ease;
    padding: 0.6rem 1.3rem;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
    background-color: rgba(214, 164, 92, 0.12);
    color: var(--primary-hover);
    border-color: var(--primary-hover);
    box-shadow: 0 0 0 3px rgba(214, 164, 92, 0.10);  /* copper halo */
}}

[role="tablist"] {{
    background: transparent !important;
    border-bottom: 1px solid var(--border);
    gap: 0 !important; padding: 0 !important;
    margin-bottom: 0.85rem;
}}
[role="tab"] {{
    background: transparent !important;
    color: var(--text-muted) !important;
    border-radius: 0 !important;
    padding: 1.1rem 2.1rem !important;
    font-weight: 400;
    font-size: 2.0rem;
    letter-spacing: -0.005em;
    border-bottom: 3px solid transparent !important;
    transition: color 0.15s ease, border-color 0.15s ease;
    position: relative;
}}
[role="tab"]:hover {{ color: var(--text) !important; }}
[role="tab"][aria-selected="true"] {{
    color: var(--primary) !important;
    border-bottom: 3px solid var(--primary) !important;
    font-weight: 500;
}}
/* Warm accent dot under the active tab — small celestial touch */
[role="tab"][aria-selected="true"]::after {{
    content: "";
    position: absolute;
    bottom: -3.5px;
    left: 50%;
    transform: translateX(-50%);
    width: 6px; height: 6px;
    background: var(--accent);
    border-radius: 50%;
    box-shadow: 0 0 8px rgba(143, 196, 176, 0.8);
}}

[data-testid="stExpander"] {{
    background: rgba(58, 63, 80, 0.25);
    border: 1px solid var(--border);
    border-radius: 4px;
    transition: border-color 0.15s ease;
}}
[data-testid="stExpander"]:hover {{ border-color: var(--border-strong); }}
[data-testid="stExpander"] summary {{
    font-weight: 500;
    color: var(--text-muted);
    font-size: 0.88rem;
    letter-spacing: 0.005em;
}}
[data-testid="stExpander"] summary:hover {{ color: var(--text); }}

.stPlotlyChart {{
    background: var(--bg-plot);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 6px;
    /* Subtle outer glow + inner vignette — frames the plot like a
       scientific figure rather than a pasted-in widget. The outer
       1px ring is a barely-visible cyan rim. */
    box-shadow:
        0 1px 0 rgba(0, 0, 0, 0.32),
        0 0 0 1px rgba(143, 196, 176, 0.06),
        inset 0 0 38px rgba(0, 0, 0, 0.28);
}}
[data-testid="stNotification"] {{
    border-radius: 3px;
    border-left-width: 2px;
    background-color: var(--bg-raised);
    color: var(--text);
}}
[data-testid="stDataFrame"] {{
    background: var(--bg-raised);
    border: 1px solid var(--border);
    border-radius: 3px;
}}
[data-testid="stToggle"] label, [data-testid="stCheckbox"] label, [data-baseweb="radio"] label {{
    color: var(--text);
    font-size: 1.0rem;
    font-weight: 400;
}}
.stSlider [data-baseweb="slider"] [role="slider"] {{
    background: var(--primary) !important;
    border-color: var(--primary) !important;
}}
.stSlider [data-baseweb="slider"] > div > div {{ background: var(--primary) !important; }}

a, a:visited {{
    color: var(--primary);
    text-decoration: none;
    border-bottom: 1px dotted var(--primary);
}}
a:hover {{ color: var(--primary-hover); border-bottom-color: var(--primary-hover); }}

/* File uploader */
[data-testid="stFileUploader"] section {{
    background: var(--bg-raised) !important;
    border: 1px dashed var(--border-strong) !important;
    border-radius: 3px !important;
}}
[data-testid="stFileUploader"] section:hover {{
    border-color: var(--primary) !important;
}}

/* Code blocks */
code, pre {{
    background: var(--bg-plot) !important;
    color: var(--accent) !important;
    border: 1px solid var(--border);
    border-radius: 3px;
    font-family: {FONT_MONO} !important;
}}

::-webkit-scrollbar {{ width: 9px; height: 9px; }}
::-webkit-scrollbar-track {{ background: var(--bg); }}
::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 0; }}
::-webkit-scrollbar-thumb:hover {{ background: var(--border-strong); }}
::selection {{ background: rgba(214, 164, 92, 0.38); color: var(--text); }}

/* Detail panel for click-inspect — uses the cyan accent on its left rule
   to mark "this is what you've selected" — distinct from the copper used
   for interactive elements throughout the rest of the chrome. */
.detail-panel {{
    background: linear-gradient(180deg,
        rgba(31, 32, 48, 0.92) 0%,
        rgba(31, 32, 48, 0.62) 100%);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 4px;
    padding: 1.1rem 1.2rem 0.95rem 1.2rem;
    font-size: 0.92rem;
    line-height: 1.55;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.32);
}}
.detail-panel h5 {{
    margin: 0 0 0.75rem 0;
    font-family: {FONT_SERIF};
    color: var(--accent);
    font-size: 1.05rem;
    font-weight: 500;
    letter-spacing: -0.005em;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(143, 196, 176, 0.28);
}}
.detail-panel .kv {{
    display: flex; justify-content: space-between; align-items: baseline;
    padding: 0.28rem 0;
    border-bottom: 1px dashed rgba(74, 81, 98, 0.55);
    font-variant-numeric: tabular-nums;
}}
.detail-panel .kv:last-child {{ border-bottom: none; }}
.detail-panel .k {{
    color: var(--text-muted);
    font-size: 0.85rem;
    letter-spacing: 0.005em;
}}
.detail-panel .v {{
    color: var(--text);
    font-family: {FONT_MONO};
    font-size: 0.88rem;
    font-weight: 400;
}}

/* ======================================================================
   v2.1 Design system additions — inspired by Linear, Vercel, Stripe,
   Anthropic Console, Observable. Hero, KPI cards, section headers,
   glass surfaces, footer.
   ====================================================================== */

/* Hide Streamlit's default H1 — we render our own hero instead. */
.stApp h1:first-of-type:not(.hero-title) {{ display: none; }}

/* ---- Hero ----------------------------------------------------------- */
.hero {{
    display: grid;
    grid-template-columns: 1fr auto;
    align-items: end;
    gap: 0.8rem 2rem;
    padding: 0.2rem 0 0.85rem 0;
    margin: 0 0 0.9rem 0;
    border-bottom: 1px solid var(--border);
    position: relative;
}}
.hero::after {{
    /* Sub-rule: copper "ignition" → cyan → fade, sitting on the border */
    content: "";
    position: absolute;
    left: 0; right: 0; bottom: -1px;
    height: 1.5px;
    background: linear-gradient(90deg,
        rgba(214,164,92,0.95) 0%,
        rgba(143,196,176,0.7) 12%,
        rgba(143,196,176,0.32) 38%,
        rgba(143,196,176,0) 72%);
    pointer-events: none;
}}
.hero-eyebrow {{
    color: var(--accent);
    font-family: {FONT_MONO};
    font-size: 0.78rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.22em;
    margin-bottom: 0.45rem;
}}
.hero-eyebrow .dot {{
    display: inline-block;
    width: 4px; height: 4px;
    background: var(--accent);
    border-radius: 50%;
    margin: 0 0.45rem;
    transform: translateY(-2px);
    box-shadow: 0 0 6px rgba(143, 196, 176, 0.7);
}}
.hero-title-row {{
    display: flex;
    align-items: baseline;
    flex-wrap: wrap;
    gap: 0.4rem 1.1rem;
    margin: 0 0 0.35rem 0;
}}
.hero-title {{
    font-family: {FONT_SERIF};
    font-size: 2.9rem;
    font-weight: 500;
    color: var(--text);
    letter-spacing: -0.022em;
    line-height: 1.0;
    margin: 0;
}}
.hero-subtitle {{
    font-family: {FONT_SERIF};
    font-style: italic;
    font-size: 1.18rem;
    font-weight: 400;
    color: var(--text-muted);
    line-height: 1.35;
    /* Sits at the baseline of the title — feels like a continuation
       rather than a separate heading line. */
    letter-spacing: 0;
}}
.hero-tagline {{
    font-family: {FONT_SERIF};
    font-size: 1.14rem;
    font-weight: 400;
    color: var(--text-muted);
    line-height: 1.5;
    margin: 0;
    /* Single-line tagline — paper title in italic plus a small bracketed
       link to NASA ADS. Whitespace nowrap on most browsers, but allow
       wrap on narrow viewports so the link doesn't disappear. */
    white-space: normal;
}}
.hero-tagline em {{
    color: var(--text);
    font-style: italic;
    font-weight: 500;
    letter-spacing: -0.005em;
}}
.hero-paperlink {{
    color: var(--accent);
    font-family: {FONT_SANS};
    font-style: normal;
    font-size: 0.92rem;
    font-weight: 500;
    letter-spacing: 0.02em;
    text-decoration: none;
    border: none;
    padding: 0 0.05rem;
    transition: color 0.12s ease, letter-spacing 0.12s ease;
    white-space: nowrap;
}}
.hero-paperlink:hover {{
    color: var(--primary-hover);
    letter-spacing: 0.025em;
}}
.hero-meta {{
    display: flex; flex-direction: column;
    align-items: flex-end;
    gap: 0.55rem;
    text-align: right;
    padding-bottom: 0.2rem;
}}
.pill {{
    display: inline-flex; align-items: center; gap: 0.4rem;
    font-family: {FONT_MONO};
    font-size: 0.82rem;
    font-weight: 500;
    letter-spacing: 0.05em;
    padding: 0.3rem 0.8rem;
    border-radius: 99px;
    background: rgba(143, 196, 176, 0.08);
    border: 1px solid rgba(143, 196, 176, 0.4);
    color: var(--accent);
}}
.pill .live-dot {{
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--accent);
    box-shadow: 0 0 6px rgba(143, 196, 176, 0.85);
    animation: pulse 2.4s ease-in-out infinite;
}}
@keyframes pulse {{
    0%, 100% {{ opacity: 1; transform: scale(1); }}
    50%      {{ opacity: 0.55; transform: scale(0.85); }}
}}
.hero-doi {{
    color: var(--text-muted);
    font-family: {FONT_MONO};
    font-size: 0.86rem;
    letter-spacing: 0.005em;
    text-decoration: none;
    border-bottom: 1px dotted rgba(214, 164, 92, 0.55);
}}
.hero-doi:hover {{
    color: var(--primary);
    border-bottom-color: var(--primary);
}}

/* ---- KPI strip (compact, single-row inline stats) ----------------- */
.kpi-strip {{
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0 1.6rem;
    margin: 0 0 0.9rem 0;
    padding: 0.55rem 1.0rem;
    background: linear-gradient(180deg,
        rgba(31, 32, 48, 0.55) 0%,
        rgba(31, 32, 48, 0.32) 100%);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border: 1px solid var(--border);
    border-left: 2px solid var(--accent);
    border-radius: 4px;
}}
.kpi-strip .kpi-stat {{
    display: inline-flex;
    align-items: baseline;
    gap: 0.55rem;
    padding: 0.15rem 0;
    flex-wrap: nowrap;
}}
.kpi-strip .kpi-stat + .kpi-stat {{
    border-left: 1px solid var(--border);
    padding-left: 1.6rem;
    margin-left: 0;
}}
.kpi-strip .kpi-label {{
    color: var(--text-subtle);
    font-family: {FONT_SANS};
    font-size: 0.74rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.18em;
}}
.kpi-strip .kpi-value {{
    color: var(--text);
    font-family: {FONT_SERIF};
    font-size: 1.2rem;
    font-weight: 500;
    letter-spacing: -0.01em;
    line-height: 1;
    font-variant-numeric: tabular-nums;
}}
.kpi-strip .kpi-subtext {{
    color: var(--text-muted);
    font-family: {FONT_SANS};
    font-size: 0.85rem;
    font-weight: 400;
    font-variant-numeric: tabular-nums;
    margin-left: 0.15rem;
}}

/* ---- Section header (compact inline header at top of each tab) ----- */
.section-header {{
    display: flex;
    align-items: baseline;
    flex-wrap: wrap;
    gap: 0.3rem 1.1rem;
    margin: 0.1rem 0 0.7rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid var(--border);
}}
.section-eyebrow {{
    color: var(--accent);
    font-family: {FONT_MONO};
    font-size: 0.74rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.2em;
}}
.section-title {{
    color: var(--text);
    font-family: {FONT_SERIF};
    font-size: 1.25rem;
    font-weight: 500;
    letter-spacing: -0.012em;
    margin: 0;
    line-height: 1.2;
}}
.section-tagline {{
    color: var(--text-muted);
    font-family: {FONT_SANS};
    font-size: 0.92rem;
    font-weight: 400;
    line-height: 1.4;
    margin: 0;
    flex: 1 1 auto;
    min-width: 18rem;
}}

/* ---- Glass treatment on existing surfaces --------------------------- */
[data-testid="stExpander"] {{
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
}}
[data-baseweb="popover"] {{
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
}}
.detail-panel {{
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
}}

/* ---- Footer --------------------------------------------------------- */
.app-footer {{
    margin: 3.5rem 0 0.8rem 0;
    padding: 1.8rem 0 1.2rem 0;
    border-top: 1px solid var(--border);
    position: relative;
    display: grid;
    grid-template-columns: 1.4fr 1fr 1fr 1fr;
    gap: 2rem;
}}
.app-footer::before {{
    content: "";
    position: absolute;
    left: 0; right: 0; top: -1px;
    height: 1.5px;
    background: linear-gradient(90deg,
        rgba(143,196,176,0) 0%,
        rgba(143,196,176,0.5) 25%,
        rgba(214,164,92,0.65) 50%,
        rgba(143,196,176,0.5) 75%,
        rgba(143,196,176,0) 100%);
    pointer-events: none;
}}
.footer-brand {{
    font-family: {FONT_SERIF};
    color: var(--text);
    font-size: 1.0rem;
    font-weight: 500;
    letter-spacing: -0.005em;
    margin-bottom: 0.35rem;
}}
.footer-tagline {{
    color: var(--text-subtle);
    font-family: {FONT_SANS};
    font-size: 0.8rem;
    line-height: 1.55;
}}
.footer-heading {{
    color: var(--text-subtle);
    font-family: {FONT_SANS};
    font-size: 0.66rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    margin-bottom: 0.55rem;
}}
.footer-link {{
    display: block;
    color: var(--text-muted);
    font-family: {FONT_SANS};
    font-size: 0.84rem;
    text-decoration: none;
    margin-bottom: 0.35rem;
    border-bottom: none !important;
    transition: color 0.12s ease;
}}
.footer-link:hover {{ color: var(--primary); }}
.footer-link .arrow {{
    color: var(--text-subtle);
    margin-left: 0.25rem;
    transition: transform 0.12s ease, color 0.12s ease;
    display: inline-block;
}}
.footer-link:hover .arrow {{
    color: var(--accent);
    transform: translate(2px, -2px);
}}
.footer-meta {{
    color: var(--text-subtle);
    font-family: {FONT_MONO};
    font-size: 0.74rem;
    line-height: 1.7;
    letter-spacing: 0.01em;
}}
.footer-authors {{
    color: var(--text-muted);
    font-family: {FONT_SANS};
    font-size: 0.84rem;
    line-height: 1.55;
}}

/* ---- Best-fit (χ²) result panel ------------------------------------- */
.fit-panel {{
    margin: 0.7rem 0 0.4rem 0;
    padding: 0.85rem 1.0rem 0.7rem 1.0rem;
    background: linear-gradient(180deg,
        rgba(31, 32, 48, 0.7) 0%,
        rgba(31, 32, 48, 0.4) 100%);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 4px;
}}
.fit-panel h5 {{
    margin: 0 0 0.55rem 0;
    color: var(--accent);
    font-family: {FONT_SERIF};
    font-size: 1.0rem;
    font-weight: 500;
    letter-spacing: -0.005em;
    padding-bottom: 0.4rem;
    border-bottom: 1px dashed rgba(143, 196, 176, 0.32);
}}
.fit-row {{
    display: flex; align-items: baseline; flex-wrap: wrap;
    gap: 0.5rem 0.8rem;
    padding: 0.32rem 0;
    border-bottom: 1px dashed rgba(74, 81, 98, 0.45);
    font-family: {FONT_SANS};
    font-size: 0.92rem;
    color: var(--text-muted);
}}
.fit-row:last-child {{ border-bottom: none; }}
.fit-name {{
    color: var(--text);
    font-weight: 500;
    font-family: {FONT_SERIF};
    font-size: 1.0rem;
    min-width: 6ch;
}}
.fit-pill {{
    color: var(--primary);
    background: rgba(214, 164, 92, 0.10);
    border: 1px solid rgba(214, 164, 92, 0.35);
    border-radius: 99px;
    padding: 0.08rem 0.6rem;
    font-family: {FONT_MONO};
    font-size: 0.82rem;
    font-variant-numeric: tabular-nums;
}}
.fit-meta {{
    color: var(--text-muted);
    font-family: {FONT_SANS};
    font-size: 0.9rem;
    font-variant-numeric: tabular-nums;
}}
.fit-num {{ color: var(--text); font-family: {FONT_MONO}; }}

/* ---- Bottom horizon (mirror of top) --------------------------------- */
.stApp::after {{
    content: "";
    position: fixed;
    bottom: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg,
        rgba(214,164,92,0) 0%,
        rgba(214,164,92,0.35) 25%,
        rgba(143,196,176,0.5) 50%,
        rgba(214,164,92,0.35) 75%,
        rgba(214,164,92,0) 100%);
    z-index: 9999;
    pointer-events: none;
}}
</style>
"""


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Cepheid MESA Grid Viewer",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(_theme_css(), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loaders — WITH st.status PROGRESS (#14)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def list_sets() -> list[str]:
    return sorted(p.stem for p in DATA_DIR.glob("*.parquet"))


@st.cache_data(show_spinner=False)
def _load_sets_cached(set_names: tuple[str, ...]) -> pd.DataFrame:
    """Actual loader — cached. UI progress happens in load_sets_with_status."""
    frames = [pd.read_parquet(DATA_DIR / f"{s}.parquet") for s in set_names]
    df = pd.concat(frames, ignore_index=True)

    mags = ["abs_mag_V", "abs_mag_I", "abs_mag_J", "abs_mag_H", "abs_mag_K"]
    if all(m in df.columns for m in mags):
        df["V_minus_I"] = df["abs_mag_V"] - df["abs_mag_I"]
        df["V_minus_K"] = df["abs_mag_V"] - df["abs_mag_K"]
        df["J_minus_K"] = df["abs_mag_J"] - df["abs_mag_K"]
        df["J_minus_H"] = df["abs_mag_J"] - df["abs_mag_H"]
        df["H_minus_K"] = df["abs_mag_H"] - df["abs_mag_K"]
        # Wesenheit reddening-free magnitudes — Cepheid distance-scale
        # convention. Soszyński et al. (R_VI = 2.55) for the optical
        # Wesenheit, Ripepi et al. (R_VK = 0.13) for the near-IR one.
        # Both reduce to the absolute magnitude when interstellar
        # reddening law follows the assumed coefficient.
        df["W_VI"] = df["abs_mag_V"] - 2.55 * df["V_minus_I"]
        df["W_VK"] = df["abs_mag_V"] - 0.13 * df["V_minus_K"]
    if "star_age" in df.columns:
        df["log_age"] = np.log10(df["star_age"].clip(lower=1.0)).astype("float32")

    # Per-track evolutionary-phase classification (used in hover and
    # for the best-fit-track inference).
    if all(c in df.columns for c in ["center_h1", "center_he4"]):
        df = (df.groupby(["set", "mass", "Z", "Y"], sort=False, group_keys=False)
                .apply(_classify_track_phases))
    return df


def _classify_track_phases(g: pd.DataFrame) -> pd.DataFrame:
    """
    Tag each row of a single track with a coarse evolutionary phase.

    Uses central abundances as the simplest robust proxies — no need for
    Lnuc breakdown columns or convective-zone diagnostics that may not be
    in every parquet:

      * Pre-MS  : before central H burning sets in (Xc still ≈ initial)
      * MS      : core hydrogen burning (Xc > 1e-3)
      * Crossing 1 (H-shell) : Xc exhausted, Yc ≈ initial
                  (Hertzsprung-gap / first IS crossing)
      * Blue loop (He-core)  : core helium burning (Yc < 0.99 * initial)
      * Post He : Yc exhausted (< 1e-3)

    Phase boundaries are sequential model-number indices, so a blue-loop
    point that bounces back into the IS still reads as "Blue loop" even
    if it momentarily crosses the same Teff again.
    """
    h1 = g["center_h1"].to_numpy()
    he4 = g["center_he4"].to_numpy()
    n = len(h1)
    if n == 0:
        g = g.copy()
        g["phase"] = pd.Series([], dtype="string")
        return g

    init_h1 = float(h1[0]) if h1[0] > 0 else 0.7
    init_he4 = float(he4[0]) if he4[0] > 0 else 0.27

    # ZAMS = first model with appreciable H depletion
    zams_mask = h1 < 0.99 * init_h1
    zams = int(zams_mask.argmax()) if zams_mask.any() else 0
    # TAMS = first model with central H exhausted
    tams_mask = h1 < 1e-3
    tams = int(tams_mask.argmax()) if tams_mask.any() else n
    # He onset: after TAMS, first model with He depletion starting
    after_tams = np.arange(n) > tams
    he_start_mask = after_tams & (he4 < 0.99 * init_he4)
    he_start = int(he_start_mask.argmax()) if he_start_mask.any() else n
    # He end: after He onset, first model with central He exhausted
    after_he_start = np.arange(n) > he_start
    he_end_mask = after_he_start & (he4 < 1e-3)
    he_end = int(he_end_mask.argmax()) if he_end_mask.any() else n

    phases = np.empty(n, dtype=object)
    phases[:] = "Main sequence"
    if zams > 0:
        phases[:zams] = "Pre-MS"
    if tams < n:
        phases[tams:he_start] = "Crossing 1 · H-shell"
    if he_start < n:
        phases[he_start:he_end] = "Blue loop · He-core"
    if he_end < n:
        phases[he_end:] = "Post He-burning"

    g = g.copy()
    g["phase"] = phases
    return g


def load_sets_with_status(set_names: tuple[str, ...]) -> pd.DataFrame:
    # If already cached, this is instant and the status block completes immediately.
    with st.status("Loading track data…", expanded=False) as status:
        status.update(label=f"Reading {len(set_names)} parquet file(s)…")
        df = _load_sets_cached(set_names)
        status.update(
            label=f"Loaded {len(df):,} rows across "
                  f"{df.groupby(['set','mass','Z','Y'],sort=False).ngroups} tracks",
            state="complete", expanded=False,
        )
    return df


# --- IS loaders ------------------------------------------------------------
TAB16_COLUMNS = [
    "is_id", "set", "crossing", "mass", "Z", "X",
    "log_age", "log_Teff", "log_L", "log_R", "Yc",
    "P_F", "P_1O",
    "M_V", "M_I", "M_J", "M_H", "M_K",
]

TAB6_COLUMNS = [
    "set", "Z", "mass",
    "log_tcross_1", "log_age_1", "Pdot_over_P_1",
    "log_tcross_2", "log_age_2", "Pdot_over_P_2",
    "log_tcross_3", "log_age_3", "Pdot_over_P_3",
]


def _find(fname: str) -> Path | None:
    for d in (MODELS_DIR, DATA_DIR, ROOT):
        p = d / fname
        if p.exists():
            return p
    return None


def _read_positional(path: Path, columns: list[str]) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(
            path, sep=r"\s+", names=columns, header=None,
            engine="python", skip_blank_lines=True, comment="#",
            # IMPORTANT: '*' is NOT in na_values. In Tab6, an asterisk
            # suffixes legitimate log_age values to flag "the tip of
            # the blue loop falls inside the IS" (see README in
            # models/). Treating '*' as NaN destroys those values.
            # We strip the asterisk *after* parsing, before coercing
            # to numeric, so the value is preserved.
            # 'x' is correctly NaN — Tab16 uses it to flag non-
            # converged RSP models that have no defined period.
            na_values=["-", "x", "X", "--"],
        )
    except Exception:
        return None
    string_cols = {"is_id", "set", "crossing"}
    for c in columns:
        if c in string_cols:
            continue
        # Strip Tab6's "loop-tip-in-IS" asterisk flag from numeric
        # columns before coercion — preserves ~540 valid log_age
        # values that would otherwise be lost across both Tab6 files.
        if df[c].dtype == object:
            df[c] = (df[c].astype(str)
                          .str.replace("*", "", regex=False)
                          .str.strip())
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in columns:
        if c in string_cols and df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()
    return df.dropna(how="all")


@st.cache_data(show_spinner=False)
def load_is_data() -> dict[str, dict]:
    out = {}
    specs = [
        ("tab16_hot",  "Tab16_online_hot_IS.dat",  TAB16_COLUMNS),
        ("tab16_cool", "Tab16_online_cool_IS.dat", TAB16_COLUMNS),
        ("tab6_hot",   "Tab6_online_hot_IS.dat",   TAB6_COLUMNS),
        ("tab6_cool",  "Tab6_online_cool_IS.dat",  TAB6_COLUMNS),
    ]
    for key, fname, cols in specs:
        path = _find(fname)
        if path is None:
            continue
        df = _read_positional(path, cols)
        if df is None or df.empty:
            continue
        # Tab16 → derive Wesenheit and log-period columns so the
        # P–L tab can plot them directly.
        if key.startswith("tab16"):
            if all(c in df.columns for c in ["M_V", "M_I", "M_K"]):
                df["W_VI"] = df["M_V"] - 2.55 * (df["M_V"] - df["M_I"])
                df["W_VK"] = df["M_V"] - 0.13 * (df["M_V"] - df["M_K"])
            for col_p, col_log in [("P_F", "log_P_F"), ("P_1O", "log_P_1O")]:
                if col_p in df.columns:
                    p = pd.to_numeric(df[col_p], errors="coerce")
                    df[col_log] = np.log10(p.where(p > 0))
        out[key] = {"df": df, "path": path, "n_rows": len(df)}
    return out


# ---------------------------------------------------------------------------
# HRD scaffolding — spectral-type bands + isoradius lines (HRD only)
# ---------------------------------------------------------------------------
# Harvard spectral-type edges in log_Teff. Order is hot → cool.
# Sources: Gray & Corbally, Pecaut & Mamajek 2013 (standard dwarf scale).
# We use these for bands regardless of luminosity class; it's a visual guide,
# not a classification tool, and over the range of a Cepheid HRD the shifts
# between dwarf/supergiant boundaries are small.
SPECTRAL_TYPES = [
    # (label, hotter_edge, cooler_edge)  in log_Teff
    ("O", 4.90, 4.48),
    ("B", 4.48, 4.01),
    ("A", 4.01, 3.88),
    ("F", 3.88, 3.78),
    ("G", 3.78, 3.72),
    ("K", 3.72, 3.56),
    ("M", 3.56, 3.40),
]

# Spectral-type fills — bright enough to see on the dark slate canvas.
# Different hue per type but all low-saturation so tracks still dominate.
# Opacity tuned empirically against bg = #262a36.
SPECTRAL_COLORS = {
    "O": "rgba(120, 160, 220, 0.22)",   # cool blue
    "B": "rgba(150, 185, 225, 0.20)",   # soft blue
    "A": "rgba(200, 210, 225, 0.18)",   # pale blue-white
    "F": "rgba(225, 220, 200, 0.17)",   # cream
    "G": "rgba(235, 210, 165, 0.20)",   # warm cream
    "K": "rgba(230, 175, 135, 0.22)",   # soft amber
    "M": "rgba(220, 140, 120, 0.24)",   # warm coral
}


def add_hrd_scaffolding(
    fig: go.Figure,
    x: str, y: str,
    *,
    show_spectral: bool = True,
    show_isoradius: bool = True,
    isoradius_log_r: tuple[int, ...] = (-1, 0, 1, 2, 3, 4),
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
):
    """
    Paint spectral-type bands and isoradius diagonals *behind* the tracks
    on an HRD plot (log_Teff on X, log_L on Y).

    `x_range` / `y_range` should be the DATA ranges in (lo, hi) form.
    Scaffolding is clipped to these so it fills the visible plot tightly —
    no wasted empty canvas.

    - Spectral bands: coloured vertical rectangles with letter labels above.
      Uses layer='below' so tracks sit on top.
    - Isoradius lines: from L = 4πR²σT⁴
      →  log(L/L☉) = 2 log(R/R☉) + 4 log(Teff/Teff_☉) with Teff_☉ = 5772 K.

    Only applies if x == 'log_Teff' and y == 'log_L'. No-op otherwise.
    """
    if x != "log_Teff" or y != "log_L":
        return

    # Default fallback range if caller didn't supply data-aware ranges
    if x_range is None:
        x_range = (3.40, 4.90)
    if y_range is None:
        y_range = (-1.0, 6.5)

    logT_lo, logT_hi = sorted(x_range)   # lo < hi always
    logL_lo, logL_hi = sorted(y_range)

    # --- Spectral bands -----------------------------------------------------
    if show_spectral:
        shapes = []
        annotations = []
        for label, t_hot, t_cool in SPECTRAL_TYPES:
            # Only draw if this band overlaps the visible X range at all
            if t_hot < logT_lo or t_cool > logT_hi:
                continue
            t0 = max(t_cool, logT_lo)
            t1 = min(t_hot, logT_hi)
            if t1 - t0 < 0.001:
                continue
            shapes.append(dict(
                type="rect",
                xref="x", yref="paper",
                x0=t0, x1=t1,
                y0=0, y1=1,
                fillcolor=SPECTRAL_COLORS.get(label, "rgba(180,180,180,0.15)"),
                line=dict(width=0),
                layer="below",
            ))
            annotations.append(dict(
                x=(t0 + t1) / 2.0,
                y=1.012,  # slight extra clearance over plot frame
                xref="x", yref="paper",
                text=f"<b>{label}</b>",
                showarrow=False,
                font=dict(size=13, color=THEME["text"],
                          family=FONT_SERIF),
                xanchor="center", yanchor="bottom",
            ))

        existing_shapes = list(fig.layout.shapes or [])
        fig.update_layout(shapes=existing_shapes + shapes)
        existing_annot = list(fig.layout.annotations or [])
        fig.update_layout(annotations=existing_annot + annotations)

    # --- Isoradius lines ----------------------------------------------------
    if show_isoradius:
        logT_sun = 3.7614
        for logR in isoradius_log_r:
            # y = 2 logR + 4 (logT − logT_sun). Line is straight in logT.
            # Find (xa, ya), (xb, yb) = endpoints intersecting the visible box.
            # Compute y at x=logT_lo and x=logT_hi
            y_at_lo = 2 * logR + 4 * (logT_lo - logT_sun)
            y_at_hi = 2 * logR + 4 * (logT_hi - logT_sun)

            # Line is out of view entirely if both endpoints are on the same
            # side of [logL_lo, logL_hi].
            if (y_at_lo < logL_lo and y_at_hi < logL_lo) or \
               (y_at_lo > logL_hi and y_at_hi > logL_hi):
                continue

            # Clip to y-limits: invert y = 2 logR + 4(x - logT_sun)
            # → x = logT_sun + (y - 2 logR) / 4
            def x_at(y_target):
                return logT_sun + (y_target - 2 * logR) / 4.0

            xa, ya = logT_lo, y_at_lo
            xb, yb = logT_hi, y_at_hi
            if ya < logL_lo:
                xa, ya = x_at(logL_lo), logL_lo
            elif ya > logL_hi:
                xa, ya = x_at(logL_hi), logL_hi
            if yb < logL_lo:
                xb, yb = x_at(logL_lo), logL_lo
            elif yb > logL_hi:
                xb, yb = x_at(logL_hi), logL_hi

            # After clipping, if the line is really short skip it
            if abs(xa - xb) < 0.005 and abs(ya - yb) < 0.005:
                continue

            fig.add_trace(go.Scatter(
                x=[xa, xb], y=[ya, yb],
                mode="lines",
                line=dict(
                    color="rgba(178, 184, 196, 0.5)",
                    width=1.0,
                    dash="dash",
                ),
                hoverinfo="skip",
                showlegend=False,
                name=f"R = 10^{logR} R☉",
            ))

            # Label at the HOT end (larger logT → lower-left of reversed axis)
            # Pick the endpoint with larger logT
            if xa > xb:
                lx, ly = xa, ya
            else:
                lx, ly = xb, yb
            # Only draw label if it sits at the visible hot edge (not clipped
            # to a y-limit at a cooler point)
            if logR == 0:
                label = "R = R<sub>⊙</sub>"
            else:
                label = f"R = 10<sup>{logR}</sup> R<sub>⊙</sub>"
            fig.add_annotation(
                x=lx, y=ly,
                text=label,
                showarrow=False,
                xanchor="left", yanchor="bottom",
                xshift=4, yshift=3,
                font=dict(size=11, color="rgba(200, 206, 218, 0.92)",
                          family=FONT_SANS),
                bgcolor="rgba(22, 23, 36, 0.55)",
                borderpad=2,
                bordercolor="rgba(0,0,0,0)",
            )


def add_is_overlay(
    fig: go.Figure, x: str, y: str,
    sel_zs: set[float] | None,
    is_identifier: str = "b",
    crossings: list[int] | None = None,
):
    """
    Draw the instability-strip overlay on an HRD.

    For each (edge identifier, crossing) combination, build a thin band
    polygon from the *hot* + *cool* Tab16 determinations of that edge.
    Colour encodes the edge identity (so the blue edge looks blue and the
    red edge looks red — astronomical convention), dash style encodes the
    crossing number (1=solid, 2=dash, 3=dot).

    The previous implementation drew one polygon over a *mixture* of all
    selected edges and crossings, which produced a chaotic zigzag because
    the closed-polygon line was forced to connect points belonging to
    physically distinct curves. This version keeps each curve separate.
    """
    if "teff" not in x.lower() or "log_l" not in y.lower():
        return
    data = load_is_data()
    hot = data.get("tab16_hot", {}).get("df")
    cool = data.get("tab16_cool", {}).get("df")
    if hot is None or cool is None:
        return

    # What edges to draw
    if is_identifier == "all":
        edges = ["b", "r", "m"]
    elif is_identifier in ("b", "r", "m"):
        edges = [is_identifier]
    else:
        return

    if not crossings:
        crossings = [1, 2, 3]

    # Astronomical convention: blue edge in blue, red edge in red. The
    # midline gets a neutral aged-paper grey so it doesn't compete.
    EDGE_COLOR = {
        "b": "#6db4ff",   # saturated celestial blue
        "r": "#ff6b6b",   # saturated coral
        "m": "#b5ad97",   # neutral (text-muted)
    }
    EDGE_LABEL = {"b": "Blue edge", "r": "Red edge", "m": "Midline"}
    DASH_BY_CROSSING = {1: "solid", 2: "dash", 3: "dot"}

    def _filter(df: pd.DataFrame, edge: str, cr: int) -> pd.DataFrame:
        m = df["log_Teff"].notna() & df["log_L"].notna()
        m &= (df["is_id"] == edge)
        m &= (df["crossing"] == f"{cr}c")
        if sel_zs:
            zv = df["Z"].fillna(-1)
            m &= zv.apply(
                lambda z: any(abs(float(z) - zk) < 1e-4 for zk in sel_zs)
            )
        return df[m].copy()

    for edge in edges:
        color = EDGE_COLOR[edge]
        for cr in crossings:
            h = _filter(hot, edge, cr).sort_values("log_L")
            c = _filter(cool, edge, cr).sort_values("log_L", ascending=False)
            if h.empty or c.empty:
                continue
            xp = np.concatenate([h["log_Teff"].values,
                                 c["log_Teff"].values])
            yp = np.concatenate([h["log_L"].values,
                                 c["log_L"].values])
            dash = DASH_BY_CROSSING.get(cr, "solid")
            # IS bands intentionally NOT in the legend — colours and
            # dash patterns are documented in the Help tab and the
            # popover. Keeps the figure-area legend uncluttered.
            fig.add_trace(go.Scatter(
                x=xp, y=yp,
                fill="toself",
                fillcolor=_to_rgba(color, 0.14),
                line=dict(color=color, width=1.7, dash=dash),
                mode="lines",
                hoverinfo="skip",
                showlegend=False,
            ))


# ---------------------------------------------------------------------------
# Observations upload (#9)  — flexible column name matching
# ---------------------------------------------------------------------------
# Maps our canonical column names to likely alternatives in uploaded CSVs
OBS_ALIASES = {
    "log_Teff":  ["log_teff", "log_Teff", "logTeff", "logT_eff", "log_T", "log_t",
                  "logteff"],
    "Teff":      ["teff", "T_eff", "T_eff_K", "Teff_K"],
    "log_L":     ["log_l", "log_L", "logL", "logL_Lsun", "logLuminosity"],
    "L":         ["l", "L", "luminosity", "L_Lsun", "L_sun"],
    "log_g":     ["log_g", "logg", "log_G"],
    "log_R":     ["log_r", "log_R", "logR", "logR_Rsun"],
    "R":         ["r", "R", "radius", "R_Rsun"],
    "period":    ["period", "P", "P_F", "P0", "period_days", "period_day",
                  "pulsation_period"],
    "mass":      ["mass", "M", "M_Msun", "initial_mass", "stellar_mass"],
    "Z":         ["Z", "z", "metallicity", "feh_Z"],
    "FeH":       ["FeH", "Fe_H", "Fe/H", "[Fe/H]", "feh", "FEH"],
    "M_V":       ["M_V", "MV", "absV", "abs_V", "M_V_mag"],
    "M_I":       ["M_I", "MI", "absI", "abs_I"],
    "M_K":       ["M_K", "MK", "absK", "abs_K"],
    "V-I":       ["V-I", "V_I", "V_minus_I", "VminusI", "V_I_col"],
    "name":      ["name", "star", "star_name", "id", "ID", "designation"],
}


def _match_alias(colnames_lower: list[str], candidates: list[str]) -> str | None:
    """Return first matching original column name from a candidate list."""
    cand_lower = [c.lower() for c in candidates]
    for orig_lower in colnames_lower:
        if orig_lower in cand_lower:
            return orig_lower
    return None


def resolve_obs_column(obs_df: pd.DataFrame, canonical: str) -> str | None:
    """
    Return the actual column name in obs_df matching our canonical name,
    or None if not present. Case-insensitive, strips whitespace/punct.
    """
    if canonical not in OBS_ALIASES:
        return canonical if canonical in obs_df.columns else None
    # Normalise: lowercase, strip spaces and leading/trailing brackets
    def norm(s: str) -> str:
        return re.sub(r"[\s\[\]]+", "", s.lower())
    colmap = {norm(c): c for c in obs_df.columns}
    for alias in OBS_ALIASES[canonical]:
        n = norm(alias)
        if n in colmap:
            return colmap[n]
    return None


def resolve_err_column(obs_df: pd.DataFrame, axis_col: str) -> pd.Series | None:
    """
    Look up an uncertainty column for a given canonical axis name. Tries a
    handful of common conventions; returns the parsed numeric series or None.

    Examples it will pick up:
        log_Teff  →  log_Teff_err / e_log_Teff / sigma_log_Teff / log_Teff_sig
        Teff      →  Teff_err / eTeff / sig_Teff
        log_L     →  log_L_err / e_log_L
        M_V       →  M_V_err / e_M_V

    No automatic σ_logTeff = σ_T / (T ln 10) propagation — the user is
    expected to provide errors in the units of their axis.
    """
    candidates = [
        f"{axis_col}_err", f"{axis_col}err", f"e_{axis_col}",
        f"e{axis_col}", f"err_{axis_col}",
        f"sigma_{axis_col}", f"sig_{axis_col}", f"{axis_col}_sigma",
    ]
    for cand in candidates:
        col = resolve_obs_column(obs_df, cand)
        if col is not None:
            return pd.to_numeric(obs_df[col], errors="coerce")
    return None


def derive_obs_axes(obs_df: pd.DataFrame, x: str, y: str) -> tuple[pd.Series | None, pd.Series | None, list[str]]:
    """
    Given current x/y axis names, return (x_vals, y_vals, messages) from
    the observations df — deriving log_Teff from Teff if needed, etc.
    """
    msgs: list[str] = []

    def get_axis(axis_name: str) -> pd.Series | None:
        # direct match
        col = resolve_obs_column(obs_df, axis_name)
        if col is not None:
            return pd.to_numeric(obs_df[col], errors="coerce")

        # derivations
        if axis_name == "log_Teff":
            c = resolve_obs_column(obs_df, "Teff")
            if c is not None:
                vals = pd.to_numeric(obs_df[c], errors="coerce")
                msgs.append(f"derived log_Teff from {c}")
                return np.log10(vals.clip(lower=1.0))
        if axis_name == "log_L":
            c = resolve_obs_column(obs_df, "L")
            if c is not None:
                vals = pd.to_numeric(obs_df[c], errors="coerce")
                msgs.append(f"derived log_L from {c}")
                return np.log10(vals.clip(lower=1e-6))
        if axis_name == "log_R":
            c = resolve_obs_column(obs_df, "R")
            if c is not None:
                vals = pd.to_numeric(obs_df[c], errors="coerce")
                msgs.append(f"derived log_R from {c}")
                return np.log10(vals.clip(lower=1e-6))
        if axis_name == "V_minus_I":
            c = resolve_obs_column(obs_df, "V-I")
            if c is not None:
                return pd.to_numeric(obs_df[c], errors="coerce")
        return None

    xv = get_axis(x)
    yv = get_axis(y)
    return xv, yv, msgs


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def downsample(df: pd.DataFrame, max_pts: int) -> pd.DataFrame:
    if max_pts <= 0:
        return df
    pieces = []
    for _, g in df.groupby(["set", "mass", "Z", "Y"], sort=False):
        if len(g) <= max_pts:
            pieces.append(g)
        else:
            step = max(1, len(g) // max_pts)
            pieces.append(g.iloc[::step])
    return pd.concat(pieces, ignore_index=True)


# Continuous "mass / Z / Y" colorscale — replaces the default Viridis
# (whose darkest end is so close to the page background that low-mass
# tracks fade out and the bottom of the colorbar merges into the warm
# K/M spectral-type band). Goes cool → warm and stays clearly visible
# against the ink-black Nocturne canvas at every stop.
NOCTURNE_SEQ = [
    [0.00, "#5d9cff"],   # saturated celestial blue (low end, still bright)
    [0.25, "#8fc4b0"],   # pale celadon
    [0.50, "#c9c19c"],   # warm khaki
    [0.75, "#d6a45c"],   # antique brass
    [1.00, "#f5dba0"],   # bright cream (high end)
]

QUAL_PALETTE = [
    # Curated 8-colour qualitative palette tuned for the Nocturne (ink
    # blue-black) backdrop. Lightness held roughly constant so no single
    # track dominates; hues span warm brass → celadon via amber, jade,
    # rose and slate for an evenly-spaced rotation. First entry matches
    # the UI primary so a single-set view feels integrated with the chrome.
    "#d6a45c",  # antique brass (matches primary)
    "#8fc4b0",  # pale celadon (matches accent)
    "#c2a4d4",  # mauve
    "#e0b878",  # old gold
    "#9bc09a",  # sage
    "#d4716e",  # vintage rose
    "#8ea8c4",  # slate blue
    "#cdbfa0",  # parchment
]


def _to_rgba(color: str, alpha: float) -> str:
    c = color.strip()
    if c.startswith("rgba("):
        r, g, b, _ = c[5:-1].split(",", 3)
        return f"rgba({r},{g},{b},{alpha})"
    if c.startswith("rgb("):
        return f"rgba({c[4:-1]},{alpha})"
    if c.startswith("#"):
        h = c.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"
    return c


def get_color_map(values, kind: str) -> dict:
    values = list(values)
    if kind == "continuous" and len(values) > 1:
        scale = NOCTURNE_SEQ
        ordered = sorted(values)
        return {v: pc.sample_colorscale(scale, i / max(len(ordered) - 1, 1))[0]
                for i, v in enumerate(ordered)}
    return {v: QUAL_PALETTE[i % len(QUAL_PALETTE)] for i, v in enumerate(values)}


AUTO_FLIP_X = {"log_Teff"}
AUTO_FLIP_Y = {"abs_mag_V", "abs_mag_I", "abs_mag_J", "abs_mag_H", "abs_mag_K"}
DASH_STYLES = ["solid", "dash", "dashdot", "dot", "longdash", "longdashdot"]


# ---------------------------------------------------------------------------
# Plotly modebar config — high-res publication export
# ---------------------------------------------------------------------------
def plotly_config(filename: str = "cepheid_viewer", fmt: str = "png") -> dict:
    """
    Centralised Plotly chart config. Customises the camera-icon download so
    figures come out at print-quality resolution (1600×1100 @ 2× scale ≈ 600 dpi
    for a one-column journal figure) and with a sensible filename. Set fmt='svg'
    when a vector export is preferred (e.g. for LaTeX/PDF inclusion).
    """
    return {
        "displaylogo": False,
        "scrollZoom": True,
        "toImageButtonOptions": {
            "format": fmt,        # 'png' | 'svg' | 'jpeg' | 'webp'
            "filename": filename,
            "height": 1100,
            "width": 1600,
            "scale": 2,
        },
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    }


# ---------------------------------------------------------------------------
# Axis presets (#3)
# ---------------------------------------------------------------------------
# Each preset sets X, Y, flip_x, flip_y, log_x, log_y
AXIS_PRESETS: dict[str, dict] = {
    "HRD":           {"x": "log_Teff",  "y": "log_L",      "fx": True,  "fy": False, "lx": False, "ly": False},
    "Kiel":          {"x": "log_Teff",  "y": "log_g",      "fx": True,  "fy": True,  "lx": False, "ly": False},
    "Nuclear":       {"x": "star_age",  "y": "center_h1",  "fx": False, "fy": False, "lx": True,  "ly": False},
    "Luminosity":    {"x": "star_age",  "y": "log_L",      "fx": False, "fy": False, "lx": True,  "ly": False},
    "CMD (V−I)":     {"x": "V_minus_I", "y": "abs_mag_V",  "fx": False, "fy": True,  "lx": False, "ly": False},
    "CMD (V−K)":     {"x": "V_minus_K", "y": "abs_mag_V",  "fx": False, "fy": True,  "lx": False, "ly": False},
    "Wesenheit V,I": {"x": "V_minus_I", "y": "W_VI",       "fx": False, "fy": True,  "lx": False, "ly": False},
    "Wesenheit V,K": {"x": "V_minus_K", "y": "W_VK",       "fx": False, "fy": True,  "lx": False, "ly": False},
    "Custom":        None,  # user picks
}


def build_title(sets: list[str], comps: list[tuple[float, float]]) -> str:
    text_color = THEME["text"]
    muted = THEME["text_muted"]
    subtle = THEME["text_subtle"]
    accent = THEME["accent"]
    if len(sets) == 1:
        s = sets[0]
        # Long name already carries the set ID in parentheses
        # ("Reference grid (O24)") — no need to repeat the bare ID.
        long = set_long_name(s)
        line1 = (
            f"<span style='color:{text_color};font-weight:500;font-size:16px;"
            f"font-family:{FONT_SERIF};letter-spacing:-0.005em'>{long}</span>"
        )
    else:
        # For comparisons we keep the compact set IDs — the long names get
        # too verbose when more than two are stacked. The descriptive
        # version is one click away in the sidebar dropdown.
        joined = (f"<span style='color:{accent};margin:0 0.4em;"
                  f"font-weight:400;font-size:13px'>·</span>").join(sets)
        line1 = (f"<span style='color:{text_color};font-weight:500;font-size:16px;"
                 f"font-family:{FONT_SERIF};letter-spacing:-0.005em'>{joined}</span>"
                 f"<span style='color:{subtle};font-size:11.5px;"
                 f"font-family:{FONT_SANS};margin-left:0.6rem'>"
                 f"{len(sets)} sets</span>")
    if len(comps) == 1:
        z, y = comps[0]
        feh = feh_for(z)
        c = f"Z = {z:.4f},  Y = {y:.4f}"
        if feh is not None:
            c += f",  [Fe/H] = {feh:+.2f}"
        line2 = (f"<span style='color:{muted};font-size:12px;"
                 f"font-family:{FONT_SANS};font-variant-numeric:tabular-nums'>"
                 f"{c}</span>")
    elif len(comps) > 1:
        line2 = (f"<span style='color:{muted};font-size:12px;"
                 f"font-family:{FONT_SANS}'>{len(comps)} compositions</span>")
    else:
        line2 = ""
    return line1 + ("<br>" + line2 if line2 else "")


def _axis_style() -> dict:
    text = THEME["text"]
    muted = THEME["text_muted"]
    subtle = THEME["text_subtle"]
    return dict(
        showline=True, linewidth=1.1, linecolor=muted,
        mirror=True,
        ticks="inside", tickwidth=1.1, ticklen=6, tickcolor=muted,
        minor=dict(ticks="inside", ticklen=3, tickcolor=subtle, showgrid=False),
        gridcolor="rgba(245,246,248,0.07)", zeroline=False,
        # Serif axis titles for the academic-figure feel; sans tick labels
        # so numerals stay crisp. `title_font` is plotly's flat shorthand
        # for `title.font` and avoids colliding with the
        # `title=dict(text=...)` keyword passed in build_figure.
        title_font=dict(size=15, color=text, family=FONT_SERIF),
        tickfont=dict(size=12.5, color=muted, family=FONT_SANS),
        automargin=True,
    )


def build_figure(
    df: pd.DataFrame,
    x: str, y: str,
    color_by: str,
    flip_x: bool, flip_y: bool,
    log_x: bool, log_y: bool,
    height: int,
    show_labels: bool,
    title: str = "",
    show_legend: bool = False,
    *,
    monochrome: bool = False,
    line_width: float = 2.2,
    line_alpha: float = 0.75,
    enable_click: bool = False,
) -> go.Figure:
    fig = go.Figure()
    groups = list(df.groupby(["set", "mass", "Z", "Y"], sort=False))
    n_groups = len(groups)

    # Per-set dash mapping — always used when multiple sets are compared,
    # so sets stay distinguishable even when color is encoding mass/Z/Y.
    set_list = list(dict.fromkeys(df["set"].tolist()))
    n_sets = len(set_list)
    multi_set = n_sets > 1
    set_dash_map = {s: DASH_STYLES[i % len(DASH_STYLES)]
                    for i, s in enumerate(set_list)}

    # Decide Scatter vs Scattergl. Dash patterns require SVG (Scattergl ignores
    # the 'dash' property). When multi-set comparison is active, raise the SVG
    # threshold so the dash distinction is preserved.
    svg_threshold = 60 if multi_set else 30
    if enable_click and n_groups <= 80:
        ScatterClass = go.Scatter  # click events need SVG too
    else:
        ScatterClass = go.Scatter if n_groups <= svg_threshold else go.Scattergl
    is_svg = ScatterClass is go.Scatter

    is_continuous = color_by in {"mass", "Z", "Y"} and not monochrome
    if not monochrome:
        cmap = get_color_map(
            df[color_by].unique(),
            "continuous" if is_continuous else "qualitative",
        )

    # Old "monochrome dash map" — kept separately for backward compat with
    # the monochrome code path below; it just aliases to the shared map.
    mono_dash_map = set_dash_map if monochrome else {}

    mono_color = f"rgba(220,225,232,{line_alpha})"
    shown_legend_keys = set()
    # Track which sets have already had a dash-legend entry shown
    dash_legend_shown: set[str] = set()

    for (s, M, Z, Y), g in groups:
        if monochrome:
            color = mono_color
            dash = set_dash_map.get(s, "solid")
            show_mono_legend = multi_set
            legend_name = s
            show_this = show_mono_legend and legend_name not in shown_legend_keys
            shown_legend_keys.add(legend_name)
        else:
            cval = {"set": s, "mass": float(M), "Z": float(Z), "Y": float(Y)}[color_by]
            color = _to_rgba(cmap[cval], line_alpha)
            # Apply per-set dash style whenever there are multiple sets
            dash = set_dash_map.get(s, "solid") if multi_set else "solid"
            disp = cval if color_by == "set" else f"{float(cval):.3g}"
            legend_name = f"{color_by} = {disp}"
            show_this = (show_legend and not is_continuous
                         and legend_name not in shown_legend_keys)
            shown_legend_keys.add(legend_name)

        # customdata for click-inspect + per-point hover phase
        # [set, mass, Z, Y, star_age, model_number, log_Teff, log_L,
        #  center_h1, center_he4, log_g, log_R, phase]
        cols_for_cd = ["star_age", "model_number", "log_Teff", "log_L",
                       "center_h1", "center_he4", "log_g", "log_R"]
        cd_cols = [g[c].values if c in g.columns
                   else np.full(len(g), np.nan) for c in cols_for_cd]
        # Phase is a per-row classification — included so hover can
        # show "what evolutionary stage am I in?" without a separate UI.
        phase_arr = (g["phase"].astype(str).values
                     if "phase" in g.columns
                     else np.full(len(g), "—"))
        customdata = np.column_stack([
            np.full(len(g), s),
            np.full(len(g), float(M)),
            np.full(len(g), float(Z)),
            np.full(len(g), float(Y)),
        ] + cd_cols + [phase_arr])

        line_kwargs = dict(width=line_width, color=color)
        if is_svg:
            line_kwargs["dash"] = dash
        feh_val = feh_for(float(Z))
        feh_str = f"  [Fe/H] = {feh_val:+.2f}" if feh_val is not None else ""
        # customdata index for phase string is len(base) + len(cd_cols) - 1
        # i.e. position 4 (set,M,Z,Y) + 8 (cd_cols) = index 12.
        phase_cd_idx = 4 + len(cd_cols)
        trace = ScatterClass(
            x=g[x], y=g[y],
            mode="lines",
            line=line_kwargs,
            name=legend_name if show_this else "",
            legendgroup=str(legend_name),
            showlegend=show_this,
            customdata=customdata,
            hovertemplate=(
                f"<b>{s}</b>  M = {float(M):.1f} M☉<br>"
                f"Z = {float(Z):.4f}  Y = {float(Y):.4f}{feh_str}<br>"
                f"<i>Phase:</i> %{{customdata[{phase_cd_idx}]}}<br>"
                f"{ax_label(x)}: %{{x:.4g}}<br>"
                f"{ax_label(y)}: %{{y:.4g}}<extra></extra>"
            ),
        )
        fig.add_trace(trace)

    # Per-set dash legend — shown when:
    # - multi-set comparison is active AND
    # - color isn't already encoding 'set' (in which case the main legend covers it) AND
    # - SVG traces are used (dashes actually rendered) AND
    # - we're not in monochrome mode (which has its own dash legend path)
    needs_dash_legend = (
        multi_set
        and is_svg
        and not monochrome
        and color_by != "set"
    )
    if needs_dash_legend:
        # One invisible "legend-only" line per set, using a neutral color so
        # the legend entry reads as dash-style → set name, independent of
        # the continuous colormap used for tracks.
        legend_line_color = "rgba(220, 225, 232, 0.85)"
        for s in set_list:
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode="lines",
                line=dict(
                    width=line_width,
                    color=legend_line_color,
                    dash=set_dash_map.get(s, "solid"),
                ),
                name=s,
                legendgroup=f"__set_{s}",
                showlegend=True,
                hoverinfo="skip",
            ))

    if show_labels:
        masses_seen: dict[float, dict] = {}
        for (s, M, Z, Y), g in groups:
            key = round(float(M), 2)
            if key in masses_seen:
                continue
            # Anchor at the COOLEST point of the track (typically the RGB tip
            # or red-edge of the blue loop). This separates labels by both
            # Teff and L and — on an HRD — places them to the right side of
            # the plot where tracks naturally fan out.
            if x == "log_Teff":
                anchor_idx = int(g[x].idxmin() - g.index[0])
                anchor_row = g.iloc[anchor_idx]
                xshift, yshift = 14, 0
                xanchor = "left"
                yanchor = "middle"
            else:
                # Non-HRD axes: label at track start as before
                anchor_row = g.iloc[0]
                xshift, yshift = -12, 12
                xanchor = "right"
                yanchor = "bottom"

            if monochrome:
                label_color = "#eceff3"
            else:
                cval = {"set": s, "mass": float(M),
                        "Z": float(Z), "Y": float(Y)}[color_by]
                label_color = cmap[cval]
            masses_seen[key] = {
                "x": float(anchor_row[x]), "y": float(anchor_row[y]),
                "color": label_color,
                "xshift": xshift, "yshift": yshift,
                "xanchor": xanchor, "yanchor": yanchor,
            }
        for M, info in masses_seen.items():
            fig.add_annotation(
                x=info["x"], y=info["y"],
                text=f"{M:.1f} M<sub>⊙</sub>",
                showarrow=False,
                xshift=info["xshift"], yshift=info["yshift"],
                xanchor=info["xanchor"], yanchor=info["yanchor"],
                font=dict(size=11, color=info["color"], family=FONT_SANS),
                # Subdued bg pill — readable on the warm K/M bands but
                # doesn't read as a chip-on-the-plot.
                bgcolor="rgba(18, 19, 30, 0.55)",
                borderpad=2.5,
                bordercolor="rgba(0, 0, 0, 0)",
                borderwidth=0,
            )

    if is_continuous:
        vals = sorted(df[color_by].unique())
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(
                colorscale=NOCTURNE_SEQ,
                cmin=float(min(vals)), cmax=float(max(vals)),
                showscale=True,
                colorbar=dict(
                    title=dict(text=ax_label(color_by), side="right",
                               font=dict(size=13.5, color=THEME["text"],
                                         family=FONT_SERIF)),
                    thickness=10, len=0.7, x=1.02, xanchor="left",
                    tickfont=dict(size=11.5, color=THEME["text_muted"],
                                  family=FONT_SANS),
                    outlinewidth=0, bgcolor="rgba(0,0,0,0)",
                    ticks="outside", ticklen=4,
                    tickcolor=THEME["text_subtle"],
                ),
                size=0.1,
            ),
            hoverinfo="skip", showlegend=False,
        ))

    axis_style = _axis_style()
    paper_bg = THEME["bg_plot"]

    fig.update_layout(
        template="plotly_dark",
        height=height,
        # Generous top margin so the multi-line plot title (set name +
        # composition line) doesn't clip against the plot frame. Also
        # leaves headroom for the spectral-band letter labels on HRDs.
        margin=dict(l=78, r=78, t=110 if title else 38, b=64),
        font=dict(family=FONT_SANS, size=13, color=THEME["text"]),
        paper_bgcolor=paper_bg,
        plot_bgcolor=paper_bg,
        xaxis=dict(title=dict(text=ax_label(x)), **axis_style),
        yaxis=dict(title=dict(text=ax_label(y)), **axis_style),
        legend=dict(
            # Bottom-right of the plot frame: typically the emptiest
            # quadrant on a Cepheid HRD (cool, low-L is unpopulated by
            # 5–10 M☉ tracks). Smaller font + nearly-transparent bg so
            # the legend reads as an annotation, not a panel that fights
            # with the data or the mass-track labels.
            font=dict(size=10.5, color=THEME["text_muted"],
                      family=FONT_SANS),
            itemsizing="constant",
            bgcolor="rgba(22,23,36,0.55)",
            bordercolor="rgba(51,52,74,0.0)", borderwidth=0,
            x=0.99, xanchor="right",
            y=0.02, yanchor="bottom",
            itemwidth=30,
            tracegroupgap=2,
        ),
        showlegend=(
            show_legend
            or (monochrome and len(mono_dash_map) > 1)
            or needs_dash_legend
        ),
        hovermode="closest",
        dragmode="pan",
        hoverlabel=dict(
            bgcolor="rgba(22,23,36,0.96)",
            bordercolor="rgba(143,196,176,0.65)",  # cyan accent border
            font=dict(family=FONT_SANS, size=13, color=THEME["text"]),
            align="left",
        ),
    )

    if title:
        fig.update_layout(title=dict(
            text=title, x=0.0, xanchor="left",
            # y=1.0 with yref="container" pins the title to the top of the
            # full chart area (above plot frame, inside top margin) rather
            # than the plot frame itself — avoids clipping at the top.
            y=0.965, yanchor="top",
            yref="container",
            pad=dict(l=4, t=8),
            font=dict(family=FONT_SERIF, size=16, color=THEME["text"]),
        ))

    if flip_x: fig.update_xaxes(autorange="reversed")
    if flip_y: fig.update_yaxes(autorange="reversed")
    if log_x:  fig.update_xaxes(type="log")
    if log_y:  fig.update_yaxes(type="log")
    return fig


# ---------------------------------------------------------------------------
# Age animation (#10)
# ---------------------------------------------------------------------------
def add_age_animation(
    fig: go.Figure,
    df: pd.DataFrame,
    x: str, y: str,
    n_frames: int = 40,
    frame_duration_ms: int = 60,
) -> go.Figure:
    """
    Add Plotly animation frames + play/pause/slider. Each frame shows track
    segments with star_age <= t_frame. Current visible traces (the full tracks)
    are set to very low opacity to act as "ghost" background; the frames
    overlay growing line segments on top.

    `frame_duration_ms` controls playback speed — smaller = faster.
    """
    if "star_age" not in df.columns or df.empty:
        return fig

    # Build frames at log-spaced age slices (log looks smoother than linear for
    # stellar evolution since early ages pass fast).
    age_vals = df["star_age"].values
    age_vals = age_vals[age_vals > 0]
    if len(age_vals) == 0:
        return fig

    log_min = np.log10(age_vals.min())
    log_max = np.log10(age_vals.max())
    frame_ages = np.logspace(log_min, log_max, n_frames)

    # Ghost out the existing traces (full tracks)
    n_orig = len(fig.data)
    for i in range(n_orig):
        tr = fig.data[i]
        # Skip non-track traces (colorbar dummy, IS overlay, etc.)
        if tr.mode != "lines":
            continue
        # Set track opacity low by adjusting the line colour alpha
        if hasattr(tr, "line") and tr.line and tr.line.color:
            tr.line.color = _to_rgba(tr.line.color, 0.15)

    # Create frames
    frames = []
    groups = list(df.groupby(["set", "mass", "Z", "Y"], sort=False))

    for t in frame_ages:
        frame_traces = []
        for (s, M, Z, Y), g in groups:
            sub = g[g["star_age"] <= t]
            if len(sub) < 2:
                frame_traces.append(go.Scatter(x=[None], y=[None], mode="lines"))
            else:
                frame_traces.append(go.Scatter(
                    x=sub[x], y=sub[y], mode="lines",
                    line=dict(width=2.4, color=THEME["primary"]),
                    hoverinfo="skip", showlegend=False,
                ))
        frames.append(go.Frame(
            data=frame_traces,
            name=f"{t:.2e}",
            traces=list(range(len(frame_traces))),
        ))

    # Add baseline (first-frame) traces so the figure has something to animate
    for tr in frames[0].data:
        fig.add_trace(tr)

    fig.frames = frames

    # Slider + play/pause
    sliders = [dict(
        active=0,
        currentvalue=dict(
            prefix="Age ≤ ", suffix=" yr",
            font=dict(family=FONT_MONO, size=12, color=THEME["text"]),
        ),
        steps=[dict(
            method="animate", label=f"{t:.1e}",
            args=[[f"{t:.2e}"], dict(
                frame=dict(duration=0, redraw=True),
                mode="immediate", transition=dict(duration=0),
            )],
        ) for t in frame_ages],
        pad=dict(t=30, l=10, r=10),
        bgcolor=THEME["bg_raised"], bordercolor=THEME["border"],
        font=dict(family=FONT_SANS, size=10, color=THEME["text_muted"]),
        tickcolor=THEME["text_subtle"],
    )]

    # Solid-brass play / pause buttons — high contrast against the ink
    # plot canvas. (Plotly applies one button-style across the whole
    # menu, so play and pause share the same fill.)
    updatemenus = [dict(
        type="buttons", direction="left",
        x=0.0, y=-0.13, xanchor="left", yanchor="top",
        pad=dict(t=6, l=8, r=8, b=6),
        bgcolor=THEME["primary"],          # antique brass
        bordercolor=THEME["primary_hover"],
        borderwidth=1,
        font=dict(family=FONT_SANS, size=13, color=THEME["bg"]),
        buttons=[
            dict(label="  ▶  Play  ", method="animate",
                 args=[None, dict(
                     frame=dict(duration=int(frame_duration_ms),
                                redraw=True),
                     fromcurrent=True, transition=dict(duration=0),
                 )]),
            dict(label="  ⏸  Pause  ", method="animate",
                 args=[[None], dict(
                     frame=dict(duration=0, redraw=False),
                     mode="immediate",
                 )]),
        ],
    )]

    fig.update_layout(sliders=sliders, updatemenus=updatemenus)
    fig.update_layout(margin=dict(b=130))  # room for controls
    return fig


# ---------------------------------------------------------------------------
# Hero header — eyebrow / serif title / italic tagline / right-side meta
# ---------------------------------------------------------------------------
hero_html = f"""
<div class="hero">
  <div>
    <div class="hero-eyebrow">
      Paper&nbsp;III<span class="dot"></span>Cepheid grid<span class="dot"></span>v{__version__}
    </div>
    <div class="hero-title-row">
      <h1 class="hero-title">Cepheid MESA Grid Viewer</h1>
      <span class="hero-subtitle">Interactive browser for the evolutionary-track grid</span>
    </div>
    <p class="hero-tagline">
      <em>Toward a Comprehensive Grid of Cepheid Models with MESA. III</em>
      &nbsp;<a class="hero-paperlink" target="_blank"
        href="https://ui.adsabs.harvard.edu/abs/2026arXiv260326111S/abstract">[go to paper&nbsp;↗]</a>
    </p>
  </div>
  <div class="hero-meta">
    <span class="pill"><span class="live-dot"></span>v{__version__} · live</span>
    <a class="hero-doi" href="https://doi.org/10.5281/zenodo.17987357"
       target="_blank">Zenodo&nbsp;·&nbsp;10.5281/zenodo.17987357 ↗</a>
  </div>
</div>
"""
st.markdown(hero_html, unsafe_allow_html=True)
# Reserved slot for the KPI card row — populated after the dataframe is
# loaded so the cards show the live filter state.
kpi_slot = st.empty()

available_sets = list_sets()
if not available_sets:
    st.error(f"No Parquet files in `{DATA_DIR}`. Run `python preprocess.py` first.")
    st.stop()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def fmt_mass(m): return f"{float(m):.1f} M☉"

with st.sidebar:
    st.header("Models")
    default_set = "O24" if "O24" in available_sets else available_sets[0]
    selected_sets = st.multiselect(
        "Model set(s)",
        available_sets,
        default=[default_set],
        max_selections=6,
        format_func=set_long_name,
        help="Pick the physics variant(s) to compare. "
             "Reference grid (O24) uses f_core=0.02, f_env=0.04. "
             "_AB: NACRE ¹⁴N(p,γ). _AC: GS98 solar mix. "
             "_AE: ΔY/ΔZ=2. _ML2/4/6: Reimers η=0.2/0.4/0.6.",
    )

if not selected_sets:
    st.info("Select at least one model set in the sidebar.")
    st.stop()

df_full = load_sets_with_status(tuple(selected_sets))
all_masses = sorted(df_full["mass"].unique().tolist())
composition_pairs = sorted({(float(z), float(y))
                            for z, y in zip(df_full["Z"], df_full["Y"])})

def fmt_comp(z, y):
    feh = feh_for(z)
    base = f"Z = {z:.4f},  Y = {y:.4f}"
    if feh is not None:
        base += f"   [Fe/H] = {feh:+.2f}"
    return base


with st.sidebar:
    st.divider()
    st.header("Filter")

    compare_mass = st.toggle("Compare mass range", value=True)
    if compare_mass:
        m_lo, m_hi = st.select_slider(
            "Mass range", options=all_masses,
            value=(min(all_masses), max(all_masses)),
            format_func=fmt_mass,
        )
        mass_step = st.select_slider(
            "Mass step", options=[0.5, 1.0, 2.0], value=1.0,
            format_func=lambda s: f"every {s:g} M☉",
        )
        in_range = [m for m in all_masses if m_lo <= m <= m_hi]
        sel_masses, anchor = [], None
        for m in in_range:
            if anchor is None or (m - anchor) >= (mass_step - 1e-3):
                sel_masses.append(m)
                anchor = m
        if in_range and in_range[-1] not in sel_masses:
            sel_masses.append(in_range[-1])
    else:
        default_m = 5.0 if 5.0 in all_masses else all_masses[len(all_masses) // 2]
        m = st.selectbox("Mass", all_masses, index=all_masses.index(default_m),
                         format_func=fmt_mass)
        sel_masses = [m]

    compare_comp = st.toggle("Compare compositions", value=False)
    comp_labels = [fmt_comp(z, y) for z, y in composition_pairs]
    default_comp_idx = next(
        (i for i, (z, _) in enumerate(composition_pairs) if abs(z - 0.006) < 1e-6),
        len(composition_pairs) // 2,
    )
    if compare_comp:
        sel_labels = st.multiselect(
            "Compositions", comp_labels,
            default=[comp_labels[default_comp_idx]],
        )
    else:
        lbl = st.selectbox("Composition", comp_labels, index=default_comp_idx)
        sel_labels = [lbl]
    sel_comps = [composition_pairs[comp_labels.index(l)] for l in sel_labels]
    sel_zs = {z for z, _ in sel_comps}
    sel_ys = {y for _, y in sel_comps}

    st.divider()
    st.header("Observations")
    obs_upload = st.file_uploader(
        "Upload CSV",
        type=["csv", "txt", "tsv"],
        help=(
            "Overlay observational data on the plot. Expected columns "
            "(flexible names, case-insensitive): log_Teff or Teff, "
            "log_L or L, log_g, period, name. Columns auto-map to the "
            "current X/Y axes when possible."
        ),
        key="obs_upload",
    )
    obs_marker_size = st.slider("Marker size", 4, 16, 9, step=1,
                                key="obs_marker_size")
    obs_color = st.selectbox(
        "Marker colour",
        options=["gold", "red", "cyan", "white"],
        index=0, key="obs_color",
    )
    obs_chi2_fit = st.toggle(
        "Best-fit track  (χ² minimisation)", value=True,
        key="obs_chi2",
        help="For each observation, find the closest sampled point on any "
             "track in the current filter. Reports χ² and (M, Z, age).",
    )

    # ---- "Try sample data" — a one-click way to populate the viewer
    # without going through the file picker. Loads 14 fictional Cepheids
    # from samples/sample_observations.csv. Toggling it again clears it.
    SAMPLE_CSV_PATH = ROOT / "samples" / "sample_observations.csv"
    if SAMPLE_CSV_PATH.exists():
        _sample_active = st.session_state.get("use_sample_obs", False)
        _btn_lbl = ("Clear sample data" if _sample_active
                    else "Try sample data")
        if st.button(
            _btn_lbl,
            key="use_sample_obs_btn",
            use_container_width=True,
            help=("Load 14 fictional Cepheids from "
                  "`samples/sample_observations.csv` to see how the "
                  "upload feature works without preparing your own file."),
        ):
            st.session_state["use_sample_obs"] = not _sample_active
            st.rerun()

    with st.expander("Display options"):
        max_pts = st.slider("Points per track (render)", 100, 2000, 400, step=100)
        show_labels = st.checkbox("Show mass labels on tracks", value=True)
        st.divider()
        line_width = st.slider("Line width", 1.0, 4.0, 1.8, step=0.1)
        line_alpha = st.slider("Line opacity", 0.2, 1.0, 0.82, step=0.05)
        monochrome = st.checkbox(
            "Monochrome mode  (paper-style)", value=False,
            help="All tracks in off-white; multi-set → dashed line patterns.",
        )

# Parse observations — accept either a real upload OR the sample-data
# button. A real upload always wins and clears the sample flag.
obs_df = None
obs_load_err = None
obs_source_label = None  # filled below for the "Parsed columns" expander

_use_sample = st.session_state.get("use_sample_obs", False)
if obs_upload is not None and _use_sample:
    # User has uploaded a real file while sample was active — defer to
    # the upload and silently drop the sample flag.
    st.session_state["use_sample_obs"] = False
    _use_sample = False

raw = None
if obs_upload is not None:
    raw = obs_upload.getvalue().decode("utf-8-sig", errors="replace")
    obs_source_label = obs_upload.name
elif _use_sample and SAMPLE_CSV_PATH.exists():
    raw = SAMPLE_CSV_PATH.read_text(encoding="utf-8-sig")
    obs_source_label = "sample_observations.csv (built-in)"

if raw is not None:
    try:
        # Find the first non-comment, non-blank line and inspect its
        # delimiter. Earlier code only looked at the *first* line, which
        # broke whenever the file led with `#`-comment metadata.
        first_data_line = ""
        for ln in raw.splitlines():
            s = ln.strip()
            if s and not s.startswith("#"):
                first_data_line = ln
                break
        # Vote between common delimiters by counting occurrences in the
        # first real data row. Whichever appears most wins.
        counts = {sep: first_data_line.count(sep)
                  for sep in (",", "\t", ";", "|")}
        sep = max(counts, key=counts.get) if max(counts.values()) > 0 else ","
        obs_df = pd.read_csv(
            io.StringIO(raw),
            sep=sep,
            comment="#",
            skipinitialspace=True,
            engine="python",
        )
        # Strip whitespace from header/string columns so trailing-space
        # column names don't break alias resolution.
        obs_df.columns = [c.strip() for c in obs_df.columns]
        for c in obs_df.select_dtypes(include="object").columns:
            obs_df[c] = obs_df[c].astype(str).str.strip()
    except Exception as e:
        obs_load_err = str(e)

# Show the parsed observation table back to the user — a quick visual
# confirmation that columns were resolved into separate fields rather
# than crammed into a single one (which would suggest a delimiter
# mismatch). Lives in the sidebar Observations area, beneath the upload.
if obs_df is not None and not obs_df.empty:
    with st.sidebar:
        with st.expander(
            f"Parsed columns  ·  {len(obs_df)} rows",
            expanded=False,
        ):
            _src_bits = []
            if obs_source_label:
                _src_bits.append(f"Source: `{obs_source_label}`")
            _src_bits.append(
                f"Delimiter: `{repr(sep).strip(chr(39))}`")
            _src_bits.append(f"{len(obs_df.columns)} columns")
            st.caption("  ·  ".join(_src_bits))
            st.dataframe(
                obs_df,
                use_container_width=True,
                hide_index=True,
                height=min(36 + 24 * len(obs_df), 280),
            )


df_f = df_full[
    df_full["mass"].isin(sel_masses)
    & df_full["Z"].isin(sel_zs)
    & df_full["Y"].isin(sel_ys)
]
n_tracks = df_f.groupby(["set", "mass", "Z", "Y"], sort=False).ngroups
df_plot = downsample(df_f, max_pts) if not df_f.empty else df_f

with st.sidebar:
    st.divider()
    c1, c2 = st.columns(2)
    c1.metric("Tracks", n_tracks)
    c2.metric("Points", f"{len(df_plot):,}")

    st.divider()
    with st.expander("About", expanded=False):
        st.markdown(
            f"""
**Cepheid MESA Grid Viewer**  ·  v{__version__}

Companion viewer to *Smolec et al. — Toward a Comprehensive Grid
of Cepheid Models with MESA. III* (A&A, submitted).

[Zenodo dataset (DOI 10.5281/zenodo.17987357)](https://doi.org/10.5281/zenodo.17987357)

Track grid: 26 model sets · {len(available_sets)} cached locally · {len(all_masses)} initial masses ·
{len(composition_pairs)} (Z, Y) compositions.
"""
        )
        if st.button("Clear data cache",
                     help="Drop cached parquet/IS reads. Use after editing files in data/ "
                          "or models/."):
            st.cache_data.clear()
            st.toast("Cache cleared — rerunning…")
            st.rerun()

if df_f.empty:
    st.warning("No tracks match the current filter.")
    st.stop()


# ---------------------------------------------------------------------------
# Header metadata strip — populates the placeholder reserved at top of page.
# Subdued, single-line summary of what is currently loaded + dataset DOI.
# ---------------------------------------------------------------------------
def _render_kpi_cards():
    """
    Stripe-style KPI row — four cards summarising what's currently loaded
    and filtered. Numbers in serif tabular-nums; labels in tracked
    uppercase sans; subtext in compact sans. Glass surfaces with copper /
    cyan alternating left rules.
    """
    n_sets = len(selected_sets)
    n_masses = len(sel_masses)
    n_comps = len(sel_comps)

    # Subtexts that say something useful, not just repeat the number
    sets_sub = (selected_sets[0] if n_sets == 1
                else f"{', '.join(selected_sets[:3])}"
                + (f" +{n_sets-3}" if n_sets > 3 else ""))
    if sel_masses:
        m_lo, m_hi = float(min(sel_masses)), float(max(sel_masses))
        if n_masses == 1:
            mass_sub = "single mass"
        else:
            mass_sub = f"{m_lo:.1f}–{m_hi:.1f} M☉"
    else:
        mass_sub = "—"
    if n_comps == 1:
        z, y = sel_comps[0]
        feh = feh_for(z)
        comps_sub = (f"Z = {z:.4f}"
                     + (f", [Fe/H] = {feh:+.2f}" if feh is not None else ""))
    else:
        comps_sub = f"across {n_comps} (Z, Y) pairs"
    tracks_sub = (f"{len(df_plot):,} pts rendered"
                  if 'df_plot' in globals() and len(df_plot) > 0
                  else f"{n_tracks} unique")

    cards = [
        ("Sets",         str(n_sets),    sets_sub),
        ("Tracks",       str(n_tracks),  tracks_sub),
        ("Masses",       str(n_masses),  mass_sub),
        ("Compositions", str(n_comps),   comps_sub),
    ]

    if _HAS_ELEMENTS:
        # Real MUI Card grid v4 — per-card accent identity, large serif
        # values, stronger gradient & glow, staggered fade-in entrance.
        # Each KPI gets its own colour family so the four cards read as
        # four distinct readouts rather than four copies.
        CARD_ACCENTS = [
            ("#8fc4b0", "rgba(143,196,176"),  # Sets — celadon
            ("#d6a45c", "rgba(214,164,92"),   # Tracks — brass
            ("#c2a4d4", "rgba(194,164,212"),  # Masses — amethyst
            ("#9bc09a", "rgba(155,192,154"),  # Compositions — sage
        ]
        with kpi_slot.container():
            with elements("kpi_strip_v4"):
                with mui.Stack(
                    direction="row",
                    spacing=1.6,
                    sx={
                        "width": "100%",
                        "marginBottom": "0.6rem",
                        "marginTop": "-0.2rem",
                    },
                ):
                    for i, (label, value, sub) in enumerate(cards):
                        accent, accent_rgb = CARD_ACCENTS[i % len(CARD_ACCENTS)]
                        with mui.Card(
                            elevation=0,
                            sx={
                                "flex": 1,
                                "minWidth": 0,
                                "padding": "0.7rem 1.0rem 0.7rem 1.1rem",
                                "background": (
                                    "linear-gradient(140deg, "
                                    "rgba(31,32,48,0.78) 0%, "
                                    "rgba(31,32,48,0.42) 60%, "
                                    f"{accent_rgb},0.10) 130%)"
                                ),
                                "backdropFilter": "blur(14px)",
                                "WebkitBackdropFilter": "blur(14px)",
                                "border": "1px solid #33344a",
                                "borderLeft": f"3px solid {accent}",
                                "borderRadius": "8px",
                                "color": "#f1ede0",
                                "transition": (
                                    "transform 260ms "
                                    "cubic-bezier(.2,.8,.2,1), "
                                    "border-color 260ms ease, "
                                    "box-shadow 260ms ease"
                                ),
                                "boxShadow": (
                                    "0 2px 0 rgba(0,0,0,0.32), "
                                    f"0 0 0 1px {accent_rgb},0.06)"
                                ),
                                "&:hover": {
                                    "transform": "translateY(-3px)",
                                    "borderColor": f"{accent_rgb},0.55)",
                                    "boxShadow": (
                                        "0 10px 24px rgba(0,0,0,0.42), "
                                        f"0 0 0 1px {accent_rgb},0.45), "
                                        f"0 0 22px {accent_rgb},0.16)"
                                    ),
                                },
                                "position": "relative",
                                "overflow": "hidden",
                                "@keyframes kpiIn": {
                                    "0%": {
                                        "opacity": 0,
                                        "transform": "translateY(8px)",
                                    },
                                    "100%": {
                                        "opacity": 1,
                                        "transform": "translateY(0)",
                                    },
                                },
                                "animation": (
                                    f"kpiIn 380ms cubic-bezier(.2,.8,.2,1) "
                                    f"{i * 70}ms backwards"
                                ),
                            },
                        ):
                            # Smaller corner glow — still visible, doesn't
                            # dominate the now-shorter card.
                            mui.Box(
                                sx={
                                    "position": "absolute",
                                    "top": "-40%", "right": "-15%",
                                    "width": "55%", "height": "150%",
                                    "background": (
                                        "radial-gradient(circle at top right, "
                                        f"{accent_rgb},0.20), "
                                        f"{accent_rgb},0) 70%)"
                                    ),
                                    "pointerEvents": "none",
                                    "zIndex": 0,
                                }
                            )
                            with mui.Box(sx={
                                "position": "relative", "zIndex": 1,
                            }):
                                # Eyebrow: dot + label
                                with mui.Stack(
                                    direction="row",
                                    spacing=0.9,
                                    alignItems="center",
                                    sx={"marginBottom": "0.2rem"},
                                ):
                                    mui.Box(sx={
                                        "width": "6px", "height": "6px",
                                        "borderRadius": "50%",
                                        "backgroundColor": accent,
                                        "boxShadow": (
                                            f"0 0 6px {accent_rgb},0.7)"
                                        ),
                                    })
                                    mui.Typography(
                                        label,
                                        sx={
                                            "color": "#b5ad97",
                                            "fontFamily":
                                                "Inter, sans-serif",
                                            "fontSize": "0.72rem",
                                            "fontWeight": 600,
                                            "textTransform": "uppercase",
                                            "letterSpacing": "0.16em",
                                        },
                                    )
                                # Inline row: big value + thin accent rule
                                # + subtext side-by-side. Compact, ~80 px
                                # tall instead of 160 px.
                                with mui.Stack(
                                    direction="row",
                                    spacing=1.2,
                                    alignItems="baseline",
                                    sx={"marginTop": "0.25rem"},
                                ):
                                    mui.Typography(
                                        value,
                                        sx={
                                            "color": "#f1ede0",
                                            "fontFamily": (
                                                "'Source Serif 4', "
                                                "'Source Serif Pro', "
                                                "Charter, serif"
                                            ),
                                            "fontSize": "1.85rem",
                                            "fontWeight": 500,
                                            "letterSpacing": "-0.022em",
                                            "lineHeight": 1.0,
                                            "fontVariantNumeric":
                                                "tabular-nums",
                                        },
                                    )
                                    mui.Box(sx={
                                        "width": "20px", "height": "2px",
                                        "background": accent,
                                        "borderRadius": "2px",
                                        "opacity": 0.75,
                                        "alignSelf": "center",
                                    })
                                    mui.Typography(
                                        sub,
                                        sx={
                                            "color": "#b5ad97",
                                            "fontFamily":
                                                "Inter, sans-serif",
                                            "fontSize": "0.82rem",
                                            "fontWeight": 400,
                                            "lineHeight": 1.3,
                                            "fontVariantNumeric":
                                                "tabular-nums",
                                            "overflow": "hidden",
                                            "textOverflow": "ellipsis",
                                            "whiteSpace": "nowrap",
                                            "minWidth": 0,
                                        },
                                    )
    else:
        # Fallback: pure HTML strip (used until streamlit-elements is
        # installed). Keeps the app functional without the new dep.
        stats_html = "".join(
            f"""
            <div class="kpi-stat">
              <span class="kpi-label">{label}</span>
              <span class="kpi-value">{value}</span>
              <span class="kpi-subtext">{sub}</span>
            </div>"""
            for label, value, sub in cards
        )
        kpi_slot.markdown(
            f'<div class="kpi-strip">{stats_html}</div>',
            unsafe_allow_html=True,
        )


_render_kpi_cards()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
NUMERIC_COLS = [
    c for c in df_plot.columns
    if c != "set" and pd.api.types.is_numeric_dtype(df_plot[c])
]

def pref(col: str, fallback: int = 0) -> str:
    return col if col in NUMERIC_COLS else NUMERIC_COLS[fallback]


tab_single, tab_multi, tab_pl, tab_cross, tab_data, tab_help = st.tabs(
    ["Single plot", "Multi-panel", "P–L diagram",
     "Crossings", "Data", "Help"]
)


# ========== SINGLE =========================================================
with tab_single:
    st.markdown(
        """
        <div class="section-header">
          <span class="section-eyebrow">Single panel</span>
          <h2 class="section-title">Evolutionary tracks</h2>
          <p class="section-tagline">Pick any axis pair · animate · click any point to inspect.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # View segmented control — full width now that toggles moved to the
    # right rail. Speed-to-ms dict still defined here for use later.
    preset = st.segmented_control(
        "View",
        options=list(AXIS_PRESETS.keys()),
        default="HRD",
        key="preset",
    ) or "HRD"
    SPEED_TO_MS = {"0.1×": 600, "0.5×": 120, "1×": 60, "2×": 30}

    # Resolve axes from preset or custom
    if preset != "Custom":
        p = AXIS_PRESETS[preset]
        x_col = p["x"] if p["x"] in NUMERIC_COLS else pref("log_Teff", 0)
        y_col = p["y"] if p["y"] in NUMERIC_COLS else pref("log_L", 1)
        flip_x, flip_y = p["fx"], p["fy"]
        log_x, log_y = p["lx"], p["ly"]
        with st.expander("Override preset (flip / log / axis swap)"):
            cc1, cc2, cc3, cc4 = st.columns(4)
            flip_x = cc1.checkbox("Flip X", value=flip_x, key="fx_pr")
            flip_y = cc2.checkbox("Flip Y", value=flip_y, key="fy_pr")
            log_x  = cc3.checkbox("Log X",  value=log_x,  key="lx_pr")
            log_y  = cc4.checkbox("Log Y",  value=log_y,  key="ly_pr")
    else:
        c_x, c_y, c_col, c_opt = st.columns([2, 2, 2, 1])
        with c_x:
            x_col = st.selectbox(
                "X axis", NUMERIC_COLS,
                index=NUMERIC_COLS.index(pref("log_Teff", 0)),
                key="x_single",
            )
        with c_y:
            y_col = st.selectbox(
                "Y axis", NUMERIC_COLS,
                index=NUMERIC_COLS.index(pref("log_L", 1)),
                key="y_single",
            )
        with c_opt:
            st.write("")
            st.write("")
            with st.popover("Axis options", use_container_width=True):
                flip_x = st.checkbox("Flip X", value=(x_col in AUTO_FLIP_X), key="fx")
                flip_y = st.checkbox("Flip Y", value=(y_col in AUTO_FLIP_Y), key="fy")
                log_x  = st.checkbox("Log X",  value=False, key="lx")
                log_y  = st.checkbox("Log Y",  value=False, key="ly")

    # Colour-by — full width now. Popovers moved to the right rail
    # beneath the plot.
    is_hrd = (x_col == "log_Teff" and y_col == "log_L")
    default_color = "set" if len(selected_sets) > 1 else "mass"
    color_by = st.segmented_control(
        "Colour by",
        options=["set", "mass", "Z", "Y"],
        default=default_color,
        key="color_single",
    ) or default_color

    # ----------------------------------------------------------------
    # Plot + right rail layout. Right rail contains:
    #   1. Point detail panel (filled after the plot is rendered)
    #   2. Click-to-inspect toggle
    #   3. Animate-age toggle (+ speed selector when on)
    #   4. Instability strip popover
    #   5. HR scaffolding popover
    # The right-rail widgets are rendered FIRST so their state can drive
    # the figure construction below.
    # ----------------------------------------------------------------
    plot_col, right_col = st.columns([4, 1.55])

    with right_col:
        # Reserve a slot for the detail panel — populated after the plot
        # is rendered (so we have the click event in hand).
        detail_slot = st.empty()
        st.markdown(
            "<div style='border-top:1px solid var(--border);"
            "margin:0.7rem 0 0.6rem 0;opacity:0.55'></div>",
            unsafe_allow_html=True,
        )

        click_mode = st.toggle(
            "Click to inspect", value=True, key="click_mode_rail",
            help="Tap a point on a track to open the detail panel.",
        )
        animate = st.toggle(
            "Animate age", value=False, key="animate_rail",
            help="Play the tracks growing from ZAMS along star_age.",
        )
        if animate:
            anim_speed = st.segmented_control(
                "Speed",
                options=list(SPEED_TO_MS.keys()),
                default="1×",
                key="anim_speed",
                label_visibility="collapsed",
                help="Playback speed. 0.1× is slowest, 2× the fastest.",
            ) or "1×"
            anim_frame_ms = SPEED_TO_MS[anim_speed]
        else:
            anim_frame_ms = 60

        with st.popover("Instability strip", use_container_width=True):
            is_data_all = load_is_data()
            is_available = ("tab16_hot" in is_data_all
                            and "tab16_cool" in is_data_all)
            if is_available:
                show_is = st.checkbox("Show instability strip", value=False)
                is_id = st.selectbox(
                    "Edge to draw",
                    options=["b", "r", "m", "all"], index=0,
                    format_func=is_id_long_name,
                    help="Which IS boundary to render. The blue edge is "
                         "the hot pulsation onset, the red edge the cool "
                         "termination; midline is the strip centre.",
                )
                sel_crossings = st.multiselect(
                    "Crossings", options=[1, 2, 3], default=[1, 2, 3],
                    help="1 = Hertzsprung gap, 2 = blue-loop blueward, "
                         "3 = blue-loop redward.",
                )
                filter_by_z = st.checkbox("Filter by current Z", value=True)
            else:
                show_is = False
                is_id = "b"
                sel_crossings = [1, 2, 3]
                filter_by_z = True
                st.caption("IS files not found in models/.")

        with st.popover(
            "HR scaffolding" if is_hrd else "HR scaffolding  (HRD only)",
            use_container_width=True,
            disabled=not is_hrd,
        ):
            if is_hrd:
                show_spectral = st.checkbox(
                    "Spectral-type bands", value=True,
                    help="Faint vertical bands (O B A F G K M) with labels "
                         "above the plot. Harvard classification boundaries.",
                )
                show_isoradius = st.checkbox(
                    "Isoradius lines", value=True,
                    help="Dashed gray lines of constant R, from "
                         "L = 4πR²σT⁴. Labeled at the hot end.",
                )
                iso_range = st.select_slider(
                    "Isoradius range (log R/R☉)",
                    options=[-1, 0, 1, 2, 3, 4, 5],
                    value=(0, 4),
                    help="Which orders of magnitude of R to draw.",
                )
            else:
                show_spectral = False
                show_isoradius = False
                iso_range = (0, 4)

    # Track-count warnings — render outside the right rail so they sit
    # above the plot, full width.
    if n_tracks > 80:
        st.warning(
            f"Rendering {n_tracks} tracks — narrow the selection for snappier panning."
        )
    if n_tracks > 80 and click_mode:
        st.caption("Click-to-inspect disabled above 80 tracks for performance.")

    # Build figure
    fig = build_figure(
        df_plot, x_col, y_col, color_by,
        flip_x, flip_y, log_x, log_y,
        height=700 if not animate else 650,
        show_labels=show_labels and not animate,
        title=build_title(selected_sets, sel_comps),
        show_legend=(color_by == "set" and len(selected_sets) > 1),
        monochrome=monochrome,
        line_width=line_width,
        line_alpha=line_alpha,
        enable_click=click_mode,
    )

    # HR scaffolding (spectral-type bands + isoradius lines) — HRD only
    # Added AFTER build_figure so fig.data exists, but shapes use layer='below'
    # and isoradius traces are prepended to fig.data so tracks render on top.
    if is_hrd and (show_spectral or show_isoradius):
        # Compute data-driven axis ranges with a small margin, so the plot
        # fills its canvas instead of showing empty space out to T = 10^5 K.
        x_vals = df_plot[x_col].dropna().values
        y_vals = df_plot[y_col].dropna().values
        if len(x_vals) > 0 and len(y_vals) > 0:
            x_lo, x_hi = float(x_vals.min()), float(x_vals.max())
            y_lo, y_hi = float(y_vals.min()), float(y_vals.max())
            # Margins: ~6% of each range
            x_pad = max(0.05, 0.06 * (x_hi - x_lo))
            y_pad = max(0.15, 0.06 * (y_hi - y_lo))
            x_data_range = (x_lo - x_pad, x_hi + x_pad)
            y_data_range = (y_lo - y_pad, y_hi + y_pad)

            # Pin the figure axes to this range (overrides autorange).
            # Plotly's X axis is reversed for HRD — pass the range in ascending
            # order and 'autorange="reversed"' still reverses the display.
            fig.update_xaxes(
                range=[x_data_range[1], x_data_range[0]]  # reversed
                if flip_x else [x_data_range[0], x_data_range[1]],
                autorange=False,
            )
            fig.update_yaxes(
                range=[y_data_range[1], y_data_range[0]]
                if flip_y else [y_data_range[0], y_data_range[1]],
                autorange=False,
            )
        else:
            x_data_range = (3.40, 4.90)
            y_data_range = (-1.0, 6.5)

        # Snapshot existing traces (tracks + colorbar dummy) before clearing
        track_traces = list(fig.data)
        fig.data = ()

        add_hrd_scaffolding(
            fig, x_col, y_col,
            show_spectral=show_spectral,
            show_isoradius=show_isoradius,
            isoradius_log_r=tuple(range(iso_range[0], iso_range[1] + 1)),
            x_range=x_data_range,
            y_range=y_data_range,
        )
        # Re-add the track traces on top of the isoradius lines
        for tr in track_traces:
            fig.add_trace(tr)

    if show_is:
        add_is_overlay(
            fig, x_col, y_col,
            sel_zs=sel_zs if filter_by_z else None,
            is_identifier=is_id,
            crossings=sel_crossings,
        )

    # Overlay observations + best-fit-track inference + error rectangles
    obs_plotted = False
    obs_msg = None
    fits_html = ""  # filled by chi² block, rendered below the plot
    if obs_df is not None and not obs_df.empty:
        xv, yv, derive_msgs = derive_obs_axes(obs_df, x_col, y_col)
        if xv is not None and yv is not None:
            valid = xv.notna() & yv.notna()
            if valid.any():
                name_col = resolve_obs_column(obs_df, "name")
                period_col = resolve_obs_column(obs_df, "period")
                xerr = resolve_err_column(obs_df, x_col)
                yerr = resolve_err_column(obs_df, y_col)

                def _label(i):
                    parts = []
                    if name_col:
                        parts.append(f"<b>{obs_df[name_col].iloc[i]}</b>")
                    parts.append(f"{ax_label(x_col)} = {xv.iloc[i]:.3f}")
                    parts.append(f"{ax_label(y_col)} = {yv.iloc[i]:.3f}")
                    if period_col:
                        try:
                            p = float(obs_df[period_col].iloc[i])
                            if np.isfinite(p):
                                parts.append(f"P = {p:.3f} d")
                        except Exception:
                            pass
                    return "<br>".join(parts)

                color_map = {
                    "gold": THEME["accent"], "red": "#e86570",
                    "cyan": "#7fd4d4", "white": "#f5f6f8",
                }
                marker_color = color_map.get(obs_color, THEME["accent"])

                # --- Error rectangles ----------------------------------
                # 2σ × 2σ box centred on each obs point. Drawn underneath
                # the marker so the marker stays visible.
                if xerr is not None or yerr is not None:
                    for i in np.where(valid)[0]:
                        xi = float(xv.iloc[i])
                        yi = float(yv.iloc[i])
                        sx = (float(xerr.iloc[i])
                              if xerr is not None and pd.notna(xerr.iloc[i])
                              else 0.0)
                        sy = (float(yerr.iloc[i])
                              if yerr is not None and pd.notna(yerr.iloc[i])
                              else 0.0)
                        if sx <= 0 and sy <= 0:
                            continue
                        # Use a small floor so a 1-D error still draws a
                        # narrow strip rather than a degenerate line.
                        sxd = max(sx, 1e-6)
                        syd = max(sy, 1e-6)
                        fig.add_shape(
                            type="rect",
                            xref="x", yref="y",
                            x0=xi - sxd, x1=xi + sxd,
                            y0=yi - syd, y1=yi + syd,
                            line=dict(color=marker_color, width=1.0),
                            fillcolor=_to_rgba(marker_color, 0.12),
                            layer="below",
                        )

                fig.add_trace(go.Scatter(
                    x=xv[valid], y=yv[valid],
                    mode="markers",
                    marker=dict(
                        symbol="circle-open",
                        size=obs_marker_size,
                        color=marker_color,
                        line=dict(width=2, color=marker_color),
                    ),
                    name="Observations",
                    hovertext=[_label(i) for i in np.where(valid)[0]],
                    hoverinfo="text",
                    showlegend=True,
                ))
                obs_plotted = True
                msg_bits = []
                if derive_msgs:
                    msg_bits.extend(derive_msgs)
                msg_bits.append(f"{int(valid.sum())} point"
                                + ("s" if int(valid.sum()) != 1 else ""))
                obs_msg = "Observations plotted (" + "; ".join(msg_bits) + ")."

                # --- χ² best-fit track inference -----------------------
                if (obs_chi2_fit and len(df_f) > 0
                        and x_col in df_f.columns and y_col in df_f.columns):
                    track_x = df_f[x_col].to_numpy(dtype=float)
                    track_y = df_f[y_col].to_numpy(dtype=float)
                    finite = np.isfinite(track_x) & np.isfinite(track_y)
                    if finite.any():
                        tx = track_x[finite]
                        ty = track_y[finite]
                        # Lookup tables from df_f index space to compact
                        # filtered track-row arrays
                        track_rows = df_f.iloc[np.where(finite)[0]]
                        results = []
                        # Default σ if user didn't provide error columns:
                        # rough relative uncertainties so fit penalises
                        # gross mismatches but isn't tyrannical.
                        default_sx = 0.02 * (np.nanmax(tx) - np.nanmin(tx)
                                             or 1.0)
                        default_sy = 0.02 * (np.nanmax(ty) - np.nanmin(ty)
                                             or 1.0)
                        for i in np.where(valid)[0]:
                            xi = float(xv.iloc[i])
                            yi = float(yv.iloc[i])
                            sx = (float(xerr.iloc[i])
                                  if xerr is not None
                                  and pd.notna(xerr.iloc[i])
                                  and float(xerr.iloc[i]) > 0
                                  else default_sx)
                            sy = (float(yerr.iloc[i])
                                  if yerr is not None
                                  and pd.notna(yerr.iloc[i])
                                  and float(yerr.iloc[i]) > 0
                                  else default_sy)
                            chi2 = ((tx - xi) / max(sx, 1e-9))**2 \
                                 + ((ty - yi) / max(sy, 1e-9))**2
                            if not np.isfinite(chi2).any():
                                continue
                            j = int(np.nanargmin(chi2))
                            best_row = track_rows.iloc[j]
                            results.append({
                                "name": (str(obs_df[name_col].iloc[i])
                                         if name_col else f"#{i+1}"),
                                "set":  str(best_row["set"]),
                                "mass": float(best_row["mass"]),
                                "Z":    float(best_row["Z"]),
                                "log_age": (float(best_row["log_age"])
                                            if "log_age" in df_f.columns
                                            else float("nan")),
                                "model": (int(best_row["model_number"])
                                          if "model_number" in df_f.columns
                                          else -1),
                                "chi2": float(chi2[j]),
                            })
                        if results:
                            rows = "".join(
                                f"<div class='fit-row'>"
                                f"<span class='fit-name'>{r['name']}</span>"
                                f"<span class='fit-pill'>χ² = {r['chi2']:.2f}</span>"
                                f"<span class='fit-meta'>"
                                f"{r['set']} · "
                                f"<span class='fit-num'>{r['mass']:.1f}</span> M☉"
                                f" · Z = <span class='fit-num'>{r['Z']:.4f}</span>"
                                + (f" · log(age) = <span class='fit-num'>"
                                   f"{r['log_age']:.2f}</span>"
                                   if np.isfinite(r['log_age']) else "")
                                + f" · model #<span class='fit-num'>{r['model']}</span>"
                                f"</span></div>"
                                for r in results
                            )
                            fits_html = (
                                "<div class='fit-panel'>"
                                "<h5>Best-fit tracks  ·  χ² minimisation</h5>"
                                f"{rows}</div>"
                            )
            else:
                obs_msg = "Observations file loaded but no rows had valid axes."
        else:
            missing = []
            if xv is None:
                missing.append(ax_label(x_col))
            if yv is None:
                missing.append(ax_label(y_col))
            obs_msg = ("Observations uploaded but no matching columns for "
                       + ", ".join(missing) + ". "
                       + "Columns found: " + ", ".join(obs_df.columns.tolist()))
    elif obs_load_err:
        obs_msg = f"Could not parse observations CSV: {obs_load_err}"

    # Animation (#10) — built after overlays are added
    if animate:
        if n_tracks > 40:
            st.caption(
                f"Animation limited when tracks > 40 (have {n_tracks}). "
                "Narrow the mass range for smoother playback."
            )
        fig = add_age_animation(
            fig, df_plot, x_col, y_col,
            n_frames=40,
            frame_duration_ms=anim_frame_ms,
        )

    # The right-rail layout was set up earlier (controls live there);
    # render the plot inside `plot_col` we already created.
    detail_col = right_col  # alias so the detail-panel block below
                            # keeps reading naturally
    with plot_col:
        # Filename baked into the modebar download — encodes view + sets so
        # exported PNGs are self-describing without renaming.
        fname = f"cepheid_{preset.replace(' ', '').lower()}_{'_'.join(selected_sets)}"
        # Capture click events (#1)
        if click_mode and n_tracks <= 80 and not animate:
            event = st.plotly_chart(
                fig, use_container_width=True,
                key="single_plot",
                on_select="rerun",
                selection_mode=("points",),
                config=plotly_config(filename=fname),
            )
        else:
            event = None
            st.plotly_chart(
                fig, use_container_width=True,
                config=plotly_config(filename=fname),
            )

        if obs_msg:
            if obs_plotted:
                st.caption(obs_msg)
            else:
                st.warning(obs_msg)

        if fits_html:
            st.markdown(fits_html, unsafe_allow_html=True)

    # Detail panel — Material-UI Card if available, falls back to HTML.
    # Writes into `detail_slot` reserved at the TOP of the right rail
    # (above the click / animate / IS / scaffolding controls).
    if detail_col is not None:
        with detail_slot.container():
            clicked = None
            if event is not None and getattr(event, "selection", None):
                pts = event.selection.get("points", [])
                if pts:
                    clicked = pts[0]

            def _fmt_cd(v, n=4):
                try:
                    f = float(v)
                    if np.isnan(f): return "—"
                    if abs(f) >= 1e4 or (abs(f) < 1e-3 and f != 0):
                        return f"{f:.{n}e}"
                    return f"{f:.{n}g}"
                except Exception:
                    return str(v)

            # Common card sx — used in both empty and populated states.
            CARD_SX = {
                "background": (
                    "linear-gradient(180deg, "
                    "rgba(31,32,48,0.92) 0%, "
                    "rgba(31,32,48,0.62) 100%)"
                ),
                "backdropFilter": "blur(12px)",
                "WebkitBackdropFilter": "blur(12px)",
                "border": "1px solid #33344a",
                "borderLeft": "3px solid #8fc4b0",
                "borderRadius": "8px",
                "color": "#f1ede0",
                "boxShadow": "0 4px 18px rgba(0,0,0,0.36)",
                "overflow": "hidden",
            }

            if not _HAS_ELEMENTS:
                # Plain HTML fallback (kept short — full table only if MUI works)
                if clicked is None:
                    st.markdown(
                        '<div class="detail-panel">'
                        '<h5>Point detail</h5>'
                        '<p style="color:var(--text-muted);font-size:0.85rem;'
                        'margin:0">Click any point on a track to inspect it.</p>'
                        '</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div class="detail-panel"><h5>Track point</h5>'
                        '<p>Install <code>streamlit-elements</code> for the '
                        'full detail panel.</p></div>',
                        unsafe_allow_html=True,
                    )
            elif clicked is None:
                # Empty state — quiet placeholder card with a hint
                with elements("detail_panel_empty"):
                    with mui.Card(elevation=0, sx={**CARD_SX,
                                                   "padding": "1.4rem 1.3rem"}):
                        mui.Typography(
                            "Point detail",
                            sx={
                                "fontFamily": ("'Source Serif 4', "
                                               "'Source Serif Pro', serif"),
                                "fontSize": "1.05rem",
                                "fontWeight": 500,
                                "color": "#8fc4b0",
                                "letterSpacing": "-0.005em",
                                "marginBottom": "0.6rem",
                                "paddingBottom": "0.5rem",
                                "borderBottom": "1px dashed rgba(143,196,176,0.28)",
                            },
                        )
                        mui.Typography(
                            "Click any point on a track to inspect "
                            "it. The phase, mass, age, central abundances, "
                            "and surface diagnostics will appear here.",
                            sx={
                                "color": "#b5ad97",
                                "fontFamily": "Inter, sans-serif",
                                "fontSize": "0.92rem",
                                "lineHeight": 1.55,
                            },
                        )
            else:
                # Populated state — extract everything, then render an
                # MUI Card with a header (Phase chip on the right), a
                # divider, and a styled key/value list.
                cd = clicked.get("customdata", [])

                set_name = str(cd[0]) if len(cd) > 0 else "—"
                M_val    = _fmt_cd(cd[1], 3) if len(cd) > 1 else "—"
                Z_val    = _fmt_cd(cd[2], 4) if len(cd) > 2 else "—"
                Y_val    = _fmt_cd(cd[3], 4) if len(cd) > 3 else "—"
                age      = _fmt_cd(cd[4], 4) if len(cd) > 4 else "—"
                modnum   = _fmt_cd(cd[5], 1) if len(cd) > 5 else "—"
                lT       = _fmt_cd(cd[6], 4) if len(cd) > 6 else "—"
                lL       = _fmt_cd(cd[7], 4) if len(cd) > 7 else "—"
                h1       = _fmt_cd(cd[8], 3) if len(cd) > 8 else "—"
                he4      = _fmt_cd(cd[9], 3) if len(cd) > 9 else "—"
                lg       = _fmt_cd(cd[10], 3) if len(cd) > 10 else "—"
                lR       = _fmt_cd(cd[11], 3) if len(cd) > 11 else "—"
                phase    = str(cd[12]) if len(cd) > 12 else "—"

                feh = feh_for(float(cd[2])) if len(cd) > 2 else None
                feh_str = f"{feh:+.2f}" if feh is not None else "—"

                # Phase → chip color (warm for shell/late, cool for MS/loop)
                phase_chip_color = {
                    "Pre-MS":               "#9ab8e0",
                    "Main sequence":        "#8fc4b0",
                    "Crossing 1 · H-shell": "#e0b878",
                    "Blue loop · He-core":  "#d6a45c",
                    "Post He-burning":      "#d4716e",
                }.get(phase, "#b5ad97")

                rows = [
                    ("Set",          set_name),
                    ("Mass",         f"{M_val} M☉"),
                    ("Z",            Z_val),
                    ("Y",            Y_val),
                    ("[Fe/H]",       feh_str),
                    ("Age",          f"{age} yr"),
                    ("Model #",      modnum),
                    ("log Teff",     lT),
                    ("log L",        lL),
                    ("log g",        lg),
                    ("log R",        lR),
                    ("Xc (¹H)",      h1),
                    ("Yc (⁴He)",     he4),
                ]

                with elements("detail_panel_filled"):
                    with mui.Card(elevation=0, sx=CARD_SX):
                        with mui.Box(sx={
                            "display": "flex",
                            "alignItems": "center",
                            "justifyContent": "space-between",
                            "padding": "0.85rem 1.1rem 0.7rem 1.2rem",
                            "borderBottom": "1px solid rgba(74,81,98,0.55)",
                        }):
                            mui.Typography(
                                "Track point",
                                sx={
                                    "fontFamily": ("'Source Serif 4', "
                                                   "'Source Serif Pro', serif"),
                                    "fontSize": "1.1rem",
                                    "fontWeight": 500,
                                    "color": "#f1ede0",
                                    "letterSpacing": "-0.005em",
                                },
                            )
                            mui.Chip(
                                label=phase,
                                size="small",
                                sx={
                                    "color": phase_chip_color,
                                    "borderColor": f"{phase_chip_color}80",
                                    "backgroundColor":
                                        f"{phase_chip_color}1f",
                                    "border": f"1px solid {phase_chip_color}80",
                                    "fontFamily": "Inter, sans-serif",
                                    "fontSize": "0.74rem",
                                    "fontWeight": 500,
                                    "letterSpacing": "0.02em",
                                    "height": "22px",
                                },
                            )

                        with mui.Box(sx={
                            "padding": "0.55rem 1.1rem 0.85rem 1.2rem",
                        }):
                            for i, (k, v) in enumerate(rows):
                                with mui.Box(sx={
                                    "display": "flex",
                                    "alignItems": "baseline",
                                    "justifyContent": "space-between",
                                    "padding": "0.32rem 0",
                                    "borderBottom": (
                                        "1px dashed rgba(74,81,98,0.4)"
                                        if i < len(rows) - 1 else "none"
                                    ),
                                }):
                                    mui.Typography(
                                        k,
                                        sx={
                                            "color": "#b5ad97",
                                            "fontFamily": "Inter, sans-serif",
                                            "fontSize": "0.84rem",
                                            "letterSpacing": "0.005em",
                                        },
                                    )
                                    mui.Typography(
                                        v,
                                        sx={
                                            "color": "#f1ede0",
                                            "fontFamily": ("'JetBrains Mono', "
                                                           "ui-monospace, "
                                                           "monospace"),
                                            "fontSize": "0.86rem",
                                            "fontVariantNumeric":
                                                "tabular-nums",
                                        },
                                    )


# ========== MULTI-PANEL ====================================================
with tab_multi:
    st.markdown(
        """
        <div class="section-header">
          <span class="section-eyebrow">Synced panels</span>
          <h2 class="section-title">Multi-panel comparison</h2>
          <p class="section-tagline">2–4 synchronised panels — each its own axes, same filter.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    n_panels = st.segmented_control(
        "Panels", options=[2, 3, 4], default=2, key="n_multi",
    ) or 2

    view_defaults = [
        ("log_Teff",    "log_L",      True,  False),
        ("log_Teff",    "log_g",      True,  True),
        ("star_age",    "center_h1",  False, False),
        ("star_age",    "log_L",      False, False),
    ]

    cols_ui = st.columns(n_panels)
    for i, c in enumerate(cols_ui):
        with c:
            dflt_x, dflt_y, dflt_fx, dflt_fy = view_defaults[i % len(view_defaults)]
            col_x, col_y = st.columns(2)
            with col_x:
                x = st.selectbox(f"X {i+1}", NUMERIC_COLS,
                                 index=NUMERIC_COLS.index(pref(dflt_x, 0)),
                                 key=f"mx{i}")
            with col_y:
                y = st.selectbox(f"Y {i+1}", NUMERIC_COLS,
                                 index=NUMERIC_COLS.index(pref(dflt_y, 1)),
                                 key=f"my{i}")
            auto_fx = x in AUTO_FLIP_X or dflt_fx
            auto_fy = y in AUTO_FLIP_Y or dflt_fy
            f = build_figure(
                df_plot, x, y,
                color_by if i == 0 else
                    ("set" if len(selected_sets) > 1 else "mass"),
                auto_fx, auto_fy, False, False,
                height=450,
                show_labels=show_labels and (n_tracks <= 30),
                title="", show_legend=False,
                monochrome=monochrome,
                line_width=line_width, line_alpha=line_alpha,
            )
            st.plotly_chart(
                f, use_container_width=True,
                config=plotly_config(filename=f"cepheid_panel{i+1}_{x}_vs_{y}"),
            )


# ========== P–L DIAGRAM ====================================================
with tab_pl:
    st.markdown(
        """
        <div class="section-header">
          <span class="section-eyebrow">Tab 16 · pulsation</span>
          <h2 class="section-title">Period–Luminosity diagram</h2>
          <p class="section-tagline">
            log P (fundamental or first overtone) vs absolute magnitude or
            Wesenheit index, sampled along the instability-strip edges.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    is_data_pl = load_is_data()
    if "tab16_hot" not in is_data_pl or "tab16_cool" not in is_data_pl:
        st.warning("Tab 16 not found in `models/`. Place "
                   "`Tab16_online_hot_IS.dat` and `Tab16_online_cool_IS.dat` "
                   "alongside the other data files.")
    else:
        # Build a long-form dataframe stitching hot+cool, tagged by edge
        hot = is_data_pl["tab16_hot"]["df"].copy()
        hot["edge"] = "blue"
        cool = is_data_pl["tab16_cool"]["df"].copy()
        cool["edge"] = "red"
        df_pl_full = pd.concat([hot, cool], ignore_index=True)
        # Restrict to the sets currently selected in the sidebar so the
        # P–L respects the same filter as the rest of the app.
        df_pl = df_pl_full[df_pl_full["set"].isin(selected_sets)].copy()
        if df_pl.empty:
            st.info("No Tab 16 rows for the currently selected model set(s).")
        else:
            # Preset selector
            PL_PRESETS: dict[str, dict] = {
                "P–L (V)":         {"x": "log_P_F",  "y": "M_V"},
                "P–L (I)":         {"x": "log_P_F",  "y": "M_I"},
                "P–L (K)":         {"x": "log_P_F",  "y": "M_K"},
                "P–L Wesenheit VI":{"x": "log_P_F",  "y": "W_VI"},
                "P–L Wesenheit VK":{"x": "log_P_F",  "y": "W_VK"},
                "P–L (1O)":        {"x": "log_P_1O", "y": "M_V"},
                "P–C (V−I)":       {"x": "log_P_F",  "y": "V_minus_I_pl"},  # derived below
            }
            # Derived V−I for P–C plots inside the IS frame
            if "M_V" in df_pl.columns and "M_I" in df_pl.columns:
                df_pl["V_minus_I_pl"] = df_pl["M_V"] - df_pl["M_I"]
            if "M_V" in df_pl.columns and "M_K" in df_pl.columns:
                df_pl["V_minus_K_pl"] = df_pl["M_V"] - df_pl["M_K"]

            row1, row2, row3 = st.columns([3, 1.6, 1.6])
            with row1:
                pl_preset = st.segmented_control(
                    "Diagram", options=list(PL_PRESETS.keys()),
                    default="P–L Wesenheit VI", key="pl_preset",
                ) or "P–L Wesenheit VI"
            with row2:
                pl_color = st.segmented_control(
                    "Colour by", options=["Z", "mass", "set", "edge"],
                    default="Z", key="pl_color",
                ) or "Z"
            with row3:
                edge_filter = st.multiselect(
                    "Edges", options=["blue", "red"],
                    default=["blue", "red"], key="pl_edges",
                )
            cross_filter = st.multiselect(
                "Crossings", options=["1c", "2c", "3c"],
                default=["1c", "2c", "3c"], key="pl_cross",
                help="Tab 16 crossing labels.",
            )

            # Filter
            mask = (df_pl["edge"].isin(edge_filter)
                    & df_pl["crossing"].astype(str).isin(cross_filter))
            df_pl_f = df_pl[mask].copy()
            xcol = PL_PRESETS[pl_preset]["x"]
            ycol = PL_PRESETS[pl_preset]["y"]
            if xcol not in df_pl_f.columns or ycol not in df_pl_f.columns:
                st.info("Selected diagram needs columns not present "
                        "in the IS table.")
            else:
                df_pl_f = df_pl_f.dropna(subset=[xcol, ycol])
                if df_pl_f.empty:
                    st.info("No valid rows for this diagram.")
                else:
                    fig_pl = go.Figure()
                    if pl_color in {"Z", "mass"}:
                        cmap_pl = get_color_map(
                            sorted(df_pl_f[pl_color].dropna().unique()),
                            "continuous",
                        )
                        for v, gv in df_pl_f.groupby(pl_color, sort=True):
                            fig_pl.add_trace(go.Scattergl(
                                x=gv[xcol], y=gv[ycol],
                                mode="markers",
                                marker=dict(
                                    size=7, color=cmap_pl[v],
                                    line=dict(width=0.5,
                                              color="rgba(255,255,255,0.4)"),
                                    symbol=("diamond" if "edge" in gv
                                            and (gv["edge"] == "red").all()
                                            else "circle"),
                                ),
                                name=f"{pl_color}={v}",
                                showlegend=False,
                                hovertemplate=(
                                    f"M = %{{customdata[0]:.1f}} M☉<br>"
                                    f"Z = %{{customdata[1]:.4f}}<br>"
                                    f"crossing %{{customdata[2]}}<br>"
                                    f"{ax_label(xcol)}: %{{x:.4g}}<br>"
                                    f"{ax_label(ycol)}: %{{y:.4g}}"
                                    "<extra></extra>"
                                ),
                                customdata=np.column_stack([
                                    gv["mass"], gv["Z"], gv["crossing"],
                                ]),
                            ))
                        # Continuous colorbar
                        vals = sorted(df_pl_f[pl_color].dropna().unique())
                        fig_pl.add_trace(go.Scatter(
                            x=[None], y=[None], mode="markers",
                            marker=dict(
                                colorscale=NOCTURNE_SEQ,
                                cmin=float(min(vals)),
                                cmax=float(max(vals)),
                                showscale=True,
                                colorbar=dict(
                                    title=dict(text=ax_label(pl_color),
                                               side="right",
                                               font=dict(family=FONT_SERIF,
                                                         size=14,
                                                         color=THEME["text"])),
                                    thickness=10, len=0.7,
                                    x=1.02, xanchor="left",
                                    tickfont=dict(family=FONT_SANS,
                                                  size=11.5,
                                                  color=THEME["text_muted"]),
                                    outlinewidth=0,
                                ),
                                size=0.1,
                            ),
                            hoverinfo="skip", showlegend=False,
                        ))
                    else:
                        # Categorical (set or edge)
                        cats = sorted(df_pl_f[pl_color].dropna().unique())
                        cmap_q = {c: QUAL_PALETTE[i % len(QUAL_PALETTE)]
                                  for i, c in enumerate(cats)}
                        for v, gv in df_pl_f.groupby(pl_color, sort=False):
                            fig_pl.add_trace(go.Scattergl(
                                x=gv[xcol], y=gv[ycol],
                                mode="markers",
                                marker=dict(
                                    size=7, color=cmap_q[v],
                                    line=dict(width=0.5,
                                              color="rgba(255,255,255,0.4)"),
                                ),
                                name=str(v),
                                hovertemplate=(
                                    f"<b>{pl_color}={v}</b><br>"
                                    f"M = %{{customdata[0]:.1f}} M☉<br>"
                                    f"Z = %{{customdata[1]:.4f}}<br>"
                                    f"crossing %{{customdata[2]}}<br>"
                                    f"{ax_label(xcol)}: %{{x:.4g}}<br>"
                                    f"{ax_label(ycol)}: %{{y:.4g}}"
                                    "<extra></extra>"
                                ),
                                customdata=np.column_stack([
                                    gv["mass"], gv["Z"], gv["crossing"],
                                ]),
                            ))

                    axis_style = _axis_style()
                    fig_pl.update_layout(
                        template="plotly_dark",
                        height=620,
                        margin=dict(l=78, r=78, t=42, b=64),
                        font=dict(family=FONT_SANS, size=13,
                                  color=THEME["text"]),
                        paper_bgcolor=THEME["bg_plot"],
                        plot_bgcolor=THEME["bg_plot"],
                        xaxis=dict(title=dict(text=ax_label(xcol)),
                                   **axis_style),
                        yaxis=dict(title=dict(text=ax_label(ycol)),
                                   **axis_style),
                        showlegend=(pl_color in {"set", "edge"}),
                        legend=dict(
                            font=dict(size=12, color=THEME["text"]),
                            bgcolor="rgba(22,23,36,0.78)",
                            bordercolor="rgba(51,52,74,0.55)", borderwidth=1,
                            x=0.99, xanchor="right",
                            y=0.99, yanchor="top",
                        ),
                        hovermode="closest", dragmode="pan",
                    )
                    # Magnitudes/Wesenheit get reversed Y axis (brighter up)
                    if ycol in ("M_V", "M_I", "M_J", "M_H", "M_K",
                                "W_VI", "W_VK"):
                        fig_pl.update_yaxes(autorange="reversed")
                    st.plotly_chart(
                        fig_pl, use_container_width=True,
                        config=plotly_config(
                            filename=f"cepheid_PL_{xcol}_vs_{ycol}"),
                    )
                    st.caption(
                        f"{len(df_pl_f):,} IS samples · "
                        f"{df_pl_f['mass'].nunique()} masses · "
                        f"{df_pl_f['Z'].nunique()} compositions · "
                        f"edges = {', '.join(edge_filter)} · "
                        f"crossings = {', '.join(cross_filter)}."
                    )


# ========== CROSSINGS ======================================================
with tab_cross:
    st.markdown(
        """
        <div class="section-header">
          <span class="section-eyebrow">Tab 6 · period change</span>
          <h2 class="section-title">Crossing-time statistics</h2>
          <p class="section-tagline">
            log t<sub>cross</sub> and Ṗ/P across the three IS crossings,
            faceted by crossing number.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    is_data_cr = load_is_data()
    if "tab6_hot" not in is_data_cr and "tab6_cool" not in is_data_cr:
        st.warning("Tab 6 not found in `models/`. Place "
                   "`Tab6_online_hot_IS.dat` and `Tab6_online_cool_IS.dat` "
                   "alongside the other data files.")
    else:
        # Stitch hot+cool, then melt to long form
        frames_t6 = []
        for k in ("tab6_hot", "tab6_cool"):
            if k in is_data_cr:
                _df = is_data_cr[k]["df"].copy()
                _df["__edge"] = "blue" if k.endswith("hot") else "red"
                frames_t6.append(_df)
        df_t6 = pd.concat(frames_t6, ignore_index=True)
        df_t6 = df_t6[df_t6["set"].isin(selected_sets)].copy()
        if df_t6.empty:
            st.info("No Tab 6 rows for the currently selected set(s).")
        else:
            long_rows = []
            for c in (1, 2, 3):
                cols_c = {f"log_tcross_{c}": "log_tcross",
                          f"log_age_{c}":    "log_age",
                          f"Pdot_over_P_{c}": "Pdot_over_P"}
                if all(k in df_t6.columns for k in cols_c):
                    sub = df_t6[["set", "Z", "mass", "__edge"] +
                                list(cols_c.keys())].copy()
                    sub = sub.rename(columns=cols_c)
                    sub["crossing"] = c
                    long_rows.append(sub)
            df_t6_long = pd.concat(long_rows, ignore_index=True)
            df_t6_long = df_t6_long.dropna(
                subset=["log_tcross", "Pdot_over_P"], how="all")

            cr_y = st.segmented_control(
                "Quantity", options=["log_tcross", "Pdot_over_P"],
                default="log_tcross", key="cr_y",
                format_func=lambda c: {
                    "log_tcross":  "Crossing time  log t",
                    "Pdot_over_P": "Period change  Ṗ/P",
                }.get(c, c),
            ) or "log_tcross"
            cr_x = st.segmented_control(
                "x-axis", options=["mass", "log_age"],
                default="mass", key="cr_x",
            ) or "mass"

            df_cr = df_t6_long.dropna(subset=[cr_x, cr_y])
            if df_cr.empty:
                st.info("No valid rows after filtering.")
            else:
                # 3-panel facet by crossing number
                fig_cr = go.Figure()
                # Use one subplot per crossing
                from plotly.subplots import make_subplots
                fig_cr = make_subplots(
                    rows=1, cols=3, shared_yaxes=True,
                    horizontal_spacing=0.04,
                    subplot_titles=[f"Crossing {c}" for c in (1, 2, 3)],
                )
                vals_z = sorted(df_cr["Z"].dropna().unique())
                cmap_z = get_color_map(vals_z, "continuous")
                for col, c in enumerate((1, 2, 3), start=1):
                    sub = df_cr[df_cr["crossing"] == c]
                    for z, gz in sub.groupby("Z", sort=True):
                        fig_cr.add_trace(
                            go.Scattergl(
                                x=gz[cr_x], y=gz[cr_y],
                                mode="markers",
                                marker=dict(
                                    size=8, color=cmap_z[z],
                                    line=dict(width=0.5,
                                              color="rgba(255,255,255,0.4)"),
                                ),
                                name=f"Z={z}",
                                showlegend=False,
                                hovertemplate=(
                                    f"M = %{{x:.1f}} M☉<br>"
                                    if cr_x == "mass" else
                                    "log(age) = %{x:.3f}<br>"
                                ) + (
                                    f"{ax_label(cr_y)}: %{{y:.4g}}<br>"
                                    f"crossing {c}<br>Z = {z}"
                                    "<extra></extra>"
                                ),
                            ),
                            row=1, col=col,
                        )
                    # axis labels per panel
                    fig_cr.update_xaxes(title_text=ax_label(cr_x),
                                        row=1, col=col)
                fig_cr.update_yaxes(title_text=ax_label(cr_y), row=1, col=1)
                # Continuous colorbar via dummy trace
                if vals_z:
                    fig_cr.add_trace(go.Scatter(
                        x=[None], y=[None], mode="markers",
                        marker=dict(
                            colorscale=NOCTURNE_SEQ,
                            cmin=float(min(vals_z)),
                            cmax=float(max(vals_z)),
                            showscale=True,
                            colorbar=dict(
                                title=dict(text="Z", side="right",
                                           font=dict(family=FONT_SERIF,
                                                     size=14,
                                                     color=THEME["text"])),
                                thickness=10, len=0.7,
                                x=1.02, xanchor="left",
                                tickfont=dict(family=FONT_SANS,
                                              size=11.5,
                                              color=THEME["text_muted"]),
                                outlinewidth=0,
                            ),
                            size=0.1,
                        ),
                        hoverinfo="skip", showlegend=False,
                    ))

                # Common cosmetics — apply tick fonts to all axes
                for ax in fig_cr.layout:
                    if ax.startswith("xaxis") or ax.startswith("yaxis"):
                        fig_cr.layout[ax].update(
                            showline=True, linewidth=1.0,
                            linecolor=THEME["text_muted"], mirror=True,
                            ticks="inside", ticklen=5,
                            tickcolor=THEME["text_muted"],
                            gridcolor="rgba(245,246,248,0.06)",
                            zeroline=False,
                            tickfont=dict(family=FONT_SANS, size=12,
                                          color=THEME["text_muted"]),
                            title_font=dict(family=FONT_SERIF, size=14,
                                            color=THEME["text"]),
                        )
                fig_cr.update_layout(
                    template="plotly_dark",
                    height=520,
                    margin=dict(l=78, r=78, t=58, b=58),
                    font=dict(family=FONT_SANS, size=13,
                              color=THEME["text"]),
                    paper_bgcolor=THEME["bg_plot"],
                    plot_bgcolor=THEME["bg_plot"],
                    showlegend=False,
                    hovermode="closest",
                )
                # Subplot titles styling
                for ann in fig_cr.layout.annotations:
                    ann.update(font=dict(family=FONT_SERIF, size=14,
                                         color=THEME["text"]))
                st.plotly_chart(
                    fig_cr, use_container_width=True,
                    config=plotly_config(
                        filename=f"cepheid_crossings_{cr_y}_vs_{cr_x}"),
                )
                st.caption(
                    f"Tab 6 — {len(df_cr):,} (set, mass, Z, crossing) rows. "
                    "Crossing 1 = Hertzsprung gap; "
                    "2 = blue-loop blueward; 3 = blue-loop redward."
                )


# ========== DATA ===========================================================
with tab_data:
    st.markdown(
        f"""
        <div class="section-header">
          <span class="section-eyebrow">Filtered subset</span>
          <h2 class="section-title">Data inspector</h2>
          <p class="section-tagline">{len(df_f):,} rows · preview shows first 5,000 · download is full set.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(df_f.head(5000), height=480, use_container_width=True)
    st.download_button(
        "Download CSV",
        df_f.to_csv(index=False).encode(),
        file_name="cepheid_subset.csv",
        mime="text/csv",
    )


# ========== HELP ===========================================================
with tab_help:
    st.markdown(
        """
        <div class="section-header">
          <span class="section-eyebrow">Quick start</span>
          <h2 class="section-title">Documentation</h2>
          <p class="section-tagline">How to use it · what each preset means · how to cite.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("""
#### Quick start
1. Sidebar → pick a model set (default `O24`, the reference grid).
2. Choose mass range, step size, and composition. `[Fe/H]` shown next to Z.
3. **View** presets above the plot: *HRD*, *Kiel*, *Nuclear*, *Luminosity*,
   *CMD*, or *Custom*.
4. Toggle **Animate age** to play tracks growing from ZAMS.
5. Toggle **Click to inspect** to open a detail panel on any point.
6. **Instability strip** popover: draw the IS wedge from Tab 16.

#### Uploading observations
Drop a CSV in the sidebar to overlay observed stars as markers.
Flexible column-name detection — any of these work:

- Temperature: `log_Teff`, `logTeff`, `Teff`, `T_eff`
- Luminosity : `log_L`, `logL`, `L`
- Period     : `period`, `P`, `P_F`
- Name/ID    : `name`, `star`, `id`

If a direct axis column isn't present, the app tries to derive it
(e.g. `log_Teff = log10(Teff)`). A status line below the plot tells you
what was matched.

#### Instability strip
Uses **Tab 16** from the Zenodo dataset. Each row is a single IS point
with `log_Teff`, `log_L`, periods, and magnitudes.
- `b`/`r`/`m` = blue / red / midline edge.
- Hot-file and cool-file contain the two endpoints; the app fills the
  polygon between them.
- Crossings 1/2/3 = Hertzsprung gap / blue-loop blueward / blue-loop redward.

**Tab 6** carries crossing durations, entry ages, and Ṗ/P.

#### Model sets (Table 3)
**O-prefix:** tens digit = f<sub>core</sub>, units digit = f<sub>env</sub>.
**O24** (ref) = f<sub>cor</sub>=0.02, f<sub>env</sub>=0.04.
Variants: `_ML2/4/6` Reimers η=0.2/0.4/0.6; `_AB` NACRE ¹⁴N(p,γ);
`_AC` GS98 solar mix; `_AE` ΔY/ΔZ=2.0.
""")

    st.divider()
    st.markdown("#### Reference MESA inlist")
    st.caption(
        "Verbatim from the Zenodo archive — the inlist used to compute the "
        "reference grid. Shown here for the O24 set at "
        "M = 5.0 M☉, Z = 0.0060, Y = 0.2575. Other (M, Z, Y) combinations "
        "differ only in `initial_mass`, `new_z`, `new_y`, and `Zbase`. "
        "Variants `_ML*`, `_AB`, `_AC`, `_AE` adjust `Reimers_scaling_factor`, "
        "the nuclear reaction rates, opacity prefixes, or Y choice "
        "respectively."
    )

    # Read the inlist from disk so the help text always matches the
    # actual file shipped in models/. Falls back to a placeholder if the
    # file isn't there.
    _inlist_path = (MODELS_DIR /
                    "inlist_project_O24_5.0_0.0060_0.2575")
    if _inlist_path.exists():
        try:
            _inlist_text = _inlist_path.read_text(encoding="utf-8")
        except Exception as _e:
            _inlist_text = f"! could not read inlist: {_e}"
    else:
        _inlist_text = ("! Reference inlist not found. Place\n"
                        "!   inlist_project_O24_5.0_0.0060_0.2575\n"
                        "! in the models/ folder (it ships with the "
                        "Zenodo archive).")

    # A single code block whose content swaps between a short preview
    # (first ~22 lines — the complete &star_job namelist) and the full
    # inlist, with a "View full inlist" button below it. One code block
    # only — avoids the confusing two-stacked-boxes effect.
    _inlist_lines = _inlist_text.splitlines()
    _preview_n = 22
    _is_truncatable = len(_inlist_lines) > _preview_n

    show_full_inlist = st.session_state.get("inlist_show_full", False)

    if not _is_truncatable or show_full_inlist:
        _to_show = _inlist_text
    else:
        _to_show = "\n".join(_inlist_lines[:_preview_n])
        _to_show += (f"\n\n! … {len(_inlist_lines) - _preview_n} "
                     "more lines — click below to view in full …")
    st.code(_to_show, language="fortran")

    if _is_truncatable:
        _btn_label = ("Hide full inlist  ↑" if show_full_inlist
                      else f"View full inlist  ·  {len(_inlist_lines)} lines  ↓")
        if st.button(_btn_label, key="inlist_show_full_btn"):
            st.session_state["inlist_show_full"] = not show_full_inlist
            st.rerun()

    if _inlist_path.exists():
        st.download_button(
            "Download inlist",
            _inlist_text.encode("utf-8"),
            file_name=_inlist_path.name,
            mime="text/plain",
            help="Save a local copy of the reference inlist.",
        )

    st.divider()
    st.markdown("#### Citation")
    st.caption(
        "If you use this viewer or the Smolec et al. grid in a publication, "
        "please cite the paper:"
    )
    # Raw string keeps the LaTeX accent backslashes literal.
    bibtex = r"""@ARTICLE{2026arXiv260326111S,
       author = {{Smolec}, R. and {Zi{\'o}{\l}kowska}, O. and {Singh Rathour}, R. and {Hocd{\'e}}, V. and {Wielg{\'o}rski}, P.},
        title = "{Toward a Comprehensive Grid of Cepheid Models with MESA. III. Evolutionary and Pulsation Relations for Models with Core and Envelope Overshooting}",
      journal = {arXiv e-prints},
     keywords = {Solar and Stellar Astrophysics},
         year = 2026,
        month = mar,
          eid = {arXiv:2603.26111},
        pages = {arXiv:2603.26111},
          doi = {10.48550/arXiv.2603.26111},
archivePrefix = {arXiv},
       eprint = {2603.26111},
 primaryClass = {astro-ph.SR},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2026arXiv260326111S},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}"""
    st.code(bibtex, language="bibtex")


# ---------------------------------------------------------------------------
# Footer — multi-column credits / links / metadata.
# Sits at the very bottom of the page, mirrors the top horizon bar with a
# subtle gradient rule.
# ---------------------------------------------------------------------------
import datetime as _dt
_now_utc = _dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
footer_html = f"""
<div class="app-footer">
  <div>
    <div class="footer-brand">Cepheid MESA Grid Viewer</div>
    <div class="footer-tagline">
      Built with Streamlit + Plotly · WebGL evolutionary tracks across
      a published Cepheid model grid.
    </div>
  </div>
  <div>
    <div class="footer-heading">Paper</div>
    <a class="footer-link"
       href="https://doi.org/10.5281/zenodo.17987357" target="_blank">
      Smolec et&nbsp;al. 2026<span class="arrow">↗</span>
    </a>
    <a class="footer-link"
       href="https://doi.org/10.5281/zenodo.17987357" target="_blank">
      Zenodo dataset<span class="arrow">↗</span>
    </a>
    <a class="footer-link" href="#" onclick="return false;">
      MESA r24.08.1
    </a>
  </div>
  <div>
    <div class="footer-heading">Authors</div>
    <div class="footer-authors">
      R.&nbsp;Smolec · O.&nbsp;Ziółkowska · R.&nbsp;S.&nbsp;Rathour ·
      V.&nbsp;Hocdé · P.&nbsp;Wielgórski
    </div>
  </div>
  <div>
    <div class="footer-heading">Build</div>
    <div class="footer-meta">
      v{__version__}<br>
      page rendered<br>
      {_now_utc}<br>
      <span style="color:{THEME['accent']}">●</span> all systems nominal
    </div>
  </div>
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)
