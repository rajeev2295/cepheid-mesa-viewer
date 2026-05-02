"""
Preprocess Cepheid evolutionary tracks from Zenodo tgz archives.

Walks every evoltracks_*.tgz in ./models/, reads each history.dat_<set>_M_Z_Y
inside, and writes one compact Parquet file per archive to ./data/.

Data format inside the archives (confirmed from the actual files):
    - Tab/whitespace-separated text
    - Single header row with 26 column names
    - No MESA-style multi-row metadata block
    - Columns: model_number, star_age, star_mass, log_Teff, log_L, log_R,
               log_g, log_cntr_P, log_cntr_Rho, log_cntr_T, center_mu,
               center_h1/he4/c12/n14/o16, surface_h1/he4/c12/n14/o16,
               abs_mag_V/I/J/H/K

Run:
    python preprocess.py                 # full resolution
    python preprocess.py --stride 5      # keep every 5th row (smaller files)
"""

from __future__ import annotations

import argparse
import os
import re
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "data"

# Filename pattern: history.dat_<SET>_<M.M>_<Z.ZZZZ>_<Y.YYYY>
# SET may contain underscores (e.g. O00_AB, O24_ML2). The last three
# fields are always numeric with a decimal point, so greedy .+ plus
# regex backtracking picks SET correctly.
FNAME_RE = re.compile(
    r"history\.dat_(?P<set>.+)_(?P<mass>\d+\.\d+)_(?P<z>\d+\.\d+)_(?P<y>\d+\.\d+)$"
)


def parse_track(fobj) -> pd.DataFrame | None:
    """Parse one history.dat_* file. Single header row, whitespace-separated."""
    try:
        df = pd.read_csv(fobj, sep=r"\s+", engine="c")
        if df.empty:
            return None
        return df
    except Exception as e:
        print(f"    parse error: {e}")
        return None


def parse_filename(name: str):
    base = os.path.basename(name)
    m = FNAME_RE.match(base)
    if not m:
        return None
    return (
        m.group("set"),
        float(m.group("mass")),
        float(m.group("z")),
        float(m.group("y")),
    )


def iter_tracks(tgz_path: Path):
    """Yield (set, M, Z, Y, DataFrame) for each history.dat_* in the archive."""
    with tarfile.open(tgz_path, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            meta = parse_filename(member.name)
            if meta is None:
                continue
            fobj = tar.extractfile(member)
            if fobj is None:
                continue
            df = parse_track(fobj)
            if df is None:
                continue
            yield (*meta, df)


def downcast(df: pd.DataFrame) -> pd.DataFrame:
    """Shrink memory/disk: float64 -> float32, int64 -> int32 when safe."""
    for c in df.select_dtypes("float64").columns:
        df[c] = df[c].astype("float32")
    for c in df.select_dtypes("int64").columns:
        col = df[c]
        if col.min() >= np.iinfo(np.int32).min and col.max() <= np.iinfo(np.int32).max:
            df[c] = col.astype("int32")
    return df


def process_archive(tgz_path: Path, stride: int = 1) -> pd.DataFrame | None:
    frames = []
    n_tracks = 0
    for s, M, Z, Y, df in iter_tracks(tgz_path):
        if stride > 1:
            df = df.iloc[::stride].copy()
        df["set"] = s
        df["mass"] = np.float32(M)
        df["Z"] = np.float32(Z)
        df["Y"] = np.float32(Y)
        frames.append(df)
        n_tracks += 1

    if not frames:
        return None

    big = pd.concat(frames, ignore_index=True)
    big = downcast(big)
    print(f"    tracks: {n_tracks:3d}   rows: {len(big):>10,}   cols: {big.shape[1]}")
    return big


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stride", type=int, default=1,
        help="Keep every Nth row per track (1 = all; try 5 for smaller files).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing Parquet files.",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    if not MODELS_DIR.exists():
        raise SystemExit(f"Missing folder: {MODELS_DIR}")

    tgz_files = sorted(MODELS_DIR.glob("evoltracks_*.tgz"))
    if not tgz_files:
        raise SystemExit(f"No evoltracks_*.tgz found in {MODELS_DIR}")

    print(f"Found {len(tgz_files)} archives in {MODELS_DIR}")
    print(f"Writing Parquet files to {OUTPUT_DIR}")
    if args.stride > 1:
        print(f"Downsampling: keeping every {args.stride}th row per track")
    print()

    total_mb = 0.0
    for tgz in tgz_files:
        set_name = tgz.stem.replace("evoltracks_", "")
        out_path = OUTPUT_DIR / f"{set_name}.parquet"

        if out_path.exists() and not args.force:
            print(f"[skip]  {out_path.name}  (exists; use --force to overwrite)")
            total_mb += out_path.stat().st_size / 1e6
            continue

        print(f"[read]  {tgz.name}")
        df = process_archive(tgz, stride=args.stride)
        if df is None:
            print("        (no tracks parsed)")
            continue

        df.to_parquet(out_path, compression="snappy", index=False)
        size_mb = out_path.stat().st_size / 1e6
        total_mb += size_mb
        print(f"[write] {out_path.name}  ({size_mb:.1f} MB)\n")

    print(f"\nDone. Total Parquet size: {total_mb:.1f} MB")


if __name__ == "__main__":
    main()
