"""
01_check_osm_setup.py
Verifies that all required files and Python packages are present before extraction.
"""

import os
import sys

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OSM_PBF  = os.path.join(RAW_DIR, "india-latest.osm.pbf")
CITIES_CSV = os.path.join(RAW_DIR, "city_coordinates.csv")

# ── File checks ──────────────────────────────────────────────────────────────

print("=" * 60)
print("FILE CHECKS")
print("=" * 60)

if os.path.exists(OSM_PBF):
    size_gb = os.path.getsize(OSM_PBF) / (1024 ** 3)
    print(f"[OK]  india-latest.osm.pbf  ({size_gb:.2f} GB)")
else:
    print(f"[MISSING]  {OSM_PBF}")

if os.path.exists(CITIES_CSV):
    print(f"[OK]  city_coordinates.csv")
else:
    print(f"[MISSING]  {CITIES_CSV}")

# ── City CSV preview ─────────────────────────────────────────────────────────

print()
print("=" * 60)
print("CITY COORDINATES PREVIEW")
print("=" * 60)

try:
    import pandas as pd
    df = pd.read_csv(CITIES_CSV)
    print(f"Total cities: {len(df)}")
    print()
    print(df.head().to_string(index=False))
except Exception as e:
    print(f"Could not read CSV: {e}")

# ── Package checks ───────────────────────────────────────────────────────────

print()
print("=" * 60)
print("PACKAGE CHECKS")
print("=" * 60)

REQUIRED = [
    ("pandas",    "pandas"),
    ("numpy",     "numpy"),
    ("geopandas", "geopandas"),
    ("shapely",   "shapely"),
    ("pyproj",    "pyproj"),
    ("pyrosm",    "pyrosm"),
]

missing = []
for import_name, pip_name in REQUIRED:
    try:
        mod = __import__(import_name)
        ver = getattr(mod, "__version__", "?")
        print(f"[OK]  {import_name:<12}  {ver}")
    except ImportError:
        print(f"[MISSING]  {import_name}")
        missing.append(pip_name)

if missing:
    print()
    print("Install missing packages with:")
    print(f"  pip install {' '.join(missing)}")
    sys.exit(1)
else:
    print()
    print("All packages present. Ready to run 02_extract_osm_features.py")
