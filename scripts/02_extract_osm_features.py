"""
02_extract_osm_features.py

Extracts city-level infrastructure features from india-latest.osm.pbf
for each city in city_coordinates.csv.

Output: data/intermediate/osm_feature_summary.csv
"""

import os
import warnings
import math
import traceback

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pyproj import Transformer

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OSM_PBF       = os.path.join(BASE_DIR, "data", "raw", "india-latest.osm.pbf")
CITIES_CSV    = os.path.join(BASE_DIR, "data", "raw", "city_coordinates.csv")
OUT_DIR       = os.path.join(BASE_DIR, "data", "intermediate")
OUT_CSV       = os.path.join(OUT_DIR, "osm_feature_summary.csv")

os.makedirs(OUT_DIR, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────

# Bounding box padding around city center used to read from PBF (degrees).
# ~0.15° ≈ 16 km — generous enough to cover the 5 km buffer + reprojection margin.
BBOX_PAD_DEG = 0.15

# Buffer radii
POLICE_RADIUS_M    = 5_000
TRANSPORT_RADIUS_M = 3_000
STREETLIGHT_RADIUS_M = 2_000
ROAD_RADIUS_M      = 5_000
LANDUSE_RADIUS_M   = 5_000
EDUCATION_RADIUS_M = 5_000

# India uses UTM zones 43N–46N.  We pick the zone per city at runtime.
def _utm_epsg(lon: float) -> int:
    """Return EPSG code for the UTM zone covering a given longitude."""
    zone = int((lon + 180) / 6) + 1
    return 32600 + zone   # northern hemisphere


def _project(gdf: gpd.GeoDataFrame, epsg: int) -> gpd.GeoDataFrame:
    return gdf.to_crs(epsg=epsg)


def _city_center_m(lat: float, lon: float, epsg: int) -> Point:
    """Return city center as a projected Point."""
    t = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    x, y = t.transform(lon, lat)
    return Point(x, y)


def _buffer_circle(center_m: Point, radius_m: float) -> object:
    return center_m.buffer(radius_m)


def _count_within(gdf: gpd.GeoDataFrame, buffer) -> int:
    if gdf is None or gdf.empty:
        return 0
    return int(gdf[gdf.geometry.within(buffer)].shape[0])


def _nearest_distance_km(gdf: gpd.GeoDataFrame, center_m: Point) -> float:
    """Return distance in km to nearest feature; NaN if none."""
    if gdf is None or gdf.empty:
        return float("nan")
    dists = gdf.geometry.distance(center_m)
    if dists.empty:
        return float("nan")
    return float(dists.min()) / 1_000.0


def _road_length_km(gdf: gpd.GeoDataFrame, buffer) -> float:
    if gdf is None or gdf.empty:
        return 0.0
    clipped = gdf[gdf.geometry.intersects(buffer)].copy()
    if clipped.empty:
        return 0.0
    clipped["geometry"] = clipped.geometry.intersection(buffer)
    return float(clipped.geometry.length.sum()) / 1_000.0


# ── OSM tag filters ───────────────────────────────────────────────────────────

POLICE_TAGS       = {"amenity": ["police"]}
BUS_STOP_TAGS     = {"highway": ["bus_stop"]}
BUS_STATION_TAGS  = {"amenity": ["bus_station"]}
RAILWAY_TAGS      = {"railway": ["station", "halt"]}
STOP_POS_TAGS     = {"public_transport": ["stop_position"]}
STREETLIGHT_TAGS  = {"highway": ["street_lamp"]}
EDUCATION_TAGS    = {"amenity": ["school", "college", "university"]}
LANDUSE_TAGS      = {"landuse": ["commercial", "residential", "industrial", "retail"]}


def _get_pois(osm, custom_filter: dict) -> gpd.GeoDataFrame | None:
    """
    Fetch POIs from pyrosm OSM object using a custom filter.
    Returns a GeoDataFrame (points) or None on failure.
    """
    try:
        gdf = osm.get_pois(custom_filter=custom_filter)
        if gdf is None or gdf.empty:
            return None
        # keep only point geometry, centroids for polygons
        gdf = gdf.copy()
        gdf["geometry"] = gdf.geometry.apply(
            lambda g: g.centroid if g is not None and g.geom_type != "Point" else g
        )
        gdf = gdf[gdf.geometry.notna()]
        return gdf
    except Exception:
        return None


def _get_landuse(osm) -> gpd.GeoDataFrame | None:
    try:
        gdf = osm.get_landuse()
        if gdf is None or gdf.empty:
            return None
        # centroid for distance/within checks
        gdf = gdf.copy()
        gdf["geometry"] = gdf.geometry.apply(
            lambda g: g.centroid if g is not None and g.geom_type != "Point" else g
        )
        return gdf
    except Exception:
        return None


def _get_network(osm) -> gpd.GeoDataFrame | None:
    try:
        net = osm.get_network(network_type="all")
        if net is None or net.empty:
            return None
        return net
    except Exception:
        return None


# ── Per-city extraction ───────────────────────────────────────────────────────

def extract_city(row: pd.Series, idx: int, total: int) -> dict:
    city   = row["city"]
    lat    = float(row["latitude"])
    lon    = float(row["longitude"])
    epsg   = _utm_epsg(lon)
    notes  = []

    print(f"  Processing {idx}/{total}: {city}")

    result = {
        "city": city,
        "latitude": lat,
        "longitude": lon,
        "police_station_count_5km": 0,
        "nearest_police_station_km": float("nan"),
        "public_transport_count_3km": 0,
        "bus_stop_count_3km": 0,
        "bus_station_count_3km": 0,
        "railway_station_count_3km": 0,
        "street_light_count_2km": 0,
        "road_length_km_5km": 0.0,
        "road_density_km_per_sqkm_5km": 0.0,
        "commercial_landuse_count_5km": 0,
        "residential_landuse_count_5km": 0,
        "industrial_landuse_count_5km": 0,
        "retail_landuse_count_5km": 0,
        "education_poi_count_5km": 0,
        "osm_extraction_notes": "",
    }

    try:
        import pyrosm

        # Build bounding box [minx, miny, maxx, maxy] = [W, S, E, N]
        # pyrosm requires a list (not a tuple)
        bbox = [
            lon - BBOX_PAD_DEG,
            lat - BBOX_PAD_DEG,
            lon + BBOX_PAD_DEG,
            lat + BBOX_PAD_DEG,
        ]

        osm = pyrosm.OSM(OSM_PBF, bounding_box=bbox)

        center_m = _city_center_m(lat, lon, epsg)

        buf_police     = _buffer_circle(center_m, POLICE_RADIUS_M)
        buf_transport  = _buffer_circle(center_m, TRANSPORT_RADIUS_M)
        buf_light      = _buffer_circle(center_m, STREETLIGHT_RADIUS_M)
        buf_road       = _buffer_circle(center_m, ROAD_RADIUS_M)
        buf_landuse    = _buffer_circle(center_m, LANDUSE_RADIUS_M)
        buf_education  = _buffer_circle(center_m, EDUCATION_RADIUS_M)

        # ── Police ──────────────────────────────────────────────────────────
        police_gdf = _get_pois(osm, POLICE_TAGS)
        if police_gdf is not None and not police_gdf.empty:
            police_proj = _project(police_gdf, epsg)
            result["police_station_count_5km"] = _count_within(police_proj, buf_police)
            result["nearest_police_station_km"] = _nearest_distance_km(police_proj, center_m)
        else:
            notes.append("no police data")

        # ── Public transport ─────────────────────────────────────────────────
        bus_stop_gdf = _get_pois(osm, BUS_STOP_TAGS)
        bus_stn_gdf  = _get_pois(osm, BUS_STATION_TAGS)
        rail_gdf     = _get_pois(osm, RAILWAY_TAGS)
        stop_pos_gdf = _get_pois(osm, STOP_POS_TAGS)

        bus_stop_count = bus_stn_count = rail_count = 0

        if bus_stop_gdf is not None and not bus_stop_gdf.empty:
            bus_stop_proj  = _project(bus_stop_gdf, epsg)
            bus_stop_count = _count_within(bus_stop_proj, buf_transport)

        if bus_stn_gdf is not None and not bus_stn_gdf.empty:
            bus_stn_proj  = _project(bus_stn_gdf, epsg)
            bus_stn_count = _count_within(bus_stn_proj, buf_transport)
        else:
            bus_stn_count = 0

        if rail_gdf is not None and not rail_gdf.empty:
            rail_proj  = _project(rail_gdf, epsg)
            rail_count = _count_within(rail_proj, buf_transport)

        # stop_position adds to total transport count
        stop_count = 0
        if stop_pos_gdf is not None and not stop_pos_gdf.empty:
            stop_proj  = _project(stop_pos_gdf, epsg)
            stop_count = _count_within(stop_proj, buf_transport)

        total_transport = bus_stop_count + bus_stn_count + rail_count + stop_count

        result["bus_stop_count_3km"]          = bus_stop_count
        result["bus_station_count_3km"]       = bus_stn_count
        result["railway_station_count_3km"]   = rail_count
        result["public_transport_count_3km"]  = total_transport

        if total_transport == 0:
            notes.append("no public transport data")

        # ── Street lights ────────────────────────────────────────────────────
        light_gdf = _get_pois(osm, STREETLIGHT_TAGS)
        if light_gdf is not None and not light_gdf.empty:
            light_proj = _project(light_gdf, epsg)
            result["street_light_count_2km"] = _count_within(light_proj, buf_light)
        else:
            notes.append("no street light data")

        # ── Roads ────────────────────────────────────────────────────────────
        road_gdf = _get_network(osm)
        if road_gdf is not None and not road_gdf.empty:
            road_proj   = _project(road_gdf, epsg)
            road_len_km = _road_length_km(road_proj, buf_road)
            area_sqkm   = (math.pi * (ROAD_RADIUS_M / 1_000) ** 2)
            result["road_length_km_5km"]             = round(road_len_km, 3)
            result["road_density_km_per_sqkm_5km"]   = round(road_len_km / area_sqkm, 4)
        else:
            notes.append("no road network data")

        # ── Landuse ──────────────────────────────────────────────────────────
        landuse_gdf = _get_landuse(osm)
        if landuse_gdf is not None and not landuse_gdf.empty:
            landuse_proj = _project(landuse_gdf, epsg)
            for kind in ["commercial", "residential", "industrial", "retail"]:
                subset = landuse_proj[landuse_proj.get("landuse", pd.Series(dtype=str)) == kind]
                result[f"{kind}_landuse_count_5km"] = _count_within(subset, buf_landuse)
        else:
            notes.append("no landuse data")

        # ── Education ────────────────────────────────────────────────────────
        edu_gdf = _get_pois(osm, EDUCATION_TAGS)
        if edu_gdf is not None and not edu_gdf.empty:
            edu_proj = _project(edu_gdf, epsg)
            result["education_poi_count_5km"] = _count_within(edu_proj, buf_education)
        else:
            notes.append("no education data")

    except Exception as exc:
        msg = f"EXTRACTION ERROR: {type(exc).__name__}: {exc}"
        print(f"    [ERROR] {city}: {msg}")
        traceback.print_exc()
        notes.append(msg)

    result["osm_extraction_notes"] = "; ".join(notes)
    return result


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("NariSafe — OSM Feature Extraction")
    print("=" * 60)

    cities = pd.read_csv(CITIES_CSV)
    total  = len(cities)
    print(f"Cities to process: {total}")
    print(f"Output: {OUT_CSV}")
    print()

    # Load partial results if a previous run was interrupted
    if os.path.exists(OUT_CSV):
        done_df   = pd.read_csv(OUT_CSV)
        done_set  = set(done_df["city"].tolist())
        results   = done_df.to_dict("records")
        print(f"Resuming — {len(done_set)} cities already done.")
    else:
        done_set = set()
        results  = []

    for i, row in cities.iterrows():
        city = row["city"]
        if city in done_set:
            print(f"  Skipping {i+1}/{total}: {city} (already done)")
            continue

        rec = extract_city(row, i + 1, total)
        results.append(rec)

        # Save after every city so a crash loses at most one city
        pd.DataFrame(results).to_csv(OUT_CSV, index=False)

    print()
    print("=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)

    final = pd.DataFrame(results)
    print(f"Rows saved : {len(final)}")
    print(f"Output file: {OUT_CSV}")

    missing_vals = final.isnull().sum()
    missing_vals = missing_vals[missing_vals > 0]
    if not missing_vals.empty:
        print("\nColumns with missing values:")
        print(missing_vals.to_string())
    else:
        print("\nNo missing values.")

    flagged = final[final["osm_extraction_notes"].str.strip() != ""]
    if not flagged.empty:
        print(f"\nCities with extraction notes ({len(flagged)}):")
        for _, r in flagged.iterrows():
            print(f"  {r['city']}: {r['osm_extraction_notes']}")

    print()
    print(final.to_string(index=False))


if __name__ == "__main__":
    main()
