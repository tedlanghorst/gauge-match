import os
from pathlib import Path
from datetime import datetime
import shutil

import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pandas as pd
import geopandas as gpd
import folium
from tqdm import tqdm
from flask import Flask, render_template, request, redirect, url_for



# --- Configuration & Setup ---

USE_MOCK_DATA = False  # <-- set to False when you want to load the real datasets


# Data paths
GAUGE_FILE = Path("./data/gauges.gpkg")
MATCH_FILE = Path("./data/matches.csv")
BACKUP_DIR = Path("./data/backups")
MERIT_DIR = Path("/Users/Ted/Documents/MERIT-BASINS/")

# Define key column names
GAUGE_ID_COL = "site_id"        
MERIT_ID_COL = "COMID" 
MERIT_AREA_COL = "uparea"  
GAUGE_AREA_COL = "area" 
GAUGE_DISCHARGE_COL = "mean_discharge" 

SEARCH_BUFFER_METERS = 2000 # Buffer around gauge to find candidates (in meters)
AREA_TOLERANCE_PERCENT = 100  #Filter candidates within this pct of gauge area
AREA_AUTO_MATCH_PERCENT = 10 # Automatically match reaches by area if less than this pct

BACKUP_INTERVAL = 50 

# 5. Initialize Flask App
app = Flask(__name__)

# Create backup directory if it doesn't exist
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

# --- Load Data (Run once on startup) ---

print("Loading datasets...")
try:
    gdf_gauges = gpd.read_file(GAUGE_FILE)

    merit_gdfs = []
    for b in tqdm(range(1,10), desc="Loading MERIT files"):
        filename = MERIT_DIR / f"riv_pfaf_{b}_MERIT_Hydro_v07_Basins_v01_bugfix1.shp"
        merit_gdfs.append(gpd.read_file(filename))
    print("Concatenating MERIT dataframes")
    gdf_merit = gpd.GeoDataFrame(pd.concat(merit_gdfs))

    # Ensure data is in a projected CRS for accurate buffering
    # We'll use a common global projection: EPSG:3857 (Web Mercator)
    if gdf_gauges.crs.to_epsg() != 3857:
        gdf_gauges = gdf_gauges.to_crs(epsg=3857)
    if gdf_merit.crs.to_epsg() != 3857:
        gdf_merit = gdf_merit.to_crs(epsg=3857)

    # Create a spatial index for fast querying
    print("Building spatial index for MERIT...")
    gdf_merit.sindex

    # Reproject copies to WGS84 (EPSG:4326) for Folium mapping
    gdf_gauges_wgs84 = gdf_gauges.to_crs(epsg=4326)
    gdf_merit_wgs84 = gdf_merit.to_crs(epsg=4326)
    
    print("Data loaded successfully.")

except Exception as e:
    print(f"Error loading data: {e}")
    print("Please ensure 'gauges.gpkg' and 'merit_hydro.gpkg' exist in /data")
    exit()

# --- Helper Functions ---

def create_backup():
    """Creates a timestamped backup of the matches CSV."""
    if not MATCH_FILE.is_file():
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = BACKUP_DIR / f"matches_backup_{timestamp}.csv"
    shutil.copy2(MATCH_FILE, backup_file)
    print(f"Created backup: {backup_file}")

def get_matched_data():
    """Reads the output CSV and returns the dataframe."""
    if not MATCH_FILE.is_file():
        return pd.DataFrame(columns=[GAUGE_ID_COL, MERIT_ID_COL, "comments"])
    try:
        df = pd.read_csv(MATCH_FILE)
        # Add comments column if it doesn't exist (for backward compatibility)
        if "comments" not in df.columns:
            df["comments"] = ""
        return df
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=[GAUGE_ID_COL, MERIT_ID_COL, "comments"])

def get_matched_ids():
    """Returns a set of gauge IDs already matched."""
    df_matches = get_matched_data()
    return set(df_matches[GAUGE_ID_COL].unique())

def get_progress_stats():
    """Returns statistics about matching progress."""
    total_gauges = len(gdf_gauges)
    matched_ids = get_matched_ids()
    matched_count = len(matched_ids)
    remaining = total_gauges - matched_count
    percent_complete = (matched_count / total_gauges * 100) if total_gauges > 0 else 0
    
    return {
        "total": total_gauges,
        "matched": matched_count,
        "remaining": remaining,
        "percent": round(percent_complete, 1)
    }

def get_gauge_by_id(gauge_id):
    """Returns a gauge row by its ID, or None if not found."""
    matches = gdf_gauges[gdf_gauges[GAUGE_ID_COL] == gauge_id]
    if len(matches) > 0:
        return matches.iloc[0]
    return None

def get_next_unmatched_gauge():
    """Finds the first gauge in the GeoDataFrame that is not in the matched set."""
    matched_ids = get_matched_ids()
    for _, gauge in gdf_gauges.iterrows():
        if gauge[GAUGE_ID_COL] not in matched_ids:
            return gauge
    return None # All gauges are matched

def get_previous_gauge():
    """Returns the most recently matched gauge ID."""
    df_matches = get_matched_data()
    if len(df_matches) > 0:
        return df_matches.iloc[-1][GAUGE_ID_COL]
    return None

def calculate_area_diff(gauge_area, merit_area):
    """Calculates the percentage difference between gauge and MERIT areas.
    Returns None if calculation cannot be performed."""
    try:
        gauge_area = float(gauge_area)
        merit_area = float(merit_area)
        if gauge_area > 0:
            diff = ((merit_area - gauge_area) / gauge_area)* 100
            return round(diff, 1)
    except (ValueError, TypeError, ZeroDivisionError):
        pass
    return None

def calculate_runoff(discharge_m3s, area_km2):
    """Calculates runoff in mm/yr from discharge (m3/s) and area (km2).
    Returns None if calculation cannot be performed."""
    try:
        discharge = float(discharge_m3s)
        area = float(area_km2)
        
        if area > 0 and discharge >= 0:
            # 1. Convert discharge from m3/s to m3/yr
            seconds_per_year = 60 * 60 * 24 * 365.25
            volume_m3_per_year = discharge * seconds_per_year
            # 2. Convert area from km2 to m2
            area_m2 = area * 1_000_000
            # 3. Calculate runoff depth in m/yr
            runoff_m_per_year = volume_m3_per_year / area_m2
            # 4. Convert to mm/yr and round
            return round(runoff_m_per_year * 1000, 1)
    except (ValueError, TypeError, ZeroDivisionError, AttributeError):
        pass
    return None

def get_candidates(gauge_geom, gauge_area=None):
    """Finds MERIT polygons that intersect a buffer around the gauge geometry, sorted by distance.
    If gauge_area is provided, filters candidates to within AREA_TOLERANCE_PERCENT.
    
    Returns:
        tuple: (candidates_gdf, skip_reason)
            - candidates_gdf: GeoDataFrame of matching candidates
            - skip_reason: None if candidates found, or string explaining why no candidates
    """
    buffer = gauge_geom.buffer(SEARCH_BUFFER_METERS)
    
    # Use spatial index to find possible matches
    possible_matches_idx = list(gdf_merit.sindex.intersection(buffer.bounds))
    
    if not possible_matches_idx:
        return gpd.GeoDataFrame(columns=gdf_merit.columns, crs=gdf_merit.crs), "NO_CANDIDATES"
        
    # Get the actual candidates and perform a precise intersection
    possible_matches = gdf_merit.iloc[possible_matches_idx]
    candidates = possible_matches[possible_matches.intersects(buffer)].copy()
    
    if candidates.empty:
        return candidates, "NO_CANDIDATES"
    
    # Filter by area if gauge area is available and valid
    if gauge_area is not None and gauge_area > 0:
        lower_bound = gauge_area * (1 - AREA_TOLERANCE_PERCENT / 100)
        upper_bound = gauge_area * (1 + AREA_TOLERANCE_PERCENT / 100)
        candidates = candidates[
            (candidates[MERIT_AREA_COL] >= lower_bound) & 
            (candidates[MERIT_AREA_COL] <= upper_bound)
        ].copy()
        
        if candidates.empty:
            return candidates, f"NO_CANDIDATES_BY_AREA"
    
    # Calculate distance from gauge point to each candidate
    if not candidates.empty:
        candidates['distance_to_gauge'] = candidates.geometry.distance(gauge_geom)
        candidates = candidates.sort_values('distance_to_gauge')
    
    return candidates, None


def create_map(gauge_wgs84, candidates_wgs84, candidate_list, all_merit_nearby_wgs84):
    """Creates a Folium map with the gauge, candidates, all nearby MERIT polygons, and layers."""

    center_lat = gauge_wgs84.geometry.y
    center_lon = gauge_wgs84.geometry.x
    m = folium.Map(location=[center_lat, center_lon], zoom_start=14)
    
    # Add base layers
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satellite",
        overlay=False,
        control=True
    ).add_to(m)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)

    # Add all nearby MERIT polygons as background (thin grey lines)
    if not all_merit_nearby_wgs84.empty:
        folium.GeoJson(
            all_merit_nearby_wgs84,
            style_function=lambda x: {
                "fillColor":  "#999999",
                "color": "#999999",
                "weight": 2,
                "fillOpacity": 0,
                "opacity": 0.5,
            },
            name="All MERIT Basins (nearby)"
        ).add_to(m)

    # Add gauge
    folium.Marker(
        [center_lat, center_lon],
        popup=f"Gauge: {gauge_wgs84[GAUGE_ID_COL]}",
        icon=folium.Icon(color="red", icon="tint"),
    ).add_to(m)

    if not candidates_wgs84.empty:
        # Map MERIT ID to color (removed label mapping)
        id_to_color = {c["id"]: c["color"] for c in candidate_list}

        def style_function(feature):
            cid = feature["properties"][MERIT_ID_COL]
            color = id_to_color.get(cid, "#3186cc")
            return {
                "fillColor": color,
                "color": color,
                "weight": 4, 
                "fillOpacity": 0.5,
            }

        def highlight_function(feature):
            cid = feature["properties"][MERIT_ID_COL]
            color = id_to_color.get(cid, "#FF0000")
            return {
                "fillColor": color,
                "color": "black",
                "weight": 5,  # Increased from 3 to 5
                "fillOpacity": 0.8,
            }

        # Tooltip showing multiple fields
        tooltip_fields = [MERIT_ID_COL, MERIT_AREA_COL]
        for f in ["slope", "elev_mean", "stream_order"]:  # optional extras if exist
            if f in candidates_wgs84.columns:
                tooltip_fields.append(f)

        tooltip = folium.features.GeoJsonTooltip(
            fields=tooltip_fields,
            aliases=[f"{f.replace('_', ' ').title()}:" for f in tooltip_fields],
            sticky=True
        )

        gjson = folium.GeoJson(
            candidates_wgs84,
            style_function=style_function,
            highlight_function=highlight_function,
            tooltip=tooltip,
            name="Candidate Matches"
        )

        gjson.add_to(m)

    folium.LayerControl().add_to(m)
    return m._repr_html_()


def auto_record_no_match(gauge_id, reason):
    """Automatically records a 'None' match with a reason."""
    new_match = {
        GAUGE_ID_COL: [gauge_id],
        MERIT_ID_COL: ["None"],
        "comments": [reason]
    }
    df_new = pd.DataFrame(new_match)
    
    # Check if file exists and has content
    file_exists = MATCH_FILE.is_file() and os.path.getsize(MATCH_FILE) > 0
    
    if file_exists:
        df_existing = get_matched_data()
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(MATCH_FILE, mode='w', header=True, index=False)
    else:
        df_new.to_csv(MATCH_FILE, mode='w', header=True, index=False)
    
    # Create backup periodically
    df_all = get_matched_data()
    if len(df_all) % BACKUP_INTERVAL == 0:
        create_backup()

def auto_record_match(gauge_id, merit_id, reason):
    """Automatically records a successful match with a reason."""
    new_match = {
        GAUGE_ID_COL: [gauge_id],
        MERIT_ID_COL: [merit_id], # <-- Use the provided merit_id
        "comments": [reason]
    }
    df_new = pd.DataFrame(new_match)
    
    # Check if file exists and has content
    file_exists = MATCH_FILE.is_file() and os.path.getsize(MATCH_FILE) > 0
    
    if file_exists:
        df_existing = get_matched_data()
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(MATCH_FILE, mode='w', header=True, index=False)
    else:
        df_new.to_csv(MATCH_FILE, mode='w', header=True, index=False)
    
    # Create backup periodically
    df_all = get_matched_data()
    if len(df_all) % BACKUP_INTERVAL == 0:
        create_backup()
    
    print(f"Auto-recorded match for {gauge_id}: {merit_id} ({reason})")


# --- Flask Routes ---

@app.route("/")
def index():
    """Main page: finds next gauge, gets candidates, and renders the UI."""
    
    # Check if we need to jump to a specific gauge or go back
    jump_to = request.args.get('jump_to')
    go_back = request.args.get('back')
    
    gauge = None
    existing_match = None
    candidates = None
    gauge_area = None
    
    if jump_to:
        gauge = get_gauge_by_id(jump_to)
        if gauge is None:
            return f"<h1>Gauge ID '{jump_to}' not found!</h1><a href='/'>Go back</a>"
        existing_match = None
    elif go_back:
        prev_id = get_previous_gauge()
        if prev_id:
            gauge = get_gauge_by_id(prev_id)
            # Get the existing match info
            df_matches = get_matched_data()
            existing_match = df_matches[df_matches[GAUGE_ID_COL] == prev_id].iloc[-1]
        else:
            return "<h1>No previous gauge to go back to!</h1><a href='/'>Go back</a>"
    else:
        # This is the "find next" path.
        # Loop until we find a gauge that needs manual review.
        while True:
            gauge = get_next_unmatched_gauge()
            existing_match = None
            
            if gauge is None:
                stats = get_progress_stats()
                return f"<h1>All {stats['total']} gauges have been matched!</h1>"

            # Get gauge area for filtering
            gauge_area = gauge.get(GAUGE_AREA_COL, None)
            
            # 2. Find potential candidates in the MERIT dataset
            candidates, skip_reason = get_candidates(gauge.geometry, gauge_area)
            
            # AUTO-SKIP: If no candidates found, record and move to next
            if candidates.empty and skip_reason:
                auto_record_no_match(gauge[GAUGE_ID_COL], f"Auto-skipped: {skip_reason}")
                continue  # <-- REPLACED REDIRECT

            # AUTO-MATCH: If we have gauge area we can attempt auto matching
            if gauge_area and gauge_area > 0 and not candidates.empty:
                candidates['area_diff_percent'] = candidates.apply(
                    lambda row: calculate_area_diff(gauge_area, row.get(MERIT_AREA_COL)), 
                    axis=1
                )
                valid_candidates = candidates.dropna(subset=['area_diff_percent'])
                valid_automatches = valid_candidates[valid_candidates['area_diff_percent'].abs()<AREA_AUTO_MATCH_PERCENT]
                
                if not valid_automatches.empty:
                    # Check topology:
                    candidate_comids = set(valid_automatches[MERIT_ID_COL])
                    candidate_nextdown_ids = set(valid_automatches['NextDownID'])
                    source_reaches = candidate_comids - candidate_nextdown_ids
                    num_sources = len(source_reaches)
                    if num_sources > 1:
                        # Pick between multiple valid automatches from different sources
                        break
                
                    best_match = valid_automatches.loc[valid_automatches['area_diff_percent'].abs().idxmin()]
                    best_diff = best_match['area_diff_percent']
                    best_id = best_match[MERIT_ID_COL]
                    
                    comment = f"Auto-match area diff: {best_diff:.1f}%"
                    auto_record_match(gauge[GAUGE_ID_COL], best_id, comment)
                    continue 

            # If we reached this point, the gauge was not auto-skipped
            # or auto-matched. It needs manual review.
            break # <-- EXIT THE WHILE LOOP

    # If we jumped or went back, candidates/area are not set yet.
    if jump_to or go_back:
        if gauge is None: # Should be handled above, but as a safeguard
             return "<h1>Gauge not found.</h1><a href='/'>Go back</a>"
        gauge_area = gauge.get(GAUGE_AREA_COL, None)
        candidates, _ = get_candidates(gauge.geometry, gauge_area) # We don't care about skip_reason here

    # --- From here, the original code logic continues ---
    # `gauge`, `gauge_area`, and `candidates` are now set correctly
    # regardless of which path (jump, back, or auto-match-loop) was taken.

    # 3. Get WGS84 versions for mapping
    gauge_wgs84 = gdf_gauges_wgs84[gdf_gauges_wgs84[GAUGE_ID_COL] == gauge[GAUGE_ID_COL]].iloc[0]
    
    # Get nearby MERIT polygons for context (larger buffer)
    context_buffer = gauge.geometry.buffer(SEARCH_BUFFER_METERS * 3)
    context_idx = list(gdf_merit.sindex.intersection(context_buffer.bounds))
    if context_idx:
        all_merit_nearby = gdf_merit.iloc[context_idx]
        all_merit_nearby_ids = all_merit_nearby[MERIT_ID_COL].tolist()
        all_merit_nearby_wgs84 = gdf_merit_wgs84[gdf_merit_wgs84[MERIT_ID_COL].isin(all_merit_nearby_ids)]
    else:
        all_merit_nearby_wgs84 = gpd.GeoDataFrame(columns=gdf_merit_wgs84.columns, crs=gdf_merit_wgs84.crs)
    
    if not candidates.empty:
         candidate_ids = candidates[MERIT_ID_COL].tolist()
         candidates_wgs84 = gdf_merit_wgs84[gdf_merit_wgs84[MERIT_ID_COL].isin(candidate_ids)]
         # Preserve the sort order from candidates
         candidates_wgs84 = candidates_wgs84.set_index(MERIT_ID_COL).loc[candidate_ids].reset_index()
    else:
        candidates_wgs84 = gpd.GeoDataFrame(columns=gdf_merit.columns, crs=gdf_merit_wgs84.crs)

    # 4. Prepare candidate data for the template
    colors = [mcolors.rgb2hex(cm.tab10(i % 10)) for i in range(len(candidates))]
    gauge_discharge = gauge.get(GAUGE_DISCHARGE_COL, None)

    candidate_list = []
    for i, (_, row) in enumerate(candidates.iterrows()):
        merit_area = row.get(MERIT_AREA_COL, 0)
        area_diff = calculate_area_diff(gauge_area, merit_area)
        runoff_mm_yr = calculate_runoff(gauge_discharge, merit_area)

        candidate_list.append({
            "id": row[MERIT_ID_COL],
            "area": round(merit_area, 0) if merit_area else None,
            "area_diff": area_diff,  # Keep as None or numeric
            "runoff": runoff_mm_yr,  # Keep as None or numeric
            "color": colors[i],
            "other_prop": row.get("other_merit_prop", "N/A")
        })

    # 5. Create Google Maps link
    lat = gauge_wgs84.geometry.y
    lon = gauge_wgs84.geometry.x
    gmaps_link = f"https://www.google.com/maps/@{lat},{lon},15z/data=!3m1!1e3"

    # 6. Create the Folium map
    map_html = create_map(gauge_wgs84, candidates_wgs84, candidate_list, all_merit_nearby_wgs84)

    # 7. Get progress stats
    progress = get_progress_stats()

    # 8. Get existing comment if going back
    existing_comment = ""
    existing_selection = None
    if go_back and existing_match is not None:
        comment_value = existing_match.get("comments", "")
        # Handle NaN values from pandas
        existing_comment = "" if pd.isna(comment_value) else str(comment_value)
        existing_selection = existing_match.get(MERIT_ID_COL)

    # 9. Render the template
    return render_template(
        "index.html",
        gauge=gauge.to_dict(),
        gauge_id=gauge[GAUGE_ID_COL],
        gauge_area=gauge_area if gauge_area is not None else "N/A",
        gauge_discharge=gauge_discharge if gauge_discharge is not None else "N/A",
        candidates=candidate_list,
        gmaps_link=gmaps_link,
        map_html=map_html,
        progress=progress,
        is_editing=go_back is not None,
        existing_comment=existing_comment,
        existing_selection=existing_selection,
        area_tolerance=AREA_TOLERANCE_PERCENT
    )

@app.route("/submit", methods=["POST"])
def submit():
    """Handles the form submission, saving the match to the CSV."""
    
    # 1. Get data from the form
    gauge_id = request.form.get("gauge_id")
    selected_id = request.form.get("selection") # This will be the MERIT ID or 'skip'/'none'
    comments = request.form.get("comments", "").strip()
    is_editing = request.form.get("is_editing") == "true"

    # 2. If editing, remove the old entry first
    if is_editing:
        df_matches = get_matched_data()
        # Remove all rows with this gauge_id
        df_matches = df_matches[df_matches[GAUGE_ID_COL] != gauge_id]
        # Write back without this gauge - ensure all columns are present
        if not df_matches.empty:
            df_matches.to_csv(MATCH_FILE, mode='w', header=True, index=False)
        else:
            # If empty after removing, create empty file with headers
            pd.DataFrame(columns=[GAUGE_ID_COL, MERIT_ID_COL, "comments"]).to_csv(
                MATCH_FILE, mode='w', header=True, index=False
            )

    # 3. Prepare the new row for the CSV
    new_match = {
        GAUGE_ID_COL: [gauge_id],
        MERIT_ID_COL: [selected_id],
        "comments": [comments]
    }
    df_new = pd.DataFrame(new_match)

    # 4. Write to CSV - append mode
    # Check if file exists and has content
    file_exists = MATCH_FILE.is_file() and os.path.getsize(MATCH_FILE) > 0
    
    if file_exists:
        # Read existing, ensure it has all columns, then append
        df_existing = get_matched_data()
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(MATCH_FILE, mode='w', header=True, index=False)
    else:
        # New file - write with header
        df_new.to_csv(MATCH_FILE, mode='w', header=True, index=False)

    # 5. Create backup periodically
    df_all = get_matched_data()
    if len(df_all) % BACKUP_INTERVAL == 0:
        create_backup()

    # 6. Redirect back to the index to load the next gauge
    return redirect(url_for("index"))

# --- Run the App ---

if __name__ == "__main__":
    print("Starting Flask app. Open http://127.0.0.1:5000 in your browser.")
    app.run(debug=True, port=5000)