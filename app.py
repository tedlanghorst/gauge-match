import re
from pathlib import Path
from datetime import datetime
import shutil

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pandas as pd
import geopandas as gpd
import folium
from tqdm import tqdm
from translate import Translator
from flask import Flask, render_template, request, redirect, url_for, jsonify, session


# --- Configuration & Setup ---
# Data paths
DATA_DIR = Path("./data")
BACKUP_DIR = DATA_DIR / "backups"
MERIT_DIR = Path("/Users/Ted/Documents/MERIT-BASINS/")

# Define key column names
GAUGE_ID_COL = "site_id"        
MERIT_ID_COL = "COMID" 
MERIT_AREA_COL = "uparea"  
GAUGE_AREA_COL = "area" 
GAUGE_DISCHARGE_COL = "mean_discharge" 
GAUGE_DISCHARGE_COUNT_COL = "count_discharge"

SEARCH_BUFFER_METERS = 2000
AREA_TOLERANCE_PERCENT = 100
AREA_AUTO_MATCH_PERCENT = 10

BACKUP_INTERVAL = 50 
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

translator = Translator(to_lang="en")

app = Flask(__name__)
app.secret_key = "your-secret-key-here-change-this"  # Required for sessions

# Global data storage
LOADED_DATA = {}

# --- Data Loading Functions ---

def get_available_gauge_files():
    """Returns list of available .gpkg files in data directory."""
    if not DATA_DIR.exists():
        return []
    gpkg_files = list(DATA_DIR.glob("*.gpkg"))
    return [f.stem for f in sorted(gpkg_files)]

def load_gauge_file(gauge_file_stem):
    """Loads a specific gauge file and its corresponding match CSV."""
    gauge_file = DATA_DIR / f"{gauge_file_stem}.gpkg"
    match_file = DATA_DIR / f"{gauge_file_stem}.csv"
    
    if not gauge_file.exists():
        raise FileNotFoundError(f"Gauge file not found: {gauge_file}")
    
    print(f"Loading gauge file: {gauge_file}")
    gdf_gauges = gpd.read_file(gauge_file)
    
    # Ensure projected CRS
    if gdf_gauges.crs.to_epsg() != 3857:
        gdf_gauges = gdf_gauges.to_crs(epsg=3857)
    
    gdf_gauges_wgs84 = gdf_gauges.to_crs(epsg=4326)
    
    return {
        'gdf_gauges': gdf_gauges,
        'gdf_gauges_wgs84': gdf_gauges_wgs84,
        'match_file': match_file,
        'gauge_file_stem': gauge_file_stem
    }

def load_merit_data():
    """Loads MERIT data (only once, shared across all gauge files)."""
    if 'gdf_merit' in LOADED_DATA:
        return
    
    merit_gdfs = []
    for b in tqdm(range(1, 10), desc="Loading MERIT files"):
        filename = MERIT_DIR / f"riv_pfaf_{b}_MERIT_Hydro_v07_Basins_v01_bugfix1.shp"
        if filename.exists():
            merit_gdfs.append(gpd.read_file(filename))
    
    print("Concatenating MERIT dataframes...")
    gdf_merit = gpd.GeoDataFrame(pd.concat(merit_gdfs, ignore_index=True))
    
    if gdf_merit.crs.to_epsg() != 3857:
        gdf_merit = gdf_merit.to_crs(epsg=3857)
    
    print("Building spatial index for MERIT...")
    gdf_merit.sindex
    
    gdf_merit_wgs84 = gdf_merit.to_crs(epsg=4326)
    
    LOADED_DATA['gdf_merit'] = gdf_merit
    LOADED_DATA['gdf_merit_wgs84'] = gdf_merit_wgs84
    print("MERIT data loaded successfully.")

def get_current_data():
    """Returns the currently selected gauge data from session."""
    gauge_file = session.get('current_gauge_file')
    if not gauge_file:
        return None # Force user to index to select a file
    
    # Load if not already loaded
    if gauge_file not in LOADED_DATA:
        try:
            LOADED_DATA[gauge_file] = load_gauge_file(gauge_file)
        except Exception as e:
            print(f"Failed to load {gauge_file}: {e}")
            session.pop('current_gauge_file', None) # Clear bad session var
            return None # Force user to index
            
    return LOADED_DATA.get(gauge_file)

# --- Helper Functions ---
def create_backup(source_file):
    """Creates a timestamped backup of the matches CSV."""
    if not source_file.is_file():
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = BACKUP_DIR / f"matches_backup_{timestamp}.csv"
    shutil.copy2(source_file, backup_file)
    print(f"Created backup: {backup_file}")


def get_matched_data(match_file):
    """
    Reads the match CSV. If it's in the old format, it migrates it to the
    new format (site_id, COMID, status, comments) in memory.
    """
    new_cols = [GAUGE_ID_COL, MERIT_ID_COL, "status", "comments"]
    if not match_file.is_file():
        return pd.DataFrame(columns=new_cols)

    try:
        df = pd.read_csv(match_file, dtype={GAUGE_ID_COL: str, MERIT_ID_COL: str, "comments":str})
        if df.empty:
            return pd.DataFrame(columns=new_cols)

        # --- MIGRATION LOGIC ---
        if "status" not in df.columns:
            print(f"Migrating old file: {match_file.name}")
            df_new = pd.DataFrame(columns=new_cols)
            df_new[GAUGE_ID_COL] = df[GAUGE_ID_COL].astype(str)
            df_new["comments"] = df.get("comments", "").fillna("")
            
            # 1. Handle old COMID values (NONE, SKIP, etc.)
            df_new[MERIT_ID_COL] = pd.to_numeric(df[MERIT_ID_COL], errors='coerce')
            
            # 2. Create status column based on old values
            df_new["status"] = "MATCHED" # Default
            df_new.loc[df[MERIT_ID_COL] == "None", "status"] = "NONE_SELECTED"
            df_new.loc[df[MERIT_ID_COL] == "SKIP", "status"] = "SKIPPED"
            df_new.loc[df[MERIT_ID_COL] == "NO_CANDIDATES", "status"] = "NO_CANDIDATES"
            df_new.loc[df[MERIT_ID_COL] == "REMOVED_DUPLICATE", "status"] = "REMOVED_DUPLICATE"
        
            df = df_new.copy()

        # --- END MIGRATION ---

        # Ensure all columns exist even if file was empty
        for col in new_cols:
            if col not in df.columns:
                df[col] = pd.NA if col == MERIT_ID_COL else ""
        
        # Ensure correct types for new format
        df[GAUGE_ID_COL] = df[GAUGE_ID_COL].astype(str)
        df[MERIT_ID_COL] = pd.to_numeric(df[MERIT_ID_COL], errors='coerce')
        df["status"] = df["status"].astype(str).fillna("MATCHED")
        df["comments"] = df["comments"].astype(str).fillna("")
        df["comments"] = df['comments'].str.replace('nan', '')

        return df

    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=new_cols)

def get_matched_ids(match_file):
    """Returns a set of gauge IDs that are *already in the match file*."""
    df_matches = get_matched_data(match_file)
    # Exclude the ids that are in the file but need review
    df_matches = df_matches[df_matches['status'] != 'NEEDS_REVIEW']
    return set(df_matches[GAUGE_ID_COL].unique())


def get_duplicate_reaches(match_file):
    """Returns reaches that have multiple gauges matched (but not MATCHED_DEDUPE_OK)."""
    df = get_matched_data(match_file)
    if df.empty:
        return pd.DataFrame(columns=df.columns)

    # We only care about active, valid matches for duplication
    valid_match_statuses = ["MATCHED"]
    df_valid = df[df["status"].isin(valid_match_statuses)].copy()
    
    # Drop any with null COMIDs (shouldn't happen, but safe)
    df_valid = df_valid.dropna(subset=[MERIT_ID_COL])
    if df_valid.empty:
        return pd.DataFrame(columns=df.columns)
            
    # Group by MERIT ID and count
    counts = df_valid.groupby(MERIT_ID_COL).size()
    duplicate_reaches = counts[counts > 1].index.tolist()
    
    return df_valid[df_valid[MERIT_ID_COL].isin(duplicate_reaches)].sort_values(MERIT_ID_COL)


def get_next_duplicate_reach(match_file):
    """Gets the first reach with duplicates that needs review."""
    duplicates = get_duplicate_reaches(match_file)
    if duplicates.empty:
        return None, []
    
    first_reach = duplicates.iloc[0][MERIT_ID_COL]
    gauges_for_reach = duplicates[duplicates[MERIT_ID_COL] == first_reach]
    return first_reach, gauges_for_reach

def get_progress_stats(match_file, gdf_gauges):
    """Returns statistics about matching progress."""
    total_gauges = len(gdf_gauges)
    # Get all IDs in the file, regardless of status
    matched_ids = get_matched_ids(match_file)
    matched_count = len(matched_ids)
    remaining = total_gauges - matched_count
    percent_complete = (matched_count / total_gauges * 100) if total_gauges > 0 else 0
    
    return {
        "total": total_gauges,
        "matched": matched_count,
        "remaining": remaining,
        "percent": round(percent_complete, 1)
    }

def get_dedup_progress_stats(match_file):
    """Returns statistics about deduplication progress."""
    duplicates = get_duplicate_reaches(match_file)
    if duplicates.empty:
        return {"total": 0, "remaining": 0, "percent": 100.0}
    
    unique_reaches = duplicates[MERIT_ID_COL].nunique()
    return {
        "total": unique_reaches,
        "remaining": unique_reaches, # This just shows total, not "progress"
        "percent": 0.0 # Progress isn't linear here
    }

def get_gauge_by_id(gauge_id, gdf_gauges):
    """Returns a gauge row by its ID, or None if not found."""
    # Ensure comparison is string-to-string
    matches = gdf_gauges[gdf_gauges[GAUGE_ID_COL].astype(str) == str(gauge_id)]
    if len(matches) > 0:
        return matches.iloc[0]
    return None

def get_next_unmatched_gauge(match_file, gdf_gauges):
    """
    Finds the next gauge to review.
    Priority 1: Any gauge with status 'NEEDS_REVIEW'.
    Priority 2: Any gauge not yet in the match file.
    """
    df_matches = get_matched_data(match_file)
    
    # Priority 1: Find gauges flagged for manual review
    if "status" in df_matches.columns:
        needs_review = df_matches[df_matches["status"] == "NEEDS_REVIEW"]
        if not needs_review.empty:
            gauge_id = needs_review.iloc[0][GAUGE_ID_COL]
            gauge = get_gauge_by_id(gauge_id, gdf_gauges)
            if gauge is not None:
                print(f"Loading gauge {gauge_id} (flagged for review).")
                return gauge, True # Return gauge and "is_flagged"
            else:
                print(f"Warning: Flagged gauge {gauge_id} not in source file. Skipping.")
                
    # Priority 2: Find first gauge not in the match file at all
    matched_ids = set(df_matches[GAUGE_ID_COL])
    for _, gauge in gdf_gauges.iterrows():
        if gauge[GAUGE_ID_COL] not in matched_ids:
            return gauge, False # Return gauge and "not_flagged"
            
    return None, False # No gauges left

def get_previous_gauge(match_file):
    """Returns the most recently matched gauge ID."""
    df_matches = get_matched_data(match_file)
    if len(df_matches) > 0:
        return df_matches.iloc[-1][GAUGE_ID_COL]
    return None

def calculate_area_diff(gauge_area, merit_area):
    """Calculates the percentage difference between areas."""
    try:
        gauge_area = float(gauge_area)
        merit_area = float(merit_area)
        mean_area = (gauge_area + merit_area) / 2
        if gauge_area > 0:
            diff = ((merit_area - gauge_area) / mean_area) * 100
            return round(diff, 1)
    except (ValueError, TypeError, ZeroDivisionError):
        pass
    return None

def calculate_runoff(discharge_m3s, area_km2):
    """Calculates runoff in mm/yr from discharge and area."""
    try:
        discharge = float(discharge_m3s)
        area = float(area_km2)
        
        if area > 0 and discharge >= 0:
            seconds_per_year = 60 * 60 * 24 * 365.25
            volume_m3_per_year = discharge * seconds_per_year
            area_m2 = area * 1_000_000
            runoff_m_per_year = volume_m3_per_year / area_m2
            return round(runoff_m_per_year * 1000, 1)
    except (ValueError, TypeError, ZeroDivisionError, AttributeError):
        pass
    return None


def get_candidates(gauge_geom, gauge_area=None):
    """Finds MERIT polygons that intersect a buffer around the gauge."""
    gdf_merit = LOADED_DATA['gdf_merit']
    
    buffer = gauge_geom.buffer(SEARCH_BUFFER_METERS)
    possible_matches_idx = list(gdf_merit.sindex.intersection(buffer.bounds))
    
    if not possible_matches_idx:
        return gpd.GeoDataFrame(columns=gdf_merit.columns, crs=gdf_merit.crs), "NO_CANDIDATES"
    
    possible_matches = gdf_merit.iloc[possible_matches_idx]
    candidates = possible_matches[possible_matches.intersects(buffer)].copy()
    
    if candidates.empty:
        return candidates, "NO_CANDIDATES"
    
    if not candidates.empty:
        candidates['distance_to_gauge'] = candidates.geometry.distance(gauge_geom)
        candidates = candidates.sort_values('distance_to_gauge')
    
    return candidates, None

def create_map(gauge_wgs84, candidates_wgs84, candidate_list, all_merit_nearby_wgs84):
    """Creates a Folium map with the gauge and candidates."""
    center_lat = gauge_wgs84.geometry.y
    center_lon = gauge_wgs84.geometry.x
    m = folium.Map(location=[center_lat, center_lon], zoom_start=14)
    
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satellite",
        overlay=False,
        control=True
    ).add_to(m)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)

    if not all_merit_nearby_wgs84.empty:
        folium.GeoJson(
            all_merit_nearby_wgs84,
            style_function=lambda x: {
                "fillColor": "#999999",
                "color": "#999999",
                "weight": 2,
                "fillOpacity": 0,
                "opacity": 0.5,
            },
            name="All MERIT Basins (nearby)"
        ).add_to(m)

    folium.Marker(
        [center_lat, center_lon],
        popup=f"Gauge: {gauge_wgs84[GAUGE_ID_COL]}",
        icon=folium.Icon(color="red", icon="tint"),
    ).add_to(m)

    if not candidates_wgs84.empty:
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
                "weight": 5,
                "fillOpacity": 0.8,
            }

        tooltip_fields = [MERIT_ID_COL, MERIT_AREA_COL]
        for f in ["slope", "elev_mean", "stream_order"]:
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

def create_dedup_map(reach_wgs84_df, gauges_wgs84, gauge_list, all_merit_nearby_wgs84):
    """Creates a map for deduplication showing one reach with multiple gauges."""
    # Center on reach centroid
    center_lat = reach_wgs84_df.geometry.iloc[0].centroid.y
    center_lon = reach_wgs84_df.geometry.iloc[0].centroid.x
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13)
    
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satellite",
        overlay=False,
        control=True
    ).add_to(m)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)

    if not all_merit_nearby_wgs84.empty:
        folium.GeoJson(
            all_merit_nearby_wgs84,
            style_function=lambda x: {
                "fillColor": "#999999",
                "color": "#999999",
                "weight": 2,
                "fillOpacity": 0,
                "opacity": 0.5,
            },
            name="All MERIT Basins (nearby)"
        ).add_to(m)

    # Highlight the reach in question
    folium.GeoJson(
        reach_wgs84_df,  
        style_function=lambda x: {
            "fillColor": "#4CAF50",
            "color": "#2E7D32",
            "weight": 4,
            "fillOpacity": 0.4,
        },
        name="Selected MERIT Reach"
    ).add_to(m)

    # Add gauges as markers
    if not gauges_wgs84.empty:
        id_to_color = {g["id"]: g["color"] for g in gauge_list}
        
        for _, gauge in gauges_wgs84.iterrows():
            gauge_id = gauge[GAUGE_ID_COL]
            color_hex = id_to_color.get(gauge_id, "#0000FF") # Default to blue hex
            
            # Use CircleMarker to allow arbitrary hex colors
            folium.CircleMarker(
                location=[gauge.geometry.y, gauge.geometry.x],
                radius=7,
                popup=f"Gauge: {gauge_id}<br>Name: {gauge.get('name', 'N/A')}",
                tooltip=f"Gauge: {gauge_id}",
                color="#000000", # Black border for visibility
                weight=1,
                fill=True,
                fillColor=color_hex,
                fillOpacity=0.9,
            ).add_to(m)

    folium.LayerControl().add_to(m)
    return m._repr_html_()

def _save_match_row(match_file, gauge_id, merit_id, status, comments = ""):
    """Internal helper to save a single row to the match file."""
    
    # Prepare new row
    new_row = {
        GAUGE_ID_COL: [str(gauge_id)],
        MERIT_ID_COL: [merit_id], # Should be numeric or pd.NA
        "status": [str(status)],
        "comments": [str(comments)]
    }
    df_new = pd.DataFrame(new_row)
    
    # Read existing data (which will be in the new, clean format)
    df_existing = get_matched_data(match_file)

    # Remove any old entry for this gauge_id
    df_existing = df_existing[df_existing[GAUGE_ID_COL] != str(gauge_id)]

    # Concat and save
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    
    # Re-order columns to be clean
    df_combined = df_combined[[GAUGE_ID_COL, MERIT_ID_COL, "status", "comments"]]
    
    # Save with float_format to avoid .0 on integer COMIDs
    df_combined.to_csv(match_file, index=False, float_format='%.0f')

    if len(df_combined) % BACKUP_INTERVAL == 0:
        create_backup(match_file)


# --- Flask Routes ---

@app.route("/", methods=["GET", "POST"])
def index():
    """Main landing page to select file and mode."""
    
    if request.method == "POST":
        gauge_file = request.form.get("gauge_file")
        mode = request.form.get("mode")

        if not gauge_file or not mode:
            # Add flash messaging here for a better user experience
            return redirect(url_for('index'))

        # Load data if not already loaded
        if gauge_file not in LOADED_DATA:
            try:
                LOADED_DATA[gauge_file] = load_gauge_file(gauge_file)
            except Exception as e:
                # Add flash message here: "Error loading file: {e}"
                print(f"Error loading file: {e}")
                return redirect(url_for('index')) # Stay on index page
        
        session['current_gauge_file'] = gauge_file
        session['current_mode'] = mode
        
        if mode == 'dedup':
            return redirect(url_for('dedup_index'))
        else: # Default to match
            return redirect(url_for('match_index'))

    # GET request
    available_files = get_available_gauge_files()
    if not available_files:
        return "<h1>Error: No .gpkg files found in data directory.</h1>"

    current_file = session.get('current_gauge_file')
    current_mode = session.get('current_mode', 'match') # Default to match
    
    return render_template("index.html",
                           available_files=available_files,
                           current_file=current_file,
                           current_mode=current_mode)

@app.route("/switch_mode/<string:new_mode>")
def switch_mode(new_mode):
    """Switches the mode in the session and redirects."""
    if 'current_gauge_file' not in session:
        # If no file is set, go back to index
        return redirect(url_for('index'))
        
    if new_mode == 'dedup':
        session['current_mode'] = 'dedup'
        return redirect(url_for('dedup_index'))
    elif new_mode == 'match':
        session['current_mode'] = 'match'
        return redirect(url_for('match_index'))
    else:
        # Invalid mode, go back to index
        return redirect(url_for('index'))

@app.route("/match")
def match_index():
    """Matching mode: finds next gauge and renders UI."""
    data = get_current_data()
    mode = session.get('current_mode')
    
    if not data or mode != 'match':
        return redirect(url_for('index'))
    
    current_file = data['gauge_file_stem']
    gdf_gauges = data['gdf_gauges']
    gdf_gauges_wgs84 = data['gdf_gauges_wgs84']
    match_file = data['match_file']
    gdf_merit = LOADED_DATA['gdf_merit']
    gdf_merit_wgs84 = LOADED_DATA['gdf_merit_wgs84']
    
    jump_to = request.args.get('jump_to')
    go_back = request.args.get('back')
    
    gauge = None
    existing_match = None
    candidates = None
    gauge_area = None
    is_flagged_for_review = False
    
    df_matches = get_matched_data(match_file) # Load once
    
    if jump_to:
        gauge = get_gauge_by_id(jump_to, gdf_gauges)
        if gauge is None:
            return f"<h1>Gauge ID '{jump_to}' not found!</h1><a href='/match'>Go back</a>"
    elif go_back:
        prev_id = get_previous_gauge(match_file)
        if prev_id:
            gauge = get_gauge_by_id(prev_id, gdf_gauges)
            existing_match_rows = df_matches[df_matches[GAUGE_ID_COL] == str(prev_id)]
            if not existing_match_rows.empty:
                existing_match = existing_match_rows.iloc[0]
        else:
            return "<h1>No previous gauge to go back to!</h1><a href='/match'>Go back</a>"
    else:
        # Auto-match loop
        while True:
            gauge, is_flagged_for_review = get_next_unmatched_gauge(match_file, gdf_gauges)
            
            if gauge is None:
                stats = get_progress_stats(match_file, gdf_gauges)
                return f"<h1>All {stats['total']} gauges have been matched!</h1><a href='/switch_mode/dedup'>Switch to deduplication mode</a>"

            if is_flagged_for_review:
                print(f"Gauge {gauge[GAUGE_ID_COL]} is flagged for review. Bypassing auto-match.")
                break # break manually review this gauge

            gauge_area = gauge.get(GAUGE_AREA_COL, None)
            candidates, skip_reason = get_candidates(gauge.geometry, gauge_area)
            
            if candidates.empty and skip_reason:
                _save_match_row(
                    match_file=match_file,
                    gauge_id=gauge[GAUGE_ID_COL],
                    merit_id=pd.NA,
                    status="NO_CANDIDATES",
                    comments=skip_reason
                )
                continue # to next gauge

            # If we have candidates, check if we can automatically match this gauge
            if gauge_area and gauge_area > 0 and not candidates.empty:
                candidates['area_diff_percent'] = candidates.apply(
                    lambda row: calculate_area_diff(gauge_area, row.get(MERIT_AREA_COL)), 
                    axis=1
                )
                valid_candidates = candidates.dropna(subset=['area_diff_percent'])
                valid_automatches = valid_candidates[valid_candidates['area_diff_percent'].abs() < AREA_AUTO_MATCH_PERCENT]
                
                if not valid_automatches.empty:
                    # Check that the closest gauge is also the best area match
                    if valid_automatches.idxmin('area_diff_percent') != 0:
                        break # manually review this gauge

                    # Check for confluences
                    candidate_comids = set(valid_automatches[MERIT_ID_COL])
                    candidate_nextdown_ids = set(valid_automatches['NextDownID'])
                    source_reaches = candidate_comids - candidate_nextdown_ids
                    if len(source_reaches) >= 1:
                        break # manually review this gauge

                    # Auto match
                    best_match = valid_automatches.loc[valid_automatches['area_diff_percent'].abs().idxmin()]
                    best_id = best_match[MERIT_ID_COL]
                    
                    _save_match_row(
                        match_file=match_file,
                        gauge_id=gauge[GAUGE_ID_COL],
                        merit_id=best_id,
                        status="AUTO-MATCH",
                    )
                    
                    continue # to next gauge
            
            # If auto-match logic fails or doesn't run, break to render
            break

    if jump_to or go_back:
        if gauge is None:
            return "<h1>Gauge not found.</h1><a href='/match'>Go back</a>"
        gauge_area = gauge.get(GAUGE_AREA_COL, None)
        candidates, _ = get_candidates(gauge.geometry, gauge_area)
    elif gauge is None: # Should be caught above, but as a safeguard
         return "<h1>Error: Could not find a gauge.</h1><a href='/'>Back home</a>"

    # Get candidates if not already populated (e.g., for flagged or back/jump)
    if candidates is None:
        gauge_area = gauge.get(GAUGE_AREA_COL, None)
        candidates, _ = get_candidates(gauge.geometry, gauge_area)

    gauge_wgs84 = gdf_gauges_wgs84[gdf_gauges_wgs84[GAUGE_ID_COL] == gauge[GAUGE_ID_COL]].iloc[0]
    
    context_buffer = gauge.geometry.buffer(SEARCH_BUFFER_METERS * 3)
    context_idx = list(gdf_merit.sindex.intersection(context_buffer.bounds))

    all_merit_nearby_wgs84 = gpd.GeoDataFrame(columns=gdf_merit_wgs84.columns, crs=gdf_merit_wgs84.crs)
    
    if context_idx:
        all_merit_nearby = gdf_merit.iloc[context_idx]
        all_merit_nearby_ids = all_merit_nearby[MERIT_ID_COL].tolist()
        all_merit_nearby_wgs84 = gdf_merit_wgs84[gdf_merit_wgs84[MERIT_ID_COL].isin(all_merit_nearby_ids)]
    
    if not candidates.empty:
        candidate_ids = candidates[MERIT_ID_COL].tolist()
        candidates_wgs84 = gdf_merit_wgs84[gdf_merit_wgs84[MERIT_ID_COL].isin(candidate_ids)]
        candidates_wgs84 = candidates_wgs84.set_index(MERIT_ID_COL).loc[candidate_ids].reset_index()
    else:
        candidates_wgs84 = gpd.GeoDataFrame(columns=gdf_merit.columns, crs=gdf_merit_wgs84.crs)

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
            "area_diff": area_diff,
            "runoff": runoff_mm_yr,
            "color": colors[i],
        })

    lat = gauge_wgs84.geometry.y
    lon = gauge_wgs84.geometry.x
    gmaps_link = f"https://www.google.com/maps/@{lat},{lon},15z/data=!3m1!1e3"

    map_html = create_map(gauge_wgs84, candidates_wgs84, candidate_list, all_merit_nearby_wgs84)

    progress = get_progress_stats(match_file, gdf_gauges)

    existing_comment = ""
    existing_selection = None
    if go_back and existing_match is not None:
        existing_comment = existing_match.get("comments", "")
        status = existing_match.get("status")
        
        if status in ["NONE_SELECTED", "NO_CANDIDATES"]:
            existing_selection = "NONE"
        elif status == "SKIPPED":
            existing_selection = "SKIP"
        elif status in ["MATCHED", "MATCHED_DEDUPE_OK"]:
            existing_selection = existing_match.get(MERIT_ID_COL)
            if pd.isna(existing_selection):
                existing_selection = None
            else:
                existing_selection = int(existing_selection)

    return render_template(
        "match.html",
        gauge=gauge.to_dict(),
        gauge_id=gauge[GAUGE_ID_COL],
        gauge_name=gauge.get('name', 'N/A'),
        gauge_area=gauge_area if gauge_area is not None else "N/A",
        gauge_discharge=gauge_discharge if gauge_discharge is not None else "N/A",
        candidates=candidate_list,
        gmaps_link=gmaps_link,
        map_html=map_html,
        progress=progress,
        is_editing=go_back is not None,
        is_flagged=is_flagged_for_review, # Pass this new flag
        existing_comment=existing_comment,
        existing_selection=existing_selection,
        current_file=current_file,
        mode='match'
    )


@app.route("/dedup")
def dedup_index():
    """Deduplication mode: review reaches with multiple gauges."""
    data = get_current_data()
    mode = session.get('current_mode')

    if not data or mode != 'dedup':
        return redirect(url_for('index'))

    current_file = data['gauge_file_stem']
    gdf_gauges = data['gdf_gauges']
    gdf_gauges_wgs84 = data['gdf_gauges_wgs84']
    match_file = data['match_file']
    gdf_merit = LOADED_DATA['gdf_merit']
    gdf_merit_wgs84 = LOADED_DATA['gdf_merit_wgs84']
    
    reach_id, gauges_df = get_next_duplicate_reach(match_file)
    
    if reach_id is None:
        return "<h1>No duplicate reaches to review!</h1><p>All reaches have been deduplicated or no duplicates exist.</p><a href='/switch_mode/match'>Back to matching mode</a>"
    
    reach_df = gdf_merit[gdf_merit[MERIT_ID_COL] == reach_id]
    # if reach_df.empty:
    #      print(f"Warning: Reach ID {reach_id} in CSV not found in MERIT data. Skipping.")
    #      # This is a data inconsistency. We should try to "fix" it by
    #      # flagging these gauges for review.
    #      df_all_matches = get_matched_data(match_file)
    #      gauge_ids_to_flag = gauges_df[GAUGE_ID_COL].tolist()
         
    #      for gid in gauge_ids_to_flag:
    #         _save_match_row(
    #             match_file=match_file,
    #             gauge_id=gid,
    #             merit_id=pd.NA,
    #             status="NEEDS_REVIEW",
    #             comments=f"Flagged: Original COMID {reach_id} not in source files."
    #         )
    #      return redirect(url_for("dedup_index")) # Reload the page

    reach = reach_df.iloc[0]
    reach_wgs84_df = gdf_merit_wgs84[gdf_merit_wgs84[MERIT_ID_COL] == reach_id]
    
    gauge_ids = gauges_df[GAUGE_ID_COL].tolist()
    gauges_for_reach = gdf_gauges[gdf_gauges[GAUGE_ID_COL].isin(gauge_ids)]
    gauges_for_reach_wgs84 = gdf_gauges_wgs84[gdf_gauges_wgs84[GAUGE_ID_COL].isin(gauge_ids)]
    
    colors = [mcolors.rgb2hex(cm.tab10(i % 10)) for i in range(len(gauges_for_reach))]
    gauge_list = []
    
    reach_area = reach.get(MERIT_AREA_COL, None)
    
    for i, (_, gauge) in enumerate(gauges_for_reach.iterrows()):
        gauge_id = gauge[GAUGE_ID_COL]
        gauge_area = gauge.get(GAUGE_AREA_COL, None)
        gauge_discharge = gauge.get(GAUGE_DISCHARGE_COL, None)
        gauge_discharge_count = gauge.get(GAUGE_DISCHARGE_COUNT_COL, None)
        min_date = gauge.get('min_date', 'N/A')
        max_date = gauge.get('max_date', 'N/A')
        
        runoff = calculate_runoff(gauge_discharge, reach_area) if gauge_discharge and reach_area else None
        
        existing_comment = gauges_df[gauges_df[GAUGE_ID_COL] == gauge_id]['comments'].iloc[0]
        
        gauge_list.append({
            "id": gauge_id,
            "name": gauge.get('name', 'N/A'),
            "area": round(gauge_area, 0) if gauge_area else "N/A",
            "discharge": round(gauge_discharge, 2) if gauge_discharge else "N/A",
            "discharge_count": gauge_discharge_count if gauge_discharge_count else "N/A",
            "min_date": min_date,
            "max_date": max_date,
            "runoff": runoff,
            "color": colors[i],
            "comment": existing_comment
        })
    
    context_buffer = reach.geometry.buffer(SEARCH_BUFFER_METERS * 3)
    context_idx = list(gdf_merit.sindex.intersection(context_buffer.bounds))
    if context_idx:
        all_merit_nearby = gdf_merit.iloc[context_idx]
        all_merit_nearby_ids = all_merit_nearby[MERIT_ID_COL].tolist()
        all_merit_nearby_wgs84 = gdf_merit_wgs84[gdf_merit_wgs84[MERIT_ID_COL].isin(all_merit_nearby_ids)]
    else:
        all_merit_nearby_wgs84 = gpd.GeoDataFrame(columns=gdf_merit_wgs84.columns, crs=gdf_merit_wgs84.crs)
    
    map_html = create_dedup_map(reach_wgs84_df, gauges_for_reach_wgs84, gauge_list, all_merit_nearby_wgs84)
    
    progress = get_dedup_progress_stats(match_file)
    
    lat = reach_wgs84_df.geometry.iloc[0].centroid.y
    lon = reach_wgs84_df.geometry.iloc[0].centroid.x
    gmaps_link = f"https://www.google.com/maps/@{lat},{lon},15z/data=!3m1!1e3"
    
    return render_template(
        "dedup.html",
        reach_id=int(reach_id),
        reach_area=round(reach_area, 0) if reach_area else "N/A",
        gauges=gauge_list,
        map_html=map_html,
        progress=progress,
        gmaps_link=gmaps_link,
        current_file=current_file,
        mode='dedup'
    )


@app.route("/submit", methods=["POST"])
def submit():
    """Handles match form submission."""
    data = get_current_data()
    if not data:
        return redirect(url_for('index'))
    
    match_file = data['match_file']
    
    gauge_id = request.form.get("gauge_id")
    selected_id_str = request.form.get("selection")
    comments = request.form.get("comments", "").strip()
    
    merit_id = pd.NA
    status = "SKIPPED" # Default
    
    if selected_id_str == "NONE":
        status = "NONE_SELECTED"
    elif selected_id_str == "SKIP":
        status = "SKIPPED"
    else:
        # It's a COMID
        merit_id = pd.to_numeric(selected_id_str, errors='coerce')
        if pd.isna(merit_id):
             # This shouldn't happen, but good to check
             return "<h1>Error: Invalid COMID submitted.</h1>"
        status = "MATCHED"

    _save_match_row(
        match_file=match_file,
        gauge_id=gauge_id,
        merit_id=merit_id,
        status=status,
        comments=comments
    )

    return redirect(url_for("match_index"))


@app.route("/submit_dedup", methods=["POST"])
def submit_dedup():
    """Handles deduplication form submission."""
    data = get_current_data()
    if not data:
        return redirect(url_for('index'))
    
    match_file = data['match_file']
    
    reach_id_str = request.form.get("reach_id")
    reach_id_num = pd.to_numeric(reach_id_str, errors='coerce')
    
    if pd.isna(reach_id_num):
        return "<h1>Error: Invalid reach_id</h1><a href='/'>Back home</a>"
    
    # Get the list of all gauge IDs that were on the page
    gauge_ids_on_page_str = request.form.get("gauge_ids_on_page", "")
    if not gauge_ids_on_page_str:
        return redirect(url_for("dedup_index"))
        
    gauge_ids_on_page = gauge_ids_on_page_str.split(',')
    
    # Get the original gauge data for comments
    df_matches = get_matched_data(match_file)
    merit_id_numeric_series = pd.to_numeric(df_matches[MERIT_ID_COL], errors='coerce')
    is_target_reach = (merit_id_numeric_series == reach_id_num)
    gauges_for_reach = df_matches[is_target_reach].copy()
    gauges_for_reach[GAUGE_ID_COL] = gauges_for_reach[GAUGE_ID_COL].astype(str)

    # We will save all rows, so first remove all old ones for this reach
    df_matches = df_matches[~is_target_reach]
    
    # Read the action for each gauge
    actions = {}
    gauges_to_keep_count = 0
    for gauge_id in gauge_ids_on_page:
        # Default to 'remove' if something is wrong
        action = request.form.get(f"action_{gauge_id}", "remove")
        actions[gauge_id] = action
        if action == "keep":
            gauges_to_keep_count += 1
        

    # Process each gauge based on its selected action
    for gauge_id, action in actions.items():

        if action == "keep":
            new_status = "MATCHED_DEDUPE_OK" if gauges_to_keep_count > 1 else "MATCHED"
            new_merit_id = reach_id_num

            old_gauge_match = gauges_for_reach[gauges_for_reach[GAUGE_ID_COL] == gauge_id]
            old_comment = str(old_gauge_match.iloc[0]['comments'])
            pattern = r"Auto-match area diff: [\d.]+%"
            new_comment = re.sub(pattern, "", old_comment).strip()
        
        elif action == "review":
            new_status = "NEEDS_REVIEW"
            new_merit_id = pd.NA

            new_comment = f"Flagged from deduplication of COMID {reach_id_str}"
            
        elif action == "remove":
            new_status = "REMOVED_DUPLICATE"
            new_merit_id = pd.NA
            
            new_comment = f"Removed from COMID {reach_id_str} during deduplication"

        new_row = pd.DataFrame({
            GAUGE_ID_COL: [gauge_id],
            MERIT_ID_COL: [new_merit_id],
            "status": [new_status],
            "comments": [new_comment]
        })
        df_matches = pd.concat([df_matches, new_row], ignore_index=True)

    # Save the updated dataframe
    df_matches = df_matches[[GAUGE_ID_COL, MERIT_ID_COL, "status", "comments"]]
    df_matches.to_csv(match_file, index=False, float_format='%.0f')
    
    if len(df_matches) % BACKUP_INTERVAL == 0:
        create_backup(match_file)
    
    return redirect(url_for("dedup_index"))

@app.route("/translate/<gauge_id>")
def translate_name(gauge_id):
    """Endpoint to translate gauge name asynchronously."""
    data = get_current_data()
    if not data:
        return jsonify({"error": "No gauge file loaded"}), 404
    
    try:
        gauge = get_gauge_by_id(gauge_id, data['gdf_gauges'])
        if gauge is None:
            return jsonify({"error": "Gauge not found"}), 404
        
        gauge_name = gauge.get('name', '')
        
        if not gauge_name or gauge_name == 'N/A':
            return jsonify({"translated": "N/A"})
        
        try:
            translated = translator.translate(gauge_name)
            return jsonify({"translated": translated})
        except Exception as e:
            print(f"Translation error for {gauge_id}: {e}")
            return jsonify({"translated": f"[Translation failed: {gauge_name}]"})
            
    except Exception as e:
        print(f"Error in translate endpoint: {e}")
        return jsonify({"error": str(e)}), 500


# --- Startup ---
if __name__ == "__main__":
    try:
        load_merit_data()
        print("\nStarting Flask app. Open http://127.0.0.1:5000 in your browser.")
        app.run(port=5000, debug=True)
    except Exception as e:
        print(f"Error loading data: {e}")
        exit()