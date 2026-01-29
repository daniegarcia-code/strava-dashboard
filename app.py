import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import numpy as np
from datetime import datetime, timedelta
import google.generativeai as genai

# --- PAGE CONFIG ---
st.set_page_config(page_title="Endurance Lab Pro", page_icon="🚴", layout="wide")

# --- 1. SETUP AUTHENTICATION ---
try:
    CLIENT_ID = st.secrets["strava_client_id"]
    CLIENT_SECRET = st.secrets["strava_client_secret"]
    if "gemini_api_key" in st.secrets:
        genai.configure(api_key=st.secrets["gemini_api_key"])
        GEMINI_AVAILABLE = True
    else:
        GEMINI_AVAILABLE = False
except (KeyError, FileNotFoundError):
    st.error("Secrets not found! Please check your .streamlit/secrets.toml file.")
    st.stop()

# REDIRECT_URI = "http://localhost:8501" # Uncomment for local testing
REDIRECT_URI = "https://strava-dashboard-f2xuhecncj4hh7tpmpgupx.streamlit.app" 

def get_auth_url():
    # UPDATED: Scope set to read_all for private activities
    return (
        f"https://www.strava.com/oauth/authorize?client_id={CLIENT_ID}"
        f"&response_type=code&redirect_uri={REDIRECT_URI}"
        f"&approval_prompt=force&scope=activity:read_all"
    )

def exchange_code_for_token(code):
    res = requests.post(
        "https://www.strava.com/oauth/token",
        data={
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "code": code,
            "grant_type": "authorization_code",
        },
    )
    return res.json()

# --- HELPER: POLYLINE DECODER ---
def decode_polyline(polyline_str):
    if not polyline_str: return []
    index, lat, lng = 0, 0, 0
    coordinates = []
    changes = {'latitude': 0, 'longitude': 0}
    while index < len(polyline_str):
        for unit in ['latitude', 'longitude']:
            shift, result = 0, 0
            while True:
                byte = ord(polyline_str[index]) - 63
                index += 1
                result |= (byte & 0x1f) << shift
                shift += 5
                if not byte >= 0x20: break
            if (result & 1): changes[unit] = ~(result >> 1)
            else: changes[unit] = (result >> 1)
        lat += changes['latitude']
        lng += changes['longitude']
        coordinates.append([lng / 100000.0, lat / 100000.0])
    return coordinates

# --- HELPER: TIME FORMATTER ---
def format_seconds(seconds):
    if not seconds or np.isnan(seconds): return "0s"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0: return f"{h}h {m}m"
    return f"{m}m {s}s"

# --- DATA FETCHING ---
@st.cache_data(ttl=300)
def fetch_activities(access_token, num_activities=500):
    url = "https://www.strava.com/api/v3/athlete/activities"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {"per_page": 200, "page": 1} 
    all_data = []
    while len(all_data) < num_activities:
        r = requests.get(url, headers=headers, params=params)
        if r.status_code == 200:
            batch = r.json()
            if not batch: break
            all_data.extend(batch)
            params['page'] += 1
        else: break
    return all_data[:num_activities]

@st.cache_data(ttl=300)
def fetch_streams(access_token, activity_id):
    url = f"https://www.strava.com/api/v3/activities/{activity_id}/streams"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {"keys": "time,watts,heartrate,velocity_smooth,cadence,grade_smooth", "key_by_type": "true"}
    r = requests.get(url, headers=headers, params=params)
    if r.status_code == 200: return r.json()
    return None

# --- METRIC CALCULATION ---
def calculate_training_load(df, ftp):
    if ftp <= 0: ftp = 200 
    
    # Ensure columns exist
    for col in ['average_watts', 'average_heartrate', 'average_speed', 'moving_time', 'elapsed_time', 'total_elevation_gain']:
        if col not in df.columns:
            df[col] = 0.0
            
    # Force numeric types to prevent math errors
    df['average_watts'] = pd.to_numeric(df['average_watts'], errors='coerce').fillna(0)
    df['average_heartrate'] = pd.to_numeric(df['average_heartrate'], errors='coerce').fillna(0)
    df['moving_time'] = pd.to_numeric(df['moving_time'], errors='coerce').fillna(0)
    df['elapsed_time'] = pd.to_numeric(df['elapsed_time'], errors='coerce').fillna(0)
    df['total_elevation_gain'] = pd.to_numeric(df['total_elevation_gain'], errors='coerce').fillna(0)
    
    # 1. Calculate Power-based TSS first
    df['IF'] = df['average_watts'] / ftp
    df['tss_score'] = (df['moving_time'] * df['average_watts'] * df['IF'] * 1.05) / (ftp * 3600) * 100
    
    # 2. Aggressive Fallback for Manual/No-Power rides
    mask_zero_tss = (df['tss_score'].fillna(0) <= 0.1) & (df['moving_time'] > 0)
    df.loc[mask_zero_tss, 'tss_score'] = (df.loc[mask_zero_tss, 'moving_time'] / 3600) * 60
    
    # 3. Final cleanup of TSS column
    df['tss_score'] = df['tss_score'].fillna(0)

    df['efficiency_factor'] = df['average_watts'] / df['average_heartrate']
    
    if 'weighted_average_watts' in df.columns:
        df['variability_index'] = df['weighted_average_watts'] / df['average_watts']
    else:
        df['variability_index'] = np.nan

    if 'average_speed' in df.columns:
        df['average_speed_mph'] = df['average_speed'] * 2.23694
    else:
        df['average_speed_mph'] = 0

    df['kilojoules'] = df.get('kilojoules', pd.Series([0]*len(df))).fillna(0)
    
    if 'device_name' not in df.columns:
        df['device_name'] = "Unknown" 
    else:
        df['device_name'] = df['device_name'].fillna("Unknown")
    
    return df

# --- DEDUPLICATION LOGIC ---
def deduplicate_rides(df):
    if df.empty: return df
    
    # Initialize link column
    df['hr_source_id'] = None
    
    df = df.sort_values('start_date_local')
    drop_indices = []
    
    indices = df.index.tolist()
    
    for i in range(len(indices) - 1):
        idx_curr = indices[i]
        idx_next = indices[i+1]
        
        row_curr = df.loc[idx_curr]
        row_next = df.loc[idx_next]
        
        # Check overlapping time (15 mins)
        time_diff = abs((row_next['start_date_local'] - row_curr['start_date_local']).total_seconds())
        
        if time_diff < 900: 
            dev1 = str(row_curr['device_name']).lower()
            dev2 = str(row_next['device_name']).lower()
            
            wahoo_idx = None
            apple_idx = None
            
            # Identify primary vs secondary
            if 'wahoo' in dev1 and 'apple' in dev2:
                wahoo_idx = idx_curr
                apple_idx = idx_next
            elif 'apple' in dev1 and 'wahoo' in dev2:
                apple_idx = idx_curr
                wahoo_idx = idx_next
            
            if wahoo_idx is not None and apple_idx is not None:
                # 1. Update Summary Stats
                apple_hr = df.loc[apple_idx, 'average_heartrate']
                if pd.notnull(apple_hr) and apple_hr > 0:
                    df.loc[wahoo_idx, 'average_heartrate'] = apple_hr
                    # Recalculate EF
                    watts = df.loc[wahoo_idx, 'average_watts']
                    if watts > 0:
                        df.loc[wahoo_idx, 'efficiency_factor'] = watts / apple_hr
                
                # 2. Store Source ID for Stream Fetching
                df.loc[wahoo_idx, 'hr_source_id'] = df.loc[apple_idx, 'id']
                
                # 3. Mark for Deletion
                drop_indices.append(apple_idx)
    
    return df.drop(drop_indices)

def calculate_pmc(df):
    df = df.sort_values('start_date_local', ascending=True)
    
    # Robust Date Processing
    df['start_date_local'] = pd.to_datetime(df['start_date_local'], utc=True, errors='coerce')
    df = df.dropna(subset=['start_date_local'])
    
    if df.empty:
        return pd.DataFrame({'date': [], 'CTL': [], 'ATL': [], 'TSB': []})

    # Normalize to naive date objects for grouping
    df['date_clean'] = df['start_date_local'].dt.date
    df['tss_score'] = pd.to_numeric(df['tss_score'], errors='coerce').fillna(0)

    daily_tss = df.groupby('date_clean')['tss_score'].sum()
    
    if not daily_tss.empty:
        idx = pd.date_range(start=daily_tss.index.min(), end=daily_tss.index.max(), freq='D')
        daily_tss = daily_tss.reindex(idx, fill_value=0)
    
    ctl = daily_tss.ewm(span=42, adjust=False).mean()
    atl = daily_tss.ewm(span=7, adjust=False).mean()
    tsb = ctl - atl
    
    pmc_df = pd.DataFrame({'CTL': ctl, 'ATL': atl, 'TSB': tsb})
    # Reset index to make 'date' a column
    pmc_df = pmc_df.reset_index().rename(columns={'index': 'date'})
    # FIX: Ensure PMC date column is timezone-naive to match st.date_input
    pmc_df['date'] = pmc_df['date'].dt.tz_localize(None)
    
    return pmc_df

def calculate_advanced_metrics(streams, ftp, max_hr):
    metrics = {}
    
    if 'watts' in streams:
        watts = pd.Series(streams['watts']['data'])
        metrics['np'] = np.power(watts.rolling(30).mean().pow(4).mean(), 0.25)
        metrics['avg_pwr'] = watts.mean()
        metrics['vi'] = metrics['np'] / metrics['avg_pwr'] if metrics['avg_pwr'] > 0 else 1.0
        metrics['if'] = metrics['np'] / ftp
        
        # Extended Power Curve
        std_durations = [1, 5, 15, 30, 60, 180, 300, 600, 1200, 1800, 2700, 3600, 5400, 7200, 10800, 14400, 18000]
        metrics['pdc'] = {}
        for d in std_durations:
            if d >= 3600: label = f"{d/3600:.1f}h"
            elif d >= 60: label = f"{int(d/60)}m"
            else: label = f"{d}s"
            
            if len(watts) > d:
                metrics['pdc'][label] = watts.rolling(window=d).mean().max()
            else:
                pass 
            
        zones = [0, 0.55*ftp, 0.75*ftp, 0.90*ftp, 1.05*ftp, 1.20*ftp, 5000]
        labels = ["Z1 Active Recovery", "Z2 Endurance", "Z3 Tempo", "Z4 Threshold", "Z5 VO2Max", "Z6 Anaerobic"]
        watts_cut = pd.cut(watts, bins=zones, labels=labels)
        metrics['zone_dist'] = watts_cut.value_counts(sort=False)
    else:
        metrics['np'] = None

    if 'heartrate' in streams:
        hr = pd.Series(streams['heartrate']['data'])
        metrics['avg_hr'] = hr.mean()
        metrics['max_hr'] = hr.max()
        
        if max_hr > 0:
            hr_zones = [0, 0.60*max_hr, 0.70*max_hr, 0.80*max_hr, 0.90*max_hr, 300]
            hr_labels = ["Z1 Recovery", "Z2 Endurance", "Z3 Tempo", "Z4 Threshold", "Z5 Anaerobic"]
            hr_cut = pd.cut(hr, bins=hr_zones, labels=hr_labels)
            total_seconds = len(hr)
            if total_seconds > 0:
                metrics['hr_zone_dist'] = (hr_cut.value_counts(sort=False) / total_seconds) * 100
        
        if 'watts' in streams:
            half = len(watts) // 2
            if half > 0:
                p1, h1 = watts.iloc[:half].mean(), hr.iloc[:half].mean()
                p2, h2 = watts.iloc[half:].mean(), hr.iloc[half:].mean()
                if h1 > 0 and h2 > 0:
                    r1, r2 = p1/h1, p2/h2
                    metrics['decoupling'] = (r1 - r2) / r1 * 100
                else: metrics['decoupling'] = 0
            else: metrics['decoupling'] = 0
        else: metrics['decoupling'] = None
    else:
        metrics['avg_hr'] = None

    if 'velocity_smooth' in streams:
        speed = pd.Series(streams['velocity_smooth']['data']) * 2.23694 
        metrics['avg_speed'] = speed.mean()
        metrics['max_speed'] = speed.max()
    else: metrics['avg_speed'] = None

    if 'cadence' in streams:
        cad = pd.Series(streams['cadence']['data'])
        metrics['avg_cadence'] = cad[cad > 0].mean()
    else: metrics['avg_cadence'] = None

    return metrics

def ask_gemini(metrics, streams, question):
    if not GEMINI_AVAILABLE: return "Gemini API Key not found."
    
    try:
        working_model_name = None
        all_models = [m.name for m in genai.list_models()]
        priorities = ['models/gemini-1.5-flash', 'models/gemini-pro', 'models/gemini-1.0-pro']
        
        for p in priorities:
            if p in all_models:
                working_model_name = p
                break
        
        if not working_model_name:
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    working_model_name = m.name
                    break
        
        if not working_model_name:
            return f"Error: No valid Gemini models found. Available: {all_models}"
            
        model = genai.GenerativeModel(working_model_name)
        
    except Exception as e:
        return f"Error detecting models: {e}"

    csv_context = ""
    try:
        raw_data = {}
        if streams:
            for key, val in streams.items():
                if 'data' in val:
                    raw_data[key] = val['data']
            
            if raw_data:
                # Trim to min length
                min_len = min(len(v) for v in raw_data.values())
                raw_data_synced = {k: v[:min_len] for k, v in raw_data.items()}
                csv_df = pd.DataFrame(raw_data_synced)
                csv_context = csv_df.to_csv(index=False)
    except Exception as e:
        csv_context = f"[Error processing raw data: {e}]"

    safe_get = lambda k: metrics.get(k) or 0
    np_val = f"{safe_get('np'):.0f}" if metrics.get('np') else "N/A"
    
    prompt = f"""
    Analyze this ride data:
    - NP: {np_val} W
    - IF: {safe_get('if'):.2f}
    - VI: {safe_get('vi'):.2f}
    - Decoupling: {safe_get('decoupling'):.1f}%
    - Avg HR: {safe_get('avg_hr'):.0f} bpm
    
    Full Data (CSV):
    {csv_context}
    
    User Question: "{question}"
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error from {working_model_name}: {e}"

# --- APP LOGIC ---
if "access_token" not in st.session_state:
    query_params = st.query_params
    if "code" in query_params:
        code = query_params["code"]
        token_data = exchange_code_for_token(code)
        if "access_token" in token_data:
