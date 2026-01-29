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

# REDIRECT_URI = "https://strava-dashboard-f2xuhecncj4hh7tpmpgupx.streamlit.app" 
REDIRECT_URI = "https://strava-dashboard-f2xuhecncj4hh7tpmpgupx.streamlit.app" 

def get_auth_url():
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

    # 4. Calculate Efficiency Factor (Avg Power / Avg HR)
    df['efficiency_factor'] = df.apply(
        lambda row: row['average_watts'] / row['average_heartrate'] 
        if row['average_heartrate'] > 0 and row['average_watts'] > 0 else None, axis=1
    )
    
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
    # Ensure date processing is robust and NAIVE (no timezone)
    df = df.sort_values('start_date_local', ascending=True)
    
    # 1. Force conversion to datetime, handling errors
    df['start_date_local'] = pd.to_datetime(df['start_date_local'], errors='coerce')
    
    # 2. Drop invalid dates
    df = df.dropna(subset=['start_date_local'])
    
    if df.empty:
        return pd.DataFrame({'date': [], 'CTL': [], 'ATL': [], 'TSB': []})

    # 3. Create timezone-naive date column for grouping
    df['date_clean'] = df['start_date_local'].dt.date
    df['tss_score'] = pd.to_numeric(df['tss_score'], errors='coerce').fillna(0)

    # 4. Group
    daily_tss = df.groupby('date_clean')['tss_score'].sum()
    
    if not daily_tss.empty:
        idx = pd.date_range(start=daily_tss.index.min(), end=daily_tss.index.max(), freq='D')
        daily_tss = daily_tss.reindex(idx, fill_value=0)
    
    # 5. Calculate Metrics
    ctl = daily_tss.ewm(span=42, adjust=False).mean()
    atl = daily_tss.ewm(span=7, adjust=False).mean()
    tsb = ctl - atl
    
    pmc_df = pd.DataFrame({'CTL': ctl, 'ATL': atl, 'TSB': tsb})
    pmc_df = pmc_df.reset_index().rename(columns={'index': 'date'})
    
    # 6. Final safety: Ensure output date column is standard datetime64[ns]
    pmc_df['date'] = pd.to_datetime(pmc_df['date'])
    
    return pmc_df

def calculate_advanced_metrics(streams, ftp, max_hr):
    # Initialize ALL keys to None to prevent KeyErrors if streams are missing
    metrics = {
        'np': None, 'if': None, 'vi': None, 'avg_pwr': None,
        'avg_hr': None, 'max_hr': None, 'decoupling': None,
        'avg_speed': None, 'max_speed': None, 'avg_cadence': None,
        'zone_dist': None, 'hr_zone_dist': None, 'pdc': {}
    }
    
    if 'watts' in streams:
        watts = pd.Series(streams['watts']['data'])
        metrics['np'] = np.power(watts.rolling(30).mean().pow(4).mean(), 0.25)
        metrics['avg_pwr'] = watts.mean()
        metrics['vi'] = metrics['np'] / metrics['avg_pwr'] if metrics['avg_pwr'] > 0 else 1.0
        metrics['if'] = metrics['np'] / ftp
        
        # Extended Power Curve
        std_durations = [1, 5, 15, 30, 60, 180, 300, 600, 1200, 1800, 2700, 3600, 5400, 7200, 10800, 14400, 18000]
        for d in std_durations:
            if d >= 3600: label = f"{d/3600:.1f}h"
            elif d >= 60: label = f"{int(d/60)}m"
            else: label = f"{d}s"
            
            if len(watts) > d:
                metrics['pdc'][label] = watts.rolling(window=d).mean().max()
            
        zones = [0, 0.55*ftp, 0.75*ftp, 0.90*ftp, 1.05*ftp, 1.20*ftp, 5000]
        labels = ["Z1 Active Recovery", "Z2 Endurance", "Z3 Tempo", "Z4 Threshold", "Z5 VO2Max", "Z6 Anaerobic"]
        watts_cut = pd.cut(watts, bins=zones, labels=labels)
        metrics['zone_dist'] = watts_cut.value_counts(sort=False)

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
            watts = pd.Series(streams['watts']['data']) # Re-get watts safely
            half = len(watts) // 2
            if half > 0:
                p1, h1 = watts.iloc[:half].mean(), hr.iloc[:half].mean()
                p2, h2 = watts.iloc[half:].mean(), hr.iloc[half:].mean()
                if h1 > 0 and h2 > 0:
                    r1, r2 = p1/h1, p2/h2
                    metrics['decoupling'] = (r1 - r2) / r1 * 100
                else: metrics['decoupling'] = 0

    if 'velocity_smooth' in streams:
        speed = pd.Series(streams['velocity_smooth']['data']) * 2.23694 
        metrics['avg_speed'] = speed.mean()
        metrics['max_speed'] = speed.max()

    if 'cadence' in streams:
        cad = pd.Series(streams['cadence']['data'])
        metrics['avg_cadence'] = cad[cad > 0].mean()

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
            st.session_state["access_token"] = token_data["access_token"]
            st.session_state["athlete_name"] = token_data["athlete"]["firstname"]
            st.query_params.clear()
            st.rerun()
        else: st.error("Login failed.")
    else:
        st.title("🚴 Endurance Lab Pro")
        st.write("Connect Strava to access advanced analytics.")
        st.link_button("Connect with Strava", get_auth_url())
        st.stop()

# FIX: Robustly get athlete name to prevent KeyError on reboot
athlete_name = st.session_state.get("athlete_name", "Athlete")
st.title(f"Endurance Lab: {athlete_name}")

with st.spinner("Analyzing training history..."):
    raw_data = fetch_activities(st.session_state["access_token"])
if not raw_data:
    st.warning("No rides found.")
    st.stop()

df = pd.json_normalize(raw_data)
# FIX: GLOBAL TIMEZONE STRIP - Convert to datetime, then strip timezone IMMEDIATELY
df['start_date_local'] = pd.to_datetime(df['start_date_local']).dt.tz_localize(None)
df['date_filter'] = df['start_date_local'] 
df['distance_miles'] = df['distance'] / 1609.34

# SIDEBAR
with st.sidebar:
    st.header("⚙️ Settings")
    ftp_input = st.number_input("Your FTP (Watts)", min_value=100, max_value=500, value=250)
    max_hr_input = st.number_input("Max Heart Rate (bpm)", min_value=100, max_value=220, value=190)
    st.divider()
    st.header("📅 Time Frame")
    min_date, max_date = df['date_filter'].min().date(), df['date_filter'].max().date()
    default_start = max(min_date, max_date - timedelta(days=90))
    date_range = st.date_input("Select Range", value=(default_start, max_date), min_value=min_date, max_value=max_date)
    if st.button("Log Out"):
        st.session_state.clear()
        st.rerun()

# 1. Calc Load
df = calculate_training_load(df, ftp_input)

# 2. DEDUPLICATE (Consolidate Wahoo + Apple Watch)
df = deduplicate_rides(df)

# 3. Calc PMC
pmc_data_full = calculate_pmc(df)

if len(date_range) == 2:
    # Ensure start_ts/end_ts are timezone naive
    start_ts = pd.Timestamp(date_range[0])
    end_ts = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    
    # Filter using naive datetimes (safe)
    df_display = df[(df['date_filter'] >= start_ts) & (df['date_filter'] <= end_ts)].copy()
    pmc_mask = (pmc_data_full['date'] >= start_ts) & (pmc_data_full['date'] <= end_ts)
    pmc_display = pmc_data_full[pmc_mask]
else:
    df_display = df.copy()
    pmc_display = pmc_data_full.copy()

tab1, tab2, tab3 = st.tabs(["📈 Performance & Maps", "🔬 Deep Analysis", "📋 Ride Log"])

with tab1:
    st.subheader("Training Summary")
    
    def get_summary_stats(dframe):
        if dframe.empty:
            return 0, 0, 0, 0
        dist = dframe['distance_miles'].sum()
        hours = dframe['moving_time'].sum() / 3600
        elev = dframe['total_elevation_gain'].sum() * 3.28084
        count = len(dframe)
        return dist, hours, elev, count

    now = datetime.now()
    date_7d = now - timedelta(days=7)
    date_30d = now - timedelta(days=30)
    date_ytd = datetime(now.year, 1, 1)

    d7_stats = get_summary_stats(df[df['date_filter'] >= date_7d])
    d30_stats = get_summary_stats(df[df['date_filter'] >= date_30d])
    ytd_stats = get_summary_stats(df[df['date_filter'] >= date_ytd])

    summary_data = {
        "Metric": ["Distance (miles)", "Time (hours)", "Elevation (ft)", "Rides"],
        "Last 7 Days": [f"{d7_stats[0]:.1f}", f"{d7_stats[1]:.1f}", f"{d7_stats[2]:,.0f}", f"{d7_stats[3]}"],
        "Last 30 Days": [f"{d30_stats[0]:.1f}", f"{d30_stats[1]:.1f}", f"{d30_stats[2]:,.0f}", f"{d30_stats[3]}"],
        "Year to Date": [f"{ytd_stats[0]:.1f}", f"{ytd_stats[1]:.1f}", f"{ytd_stats[2]:,.0f}", f"{ytd_stats[3]}"]
    }
    
    summary_df = pd.DataFrame(summary_data).set_index("Metric")
    st.dataframe(summary_df, use_container_width=True)
    
    st.divider()

    st.subheader("Performance Management")
    if not pmc_display.empty:
        c1, c2, c3 = st.columns(3)
        c1.metric("Fitness (CTL)", f"{pmc_display['CTL'].iloc[-1]:.1f}")
        c2.metric("Fatigue (ATL)", f"{pmc_display['ATL'].iloc[-1]:.1f}")
        c3.metric("Form (TSB)", f"{pmc_display['TSB'].iloc[-1]:.1f}")
        
        fig_pmc = go.Figure()
        fig_pmc.add_trace(go.Scatter(x=pmc_display['date'], y=pmc_display['CTL'], name='Fitness', line=dict(color='blue')))
        fig_pmc.add_trace(go.Scatter(x=pmc_display['date'], y=pmc_display['ATL'], name='Fatigue', line=dict(color='magenta')))
        fig_pmc.add_trace(go.Bar(x=pmc_display['date'], y=pmc_display['TSB'], name='Form', marker_color='orange', opacity=0.5))
        fig_pmc.update_layout(height=400, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig_pmc, use_container_width=True)

    # --- NEW: Efficiency Factor Chart (Robust Manual Trendline) ---
    st.divider()
    st.subheader("Efficiency Factor (EF) Over Time")
    
    # Filter for valid EF rides
    ef_df = df_display.dropna(subset=['efficiency_factor']).copy()
    ef_df = ef_df[ef_df['efficiency_factor'] > 0] # Remove zeros
    
    if not ef_df.empty:
        # Calculate simple 5-ride moving average for trend
        ef_df = ef_df.sort_values('start_date_local')
        ef_df['EF_Trend'] = ef_df['efficiency_factor'].rolling(window=5, min_periods=1).mean()

        fig_ef = px.scatter(
            ef_df, 
            x="start_date_local", 
            y="efficiency_factor", 
            hover_data=["name", "average_watts", "average_heartrate"],
            labels={"efficiency_factor": "EF (Watts/HR)", "start_date_local": "Date"}
        )
        # Add the points
        fig_ef.update_traces(marker=dict(size=8, color='green', opacity=0.6))
        
        # Add the manual trendline
        fig_ef.add_trace(go.Scatter(
            x=ef_df['start_date_local'], 
            y=ef_df['EF_Trend'], 
            mode='lines', 
            name='Trend (5-Ride Avg)',
            line=dict(color='darkgreen', width=3)
        ))

        fig_ef.update_layout(height=400)
        st.plotly_chart(fig_ef, use_container_width=True)
        st.caption("Note: Efficiency Factor (EF) = Avg Power / Avg Heart Rate. Higher is better.")
    else:
        st.info("Not enough data to calculate Efficiency Factor (requires Power and Heart Rate).")

with tab2:
    st.write("### Single Ride Deep Dive")
    if not df_display.empty:
        df_display['label'] = df_display['start_date_local'].dt.strftime('%Y-%m-%d %H:%M') + " - " + df_display['name'] + " [" + df_display['device_name'].astype(str) + "]"
        ride_options = df_display[['label', 'id']].sort_values('label', ascending=False)
        selected_ride_label = st.selectbox("Choose a Ride:", ride_options['label'])
        selected_id = ride_options[ride_options['label'] == selected_ride_label]['id'].values[0]

        with st.spinner("Fetching details..."):
            streams = fetch_streams(st.session_state["access_token"], selected_id)
            
            # --- MERGE STREAMS LOGIC ---
            selected_row = df_display[df_display['id'] == selected_id].iloc[0]
            if selected_row.get('hr_source_id'):
                try:
                    hr_id = int(selected_row['hr_source_id'])
                    hr_streams = fetch_streams(st.session_state["access_token"], hr_id)
                    if hr_streams and 'heartrate' in hr_streams:
                        streams['heartrate'] = hr_streams['heartrate']
                        st.toast(f"Merged Heart Rate from Apple Watch ride (ID: {hr_id})")
                except Exception as e:
                    st.error(f"Failed to merge HR streams: {e}")
            # ---------------------------
        
        if streams:
            data = calculate_advanced_metrics(streams, ftp_input, max_hr_input)
            
            # --- UPDATED: 3 Split Tables ---
            st.subheader("Ride Metrics")
            
            t1, t2, t3 = st.columns(3)
            
            with t1:
                st.caption("📏 Speed & Distance")
                table1 = {
                    "Metric": ["Distance", "Total Time", "Moving Time", "Avg Speed", "Max Speed", "Avg Cadence"],
                    "Value": [
                        f"{selected_row['distance_miles']:.1f} mi",
                        format_seconds(selected_row.get('elapsed_time', 0)),
                        format_seconds(selected_row.get('moving_time', 0)),
                        f"{data['avg_speed']:.1f} mph" if data['avg_speed'] else "N/A",
                        f"{data['max_speed']:.1f} mph" if data['max_speed'] else "N/A",
                        f"{data['avg_cadence']:.0f} rpm" if data['avg_cadence'] else "N/A"
                    ]
                }
                st.dataframe(pd.DataFrame(table1), hide_index=True, use_container_width=True)

            with t2:
                st.caption("❤️ Heart Rate")
                table2 = {
                    "Metric": ["Avg HR", "Max HR", "Decoupling"],
                    "Value": [
                        f"{data['avg_hr']:.0f} bpm" if data['avg_hr'] else "N/A",
                        f"{data['max_hr']:.0f} bpm" if data['max_hr'] else "N/A",
                        f"{data['decoupling']:.1f}%" if data['decoupling'] else "N/A"
                    ]
                }
                st.dataframe(pd.DataFrame(table2), hide_index=True, use_container_width=True)

            with t3:
                st.caption("⚡ Power & Load")
                table3 = {
                    "Metric": ["Avg Power", "Norm Power (NP)", "Work", "Intensity (IF)", "Variability (VI)", "TSS", "Efficiency (EF)"],
                    "Value": [
                        f"{data['avg_pwr']:.0f} W" if data.get('avg_pwr') else "N/A",
                        f"{data['np']:.0f} W" if data['np'] else "N/A",
                        f"{selected_row.get('kilojoules', 0):.0f} kJ",
                        f"{data['if']:.2f}" if data['if'] else "N/A",
                        f"{data['vi']:.2f}" if data['vi'] else "N/A",
                        f"{selected_row.get('tss_score', 0):.0f}",
                        f"{data['np']/data['avg_hr']:.2f}" if (data['np'] and data['avg_hr']) else "N/A"
                    ]
                }
                st.dataframe(pd.DataFrame(table3), hide_index=True, use_container_width=True)

            # --- MAP SECTION (Moved to Tab 2) ---
            st.divider()
            st.subheader("Route Map")
            if selected_row.get('map.summary_polyline'):
                decoded_path = decode_polyline(selected_row['map.summary_polyline'])
                if decoded_path:
                    # Calculate center for view state
                    lats = [p[1] for p in decoded_path]
                    lngs = [p[0] for p in decoded_path]
                    mid_lat = sum(lats) / len(lats)
                    mid_lng = sum(lngs) / len(lngs)

                    view_state = pdk.ViewState(
                        latitude=mid_lat,
                        longitude=mid_lng,
                        zoom=11,
                        pitch=0
                    )
                    
                    # 1. Satellite Background Layer (Esri World Imagery - Free/Public)
                    satellite_layer = pdk.Layer(
                        "TileLayer",
                        data="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                        min_zoom=0,
                        max_zoom=19,
                        tileSize=256,
                    )

                    # 2. The Route Line
                    route_layer = pdk.Layer(
                        "PathLayer",
                        data=[{"path": decoded_path}],
                        get_path="path",
                        get_color=[255, 75, 75], # Bright Red
                        width_min_pixels=3,
                        opacity=1.0
                    )
                    
                    st.pydeck_chart(pdk.Deck(
                        map_style=None, # Disable Mapbox style to rely on TileLayer
                        initial_view_state=view_state,
                        layers=[satellite_layer, route_layer] 
                    ))
                else:
                    st.info("No GPS data for this ride.")
            else:
                st.info("No map available.")

            # --- NEW: Robust Interactive Chart ---
            st.divider()
            st.subheader("📊 Power vs. Heart Rate Analysis")
            
            try:
                # Prepare Chart Data
                chart_data = {}
                if 'time' in streams:
                    chart_data['Time'] = streams['time']['data']
                    if 'watts' in streams:
                        chart_data['Power'] = streams['watts']['data']
                    if 'heartrate' in streams:
                        chart_data['Heart Rate'] = streams['heartrate']['data']
                    
                    if chart_data and len(chart_data) > 1:
                        min_len = min(len(v) for v in chart_data.values())
                        chart_df = pd.DataFrame({k: v[:min_len] for k, v in chart_data.items()})
                        
                        # --- UPDATED: Convert Seconds to Minutes ---
                        if 'Time' in chart_df.columns:
                            chart_df['Time'] = chart_df['Time'] / 60
                        
                        if not chart_df.empty:
                            fig = go.Figure()
                            
                            # Smart Axis Logic
                            has_power = 'Power' in chart_df.columns
                            has_hr = 'Heart Rate' in chart_df.columns
                            
                            # 1. Power Trace
                            if has_power:
                                fig.add_trace(go.Scatter(
                                    x=chart_df['Time'], y=chart_df['Power'],
                                    name="Power (W)", line=dict(color='#FFA500', width=1), opacity=0.7
                                ))
                            
                            # 2. Heart Rate Trace
                            if has_hr:
                                # If Power exists, map HR to Y2. If not, HR stays on Y1.
                                y_axis_name = "y2" if has_power else "y"
                                fig.add_trace(go.Scatter(
                                    x=chart_df['Time'], y=chart_df['Heart Rate'],
                                    name="Heart Rate (bpm)", line=dict(color='#FF4B4B', width=2),
                                    yaxis=y_axis_name
                                ))

                            # 3. Dynamic Layout
                            layout_opts = dict(
                                height=400,
                                hovermode="x unified",
                                xaxis=dict(title="Duration (minutes)"), 
                                margin=dict(l=0, r=0, t=30, b=0),
                                legend=dict(orientation="h", y=1.1)
                            )
                            
                            if has_power:
                                layout_opts['yaxis'] = dict(
                                    title=dict(text="Power (Watts)", font=dict(color="#FFA500")),
                                    tickfont=dict(color="#FFA500")
                                )
                                
                            if has_hr:
                                if has_power:
                                    # Dual Axis Mode
                                    layout_opts['yaxis2'] = dict(
                                        title=dict(text="Heart Rate (bpm)", font=dict(color="#FF4B4B")),
                                        tickfont=dict(color="#FF4B4B"),
                                        overlaying="y",
                                        side="right"
                                    )
                                else:
                                    # Single Axis Mode (HR only)
                                    layout_opts['yaxis'] = dict(
                                        title=dict(text="Heart Rate (bpm)", font=dict(color="#FF4B4B")),
                                        tickfont=dict(color="#FF4B4B")
                                    )

                            fig.update_layout(**layout_opts)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Chart data is empty.")
                    else:
                        st.info("Insufficient data for Power/HR chart.")
            except Exception as e:
                st.warning(f"Could not render chart: {e}")
            # -------------------------------------

            st.divider()
            
            try:
                export_data = {}
                for k, v in streams.items():
                    if 'data' in v:
                        export_data[k] = v['data']
                
                lengths = [len(v) for v in export_data.values()]
                if lengths:
                    min_len = min(lengths)
                    export_data_synced = {k: v[:min_len] for k, v in export_data.items()}
                    csv_df = pd.DataFrame(export_data_synced)
                    csv_file = csv_df.to_csv(index=False).encode('utf-8')
                    
                    st.download_button(
                        label="Download Ride Data (CSV)",
                        data=csv_file,
                        file_name=f"ride_{selected_id}.csv",
                        mime="text/csv",
                    )
            except Exception as e:
                st.info(f"CSV download not available: {e}")

            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                if data['np']:
                    st.subheader("Power Curve")
                    dur = list(data['pdc'].keys())
                    pwr = list(data['pdc'].values())
                    st.plotly_chart(px.line(x=dur, y=pwr, title="Mean Max Power"), use_container_width=True)
                else:
                    st.info("No Power Curve available.")
            
            with col_chart2:
                if 'zone_dist' in data:
                    st.subheader("Time in Power Zones")
                    st.plotly_chart(px.bar(data['zone_dist'], orientation='h', title="Power Distribution", labels={'value': 'Seconds', 'index': 'Zone'}), use_container_width=True)
                
                if 'hr_zone_dist' in data:
                    st.subheader("Time in HR Zones (%)")
                    st.plotly_chart(px.bar(data['hr_zone_dist'], orientation='h', 
                                         title="Heart Rate Distribution",
                                         labels={'value': 'Percentage (%)', 'index': 'Zone'},
                                         text_auto='.1f'), use_container_width=True)

            if GEMINI_AVAILABLE:
                st.divider()
                st.subheader("🤖 AI Coach")
                q = st.text_input("Ask Gemini about this ride:", placeholder="How was my pacing?")
                if st.button("Ask Coach"):
                    with st.spinner("Thinking..."):
                        st.markdown(ask_gemini(data, streams, q))
        else: st.error("Could not load ride data.")
    else: st.warning("No rides in selected date range.")

with tab3:
    st.subheader("Ride Log")
    cols = [
        'start_date_local', 'name', 'device_name', 'distance_miles', 'average_speed_mph', 
        'average_watts', 'variability_index', 'IF', 'efficiency_factor', 
        'average_heartrate', 'tss_score'
    ]
    valid_cols = [c for c in cols if c in df_display.columns]
    
    st.dataframe(
        df_display[valid_cols]
        .sort_values('start_date_local', ascending=False)
        .style.format("{:.2f}", subset=[c for c in ['distance_miles', 'average_speed_mph', 'average_watts', 'variability_index', 'IF', 'efficiency_factor', 'tss_score'] if c in df_display.columns])
    )
