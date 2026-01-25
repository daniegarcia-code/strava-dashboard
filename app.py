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
    
    if 'average_watts' not in df.columns: df['average_watts'] = 0
    if 'average_heartrate' not in df.columns: df['average_heartrate'] = 0
    if 'average_speed' not in df.columns: df['average_speed'] = 0
    
    df['IF'] = df['average_watts'] / ftp
    df['tss_score'] = (df['moving_time'] * df['average_watts'] * df['IF'] * 1.05) / (ftp * 3600) * 100
    
    mask_no_power = (df['average_watts'] < 1) & (df['average_heartrate'] > 1)
    df.loc[mask_no_power, 'tss_score'] = (df.loc[mask_no_power, 'moving_time'] / 3600) * 60
    
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
    
    # --- Device Name Handling ---
    if 'device_name' not in df.columns:
        df['device_name'] = "Unknown" 
    else:
        df['device_name'] = df['device_name'].fillna("Unknown")
    
    return df

def calculate_pmc(df):
    df = df.sort_values('start_date_local', ascending=True)
    df['date_clean'] = df['start_date_local'].dt.tz_localize(None)
    df = df.set_index('date_clean')
    full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
    daily_tss = df['tss_score'].resample('D').sum().reindex(full_idx, fill_value=0)
    
    ctl = daily_tss.ewm(span=42, adjust=False).mean()
    atl = daily_tss.ewm(span=7, adjust=False).mean()
    tsb = ctl - atl
    
    pmc_df = pd.DataFrame({'CTL': ctl, 'ATL': atl, 'TSB': tsb})
    return pmc_df.reset_index().rename(columns={'index': 'date'})

def calculate_advanced_metrics(streams, ftp):
    metrics = {}
    
    if 'watts' in streams:
        watts = pd.Series(streams['watts']['data'])
        metrics['np'] = np.power(watts.rolling(30).mean().pow(4).mean(), 0.25)
        metrics['avg_pwr'] = watts.mean()
        metrics['vi'] = metrics['np'] / metrics['avg_pwr'] if metrics['avg_pwr'] > 0 else 1.0
        metrics['if'] = metrics['np'] / ftp
        
        # List of durations to calculate (in seconds)
        std_durations = [1, 5, 15, 30, 60, 180, 300, 600, 1200, 1800, 2700, 3600, 5400, 7200, 10800, 14400, 18000]
        metrics['pdc'] = {}
        for d in std_durations:
            # Format Label
            if d >= 3600:
                label = f"{d/3600:.1f}h"
            elif d >= 60:
                label = f"{int(d/60)}m"
            else:
                label = f"{d}s"
            
            # Calculate only if ride is long enough
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

# --- ROBUST AI FUNCTION (AUTO-DETECT & ERROR SAFE & FULL DATA) ---
def ask_gemini(metrics, streams, question): # Added streams arg
    if not GEMINI_AVAILABLE: return "Gemini API Key not found."
    
    # 1. AUTO-DETECT AVAILABLE MODELS
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

    # 2. PREPARE RAW DATA (CSV FORMAT) FOR AI
    csv_context = ""
    try:
        # Build dictionary from streams
        raw_data = {}
        if streams:
            for key, val in streams.items():
                if 'data' in val:
                    raw_data[key] = val['data']
            
            # Ensure equal lengths for DataFrame creation
            if raw_data:
                min_len = min(len(v) for v in raw_data.values())
                # Truncate to matching length
                raw_data_synced = {k: v[:min_len] for k, v in raw_data.items()}
                
                # Create DataFrame and stringify
                df_context = pd.DataFrame(raw_data_synced)
                # We limit rows if huge, but 1.5 Flash handles ~1M tokens, 
                # so full ride data (e.g. 10k-20k rows) is usually fine.
                csv_context = df_context.to_csv(index=False)
    except Exception as e:
        csv_context = f"[Error processing raw data: {e}]"

    # 3. GENERATE CONTENT
    safe_get = lambda k: metrics.get(k) or 0
    np_val = f"{safe_get('np'):.0f}" if metrics.get('np') else "N/A"
    
    prompt = f"""
    You are an expert cycling coach. Analyze this ride data.
    
    ### Ride Summary Metrics:
    - NP: {np_val} W
    - IF: {safe_get('if'):.2f}
    - VI: {safe_get('vi'):.2f}
    - Decoupling: {safe_get('decoupling'):.1f}%
    - Avg HR: {safe_get('avg_hr'):.0f} bpm
    
    ### Full Second-by-Second Data (CSV):
    {csv_context}
    
    ### User Question:
    "{question}"
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

st.title(f"Endurance Lab: {st.session_state['athlete_name']}")

with st.spinner("Analyzing training history..."):
    raw_data = fetch_activities(st.session_state["access_token"])
if not raw_data:
    st.warning("No rides found.")
    st.stop()

df = pd.json_normalize(raw_data)
df['start_date_local'] = pd.to_datetime(df['start_date_local'])
df['date_filter'] = df['start_date_local'].dt.tz_localize(None) 
df['distance_miles'] = df['distance'] / 1609.34

# SIDEBAR
with st.sidebar:
    st.header("⚙️ Settings")
    ftp_input = st.number_input("Your FTP (Watts)", min_value=100, max_value=500, value=250)
    st.divider()
    st.header("📅 Time Frame")
    min_date, max_date = df['date_filter'].min().date(), df['date_filter'].max().date()
    default_start = max(min_date, max_date - timedelta(days=90))
    date_range = st.date_input("Select Range", value=(default_start, max_date), min_value=min_date, max_value=max_date)
    if st.button("Log Out"):
        st.session_state.clear()
        st.rerun()

df = calculate_training_load(df, ftp_input)
pmc_data_full = calculate_pmc(df)

if len(date_range) == 2:
    start_ts = pd.Timestamp(date_range[0])
    end_ts = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    df_display = df[(df['date_filter'] >= start_ts) & (df['date_filter'] <= end_ts)].copy()
    pmc_display = pmc_data_full[(pmc_data_full['date'] >= start_ts) & (pmc_data_full['date'] <= end_ts)]
else:
    df_display = df.copy()
    pmc_display = pmc_data_full.copy()

tab1, tab2, tab3 = st.tabs(["📈 Performance & Maps", "🔬 Deep Analysis", "📋 Ride Log"])

with tab1:
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

    st.divider()
    st.subheader(f"Heatmap ({len(df_display)} rides)")
    if 'map.summary_polyline' in df_display.columns:
        map_df = df_display[df_display['map.summary_polyline'].notna()].copy()
        if not map_df.empty:
            map_df['path'] = map_df['map.summary_polyline'].apply(decode_polyline)
            map_df = map_df[map_df['path'].map(len) > 0]
            if not map_df.empty:
                layer = pdk.Layer("PathLayer", data=map_df, get_path="path", get_color=[255, 75, 75], width_min_pixels=2, opacity=0.8)
                start_pt = map_df.iloc[0]['path'][0]
                view_state = pdk.ViewState(latitude=start_pt[1], longitude=start_pt[0], zoom=10)
                st.pydeck_chart(pdk.Deck(map_style="mapbox://styles/mapbox/light-v9", initial_view_state=view_state, layers=[layer]))
            else: st.info("No valid routes found.")
        else: st.info("No map data available.")

with tab2:
    st.write("### Single Ride Deep Dive")
    if not df_display.empty:
        df_display['label'] = df_display['start_date_local'].dt.strftime('%Y-%m-%d %H:%M') + " - " + df_display['name'] + " [" + df_display['device_name'].astype(str) + "]"
        ride_options = df_display[['label', 'id']].sort_values('label', ascending=False)
        selected_ride_label = st.selectbox("Choose a Ride:", ride_options['label'])
        selected_id = ride_options[ride_options['label'] == selected_ride_label]['id'].values[0]

        with st.spinner("Fetching details..."):
            streams = fetch_streams(st.session_state["access_token"], selected_id)
        
        if streams:
            data = calculate_advanced_metrics(streams, ftp_input)
            
            st.markdown("#### ⚡ Power Stats")
            if data['np']:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Normalized Power", f"{data['np']:.0f} W")
                c2.metric("Intensity (IF)", f"{data['if']:.2f}")
                c3.metric("Variability (VI)", f"{data['vi']:.2f}")
                work_kj = df[df['id']==selected_id]['kilojoules'].values[0]
                c4.metric("Work", f"{work_kj:.0f} kJ")
            else: st.info("No Power Data.")

            st.markdown("#### ❤️ Heart Rate Stats")
            if data['avg_hr']:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Avg HR", f"{data['avg_hr']:.0f} bpm")
                c2.metric("Max HR", f"{data['max_hr']:.0f} bpm")
                if data['decoupling']: c3.metric("Decoupling", f"{data['decoupling']:.1f}%")
                else: c3.metric("Decoupling", "N/A")
                if data['np'] and data['avg_hr'] > 0: c4.metric("Efficiency (EF)", f"{data['np']/data['avg_hr']:.2f}")
            else: st.info("No Heart Rate Data.")

            st.markdown("#### 💨 Speed & Cadence")
            c1, c2, c3, c4 = st.columns(4)
            if data['avg_speed']:
                c1.metric("Avg Speed", f"{data['avg_speed']:.1f} mph")
                c2.metric("Max Speed", f"{data['max_speed']:.1f} mph")
            if data['avg_cadence']: c3.metric("Avg Cadence", f"{data['avg_cadence']:.0f} rpm")
            
            st.divider()
            
            # --- CSV Download Button ---
            try:
                export_data = {}
                for k, v in streams.items():
                    if 'data' in v:
                        export_data[k] = v['data']
                
                lengths = [len(v) for v in export_data.values()]
                if lengths and all(x == lengths[0] for x in lengths):
                    csv_df = pd.DataFrame(export_data)
                    csv_file = csv_df.to_csv(index=False).encode('utf-8')
                    
                    st.download_button(
                        label="Download Ride Data (CSV)",
                        data=csv_file,
                        file_name=f"ride_{selected_id}.csv",
                        mime="text/csv",
                    )
            except Exception as e:
                st.info(f"CSV download not available for this activity type.")
            # ----------------------------------

            if data['np']:
                col_chart1, col_chart2 = st.columns(2)
                with col_chart1:
                    st.subheader("Power Curve")
                    dur = list(data['pdc'].keys())
                    pwr = list(data['pdc'].values())
                    st.plotly_chart(px.line(x=dur, y=pwr, title="Mean Max Power"), use_container_width=True)
                
                with col_chart2:
                    st.subheader("Time in Zones")
                    if 'zone_dist' in data:
                        st.plotly_chart(px.bar(data['zone_dist'], orientation='h', title="Power Distribution", labels={'value': 'Seconds', 'index': 'Zone'}), use_container_width=True)

            if GEMINI_AVAILABLE:
                st.divider()
                st.subheader("🤖 AI Coach")
                q = st.text_input("Ask Gemini about this ride:", placeholder="How was my pacing?")
                if st.button("Ask Coach"):
                    with st.spinner("Thinking..."):
                        # Updated call passing streams
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
