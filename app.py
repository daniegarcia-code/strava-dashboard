import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk  # IMPORT PYDECK FOR MAPS
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
REDIRECT_URI = "https://strava-dashboard-bavkdzxtyephsasu7k9q7b.streamlit.app" 

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

# --- HELPER: POLYLINE DECODER (For Maps) ---
def decode_polyline(polyline_str):
    """Decodes Strava's summary polyline into [Lon, Lat] coordinates for Pydeck."""
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
                if not byte >= 0x20:
                    break
            if (result & 1):
                changes[unit] = ~(result >> 1)
            else:
                changes[unit] = (result >> 1)

        lat += changes['latitude']
        lng += changes['longitude']
        # Pydeck expects [Longitude, Latitude]
        coordinates.append([lng / 100000.0, lat / 100000.0])

    return coordinates

# --- DATA FETCHING FUNCTIONS ---
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
        else:
            break
            
    return all_data[:num_activities]

@st.cache_data(ttl=300)
def fetch_streams(access_token, activity_id):
    url = f"https://www.strava.com/api/v3/activities/{activity_id}/streams"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {"keys": "time,watts,heartrate,velocity_smooth", "key_by_type": "true"}
    
    r = requests.get(url, headers=headers, params=params)
    if r.status_code == 200:
        return r.json()
    return None

# --- METRIC CALCULATION FUNCTIONS ---
def calculate_training_load(df, ftp):
    if ftp <= 0: ftp = 200 
    
    if 'average_watts' not in df.columns: df['average_watts'] = 0
    if 'average_heartrate' not in df.columns: df['average_heartrate'] = 0
    
    df['IF'] = df['average_watts'] / ftp
    df['tss_score'] = (df['moving_time'] * df['average_watts'] * df['IF'] * 1.05) / (ftp * 3600) * 100
    
    mask_no_power = (df['average_watts'] < 1) & (df['average_heartrate'] > 1)
    df.loc[mask_no_power, 'tss_score'] = (df.loc[mask_no_power, 'moving_time'] / 3600) * 60
    
    df['efficiency_factor'] = df['average_watts'] / df['average_heartrate']
    df['kilojoules'] = df.get('kilojoules', pd.Series([0]*len(df))).fillna(0)
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
    if 'watts' not in streams:
        return None, None, None, None, None

    watts = pd.Series(streams['watts']['data'])
    
    if 'heartrate' in streams:
        hr = pd.Series(streams['heartrate']['data'])
    else:
        hr = pd.Series([0] * len(watts))

    rolling_30s = watts.rolling(window=30, min_periods=1).mean()
    np_val = np.power(rolling_30s.pow(4).mean(), 0.25)
    
    avg_pwr = watts.mean()
    if avg_pwr > 0:
        vi_val = np_val / avg_pwr
    else:
        vi_val = 1.0

    if_val = np_val / ftp
    
    half_idx = len(watts) // 2
    if hr.sum() > 0: 
        p1 = watts.iloc[:half_idx].mean()
        h1 = hr.iloc[:half_idx].mean()
        r1 = p1 / h1 if h1 > 0 else 0
        p2 = watts.iloc[half_idx:].mean()
        h2 = hr.iloc[half_idx:].mean()
        r2 = p2 / h2 if h2 > 0 else 0
        decoupling = (r1 - r2) / r1 * 100 if r1 > 0 else 0
    else:
        decoupling = 0
    
    durations = [1, 5, 15, 30, 60, 180, 300, 600, 1200]
    pdc_data = {}
    for d in durations:
        if d > 60: label = f"{int(d/60)}m"
        else: label = f"{d}s"
            
        if len(watts) > d:
            pdc_data[label] = watts.rolling(window=d).mean().max()
        else:
            pdc_data[label] = 0
            
    return np_val, if_val, vi_val, decoupling, pdc_data

def ask_gemini(metrics, question):
    if not GEMINI_AVAILABLE: return "Gemini API Key not found."
    model = genai.GenerativeModel('gemini-pro')
    prompt = f"""
    You are an expert cycling coach. Analyze this ride:
    - NP: {metrics['np']:.0f} W
    - IF: {metrics['if']:.2f}
    - VI: {metrics['vi']:.2f}
    - Decoupling: {metrics['decoupling']:.1f}%
    - Work: {metrics['work']:.0f} kJ
    User Question: "{question}"
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {e}"

# --- 2. AUTHENTICATION LOGIC ---
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
        else:
            st.error("Login failed.")
    else:
        st.title("🚴 Endurance Lab Pro")
        st.write("Connect Strava to access advanced analytics.")
        st.link_button("Connect with Strava", get_auth_url())
        st.stop()

# --- 3. MAIN DASHBOARD ---
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
    min_date = df['date_filter'].min().date()
    max_date = df['date_filter'].max().date()
    default_start = max_date - timedelta(days=90)
    if default_start < min_date: default_start = min_date
    date_range = st.date_input("Select Range", value=(default_start, max_date), min_value=min_date, max_value=max_date)
    
    if st.button("Log Out"):
        st.session_state.clear()
        st.rerun()

# CALCULATE
df = calculate_training_load(df, ftp_input)
pmc_data_full = calculate_pmc(df)

# FILTER
if len(date_range) == 2:
    start_d, end_d = date_range
    start_ts = pd.Timestamp(start_d)
    end_ts = pd.Timestamp(end_d) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    
    mask_pmc = (pmc_data_full['date'] >= start_ts) & (pmc_data_full['date'] <= end_ts)
    pmc_data_display = pmc_data_full.loc[mask_pmc]
    
    mask_df = (df['date_filter'] >= start_ts) & (df['date_filter'] <= end_ts)
    df_display = df.loc[mask_df]
else:
    pmc_data_display = pmc_data_full
    df_display = df

# --- TABS UI ---
tab1, tab2, tab3 = st.tabs(["📈 Performance & Maps", "🔬 Deep Analysis", "📋 Ride Log"])

with tab1:
    st.subheader("Performance Management Chart")
    if not pmc_data_display.empty:
        curr_ctl = pmc_data_display['CTL'].iloc[-1]
        curr_atl = pmc_data_display['ATL'].iloc[-1]
        curr_tsb = pmc_data_display['TSB'].iloc[-1]
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Fitness (CTL)", f"{curr_ctl:.1f}")
        c2.metric("Fatigue (ATL)", f"{curr_atl:.1f}")
        c3.metric("Form (TSB)", f"{curr_tsb:.1f}")

        fig_pmc = go.Figure()
        fig_pmc.add_trace(go.Scatter(x=pmc_data_display['date'], y=pmc_data_display['CTL'], mode='lines', name='Fitness (CTL)', line=dict(color='blue')))
        fig_pmc.add_trace(go.Scatter(x=pmc_data_display['date'], y=pmc_data_display['ATL'], mode='lines', name='Fatigue (ATL)', line=dict(color='magenta')))
        fig_pmc.add_trace(go.Bar(x=pmc_data_display['date'], y=pmc_data_display['TSB'], name='Form (TSB)', marker_color='orange', opacity=0.5))
        fig_pmc.update_layout(height=450, title="Fitness vs. Fatigue vs. Form", template="plotly_white")
        st.plotly_chart(fig_pmc, use_container_width=True)
    else:
        st.warning("No data in selected date range.")
    
    # --- ROUTE MAPS SECTION ---
    st.divider()
    st.subheader(f"Route Heatmap ({len(df_display)} rides)")
    
    # 1. Prepare Map Data
    # Check if map data exists in Strava response
    if 'map.summary_polyline' in df_display.columns:
        # Filter rows with valid polylines
        map_df = df_display[df_display['map.summary_polyline'].notna()].copy()
        
        if not map_df.empty:
            # Decode polylines into paths
            map_df['path'] = map_df['map.summary_polyline'].apply(decode_polyline)
            # Remove empty paths (failed decodes)
            map_df = map_df[map_df['path'].map(len) > 0]
            
            # 2. Define Pydeck Layer
            layer = pdk.Layer(
                "PathLayer",
                data=map_df,
                get_path="path",
                get_color=[255, 75, 75], # Red color for routes
                width_min_pixels=2,
                opacity=0.8
            )
            
            # 3. Calculate Viewport (Center the map on the most recent ride)
            # We take the first point of the most recent ride as the center
            if not map_df.empty:
                recent_path = map_df.iloc[0]['path']
                start_pt = recent_path[0] # [Lon, Lat]
                view_state = pdk.ViewState(
                    latitude=start_pt[1],
                    longitude=start_pt[0],
                    zoom=10,
                    pitch=0
                )
                
                # 4. Render Map
                st.pydeck_chart(pdk.Deck(
                    map_style="mapbox://styles/mapbox/light-v9",
                    initial_view_state=view_state,
                    layers=[layer],
                    tooltip={"text": "{name}"}
                ))
            else:
                st.info("No route data available to plot.")
        else:
            st.info("No rides with map data found in this period.")
    else:
        st.info("Map data not available in this activity feed.")
    # --------------------------

    st.subheader("Weekly Volume")
    if not df_display.empty:
        df_display['week_start'] = df_display['start_date_local'].dt.to_period('W').apply(lambda r: r.start_time)
        weekly_stats = df_display.groupby('week_start').agg({'distance_miles': 'sum', 'tss_score': 'sum'}).reset_index()
        fig_vol = px.bar(weekly_stats, x='week_start', y='distance_miles', title="Weekly Mileage")
        st.plotly_chart(fig_vol, use_container_width=True)

with tab2:
    st.write("### Single Ride Deep Dive")
    if not df_display.empty:
        ride_options = df_display[['name', 'id', 'start_date_local']].sort_values('start_date_local', ascending=False)
        selected_ride_name = st.selectbox("Choose a Ride:", ride_options['name'])
        selected_row = ride_options[ride_options['name'] == selected_ride_name]
        
        if not selected_row.empty:
            selected_id = selected_row['id'].values[0]
            if st.button(f"Analyze '{selected_ride_name}'"):
                with st.spinner("Fetching stream data..."):
                    streams = fetch_streams(st.session_state["access_token"], selected_id)
                if streams:
                    np_val, if_val, vi_val, decoupling, pdc_data = calculate_advanced_metrics(streams, ftp_input)
                    if np_val is not None:
                        m1, m2, m3, m4, m5 = st.columns(5)
                        m1.metric("Norm. Power", f"{np_val:.0f} w")
                        m2.metric("Intensity (IF)", f"{if_val:.2f}")
                        m3.metric("Variability (VI)", f"{vi_val:.2f}")
                        m4.metric("Decoupling", f"{decoupling:.1f} %")
                        work_kj = df[df['id']==selected_id]['kilojoules'].values[0]
                        m5.metric("Work", f"{work_kj:.0f} kJ")

                        st.subheader("Power Duration Curve")
                        durations = list(pdc_data.keys())
                        powers = list(pdc_data.values())
                        fig_pdc = px.line(x=durations, y=powers, markers=True, title="Mean Max Power")
                        st.plotly_chart(fig_pdc, use_container_width=True)
                        
                        st.divider()
                        st.subheader("🤖 AI Coach")
                        if GEMINI_AVAILABLE:
                            user_q = st.text_input("Ask Gemini about this ride:", placeholder="e.g., How was my pacing?")
                            if st.button("Ask Coach"):
                                ride_metrics = {'np': np_val, 'if': if_val, 'vi': vi_val, 'decoupling': decoupling, 'work': work_kj, 'pdc': pdc_data}
                                with st.spinner("Gemini is analyzing..."):
                                    ai_response = ask_gemini(ride_metrics, user_q)
                                    st.markdown(f"**Coach says:**\n\n{ai_response}")
                        else:
                            st.warning("Add `gemini_api_key` to secrets to enable AI.")
                    else:
                        st.warning("No Power data.")
                else:
                    st.error("Could not fetch details.")
        else:
            st.error("Selection Error.")
    else:
        st.warning("No rides available.")

with tab3:
    st.subheader("Ride Log")
    display_cols = ['start_date_local', 'name', 'distance_miles', 'average_watts', 'average_heartrate', 'efficiency_factor', 'IF', 'tss_score', 'kilojoules']
    valid_cols = [c for c in display_cols if c in df_display.columns]
    if not df_display.empty:
        st.dataframe(df_display[valid_cols].sort_values('start_date_local', ascending=False).style.format("{:.1f}", subset=[c for c in ['distance_miles', 'average_watts', 'average_heartrate', 'efficiency_factor', 'tss_score', 'kilojoules', 'IF'] if c in df_display.columns]))
