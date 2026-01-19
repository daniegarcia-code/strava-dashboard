import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="Endurance Lab Pro", page_icon="🚴", layout="wide")

# --- 1. SETUP AUTHENTICATION ---
# Load secrets
try:
    CLIENT_ID = st.secrets["strava_client_id"]
    CLIENT_SECRET = st.secrets["strava_client_secret"]
except (KeyError, FileNotFoundError):
    st.error("Secrets not found! Please check your .streamlit/secrets.toml file.")
    st.stop()

# Set Redirect URI (Must match Strava Settings)
REDIRECT_URI = "https://strava-dashboard-bavkdzxtyephsasu7k9q7b.streamlit.app" 
# FOR LOCAL TESTING UNCOMMENT BELOW:
# REDIRECT_URI = "http://localhost:8501"

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

# --- DATA FETCHING FUNCTIONS ---
@st.cache_data(ttl=300)
def fetch_activities(access_token, num_activities=100):
    """Fetches summary list of activities for the PMC."""
    url = "https://www.strava.com/api/v3/athlete/activities"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {"per_page": num_activities}
    
    all_data = []
    r = requests.get(url, headers=headers, params=params)
    if r.status_code == 200:
        return r.json()
    return []

@st.cache_data(ttl=300)
def fetch_streams(access_token, activity_id):
    """Fetches second-by-second data for ONE ride (Deep Dive)."""
    url = f"https://www.strava.com/api/v3/activities/{activity_id}/streams"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {"keys": "time,watts,heartrate,velocity_smooth", "key_by_type": "true"}
    
    r = requests.get(url, headers=headers, params=params)
    if r.status_code == 200:
        return r.json()
    return None

# --- METRIC CALCULATION FUNCTIONS ---

def calculate_training_load(df, ftp):
    """Calculates TSS, IF, and Efficiency Factor based on summary data."""
    # Avoid division by zero
    if ftp <= 0: ftp = 200 

    # 1. Estimate Intensity Factor (IF) = Avg Power / FTP
    # (Note: True IF uses NP, but we use AvgPwr for the summary list to save API calls)
    df['IF'] = df['average_watts'] / ftp
    
    # 2. Estimate TSS = (sec * avg_watts * IF) / (FTP * 3600) * 100
    # We use a slight correction factor (1.05) to approximate NP vs AvgPwr difference
    df['tss_score'] = (df['moving_time'] * df['average_watts'] * df['IF'] * 1.05) / (ftp * 3600) * 100
    
    # 3. Efficiency Factor (EF) = Avg Watts / Avg HR
    df['efficiency_factor'] = df['average_watts'] / df['average_heartrate']
    
    # 4. Work (kJ)
    df['kilojoules'] = df['kilojoules'].fillna(0)
    
    return df

def calculate_pmc(df):
    """Calculates CTL (Fitness), ATL (Fatigue), and TSB (Form)."""
    # Sort by date ascending for rolling calc
    df = df.sort_values('start_date_local', ascending=True)
    
    # Reindex to fill missing days with 0 TSS so decay works correctly
    df = df.set_index('start_date_local')
    full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
    daily_tss = df['tss_score'].resample('D').sum().reindex(full_idx, fill_value=0)
    
    # Calculate exponentially weighted moving averages
    # CTL = 42 day time constant
    # ATL = 7 day time constant
    ctl = daily_tss.ewm(span=42, adjust=False).mean()
    atl = daily_tss.ewm(span=7, adjust=False).mean()
    tsb = ctl - atl
    
    pmc_df = pd.DataFrame({'CTL': ctl, 'ATL': atl, 'TSB': tsb})
    return pmc_df.reset_index().rename(columns={'index': 'date'})

def calculate_advanced_metrics(streams, ftp):
    """Calculates strict NP, Decoupling, and PDC from Stream data."""
    try:
        watts = pd.Series(streams['watts']['data'])
        hr = pd.Series(streams['heartrate']['data'])
        time = pd.Series(streams['time']['data'])
    except KeyError:
        return None, None, None

    # 1. Normalized Power (NP)
    # 30s rolling average, raised to 4th power, mean, 4th root
    rolling_30s = watts.rolling(window=30, min_periods=1).mean()
    np_val = np.power(rolling_30s.pow(4).mean(), 0.25)
    
    # 2. Intensity Factor (True IF)
    if_val = np_val / ftp
    
    # 3. Aerobic Decoupling (Pw:HR)
    # Split ride into first half and second half
    half_idx = len(watts) // 2
    
    p1 = watts.iloc[:half_idx].mean()
    h1 = hr.iloc[:half_idx].mean()
    r1 = p1 / h1 if h1 > 0 else 0
    
    p2 = watts.iloc[half_idx:].mean()
    h2 = hr.iloc[half_idx:].mean()
    r2 = p2 / h2 if h2 > 0 else 0
    
    decoupling = (r1 - r2) / r1 * 100 if r1 > 0 else 0
    
    # 4. Power Duration Curve (PDC)
    # Calculate Max Mean Power for key durations
    durations = [1, 5, 15, 30, 60, 180, 300, 600, 1200] # seconds
    pdc_data = {}
    for d in durations:
        if len(watts) > d:
            pdc_data[f"{d}s"] = watts.rolling(window=d).mean().max()
        else:
            pdc_data[f"{d}s"] = 0
            
    return np_val, if_val, decoupling, pdc_data

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
        st.write("Connect Strava to access advanced analytics (PMC, PDC, Decoupling).")
        st.link_button("Connect with Strava", get_auth_url())
        st.stop()

# --- 3. MAIN DASHBOARD ---
st.title(f"Endurance Lab: {st.session_state['athlete_name']}")

# Sidebar: User Inputs for Math
with st.sidebar:
    st.header("⚙️ Athlete Settings")
    ftp_input = st.number_input("Your FTP (Watts)", min_value=100, max_value=500, value=250)
    st.info("FTP is required to calculate TSS, IF, and CTL.")
    if st.button("Log Out"):
        st.session_state.clear()
        st.rerun()

# Fetch Summary Data
with st.spinner("Analyzing training history..."):
    raw_data = fetch_activities(st.session_state["access_token"])

if not raw_data:
    st.warning("No rides found.")
    st.stop()

# Process Summary Data
df = pd.json_normalize(raw_data)
df['start_date_local'] = pd.to_datetime(df['start_date_local'])
df['distance_miles'] = df['distance'] / 1609.34
# Ensure we have power data
if 'average_watts' not in df.columns:
    df['average_watts'] = 0
df['average_watts'] = df['average_watts'].fillna(0)
df['average_heartrate'] = df.get('average_heartrate', pd.Series([0]*len(df))).fillna(0)
df['kilojoules'] = df.get('kilojoules', pd.Series([0]*len(df)))

# Calculate Macro Metrics
df = calculate_training_load(df, ftp_input)
pmc_data = calculate_pmc(df)

# --- TABS UI ---
tab1, tab2, tab3 = st.tabs(["📈 Performance Management (PMC)", "🔬 Deep Analysis", "📋 Ride Log"])

# TAB 1: PMC CHART
with tab1:
    st.subheader("Performance Management Chart")
    
    # Current Stats
    curr_ctl = pmc_data['CTL'].iloc[-1]
    curr_atl = pmc_data['ATL'].iloc[-1]
    curr_tsb = pmc_data['TSB'].iloc[-1]
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Fitness (CTL)", f"{curr_ctl:.1f}", delta_color="normal")
    c2.metric("Fatigue (ATL)", f"{curr_atl:.1f}", delta_color="inverse")
    c3.metric("Form (TSB)", f"{curr_tsb:.1f}")

    # The PMC Chart
    fig_pmc = go.Figure()
    # CTL (Fitness) - Blue Line
    fig_pmc.add_trace(go.Scatter(x=pmc_data['date'], y=pmc_data['CTL'], mode='lines', name='Fitness (CTL)', line=dict(color='blue')))
    # ATL (Fatigue) - Pink Line
    fig_pmc.add_trace(go.Scatter(x=pmc_data['date'], y=pmc_data['ATL'], mode='lines', name='Fatigue (ATL)', line=dict(color='magenta')))
    # TSB (Form) - Bar/Area
    fig_pmc.add_trace(go.Bar(x=pmc_data['date'], y=pmc_data['TSB'], name='Form (TSB)', marker_color='orange', opacity=0.5))
    
    fig_pmc.update_layout(height=450, title="Fitness vs. Fatigue vs. Form", template="plotly_white")
    st.plotly_chart(fig_pmc, use_container_width=True)
    
    st.write("---")
    # Weekly Stats
    st.subheader("Weekly Volume & Intensity")
    df['week_start'] = df['start_date_local'].dt.to_period('W').apply(lambda r: r.start_time)
    weekly_stats = df.groupby('week_start').agg({'distance_miles': 'sum', 'tss_score': 'sum', 'moving_time': 'sum'}).reset_index()
    
    col_a, col_b = st.columns(2)
    with col_a:
        fig_vol = px.bar(weekly_stats, x='week_start', y='distance_miles', title="Weekly Mileage")
        st.plotly_chart(fig_vol, use_container_width=True)
    with col_b:
        fig_tss = px.bar(weekly_stats, x='week_start', y='tss_score', title="Weekly Training Stress (TSS)")
        st.plotly_chart(fig_tss, use_container_width=True)

# TAB 2: DEEP DIVE (STREAMS)
with tab2:
    st.write("### Single Ride Deep Dive")
    st.info("Select a ride below to calculate Normalized Power, Decoupling, and PDC. (Requires fetching detailed stream data)")
    
    # Dropdown to select ride
    ride_options = df[['name', 'id', 'start_date_local']].sort_values('start_date_local', ascending=False)
    selected_ride_name = st.selectbox("Choose a Ride:", ride_options['name'])
    
    # Get ID
    selected_id = ride_options[ride_options['name'] == selected_ride_name]['id'].values[0]
    
    if st.button(f"Analyze '{selected_ride_name}'"):
        with st.spinner("Fetching second-by-second stream data..."):
            streams = fetch_streams(st.session_state["access_token"], selected_id)
            
        if streams:
            # Calculate Advanced Metrics
            np_val, if_val, decoupling, pdc_data = calculate_advanced_metrics(streams, ftp_input)
            
            if np_val:
                # Top Row Metrics
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Normalized Power (NP)", f"{np_val:.0f} w")
                m2.metric("Intensity Factor (IF)", f"{if_val:.2f}")
                m3.metric("Aerobic Decoupling", f"{decoupling:.1f} %", delta=None, help="Positive is drift (fatigue). <5% is good.")
                m4.metric("Work", f"{df[df['id']==selected_id]['kilojoules'].values[0]:.0f} kJ")

                # PDC Chart
                st.subheader("Power Duration Curve")
                durations = list(pdc_data.keys())
                powers = list(pdc_data.values())
                fig_pdc = px.line(x=durations, y=powers, markers=True, title="Mean Max Power")
                fig_pdc.update_layout(xaxis_title="Duration", yaxis_title="Watts")
                st.plotly_chart(fig_pdc, use_container_width=True)
                
                # Zone Distribution
                st.subheader("Power Zones")
                watts_series = pd.Series(streams['watts']['data'])
                # Simple 5 Zone Model based on FTP
                zones = [0, 0.55*ftp_input, 0.75*ftp_input, 0.90*ftp_input, 1.05*ftp_input, 1.20*ftp_input, 2000]
                labels = ["Z1 Recovery", "Z2 Endurance", "Z3 Tempo", "Z4 Threshold", "Z5 VO2Max", "Z6 Anaerobic"]
                watts_series_cut = pd.cut(watts_series, bins=zones, labels=labels)
                zone_dist = watts_series_cut.value_counts(sort=False)
                
                fig_zones = px.bar(zone_dist, orientation='h', title="Time in Zones")
                st.plotly_chart(fig_zones, use_container_width=True)
                
            else:
                st.warning("No Power data found in this ride's streams.")
        else:
            st.error("Could not fetch details for this ride.")

# TAB 3: DATA TABLE
with tab3:
    st.subheader("All Rides (Summary)")
    st.dataframe(
        df[['start_date_local', 'name', 'distance_miles', 'average_watts', 'average_heartrate', 'efficiency_factor', 'tss_score', 'IF', 'kilojoules']]
        .sort_values('start_date_local', ascending=False)
        .style.format("{:.1f}", subset=['distance_miles', 'average_watts', 'tss_score', 'efficiency_factor'])
    )
