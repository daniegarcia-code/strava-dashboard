import streamlit as st
import requests
import pandas as pd
import plotly.express as px

# --- PAGE CONFIG ---
st.set_page_config(page_title="Endurance Lab", page_icon="🚴")

# --- 1. SETUP AUTHENTICATION ---
# Load the application keys from secrets.toml
CLIENT_ID = st.secrets["strava_client_id"]
CLIENT_SECRET = st.secrets["strava_client_secret"]

# IMPORTANT: This URL must match your deployed app URL exactly.
# For local testing use: "http://localhost:8501"
# For Cloud use: "https://your-app-name.streamlit.app"
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

def fetch_activities(access_token, num_activities=50):
    url = "https://www.strava.com/api/v3/athlete/activities"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {"per_page": num_activities}
    
    r = requests.get(url, headers=headers, params=params)
    if r.status_code == 200:
        return r.json()
    return []

# --- 2. AUTHENTICATION LOGIC ---

# A. Check if user is already logged in (session state)
if "access_token" not in st.session_state:
    
    # B. Check if Strava just redirected them back with a 'code'
    # Streamlit now uses st.query_params for URL parameters
    if "code" in st.query_params:
        code = st.query_params["code"]
        
        # 1. Swap the code for a token
        token_data = exchange_code_for_token(code)
        
        if "access_token" in token_data:
            # Success! Save their identity to the session
            st.session_state["access_token"] = token_data["access_token"]
            st.session_state["athlete_name"] = token_data["athlete"]["firstname"]
            
            # 2. Clear the URL (hide the ugly code) and reload
            st.query_params.clear()
            st.rerun()
        else:
            st.error("Login failed. Strava didn't accept the code.")
            st.write(token_data) # Print error for debugging
    
    else:
        # C. Show the Login Button (The "Gate")
        st.title("🚴 Endurance Lab")
        st.write("## Analyze Your Training")
        st.markdown("Connect your Strava account to view your dashboard.")
        
        # The Button
        st.link_button("Connect with Strava", get_auth_url())
        
        # Stop the app here. Don't run the dashboard code below.
        st.stop() 

# --- 3. THE DASHBOARD (Only runs if logged in) ---
st.sidebar.write(f"Logged in as: **{st.session_state['athlete_name']}**")
if st.sidebar.button("Log Out"):
    st.session_state.clear()
    st.rerun()

st.title(f"Welcome, {st.session_state['athlete_name']}!")

# Fetch Data Live
with st.spinner("Downloading your rides..."):
    data = fetch_activities(st.session_state["access_token"])

if not data:
    st.warning("No rides found!")
    st.stop()

# Convert to DataFrame
df = pd.json_normalize(data)
df['start_date_local'] = pd.to_datetime(df['start_date_local'])
df['distance_miles'] = df['distance'] / 1609.34
df['elevation_ft'] = df['total_elevation_gain'] * 3.28084

# VISUALS
col1, col2, col3 = st.columns(3)
col1.metric("Recent Rides", len(df))
col2.metric("Total Miles", f"{df['distance_miles'].sum():.1f} mi")
col3.metric("Elevation", f"{df['elevation_ft'].sum():,.0f} ft")

# Chart
fig = px.bar(df, x='start_date_local', y='distance_miles', title="Recent Volume")
st.plotly_chart(fig)

st.dataframe(df[['name', 'start_date_local', 'distance_miles', 'average_watts']])



