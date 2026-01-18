import streamlit as st
import requests
import pandas as pd
import plotly.express as px

# --- PAGE CONFIG ---
st.set_page_config(page_title="Endurance Lab", page_icon="🚴")

# --- 1. SETUP AUTHENTICATION ---
# We get these secrets from the Cloud Config (later)
CLIENT_ID = st.secrets["strava"]["client_id"]
CLIENT_SECRET = st.secrets["strava"]["client_secret"]
# IMPORTANT: This URL must match what you set in Strava Settings later
REDIRECT_URI = "https://your-app-name.streamlit.app" 

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
    
    all_data = []
    # Fetch just one page for speed in this demo
    r = requests.get(url, headers=headers, params=params)
    if r.status_code == 200:
        return r.json()
    return []

# --- 2. AUTHENTICATION LOGIC ---
# A. Check if we are already logged in
if "access_token" not in st.session_state:
    
    # B. Check if Strava just sent us back with a code
    query_params = st.query_params
    
    if "code" in query_params:
        # 1. Swap the code for a token
        code = query_params["code"]
        token_data = exchange_code_for_token(code)
        
        if "access_token" in token_data:
            st.session_state["access_token"] = token_data["access_token"]
            st.session_state["athlete_name"] = token_data["athlete"]["firstname"]
            
            # 2. Clear the URL (hide the code)
            st.query_params.clear()
            st.rerun()
        else:
            st.error("Login failed. Please try again.")
    
    else:
        # C. Show the Login Button
        st.title("🚴 Endurance Lab Login")
        st.write("Connect your Strava account to analyze your training.")
        st.link_button("Connect with Strava", get_auth_url())
        st.stop() # Stop here, don't show the dashboard yet

# --- 3. THE DASHBOARD (Only runs if logged in) ---
st.title(f"Welcome, {st.session_state['athlete_name']}!")

if st.button("Log Out"):
    st.session_state.clear()
    st.rerun()

# Fetch Data Live
with st.spinner("Fetching your latest rides..."):
    data = fetch_activities(st.session_state["access_token"])

if not data:
    st.warning("No rides found!")
    st.stop()

# Convert to DataFrame
df = pd.json_normalize(data)
df['start_date_local'] = pd.to_datetime(df['start_date_local'])
df['distance_miles'] = df['distance'] / 1609.34

# VISUALS
col1, col2 = st.columns(2)
col1.metric("Recent Rides", len(df))
col1.metric("Total Miles", f"{df['distance_miles'].sum():.1f} mi")

# Chart
fig = px.bar(df, x='start_date_local', y='distance_miles', title="Recent Volume")
st.plotly_chart(fig)

st.dataframe(df[['name', 'start_date_local', 'distance_miles', 'average_watts']])