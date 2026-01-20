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
    
    df['IF'] = df
