import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import plotly.graph_objects as go
import os

# ── Page Config ──
st.set_page_config(
    page_title="IPL Win Predictor",
    page_icon="🏏",
    layout="wide"
)

# ── Load Model & Encoders ──
BASE = os.path.join(os.path.dirname(__file__), '..', 'model')

@st.cache_resource
def load_model():
    with open(f'{BASE}/ipl_model.pkl',  'rb') as f: model      = pickle.load(f)
    with open(f'{BASE}/le_batting.pkl', 'rb') as f: le_batting = pickle.load(f)
    with open(f'{BASE}/le_bowling.pkl', 'rb') as f: le_bowling = pickle.load(f)
    with open(f'{BASE}/le_venue.pkl',   'rb') as f: le_venue   = pickle.load(f)
    with open(f'{BASE}/teams.json',     'r')  as f: teams      = json.load(f)
    with open(f'{BASE}/venues.json',    'r')  as f: venues     = json.load(f)
    return model, le_batting, le_bowling, le_venue, teams, venues

model, le_batting, le_bowling, le_venue, teams, venues = load_model()

# ── Styling ──
st.markdown("""
<style>
    .main { background-color: #0a0a1a; }
    .stMetric { background-color: #1a1a2e; border-radius: 10px; padding: 10px; }
    .win-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 15px; padding: 25px;
        text-align: center; margin: 10px 0;
        border: 1px solid #2E74B5;
    }
    .team-name { font-size: 22px; font-weight: bold; color: #ffffff; }
    .win-pct   { font-size: 48px; font-weight: bold; color: #2E74B5; }
    .win-label { font-size: 14px; color: #888888; }
</style>
""", unsafe_allow_html=True)

# ── Header ──
st.markdown("""
<div style='text-align:center; padding: 20px 0'>
    <h1 style='color:#2E74B5; font-size:42px'>🏏 IPL Win Predictor</h1>
    <p style='color:#888888; font-size:16px'>
        Real-time win probability using ML trained on 17 years of IPL data
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Sidebar — Match Setup ──
st.sidebar.markdown("## 🏟️ Match Setup")
st.sidebar.markdown("---")

batting_team = st.sidebar.selectbox("🏏 Batting Team",  sorted(teams))
bowling_team = st.sidebar.selectbox("🎯 Bowling Team",
    [t for t in sorted(teams) if t != batting_team])
venue        = st.sidebar.selectbox("🏟️ Venue", sorted(venues))

st.sidebar.markdown("---")
st.sidebar.markdown("## 📊 Current Match State")

target       = st.sidebar.number_input("🎯 Target (runs)",        min_value=50,  max_value=300, value=180)
current_over = st.sidebar.slider("⏱️ Current Over",               min_value=1,   max_value=20,  value=10)
runs_scored  = st.sidebar.number_input("🏃 Runs Scored",          min_value=0,   max_value=300, value=80)
wickets_lost = st.sidebar.slider("💀 Wickets Lost",               min_value=0,   max_value=10,  value=2)

# ── Derived Features ──
runs_needed = max(target - runs_scored, 0)
balls_left  = max((20 - current_over) * 6, 0)
crr         = round(runs_scored / current_over, 2) if current_over > 0 else 0
rrr         = round((runs_needed / (balls_left / 6)), 2) if balls_left > 0 else 99

# ── Predict Button ──
st.sidebar.markdown("---")
predict_btn = st.sidebar.button("🔮 Predict Win Probability", use_container_width=True)

# ── Main Content ──
col1, col2, col3, col4 = st.columns(4)
col1.metric("🎯 Target",        f"{target} runs")
col2.metric("🏃 Runs Needed",   f"{runs_needed} runs")
col3.metric("⚡ Required RR",   f"{rrr:.2f}")
col4.metric("📈 Current RR",    f"{crr:.2f}")

st.markdown("---")

if predict_btn:

    # Validate inputs
    if runs_scored >= target:
        st.success(f"🎉 {batting_team} have already won!")
    elif wickets_lost == 10:
        st.error(f"💀 {batting_team} are all out! {bowling_team} win!")
    elif balls_left == 0:
        st.error("⏱️ Match is over!")
    else:
        # Encode inputs
        try:
            bat_enc   = le_batting.transform([batting_team])[0]
            bowl_enc  = le_bowling.transform([bowling_team])[0]
            venue_enc = le_venue.transform([venue])[0]
        except ValueError as e:
            st.error(f"Encoding error: {e}")
            st.stop()

        # Build feature vector
        features = np.array([[
            bat_enc, bowl_enc, venue_enc,
            target, current_over, runs_scored,
            wickets_lost, runs_needed, balls_left,
            crr, rrr
        ]])

        # Predict
        win_prob  = model.predict_proba(features)[0][1]
        lose_prob = 1 - win_prob

        # ── Win Probability Cards ──
        c1, c2 = st.columns(2)

        with c1:
            st.markdown(f"""
            <div class='win-card'>
                <div class='team-name'>🏏 {batting_team}</div>
                <div class='win-pct'>{win_prob*100:.1f}%</div>
                <div class='win-label'>Win Probability</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class='win-card'>
                <div class='team-name'>🎯 {bowling_team}</div>
                <div class='win-pct'>{lose_prob*100:.1f}%</div>
                <div class='win-label'>Win Probability</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Probability Gauge ──
        fig_gauge = go.Figure(go.Indicator(
            mode  = "gauge+number",
            value = win_prob * 100,
            title = {'text': f"{batting_team} Win Probability (%)"},
            gauge = {
                'axis'  : {'range': [0, 100]},
                'bar'   : {'color': '#2E74B5'},
                'steps' : [
                    {'range': [0,  40],  'color': '#ff4444'},
                    {'range': [40, 60],  'color': '#ffaa00'},
                    {'range': [60, 100], 'color': '#44bb44'},
                ],
                'threshold': {
                    'line' : {'color': 'white', 'width': 4},
                    'thickness': 0.75,
                    'value': win_prob * 100
                }
            }
        ))
        fig_gauge.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_gauge, use_container_width=True)

        # ── Match Situation Analysis ──
        st.markdown("### 📊 Match Situation Analysis")

        c1, c2, c3 = st.columns(3)

        # RRR vs CRR
        with c1:
            fig_rr = go.Figure(go.Bar(
                x=['Current RR', 'Required RR'],
                y=[crr, rrr],
                marker_color=['#2E74B5' if crr >= rrr else '#ff4444',
                              '#44bb44' if rrr <= 10 else '#ff4444'],
                text=[f'{crr:.2f}', f'{rrr:.2f}'],
                textposition='auto'
            ))
            fig_rr.update_layout(
                title='Run Rate Analysis',
                height=300,
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_rr, use_container_width=True)

        # Wickets remaining
        with c2:
            wickets_left = 10 - wickets_lost
            fig_wkt = go.Figure(go.Bar(
                x=['Wickets Lost', 'Wickets Left'],
                y=[wickets_lost, wickets_left],
                marker_color=['#ff4444', '#44bb44'],
                text=[wickets_lost, wickets_left],
                textposition='auto'
            ))
            fig_wkt.update_layout(
                title='Wicket Situation',
                height=300,
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_wkt, use_container_width=True)

        # Runs progress
        with c3:
            fig_runs = go.Figure(go.Bar(
                x=['Runs Scored', 'Runs Needed'],
                y=[runs_scored, runs_needed],
                marker_color=['#2E74B5', '#ffaa00'],
                text=[runs_scored, runs_needed],
                textposition='auto'
            ))
            fig_runs.update_layout(
                title='Runs Progress',
                height=300,
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_runs, use_container_width=True)

        # ── Verdict ──
        st.markdown("---")
        if win_prob >= 0.65:
            st.success(f"✅ **Strong Win Prediction** — {batting_team} are in control with {win_prob*100:.1f}% win probability.")
        elif win_prob >= 0.45:
            st.warning(f"⚔️ **Close Contest** — Could go either way. {batting_team} at {win_prob*100:.1f}%.")
        else:
            st.error(f"🚨 **Under Pressure** — {batting_team} need something special. Win probability: {win_prob*100:.1f}%.")

# ── Footer ──
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#555; font-size:13px'>"
    "Built by Shaik Azam Hussain · IPL Win Predictor · "
    "Trained on 17 years of IPL data (2008–2024) · "
    "<a href='https://github.com/azam-1125' style='color:#2E74B5'>GitHub</a>"
    "</div>",
    unsafe_allow_html=True
)