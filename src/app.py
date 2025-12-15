import streamlit as st
import pandas as pd
import joblib
import yaml
import sqlite3
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime

# --- CONFIG PAGE ---
st.set_page_config(page_title="Mission Control Dashboard", layout="wide")

# --- 1. LOAD ASSETS ---
@st.cache_resource
def load_assets():
    with open("configs/data.yaml", "r") as f:
        config = yaml.safe_load(f)
    model = joblib.load("models/best_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return config, model, scaler

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/streaming_source.csv")

# --- 2. DATABASE ---
def init_db():
    conn = sqlite3.connect('inference_logs.db', check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS prediction_logs 
                 (id INTEGER PRIMARY KEY, timestamp TEXT, unit_id INTEGER, cycle INTEGER, 
                 prediction INTEGER, probability REAL, status TEXT)''')
    conn.commit()
    return conn

def log_to_db(conn, unit_id, cycle, pred, prob):
    c = conn.cursor()
    status = "CRITICAL" if pred == 1 else "NORMAL"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO prediction_logs (timestamp, unit_id, cycle, prediction, probability, status) VALUES (?, ?, ?, ?, ?, ?)", 
              (timestamp, unit_id, cycle, pred, prob, status))
    conn.commit()

# --- 3. MAIN APP ---
def main():
    config, model, scaler = load_assets()
    df_stream = load_data()
    conn = init_db()

    # --- STATE MANAGEMENT (Otak dari Logika Baru) ---
    # Status Simulasi: 'IDLE', 'RUNNING', 'PAUSED'
    if "sim_state" not in st.session_state:
        st.session_state.sim_state = "IDLE"
    
    # Data Persisten
    if "chart_data" not in st.session_state:
        st.session_state.chart_data = pd.DataFrame(columns=['Cycle', 'Sensor_11', 'Sensor_4', 'Sensor_9'])
    if "logs_data" not in st.session_state:
        st.session_state.logs_data = []
        
    # Penanda Posisi Data (Agar bisa Resume)
    if "current_index" not in st.session_state:
        st.session_state.current_index = 0

    # --- SIDEBAR ---
    st.sidebar.title("🎛️ Flight Control")
    available_units = df_stream['unit_number'].unique()
    selected_engine = st.sidebar.selectbox("Select Engine Unit", available_units)
    speed = st.sidebar.slider("Simulation Speed", 0.05, 1.0, 0.1)
    
    st.sidebar.divider()
    
    # --- LOGIKA TOMBOL CANGGIH ---
    col_btn1, col_btn2 = st.sidebar.columns(2)
    
    # 1. TOMBOL START (RESET OTOMATIS)
    # Logika: Selalu me-reset state ke awal dan mengubah status jadi RUNNING
    if col_btn1.button("▶️ Start / Reset", type="primary", use_container_width=True):
        st.session_state.sim_state = "RUNNING"
        st.session_state.chart_data = pd.DataFrame(columns=['Cycle', 'Sensor_11', 'Sensor_4', 'Sensor_9'])
        st.session_state.logs_data = []
        st.session_state.current_index = 0
        st.rerun()

    # 2. TOMBOL JEDA / LANJUT (TOGGLE)
    # Logika: Muncul beda tulisan tergantung status saat ini
    if st.session_state.sim_state == "RUNNING":
        if col_btn2.button("⏸️ Jeda (Pause)", use_container_width=True):
            st.session_state.sim_state = "PAUSED"
            st.rerun()
            
    elif st.session_state.sim_state == "PAUSED":
        if col_btn2.button("▶️ Lanjut (Resume)", use_container_width=True):
            st.session_state.sim_state = "RUNNING"
            st.rerun()
            
    else: # IDLE
        col_btn2.button("⏹️ Berhenti", disabled=True, use_container_width=True)

    # --- MAIN CONTENT ---
    st.title("🚀 Engine Telemetry System")
    st.caption(f"Unit: {selected_engine} | Status: **{st.session_state.sim_state}**")
    st.divider()

    # Metrics Layout
    col1, col2, col3, col4 = st.columns(4)
    with col1: metric_status = st.empty()
    with col2: metric_prob = st.empty()
    with col3: metric_s4 = st.empty()
    with col4: metric_rpm = st.empty()

    # Default Metrics Display
    metric_status.info("Ready")
    metric_prob.metric("Risk Prob", "-")
    metric_s4.metric("EGT", "-")
    metric_rpm.metric("RPM", "-")

    # --- TABS LOGIC (Permintaan Kedua: Tab History) ---
    tab_chart, tab_history = st.tabs(["📈 Real-time Telemetry", "📜 Riwayat Simulasi (History)"])

    # --- TAB 1: GRAFIK ---
    with tab_chart:
        chart_placeholder = st.empty()
        
        # Fungsi helper untuk gambar grafik
        def draw_chart():
            if st.session_state.chart_data.empty:
                # Grafik Kosong
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                                    subplot_titles=("Sensor 11 (Pressure)", "Sensor 4 (Temp)", "Sensor 9 (RPM)"))
                fig.update_layout(height=500, title_text="Waiting for Start...")
            else:
                # Grafik Isi
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                                    subplot_titles=("S11: Pressure (psia)", "S4: Temp (°R)", "S9: Speed (rpm)"))
                
                fig.add_trace(go.Scatter(x=st.session_state.chart_data['Cycle'], y=st.session_state.chart_data['Sensor_11'],
                                         mode='lines', name='Pressure', line=dict(color='#00CC96')), row=1, col=1)
                fig.add_trace(go.Scatter(x=st.session_state.chart_data['Cycle'], y=st.session_state.chart_data['Sensor_4'],
                                         mode='lines', name='Temp', line=dict(color='#FFA15A')), row=2, col=1)
                fig.add_trace(go.Scatter(x=st.session_state.chart_data['Cycle'], y=st.session_state.chart_data['Sensor_9'],
                                         mode='lines', name='RPM', line=dict(color='#AB63FA')), row=3, col=1)
                
                fig.update_layout(height=600, showlegend=False, margin=dict(t=30, b=30))
            
            chart_placeholder.plotly_chart(fig, use_container_width=True)

        # Gambar grafik awal (sebelum loop atau saat pause)
        draw_chart()

    # --- TAB 2: HISTORY ---
    with tab_history:
        col_hist_1, col_hist_2 = st.columns([4, 1])
        with col_hist_1:
            st.markdown("##### Log Data Simulasi")
        with col_hist_2:
            # Tombol Hapus History
            if st.button("🗑️ Hapus History"):
                st.session_state.logs_data = []
                st.session_state.chart_data = pd.DataFrame(columns=['Cycle', 'Sensor_11', 'Sensor_4', 'Sensor_9'])
                c = conn.cursor()
                c.execute("DELETE FROM prediction_logs")
                conn.commit()
                st.rerun()

        history_placeholder = st.empty()
        if len(st.session_state.logs_data) > 0:
            df_hist = pd.DataFrame(st.session_state.logs_data)
            history_placeholder.dataframe(df_hist.sort_index(ascending=False), use_container_width=True)
        else:
            history_placeholder.info("Belum ada data history.")

    # --- CORE SIMULATION LOOP ---
    # Hanya jalan jika status RUNNING
    if st.session_state.sim_state == "RUNNING":
        engine_data = df_stream[df_stream['unit_number'] == selected_engine]
        
        # Kunci Logika Resume: Kita mulai loop dari 'current_index'
        # Bukan dari 0 lagi.
        data_to_stream = engine_data.iloc[st.session_state.current_index:]
        
        for i, row in data_to_stream.iterrows():
            # Cek jika user menekan Pause di tengah loop
            if st.session_state.sim_state != "RUNNING": 
                break
            
            # Update index agar nanti bisa resume dari sini
            st.session_state.current_index += 1
            
            # 1. PREDIKSI
            features = config['selected_features']
            input_data = row[features].values.reshape(1, -1)
            input_scaled = scaler.transform(input_data)
            
            pred = model.predict(input_scaled)[0]
            prob = model.predict_proba(input_scaled)[0][1]
            cycle = int(row['time_in_cycles'])
            
            # 2. UPDATE STATE (Chart)
            new_row = pd.DataFrame({
                'Cycle': [cycle], 'Sensor_11': [row['sensor_11']],
                'Sensor_4': [row['sensor_4']], 'Sensor_9': [row['sensor_9']]
            })
            st.session_state.chart_data = pd.concat([st.session_state.chart_data, new_row], ignore_index=True)
            
            # 3. UPDATE STATE (History Logs)
            status_txt = "CRITICAL" if pred == 1 else "NORMAL"
            log_entry = {
                "Time": datetime.now().strftime("%H:%M:%S"),
                "Cycle": cycle,
                "Prediction": status_txt,
                "Probability": f"{prob:.2%}",
                "S11 (Press)": f"{row['sensor_11']:.2f}",
                "S4 (Temp)": f"{row['sensor_4']:.1f}",
                "S9 (RPM)": f"{row['sensor_9']:.0f}"
            }
            st.session_state.logs_data.append(log_entry)

            # 4. RENDER UI
            # Update Metrics
            if pred == 1:
                metric_status.error(f"⚠️ FAILURE (Cycle {cycle})")
            else:
                metric_status.success(f"✅ NORMAL (Cycle {cycle})")
            metric_prob.metric("Risk Prob", f"{prob:.2%}")
            metric_s4.metric("EGT", f"{row['sensor_4']:.1f}")
            metric_rpm.metric("RPM", f"{row['sensor_9']:.0f}")

            # Update Chart di Tab 1
            with tab_chart:
                draw_chart()
            
            # Update Table di Tab 2 (Realtime, update setiap 5 cycle biar gak berat)
            if cycle % 5 == 0:
                with tab_history:
                    df_hist = pd.DataFrame(st.session_state.logs_data)
                    history_placeholder.dataframe(df_hist.sort_index(ascending=False), use_container_width=True)

            # 5. SIMPAN DB
            log_to_db(conn, selected_engine, cycle, pred, prob)
            
            time.sleep(speed)

    # Indikator saat Pause
    if st.session_state.sim_state == "PAUSED":
        metric_status.warning("⏸️ SIMULASI DIJEDA")

if __name__ == "__main__":
    main()