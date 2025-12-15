import streamlit as st
import pandas as pd
import joblib
import yaml
import time
import sqlite3
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="⚙️",
    layout="wide"
)

# --- FUNGSI DATABASE (SQLite) ---
def init_db():
    """Membuat database dan tabel log jika belum ada"""
    # check_same_thread=False dibutuhkan untuk Streamlit
    conn = sqlite3.connect('inference_logs.db', check_same_thread=False)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS prediction_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            unit_id INTEGER,
            cycle INTEGER,
            sensor_11 REAL,
            sensor_12 REAL,
            prediction INTEGER,
            probability REAL,
            status TEXT
        )
    ''')
    conn.commit()
    return conn

def log_to_db(conn, unit_id, cycle, s11, s12, pred, prob):
    """Menyimpan satu baris log prediksi ke database"""
    c = conn.cursor()
    status = "CRITICAL" if pred == 1 else "NORMAL"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    c.execute('''
        INSERT INTO prediction_logs (timestamp, unit_id, cycle, sensor_11, sensor_12, prediction, probability, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (timestamp, unit_id, cycle, s11, s12, int(pred), prob, status))
    
    conn.commit()

# --- FUNGSI UTAMA (CACHED) ---
@st.cache_resource
def load_artifacts():
    """Memuat Model, Scaler, dan Config agar tidak berat saat reload"""
    
    # Paths
    config_path = Path("configs/data.yaml")
    model_path = Path("models/best_model.pkl")
    scaler_path = Path("models/scaler.pkl")
    data_path = Path("data/processed/streaming_source.csv")

    # 1. Load Config
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    # 2. Load Model & Scaler
    if not model_path.exists() or not scaler_path.exists():
        st.error("❌ Model/Scaler belum ada. Jalankan training dulu!")
        return None, None, None, None
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # 3. Load Data Streaming (Simulasi)
    if not data_path.exists():
        st.error("❌ Data streaming tidak ditemukan.")
        return None, None, None, None
    
    df_stream = pd.read_csv(data_path)
    
    return model, scaler, cfg, df_stream

def preprocess_input(row, scaler, features):
    """Scaling data input agar sesuai dengan data training"""
    # Pastikan mengambil hanya kolom features yang sesuai
    sensor_vals = {k: row[k] for k in features if k in row}
    df_single = pd.DataFrame([sensor_vals], columns=features)
    return scaler.transform(df_single)

# --- UI LAYOUT ---
def main():
    # 1. Init Database
    conn = init_db()

    # 2. Load System
    model, scaler, cfg, df_stream = load_artifacts()
    
    if model is None:
        return

    # --- SIDEBAR ---
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/1541/1541425.png", width=100)
    st.sidebar.title("🎛️ Control Panel")
    
    # Pilih Mesin
    available_engines = df_stream['unit_number'].unique()
    selected_engine = st.sidebar.selectbox("Pilih ID Mesin (Unit)", available_engines)
    
    # Filter data berdasarkan mesin
    engine_data = df_stream[df_stream['unit_number'] == selected_engine]
    
    # Kecepatan Simulasi
    speed = st.sidebar.slider("Kecepatan Simulasi (detik)", 0.05, 1.0, 0.1)
    
    # Checkbox Demo Mode
    demo_mode = st.sidebar.checkbox("⚡ Demo Mode (Jump to Failure)", value=True, 
                                    help="Langsung mulai dari 75 cycle terakhir sebelum rusak agar tidak menunggu lama.")

    # Tombol Start/Stop
    start_btn = st.sidebar.button("▶️ Mulai Monitoring", type="primary")
    stop_btn = st.sidebar.button("⏹️ Berhenti")

    # --- DASHBOARD HEADER ---
    st.title("🏭 Intelligent Predictive Maintenance System")
    st.markdown("Dashboard Real-time MLOps dengan **Inference Logging** dan **Dual-Axis Monitoring**.")
    
    # --- TABS (MONITORING vs LOGS) ---
    tab1, tab2 = st.tabs(["📡 Live Monitoring", "🗄️ Database Audit Trail"])

    # === TAB 1: LIVE MONITORING ===
    with tab1:
        col1, col2, col3 = st.columns(3)
        with col1: metric_cycle = st.empty()
        with col2: metric_status = st.empty()
        with col3: metric_prob = st.empty()
            
        st.divider()
        
        col_chart, col_alert = st.columns([2, 1])
        with col_chart:
            st.subheader(f"Sensor Telemetry - Unit {selected_engine}")
            chart_placeholder = st.empty()
        with col_alert:
            st.subheader("System Alerts")
            alert_placeholder = st.empty()

        # --- LOGIKA STREAMING ---
        if "streaming" not in st.session_state:
            st.session_state.streaming = False

        if start_btn: st.session_state.streaming = True
        if stop_btn: st.session_state.streaming = False

        if st.session_state.streaming:
            chart_data = pd.DataFrame(columns=['Cycle', 'Sensor_11', 'Sensor_12'])
            
            # Trik Demo (Jump Start)
            if demo_mode and len(engine_data) > 75:
                 engine_data = engine_data.tail(75) 
            
            # Loop Data Streaming
            for index, row in engine_data.iterrows():
                if not st.session_state.streaming:
                    break
                    
                # 1. Preprocess & Predict
                features = cfg['selected_features']
                input_vector = preprocess_input(row, scaler, features)
                
                pred = model.predict(input_vector)[0]
                prob = model.predict_proba(input_vector)[0][1]
                cycle_val = int(row['time_in_cycles'])
                
                # 2. Update Metrics UI
                metric_cycle.metric("Current Cycle", f"{cycle_val}")
                metric_prob.metric("Failure Probability", f"{prob:.1%}")
                
                if pred == 1:
                    metric_status.error("STATUS: CRITICAL")
                    alert_placeholder.warning(f"🚨 **ALERT:** Anomaly at Cycle {cycle_val}! Prob: {prob:.2%}")
                else:
                    metric_status.success("STATUS: NORMAL")
                    alert_placeholder.info("System Normal.")

                # 3. Log ke Database (Inference Logging)
                log_to_db(conn, selected_engine, cycle_val, 
                          row['sensor_11'], row['sensor_12'], pred, prob)
                
                # 4. Update Grafik (Dual Axis)
                new_row = pd.DataFrame({
                    'Cycle': [cycle_val],
                    'Sensor_11': [row['sensor_11']],
                    'Sensor_12': [row['sensor_12']]
                })
                
                chart_data = pd.concat([chart_data, new_row], ignore_index=True)
                
                # Kita tampilkan semua riwayat (tanpa .tail) agar grafik memanjang
                chart_view = chart_data 
                
                fig = make_subplots(specs=[[{"secondary_y": True}]])

                # Sumbu Kiri (Pressure - Biru)
                fig.add_trace(
                    go.Scatter(x=chart_view['Cycle'], y=chart_view['Sensor_11'], 
                               name='Pressure (S11)', line=dict(color='#29b5e8', width=2)),
                    secondary_y=False,
                )
                # Sumbu Kanan (Vibration - Merah)
                fig.add_trace(
                    go.Scatter(x=chart_view['Cycle'], y=chart_view['Sensor_12'], 
                               name='Vibration (S12)', line=dict(color='#ff4b4b', width=2)),
                    secondary_y=True,
                )

                fig.update_layout(
                    height=400,
                    margin=dict(l=10, r=10, t=10, b=10),
                    legend=dict(orientation="h", y=1.1),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                
                # Label Sumbu
                fig.update_yaxes(title_text="Pressure", secondary_y=False, showgrid=True, gridcolor='lightgray')
                fig.update_yaxes(title_text="Vibration", secondary_y=True, showgrid=False)
                
                chart_placeholder.plotly_chart(fig, use_container_width=True)
                
                time.sleep(speed)
        
        else:
            if not start_btn:
                st.info("👈 Tekan tombol 'Mulai Monitoring' untuk simulasi.")

    # === TAB 2: DATABASE LOGS ===
    with tab2:
        st.subheader("🗄️ Audit Trail Data (Stored in SQLite)")
        st.markdown("Data ini disimpan otomatis untuk keperluan **Audit** dan **Future Retraining**.")
        
        col_db_btn, col_db_space = st.columns([1, 4])
        with col_db_btn:
            refresh_db = st.button("🔄 Refresh Logs")
            
        # Tampilkan Data dari DB
        try:
            df_logs = pd.read_sql_query("SELECT * FROM prediction_logs ORDER BY id DESC", conn)
            st.dataframe(df_logs, use_container_width=True)
            
            # Tombol Download CSV
            csv = df_logs.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Logs as CSV",
                csv,
                "inference_logs.csv",
                "text/csv",
                key='download-csv'
            )
        except Exception as e:
            st.warning("Belum ada data log. Jalankan monitoring terlebih dahulu.")

if __name__ == "__main__":
    main()