import time
import pandas as pd
import joblib
import yaml
import sys
from pathlib import Path

# --- KONFIGURASI ---
CONFIG_PATH = Path("configs/data.yaml")
MODEL_DIR = Path("models")

def load_artifacts():
    print("ep Memuat Model & Scaler dari Artifact Store...")
    
    # 1. Load Config
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    
    # 2. Load Model Terbaik
    model_path = MODEL_DIR / "best_model.pkl"
    if not model_path.exists():
        sys.exit("❌ Error: best_model.pkl tidak ditemukan. Jalankan training dulu.")
    model = joblib.load(model_path)
    
    # 3. Load Scaler (PENTING: Agar standar matematikanya sama)
    scaler_path = MODEL_DIR / "scaler.pkl"
    if not scaler_path.exists():
        sys.exit("❌ Error: scaler.pkl tidak ditemukan. Jalankan preprocessing dulu.")
    scaler = joblib.load(scaler_path)
    
    print("✅ Sistem Siap. Menunggu data sensor...")
    return model, scaler, cfg

def preprocess_single_row(row, scaler, features):
    """
    Mengubah satu baris data mentah menjadi format yang dimengerti model.
    """
    # Ambil hanya fitur yang dipakai saat training
    # Kita ubah jadi DataFrame agar nama kolomnya cocok dengan scaler
    df_single = pd.DataFrame([row], columns=features)
    
    # Lakukan Scaling
    scaled_values = scaler.transform(df_single)
    
    return scaled_values

def start_streaming_inference():
    # Load semua kebutuhan
    model, scaler, cfg = load_artifacts()
    features = cfg["selected_features"]
    
    # Load Data 'Masa Depan' untuk simulasi
    stream_data_path = Path(cfg["output_dir"]) / "streaming_source.csv"
    if not stream_data_path.exists():
        sys.exit("❌ Error: Data streaming tidak ditemukan.")
    
    # Simulasi Streaming
    # Kita baca CSV ini, tapi kita proses baris per baris pakai loop
    df_stream = pd.read_csv(stream_data_path)
    
    print("\n🚀 MEMULAI ONLINE INFERENCE (Real-time Monitoring)")
    print("="*60)
    print(f"{'TIMESTAMP':<10} | {'ENGINE ID':<10} | {'PREDIKSI':<15} | {'CONFIDENCE':<10} | {'STATUS'}")
    print("-" * 60)
    
    failure_counter = 0
    
    try:
        # Loop seolah-olah data masuk satu per satu
        for index, row in df_stream.iterrows():
            
            # 1. Ambil data sensor mentah (Fitur saja)
            sensor_data = row[features].to_dict() # Hanya ambil kolom sensor
            
            # 2. Preprocess (Scaling)
            input_vector = preprocess_single_row(sensor_data, scaler, features)
            
            # 3. Predict (Inference)
            prediction = model.predict(input_vector)[0]     # 0 atau 1
            probability = model.predict_proba(input_vector)[0][1] # Seberapa yakin model? (0.0 - 1.0)
            
            # 4. Tampilkan Log Monitoring
            status = "NORMAL 🟢"
            if prediction == 1:
                status = "BAHAYA! 🔴"
                failure_counter += 1
            else:
                failure_counter = 0 # Reset jika normal kembali
                
            # Simulasi timestamp berjalan
            cycle_time = row['time_in_cycles']
            unit_id = int(row['unit_number'])
            
            print(f"Cycle {int(cycle_time):<4} | Unit {unit_id:<4}     | {status:<15} | {probability:.2f}       | {status}")
            
            # 5. MONITORING LOGIC (Simple Rule)
            # Jika terdeteksi bahaya berturut-turut pada mesin yang sama, kirim ALERT KERAS
            if failure_counter >= 3:
                print(f"   ⚠️  [ALERT SYSTEM] Peringatan Dini! Mesin {unit_id} diprediksi RUSAK dalam waktu dekat. Segera maintenance!")
                failure_counter = 0 # Reset alert
                time.sleep(1) # Beri jeda agar Alert terbaca

            # Delay agar terlihat seperti streaming (bisa dipercepat)
            time.sleep(0.1) 
            
    except KeyboardInterrupt:
        print("\n🛑 Monitoring dihentikan user.")

if __name__ == "__main__":
    start_streaming_inference()