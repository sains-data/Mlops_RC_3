import pandas as pd
import yaml
import joblib
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

# --- KONFIGURASI ---
CONFIG_PATH = Path("configs/data.yaml")

def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_preprocessing():
    print("🚀 Memulai Preprocessing untuk FD002...")
    
    # 1. Load Config & Data
    cfg = load_config(CONFIG_PATH)
    input_path = Path(cfg["output_dir"]) / "ingested_train.csv"
    output_dir = Path(cfg["output_dir"])
    model_dir = Path(cfg["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        raise FileNotFoundError("File ingested_train.csv tidak ditemukan. Jalankan data_ingestion.py dulu.")
        
    df = pd.read_csv(input_path)
    
    # 2. Filter Fitur
    # Kita mengambil kolom sensor DAN op_setting sesuai data.yaml
    features = cfg["selected_features"]
    target = "label"
    
    print(f"   Fitur yang digunakan ({len(features)}): {features}")
    
    # Pastikan kolom ada
    missing_cols = [col for col in features if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Kolom berikut hilang di CSV: {missing_cols}")
        
    X = df[features]
    y = df[target]
    
    # 3. Scaling (Normalisasi MinMax)
    # PENTING: Untuk FD002, op_setting juga perlu di-scale agar range-nya 0-1 
    # sama seperti sensor lainnya. Ini membantu model konvergen lebih cepat.
    print("   Melakukan Scaling Data (0-1)...")
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Kembalikan ke DataFrame agar nama kolom tidak hilang
    df_processed = pd.DataFrame(X_scaled, columns=features)
    df_processed[target] = y
    
    # 4. Simpan Data & Scaler
    # Simpan Scaler (PENTING untuk tahap Serving nanti!)
    scaler_path = model_dir / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"✅ Scaler disimpan di: {scaler_path}")
    
    # Simpan Data Training Final
    output_path = output_dir / "train_final.csv"
    df_processed.to_csv(output_path, index=False)
    print(f"✅ Data bersih disimpan di: {output_path}")
    print(f"   Shape Akhir: {df_processed.shape}")

if __name__ == "__main__":
    run_preprocessing()