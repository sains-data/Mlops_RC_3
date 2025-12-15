import pandas as pd
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split

# --- KONFIGURASI ---
CONFIG_PATH = Path("configs/data.yaml")

def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ingest_data():
    print("🚀 Memulai Data Ingestion untuk FD002...")
    cfg = load_config(CONFIG_PATH)
    
    # 1. Setup Path
    raw_path = Path(cfg["raw_data_path"])
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not raw_path.exists():
        raise FileNotFoundError(f"File dataset tidak ditemukan di: {raw_path}")

    # 2. Definisikan Nama Kolom (Standar NASA CMAPSS)
    # FD002 memiliki struktur kolom yang sama persis dengan FD001
    col_names = [
        "unit_number", "time_in_cycles", 
        "op_setting_1", "op_setting_2", "op_setting_3",
        "sensor_1", "sensor_2", "sensor_3", "sensor_4", "sensor_5",
        "sensor_6", "sensor_7", "sensor_8", "sensor_9", "sensor_10",
        "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
        "sensor_16", "sensor_17", "sensor_18", "sensor_19", "sensor_20", "sensor_21"
    ]
    
    # 3. Baca Data
    print(f"   Membaca file: {raw_path} ...")
    df = pd.read_csv(raw_path, sep=r"\s+", header=None, names=col_names)
    print(f"   Total Data Mentah: {df.shape}")
    
    # 4. Generate Label (RUL)
    # Kita hitung RUL mundur (Max Cycle - Current Cycle)
    # Ini berlaku untuk data Train NASA karena semua mesin berjalan sampai rusak (Run-to-Failure)
    print("   Generating RUL (Remaining Useful Life)...")
    max_cycles = df.groupby('unit_number')['time_in_cycles'].transform('max')
    df['RUL'] = max_cycles - df['time_in_cycles']
    
    # Buat Target Label (Binary Classification)
    # 1 = Bahaya (RUL <= 30), 0 = Aman
    threshold = cfg['rul_threshold']
    df['label'] = (df['RUL'] <= threshold).astype(int)
    
    # 5. Split Data: Training vs Streaming Simulation
    # Kita pisahkan mesin berdasarkan ID
    split_id = cfg['test_split_engine_id']
    
    df_train = df[df['unit_number'] <= split_id].copy()
    df_stream = df[df['unit_number'] > split_id].copy()
    
    print(f"   Split Data: Unit 1-{split_id} untuk Training, Unit {split_id+1}+ untuk Streaming")
    print(f"   Data Train: {df_train.shape}")
    print(f"   Data Stream: {df_stream.shape}")
    
    # 6. Simpan Data
    train_save_path = output_dir / "ingested_train.csv"
    stream_save_path = output_dir / "streaming_source.csv"
    
    df_train.to_csv(train_save_path, index=False)
    df_stream.to_csv(stream_save_path, index=False)
    
    print(f"✅ Data tersimpan di '{output_dir}'")

if __name__ == "__main__":
    ingest_data()