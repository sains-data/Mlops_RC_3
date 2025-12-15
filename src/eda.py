import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml

# --- KONFIGURASI ---
INPUT_FILE = Path("data/processed/initial_training_data.csv")
OUTPUT_DIR = Path("reports/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Data tidak ditemukan di {path}. Jalankan data_ingestion.py dulu.")
    return pd.read_csv(path)

def plot_sensor_behavior(df: pd.DataFrame, sensor_col: str, unit_ids: list):
    """
    Visualisasi 1: Membuktikan degradasi mesin.
    Kita plot nilai sensor dari awal sampai mesin mati (Cycle terakhir).
    """
    plt.figure(figsize=(12, 6))
    
    for uid in unit_ids:
        subset = df[df['unit_number'] == uid]
        # Plot sensor value
        plt.plot(subset['time_in_cycles'], subset[sensor_col], label=f'Engine {uid}')
        
        # Tandai titik failure (ketika label berubah jadi 1/Bahaya)
        failure_start = subset[subset['label'] == 1]['time_in_cycles'].min()
        if not pd.isna(failure_start):
            plt.axvline(failure_start, color='red', linestyle='--', alpha=0.3)

    plt.title(f"Degradasi {sensor_col} Seiring Waktu (Garis Merah = Mulai Fase Bahaya)")
    plt.xlabel("Waktu (Cycles)")
    plt.ylabel(f"Nilai {sensor_col}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = OUTPUT_DIR / f"eda_behavior_{sensor_col}.png"
    plt.savefig(save_path)
    print(f"✅ Saved behavior plot: {save_path}")
    plt.close()

def plot_correlation_heatmap(df: pd.DataFrame):
    """
    Visualisasi 2: Feature Selection.
    Mencari sensor mana yang paling berkorelasi dengan 'RUL' (Sisa Umur).
    """
    # Ambil kolom sensor dan RUL saja
    sensor_cols = [c for c in df.columns if 'sensor' in c]
    cols_to_check = sensor_cols + ['RUL']
    
    corr_matrix = df[cols_to_check].corr()
    
    plt.figure(figsize=(15, 12))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
    plt.title("Korelasi Antar Sensor dan RUL (Sisa Umur)")
    
    save_path = OUTPUT_DIR / "eda_correlation_heatmap.png"
    plt.savefig(save_path)
    print(f"✅ Saved correlation heatmap: {save_path}")
    plt.close()
    
    # Print korelasi tertinggi dengan RUL untuk rekomendasi fitur
    print("\n🔍 Top 5 Sensor Paling Berkorelasi dengan RUL (Indikator Kerusakan Terbaik):")
    rul_corr = corr_matrix['RUL'].abs().sort_values(ascending=False)
    print(rul_corr.head(6)) # Top 5 + RUL itself

def plot_label_distribution(df: pd.DataFrame):
    """
    Visualisasi 3: Cek Imbalance Data.
    Apakah data 'Aman' jauh lebih banyak dari 'Rusak'?
    """
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='label', palette='viridis')
    plt.title("Distribusi Label Target (0=Aman, 1=Bahaya)")
    plt.xlabel("Label Status")
    plt.ylabel("Jumlah Data Sample")
    
    save_path = OUTPUT_DIR / "eda_label_dist.png"
    plt.savefig(save_path)
    print(f"✅ Saved label distribution: {save_path}")
    plt.close()

def main():
    print("🚀 Memulai EDA Pipeline...")
    df = load_data(INPUT_FILE)
    
    # 1. Cek Pola Kerusakan (Sensor 11 dan 12 biasanya paling sensitif di dataset NASA)
    # Kita ambil sampel 3 mesin pertama
    plot_sensor_behavior(df, 'sensor_11', unit_ids=[1, 2, 3])
    plot_sensor_behavior(df, 'sensor_12', unit_ids=[1, 2, 3])
    
    # 2. Cek Korelasi untuk memilih fitur
    plot_correlation_heatmap(df)
    
    # 3. Cek Balance Data
    plot_label_distribution(df)
    
    print(f"\n🎉 EDA Selesai. Cek folder {OUTPUT_DIR} untuk hasil gambar.")

if __name__ == "__main__":
    main()