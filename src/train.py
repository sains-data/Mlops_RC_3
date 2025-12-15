import pandas as pd
import yaml
import joblib
import mlflow
import mlflow.sklearn
import optuna
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, accuracy_score, f1_score

# --- KONFIGURASI ---
CONFIG_PATH = Path("configs/data.yaml")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# --- FUNGSI OBJECTIVE (INTI OPTUNA) ---
# --- FUNGSI OBJECTIVE (UPDATED FOR LOGGING) ---
def objective(trial):
    # 1. Start MLflow Run (Nested = True agar rapi)
    with mlflow.start_run(nested=True):
        
        # 2. Load Data & Split
        cfg = load_config(CONFIG_PATH)
        data_path = Path(cfg["output_dir"]) / "train_final.csv"
        df = pd.read_csv(data_path)
        X = df.drop(columns=['label'])
        y = df['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 3. Suggest Hyperparameters
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 400),
            'max_depth': trial.suggest_int('max_depth', 10, 50),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 15),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2'])
        }
        
        # 4. Train Model
        model = RandomForestClassifier(**param, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # 5. Evaluate
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        
        # 6. LOGGING KE MLFLOW (Ini bagian barunya)
        # Kita catat parameter yang dipilih Optuna saat ini
        mlflow.log_params(param)
        # Kita catat hasilnya
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1", f1)
        
        # Kita beri tag agar mudah dicari bahwa ini adalah "trial"
        mlflow.set_tag("type", "optuna_trial")
        mlflow.set_tag("trial_number", trial.number)
        
        return f1

def main():
    print("🚀 Memulai Hyperparameter Tuning (Optuna) untuk FD002...")
    print("   Proses ini akan memakan waktu karena data lebih besar...")
    
    # 1. Setup MLflow
    mlflow.set_experiment("Predictive_Maintenance_FD002_Tuning")
    
    # 2. Jalan Optuna
    # Kita coba 20 kali percobaan (trials)
    study = optuna.create_study(direction='maximize', study_name="RF_FD002_Optimization")
    study.optimize(objective, n_trials=20) 
    
    print("\n🏁 Tuning Selesai!")
    print(f"✅ Best F1-Score: {study.best_value:.4f}")
    print(f"✅ Best Params: {study.best_params}")
    
    # --- RETRAIN MODEL TERBAIK ---
    print("\n💾 Menyimpan Model Pemenang...")
    
    with mlflow.start_run(run_name="Optuna_Best_Model_FD002"):
        # Load ulang data
        cfg = load_config(CONFIG_PATH)
        df = pd.read_csv(Path(cfg["output_dir"]) / "train_final.csv")
        X = df.drop(columns=['label'])
        y = df['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train pakai Best Params
        best_params = study.best_params
        best_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
        best_model.fit(X_train, y_train)
        
        # Final Metrics
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"   📊 Final Metrics -> Accuracy: {acc:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
        
        # Logging
        mlflow.log_params(best_params)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1", f1)
        mlflow.sklearn.log_model(best_model, "model")
        
        # Save Local
        joblib.dump(best_model, MODEL_DIR / "best_model.pkl")
        print(f"🏆 Model terbaik disimpan di: {MODEL_DIR / 'best_model.pkl'}")

if __name__ == "__main__":
    main()