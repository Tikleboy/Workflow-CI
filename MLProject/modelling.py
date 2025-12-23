import pandas as pd
import os
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- KONFIGURASI OTOMATIS DARI GITHUB SECRETS ---
# Kita ambil username dari Environment Variable (diset di GitHub Actions nanti)
REPO_OWNER = os.environ.get("Tickleboy") 
REPO_NAME = "https://dagshub.com/Tikleboy/Eksperimen_MSML_Stanly" # <--- GANTI INI DENGAN NAMA REPO DAGSHUB KAMU!!!

# Cek koneksi
if REPO_OWNER and REPO_NAME:
    print(f"Connecting to DagsHub: {REPO_OWNER}/{REPO_NAME}")
    dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)
else:
    print("Warning: Tidak mendeteksi credentials DagsHub. Script mungkin error jika jalan di CI.")

def train():
    print("=== Starting Training in CI Workflow ===")
    
    # Load Data
    try:
        df = pd.read_csv('data/ulasan_KAI_preprocessingcsv')
    except FileNotFoundError:
        print("Error: Dataset tidak ditemukan!")
        return

    # Fitur Sederhana
    X = df[['thumbsUpCount_scaled']] 
    y = df['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Setup Experiment
    mlflow.set_experiment("CI_Pipeline_Docker")
    mlflow.sklearn.autolog(disable=True) # Matikan autolog
    
    with mlflow.start_run() as run:
        # Training
        rf = RandomForestClassifier(n_estimators=30, max_depth=5)
        rf.fit(X_train, y_train)
        
        # Evaluasi
        preds = rf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"Accuracy: {acc}")
        
        # Logging Manual
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(rf, "model_final")
        
        # --- TRIK RAHASIA: SIMPAN RUN ID ---
        # Kita simpan ID ini ke file txt agar bisa dibaca oleh langkah selanjutnya (Docker Build)
        run_id = run.info.run_id
        with open("run_id.txt", "w") as f:
            f.write(run_id)
        print(f"Run ID {run_id} berhasil disimpan ke run_id.txt")

if __name__ == "__main__":
    train()