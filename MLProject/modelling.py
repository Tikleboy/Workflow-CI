import pandas as pd
import os
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ====================================================
# 1. KONFIGURASI OTOMATIS (DARI GITHUB SECRETS)
# ====================================================
# Kita ambil username dari Environment Variable (yang diset di file YAML)
REPO_OWNER = os.environ.get("Tickleboy")
REPO_NAME = "https://dagshub.com/Tikleboy/Eksperimen_MSML_Stanly" # <--- WAJIB GANTI INI !!!

# Cek apakah script jalan di GitHub Actions atau Lokal
if REPO_OWNER and REPO_NAME:
    print(f"[INFO] Terdeteksi berjalan di CI/CD. Konek ke: {REPO_OWNER}/{REPO_NAME}")
    dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)
else:
    print("[WARN] Tidak ada credentials DagsHub. Pastikan Anda sudah set Secrets di GitHub.")

def train():
    print("=== Memulai Training Model ===")
    
    # 2. Load Data
    try:
        # Pastikan path ini sesuai dengan struktur folder Anda
        df = pd.read_csv('data/ulasan_KAI_clean.csv')
    except FileNotFoundError:
        print("[ERROR] File dataset tidak ditemukan di 'data/ulasan_KAI_clean.csv'")
        return

    # 3. Persiapan Data (Sesuaikan fitur dengan dataset Anda)
    X = df[['thumbsUpCount_scaled']] 
    y = df['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Setup Experiment
    mlflow.set_experiment("CI_Pipeline_Docker")
    mlflow.sklearn.autolog(disable=True) # Matikan autolog agar kita bisa kontrol log
    
    with mlflow.start_run() as run:
        # Training
        rf = RandomForestClassifier(n_estimators=30, max_depth=5, random_state=42)
        rf.fit(X_train, y_train)
        
        # Evaluasi
        preds = rf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"Model Accuracy: {acc}")
        
        # Logging ke DagsHub
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(rf, "model_final")
        
        # ===========================================================
        # 5. BAGIAN PENYELAMAT: SIMPAN RUN ID KE FILE TXT
        # ===========================================================
        # Langkah ini PENTING agar langkah selanjutnya (Build Docker) 
        # tahu model mana yang harus dibungkus.
        run_id = run.info.run_id
        print(f"[INFO] Run ID berhasil didapatkan: {run_id}")
        
        with open("run_id.txt", "w") as f:
            f.write(run_id)
        print("[SUCCESS] Run ID tersimpan di file 'run_id.txt'")

if __name__ == "__main__":
    train()
