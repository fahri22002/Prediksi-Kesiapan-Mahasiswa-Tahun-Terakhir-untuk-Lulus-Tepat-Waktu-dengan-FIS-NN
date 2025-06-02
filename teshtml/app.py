from flask import Flask, render_template, request
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from model import ANFIS

# Inisialisasi Flask app
app = Flask(__name__)

# === Load dan siapkan data training untuk scaler dan encoder ===
df = pd.read_csv("Book1.csv")

# Fitur dan label
features = df[['ipk', 'progress', 'kehadiran', 'organisasi', 'magang', 'sks']].values
labels = df['kategori'].values

# Standardisasi fitur
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Encode label ke angka
encoder = LabelEncoder()
labels_encoded = encoder.fit_transform(labels)

# === Load model ANFIS ===
model = ANFIS()
# Jika punya model yang sudah dilatih:
# model.load_state_dict(torch.load("model.pth"))
model.eval()

# === Routing utama ===
@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        # Ambil input dari form
        try:
            ipk = float(request.form['ipk'])
            progress = float(request.form['progress'])
            kehadiran = float(request.form['kehadiran'])
            organisasi = float(request.form['organisasi'])
            magang = float(request.form['magang'])
            sks = float(request.form['sks'])

            # Buat array input dan scaling
            X = np.array([[ipk, progress, kehadiran, organisasi, magang, sks]])
            X_scaled = scaler.transform(X)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

            # Inference
            with torch.no_grad():
                output = model(X_tensor)
                pred_class = torch.argmax(output, dim=1).item()
                kategori = encoder.inverse_transform([pred_class])[0]

            result = f"Hasil prediksi kategori: <strong>{kategori}</strong>"

        except Exception as e:
            result = f"Terjadi kesalahan input: {e}"

    return render_template("index.html", result=result)

# === Run app ===
if __name__ == '__main__':
    app.run(debug=True)
