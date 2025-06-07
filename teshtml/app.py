from flask import Flask, render_template, request
import torch
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from model import ANFIS

# Inisialisasi Flask app
app = Flask(__name__)

# === Load model ANFIS ===
with open('../model/scaler.pkl','rb') as f:
    scaler = pickle.load(f)

with open('../model/label_encoder.pkl','rb') as f:
    encoder = pickle.load(f)
model = ANFIS()
save_path = '../model/modelANFIS.pt'
def predict(data):
    data_scaled = scaler.transform(data)
    data_tensor = torch.tensor(data_scaled,dtype=torch.float32)
    model.load_state_dict(torch.load(save_path))
    model.eval()
    with torch.no_grad():
        output = model(data_tensor) 
        pred_class = torch.argmax(output, dim=1).item()
        print(pred_class)
        pred_label = encoder.inverse_transform([pred_class])[0]
    return pred_label

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
            print(X)

            # Inference
            kategori = predict(X)
            print(kategori)

            result = f"Hasil prediksi kategori: <strong>{kategori}</strong>"

        except Exception as e:
            result = f"Terjadi kesalahan input: {e}"

    return render_template("index.html", result=result)

# === Run app ===
if __name__ == '__main__':
    app.run(debug=True)
