from flask import Flask, render_template, request
import torch
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from model import ANFIS

app = Flask(__name__)

# === Load model dan scaler ===
with open('../model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('../model/label_encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

model = ANFIS()
save_path = '../model/modelANFIS.pt'

def predict(data):
    data_scaled = scaler.transform(data)
    data_tensor = torch.tensor(data_scaled, dtype=torch.float32)
    model.load_state_dict(torch.load(save_path))
    model.eval()
    with torch.no_grad():
        output = model(data_tensor)
        pred_class = torch.argmax(output, dim=1).item()
        pred_label = encoder.inverse_transform([pred_class])[0]
    return pred_label

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    chart_data = None
    if request.method == 'POST':
        try:
            ipk = float(request.form['ipk'])
            progress = float(request.form['progress'])
            kehadiran = float(request.form['kehadiran'])
            organisasi = float(request.form['organisasi'])
            magang = float(request.form['magang'])
            sks = float(request.form['sks'])

            X = np.array([[ipk, progress, kehadiran, organisasi, magang, sks]])
            kategori = predict(X)
            result = f"Hasil prediksi kategori: <strong>{kategori}</strong>"
            chart_data = X.tolist()[0]

        except Exception as e:
            result = f"Terjadi kesalahan input: {e}"

    return render_template("index.html", result=result, chart_data=chart_data)

if __name__ == '__main__':
    app.run(debug=True)
