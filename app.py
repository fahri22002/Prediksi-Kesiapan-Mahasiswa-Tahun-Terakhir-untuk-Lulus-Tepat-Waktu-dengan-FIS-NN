import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

# === Model ===

class GaussianMF(nn.Module):
    def __init__(self, mean, sigma):
        super().__init__()
        self.mean = nn.Parameter(torch.tensor(float(mean)))
        self.sigma = nn.Parameter(torch.tensor(float(sigma)))

    def forward(self, x):
        sigma = torch.clamp(self.sigma, min=1e-3)
        return torch.exp(-0.5 * ((x - self.mean) / sigma) ** 2)

class ANFIS(nn.Module):
    def __init__(self, n_inputs=6, mfs_per_input=2, n_classes=3):
        super().__init__()
        self.n_inputs = n_inputs
        self.mfs_per_input = mfs_per_input
        self.n_rules = mfs_per_input ** n_inputs
        self.n_classes = n_classes

        self.mf_layers = nn.ModuleList([
            nn.ModuleList([GaussianMF(mean=0.5 + i, sigma=1.0) for i in range(mfs_per_input)])
            for _ in range(n_inputs)
        ])

        self.rule_weights = nn.Parameter(torch.randn(self.n_rules, n_inputs + 1, n_classes))

    def forward(self, x):
        batch_size = x.size(0)
        mf_values = []
        for i in range(self.n_inputs):
            mf_x = [mf(x[:, i]) for mf in self.mf_layers[i]]
            mf_values.append(torch.stack(mf_x, dim=1))

        from itertools import product
        rule_indices = list(product(range(self.mfs_per_input), repeat=self.n_inputs))

        firing_strengths = []
        for idx in rule_indices:
            rule_mfs = [mf_values[i][:, idx[i]] for i in range(self.n_inputs)]
            prod = torch.stack(rule_mfs, dim=0).prod(dim=0)
            firing_strengths.append(prod)

        firing_strengths = torch.stack(firing_strengths, dim=1)
        norm_strengths = firing_strengths / torch.sum(firing_strengths, dim=1, keepdim=True)

        x_aug = torch.cat([x, torch.ones(batch_size, 1)], dim=1)
        consequent = torch.einsum('bi,rij->brj', x_aug, self.rule_weights)
        output = torch.sum(norm_strengths.unsqueeze(-1) * consequent, dim=1)

        return output

# === Load & preprocess training data for scaling and label encoding ===
df = pd.read_csv("Book1.csv")
features = df[['ipk', 'progress', 'kehadiran', 'organisasi', 'magang', 'sks']].values
scaler = StandardScaler()
scaler.fit(features)

label_encoder = LabelEncoder()
label_encoder.fit(df['kategori'])

# === Load or define model ===
model = ANFIS()
# NOTE: Load trained model state here jika sudah ada:
# model.load_state_dict(torch.load('model.pth'))

# === Streamlit UI ===
st.title("Prediksi Kategori Mahasiswa (ANFIS)")

ipk = st.number_input("IPK", min_value=0.0, max_value=4.0, step=0.01)
progress = st.number_input("Progress", min_value=0.0, max_value=100.0)
kehadiran = st.number_input("Kehadiran", min_value=0.0, max_value=100.0)
organisasi = st.number_input("Organisasi", min_value=0.0, max_value=10.0)
magang = st.number_input("Magang", min_value=0.0, max_value=10.0)
sks = st.number_input("SKS", min_value=0.0, max_value=160.0)

if st.button("Prediksi"):
    # Preprocess input
    input_np = np.array([[ipk, progress, kehadiran, organisasi, magang, sks]])
    input_scaled = scaler.transform(input_np)
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        pred_class = torch.argmax(output, dim=1).item()
        kategori = label_encoder.inverse_transform([pred_class])[0]

    st.success(f"Kategori Prediksi: **{kategori}**")
