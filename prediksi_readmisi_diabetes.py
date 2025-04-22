# prediksi_readmisi_diabetes.py
# Proyek Analitik Prediktif untuk Submission Machine Learning Dicoding

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Tetapkan seed acak untuk reproduktivitas
np.random.seed(42)

# Muat dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00296/diabetic_data.csv'
data = pd.read_csv(url)

# Ambil sampel 5000 data
data = data.sample(n=5000, random_state=42)

# Persiapan Data
data.replace('?', np.nan, inplace=True)
data['race'].fillna('Unknown', inplace=True)
data['weight'].fillna('Unknown', inplace=True)
data['payer_code'].fillna('Unknown', inplace=True)
data['medical_specialty'].fillna('Unknown', inplace=True)
data.drop(['weight', 'payer_code', 'medical_specialty'], axis=1, inplace=True)

# Ubah readmitted menjadi skor risiko berkelanjutan
data['risiko_readmisi'] = data['readmitted'].map({'NO': 0, '>30': 0.5, '<30': 1})
data.drop('readmitted', axis=1, inplace=True)

# Rekayasa fitur
data['total_prosedur'] = data['num_lab_procedures'] + data['num_procedures'] + \
                         data['number_outpatient'] + data['number_emergency'] + \
                         data['number_inpatient']

# Kelompokkan usia ke dalam kategori
def kelompok_usia(age):
    if '[0-30)' in age:
        return 'Muda'
    elif '[30-60)' in age:
        return 'Setengah Baya'
    else:
        return 'Senior'
data['kelompok_usia'] = data['age'].apply(kelompok_usia)

# Enkode variabel kategorikal
le = LabelEncoder()
kolom_kategorikal = ['race', 'gender', 'kelompok_usia', 'diag_1', 'diag_2', 'diag_3'] + \
                    [col for col in data.columns if col.startswith('max_glu') or col.startswith('A1C') or col.startswith('change') or col.startswith('diabetesMed')]
for col in kolom_kategorikal:
    data[col] = le.fit_transform(data[col].astype(str))

# Hapus kolom yang tidak relevan
data.drop(['encounter_id', 'patient_nbr', 'age', 'diag_1', 'diag_2', 'diag_3'], axis=1, inplace=True)

# Pisahkan fitur dan target
X = data.drop('risiko_readmisi', axis=1)
y = data['risiko_readmisi']

# Bagi data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Skalakan fitur
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Pemodelan
model = {
    'Regresi Linear': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42)
}

# Latih dan evaluasi model
hasil = {}
for nama, model in model.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    hasil[nama] = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'R2': r2_score(y_test, y_pred)
    }

# Penyetelan hiperparameter untuk Random Forest
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None]
}
grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Perbarui hasil Random Forest
rf_terbaik = grid_search.best_estimator_
y_pred = rf_terbaik.predict(X_test)
hasil['Random Forest (Disetel)'] = {
    'MAE': mean_absolute_error(y_test, y_pred),
    'MSE': mean_squared_error(y_test, y_pred),
    'R2': r2_score(y_test, y_pred)
}

# Tampilkan hasil
hasil_df = pd.DataFrame(hasil).T
print('Performa Model:')
print(hasil_df)

# Pentingnya fitur
pentingnya_fitur = pd.DataFrame({
    'Fitur': X.columns,
    'Pentingnya': rf_terbaik.feature_importances_
}).sort_values('Pentingnya', ascending=False)
print('\nPentingnya Fitur:')
print(pentingnya_fitur)