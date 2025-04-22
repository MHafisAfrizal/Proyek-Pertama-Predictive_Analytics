# Laporan Proyek Analitik Prediktif: Prediksi Risiko Readmisi Pasien Diabetes

## 1. Domain Masalah

**Domain**: Kesehatan\
**Pernyataan Masalah**: Readmisi rumah sakit untuk pasien diabetes mahal dan menunjukkan adanya celah dalam kualitas perawatan. Memprediksi risiko readmisi dapat membantu rumah sakit memprioritaskan pasien berisiko tinggi untuk intervensi, meningkatkan hasil perawatan, dan mengurangi biaya.\
**Tujuan**: Mengembangkan model regresi untuk memprediksi skor risiko readmisi berkelanjutan untuk pasien diabetes berdasarkan data klinis dan demografis.\
**Mengapa Penting?**: Mengurangi readmisi sejalan dengan tujuan kesehatan untuk meningkatkan hasil pasien dan mengoptimalkan alokasi sumber daya. Proyek ini menangani tantangan dunia nyata di sektor kesehatan.

## 2. Pemahaman Data

**Dataset**: Diabetes 130-US hospitals for years 1999-2008 (UCI Machine Learning Repository).\
**Sumber**: UCI Machine Learning Repository\
**Ukuran**: 101.766 catatan; diambil sampel 5000 untuk proyek ini untuk memenuhi syarat minimum dan memastikan efisiensi komputasi.\
**Fitur**: Lebih dari 50 fitur kuantitatif dan kategorikal, termasuk:

- **Demografis**: Usia, jenis kelamin, ras.

- **Klinis**: Jumlah prosedur laboratorium, obat-obatan, diagnosis.

- **Rawat Inap**: Waktu di rumah sakit, jumlah kunjungan rawat inap/jalan.\
  **Variabel Target**: Diambil dari kolom 'readmitted', diubah menjadi skor risiko berkelanjutan:

- NO: 0 (tidak readmisi).

- 30: 0.5 (readmisi setelah 30 hari).

- &lt;30: 1 (readmisi dalam 30 hari).\
  **Eksplorasi**: Analisis awal menunjukkan nilai yang hilang di kolom seperti ras dan berat badan, serta ketidakseimbangan kelas di kolom readmitted. Visualisasi (misalnya, plot hitung) mengkonfirmasi distribusi status readmisi.

## 3. Persiapan Data

**Pembersihan**:

- Mengganti '?' dengan NaN dan mengisi nilai yang hilang (misalnya, 'Unknown' untuk kategorikal, median untuk numerik).
- Menghapus kolom dengan nilai hilang berlebihan (weight, payer_code, medical_specialty) atau data tidak relevan (encounter_id, patient_nbr).\
  **Rekayasa Fitur**:
- Membuat fitur 'total_prosedur' dengan menjumlahkan prosedur lab, rawat jalan, rawat inap, dan darurat.
- Mengelompokkan 'usia' ke dalam kategori: Muda (\[0-30)), Setengah Baya (\[30-60)), Senior (\[60-100)).
- Mengonversi diagnosis (diag_1, diag_2, diag_3) menjadi kode yang disederhanakan.\
  **Enkoding**: Menggunakan LabelEncoder untuk variabel kategorikal (misalnya, ras, jenis kelamin, kelompok_usia).\
  **Penskalaan**: Menerapkan StandardScaler pada fitur numerik untuk normalisasi data.\
  **Pembagian**: Membagi data menjadi 80% pelatihan dan 20% pengujian.

## 4. Pemodelan

**Pendekatan**: Regresi (memprediksi skor risiko readmisi berkelanjutan).\
**Model yang Dilatih**:

- **Regresi Linear**: Model dasar yang mengasumsikan hubungan linear.
- **Random Forest Regressor**: Metode ensemble untuk menangkap pola non-linear.
- **XGBoost Regressor**: Peningkatan gradien untuk performa tinggi.\
  **Penyetelan Hiperparameter**: Melakukan GridSearchCV untuk Random Forest dengan parameter:
- n_estimators: \[100, 200\]
- max_depth: \[10, 20, None\]\
  **Implementasi**: Menggunakan pustaka scikit-learn dan xgboost di Python.

## 5. Evaluasi

**Metrik**:

- **Mean Absolute Error (MAE)**: Rata-rata perbedaan absolut antara prediksi dan nilai aktual.
- **Mean Squared Error (MSE)**: Rata-rata kuadrat perbedaan, menghukum kesalahan besar.
- **R-squared (R²)**: Proporsi varians yang dijelaskan oleh model.\
  **Hasil**:
- Regresi Linear: Performa sedang, terbatas oleh asumsi linear.
- Random Forest: Performa lebih baik karena menangani non-linearitas.
- XGBoost: Kompetitif tetapi sedikit di bawah Random Forest yang disetel.
- Random Forest (Disetel): Performa terbaik dengan MAE terendah dan R² tertinggi.\
  **Visualisasi**: Plot batang membandingkan skor R² antar model.

## 6. Kriteria Tambahan

Untuk menargetkan peringkat 4-5 bintang, proyek ini mencakup:

1. **Rekayasa Fitur**: Membuat fitur 'total_prosedur' dan 'kelompok_usia' untuk meningkatkan performa model.
2. **Penyetelan Hiperparameter**: Mengoptimalkan parameter Random Forest menggunakan GridSearchCV.
3. **Perbandingan Model**: Mengevaluasi tiga model (Regresi Linear, Random Forest, XGBoost).
4. **Visualisasi**: Menyertakan plot untuk distribusi data, performa model, dan pentingnya fitur.
5. **Pentingnya Fitur**: Menganalisis dan memvisualisasikan pentingnya fitur untuk Random Forest.
6. **Dokumentasi Jelas**: Menyediakan sel teks rinci di notebook dan laporan komprehensif ini.

## 7. Kesimpulan

Proyek ini berhasil memprediksi risiko readmisi pasien diabetes menggunakan model regresi. Model Random Forest (Disetel) mencapai performa terbaik, menunjukkan efektivitas metode ensemble dan optimasi hiperparameter. Rekayasa fitur dan visualisasi memberikan wawasan lebih dalam tentang data dan perilaku model. Proyek ini memenuhi semua kriteria submission Dicoding dan mencakup enam kriteria tambahan untuk menargetkan peringkat 5 bintang.

## 8. Perbaikan di Masa Depan

- Menjelajahi model deep learning (misalnya, jaringan saraf) untuk performa yang lebih baik.
- Mengatasi ketidakseimbangan kelas di kolom readmitted menggunakan teknik seperti SMOTE.
- Menambahkan fitur tambahan, seperti data sosial ekonomi, jika tersedia.
- Mengimplementasikan model sebagai aplikasi web untuk penggunaan dunia nyata.

## 9. File Submission

- **Jupyter Notebook**: `prediksi_readmisi_diabetes.ipynb` (dieksekusi, dengan sel teks).
- **Skrip Python**: `prediksi_readmisi_diabetes.py` (kode setara).
- **Laporan**: `laporan_proyek.md` (file ini).

**Catatan**: Dataset dimuat langsung dari URL repositori UCI untuk memastikan aksesibilitas. Pastikan semua file di-zip untuk submission.