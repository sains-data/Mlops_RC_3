# Mlops_RC_3

# 🚀 MLOps Machine Failure Prediction System

## 📌 Deskripsi Proyek
Proyek ini merupakan implementasi **Machine Learning Operations (MLOps)** untuk membangun sistem prediksi kerusakan mesin berbasis data sensor. Sistem ini dirancang untuk memprediksi **kapan mesin akan mengalami kegagalan (failure cycle)** serta **persentase probabilitas kerusakan**, kemudian menampilkannya melalui sebuah website interaktif.

Proyek dikembangkan sebagai bagian dari **Tugas Besar Mata Kuliah Machine Learning Operations**, dengan fokus pada penerapan pipeline end-to-end mulai dari pengolahan data, training model, experiment tracking, hingga deployment.

## 🎯 Tujuan
- Membangun model machine learning untuk mendeteksi potensi kerusakan mesin berdasarkan data sensor
- Mengimplementasikan pipeline MLOps yang terstruktur dan reproducible
- Melakukan tracking eksperimen dan pemilihan model terbaik menggunakan **MLflow**
- Menyediakan antarmuka website untuk inference model menggunakan **Streamlit**

## 📂 Dataset
Dataset yang digunakan berasal dari Kaggle:

🔗 **NASA Turbofan Jet Engine Data Set (C-MAPSS)**  
https://www.kaggle.com/datasets/behrad3d/nasa-cmaps

Dataset ini merupakan dataset simulasi degradasi mesin jet turbofan (C-MAPSS) yang dikembangkan oleh NASA dan banyak digunakan untuk kasus **predictive maintenance** dan **Remaining Useful Life (RUL) prediction**.

Karakteristik dataset:
- Data sensor mesin jet turbofan (run-to-failure)
- Data berbentuk tabular (dalam bentuk .txt yang sudah di ekstrak)
- Memuat beberapa sensor dan kondisi operasi mesin
- Digunakan untuk memprediksi kegagalan mesin berdasarkan cycle

> ⚠️ Pada tahap ini, sistem masih menggunakan **dataset awal** dan belum menerapkan mekanisme penambahan dataset baru maupun retraining incremental.

## 🧠 Alur Sistem (Workflow)
1. **Data Ingestion**  
   Dataset sensor dibaca dan diproses untuk menghasilkan label kerusakan mesin.

2. **Exploratory Data Analysis (EDA)**  
   Analisis pola degradasi sensor, korelasi fitur, dan distribusi label untuk memahami karakteristik data.

3. **Preprocessing Data**  
   Pemilihan fitur dan normalisasi data agar siap digunakan oleh model.

4. **Model Training & Evaluation**  
   Model machine learning dilatih dan dievaluasi menggunakan data training dan testing (80:20).

5. **Experiment Tracking (MLflow)**  
   Seluruh eksperimen dicatat untuk membandingkan performa model dan memilih model terbaik.

6. **Deployment (Streamlit)**  
   Model terbaik dideploy ke aplikasi web untuk melakukan prediksi dan simulasi data sensor secara interaktif.

## 🌐 Fitur Aplikasi Web
- Simulasi data sensor mesin secara real-time
- Prediksi status mesin (NORMAL / CRITICAL)
- Estimasi risiko kerusakan dalam bentuk probabilitas
- Visualisasi telemetry sensor (tekanan, temperatur, dan RPM)
- Riwayat hasil prediksi selama simulasi berjalan

## 🛠️ Tools & Teknologi
- **Python**  
  Bahasa pemrograman utama yang digunakan untuk pengembangan pipeline machine learning dan MLOps.
- **Scikit-learn / Machine Learning Library**  
  Digunakan untuk membangun, melatih, dan mengevaluasi model machine learning berbasis data sensor.
- **MLflow**  
  Digunakan sebagai alat experiment tracking dan model management untuk mencatat parameter, metric, serta menyimpan model terbaik.
- **Streamlit**  
  Digunakan untuk membangun dan mendeploy aplikasi web interaktif sebagai antarmuka inferensi model.
- **Pandas & NumPy**  
  Digunakan untuk manipulasi, analisis, dan pengolahan data sensor dalam bentuk tabular.
- **joblib**  
  Digunakan untuk menyimpan dan memuat model machine learning atau objek Python secara efisien.
- **PyYAML**  
  Digunakan untuk membaca dan mengelola file konfigurasi (YAML), seperti pengaturan parameter model dan pipeline.
- **Matplotlib**  
  Digunakan untuk visualisasi data dan hasil evaluasi model dalam bentuk grafik statis.
- **Seaborn**  
  Digunakan untuk visualisasi data statistik yang lebih informatif dan eksploratif berbasis Matplotlib.
- **Plotly**  
  Digunakan untuk visualisasi data interaktif yang ditampilkan pada aplikasi web.

## 🌐 Fitur Website
- Input data sensor mesin
- Prediksi:
  - Estimasi cycle mesin akan berhenti atau rusak
  - Persentase probabilitas kerusakan
- Tampilan hasil prediksi secara real-time melalui website.

## 📊 Experiment Tracking
MLflow digunakan untuk mencatat parameter, metrik evaluasi, serta artifact model sehingga proses training dapat direproduksi dan dibandingkan dengan mudah.

## 📌 Catatan Pengembangan
- Mekanisme penambahan dataset baru dan retraining otomatis belum diterapkan
- Pipeline dirancang agar dapat dikembangkan menjadi **continuous training** pada tahap selanjutnya

## 👥 Tim
- Akmal Faiz Abdillah (122450114)
- Elok Fiola (122450051)
- Rut Junita Sari Siburian (122450103)
- Syalaisha Andina Putriansyah (122450111)

## 📄 Lisensi
Proyek ini dibuat untuk keperluan akademik dan pembelajaran.
