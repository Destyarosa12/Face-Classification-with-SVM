# Face-Classification-with-SVM
# 🎓 Face Recognition and Classification: Mahasiswa vs Dosen

Proyek ini merupakan implementasi sistem pengenalan wajah dan klasifikasi untuk membedakan antara **mahasiswa** dan **dosen**, menggunakan library `face_recognition` dan algoritma **Support Vector Machine (SVM)**. Proyek ini dibuat sebagai tugas akhir semester untuk mata kuliah *Pengenalan Pola*.

## 🚀 Fitur Utama

- 🔍 Ekstraksi fitur wajah menggunakan `face_recognition`
- 🧠 Training model klasifikasi menggunakan SVM
- 🎥 Sistem klasifikasi wajah secara real-time dari webcam
- 📊 Evaluasi performa model menggunakan confusion matrix dan classification report

## 📁 Struktur File
- ├── dataset_augmented/ # Folder dataset (tidak disertakan di repo)
- ├── extract_data.py # Ekstraksi face encoding dari dataset
- ├── train_model.py # Training model klasifikasi dengan SVM
- ├── realtime_classifier.py # Sistem klasifikasi wajah secara real-time
- ├── face_data.pkl # Hasil encoding wajah (dibuat saat runtime)
- ├── model_svm.pkl # Model hasil training (dibuat saat runtime)


## 📦 Dataset & Model

Dataset dan file hasil ekstraksi/training tidak disertakan langsung karena ukurannya besar. Silakan unduh melalui tautan berikut:

- 📁 Dataset & Augmentasi: [🔗 [Link Google Drive](https://drive.google.com/drive/folders/1woVdKpMC7AxBUeG_vqVV_LHEkuSLuLwr?usp=sharing)](#) 
- 📄 face_data.pkl: [🔗 [Link Google Drive](https://drive.google.com/file/d/1P0WtgUqOMuPli5fD0FPJmOm_-n9njnVP/view?usp=sharing)](#)
- 🧠 svm_classifier.pkl: [🔗 [Link Google Drive](https://drive.google.com/file/d/1DJW8uOLWEXecC8ScKJE5WaRcsLoUSW6r/view?usp=drive_link)](#)

## ⚙️ Cara Menjalankan

1. Pastikan semua dependensi sudah terinstall:
   ```bash
   pip install face_recognition opencv-python scikit-learn numpy
2. Jalankan proses augmentasi data:
   ```bash
   python augmentasi.py
4. Jalankan proses ekstraksi encoding wajah:
   ```bash
   python extract_face_data.py
5. Latih model SVM menggunakan data encoding:
   ```bash
   python train.py
6. Jalankan sistem klasifikasi wajah secara real-time:
   ```bash
   python realtime.py

## 🧪 Hasil Evaluasi

Model dievaluasi menggunakan dataset uji dengan hasil berikut:
- Accuracy: ±97%
- 

## 📚 Dependensi

Semua dependensi tercantum di file `requirements.txt`:
```bash
pip install -r requirements.txt
