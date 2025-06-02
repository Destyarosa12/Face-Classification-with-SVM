# Face-Classification-with-SVM
# 🎓 Face Recognition and Classification: Mahasiswa vs Dosen

Proyek ini merupakan implementasi sistem pengenalan wajah dan klasifikasi untuk membedakan antara **mahasiswa** dan **dosen**, menggunakan library `face_recognition` dan algoritma **Support Vector Machine (SVM)**. Proyek ini dibuat sebagai tugas akhir semester untuk mata kuliah *Pengenalan Pola*.

## 🚀 Fitur Utama

- 🔍 Ekstraksi fitur wajah menggunakan `face_recognition`
- 🧠 Training model klasifikasi menggunakan SVM
- 🎥 Sistem klasifikasi wajah secara real-time dari webcam
- 📊 Evaluasi performa model menggunakan confusion matrix dan classification report

## 📁 Struktur File
├── dataset_augmented/ # Folder dataset (tidak disertakan di repo)
├── extract_data.py # Ekstraksi face encoding dari dataset
├── train_model.py # Training model klasifikasi dengan SVM
├── realtime_classifier.py # Sistem klasifikasi wajah secara real-time
├── face_data.pkl # Hasil encoding wajah (dibuat saat runtime)
├── model_svm.pkl # Model hasil training (dibuat saat runtime)
