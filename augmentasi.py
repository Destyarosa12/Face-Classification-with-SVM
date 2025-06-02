import os
import cv2
import numpy as np
import random
import shutil

# Folder asal dataset dan folder tujuan hasil augmentasi
input_root = 'dataset'
output_root = 'dataset_augmented'

# Fungsi-fungsi augmentasi sederhana
def flip_horizontal(image):
    return cv2.flip(image, 1)

def rotate(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, mat, (w, h))

def adjust_brightness(image, factor):
    return np.clip(image * factor, 0, 255).astype(np.uint8)

def blur(image, ksize=3):
    return cv2.GaussianBlur(image, (ksize, ksize), 0)

def add_noise(image):
    noise = np.random.normal(0, 25, image.shape).astype(np.int16)
    noisy_img = np.clip(image.astype(np.int16) + noise, 0, 255)
    return noisy_img.astype(np.uint8)

# Loop kelas (dosen, mahasiswa)
for class_folder in ['dosen', 'mahasiswa']:
    input_class_path = os.path.join(input_root, class_folder)
    output_class_path = os.path.join(output_root, class_folder)
    os.makedirs(output_class_path, exist_ok=True)

    # Loop subjek di dalam masing-masing kelas
    for subject_name in os.listdir(input_class_path):
        subject_input_path = os.path.join(input_class_path, subject_name)
        subject_output_path = os.path.join(output_class_path, subject_name)

        if not os.path.isdir(subject_input_path):
            continue

        os.makedirs(subject_output_path, exist_ok=True)
        print(f"🔁 Memproses: {subject_input_path}")

        # Loop gambar-gambar dalam subjek
        for img_name in os.listdir(subject_input_path):
            img_path = os.path.join(subject_input_path, img_name)

            if not os.path.isfile(img_path) or not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            image = cv2.imread(img_path)
            if image is None:
                print(f"⚠️ Gagal membaca {img_path}")
                continue

            base_name = os.path.splitext(img_name)[0]

            for i in range(10):
                aug_img = image.copy()

                if random.random() < 0.5:
                    aug_img = flip_horizontal(aug_img)
                if random.random() < 0.5:
                    aug_img = rotate(aug_img, random.uniform(-15, 15))
                if random.random() < 0.5:
                    aug_img = adjust_brightness(aug_img, random.uniform(0.8, 1.2))
                if random.random() < 0.5:
                    aug_img = blur(aug_img, ksize=random.choice([3, 5]))
                if random.random() < 0.5:
                    aug_img = add_noise(aug_img)

                out_name = f"{base_name}_aug{i+1}.jpg"
                out_path = os.path.join(subject_output_path, out_name)
                cv2.imwrite(out_path, aug_img)
                print(f"✅ Saved: {out_path}")

print("\n🎉 Semua data berhasil diaugmentasi ke folder 'dataset_augmented'")
