import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from rembg import remove
from PIL import Image
import pandas as pd
import glob
import pickle

# Mapping kategori
categories = {0: 'Ikan_Baronang', 1: 'Ikan_Belanak', 2: 'Ikan_Kakap'}
LABELS = list(categories.values())
DATASET_DIR = 'dataset/asli'
IMAGE_SIZE = (128, 128)

def remove_background(img):
    try:
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img_no_bg = remove(img_pil)
        img_no_bg = img_no_bg.convert("RGB")
        return cv2.cvtColor(np.array(img_no_bg), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"[ERROR] rembg gagal: {e}")
        return img  # fallback jika gagal hapus background

def extract_hog_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features, _ = hog(
        gray,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=True
    )
    return features

def load_and_preprocess_data():
    X, y, filenames = [], [], []

    for idx, label in enumerate(LABELS):
        print(f"[INFO] Memproses kelas: {label}")
        class_dir = os.path.join(DATASET_DIR, label)
        image_paths = glob.glob(os.path.join(class_dir, "*.jpg")) + glob.glob(os.path.join(class_dir, "*.png"))

        if not image_paths:
            print(f"[WARNING] Tidak ada gambar ditemukan untuk {label}")
            continue

        for path in image_paths:
            try:
                img = cv2.imread(path)
                if img is None:
                    print(f"[WARNING] Gagal membaca: {path}")
                    continue

                img_resized = cv2.resize(img, IMAGE_SIZE)
                img_no_bg = remove_background(img_resized)
                features = extract_hog_features(img_no_bg)

                X.append(features)
                y.append(idx)
                filenames.append(os.path.basename(path))
            except Exception as e:
                print(f"[ERROR] File rusak atau gagal proses ({path}): {e}")

    if len(X) == 0:
        raise ValueError("Tidak ada data yang berhasil diproses!")

    X = np.array(X)
    y = np.array(y)
    print(f"[INFO] Ekstraksi fitur selesai. Jumlah data: {len(y)}")
    return X, y, filenames

def save_to_excel(X, y, filenames):
    print("[INFO] Menyimpan fitur ke file Excel...")
    df_X = pd.DataFrame(X)
    df_y = pd.DataFrame({'Label_Index': y})
    df_files = pd.DataFrame({'Filename': filenames})
    df_combined = pd.concat([df_files, df_y, df_X], axis=1)
    df_combined.to_excel("fish_features.xlsx", index=False)
    print("[INFO] Disimpan sebagai fish_features.xlsx")

if __name__ == "__main__":
    try:
        # Load dan preprocessing
        X, y, filenames = load_and_preprocess_data()

        # Simpan fitur ke Excel
        save_to_excel(X, y, filenames)

        # Normalisasi fitur
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # Encode label
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

        # Train KNN
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(X_train, y_train)

        # Evaluasi
        y_pred = model.predict(X_test)
        print("\n[INFO] Evaluasi Model KNN:")
        print(classification_report(y_test, y_pred, target_names=[categories[i] for i in le.classes_]))
        print("Akurasi:", accuracy_score(y_test, y_pred))

        # Simpan model, encoder, scaler
        with open("knn_model.pkl", "wb") as f:
            pickle.dump(model, f)
        with open("label_encoder.pkl", "wb") as f:
            pickle.dump(le, f)
        with open("scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)

        print("\n[INFO] Model, label encoder, dan scaler telah disimpan.")

    except Exception as e:
        print(f"[FATAL ERROR] {e}")
