import cv2
import numpy as np
from skimage.feature import hog
from rembg import remove
from PIL import Image
import pickle

# Fungsi preprocessing
def remove_background(img):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img_no_bg = remove(img_pil)
    img_no_bg = img_no_bg.convert("RGB")
    return cv2.cvtColor(np.array(img_no_bg), cv2.COLOR_RGB2BGR)

def extract_hog_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features, _ = hog(gray, pixels_per_cell=(8,8), cells_per_block=(2,2), block_norm='L2-Hys', visualize=True)
    return features

# Load model, encoder, dan scaler
with open("knn_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Label mapping
categories = {0: 'Ikan_Baronang', 1: 'Ikan_Belanak', 2: 'Ikan_Kakap'}

# Load gambar uji
img_path = "tes/01.png"
img = cv2.imread(img_path)

if img is None:
    print(f"[ERROR] Gambar tidak ditemukan atau rusak: {img_path}")
    exit()

img_resized = cv2.resize(img, (128, 128))
img_no_bg = remove_background(img_resized)
features = extract_hog_features(img_no_bg).reshape(1, -1)

# Normalisasi
features = scaler.transform(features)

# Prediksi dengan probabilitas jarak (gunakan kneighbors untuk analisis jarak)
distances, indices = model.kneighbors(features)

# Ambil jarak ke tetangga terdekat
min_distance = distances[0][0]

# Threshold jarak (sesuaikan sesuai data kamu)
THRESHOLD = 1500  # Coba atur antara 1000–3000, tergantung besar fitur

if min_distance > THRESHOLD:
    print("[RESULT] Prediksi jenis ikan: Tidak diketahui (bukan ikan target)")
else:
    pred = model.predict(features)
    predicted_label = categories[le.inverse_transform(pred)[0]]
    print(f"[RESULT] Prediksi jenis ikan: {predicted_label} (jarak={min_distance:.2f})")