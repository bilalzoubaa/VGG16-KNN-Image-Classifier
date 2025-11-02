import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier

# ============================
# Supprimer les messages TensorFlow inutiles
# ============================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ============================
# 1️⃣ Charger les images + labels
# ============================
dataset_dir = "images"  # dossier contenant cat/, dog/, panda/
image_paths = []
labels = []

for root, dirs, files in os.walk(dataset_dir):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(root, file)
            label = os.path.basename(root)
            image_paths.append(path)
            labels.append(label)

print(f"Total images trouvées : {len(image_paths)}")

# ============================
# 2️⃣ Modèle de feature extraction
# ============================
model = VGG16(weights='imagenet', include_top=False, pooling='avg')

def extract_feature_from_path(img_path):
    img = image.load_img(img_path, target_size=(224,224))
    x = image.img_to_array(img)
    return x

# ============================
# 3️⃣ Extraire les features pour toutes les images (batch + sauvegarde)
# ============================
features_file = "features.npy"

if os.path.exists(features_file):
    features = np.load(features_file)
    labels = np.load("labels.npy", allow_pickle=True)
    image_paths = np.load("image_paths.npy", allow_pickle=True).tolist()
    print("Features chargées depuis le disque ✅")
else:
    batch_size = 32
    all_imgs = []
    features_list = []

    for i, p in enumerate(image_paths):
        all_imgs.append(extract_feature_from_path(p))
        if len(all_imgs) == batch_size or i == len(image_paths)-1:
            batch = np.array(all_imgs)
            batch = preprocess_input(batch)
            batch_features = model.predict(batch, verbose=0)
            features_list.append(batch_features)
            all_imgs = []

    features = np.vstack(features_list)
    np.save("features.npy", features)
    np.save("labels.npy", np.array(labels))
    np.save("image_paths.npy", np.array(image_paths))
    print("Extraction terminée ✅")
print("Shape des features :", features.shape)

# ============================
# 4️⃣ Classification KNN
# ============================
knn_clf = KNeighborsClassifier(n_neighbors=5, metric='cosine')
knn_clf.fit(features, labels)
print("KNN entraîné pour la classification ✅")

# ============================
# 5️⃣ Fonction pour prédiction + affichage voisins avec métrique variable
# ============================
def predict_image_class(img_path, k=5, metric='cosine'):
    # Charger et prétraiter l'image test
    img = image.load_img(img_path, target_size=(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    test_feat = model.predict(x, verbose=0).flatten()

    # Prédiction avec KNN
    if metric == 'cosine':
        knn_temp = KNeighborsClassifier(n_neighbors=k, metric='cosine')
    else:
        knn_temp = KNeighborsClassifier(n_neighbors=k, metric='euclidean')

    knn_temp.fit(features, labels)
    pred_label = knn_temp.predict([test_feat])[0]
    distances, indices = knn_temp.kneighbors([test_feat], n_neighbors=k)

    print(f"\n➡️ L’image '{img_path}' est probablement : {pred_label.upper()} (métrique={metric})")
    print("Voisins les plus proches :")
    for i in indices[0]:
        print("  -", image_paths[i], "→", labels[i])

    # Affichage test + voisins
    plt.figure(figsize=(3*(k+1), 3))
    plt.subplot(1, k+1, 1)
    plt.imshow(Image.open(img_path))
    plt.title("Test")
    plt.axis("off")

    for j, idx in enumerate(indices[0]):
        plt.subplot(1, k+1, j+2)
        plt.imshow(Image.open(image_paths[idx]))
        plt.title(labels[idx])
        plt.axis("off")

    plt.show()
    return pred_label

# ============================
# 6️⃣ Tester plusieurs images + comparer deux métriques
# ============================
test_images = ["testdog.jpg", "testcat.jpg"]  # Remplacer par vos images tests

for img_path in test_images:
    predict_image_class(img_path, k=5, metric='cosine')
    predict_image_class(img_path, k=5, metric='euclidean')
