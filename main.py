"""Image classification with VGG16 transfer-learning features and a KNN classifier.

Pipeline:
    1. Walk the dataset directory and collect (image_path, label) pairs.
    2. Extract a 512-d feature vector per image with VGG16 (ImageNet weights,
       no top layer, global average pooling), caching the result to disk.
    3. Classify a query image by finding its k nearest neighbors (cosine or
       euclidean distance) among the cached features with scikit-learn's KNN.
"""
import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image

DATASET_DIR = "images"
FEATURES_FILE = "features.npy"
LABELS_FILE = "labels.npy"
IMAGE_PATHS_FILE = "image_paths.npy"
IMAGE_SIZE = (224, 224)


def load_dataset(dataset_dir=DATASET_DIR):
    """Walk dataset_dir and return (image_paths, labels).

    Each image's label is the name of its immediate parent folder
    (e.g. images/cats/xyz.jpg -> label "cats").
    """
    image_paths, labels = [], []
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(root, file))
                labels.append(os.path.basename(root))
    return image_paths, labels


def build_feature_extractor():
    """Load VGG16 (ImageNet weights, no classification head) as a feature extractor."""
    return VGG16(weights="imagenet", include_top=False, pooling="avg")


def load_image_array(img_path, target_size=IMAGE_SIZE):
    img = keras_image.load_img(img_path, target_size=target_size)
    return keras_image.img_to_array(img)


def extract_features(model, image_paths, batch_size=32):
    """Run VGG16 over image_paths in batches and stack the resulting feature vectors."""
    features_list = []
    batch = []
    for i, path in enumerate(image_paths):
        batch.append(load_image_array(path))
        if len(batch) == batch_size or i == len(image_paths) - 1:
            batch_array = preprocess_input(np.array(batch))
            features_list.append(model.predict(batch_array, verbose=0))
            batch = []
    return np.vstack(features_list)


def load_or_extract_features(model, dataset_dir=DATASET_DIR):
    """Load cached features from disk, or extract and cache them if missing."""
    if os.path.exists(FEATURES_FILE):
        features = np.load(FEATURES_FILE)
        labels = np.load(LABELS_FILE, allow_pickle=True)
        image_paths = np.load(IMAGE_PATHS_FILE, allow_pickle=True).tolist()
        print(f"Loaded cached features for {len(image_paths)} images.")
        return features, labels, image_paths

    image_paths, labels = load_dataset(dataset_dir)
    print(f"Found {len(image_paths)} images. Extracting VGG16 features...")
    features = extract_features(model, image_paths)

    np.save(FEATURES_FILE, features)
    np.save(LABELS_FILE, np.array(labels))
    np.save(IMAGE_PATHS_FILE, np.array(image_paths))
    print("Feature extraction complete.")
    return features, labels, image_paths


def predict_image_class(model, features, labels, image_paths, img_path, k=5, metric="cosine", show=True):
    """Classify img_path against the cached feature bank and optionally plot its nearest neighbors."""
    x = np.expand_dims(load_image_array(img_path), axis=0)
    x = preprocess_input(x)
    test_feature = model.predict(x, verbose=0).flatten()

    knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
    knn.fit(features, labels)
    predicted_label = knn.predict([test_feature])[0]
    _, indices = knn.kneighbors([test_feature], n_neighbors=k)

    print(f"\n'{img_path}' predicted as: {predicted_label.upper()} (metric={metric})")
    print("Nearest neighbors:")
    for i in indices[0]:
        print(f"  - {image_paths[i]} -> {labels[i]}")

    if show:
        plt.figure(figsize=(3 * (k + 1), 3))
        plt.subplot(1, k + 1, 1)
        plt.imshow(Image.open(img_path))
        plt.title("Query")
        plt.axis("off")
        for j, idx in enumerate(indices[0]):
            plt.subplot(1, k + 1, j + 2)
            plt.imshow(Image.open(image_paths[idx]))
            plt.title(labels[idx])
            plt.axis("off")
        plt.show()

    return predicted_label


def parse_args():
    parser = argparse.ArgumentParser(description="Classify an image using VGG16 features + KNN.")
    parser.add_argument(
        "images", nargs="*", default=["testdog.jpg", "testcat.jpg"],
        help="Path(s) to the image(s) to classify (default: testdog.jpg testcat.jpg).",
    )
    parser.add_argument("-k", type=int, default=5, help="Number of neighbors (default: 5).")
    parser.add_argument(
        "--metric", choices=["cosine", "euclidean"], default=None,
        help="Distance metric. Default: run both cosine and euclidean.",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Skip plotting (useful in headless/CI environments).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model = build_feature_extractor()
    features, labels, image_paths = load_or_extract_features(model)

    metrics = [args.metric] if args.metric else ["cosine", "euclidean"]
    for img_path in args.images:
        for metric in metrics:
            predict_image_class(
                model, features, labels, image_paths, img_path,
                k=args.k, metric=metric, show=not args.no_show,
            )


if __name__ == "__main__":
    main()
