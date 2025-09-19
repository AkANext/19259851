import cv2
import numpy as np
import os


def extract_features(image):
    """Udtrækker farve-histogram (HSV) ."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8],
                        [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def load_dataset(folder):
    """Indlæser billeder og fra mappen."""
    features = []
    labels = []
    label_map = {}
    label_counter = 0

    for file in os.listdir(folder):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(folder, file)
            image = cv2.imread(path)
            if image is None:
                continue

            image = cv2.resize(image, (128, 128))
            feat = extract_features(image)

            label_name = file.split("_")[0].lower()
            if label_name not in label_map:
                label_map[label_name] = label_counter
                label_counter += 1

            features.append(feat)
            labels.append(label_map[label_name])

    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int32), label_map


def train_classifier(features, labels):
    """Trainer en SVM på features og labels."""
    svm = cv2.ml.SVM_create()
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setC(2.5)
    svm.train(features, cv2.ml.ROW_SAMPLE, labels)
    return svm


def ensure_model_ready():

    train_folder = "train_fruits"
    X, y, label_map = load_dataset(train_folder)
    svm = train_classifier(X, y)
    return svm, label_map


def predict_image(model, label_map, image):
    #Forudsiger klassen for et billede.

    if image is None:
        print(":warning: Image is None")
        return None

    image = cv2.resize(image, (128, 128))
    feat = extract_features(image).reshape(1, -1).astype(np.float32)

    _, result = model.predict(feat)
    class_id = int(result[0][0])

    inv_map = {v: k for k, v in label_map.items()}
    return inv_map[class_id]


if __name__ == "__main__":
    train_folder = "train_fruits"   # mappe med træningsfiler
    test_folder = "test_fruits"     # mappe med testfiler

    print(":open_file_folder: Loading training dataset...")
    X, y, label_map = load_dataset(train_folder)
    print("Classes found:", label_map)

    print(":wrench: Training classifier...")
    svm = train_classifier(X, y)

    print("\n:crystal_ball: Predictions on test images:")
    for file in os.listdir(test_folder):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            test_path = os.path.join(test_folder, file)
            test_img = cv2.imread(test_path)
            predicted = predict_image(svm, label_map, test_img)
            if predicted:
                print(f"{file} → {predicted}")
