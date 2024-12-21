import os
import pickle
import time

import cv2
from sklearn import svm
from sklearn.model_selection import train_test_split

# Data and model directory
DATA_DIR = "data"
MODEL_PATH = "model/vehicle_recognition_model.pkl"

def extract_features(image: cv2.typing.MatLike):
    # Preprocess and extract features
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    resized = cv2.resize(blurred, (128, 128))
    edges = cv2.Canny(resized, 50, 150)
    features = edges.flatten()
    return features

def load_training_data(data_dir: str):
    # Load and labeling training data
    training_data = []
    labels: list[str] = []

    for label in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label)
        if os.path.isdir(label_path):
            for filename in os.listdir(label_path):
                file_path = os.path.join(label_path, filename)
                image = cv2.imread(file_path)
                if image is not None:
                    features = extract_features(image)
                    training_data.append(features)
                    labels.append(label)

    return training_data, labels

def train_SVM_model(training_data, labels):
    X_train, X_test, y_train, y_test = train_test_split(training_data, labels, test_size=0.2, random_state=42)
    
    SVM = svm.SVC(kernel='linear', C=4)
    SVM.fit(X_train, y_train)

    accuracy = SVM.score(X_test, y_test)
    print(f"Model accuracy: {accuracy * 100:.2f}%")

    return SVM

def save_model(model, path: str):
    os.makedirs("model", exist_ok=True)
    with open(path, "wb") as model_file:
        pickle.dump(model, model_file)

if __name__ == '__main__':
    print("Loading training data")
    start_time = time.time()
    training_data, labels = load_training_data(DATA_DIR)
    print(f"Training data loaded in {time.time() - start_time:.2f} seconds")

    print("Training SVM Model")
    start_time = time.time()
    model = train_SVM_model(training_data, labels)
    print(f"Model trained in {time.time() - start_time:.2f} seconds")

    print("Saving model")
    start_time = time.time()
    save_model(model, MODEL_PATH)
    print(f"Model saved in {time.time() - start_time:.2f} seconds")

    print(f"Model saved in {MODEL_PATH}")
