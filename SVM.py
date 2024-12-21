import pickle

import cv2

MODEL_PATH = "model/vehicle_recognition_model.pkl"

def extract_features(image: cv2.typing.MatLike):
    # Preprocess and extract features
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    resized = cv2.resize(blurred, (128, 128))
    edges = cv2.Canny(resized, 50, 150)
    features = edges.flatten()
    return features

def load_model(path: str):
    with open(path, "rb") as model_file:
        return pickle.load(model_file)
    
def predict(img):
    model = load_model(MODEL_PATH)
    
    features = extract_features(img)
    prediction = model.predict([features])[0]
        
    return (prediction, img)
