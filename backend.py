import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import warnings

from joblib import load
from keras.models import load_model
from PIL import Image

warnings.filterwarnings('ignore')
models = {}

app = FastAPI()

class PredictionRequest(BaseModel):
    input_data_path: str 
    model_type: str


@app.on_event("startup")
def load_models():
    global models
    models_path = ['breast_classifier.pkl', 'rna-seq_classifier.pkl', 'leukemia_classifier.h5',
                   'lung_classifier.h5', 'kidney_classifier.h5', 'skin_classifier.h5']
    for model_path in models_path:
        full_path = "Models/" + model_path
        with open(full_path, "rb") as file:
            if full_path.endswith('.pkl'):
                models[model_path] = load(file)
            else:
                models[model_path] = load_model(full_path)


@app.post("/predict/")
def predict(request: PredictionRequest):
    if request.model_type == "breast":
        prediction = breast_prediction(pd.read_csv(request.input_data_path))
    elif request.model_type == "rna-seq":
        prediction = rna_prediction(pd.read_csv(request.input_data_path))
    elif request.model_type == "lung":
        prediction = lung_prediction(pd.read_csv(request.input_data_path))
    elif request.model_type == "leukemia":
        prediction = leukemia_prediction(preprocess_image(request.input_data_path))
    elif request.model_type == "kidney":
        prediction = kidney_prediction(preprocess_image(request.input_data_path))
    elif request.model_type == "skin":
        prediction = skin_prediction(preprocess_image(request.input_data_path, scaled=True))
    
    if isinstance(prediction, (np.ndarray)):
            prediction = prediction.tolist()  
    return {"prediction": prediction} # Convert to list for JSON serialization


def breast_prediction(sample: pd.DataFrame) -> str:
    breast_model_data = models["breast_classifier.pkl"]
    breast_model = breast_model_data['model']

    prediction = breast_model.predict(sample.to_numpy().reshape(1, -1))
    if prediction == 1:
        return ["Malignant"]
    else: 
        return ["Benign"]


def rna_prediction(sample: pd.DataFrame) -> str:
    rna_model_data = models['rna-seq_classifier.pkl']
    rna_model = rna_model_data['model']
    pca = rna_model_data['pca']
    scaler = rna_model_data['std_scaler']
    
    scaled_sample = scaler.transform(sample.to_numpy().reshape(1, -1))
    reduced_data = pca.transform(scaled_sample)

    return rna_model.predict(reduced_data) # Labeled


def lung_prediction(sample: pd.DataFrame) -> str:
    lung_model = models['lung_classifier.h5']
    scaler = load("Models/lung_metadata.pkl")['std_scaler']
    scaled_sample = scaler.transform(sample.to_numpy().reshape(1, -1))

    prediction = lung_model.predict(scaled_sample) # Unlabeled
    if prediction >= 0.5:
        return "Malignant"
    else: 
        return "Benign"


def leukemia_prediction(image: Image.Image) -> str:
    model = models['leukemia_classifier.h5']
    prediction = model.predict(image)
    print(prediction)
    if prediction >= 0.5:
        return "Malignant"
    else: 
        return "Benign"


def kidney_prediction(image: Image.Image) -> str: 
    model = models['kidney_classifier.h5']
    prediction = model.predict(image)
    if prediction >= 0.5:
        return "Malignant"
    else: 
        return "Benign"


def skin_prediction(image: Image.Image) -> str:
    model = models['skin_classifier.h5']
    class_names = ['actinic keratosis', 'basal cell carcinoma', 'dermatofibroma',
                   'melanoma', 'nevus', 'pigmented benign keratosis', 'seborrheic keratosis',
                   'squamous cell carcinoma', 'vascular lesion']

    prediction = np.argmax(model.predict(image))
    return class_names[prediction]


def preprocess_image(file_path, scaled=False):
    image = Image.open(file_path)
    if image.mode != 'RGB':
            image = image.convert('RGB')

    image = image.resize((224, 224))
    image_array = np.array(image)
    if not scaled:
        image_array = image_array / 255.0
    return image_array.reshape(1, 224, 224, 3)

