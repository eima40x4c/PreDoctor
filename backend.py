from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File
import numpy as np
import pandas as pd

from joblib import load
from keras.models import load_model
from PIL import Image


models = {}
class PredictionRequest(BaseModel):
    input_data_path: str 
    model_type: str


app = FastAPI()


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
        prediction = leukemia_prediction()
    elif request.model_type == "kidney":
        prediction = kidney_prediction()
    elif request.model_type == "skin":
        prediction = skin_prediction()
    
    return {"prediction": prediction.tolist()} # Convert to list for JSON serialization


def breast_prediction(sample: pd.DataFrame) -> str:
    breast_model_data = models["breast_classifier.pkl"]
    breast_model = breast_model_data['model']

    means = breast_model_data['column_means']
    stds = breast_model_data['column_stds']
    sample = (sample - means) / stds

    prediction = breast_model.predict(sample)
    if prediction == 1:
        return "YES"
    else: 
        return "NO"


def rna_prediction(sample: pd.DataFrame) -> str:
    rna_model_data = models['rna-seq_classifier.pkl']
    rna_model = rna_model_data['model']
    pca = rna_model_data['pca']
    scaler = rna_model_data['std_scaler']
    
    scaled_sample = scaler.transform(sample.to_numpy().reshape(1, -1))
    reduced_data = pca.transform(scaled_sample)

    return rna_model.predict(reduced_data) # Labeled


def lung_prediction(sample: np.ndarray) -> str:
    lung_model = models['lung_classifier.h5']
    mean, std = load("Models/lung_metadata.pkl").values()
    sample['age'] = (sample['age'] - mean) / std
    
    if sample['GENDER'] == 'M': 
        sample['GENDER'] = 1
    else:
        sample['GENDER'] = 0

    prediction = lung_model.predict(sample) # Unlabeled
    if prediction == 1:
        return "YES"
    else: 
        return "NO"

# TODO: Complete
def kidney_prediction() -> str: 
    return "Placeholder"

# TODO: Complete
def skin_prediction() -> str:
    return "Also a placeholder"

# TODO: Complete
def leukemia_prediction() -> str:
    return "A placeholder.. Do you really think it's that easy?!"


# TODO: Adjust as needed
def read_image(file_path):
    image = Image.open(file_path)
    image_array = np.array(image.resize((224, 224)))  # Resize if needed
    return image_array.reshape(1, -1)  # Reshape for prediction