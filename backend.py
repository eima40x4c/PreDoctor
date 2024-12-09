from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File
import numpy as np
from joblib import load
from keras.models import load_model


models = {}
class PredictionRequest(BaseModel):
    input_data: list  # List of input features for prediction
    model_type: str  # e.g., "deep" or "ml" to distinguish the model


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
    print(models)


@app.post("/predict/")
def predict(request: PredictionRequest):
    input_data = np.array(request.input_data).reshape(1, -1)
    if request.model_type == "breast":
        prediction = breast_prediction()
    elif request.model_type == "rna-seq":
        prediction = rna_prediction()
    elif request.model_type == "lung":
        prediction = lung_prediction()
    elif request.model_type == "leukemia":
        prediction = leukemia_prediction()
    elif request.model_type == "kidney":
        prediction = kidney_prediction()
    elif request.model_type == "skin":
        prediction = skin_prediction()
    
    return {"prediction": prediction}


def breast_prediction(sample: dict):
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


def rna_prediction(sample: dict):
    rna_model_data = models['rna-seq_classifier.pkl']
    rna_model = rna_model_data['model']
    pca = rna_model_data['pca']

    low_variance_mask = rna_model_data['low_variance_mask']
    means = rna_model_data['column_means']
    stds = rna_model_data['column_stds']
    
    filtered_features = sample.iloc[:, ~low_variance_mask]
    scaled_sample = (filtered_features - means) / stds
    reduced_data = pca.transform(scaled_sample)

    return rna_model.predict(reduced_data) # Labeled


def lung_prediction(sample: dict):
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


def kidney_prediction(): 
    pass


def skin_prediction():
    pass


def leukemia_prediction():
    pass
