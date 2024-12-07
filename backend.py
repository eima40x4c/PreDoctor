from tensorflow.keras.models import load_model
from joblib import load

kidney_model = load_model('Models/kidney_classifier.h5')
leukemia_model = load_model("Models/leukemia_classifier.h5")


def breast_setup(sample: dict):
    breast_model_data = load("Models/breast_classifier.pkl")
    breast_model = breast_model_data['model']

    means = breast_model_data['column_means']
    stds = breast_model_data['column_stds']
    sample = (sample - means) / stds

    prediction = breast_model.predict(sample)
    if prediction == 1:
        prediction = "YES"
    else: 
        prediction = "NO"


def lung_setup(sample: dict):
    lung_model = load_model("Models/Lung/model.h5")
    mean, std = load("Models/Lung/metadata.pkl").values()
    sample['age'] = (sample['age'] - mean) / std
    
    if sample['GENDER'] == 'M': 
        sample['GENDER'] = 1
    else:
        sample['GENDER'] = 0

    prediction = lung_model.predict(sample) # Unlabeled
    if prediction == 1:
        prediction = "YES"
    else: 
        prediction = "NO"


def rna_setup(sample: dict):
    rna_model_data = load_model("Models/rna-seq_classifier.h5")
    rna_model = rna_model_data['model']
    pca = rna_model_data['pca']

    low_variance_mask = rna_model_data['low_variance_mask']
    means = rna_model_data['column_means']
    stds = rna_model_data['column_stds']
    
    filtered_features = sample.iloc[:, ~low_variance_mask]
    scaled_sample = (filtered_features - means) / stds
    reduced_data = pca.transform(scaled_sample)

    prediction = rna_model.predict(reduced_data) # Labeled



def kidney_setup(): 
    pass