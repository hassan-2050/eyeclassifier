# Import Packages
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torchvision import models, transforms
import torch.nn as nn
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Display logo at the top
logo = Image.open('hh.png')  # Replace with the path to your logo image
st.image(logo, width=700)  # Adjust the width as needed

# Title and sidebar
st.header('Eye Disease Classifier')
st.sidebar.subheader("Input a Disease Picture")

# Load Model
def effNetb3():
    model = models.efficientnet_b3(pretrained=False).to(device)
    in_features = 1024
    model._fc = nn.Sequential(
        nn.BatchNorm1d(num_features=in_features),    
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.BatchNorm1d(num_features=128),
        nn.Dropout(0.4),
        nn.Linear(128, 4),
    ).to(device)

    model.load_state_dict(torch.load('Weights.h5', map_location=torch.device('cpu')))
    model.eval()

    return model

# Calculating Prediction
def Predict(img):
    allClasses = ['Cataract', 'Diabetic_Retinopathy', 'Glaucoma', 'Normal']
    Mod = effNetb3()
    out = Mod(img)
    _, predicted = torch.max(out.data, 1)
    allClasses.sort()
    labelPred = allClasses[predicted]
    return labelPred

# Title bar for uploading image
st.title('Upload an Image for Classification')
file_up = st.file_uploader('Upload an Image', type="png")

# Normalizing
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# Transforming the Image
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])

if file_up is not None:
    # Display image that user uploaded
    image = Image.open(file_up).convert('RGB')
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Just a second ...")
    img = data_transform(image)
    img = torch.reshape(img, (1, 3, 224, 224))
    prob = Predict(img)
    st.write(f"Prediction: {prob}")
