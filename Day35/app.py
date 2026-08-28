import streamlit as st
import torch
import torch.nn as nn

from torchvision import models, transforms
from PIL import Image

classes = [
    "Tomato___Early_blight",
    "Tomato___healthy",
    "Tomato___Late_blight"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    len(classes)
)

torch.load(
    "Day34/resnet18_tomato.pth",
    map_location=device
)

model = model.to(device)

model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

st.title("🍅 Tomato Leaf Disease Classifier")

st.write(
    "Upload a tomato leaf image to predict its disease."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    input_tensor = transform(image)
    input_tensor = input_tensor.unsqueeze(0)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():

        outputs = model(input_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted = torch.max(
            probabilities,
            1
        )

    predicted_class = classes[predicted.item()]
    confidence_value = confidence.item() * 100

    st.subheader("Prediction")

    st.success(
        f"{predicted_class}"
    )

    st.write(
        f"Confidence: {confidence_value:.2f}%"
    )
