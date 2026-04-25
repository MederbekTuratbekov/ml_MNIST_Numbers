from fastapi import HTTPException
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from torchvision import transforms
import streamlit as st
import torch.nn as nn
import uvicorn
import torch
import io


app = FastAPI()

class CheckImage(nn.Module):
    def __init__(self):
        super().__init__()

        self.first = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.second = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 14 * 14, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    def forward(self, x):
        x = self.first(x)
        x = self.second(x)
        return x

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CheckImage()
model.load_state_dict(torch.load('mnist_numbers.pth', map_location=device))
model.to(device)
model.eval()

# @app.post('/predict')
# async def check_image(file: UploadFile = File()):
#     try:
#         image_bytes = await file.read()
#         if not image_bytes:
#             raise HTTPException(400, detail='Файл кошулган жок')
#
#         image = Image.open(io.BytesIO(image_bytes))
#         image_tensor = transform(image).unsqueeze(0).to(device)
#
#         with torch.no_grad():
#             y_prediction = model(image_tensor)
#             prediction = y_prediction.argmax(dim=1).item()
#         return {'Answer': prediction}
#
#     except Exception as e:
#         raise HTTPException(500, detail=str(e))
#
#
# if __name__ == '__main__':
#     uvicorn.run(app, host='127.0.0.1', port=8000)

st.title('Mnist Numbers Classifier Model')
st.text('Загрузите изображение с цифрой, и модель попробует её распознать.')

mnist_image = st.file_uploader('Выберите изображение', type=['PNG', 'JPG', 'JPEG', 'SVG'])

if not mnist_image:
    st.info('Загрузите изображение')
else:
    st.image(mnist_image, caption='Загруженное изображение')

    if st.button('Распознать цифру'):
        try:
            image = Image.open(mnist_image)
            image_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                y_prediction = model(image_tensor)
                prediction = y_prediction.argmax(dim=1).item()
            st.success(f'Модель думает, что это: {prediction}')

        except Exception as e:
            st.error(f'Ошибка: {str(e)}')
