import onnxruntime as ort
from torchvision import transforms
from PIL import Image
import numpy as np
import gradio as gr
import os

# Определение классов
class_names = ['cat', 'fruit', 'monkey']

# Трансформации
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Путь к модели
base_dir = os.path.dirname(os.path.abspath(__file__))
onnx_path = os.path.join(base_dir, '..', 'notebooks', 'models', 'resnet18_bs16_lr0.005.onnx')

# Загрузка модели
ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

# Функция softmax
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=1, keepdims=True)

# Функция предсказания
def predict(image):
    # Преобразование изображения
    img = Image.fromarray(image)
    img = test_transforms(img).numpy()
    img = np.expand_dims(img, axis=0)
    
    # Предсказание
    ort_inputs = {ort_session.get_inputs()[0].name: img}
    ort_outs = ort_session.run(None, ort_inputs)
    
    # Получаем вероятности
    logits = ort_outs[0]
    probabilities = softmax(logits)[0]
    
    # Находим предсказанный класс
    pred_idx = np.argmax(probabilities)
    pred_class = class_names[pred_idx]
    
    # Формируем текстовый вывод с процентами
    result = f"**Предсказанный класс: {pred_class}**\n\n"
    result += "**Вероятности по классам:**\n"
    for class_name, prob in zip(class_names, probabilities):
        percentage = prob * 100
        marker = "+" if class_name == pred_class else ""
        result += f"- **{class_name}**: {percentage:.2f}% {marker}\n"
    
    return result

iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Markdown(label="Результат классификации"),
    title="Классификатор: Обезьяна, Кот или Фрукт",
    description="Загрузите изображение"
)

# Запуск
iface.launch()
