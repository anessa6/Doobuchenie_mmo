from PIL import Image
import os

# Путь к исходной папке с изображениями
input_dir = '../data/raw'  
output_dir = '../data/processed'  
os.makedirs(output_dir, exist_ok=True)

# Классы
classes = ['fruit', 'cat', 'monkey']

for cls in classes:
    class_input_dir = os.path.join(input_dir, cls)
    class_output_dir = os.path.join(output_dir, cls)
    os.makedirs(class_output_dir, exist_ok=True)
    
    # Обработка изображения
    for filename in os.listdir(class_input_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  # Поддерживаемые форматы
            img_path = os.path.join(class_input_dir, filename)
            img = Image.open(img_path).convert('RGB')  # Преобразуем в RGB
            img_resized = img.resize((224, 224), Image.Resampling.LANCZOS)  # Ресайзинг с качеством
            output_path = os.path.join(class_output_dir, filename)
            img_resized.save(output_path)
            print(f'Обработано: {filename}')

print('Все изображения приведены к 224x224!')