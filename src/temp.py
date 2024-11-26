import os 
import glob 
from PIL import Image 
# Функция для сбора путей к изображениям и меток 
import os
import glob
from PIL import Image
from datasets import Dataset, DatasetDict  # Убедитесь, что у вас установлен пакет datasets

def load_image_paths_and_labels(data_dir): 
    dataset = DatasetDict()
    
    # Проходим по каждому подкаталогу train/test
    for train_test in os.listdir(data_dir):
        train_test_path = os.path.join(data_dir, train_test)
        
        if not os.path.isdir(train_test_path):
            continue  # Пропускаем, если это не директория
        
        image_paths = [] 
        class_labels = [] 
        subclass_labels = [] 
        
        # Проходим по каждому классу в директории train/test
        for class_dir in os.listdir(train_test_path):
            class_path = os.path.join(train_test_path, class_dir) 
            
            if not os.path.isdir(class_path):
                continue  # Пропускаем, если это не директория
            
            if class_dir == 'MEL':
                # Обрабатываем подклассы для MEL
                for subclass_dir in os.listdir(class_path): 
                    subclass_path = os.path.join(class_path, subclass_dir) 
                    
                    if os.path.isdir(subclass_path): 
                        for image_file in glob.glob(os.path.join(subclass_path, '*.jpg')):
                            image_paths.append(image_file)  # Сохраняем путь к изображению
                            class_labels.append(class_dir)
                            subclass_labels.append(subclass_dir) 
            else:
                # Обрабатываем классы без подклассов
                for image_file in glob.glob(os.path.join(class_path, '*.jpg')):
                    image_paths.append(image_file)  # Сохраняем путь к изображению
                    class_labels.append(class_dir)
                    subclass_labels.append(class_dir) 

        # Создаем датасет для текущего train/test
        dat = Dataset.from_dict({
            'image': image_paths,
            'class_label': class_labels,
            'subclass_label': subclass_labels
        })
        dat = dat.class_encode_column('subclass_label')
        dat = dat.class_encode_column('class_label')
        dataset[train_test] = dat  # Сохраняем датасет в общий словарь

    return dataset
                        # Загрузка данных 
data_dir = '/home/jovyan/ensemble-of-models/data/preprocessed' 
# Создание Dataset
dataset = load_image_paths_and_labels(data_dir)
for example in dataset['test']:
    print(example)