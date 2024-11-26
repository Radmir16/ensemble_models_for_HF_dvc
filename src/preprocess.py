import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
from dvclive import Live
import csv
import glob
from tqdm import tqdm
from datasets import DatasetDict, Dataset

def cust_load_model(data_dir): #кастомный способ загрузки датасета 
    dataset = DatasetDict()

    for train_test in os.listdir(data_dir):
        train_test_path = os.path.join(data_dir, train_test)
        
        if not os.path.isdir(train_test_path):
            continue
        
        image_paths = [] 
        class_labels = [] 
        subclass_labels = [] 
        
        for class_dir in os.listdir(train_test_path):
            class_path = os.path.join(train_test_path, class_dir) 
            
            if not os.path.isdir(class_path):
                continue 
            
            if class_dir == 'MEL':
                for subclass_dir in os.listdir(class_path): 
                    subclass_path = os.path.join(class_path, subclass_dir) 
                    
                    if os.path.isdir(subclass_path): 
                        for image_file in glob.glob(os.path.join(subclass_path, '*.jpg')):
                            image_paths.append(Image.open(image_file))
                            class_labels.append(class_dir)
                            subclass_labels.append(subclass_dir) 
            else:
                for image_file in glob.glob(os.path.join(class_path, '*.jpg')):
                    image_paths.append(Image.open(image_file))
                    class_labels.append(class_dir)
                    subclass_labels.append(class_dir) 

        dat = Dataset.from_dict({
            'image': image_paths,
            'class_label': class_labels,
            'subclass_label': subclass_labels
        })
        dataset[train_test] = dat

    return dataset
# TODO: crop should be done for model params
def resize_and_normalize(image_path, output_path, size=(224, 224), format='JPEG'):
    image = Image.open(image_path)
    image = image.resize(size)
    if image.mode in ['RGBA', 'P']:
        image = image.convert('RGB')
    image.save(output_path, format=format)


def preprocess_dataset(input_dir, output_dir, csv_path, test_size=0.2):
    os.makedirs(output_dir, exist_ok=True)
    data = pd.read_csv(csv_path)

    # Initialize DVCLive
    with Live("dvcpreproc") as live:
        total_images_processed = 0

        # Create directories for each label
        for label in data['diagnosis'].unique():
            if label == 'MEL':
                os.makedirs(os.path.join(output_dir, 'train', label,"MEL"), exist_ok=True)
                os.makedirs(os.path.join(output_dir, 'train', label,"NV"), exist_ok=True)
                os.makedirs(os.path.join(output_dir, 'test', label,"NV"), exist_ok=True)
                os.makedirs(os.path.join(output_dir, 'test', label,"MEL"), exist_ok=True)
            else:
                os.makedirs(os.path.join(output_dir, 'train', label), exist_ok=True)
                os.makedirs(os.path.join(output_dir, 'test', label), exist_ok=True)


        # Split data into training and testing
        train_df, test_df = train_test_split(data, test_size=test_size,random_state=22)


        # Process train images with a progress bar
        for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc='Processing train images'):
            image_path = os.path.join(input_dir, row['image_name'] + '.jpg')
            if str(row['melanocit']) != 'nan':
                if os.path.exists(image_path):
                    output_path = os.path.join(output_dir, 'train', row['diagnosis'], str(row['melanocit']), row['image_name'] + '.jpg')
                    resize_and_normalize(image_path, output_path)
                    total_images_processed += 1
                    live.log_metric("train_images_processed", total_images_processed)
            else:
                if os.path.exists(image_path):
                    output_path = os.path.join(output_dir, 'train', row['diagnosis'], row['image_name'] + '.jpg')
                    resize_and_normalize(image_path, output_path)
                    total_images_processed += 1
                    live.log_metric("train_images_processed", total_images_processed)

        # Process test images with a progress bar
        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc='Processing test images'):
            image_path = os.path.join(input_dir, row['image_name'] + '.jpg')
            if str(row['melanocit']) != 'nan':
                if os.path.exists(image_path):
                    output_path = os.path.join(output_dir, 'test', row['diagnosis'], str(row['melanocit']), row['image_name'] + '.jpg')
                    resize_and_normalize(image_path, output_path)
                    total_images_processed += 1
                    live.log_metric("train_images_processed", total_images_processed)
            else:
                if os.path.exists(image_path):
                    output_path = os.path.join(output_dir, 'test', row['diagnosis'], row['image_name'] + '.jpg')
                    resize_and_normalize(image_path, output_path)
                    total_images_processed += 1
                    live.log_metric("train_images_processed", total_images_processed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--csv_path', type=str, required=True)
    args = parser.parse_args()

    preprocess_dataset(args.input_dir, args.output_dir, args.csv_path)

