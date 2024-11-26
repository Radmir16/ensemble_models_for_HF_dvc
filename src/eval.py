import os
import argparse
import yaml, json
import numpy as np
from transformers import AutoModelForImageClassification, ViTFeatureExtractor, pipeline, ViTImageProcessor
import datasets
from datasets import load_metric, load_dataset
import dvclive
import evaluate
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
from markdownwriter import MarkdownWriter
import pandas as pd
from markdownwriter import *
import seaborn as sns
from preprocess import cust_load_model

LABEL_MAP = {
    "MEL": 0, "NOTMEL": 1,
}
# Загружаем несколько моделей для классификации изображений
def load_model(model_path):
    model = AutoModelForImageClassification.from_pretrained(model_path)
    return model

def save_classification_report(live, references, predictions, labels):
    # Generate the classification report as a dictionary
    report = classification_report(references, predictions, target_names=labels, output_dict=True)
    
    # Specify the path for the JSON file
    report_path = "classification_report.json"
    
    # Serialize the report dictionary to a JSON formatted string and save it to a file
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)  # `indent=4` for pretty-printing
    
    # Log the JSON file as an artifact with DVCLive
    live.log_artifact(report_path)

def save_confusion_matrix(live, references, predictions, labels):
    cm = confusion_matrix(references, predictions, labels=list(LABEL_MAP.values()))
    plt.figure(figsize=(10, 7))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("confusion_matrix.png")
    plt.close()
    live.log_artifact("confusion_matrix.png")

def evaluate_model(data_path, model_path, params_file):
    # Load parameters and model
    with open(params_file, 'r') as f:
        params = yaml.safe_load(f)
        
    # Log metrics with dvclive
    live = dvclive.Live("dvceval", report='md')
    
    model1 = load_model(model_path+'model_mel_notmel.pkl')
    model2 = load_model(model_path+'model_mel_nv.pkl')
    feature_extractor1 = ViTImageProcessor.from_pretrained(model_path+'model_mel_notmel.pkl')
    feature_extractor2 = ViTImageProcessor.from_pretrained(model_path+'model_mel_nv.pkl')

    # Prepare data and evaluation pipeline
    full_dataset = cust_load_model(data_path)
    test_dataset = full_dataset["test"]
    
    eval_pipeline1 = pipeline("image-classification", model=model1, feature_extractor=feature_extractor1)
    eval_pipeline2 = pipeline("image-classification", model=model2, feature_extractor=feature_extractor2)

    # Collect predictions and references with a progress bar
    predictions1 = []
    references1 = []
    predictions2 = []
    references2 = []
    label_map_1 = params['model_mel_notmel']["label_map"]
    label_map_2 = params['model_mel_nv']["label_map"]

    for example in tqdm(test_dataset, desc="Processing Images", leave=True):
        result1 = eval_pipeline1(example["image"])
        prediction_label1 = result1[0]['label'].split('_')[-1]
        prediction1 = label_map_1.get(prediction_label1,prediction_label1)
        reference1 = label_map_1.get(example['class_label'],example['class_label'])
        predictions1.append(prediction1)
        references1.append(reference1)
        if prediction_label1=='MEL':
            result2 = eval_pipeline2(example['image'])
            prediction_label2 = result2[0]['label'].split('_')[-1]
            prediction2 = label_map_2.get(prediction_label2,prediction_label2)
            reference2 = label_map_2.get(example['subclass_label'],example['subclass_label'])
            predictions2.append(prediction2)
            references2.append(reference2)


    f1_m1 = f1_score(references1, predictions1, average='macro')  # You can change average to 'micro', 'weighted', or None
    precision_m1 = precision_score(references1, predictions1, average='macro')
    accuracy_m1 = accuracy_score(references1, predictions1)

    f1_m2 = f1_score(references2, predictions2, average='macro')  # You can change average to 'micro', 'weighted', or None
    precision_m2 = precision_score(references2, predictions2, average='macro')
    accuracy_m2 = accuracy_score(references2, predictions2)

    print(f"F1_model_1 Score (Macro): {f1_m1}")
    print(f"F1_model_2 Score (Macro): {f1_m2}")
    # Log F1 Score and other metrics
    live.log_metric("f1_macro_model_1", f1_m1)
    live.log_metric("precision_macro_model_1", precision_m1)
    live.log_metric("accuracy_macro_model_1", accuracy_m1)
    live.log_metric("f1_macro_model_2", f1_m2)
    live.log_metric("precision_macro_model_2", precision_m2)
    live.log_metric("accuracy_macro_model_2", accuracy_m2)

    # Classification report and confusion matrix model1
    cr1 = classification_report(references1, predictions1, target_names=list(label_map_1.keys()), zero_division=0, output_dict=True)
    cm1 = confusion_matrix(references1, predictions1, labels=list(label_map_1.values()))
    
    # Classification report and confusion matrix model2
    cr2 = classification_report(references2, predictions2, target_names=list(label_map_2.keys()), zero_division=0, output_dict=True)
    cm2 = confusion_matrix(references2, predictions2, labels=list(label_map_2.values()))
    
    # Convert classification report and confusion matrix to DataFrame for better formatting
    report_df1 = pd.DataFrame(cr1).transpose()
    matrix_df1 = pd.DataFrame(cm1)

    report_df2 = pd.DataFrame(cr2).transpose()
    matrix_df2 = pd.DataFrame(cm2)

    md_writer = MarkdownWriter(f"report/report_ans.md")
    #md_writer.header1('Report')

    md_writer.header2('Params')

    md_writer.print_config(params)

    train_description = describe_dataset('data/preprocessed/train/')
    test_description = describe_dataset('data/preprocessed/test/')
    md_writer.describe_dataset_markdown(train_description, "Training Dataset Description")
    md_writer.describe_dataset_markdown(test_description, "Testing Dataset Description")

    md_writer.header2('Metrics')

    md_writer.print_data(f"F1 Score (Macro) Model1: **{f1_m1}**\n")
    md_writer.print_data(f"Accuracy Model1: **{cr1['accuracy']}**\n")

    md_writer.print_data(f"F1 Score (Macro) Model2: **{f1_m2}**\n")
    md_writer.print_data(f"Accuracy Model2: **{cr2['accuracy']}**\n")

    md_writer.header2('Classification Report')

    md_writer.print_data(report_df1.to_markdown(index=True) + '\n\n')
    md_writer.print_data(report_df2.to_markdown(index=True) + '\n\n')

    md_writer.header2('Confusion Matrix')
    matrix_df1.index = [key for key, value in sorted(label_map_1.items(), key=lambda item: item[1])]
    matrix_df1.columns = [key for key, value in sorted(label_map_1.items(), key=lambda item: item[1])]
    md_writer.print_data(matrix_df1.to_markdown(index=True) + '\n')

    md_writer.header2('Confusion Matrix Model 2')
    matrix_df2.index = [key for key, value in sorted(label_map_2.items(), key=lambda item: item[1])]
    matrix_df2.columns = [key for key, value in sorted(label_map_2.items(), key=lambda item: item[1])]
    md_writer.print_data(matrix_df2.to_markdown(index=True) + '\n')


    # Compute confusion matrix
    cm1 = confusion_matrix(references1, predictions1)
    cm2 = confusion_matrix(references2, predictions2)

    # Define the plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', xticklabels=list(label_map_1.keys()), yticklabels=list(label_map_1.keys()))
    plt.title('Confusion Matrix Model 1')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    # Save the plot model1 as an image file
    plt.savefig('report/confusion_matrix_m1.png', bbox_inches='tight')

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm2, annot=True, fmt='d', cmap='Blues', xticklabels=list(label_map_2.keys()), yticklabels=list(label_map_2.keys()))
    plt.title('Confusion Matrix Model 2')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    # Save the plot model2 as an image file
    plt.savefig('report/confusion_matrix_m2.png', bbox_inches='tight')

    md_writer.add_image('report/confusion_matrix_m1.png', 'Confusion matrix')
    md_writer.add_image('report/confusion_matrix_m2.png', 'Confusion matrix')
    # md_writer.add_image('dvctrain/static/eval/accuracy.png', 'Accuracy during the training')
    # md_writer.add_image('dvctrain/static/eval/f1.png', 'F1Score during the training')
    # md_writer.add_image('dvctrain/static/eval/loss.png', 'Loss during the training')
    md_writer.write_toc('Report '+ params['model_mel_notmel']['arch']+" "+ params['model_mel_nv']['arch'])
    
    live.next_step()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained Hugging Face model")
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data directory')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model directory')
    parser.add_argument('--params_file', type=str, required=True, help='Path to the params YAML file')
    args = parser.parse_args()

    evaluate_model(args.data_path, args.model_path, args.params_file)