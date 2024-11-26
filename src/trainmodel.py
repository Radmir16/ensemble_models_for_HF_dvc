import os
import argparse
import numpy as np
import yaml
from transformers import ViTFeatureExtractor, ViTForImageClassification, TrainingArguments, Trainer, DefaultDataCollator
from transformers import ViTImageProcessor, ViTConfig
from datasets import Dataset, load_metric
from dvclive import Live
from transformers.integrations import DVCLiveCallback
import tqdm
import torch
from datasets import DatasetDict
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    GaussianBlur,
    ToTensor,
    RandomRotation,
)
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer
import evaluate
from PIL import Image
import glob

class TrainModel:
    def __init__(self, label_map, model, output_path, epoch, batch_size, data_path):
        self.LABEL_MAP = label_map
        self.model = model
        self.output_path = output_path
        self.epoch = epoch
        self.batch_size = batch_size
        self.data_path = data_path

    def train_model(self):
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

        def melanocit_preprocess(example): #обработка для классификатора меланоцитов
            return{
                'image' : example['image'],
                'label' : example['class_label']
            }
        def melanoma_preprocess(example): #обработка для классификатора меланом и невусов
            return{
                'image' : example['image'],
                'label' : example['subclass_label']
            }
        train_dataset = cust_load_model(self.data_path)
        if list(self.LABEL_MAP.keys()) == ['MEL','NOTMEL']:
            train_dataset['train'] = train_dataset['train'].map(melanocit_preprocess)
            train_dataset['test'] = train_dataset['test'].map(melanocit_preprocess)
            train_dataset = train_dataset.class_encode_column("label")
        else:
            train_dataset['train'] = train_dataset['train'].map(melanoma_preprocess)
            train_dataset['test'] = train_dataset['test'].map(melanoma_preprocess)
            train_dataset = train_dataset.class_encode_column("label")

        image_processor  = AutoImageProcessor.from_pretrained(self.model)

        normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
        if "height" in image_processor.size:
            size = (image_processor.size["height"], image_processor.size["width"])
            crop_size = size
            max_size = None
        elif "shortest_edge" in image_processor.size:
            size = image_processor.size["shortest_edge"]
            crop_size = (size, size)
            max_size = image_processor.size.get("longest_edge")
        train_transforms = Compose(
                [
                    Resize(size),
                    ToTensor(),
                    normalize,
                ]
            )

        val_transforms = Compose(
                [
                    Resize(size),
                    ToTensor(),
                    normalize,
                ]
            )
        def preprocess_train(example_batch):
            """Apply train_transforms across a batch."""
            example_batch["pixel_values"] = [
                train_transforms(image.convert("RGB")) for image in example_batch["image"]
            ]
            return example_batch

        def preprocess_val(example_batch):
            """Apply val_transforms across a batch."""
            example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
            return example_batch

        train_dataset['train'].set_transform(preprocess_val)
        train_dataset['test'].set_transform(preprocess_val)

        acc_metric = evaluate.load("accuracy", trust_remote_code=True)
        f1_metric = evaluate.load("f1", trust_remote_code=True)

        def collate_fn(examples):
            pixel_values = torch.stack([example["pixel_values"] for example in examples])
            labels = torch.tensor([example["label"] for example in examples])
            return {"pixel_values": pixel_values, "labels": labels}

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            accuracy = acc_metric.compute(predictions=predictions, references=labels)
            f1 = f1_metric.compute(predictions=predictions, references=labels, average='macro')
            return {"accuracy": accuracy['accuracy'], "f1": f1['f1']}

        labels = train_dataset['train'].features["label"].names
        label2id, id2label = dict(), dict()
        for i, label in enumerate(labels):
            label2id[label] = i
            id2label[i] = label


        model = AutoModelForImageClassification.from_pretrained(
        self.model, 
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes = True, # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
        )

        args = TrainingArguments(
        self.output_path,
        remove_unused_columns=False,
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=5e-5,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=8,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        push_to_hub=False,
        per_device_train_batch_size=self.batch_size,
        num_train_epochs=self.epoch,
        report_to="dvclive"
        )
        with Live("dvctrain", report='md') as live:    
            trainer = Trainer(
            model,
            args,
            train_dataset=train_dataset['train'],
            eval_dataset=train_dataset['test'],
            tokenizer=image_processor,
            compute_metrics=compute_metrics,
            data_collator=collate_fn,
            )
            trainer.add_callback(DVCLiveCallback(live=live))
            trainer.train()
        trainer.save_model(self.output_path)



