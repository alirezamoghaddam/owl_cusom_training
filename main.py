from sklearn.model_selection import train_test_split
from datasets import load_dataset, DatasetDict, load_metric
from transformers import pipeline, ViTImageProcessor, AutoImageProcessor, ViTForImageClassification, TrainingArguments, ViTFeatureExtractor, Trainer
from evaluate import load
from sklearn.model_selection import train_test_split
from datasets import load_dataset, DatasetDict
from PIL import Image
from tqdm import tqdm
import os
import tensorflow as tf
import numpy as np
import torch
import torchvision.transforms as transformers

ds = load_dataset("uoft-cs/cifar10")
labels = ds['train'].features['label']
#print(labels)

model_name_or_path = 'google/vit-base-patch16-224-in21k'
processor = ViTImageProcessor.from_pretrained(model_name_or_path, use_fast=True)

def transform(example_batch):
    desired_size = (224, 224)
    resized_images = [transformers.Resize(desired_size)(x.convert("RGB")) for x in example_batch['image']]
    inputs = processor(resized_images, return_tensors='pt')
    inputs[labels] = example_batch['label']
    print(example_batch['label'])
    return inputs

prepared_ds = ds.with_transform(transform)

print(prepared_ds)

def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['label'] for x in batch])
    }

metric = load("accuracy")

def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

#processor = AutoImageProcessor.from_pretrained(model_name_or_path, use_fast=True)

model = ViTForImageClassification.from_pretrained(
    model_name_or_path,
    num_labels=labels.num_classes,
    id2label={str(i): c for i, c in enumerate(labels.names)},
    label2id={c: str(i) for i, c in enumerate(labels.names)},
    ignore_mismatched_sizes=True
)

root_dir = "./ViT_custom/"  # Path where all config files and checkpoints will be saved
training_args = TrainingArguments(
  output_dir=root_dir,
  per_device_train_batch_size=16,
  evaluation_strategy="epoch",
  save_strategy="epoch",
  fp16=True,
  num_train_epochs=20,
  logging_steps=500,
  learning_rate=2e-4,
  save_total_limit=1,
  remove_unused_columns=False,
  push_to_hub=False,
  report_to='tensorboard',
  load_best_model_at_end=True,
)


trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    train_dataset=prepared_ds["train"],
    eval_dataset=prepared_ds["test"],
    tokenizer=processor,
)



'''
save_dir = './ViT_custom/best_model/'  # Define the path to save the model
train_results = trainer.train()
trainer.save_model(save_dir)  # Save the best model
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()

metrics = trainer.evaluate(prepared_ds['test'])
trainer.log_metrics("test", metrics)
trainer.save_metrics("test", metrics)


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, recall_score

test_ds = ds['test'].with_transform(transform)
test_outputs = trainer.predict(test_ds)

y_true = test_outputs.label_ids
y_pred = test_outputs.predictions.argmax(1)

labels = test_ds.features["label"].names
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(xticks_rotation=45)

recall = recall_score(y_true, y_pred, average=None)

# Print the recall for each class
for label, score in zip(labels, recall):
  print(f"Recall for {label}: {score:.2f}")


def getPrediction(image):
    model_name_or_path = 'google/vit-base-patch16-224-in21k'
    processor = AutoImageProcessor.from_pretrained(model_name_or_path)
    vit = ViTForImageClassification.from_pretrained(save_dir)
    model = pipeline('image-classification', model=vit, feature_extractor=processor)

    result = model(image)
    return result

import cv2

image = cv2.imread("./Images/image.jpg")
print(getPrediction(image))

'''
