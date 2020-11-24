import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import models
import torchvision.transforms as transforms

from PIL import Image

import pandas as pd
from typing import Any, Callable, Optional, Tuple
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification

from VQADataset import VQADataset
from VQAModel import VQAModel
from train_optim import train_optim

# Précisez la localisation de vos données sur Google Drive
path = "boolean_answers_dataset_10000"
image_folder = "boolean_answers_dataset_images_10000"
descriptor = "boolean_answers_dataset_10000.csv"

batch_size = 2

# exemples de transformations
transform = transforms.Compose(
    [transforms.Resize((720,720)),
     transforms.ToTensor(),     
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

vqa_dataset = VQADataset(path, descriptor, image_folder, transform=transform)
lenTrain = int(vqa_dataset.__len__() * 0.8)
train_set, test_set = torch.utils.data.random_split(vqa_dataset, [lenTrain, vqa_dataset.__len__() - lenTrain])

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

train_encodings = tokenizer(list(train_set.dataset.descriptor.question), truncation=True, padding=True)
val_encodings = tokenizer(list(test_set.dataset.descriptor.question), truncation=True, padding=True)
train_set.dataset.descriptor.question = train_encodings.values
train_loader = DataLoader(train_set,batch_size=batch_size, shuffle=True, num_workers=0)

test_loader = DataLoader(test_set,batch_size=batch_size, shuffle=False, num_workers=0)


def main():

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model = VQAModel()
  
    train_optim(model, train_loader, test_loader, epochs=3, log_frequency=60, device=device, learning_rate=1e-3)


if __name__ == "__main__":
  main()

