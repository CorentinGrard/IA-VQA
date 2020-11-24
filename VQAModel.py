import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import models
import torchvision.transforms as transforms
from transformers import AutoModelForSequenceClassification

from PIL import Image

import pandas as pd
from pprint import pprint
from typing import Any, Callable, Optional, Tuple
from transformers.tokenization_auto import AutoTokenizer
from transformers.utils.dummy_pt_objects import AutoModelForQuestionAnswering


class VQAModel(torch.nn.Module):
    def __init__(self):
        super(VQAModel, self).__init__()
        self.image_embedding = torch.nn.Sequential(*(list(models.resnet152(pretrained=True).children())[:-1]))
        self.question_embedding = torch.nn.Sequential(*(list(AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased").children())[:-3]))
        self.linear1 = torch.nn.Linear(2816, 1024)
        self.linear2 = torch.nn.Linear(1024,512)
        self.linear3 = torch.nn.Linear(512, 2)

    def forward(self, images, questions):
      img_feature = self.image_embedding(images)        # output: 2048
      qst_feature = self.question_embedding(questions)  # output: 768
      x = torch.mul(img_feature, qst_feature)           # output : 2816
      x = F.relu(self.linear1(x))
      x = F.relu(self.linear2(x))
      x = self.linear3(x)
      return x