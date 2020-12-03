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
from transformers import DistilBertModel, DistilBertConfig
from transformers.utils.dummy_pt_objects import AutoModelForQuestionAnswering


class VQAModel(torch.nn.Module):
    def __init__(self, tokenizer, device):
        super(VQAModel, self).__init__()
        self.tokenizer = tokenizer
        self.device = device
        self.image_embedding = torch.nn.Sequential(*(list(models.resnet152(pretrained=True).children())[:-1]))
        # self.question_embedding = torch.nn.Sequential(*(list(AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased").children())[:-3]))
        configuration = DistilBertConfig()
        self.question_embedding =  DistilBertModel(configuration)
        self.linear1 = torch.nn.Linear(2816, 1024)
        self.linear2 = torch.nn.Linear(1024,512)
        self.linear3 = torch.nn.Linear(512, 2)

    def forward(self, input):
        images, questions = input
        encoded_question = self.tokenizer(questions, add_special_tokens=True, padding=True, truncation=True, return_tensors="pt")
        encoded_question['input_ids'] = encoded_question['input_ids'].to(self.device)
        encoded_question['attention_mask'] = encoded_question['attention_mask'].to(self.device)

        qst_feature = self.question_embedding(input_ids=encoded_question['input_ids'], attention_mask= encoded_question['attention_mask']) # output : 768
        hidden_state = qst_feature[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)

        img_feature = self.image_embedding(images) 
        img_feature = torch.flatten(img_feature,start_dim=1, end_dim=-1) 
        x = torch.cat((img_feature, pooled_output), dim=1)        # output : 2816
        x = F.relu(self.linear1(x))                               # output : 1024
        x = F.relu(self.linear2(x))                               # output : 512
        x = self.linear3(x)                                       # output : 2
        return x