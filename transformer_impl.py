from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
import torch


#(training_set -> text, label)


nb_samples_train = 10000
nb_samples_validation = 3000

train_set_ft_text = training_set['text'][:nb_samples_train]
train_set_ft_label = training_set['label'][:nb_samples_train]

valid_set_ft_text = training_set['text'][nb_samples_train:nb_samples_train+nb_samples_validation]
valid_set_ft_label = training_set['label'][nb_samples_train:nb_samples_train+nb_samples_validation]

def load_tok_model(flag):
  tokenizer = AutoTokenizer.from_pretrained(flag)
  model = AutoModelForSequenceClassification.from_pretrained(flag)
  return tokenizer, model

tokenizer, model = load_tok_model("distilbert-base-uncased")

train_encodings = tokenizer(train_set_ft_text, truncation=True, padding=True)
val_encodings = tokenizer(valid_set_ft_text, truncation=True, padding=True)

class YelpDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()} # tok generated input_ids, token_type_ids, attention_mask...
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Up to you to obtain train_encodings, training_set_label...
train_dataset = YelpDataset(train_encodings, train_set_ft_label)
val_dataset = YelpDataset(val_encodings, valid_set_ft_label)

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=1,              # total number of training epochs: 1 for the example
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=50,
)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset             # evaluation dataset
)

trainer.train()