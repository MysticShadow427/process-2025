import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
import torch
from torch import nn
from datasets import load_metric

class MultiTaskModel(nn.Module):
    def __init__(self, model_name, num_labels_classification):
        super(MultiTaskModel, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        
        # Classification head
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels_classification)
        
        # Regression head
        self.regressor = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, labels_classification=None, labels_regression=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        pooled_output = self.dropout(pooled_output)
        
        # Classification logits
        logits_classification = self.classifier(pooled_output)
        
        # Regression logits
        logits_regression = self.regressor(pooled_output).squeeze(-1)  # Squeeze to match labels shape
        
        loss = None
        if labels_classification is not None and labels_regression is not None:
            # Classification loss
            loss_fct_classification = nn.CrossEntropyLoss()
            loss_classification = loss_fct_classification(logits_classification, labels_classification)
            
            # Regression loss
            loss_fct_regression = nn.MSELoss()
            loss_regression = loss_fct_regression(logits_regression, labels_regression)
            
            # Combined loss
            loss = loss_classification + loss_regression
        elif labels_classification is not None:
            # Only classification
            loss_fct_classification = nn.CrossEntropyLoss()
            loss = loss_fct_classification(logits_classification, labels_classification)
        elif labels_regression is not None:
            # Only regression
            loss_fct_regression = nn.MSELoss()
            loss = loss_fct_regression(logits_regression, labels_regression)

        return {'loss': loss, 'logits_classification': logits_classification, 'logits_regression': logits_regression}

# Load datasets
train_df = pd.read_csv('/content/drive/MyDrive/process2025/train_dataset_bert.csv')
val_df = pd.read_csv('/content/drive/MyDrive/process2025/val_dataset_bert.csv')

model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

train_dataset = train_dataset.map(preprocess_function, batched=True)
val_dataset = val_dataset.map(preprocess_function, batched=True)

# Rename columns
train_dataset = train_dataset.rename_column('classification_label', 'labels_classification')
train_dataset = train_dataset.rename_column('regression_label', 'labels_regression')
val_dataset = val_dataset.rename_column('classification_label', 'labels_classification')
val_dataset = val_dataset.rename_column('regression_label', 'labels_regression')

num_labels_classification = len(train_df['classification_label'].unique())  

model = MultiTaskModel(model_name, num_labels_classification)

def compute_metrics(pred):
    metric_classification = load_metric("accuracy")
    logits_classification = pred.predictions['logits_classification']
    labels_classification = pred.label_ids['labels_classification']
    predictions_classification = logits_classification.argmax(dim=-1)
    acc = metric_classification.compute(predictions=predictions_classification, references=labels_classification)
    
    # Regression metrics
    metric_regression = load_metric("mean_squared_error")
    logits_regression = pred.predictions['logits_regression']
    labels_regression = pred.label_ids['labels_regression']
    mse = metric_regression.compute(predictions=logits_regression, references=labels_regression)
    
    return {
        "accuracy": acc["accuracy"],
        "mse": mse["mse"]
    }

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    save_strategy="epoch",
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=True,
    save_total_limit=1
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

model.bert.save_pretrained("/content/drive/MyDrive/process2025/bert_model")
