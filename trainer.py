import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
import matplotlib.pyplot as plt
import os
from custom_losses import classification_loss_function, regression_loss_function

class Trainer:
    def __init__(self, model, train_dataset, val_dataset, batch_size=8, learning_rate=1e-4, device=None):
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # Initialize DataLoader
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn)
        
        # Initialize optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        
        # Load loss functions
        self.classification_criterion = classification_loss_function
        self.regression_criterion = regression_loss_function

    def collate_fn(self, batch):
        # Unpack the batch into features and labels
        features, labels = zip(*batch)
        
        # Stack each type of feature separately
        fbank_features = [item[0] for item in features]
        wav2vec2_features = [item[1] for item in features]
        bert_features = [item[2] for item in features]
        
        fbank_features = torch.nn.utils.rnn.pad_sequence(fbank_features, batch_first=True)
        wav2vec2_features = torch.nn.utils.rnn.pad_sequence(wav2vec2_features, batch_first=True)
        bert_features = torch.nn.utils.rnn.pad_sequence(bert_features, batch_first=True)
        
        # Stack the labels
        classification_labels = torch.stack([label[0] for label in labels])
        regression_labels = torch.stack([label[1] for label in labels])
        
        return [fbank_features, wav2vec2_features, bert_features], (classification_labels, regression_labels)

    def train(self):
        self.model.train()
        total_loss = 0
        total_classification_loss = 0
        total_regression_loss = 0

        for batch_features, (classification_labels, regression_labels) in tqdm(self.train_loader, desc="Training"):
            # Move to device
            fbank_features, wav2vec2_features, bert_features = [f.to(self.device) for f in batch_features]
            classification_labels = classification_labels.to(self.device)
            regression_labels = regression_labels.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            logits, regression_output = self.model(fbank_features, wav2vec2_features, bert_features)

            # Compute losses
            classification_loss = self.classification_criterion(logits, classification_labels)
            regression_loss = self.regression_criterion(regression_output, regression_labels)

            # Total loss
            loss = classification_loss + regression_loss
            total_loss += loss.item()
            total_classification_loss += classification_loss.item()
            total_regression_loss += regression_loss.item()

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()

        avg_loss = total_loss / len(self.train_loader)
        avg_classification_loss = total_classification_loss / len(self.train_loader)
        avg_regression_loss = total_regression_loss / len(self.train_loader)

        print(f"Training loss: {avg_loss:.4f}")
        print(f"Classification loss: {avg_classification_loss:.4f}")
        print(f"Regression loss: {avg_regression_loss:.4f}")

        return avg_loss, avg_classification_loss, avg_regression_loss

    def validate(self):
        self.model.eval()
        total_loss = 0
        total_classification_loss = 0
        total_regression_loss = 0
        correct_classification = 0
        total_samples = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_features, (classification_labels, regression_labels) in tqdm(self.val_loader, desc="Evaluating"):
                # Move to device
                fbank_features, wav2vec2_features, bert_features = [f.to(self.device) for f in batch_features]
                classification_labels = classification_labels.to(self.device)
                regression_labels = regression_labels.to(self.device)

                # Forward pass
                logits, regression_output = self.model(fbank_features, wav2vec2_features, bert_features)

                # Compute losses
                classification_loss = self.classification_criterion(logits, classification_labels)
                regression_loss = self.regression_criterion(regression_output, regression_labels)

                # Total loss
                loss = classification_loss + regression_loss
                total_loss += loss.item()
                total_classification_loss += classification_loss.item()
                total_regression_loss += regression_loss.item()

                # Calculate classification accuracy
                _, predicted = torch.max(logits, 1)
                correct_classification += (predicted == classification_labels).sum().item()
                total_samples += classification_labels.size(0)

                # Collect all predictions and labels for metrics calculation
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(classification_labels.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader)
        avg_classification_loss = total_classification_loss / len(self.val_loader)
        avg_regression_loss = total_regression_loss / len(self.val_loader)
        accuracy = correct_classification / total_samples

        # Classification metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')

        # Regression metrics
        regression_outputs = torch.cat([regression_output for _, _, regression_output in self.val_loader], dim=0)
        rmse = torch.sqrt(torch.mean((regression_outputs - regression_labels) ** 2)).item()

        print(f"Validation loss: {avg_loss:.4f}")
        print(f"Classification loss: {avg_classification_loss:.4f}")
        print(f"Regression loss: {avg_regression_loss:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"RMSE: {rmse:.4f}")

        return avg_loss, avg_classification_loss, avg_regression_loss

    def fit(self, epochs):
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            # Train
            avg_train_loss, avg_train_classification_loss, avg_train_regression_loss = self.train()
            train_losses.append(avg_train_loss)

            # Validate
            avg_val_loss, avg_val_classification_loss, avg_val_regression_loss = self.validate()
            val_losses.append(avg_val_loss)
        
        # Plot loss curves
        self.plot_loss_curves(train_losses, val_losses)

    def plot_loss_curves(self, train_losses, val_losses):
        plt.figure(figsize=(12, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curves')
        plt.legend()
        plt.grid(True)
        plt.show()

    def save_checkpoint(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path):
        if os.path.isfile(path):
            self.model.load_state_dict(torch.load(path))
            print(f"Checkpoint loaded from {path}")
        else:
            print(f"No checkpoint found at {path}")

