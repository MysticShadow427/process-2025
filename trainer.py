import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
import matplotlib.pyplot as plt
import os
from custom_losses import RMSELoss,ClassificationLoss,SimilarityLoss
from custom_dataloader import collate_fn

class Trainer:
    def __init__(self, model, train_dataset, val_dataset, batch_size=8, learning_rate=1e-4, wt_decay=1e-3,device='cuda'):
        self.model = model.to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total number of parameters in the model: {total_params}")
        self.device = device
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)
        
        self.optimizer = AdamW(self.model.parameters(), lr=float(self.learning_rate),weight_decay=float(wt_decay))
        
        self.classification_criterion = ClassificationLoss()
        self.regression_criterion = RMSELoss()
        self.similarity_criterion = SimilarityLoss()

    def train(self):
        self.model.train()
        total_loss = 0
        total_classification_loss = 0
        total_regression_loss = 0
        total_similarity_loss = 0

        for batch_features, (classification_labels, regression_labels) in tqdm(self.train_loader, desc="Training"):
            
            fbank_features = batch_features['fbank_features'].to(self.device)
            wav2vec2_features = batch_features['wav2vec2_features'].to(self.device)
            egmap_features = batch_features['egmap_features'].to(self.device)
            trill_features = batch_features['trill_features'].to(self.device)
            phonetic_features = batch_features['phonetic_features'].to(self.device)
            bert_features = batch_features['bert_features'].to(self.device)

            classification_labels = classification_labels.to(self.device)
            regression_labels = regression_labels.to(self.device)

            self.optimizer.zero_grad()

            logits, regression_output, speech_features = self.model(fbank_features,wav2vec2_features,egmap_features,trill_features,phonetic_features)

            classification_loss = self.classification_criterion(logits, classification_labels)
            regression_loss = self.regression_criterion(regression_output.squeeze(dim=1), regression_labels)
            similarity_loss = self.similarity_criterion(speech_features,bert_features.mean(dim=1),classification_labels)

            loss = classification_loss + regression_loss + 0.8 * similarity_loss
            total_loss += loss.item()
            total_classification_loss += classification_loss.item()
            total_regression_loss += regression_loss.item()
            total_similarity_loss += similarity_loss.item()

            loss.backward()
            self.optimizer.step()

        avg_loss = total_loss / len(self.train_loader)
        avg_classification_loss = total_classification_loss / len(self.train_loader)
        avg_regression_loss = total_regression_loss / len(self.train_loader)
        avg_similarity_loss = total_similarity_loss / len(self.train_loader)

        print(f"Training loss: {avg_loss:.4f}")
        print(f"Classification loss: {avg_classification_loss:.4f}")
        print(f"Regression loss: {avg_regression_loss:.4f}")
        print(f"Similarity loss: {avg_similarity_loss:.4f}")

        return avg_loss, avg_classification_loss, avg_regression_loss, avg_similarity_loss

    def validate(self):
        self.model.eval()
        total_loss = 0
        total_classification_loss = 0
        total_regression_loss = 0
        total_similarity_loss = 0
        correct_classification = 0
        total_samples = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_features, (classification_labels, regression_labels) in tqdm(self.val_loader, desc="Evaluating"):
                
                fbank_features = batch_features['fbank_features'].to(self.device)
                wav2vec2_features = batch_features['wav2vec2_features'].to(self.device)
                egmap_features = batch_features['egmap_features'].to(self.device)
                trill_features = batch_features['trill_features'].to(self.device)
                phonetic_features = batch_features['phonetic_features'].to(self.device)
                bert_features = batch_features['bert_features'].to(self.device)

                classification_labels = classification_labels.to(self.device)
                regression_labels = regression_labels.to(self.device)

                logits, regression_output, speech_features = self.model(fbank_features,wav2vec2_features,egmap_features,trill_features,phonetic_features)

                classification_loss = self.classification_criterion(logits, classification_labels)
                regression_loss = self.regression_criterion(regression_output.squeeze(dim=1), regression_labels)
                similarity_loss = self.similarity_criterion(speech_features,bert_features.mean(dim=1),classification_labels)

                loss = classification_loss + regression_loss + similarity_loss
                total_loss += loss.item()
                total_classification_loss += classification_loss.item()
                total_regression_loss += regression_loss.item()
                total_similarity_loss += similarity_loss.item()

                _, predicted = torch.max(logits, 1)
                correct_classification += (predicted == classification_labels).sum().item()
                total_samples += classification_labels.size(0)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(classification_labels.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader)
        avg_classification_loss = total_classification_loss / len(self.val_loader)
        avg_regression_loss = total_regression_loss / len(self.val_loader)
        avg_similarity_loss = total_similarity_loss / len(self.val_loader)
        accuracy = correct_classification / total_samples

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='weighted')

        regression_outputs = torch.cat([regression_output for _, _, regression_output in self.val_loader], dim=0)
        rmse = torch.sqrt(torch.mean((regression_outputs - regression_labels) ** 2)).item()

        print(f"Validation loss: {avg_loss:.4f}")
        print(f"Classification loss: {avg_classification_loss:.4f}")
        print(f"Regression loss: {avg_regression_loss:.4f}")
        print(f"Similarity loss: {avg_similarity_loss:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")
        print(f"Macro Precision: {precision:.4f}")
        print(f"Macro Recall: {recall:.4f}")
        print(f"Weighted F1 Score: {f1:.4f}")
        print(f"RMSE: {rmse:.4f}")

        return avg_loss, avg_classification_loss, avg_regression_loss, avg_similarity_loss

    def fit(self, epochs):
        train_losses = []
        val_losses = []
        train_cls_loss = []
        val_cls_loss = []
        train_reg_loss = []
        val_reg_loss = []
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
    
            avg_train_loss, avg_train_classification_loss, avg_train_regression_loss,_ = self.train()
            train_losses.append(avg_train_loss)
            train_cls_loss.append(avg_train_classification_loss)
            train_reg_loss.append(avg_train_regression_loss)


            avg_val_loss, avg_val_classification_loss, avg_val_regression_loss,_ = self.validate()
            val_losses.append(avg_val_loss)
            val_cls_loss.append(avg_val_classification_loss)
            val_reg_loss.append(avg_val_regression_loss)
        
        self.plot_loss_curves(train_losses, val_losses,train_cls_loss,val_cls_loss,train_reg_loss,val_reg_loss)

    def plot_loss_curves(self, train_losses, val_losses,train_cls_loss,val_cls_loss,train_reg_loss,val_reg_loss):
        plt.figure(figsize=(12, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curves')
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.savefig('/content/cls_loss_curve.png')
        plt.figure(figsize=(12, 6))
        plt.plot(train_cls_loss, label='Training Loss')
        plt.plot(val_cls_loss, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curves')
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.savefig('/content/loss_curve.png')
        plt.figure(figsize=(12, 6))
        plt.plot(train_reg_loss, label='Training Loss')
        plt.plot(val_reg_loss, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curves')
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.savefig('/content/reg_loss_curve.png')

    def save_checkpoint(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path):
        if os.path.isfile(path):
            self.model.load_state_dict(torch.load(path))
            print(f"Checkpoint loaded from {path}")
        else:
            print(f"No checkpoint found at {path}")

