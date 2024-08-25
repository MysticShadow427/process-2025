import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()
    def forward(self, predictions, targets):
        mse = self.mse(predictions, targets)
        rmse = torch.sqrt(mse)
        return rmse

class ClassificationLoss(nn.Module):
    def __init__(self):
        super(ClassificationLoss, self).__init__()
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        return self.loss_fct(logits, labels)

class SimilarityLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(SimilarityLoss, self).__init__()
        self.margin = margin

    def forward(self, audio_embeddings, text_embeddings, labels):
        # Compute the distance between audio and text embeddings
        distances = F.pairwise_distance(audio_embeddings, text_embeddings)
        
        # Compute the contrastive loss
        loss = labels * torch.pow(distances, 2) + (1 - labels) * torch.pow(torch.clamp(self.margin - distances, min=0.0), 2)
        
        # Average the loss over the batch
        return torch.mean(loss)