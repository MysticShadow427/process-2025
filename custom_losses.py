import torch
import torch.nn as nn

class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()
    def forward(self, predictions, targets):
        mse = self.mse(predictions, targets)
        rmse = torch.sqrt(mse)
        return rmse

class classificationLoss(nn.Module):
    def __init__(self):
        super(classificationLoss, self).__init__()
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        return self.loss_fct(logits, labels)