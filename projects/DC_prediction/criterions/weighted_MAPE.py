import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedMAPE(nn.Module):
    def __init__(self, window_length, history_length, alpha_mse: float = 0.0, max_epoch: int = 50):
        super(WeightedMAPE, self).__init__()
        self.pred_length = window_length - history_length
        start_value = 1.0001
        end_value = 1.1999
        self.weights = torch.linspace(0, 1, self.pred_length) * (end_value - start_value) + start_value
        self.alpha_mse = alpha_mse
        self.max_epoch = max_epoch

    def forward(self, logits, annots, epoch=None):
        # logits, annots > (B, 1, 4, 2500)
        logits = logits[:, :, :, -self.pred_length:]
        annots = annots[:, :, :, -self.pred_length:]
        
        weights = self.weights.type_as(logits.data)

        abs_errors = torch.abs(logits - annots)
        MAPE_loss = torch.mean(torch.sum(
            (abs_errors / torch.abs(annots).clamp(min=1e-6)) * weights, dim=3) / torch.sum(weights))
        MSE_loss = torch.mean(torch.sum(
            F.mse_loss(logits, annots, reduction='none') * weights, dim=3) / torch.sum(weights))
        
        _alpha = (1 - min(1, (max(1, epoch) / self.max_epoch))) * self.alpha_mse if epoch else self.alpha_mse
        hybrid_errors = MSE_loss * _alpha + MAPE_loss * (1 - _alpha)
        
        loss_dict = {
            "MAPE": MAPE_loss,
            "MSE": MSE_loss,
            "hybrid": hybrid_errors}
        
        return loss_dict