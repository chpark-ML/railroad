import torch
import torch.nn as nn


class WeightedMAPE(nn.Module):
    def __init__(self, window_length, history_length):
        super(WeightedMAPE, self).__init__()
        self.pred_length = window_length - history_length
        start_value = 1.0001
        end_value = 1.1999
        self.weights = torch.linspace(0, 1, self.pred_length) * (end_value - start_value) + start_value

    def forward(self, logits, annots):
        # logits, annots > (B, 1, 4, 2500)
        logits = logits[:, :, :, -self.pred_length:]
        annots = annots[:, :, :, -self.pred_length:]
        
        weights = self.weights.type_as(logits.data)
        abs_errors = torch.abs(logits - annots)
        percentage_errors = abs_errors / torch.abs(annots).clamp(min=1e-4)
        weighted_erros = weights * percentage_errors
        
        loss = torch.mean(weighted_erros)
        return loss