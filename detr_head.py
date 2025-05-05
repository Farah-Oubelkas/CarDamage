import torch
import torch.nn as nn
from torch.nn.functional import l1_loss, cross_entropy
from transformers import DetrModel, DetrConfig
from scipy.optimize import linear_sum_assignment

class DETRHead(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        config = DetrConfig(num_labels=num_classes)
        self.detr = DetrModel(config)
        self.class_embed = nn.Linear(config.hidden_size, num_classes)
        self.bbox_embed = nn.Linear(config.hidden_size, 4)

    def forward(self, feature_map):
        outputs = self.detr(pixel_values=feature_map)
        hs = outputs.last_hidden_state  # shape: [B, N, D]

        logits = self.class_embed(hs)
        boxes = self.bbox_embed(hs).sigmoid()  # normalized boxes

        return logits, boxes

    def compute_cost_matrix(self, pred_logits, pred_boxes, gt_boxes, gt_labels):
        """
        Computes cost matrix for Hungarian matching.
        """
        B, N_pred, _ = pred_logits.shape
        cost_matrices = []

        for b in range(B):
            cls_cost = -pred_logits[b][:, gt_labels[b]].detach().cpu().numpy()
            bbox_cost = torch.cdist(pred_boxes[b], gt_boxes[b], p=1).detach().cpu().numpy()
            total_cost = cls_cost + bbox_cost
            cost_matrices.append(total_cost)

        return cost_matrices

    def hungarian_match(self, pred_logits, pred_boxes, gt_boxes, gt_labels):
        matched_indices = []
        cost_matrices = self.compute_cost_matrix(pred_logits, pred_boxes, gt_boxes, gt_labels)

        for cost_matrix in cost_matrices:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            matched_indices.append((row_ind, col_ind))

        return matched_indices
