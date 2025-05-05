from models.feature_encoder import FeatureEncoder
from models.cd_rpn_k import CDRPNK
from models.swav_module import SwAVModule
from models.detr_head import DETRHead

class CarDamageDetector(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.encoder = FeatureEncoder()
        self.rpn = CDRPNK()
        self.swav = SwAVModule()
        self.detr = DETRHead(num_classes=num_classes)

    def forward(self, x, gt_boxes=None, gt_labels=None):
        features = self.encoder(x)
        proposals = self.rpn(x)
        swav_loss = self.swav(features.flatten(2).permute(2, 0, 1))

        logits, boxes = self.detr(features)
        match_indices = None

        if gt_boxes is not None and gt_labels is not None:
            match_indices = self.detr.hungarian_match(logits, boxes, gt_boxes, gt_labels)

        return logits, boxes, swav_loss, match_indices
