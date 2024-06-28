# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified by Sagar Vaze from https://github.com/ABaldrati/CLIP4CirDemo/blob/main/model.py

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

"""
Code from: https://github.com/ABaldrati/CLIP4CirDemo/blob/main/model.py
and https://raw.githubusercontent.com/facebookresearch/genecis/main/models/combiner_model.py
"""

class Combiner(nn.Module):
    """
    Combiner module which once trained fuses textual and label information
    """

    def __init__(self, clip_feature_dim: int, projection_dim: int, hidden_dim: int):
        """
        :param clip_feature_dim: CLIP input feature dimension
        :param projection_dim: projection dimension
        :param hidden_dim: hidden dimension
        """
        super(Combiner, self).__init__()
        self.text_projection_layer = nn.Linear(clip_feature_dim, projection_dim)
        self.image_projection_layer = nn.Linear(clip_feature_dim, projection_dim)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

        self.combiner_layer = nn.Linear(projection_dim * 2, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, clip_feature_dim)

        self.dropout3 = nn.Dropout(0.5)
        self.dynamic_scalar = nn.Sequential(nn.Linear(projection_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(0.5),
                                            nn.Linear(hidden_dim, 1),
                                            nn.Sigmoid())

        self.logit_scale = 100

    @torch.jit.export
    def forward(self, text_features, label_features):
        """
        Cobmine the text features and label features. It outputs the predicted features
        :param text_features: CLIP textual features
        :param label_features: Label features
        :return: predicted textual features
        """

        text_projected_features = self.dropout1(F.relu(self.text_projection_layer(text_features)))
        label_projected_features = self.dropout2(F.relu(self.image_projection_layer(label_features)))

        raw_combined_features = torch.cat((text_projected_features, label_projected_features), -1)
        combined_features = self.dropout3(F.relu(self.combiner_layer(raw_combined_features)))
        dynamic_scalar = self.dynamic_scalar(raw_combined_features)
        output = self.output_layer(combined_features) + dynamic_scalar * text_features + (
                1 - dynamic_scalar) * label_projected_features

        return F.normalize(output)
    
    
    
class DeepCombiner(nn.Module):
    """
    Combiner module which once trained fuses textual and label information
    """

    def __init__(self, clip_feature_dim: int, projection_dim: int, hidden_dim: int):
        """
        :param clip_feature_dim: CLIP input feature dimension
        :param projection_dim: projection dimension
        :param hidden_dim: hidden dimension
        """
        super(Combiner, self).__init__()
        self.text_projection_layer = nn.Linear(clip_feature_dim, projection_dim)
        self.image_projection_layer = nn.Linear(clip_feature_dim, projection_dim)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

        self.combiner_layer = nn.Linear(projection_dim * 2, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, clip_feature_dim)

        self.dropout3 = nn.Dropout(0.5)
        self.dynamic_scalar = nn.Sequential(nn.Linear(projection_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(0.5),
                                            nn.Linear(hidden_dim, 1),
                                            nn.Sigmoid())

        self.logit_scale = 100

    @torch.jit.export
    def forward(self, text_features, label_features):
        """
        Cobmine the text features and label features. It outputs the predicted features
        :param text_features: CLIP textual features
        :param label_features: Label features
        :return: predicted textual features
        """

        text_projected_features = self.dropout1(F.relu(self.text_projection_layer(text_features)))
        label_projected_features = self.dropout2(F.relu(self.image_projection_layer(label_features)))

        raw_combined_features = torch.cat((text_projected_features, label_projected_features), -1)
        combined_features = self.dropout3(F.relu(self.combiner_layer(raw_combined_features)))
        dynamic_scalar = self.dynamic_scalar(raw_combined_features)
        output = self.output_layer(combined_features) + dynamic_scalar * text_features + (
                1 - dynamic_scalar) * label_projected_features

        return F.normalize(output)