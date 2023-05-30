# Code adapted from detectron2
# https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py
# TODO: remove detectron2 dependencies and use openmmlab equivalents

from typing import Tuple

import torch
import torch.nn as nn

from mmengine.model import BaseModule
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, MultiConfig, OptConfigType
import mmdet.models.utils.detectron as detectron

@MODELS.register_module()
class SimpleFPN(BaseModule):

    def __init__(
        self,
        d_in,
        d_out
    ):
        """
        Args:
            d_in: input shape [B, D, H, W], D is specified by this arg
            d_out: output shape [B, D', H', W'], D' is specified by this arg
        """

        super().__init__()

        # scale = 4
        d = d_in // 4
        self.layer_scale_4 = nn.Sequential(
            nn.ConvTranspose2d(d_in, d_in // 2, kernel_size=2, stride=2),
            detectron.LayerNorm(d_in // 2),
            nn.GELU(),
            nn.ConvTranspose2d(d_in // 2, d_in // 4, kernel_size=2, stride=2),
            detectron.Conv2d(
                d,
                d_out,
                kernel_size=1,
                bias=False,
                norm=detectron.LayerNorm(d_out)
            ),
            detectron.Conv2d(
                d_out,
                d_out,
                kernel_size=3,
                padding=1,
                bias=False,
                norm=detectron.LayerNorm(d_out)
            )
        )

        # scale = 2
        d = d_in // 2
        self.layer_scale_2 = nn.Sequential(
            nn.ConvTranspose2d(d_in, d_in // 2, kernel_size=2, stride=2),
            detectron.Conv2d(
                d,
                d_out,
                kernel_size=1,
                bias=False,
                norm=detectron.LayerNorm(d_out)
            ),
            detectron.Conv2d(
                d_out,
                d_out,
                kernel_size=3,
                padding=1,
                bias=False,
                norm=detectron.LayerNorm(d_out)
            )
        )

        # scale = 1
        d = d_in
        self.layer_scale_1 = nn.Sequential(
            detectron.Conv2d(
                d,
                d_out,
                kernel_size=1,
                bias=False,
                norm=detectron.LayerNorm(d_out)
            ),
            detectron.Conv2d(
                d_out,
                d_out,
                kernel_size=3,
                padding=1,
                bias=False,
                norm=detectron.LayerNorm(d_out)
            )
        )

        # scale = 0.5
        d = d_in
        self.layer_scale_0_5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            detectron.Conv2d(
                d,
                d_out,
                kernel_size=1,
                bias=False,
                norm=detectron.LayerNorm(d_out)
            ),
            detectron.Conv2d(
                d_out,
                d_out,
                kernel_size=3,
                padding=1,
                bias=False,
                norm=detectron.LayerNorm(d_out)
            )
        )

        # extra
        self.layer_extra = nn.MaxPool2d(kernel_size=1, stride=2, padding=0)

        # save args for further use
        self.d_in = d_in
        self.d_out = d_out


    def forward(self, x:Tuple[torch.Tensor]) -> tuple:
        """
        Args:
            x: input tensor of shape [B, D, H, W], D should match d_in
                should be in tuple or list form
        Return:
            results: tuple containing feature on each pyramid layer
                ordered from large to small scale
        """
        assert x[-1].shape[1] == self.d_in, f"{x[-1].shape[1]} != {self.d_in}"

        feat_layer_scale_4 = self.layer_scale_4(x[-1])
        feat_layer_scale_2 = self.layer_scale_2(x[-1])
        feat_layer_scale_1 = self.layer_scale_1(x[-1])
        feat_layer_scale_0_5 = self.layer_scale_0_5(x[-1])
        feat_extra = self.layer_extra(feat_layer_scale_0_5)
        
        results = [
            feat_layer_scale_4,
            feat_layer_scale_2,
            feat_layer_scale_1,
            feat_layer_scale_0_5,
            feat_extra
        ]

        return tuple(results)
        