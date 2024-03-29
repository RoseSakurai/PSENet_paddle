import paddle
import paddle.nn as nn
import math
import paddle.nn.functional as F
import numpy as np
import sys

from .backbone import build_backbone
from .neck import build_neck
from .head import build_head
from .utils import Conv_BN_ReLU


class PSENet(nn.Layer):
    def __init__(self,
                 backbone,
                 neck,
                 detection_head):
        super(PSENet, self).__init__()
        self.backbone = build_backbone(backbone)
        self.fpn = build_neck(neck)
        self.det_head = build_head(detection_head)

    def _upsample(self, x, size, scale=1):
        _, _, H, W = size
        return F.upsample(x, size=(H // scale, W // scale), mode='bilinear')

    def forward(self, imgs, gt_texts=None, gt_kernels=None, training_masks=None, img_metas=None, cfg=None):
        outputs = dict()

        f = self.backbone(imgs)
        f1, f2, f3, f4, = self.fpn(f[0], f[1], f[2], f[3])

        f = paddle.concat((f1, f2, f3, f4), 1)
        det_out = self.det_head(f)
        if self.training:
            det_out = self._upsample(det_out, imgs.shape)
            det_loss = self.det_head.loss(det_out, gt_texts.cast('int32'), gt_kernels.cast('int32'),
                                          training_masks.cast('int32'))
            outputs.update(det_loss)
        else:
            det_out = self._upsample(det_out, imgs.shape, 1)
            det_res = self.det_head.get_results(det_out, img_metas, cfg)
            outputs.update(det_res)
        return outputs
        
        # return det_out
