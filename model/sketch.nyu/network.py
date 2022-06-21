# encoding: utf-8

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from functools import partial
from collections import OrderedDict
from config import config
from resnet import get_resnet50
from torch_poly_lr_decay import PolynomialLRDecay
from score_utils import compute_metric, score_to_pred, print_ssc_iou
import sys

from sc_utils import export_grid
import pdb

sys.path.append("/home/yildirir/workspace/kerem/TorchSSC/furnace/")
from utils.init_func import init_weight, group_weight
from loss_utils import compute_loss


class SimpleRB(nn.Module):
    def __init__(self, in_channel, norm_layer, bn_momentum):
        super(SimpleRB, self).__init__()
        self.path = nn.Sequential(
            nn.Conv3d(in_channel, in_channel, kernel_size=3, padding=1, bias=False),
            norm_layer(in_channel, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv3d(in_channel, in_channel, kernel_size=3, padding=1, bias=False),
            norm_layer(in_channel, momentum=bn_momentum),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        conv_path = self.path(x)
        out = residual + conv_path
        out = self.relu(out)
        return out


"""
3D Residual Block，3x3x3 conv ==> 3 smaller 3D conv, refered from DDRNet
"""


class Bottleneck3D(nn.Module):
    def __init__(
        self,
        inplanes,
        planes,
        norm_layer,
        stride=1,
        dilation=[1, 1, 1],
        expansion=4,
        downsample=None,
        fist_dilation=1,
        multi_grid=1,
        bn_momentum=0.0003,
    ):
        super(Bottleneck3D, self).__init__()
        # often，planes = inplanes // 4
        self.expansion = expansion
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes, momentum=bn_momentum)
        self.conv2 = nn.Conv3d(
            planes,
            planes,
            kernel_size=(1, 1, 3),
            stride=(1, 1, stride),
            dilation=(1, 1, dilation[0]),
            padding=(0, 0, dilation[0]),
            bias=False,
        )
        self.bn2 = norm_layer(planes, momentum=bn_momentum)
        self.conv3 = nn.Conv3d(
            planes,
            planes,
            kernel_size=(1, 3, 1),
            stride=(1, stride, 1),
            dilation=(1, dilation[1], 1),
            padding=(0, dilation[1], 0),
            bias=False,
        )
        self.bn3 = norm_layer(planes, momentum=bn_momentum)
        self.conv4 = nn.Conv3d(
            planes,
            planes,
            kernel_size=(3, 1, 1),
            stride=(stride, 1, 1),
            dilation=(dilation[2], 1, 1),
            padding=(dilation[2], 0, 0),
            bias=False,
        )
        self.bn4 = norm_layer(planes, momentum=bn_momentum)
        self.conv5 = nn.Conv3d(
            planes, planes * self.expansion, kernel_size=(1, 1, 1), bias=False
        )
        self.bn5 = norm_layer(planes * self.expansion, momentum=bn_momentum)

        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

        self.downsample2 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, stride, 1), stride=(1, stride, 1)),
            nn.Conv3d(planes, planes, kernel_size=1, stride=1, bias=False),
            norm_layer(planes, momentum=bn_momentum),
        )
        self.downsample3 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(stride, 1, 1), stride=(stride, 1, 1)),
            nn.Conv3d(planes, planes, kernel_size=1, stride=1, bias=False),
            norm_layer(planes, momentum=bn_momentum),
        )
        self.downsample4 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(stride, 1, 1), stride=(stride, 1, 1)),
            nn.Conv3d(planes, planes, kernel_size=1, stride=1, bias=False),
            norm_layer(planes, momentum=bn_momentum),
        )

    def forward(self, x):
        residual = x

        out1 = self.relu(self.bn1(self.conv1(x)))
        out2 = self.bn2(self.conv2(out1))
        out2_relu = self.relu(out2)

        out3 = self.bn3(self.conv3(out2_relu))
        if self.stride != 1:
            out2 = self.downsample2(out2)
        out3 = out3 + out2
        out3_relu = self.relu(out3)

        out4 = self.bn4(self.conv4(out3_relu))
        if self.stride != 1:
            out2 = self.downsample3(out2)
            out3 = self.downsample4(out3)
        out4 = out4 + out2 + out3

        out4_relu = self.relu(out4)
        out5 = self.bn5(self.conv5(out4_relu))

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out5 + residual
        out_relu = self.relu(out)

        return out_relu


"""
Input: 60*36*60 sketch
Latent code: 15*9*15
"""


class CVAE(nn.Module):
    def __init__(self, norm_layer, bn_momentum, latent_size=16):
        super(CVAE, self).__init__()
        self.latent_size = latent_size
        self.encoder = nn.Sequential(
            nn.Conv3d(2, 3, kernel_size=3, padding=1, bias=False),
            norm_layer(3, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv3d(3, 16, kernel_size=3, padding=1, bias=False),
            norm_layer(16, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv3d(16, 16, kernel_size=3, padding=1, bias=False),
            norm_layer(16, momentum=bn_momentum),
            nn.ReLU(),
            nn.AvgPool3d(kernel_size=2, stride=2),
            nn.Conv3d(16, self.latent_size, kernel_size=3, padding=1, bias=False),
            norm_layer(self.latent_size, momentum=bn_momentum),
            nn.ReLU(),
            nn.AvgPool3d(kernel_size=2, stride=2),
            nn.Conv3d(
                self.latent_size, self.latent_size, kernel_size=3, padding=1, bias=False
            ),
            norm_layer(self.latent_size, momentum=bn_momentum),
            nn.ReLU(),
        )

        self.mean = nn.Conv3d(
            self.latent_size, self.latent_size, kernel_size=1, bias=True
        )  # predict mean.
        self.log_var = nn.Conv3d(
            self.latent_size, self.latent_size, kernel_size=1, bias=True
        )  # predict log(var).

        self.decoder_x = nn.Sequential(
            nn.Conv3d(1, 3, kernel_size=3, padding=1, bias=False),
            norm_layer(3, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv3d(3, 16, kernel_size=3, padding=1, bias=False),
            norm_layer(16, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv3d(16, 16, kernel_size=3, padding=1, bias=False),
            norm_layer(16, momentum=bn_momentum),
            nn.ReLU(),
            nn.AvgPool3d(kernel_size=2, stride=2),
            nn.Conv3d(16, self.latent_size, kernel_size=3, padding=1, bias=False),
            norm_layer(self.latent_size, momentum=bn_momentum),
            nn.ReLU(),
            nn.AvgPool3d(kernel_size=2, stride=2),
            nn.Conv3d(
                self.latent_size, self.latent_size, kernel_size=3, padding=1, bias=False
            ),
            norm_layer(self.latent_size, momentum=bn_momentum),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(
                self.latent_size * 2,
                self.latent_size,
                kernel_size=3,
                stride=2,
                padding=1,
                dilation=1,
                output_padding=1,
            ),
            norm_layer(self.latent_size, momentum=bn_momentum),
            nn.ReLU(inplace=False),
            nn.ConvTranspose3d(
                self.latent_size,
                self.latent_size,
                kernel_size=3,
                stride=2,
                padding=1,
                dilation=1,
                output_padding=1,
            ),
            norm_layer(self.latent_size, momentum=bn_momentum),
            nn.ReLU(inplace=False),
            nn.Dropout3d(0.1),
            nn.Conv3d(self.latent_size, 2, kernel_size=1, bias=True),
        )

    def forward(self, x, gt=None):
        b, c, h, w, l = x.shape

        if self.training:
            gt = gt.view(b, 1, h, w, l).float()
            for_encoder = torch.cat([x, gt], dim=1)
            enc = self.encoder(for_encoder)
            pred_mean = self.mean(enc)
            pred_log_var = self.log_var(enc)

            decoder_x = self.decoder_x(x)

            out_samples = []
            out_samples_gsnn = []
            for i in range(config.samples):
                std = pred_log_var.mul(0.5).exp_()
                eps = torch.randn(
                    [b, self.latent_size, h // 4, w // 4, l // 4]
                ).type_as(pred_log_var)
                z1 = eps * std + pred_mean
                z2 = torch.randn([b, self.latent_size, h // 4, w // 4, l // 4]).type_as(
                    pred_log_var
                )

                sketch = self.decoder(torch.cat([decoder_x, z1], dim=1))
                out_samples.append(sketch)

                sketch_gsnn = self.decoder(torch.cat([decoder_x, z2], dim=1))
                out_samples_gsnn.append(sketch_gsnn)

            sketch = torch.cat(
                [torch.unsqueeze(out_sample, dim=0) for out_sample in out_samples]
            )
            sketch = torch.mean(sketch, dim=0)
            sketch_gsnn = torch.cat(
                [torch.unsqueeze(out_sample, dim=0) for out_sample in out_samples_gsnn]
            )
            sketch_gsnn = torch.mean(sketch_gsnn, dim=0)

            return pred_mean, pred_log_var, sketch_gsnn, sketch
        else:
            out_samples = []
            for i in range(config.samples):
                z = torch.randn([b, self.latent_size, h // 4, w // 4, l // 4]).type_as(
                    x
                )
                decoder_x = self.decoder_x(x)
                out = self.decoder(torch.cat([decoder_x, z], dim=1))
                out_samples.append(out)
            sketch_gsnn = torch.cat(
                [torch.unsqueeze(out_sample, dim=0) for out_sample in out_samples]
            )
            sketch_gsnn = torch.mean(sketch_gsnn, dim=0)
            return None, None, sketch_gsnn, None


class STAGE1(nn.Module):
    def __init__(
        self,
        class_num,
        norm_layer,
        resnet_out=2048,
        feature=512,
        ThreeDinit=True,
        bn_momentum=0.1,
        pretrained_model=None,
        eval=False,
        freeze_bn=False,
    ):
        super(STAGE1, self).__init__()
        self.business_layer = []

        self.oper1 = nn.Sequential(
            nn.Conv3d(1, 3, kernel_size=3, padding=1, bias=False),
            norm_layer(3, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv3d(3, 64, kernel_size=3, padding=1, bias=False),
            norm_layer(64, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv3d(64, feature, kernel_size=3, padding=1, bias=False),
            norm_layer(feature, momentum=bn_momentum),
            nn.ReLU(inplace=False),
        )
        self.business_layer.append(self.oper1)

        self.completion_layer1 = nn.Sequential(
            Bottleneck3D(
                feature,
                feature // 4,
                bn_momentum=bn_momentum,
                expansion=4,
                stride=2,
                downsample=nn.Sequential(
                    nn.AvgPool3d(kernel_size=2, stride=2),
                    nn.Conv3d(feature, feature, kernel_size=1, stride=1, bias=False),
                    norm_layer(feature, momentum=bn_momentum),
                    # nn.ReLU(),
                ),
                norm_layer=norm_layer,
            ),  # feature --> feature*2
            Bottleneck3D(
                feature,
                feature // 4,
                bn_momentum=bn_momentum,
                norm_layer=norm_layer,
                dilation=[1, 1, 1],
            ),
            Bottleneck3D(
                feature,
                feature // 4,
                bn_momentum=bn_momentum,
                norm_layer=norm_layer,
                dilation=[2, 2, 2],
            ),
            Bottleneck3D(
                feature,
                feature // 4,
                bn_momentum=bn_momentum,
                norm_layer=norm_layer,
                dilation=[3, 3, 3],
            ),
        )
        self.business_layer.append(self.completion_layer1)

        self.completion_layer2 = nn.Sequential(
            Bottleneck3D(
                feature,
                feature // 4,
                bn_momentum=bn_momentum,
                expansion=8,
                stride=2,
                downsample=nn.Sequential(
                    nn.AvgPool3d(kernel_size=2, stride=2),
                    nn.Conv3d(
                        feature, feature * 2, kernel_size=1, stride=1, bias=False
                    ),
                    norm_layer(feature * 2, momentum=bn_momentum),
                    # nn.ReLU(),
                ),
                norm_layer=norm_layer,
            ),
            Bottleneck3D(
                feature * 2,
                feature // 2,
                bn_momentum=bn_momentum,
                norm_layer=norm_layer,
                dilation=[1, 1, 1],
            ),
            Bottleneck3D(
                feature * 2,
                feature // 2,
                bn_momentum=bn_momentum,
                norm_layer=norm_layer,
                dilation=[2, 2, 2],
            ),
            Bottleneck3D(
                feature * 2,
                feature // 2,
                bn_momentum=bn_momentum,
                norm_layer=norm_layer,
                dilation=[3, 3, 3],
            ),
        )
        self.business_layer.append(self.completion_layer2)

        self.cvae = CVAE(
            norm_layer=norm_layer,
            bn_momentum=bn_momentum,
            latent_size=config.lantent_size,
        )
        self.business_layer.append(self.cvae)

        self.classify_sketch = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConvTranspose3d(
                        feature * 2,
                        feature,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        dilation=1,
                        output_padding=1,
                    ),
                    norm_layer(feature, momentum=bn_momentum),
                    nn.ReLU(inplace=False),
                ),
                nn.Sequential(
                    nn.ConvTranspose3d(
                        feature,
                        feature,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        dilation=1,
                        output_padding=1,
                    ),
                    norm_layer(feature, momentum=bn_momentum),
                    nn.ReLU(inplace=False),
                ),
                nn.Sequential(
                    nn.Dropout3d(0.1), nn.Conv3d(feature, 2, kernel_size=1, bias=True)
                ),
            ]
        )
        self.business_layer.append(self.classify_sketch)

    def forward(self, tsdf, depth_mapping_3d, sketch_gt=None):
        """
        extract 3D feature
        """
        tsdf = self.oper1(tsdf)
        completion1 = self.completion_layer1(tsdf)
        completion2 = self.completion_layer2(completion1)

        up_sketch1 = self.classify_sketch[0](completion2)
        up_sketch1 = up_sketch1 + completion1
        up_sketch2 = self.classify_sketch[1](up_sketch1)
        pred_sketch_raw = self.classify_sketch[2](up_sketch2)

        _, pred_sketch_binary = torch.max(
            pred_sketch_raw, dim=1, keepdim=True
        )  # (b, 1, 60, 36, 60) binary-voxel sketch
        pred_mean, pred_log_var, pred_sketch_gsnn, pred_sketch = self.cvae(
            pred_sketch_binary.float(), sketch_gt
        )

        return pred_sketch_raw, pred_sketch_gsnn, pred_sketch, pred_mean, pred_log_var


class STAGE2(nn.Module):
    def __init__(
        self,
        class_num,
        norm_layer,
        resnet_out=2048,
        feature=512,
        ThreeDinit=True,
        bn_momentum=0.1,
        pretrained_model=None,
        eval=False,
        freeze_bn=False,
    ):
        super(STAGE2, self).__init__()
        self.business_layer = []

        if eval:
            self.downsample = nn.Sequential(
                nn.Conv2d(resnet_out, feature, kernel_size=1, bias=False),
                nn.BatchNorm2d(feature, momentum=bn_momentum),
                nn.ReLU(),
            )
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(resnet_out, feature, kernel_size=1, bias=False),
                nn.BatchNorm2d(feature, momentum=bn_momentum),
                nn.ReLU(),
            )
        self.business_layer.append(self.downsample)

        self.resnet_out = resnet_out
        self.feature = feature
        self.ThreeDinit = ThreeDinit

        self.pooling = nn.AvgPool3d(kernel_size=3, padding=1, stride=1)
        self.business_layer.append(self.pooling)

        self.semantic_layer1 = nn.Sequential(
            Bottleneck3D(
                feature,
                feature // 4,
                bn_momentum=bn_momentum,
                expansion=4,
                stride=2,
                downsample=nn.Sequential(
                    nn.AvgPool3d(kernel_size=2, stride=2),
                    nn.Conv3d(feature, feature, kernel_size=1, stride=1, bias=False),
                    norm_layer(feature, momentum=bn_momentum),
                ),
                norm_layer=norm_layer,
            ),  # feature --> feature*2
            Bottleneck3D(
                feature,
                feature // 4,
                bn_momentum=bn_momentum,
                norm_layer=norm_layer,
                dilation=[1, 1, 1],
            ),
            Bottleneck3D(
                feature,
                feature // 4,
                bn_momentum=bn_momentum,
                norm_layer=norm_layer,
                dilation=[2, 2, 2],
            ),
            Bottleneck3D(
                feature,
                feature // 4,
                bn_momentum=bn_momentum,
                norm_layer=norm_layer,
                dilation=[3, 3, 3],
            ),
        )
        self.business_layer.append(self.semantic_layer1)

        self.semantic_layer2 = nn.Sequential(
            Bottleneck3D(
                feature,
                feature // 4,
                bn_momentum=bn_momentum,
                expansion=8,
                stride=2,
                downsample=nn.Sequential(
                    nn.AvgPool3d(kernel_size=2, stride=2),
                    nn.Conv3d(
                        feature, feature * 2, kernel_size=1, stride=1, bias=False
                    ),
                    norm_layer(feature * 2, momentum=bn_momentum),
                ),
                norm_layer=norm_layer,
            ),
            Bottleneck3D(
                feature * 2,
                feature // 2,
                bn_momentum=bn_momentum,
                norm_layer=norm_layer,
                dilation=[1, 1, 1],
            ),
            Bottleneck3D(
                feature * 2,
                feature // 2,
                bn_momentum=bn_momentum,
                norm_layer=norm_layer,
                dilation=[2, 2, 2],
            ),
            Bottleneck3D(
                feature * 2,
                feature // 2,
                bn_momentum=bn_momentum,
                norm_layer=norm_layer,
                dilation=[3, 3, 3],
            ),
        )
        self.business_layer.append(self.semantic_layer2)

        self.classify_semantic = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConvTranspose3d(
                        feature * 2,
                        feature,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        dilation=1,
                        output_padding=1,
                    ),
                    norm_layer(feature, momentum=bn_momentum),
                    nn.ReLU(inplace=False),
                ),
                nn.Sequential(
                    nn.ConvTranspose3d(
                        feature,
                        feature,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        dilation=1,
                        output_padding=1,
                    ),
                    norm_layer(feature, momentum=bn_momentum),
                    nn.ReLU(inplace=False),
                ),
                nn.Sequential(
                    nn.Dropout3d(0.1),
                    nn.Conv3d(feature, class_num, kernel_size=1, bias=True),
                ),
            ]
        )
        self.business_layer.append(self.classify_semantic)

        self.oper_sketch = nn.Sequential(
            nn.Conv3d(2, 3, kernel_size=3, padding=1, bias=False),
            norm_layer(3, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv3d(3, 64, kernel_size=3, padding=1, bias=False),
            norm_layer(64, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv3d(64, feature, kernel_size=3, padding=1, bias=False),
            norm_layer(feature, momentum=bn_momentum),
            nn.ReLU(inplace=False),
        )
        self.oper_sketch_cvae = nn.Sequential(
            nn.Conv3d(2, 3, kernel_size=3, padding=1, bias=False),
            norm_layer(3, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv3d(3, 64, kernel_size=3, padding=1, bias=False),
            norm_layer(64, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv3d(64, feature, kernel_size=3, padding=1, bias=False),
            norm_layer(feature, momentum=bn_momentum),
            nn.ReLU(inplace=False),
        )
        self.business_layer.append(self.oper_sketch)
        self.business_layer.append(self.oper_sketch_cvae)

    def forward(self, feature2d, depth_mapping_3d, pred_sketch_raw, pred_sketch_gsnn):
        # reduce the channel of 2D feature map
        if self.resnet_out != self.feature:
            feature2d = self.downsample(feature2d)
        feature2d = F.interpolate(
            feature2d, scale_factor=16, mode="bilinear", align_corners=True
        )

        """
        project 2D feature to 3D space
        """
        b, c, h, w = feature2d.shape
        feature2d = feature2d.view(b, c, h * w).permute(0, 2, 1)  # b x h*w x c

        zerosVec = torch.zeros(b, 1, c).type_as(
            feature2d
        )  # for voxels that could not be projected from the depth map, we assign them zero vector
        segVec = torch.cat((feature2d, zerosVec), 1)

        segres = [
            torch.index_select(segVec[i], 0, depth_mapping_3d[i]) for i in range(b)
        ]
        segres = (
            torch.stack(segres).permute(0, 2, 1).contiguous().view(b, c, 60, 36, 60)
        )  # B, (channel), 60, 36, 60

        """
        init the 3D feature
        """
        if self.ThreeDinit:
            pool = self.pooling(segres)

            zero = (segres == 0).float()
            pool = pool * zero
            segres = segres + pool

        """
        extract 3D feature
        """
        sketch_proi = self.oper_sketch(pred_sketch_raw)
        sketch_proi_gsnn = self.oper_sketch_cvae(pred_sketch_gsnn)

        seg_fea = segres + sketch_proi + sketch_proi_gsnn
        semantic1 = self.semantic_layer1(seg_fea)
        semantic2 = self.semantic_layer2(semantic1)
        up_sem1 = self.classify_semantic[0](semantic2)
        up_sem1 = up_sem1 + semantic1
        up_sem2 = self.classify_semantic[1](up_sem1)
        pred_semantic = self.classify_semantic[2](up_sem2)

        return pred_semantic, None


"""
main network
"""


class Network(pl.LightningModule):
    def __init__(
        self,
        class_num,
        norm_layer,
        resnet_out=2048,
        feature=512,
        ThreeDinit=True,
        bn_momentum=0.1,
        pretrained_model=None,
        eval=False,
        freeze_bn=False,
        config=None,
    ):
        super(Network, self).__init__()

        self.automatic_optimization = False
        self.save_hyperparameters(config)
        self.business_layer = []
        self.debug = False
        self.config = config
        if eval:
            self.backbone = get_resnet50(
                num_classes=class_num,
                dilation=[1, 1, 1, 2],
                bn_momentum=self.config.bn_momentum,
                is_fpn=False,
                BatchNorm2d=nn.BatchNorm2d,
            )
        else:
            self.backbone = get_resnet50(
                num_classes=class_num,
                dilation=[1, 1, 1, 2],
                bn_momentum=config.bn_momentum,
                is_fpn=False,
                BatchNorm2d=nn.BatchNorm2d,
            )

        self.dilate = 2
        for m in self.backbone.layer4.children():
            m.apply(partial(self._nostride_dilate, dilate=self.dilate))
            self.dilate *= 2

        self.stage1 = STAGE1(
            class_num,
            norm_layer,
            resnet_out=resnet_out,
            feature=feature,
            ThreeDinit=ThreeDinit,
            bn_momentum=bn_momentum,
            pretrained_model=pretrained_model,
            eval=eval,
            freeze_bn=freeze_bn,
        )
        self.business_layer += self.stage1.business_layer

        self.stage2 = STAGE2(
            class_num,
            nn.BatchNorm3d,
            resnet_out=resnet_out,
            feature=feature,
            ThreeDinit=ThreeDinit,
            bn_momentum=bn_momentum,
            pretrained_model=pretrained_model,
            eval=eval,
            freeze_bn=freeze_bn,
        )
        self.business_layer += self.stage2.business_layer

        init_weight(
            self.business_layer,
            nn.init.kaiming_normal_,
            nn.BatchNorm3d,
            config.bn_eps,
            config.bn_momentum,
            mode="fan_in",
        )  # , nonlinearity='relu')

        state_dict = torch.load(config.pretrained_model)  # ['state_dict']
        transformed_state_dict = {}
        for k, v in state_dict.items():
            transformed_state_dict[k.replace(".bn.", ".")] = v

        self.backbone.load_state_dict(transformed_state_dict, strict=False)

    def forward(self, rgb, depth_mapping_3d, tsdf, sketch_gt=None):

        h, w = rgb.size(2), rgb.size(3)

        feature2d = self.backbone(rgb)

        (
            pred_sketch_raw,
            pred_sketch_gsnn,
            pred_sketch,
            pred_mean,
            pred_log_var,
        ) = self.stage1(tsdf, depth_mapping_3d, sketch_gt)
        pred_semantic, _ = self.stage2(
            feature2d, depth_mapping_3d, pred_sketch_raw, pred_sketch_gsnn
        )

        if self.training:
            return (
                pred_semantic,
                _,
                pred_sketch_raw,
                pred_sketch_gsnn,
                pred_sketch,
                pred_mean,
                pred_log_var,
            )
        return pred_semantic, _, pred_sketch_gsnn

    # @staticmethod
    def configure_optimizers(self):

        base_lr = self.config.lr

        """ fix the weight of resnet"""
        for param in self.backbone.parameters():
            param.requires_grad = False

        params_list = []
        for module in self.business_layer:
            params_list = group_weight(params_list, module, nn.BatchNorm2d, base_lr)

        optimizer = torch.optim.SGD(
            # self.parameters(),
            params_list,
            lr=base_lr,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
        )
        scheduler = PolynomialLRDecay(
            optimizer,
            max_decay_steps=self.config.steps_per_epoch * config.nepochs,
            end_learning_rate=1e-4,
            power=self.config.lr_power,
        )
        print(self.config.steps_per_epoch * config.nepochs)
        return [optimizer], [scheduler]  # super().configure_optimizers()

    def training_step(self, batch, batch_idx):

        img = batch["data"].float()
        hha = batch["hha_img"]
        label = batch["label"]
        label_weight = batch["label_weight"]
        tsdf = batch["tsdf"].float()
        depth_mapping_3d = batch["depth_mapping_3d"]
        sketch_gt = batch["sketch_gt"].long()
        opt = self.optimizers()
        sch = self.lr_schedulers()
        opt.zero_grad()

        (
            output,
            _,
            pred_sketch_raw,
            pred_sketch_gsnn,
            pred_sketch,
            pred_mean,
            pred_log_var,
        ) = self.forward(img, depth_mapping_3d, tsdf, sketch_gt)

        cri_weights = (torch.ones(self.config.num_classes)).type_as(tsdf)
        cri_weights[0] = self.config.empty_loss_weight
        criterion = nn.CrossEntropyLoss(
            ignore_index=255, reduction="none", weight=cri_weights.type_as(tsdf)
        )

        (
            loss,
            loss_semantic,
            loss_sketch_raw,
            loss_sketch,
            loss_sketch_gsnn,
            KLD,
        ) = compute_loss(
            label_weight,
            label,
            output,
            criterion,
            sketch_gt,
            pred_sketch_raw,
            pred_sketch,
            pred_sketch_gsnn,
            pred_log_var,
            pred_mean,
            engine=None,
            config=self.config,
        )
        self.manual_backward(loss)
        opt.step()
        sch.step()
        del (
            img,
            hha,
            tsdf,
            label_weight,
            label,
            output,
            criterion,
            sketch_gt,
            pred_sketch_raw,
            pred_sketch,
            pred_sketch_gsnn,
        )
        self.log("train/loss/total", loss, sync_dist=True, on_epoch=True, prog_bar=True)
        self.log("train/loss/semantic", loss_semantic, sync_dist=True, on_epoch=True)
        self.log(
            "train/loss/sketch_raw_loss", loss_sketch_raw, sync_dist=True, on_epoch=True
        )
        self.log("train/loss/sketch", loss_sketch, sync_dist=True, on_epoch=True)
        self.log(
            "train/loss/sketch_gsnn", loss_sketch_gsnn, sync_dist=True, on_epoch=True
        )
        self.log("train/loss/KLD", KLD, sync_dist=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if batch is None:
            return
        img = batch["data"].float()
        hha = batch["hha_img"]
        label = batch["label"]
        label_weight = batch["label_weight"]
        tsdf = batch["tsdf"].float()
        depth_mapping_3d = batch["depth_mapping_3d"]
        sketch_gt = batch["sketch_gt"].long()

        # prev = 0.5
        # b = 1.0
        # # scores = batch["scores"]
        # vis_mask = (
        #     (scores > prev) & (scores <= b)
        #     )
        # scores = scores[vis_mask]
        # self.train()
        # with torch.no_grad():
        # (
        #     output,
        #     _,
        #     pred_sketch_raw,
        #     pred_sketch_gsnn,
        #     pred_sketch,
        #     pred_mean,
        #     pred_log_var,
        # pdb.set_trace()
        output, _, _ = self.forward(img, depth_mapping_3d, tsdf, sketch_gt)
        # self.train()
        # (
        #     pred_semantic,
        #     _,
        #     pred_sketch_raw,
        #     pred_sketch_gsnn,
        #     pred_sketch,
        #     pred_mean,
        #     pred_log_var,
        # ) = self.forward(img, depth_mapping_3d, tsdf, sketch_gt)

        if self.config.dataset != "NYUv2":
            label[label == 3] = 1
            label[label == 5] = 2
        bsize = len(img)  # output.shape[0]
        batch_results = []
        for b in range(bsize):

            # pdb.set_trace()
            # output, _, pred_sketch_gsnn = self.forward(
            #     img[b].unsqueeze(0), depth_mapping_3d[b].unsqueeze(0), tsdf[b].unsqueeze(0), sketch_gt[b].unsqueeze(0)
            # )
            # # pdb.set_trace()
            score = output[b]

            score = torch.exp(score).detach().cpu()
            pred = score_to_pred(score)
            # print(pred.sum())
            if self.debug:
                prefix = batch["fn"][b]

                export_grid(
                    "{}_pred.ply".format(prefix),
                    pred
                    * label_weight[b].cpu().numpy().reshape(pred.shape).astype(int),
                )

                export_grid(
                    "{}_label_weight.ply".format(prefix),
                    label_weight[b].reshape(pred.shape).cpu().numpy().astype(int),
                )
                export_grid("{}_label.ply".format(prefix), label[b].reshape(pred.shape).cpu().numpy().astype(int))
                export_grid(
                    "{}_mapping.ply".format(prefix),
                    (depth_mapping_3d[b] != 307200).cpu().numpy().reshape(pred.shape).astype(int),
                )

            results_dict = {
                "pred": pred,
                "label": label.detach().cpu(),
                "label_weight": label_weight.detach().cpu(),
                "name": batch["fn"][b],
                "mapping": depth_mapping_3d.detach().cpu(),
            }
            batch_results.append(results_dict)
        # cri_weights = torch.ones(self.config.num_classes)
        # cri_weights[0] = self.config.empty_loss_weight
        # criterion = nn.CrossEntropyLoss(
        #     ignore_index=255, reduction="none", weight=cri_weights.type_as(tsdf)
        # )

        # (
        #     loss,
        #     loss_semantic,
        #     loss_sketch_raw,
        #     loss_sketch,
        #     loss_sketch_gsnn,
        #     KLD,
        # ) = compute_loss(
        #     label_weight,
        #     label,
        #     output,
        #     criterion,
        #     sketch_gt,
        #     pred_sketch_raw,
        #     pred_sketch,
        #     pred_sketch_gsnn,
        #     pred_log_var,
        #     pred_mean,
        #     engine=None,
        #     config=self.config,
        # )

        # del (
        #     img,
        #     hha,
        #     tsdf,
        #     label_weight,
        #     label,
        #     output,
        #     criterion,
        #     sketch_gt,
        #     pred_sketch_raw,
        #     pred_sketch,
        #     pred_sketch_gsnn,
        # )
        # self.log("val/loss/total", loss, sync_dist=True, on_epoch=True)
        # self.log("val/loss/semantic", loss_semantic, sync_dist=True, on_epoch=True)
        # self.log(
        #     "val/loss/sketch_raw_loss", loss_sketch_raw, sync_dist=True, on_epoch=True
        # )
        # self.log("val/loss/sketch", loss_sketch, sync_dist=True, on_epoch=True)
        # self.log(
        #     "val/loss/sketch_gsnn", loss_sketch_gsnn, sync_dist=True, on_epoch=True
        # )
        # self.log("val/loss/KLD", KLD, sync_dist=True, on_epoch=True)
        return {"results": batch_results}

    def validation_epoch_end(self, val_step_outputs):
        results_dicts = []

        for e in val_step_outputs:
            for x in e["results"]:
                assert type(x) == type({}), "{} must be dictionary".format(str(type(x)))
                results_dicts.append(x)
        # pdb.set_trace()

        sc, ssc = compute_metric(self.config, results_dicts, confusion=False)

        revdict = {}
        for i, j in self.config.type2class.items():
            revdict[j] = i
        for c, sem_cp in enumerate(ssc[0].tolist()):
            self.log(
                "val/sscIOU/{}".format(revdict[c]), sem_cp, sync_dist=True, batch_size=1
            )
        # lines.append("--*-- Semantic Scene Completion --*--")
        # lines.append("IOU: \n{}\n".format(str(names)))
        # lines.append("meanIOU: %f\n" % ssc[2])
        # lines.append("pixel-accuracy: %f\n" % ssc[3])
        # lines.append("")
        # lines.append("--*-- Scene Completion --*--\n")
        # lines.append("IOU: %f\n" % sc[0])
        # lines.append("pixel-accuracy: %f\n" % sc[1])  # 0 和 1 类的IOU
        # lines.append("recall: %f\n" % sc[2])

        # line = "\n".join(lines)
        # print(line)
        sscmIOU = ssc[2]
        sscPixel = ssc[3]
        scIOU = sc[0]
        scPixel = sc[1]
        scRecall = sc[2]
        # print("Val loss",val_sum_loss)
        # sscmIOU, sscPixel, scIOU, scPixel, scRecall = metric
        self.log("val/sscmIOU", sscmIOU, sync_dist=True, batch_size=1)
        self.log("val/sscPixel", sscPixel, sync_dist=True, batch_size=1)
        self.log("val/scIOU", scIOU, sync_dist=True, batch_size=1)
        self.log("val/scPixel", scPixel, sync_dist=True, batch_size=1)
        self.log("val/scRecall", scRecall, sync_dist=True, batch_size=1)

        return None

    # @staticmethod
    def _nostride_dilate(self, m, dilate):
        if isinstance(m, nn.Conv2d):
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)


if __name__ == "__main__":
    model = Network(class_num=12, norm_layer=nn.BatchNorm3d, feature=128, eval=True)
    # print(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    left = torch.rand(1, 3, 480, 640).cuda()
    right = torch.rand(1, 3, 480, 640).cuda()
    depth_mapping_3d = (
        torch.from_numpy(np.ones((1, 129600)).astype(np.int64)).long().cuda()
    )
    tsdf = torch.rand(1, 1, 60, 36, 60).cuda()

    out = model(left, depth_mapping_3d, tsdf, None)
