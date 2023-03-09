# Copyright (c) 2021. All rights reserved.
from __future__ import print_function
import math
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from models.submodule import feature_extraction, MobileV2_Residual_3D, convbn_3d, build_gwc_volume, disparity_regression


class hourglass3D(nn.Module):
    def __init__(self, in_channels):
        super(hourglass3D, self).__init__()

        self.expanse_ratio = 2

        self.conv1 = MobileV2_Residual_3D(in_channels, in_channels * 2, 2, self.expanse_ratio)

        self.conv2 = MobileV2_Residual_3D(in_channels * 2, in_channels * 2, 1, self.expanse_ratio)

        self.conv3 = MobileV2_Residual_3D(in_channels * 2, in_channels * 4, 2, self.expanse_ratio)

        self.conv4 = MobileV2_Residual_3D(in_channels * 4, in_channels * 4, 1, self.expanse_ratio)

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels * 2))

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels))

        self.redir1 = MobileV2_Residual_3D(in_channels, in_channels, 1, self.expanse_ratio)
        self.redir2 = MobileV2_Residual_3D(in_channels * 2, in_channels * 2, 1, self.expanse_ratio)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)

        return conv6


class MSNet3D(nn.Module):
    def __init__(self, maxdisp):

        super(MSNet3D, self).__init__()

        self.maxdisp = maxdisp

        self.hourglass_size = 32

        self.dres_expanse_ratio = 3

        self.num_groups = 40

        self.feature_extraction = feature_extraction()

        self.dres0 = nn.Sequential(
            MobileV2_Residual_3D(self.num_groups, self.hourglass_size, 1, self.dres_expanse_ratio),
            MobileV2_Residual_3D(self.hourglass_size, self.hourglass_size, 1, self.dres_expanse_ratio))

        self.dres1 = nn.Sequential(
            MobileV2_Residual_3D(self.hourglass_size, self.hourglass_size, 1, self.dres_expanse_ratio),
            MobileV2_Residual_3D(self.hourglass_size, self.hourglass_size, 1, self.dres_expanse_ratio))

        self.encoder_decoder1 = hourglass3D(self.hourglass_size)

        self.encoder_decoder2 = hourglass3D(self.hourglass_size)

        self.encoder_decoder3 = hourglass3D(self.hourglass_size)

        self.classif0 = nn.Sequential(convbn_3d(self.hourglass_size, self.hourglass_size, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(self.hourglass_size, 1, kernel_size=3, padding=1, stride=1, bias=False,
                                                dilation=1))
        self.classif1 = nn.Sequential(convbn_3d(self.hourglass_size, self.hourglass_size, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(self.hourglass_size, 1, kernel_size=3, padding=1, stride=1, bias=False,
                                                dilation=1))
        self.classif2 = nn.Sequential(convbn_3d(self.hourglass_size, self.hourglass_size, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(self.hourglass_size, 1, kernel_size=3, padding=1, stride=1, bias=False,
                                                dilation=1))
        self.classif3 = nn.Sequential(convbn_3d(self.hourglass_size, self.hourglass_size, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(self.hourglass_size, 1, kernel_size=3, padding=1, stride=1, bias=False,
                                                dilation=1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, L, R):
        features_left = self.feature_extraction(L)
        features_right = self.feature_extraction(R)

        volume = build_gwc_volume(features_left, features_right, self.maxdisp // 4, self.num_groups)

        cost0 = self.dres0(volume)
        cost0 = self.dres1(cost0) + cost0

        out1 = self.encoder_decoder1(cost0)
        out2 = self.encoder_decoder2(out1)
        out3 = self.encoder_decoder3(out2)

        if self.training:
            cost0 = self.classif0(cost0)
            cost1 = self.classif1(out1)
            cost2 = self.classif2(out2)
            cost3 = self.classif3(out3)

            cost0 = F.interpolate(cost0, [self.maxdisp, L.size()[2], L.size()[3]], mode='trilinear')
            cost0 = torch.squeeze(cost0, 1)
            pred0 = F.softmax(cost0, dim=1)
            pred0 = disparity_regression(pred0, self.maxdisp)

            cost1 = F.interpolate(cost1, [self.maxdisp, L.size()[2], L.size()[3]], mode='trilinear')
            cost1 = torch.squeeze(cost1, 1)
            pred1 = F.softmax(cost1, dim=1)
            pred1 = disparity_regression(pred1, self.maxdisp)

            cost2 = F.interpolate(cost2, [self.maxdisp, L.size()[2], L.size()[3]], mode='trilinear')
            cost2 = torch.squeeze(cost2, 1)
            pred2 = F.softmax(cost2, dim=1)
            pred2 = disparity_regression(pred2, self.maxdisp)

            cost3 = F.interpolate(cost3, [self.maxdisp, L.size()[2], L.size()[3]], mode='trilinear')
            cost3 = torch.squeeze(cost3, 1)
            pred3 = F.softmax(cost3, dim=1)
            pred3 = disparity_regression(pred3, self.maxdisp)

            return [pred0, pred1, pred2, pred3]

        else:
            cost3 = self.classif3(out3)
            cost3 = F.interpolate(cost3, [self.maxdisp, L.size()[2], L.size()[3]], mode='trilinear')
            cost3 = torch.squeeze(cost3, 1)
            pred3 = F.softmax(cost3, dim=1)
            pred3 = disparity_regression(pred3, self.maxdisp)

            return [pred3]
