import torch
from torch import nn

from modules.conv import conv, conv_dw, conv_dw_no_bn


class Cpm(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.align = conv(in_channels, out_channels, kernel_size=1, padding=0, bn=False)
        self.trunk = nn.Sequential(
            conv_dw_no_bn(out_channels, out_channels),
            conv_dw_no_bn(out_channels, out_channels),
            conv_dw_no_bn(out_channels, out_channels)
        )
        self.conv = conv(out_channels, out_channels, bn=False)

    def forward(self, x):
        x = self.align(x)
        x = self.conv(x + self.trunk(x))
        return x

class Cat_rgbd(nn.Module):
    def __init__(self, in_channels, out_channels,pooling = False,kernel_size=2):
        super().__init__()
        self.align = conv(in_channels, out_channels, kernel_size=1, padding=0, bn=False)
        self.avgpool = nn.AvgPool2d(kernel_size)
        self.pooling = pooling

    def forward(self, x):
        x = self.align(x)
        if self.pooling == True:
            x = self.avgpool(x)
        return x

class InitialStage(nn.Module):
    def __init__(self, num_channels, num_heatmaps, num_pafs):
        super().__init__()
        self.trunk = nn.Sequential(
            conv(num_channels, num_channels, bn=False),
            conv(num_channels, num_channels, bn=False),
            conv(num_channels, num_channels, bn=False)
        )
        self.heatmaps = nn.Sequential(
            conv(num_channels, 512, kernel_size=1, padding=0, bn=False),
            conv(512, num_heatmaps, kernel_size=1, padding=0, bn=False, relu=False)
        )
        self.pafs = nn.Sequential(
            conv(num_channels, 512, kernel_size=1, padding=0, bn=False),
            conv(512, num_pafs, kernel_size=1, padding=0, bn=False, relu=False)
        )

    def forward(self, x):
        trunk_features = self.trunk(x)
        heatmaps = self.heatmaps(trunk_features)
        pafs = self.pafs(trunk_features)
        return [heatmaps, pafs]


class RefinementStageBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.initial = conv(in_channels, out_channels, kernel_size=1, padding=0, bn=False)
        self.trunk = nn.Sequential(
            conv(out_channels, out_channels),
            conv(out_channels, out_channels, dilation=2, padding=2)
        )

    def forward(self, x):
        initial_features = self.initial(x)
        trunk_features = self.trunk(initial_features)
        return initial_features + trunk_features


class RefinementStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_heatmaps, num_pafs):
        super().__init__()
        self.trunk = nn.Sequential(
            RefinementStageBlock(in_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels)
        )
        self.heatmaps = nn.Sequential(
            conv(out_channels, out_channels, kernel_size=1, padding=0, bn=False),
            conv(out_channels, num_heatmaps, kernel_size=1, padding=0, bn=False, relu=False)
        )
        self.pafs = nn.Sequential(
            conv(out_channels, out_channels, kernel_size=1, padding=0, bn=False),
            conv(out_channels, num_pafs, kernel_size=1, padding=0, bn=False, relu=False)
        )

    def forward(self, x):
        trunk_features = self.trunk(x)
        heatmaps = self.heatmaps(trunk_features)
        pafs = self.pafs(trunk_features)
        return [heatmaps, pafs]


class PoseEstimationWithMobileNet(nn.Module):
    def __init__(self, num_refinement_stages=1, num_channels=128, num_heatmaps=19, num_pafs=38):
        super().__init__()
        self.model = nn.Sequential(
            conv(     3,  32, stride=2, bias=False), # 112*112
            conv_dw( 32,  64),
            conv_dw( 64, 128, stride=2),             # 56*56
            conv_dw(128, 128),
            conv_dw(128, 256, stride=2),             # 28*28
            conv_dw(256, 256),
            conv_dw(256, 512),  # conv4_2
            conv_dw(512, 512, dilation=2, padding=2),# 28*28
            conv_dw(512, 512),
            conv_dw(512, 512),
            conv_dw(512, 512),
            conv_dw(512, 512)   # conv5_5            # 28*28
        )
        for p in self.model.parameters():
            p.requires_grad = False
        print('model frozen')
        self.model_rgbd = nn.Sequential(
            conv(     3,  32, stride=2, bias=False),
            conv_dw( 32,  64),
            conv_dw( 64, 128, stride=2),
            conv_dw(128, 128),
            conv_dw(128, 256, stride=2),
            conv_dw(256, 256),
            conv_dw(256, 512),  # conv4_2
            conv_dw(512, 512, dilation=2, padding=2),
            conv_dw(512, 512),
            conv_dw(512, 512),
            conv_dw(512, 512),
            conv_dw(512, 512)   # conv5_5
        )
        for p in self.model_rgbd.parameters():
            p.requires_grad = False
        print('modelrgbd frozen')
        self.rgb_rgbd_0 = Cat_rgbd(64,128,pooling = True,kernel_size = 4)
        self.rgb_rgbd_1 = Cat_rgbd(256,128,pooling = True,kernel_size = 2)
        self.rgb_rgbd_2 = Cat_rgbd(512,128,pooling = False)
        self.rgb_rgbd_3 = Cat_rgbd(1024,128,pooling = False)
        self.rgb_rgbd_4 = Cat_rgbd(1024,128,pooling = False)
        self.cpm_fusion = Cpm(128*5,128)
        # self.cpm = Cpm(512, num_channels)
        # self.cpm_rgbd = Cpm(512, num_channels)
        # self.rgb_rgbd = conv(num_channels * 2, num_channels, kernel_size = 1, padding=0)
        self.initial_stage = InitialStage(num_channels, num_heatmaps, num_pafs)
        self.refinement_stages = nn.ModuleList()
        for idx in range(num_refinement_stages):
            self.refinement_stages.append(RefinementStage(num_channels + num_heatmaps + num_pafs, num_channels,
                                                          num_heatmaps, num_pafs))

    def forward(self, x):     
        rgb, depth = x.split([3, 1], dim=1) 
        depth = depth.repeat(1,3,1,1)
        rgbd_stage_0 = self.rgb_rgbd_0(torch.cat([self.model[0](rgb), self.model_rgbd[0](depth)], dim=1))
        rgbd_stage_1 = self.rgb_rgbd_1(torch.cat([self.model[0:3](rgb), self.model_rgbd[0:3](depth)], dim=1))
        rgbd_stage_2 = self.rgb_rgbd_2(torch.cat([self.model[0:5](rgb), self.model_rgbd[0:5](depth)], dim=1))
        rgbd_stage_3 = self.rgb_rgbd_3(torch.cat([self.model[0:8](rgb), self.model_rgbd[0:8](depth)], dim=1))
        rgbd_stage_4 = self.rgb_rgbd_4(torch.cat([self.model[0:10](rgb), self.model_rgbd[0:10](depth)], dim=1))
        backbone_features = self.cpm_fusion(torch.cat([rgbd_stage_0,rgbd_stage_1,rgbd_stage_2,rgbd_stage_3,rgbd_stage_4],dim=1))
        # rgb_backbone_features = self.model(rgb)
        # rgb_backbone_features = self.cpm(rgb_backbone_features)
        # depth_backbone_features = self.model_rgbd(rgbd)
        # depth_backbone_features = self.cpm_rgbd(depth_backbone_features)
        # backbone_features = self.rgb_rgbd(torch.cat([rgb_backbone_features, depth_backbone_features], dim=1)) #拼接 + 1x1卷积

        stages_output = self.initial_stage(backbone_features)
        for refinement_stage in self.refinement_stages:
            stages_output.extend(
                refinement_stage(torch.cat([backbone_features, stages_output[-2], stages_output[-1]], dim=1)))

        return stages_output
