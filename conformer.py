import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from MobileNetV2 import mobilenet_v2
import time

writer = SummaryWriter('log/run' + time.strftime("%d-%m"))

# Depthwise Separable Convolution
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class JLModule(nn.Module):
    def __init__(self, backbone):
        super(JLModule, self).__init__()
        self.backbone = backbone

    def load_pretrained_model(self, model_path):
        pretrained_dict = torch.load(model_path)
        model_dict = self.backbone.state_dict()
        pretrained_dict = {ka: va for ka, va in pretrained_dict.items() if ka in model_dict}
        model_dict.update(pretrained_dict)
        self.backbone.load_state_dict(model_dict)

    def forward(self, x):
        convr = self.backbone(x)
        return convr[1], convr[2], convr[3], convr[4]

class ShuffleChannelAttention(nn.Module):
    def __init__(self, channel=64, reduction=16, groups=8):
        super(ShuffleChannelAttention, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.g = groups
        self.se = nn.Sequential(
            DepthwiseSeparableConv(channel, channel // reduction, kernel_size=1, padding=0),
            nn.ReLU(),
            DepthwiseSeparableConv(channel // reduction, channel, kernel_size=3)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape
        residual = x
        max_result = self.maxpool(x)
        shuffled_in = max_result.view(b, self.g, c // self.g, 1, 1).permute(0, 2, 1, 3, 4).reshape(b, c, 1, 1)
        max_out = self.se(shuffled_in)
        output1 = self.sigmoid(max_out).view(b, c, 1, 1)
        output2 = self.sigmoid(max_result)
        output = output1 + output2
        return (output * x) + residual

class LDELayer(nn.Module):
    def __init__(self):
        super(LDELayer, self).__init__()
        self.operation_stage_1 = nn.Sequential(DepthwiseSeparableConv(32, 32), nn.ReLU())
        self.ca_1 = ShuffleChannelAttention(channel=32, reduction=16, groups=2)
        self.last_conv1x1 = nn.Conv2d(32, 1, 1, 1)

    def forward(self, x):
        rgb_1 = self.operation_stage_1(x)
        depth_1 = self.ca_1(x)
        rgbd_fusion_1 = x + (rgb_1 * depth_1)
        last_out = self.last_conv1x1(rgbd_fusion_1)
        return last_out

class CoarseLayer(nn.Module):
    def __init__(self):
        super(CoarseLayer, self).__init__()
        self.relu = nn.ReLU()
        self.conv_r = nn.Sequential(
            DepthwiseSeparableConv(320, 160, kernel_size=1, padding=0),
            self.relu,
            nn.Conv2d(160, 1, 1, 1)
        )

    def forward(self, x):
        return self.conv_r(x)

class GDELayer(nn.Module):
    def __init__(self):
        super(GDELayer, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.convH = nn.Sequential(
            DepthwiseSeparableConv(320, 160, kernel_size=1, padding=0),
            self.relu,
            nn.Conv2d(160, 1, 1, 1)
        )
        self.convM = nn.Sequential(
            DepthwiseSeparableConv(96, 32, kernel_size=1, padding=0),
            self.relu,
            nn.Conv2d(32, 1, 1, 1)
        )
        self.upsampling = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, convr4, convr5, coarse_sal_rgb):
        convr5_rgb_part = self.convH(convr5)
        salr = self.sigmoid(coarse_sal_rgb)
        rgb_h = (1 - salr) * convr5_rgb_part

        convr4_rgb_part = self.convM(convr4)
        coarse_sal_rgb1 = self.upsampling(coarse_sal_rgb)
        salr = self.sigmoid(coarse_sal_rgb1)
        rgb_m = (1 - salr) * convr4_rgb_part

        return rgb_h, rgb_m

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.upsample = nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up2 = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1)
        self.up21 = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1)
        self.act = nn.Sigmoid()

    def forward(self, lde_out, rgb_h, rgb_m):
        lde_out1 = self.upsample(lde_out)
        edge_rgbd0 = self.act(self.up21(self.up21(lde_out1)))
        sal_final = edge_rgbd0 + self.up21(self.up2(self.up2(self.up2(rgb_m + self.up2(rgb_h)))))
        return sal_final, edge_rgbd0

class JL_DCF(nn.Module):
    def __init__(self, JLModule, lde_layers, coarse_layer, gde_layers, decoder):
        super(JL_DCF, self).__init__()
        self.JLModule = JLModule
        self.lde = lde_layers
        self.coarse_layer = coarse_layer
        self.gde_layers = gde_layers
        self.decoder = decoder

    def forward(self, x):
        conv1r, conv2r, conv3r, conv4r = self.JLModule(x)
        lde_out = self.lde(conv2r)
        coarse_sal_rgb = self.coarse_layer(conv4r)
        rgb_h, rgb_m = self.gde_layers(conv3r, conv4r, coarse_sal_rgb)
        sal_final, edge_rgbd0 = self.decoder(lde_out, rgb_h, rgb_m)
        return sal_final, coarse_sal_rgb, edge_rgbd0, lde_out, rgb_h, rgb_m

def build_model(network='conformer', base_model_cfg='conformer'):
    backbone = mobilenet_v2()
    return JL_DCF(JLModule(backbone), LDELayer(), CoarseLayer(), GDELayer(), Decoder())
