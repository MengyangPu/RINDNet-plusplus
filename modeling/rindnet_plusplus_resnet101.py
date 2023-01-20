import math
import torch.nn.functional as F
import torch
import torch.nn as nn
import torchvision.models as models
from modeling.backbone.resnet101 import ResNet101
from modeling.attention_module import ATT

class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, expansion=4, dilation_rate=1):
        super(Bottleneck, self).__init__()
        self.expansion = expansion
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        pad = 2 if dilation_rate == 2 else 1
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=pad, bias=False, dilation=dilation_rate)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        ## output channel: 4*inplanes
        return out

class RI_Decoder(nn.Module):
    def __init__(self):
        super(RI_Decoder, self).__init__()
        # fuse c4 and c5
        self.c5 = nn.Sequential(
            nn.Conv2d(2048, 16, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 16, kernel_size=16, stride=8, padding=4, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True)
            )
        # c5_3 * c3 ====> c3_1
        # c3 ---up---> c3_up
        self.up3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        )
        self.conv23 = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )
        # c5_2 * c2 ====> c2_1
        # c2_1 ---up---> c2_up
        self.up2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv12 = nn.Sequential(
            nn.Conv2d(128, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True)
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True))

    def forward(self, c1, c2, c3, c5):
        #[1, 64, 320, 320]   [1, 256, 160, 160]   [1, 512, 80, 80]   [1, 1024, 40, 40]    [1, 2048, 40, 40]
        c5 = self.c5(c5)    #[1, 16, 320, 320]

        c3_up = self.up3(c3)  #[1, 256, 160, 160]
        c23 = self.conv23(torch.cat([c2, c3_up], dim=1))    #[1, 64, 160, 160]

        c2_up = self.up2(c23)    #[1, 64, 320, 320]
        c12 = self.conv12(torch.cat([c1, c2_up], dim=1))    #[1, 16, 320, 320]

        outs = self.fuse(torch.cat([c12, c5], dim=1))    #[1, 32, 320, 320]
        return outs


class DN_Decoder(nn.Module):
    def __init__(self):
        super(DN_Decoder, self).__init__()
        self.inplanes = 64
        self.c5 = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True))

        self.c5_depth = nn.Sequential(
            nn.Conv2d(256, 16, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 16, kernel_size=8, stride=4, padding=2, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True)
        )

        self.c5_normal = nn.Sequential(
            nn.Conv2d(256, 16, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 16, kernel_size=8, stride=4, padding=2, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True)
        )

        self.c5_dn = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            self._make_resblock(Bottleneck, 256, 64),
            self._make_resblock(Bottleneck, 256, 8, expansion=2),
            nn.ConvTranspose2d(16, 16, kernel_size=8, stride=4, padding=2, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True)
        )

    def _make_resblock(self, block, inplanes, planes, stride=1, expansion=4):
        downsample = None
        if stride != 1 or self.inplanes != planes * expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * expansion),
            )

        return block(inplanes, planes, stride, downsample,expansion=expansion)

    def forward(self, c5):
        c5 = self.c5(c5)#[1, 256, 40, 40]
        c5_depth = self.c5_depth(c5)
        c5_normal = self.c5_normal(c5)
        c5_share = self.c5_dn(c5)
        return torch.cat([c5_depth,c5_share],dim=1), torch.cat([c5_normal,c5_share],dim=1)

class RI_OutHead(nn.Module):
    def __init__(self):
        super(RI_OutHead, self).__init__()
        self.conv_stage = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1), nn.ReLU(inplace=True)
        )
        self.out = nn.Conv2d(1, 1, kernel_size=1)

    def forward(self, x):
        x = self.conv_stage(x)
        x = self.out(x)
        return x

class DN_OutHead(nn.Module):
    def __init__(self):
        super(DN_OutHead, self).__init__()
        self.conv_stage = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True)
        )
        self.out = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        x = self.conv_stage(x)
        x = self.out(x)
        return x

class MyNet(nn.Module):

    def __init__(self, nclass=5, backbone_name='resnet101', norm_layer=nn.BatchNorm2d):
        self.inplanes = 64
        super(MyNet, self).__init__()
        self.nclass = nclass
        self.backbone_name = backbone_name
        self.norm_layer = norm_layer

        # Conv_Backbone ResNet50
        self.backbone = ResNet101(pretrained=True)

        # Attention Module
        self.att = ATT()

        # Decoder
        self.R_Decoder = RI_Decoder()
        self.I_Decoder = RI_Decoder()
        self.DN_Decoder = DN_Decoder()
        # Decision Head
        self.head_r = RI_OutHead()
        self.head_i = RI_OutHead()
        self.head_n = DN_OutHead()
        self.head_d = DN_OutHead()

        ## init param
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        print('Initialize ResNet ')
        self.load_resnet()

    def load_resnet(self):
        print('Loading pre-trained resnet101')
        resnet101 = models.resnet101(pretrained=True)
        pretrained_dict = resnet101.state_dict()

        ignore_keys = ['fc.weight', 'fc.bias']
        model_dict = self.backbone.state_dict()

        for k, v in list(pretrained_dict.items()):
            if k in ignore_keys:
                pretrained_dict.pop(k)
        model_dict.update(pretrained_dict)
        self.backbone.load_state_dict(model_dict)

    def forward(self, x):  ## when x: (1, 3, 320, 320)
        ## resnet-50
        c1, c2, c3, c4, c5 = self.backbone(x) #[1, 64, 320, 320]   [1, 256, 160, 160]   [1, 512, 80, 80]   [1, 1024, 40, 40]    [1, 2048, 40, 40]

        # attention map
        att_outs = self.att(x)  # (1, 5, 320, 320)
        boundary_soft = torch.softmax(att_outs,1)

        # feature maps from RI-Decoder
        r_decoder_features = self.R_Decoder(c1, c2, c3, c5)
        i_decoder_features = self.I_Decoder(c1, c2, c3, c5)
        d_decoder_features, n_decoder_features = self.DN_Decoder(c5)

        out_reflectance = self.head_r(r_decoder_features)
        out_illumination = self.head_i(i_decoder_features)
        out_depth = self.head_d(d_decoder_features)
        out_normal = self.head_n(n_decoder_features)

        out_depth = out_depth * (1.0 + boundary_soft[:, 1:2, :, :])
        out_normal = out_normal * (1.0 + boundary_soft[:, 2:3, :, :])
        out_reflectance = out_reflectance * (1.0 + boundary_soft[:, 3:4, :, :])
        out_illumination = out_illumination * (1.0 + boundary_soft[:, 4:5, :, :])

        out_depth = torch.sigmoid(out_depth)
        out_normal = torch.sigmoid(out_normal)
        out_reflectance = torch.sigmoid(out_reflectance)
        out_illumination = torch.sigmoid(out_illumination)

        return att_outs, out_depth,out_normal,out_reflectance,out_illumination


if __name__ == '__main__':
    model = MyNet()
    dummy_input = torch.rand(1, 3, 320, 320)
    output = model(dummy_input)
    for out in output:
        print(out.size())