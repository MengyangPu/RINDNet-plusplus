import math
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision.models as models

def Conv_Stage(input_dim,dim_list, bias=True, output_map=False):
    num_layers = len(dim_list)
    dim_list = [input_dim] + dim_list

    layers = []
    for i in range(num_layers):
        layer = nn.Sequential(
            nn.Conv2d(dim_list[i], dim_list[i+1], kernel_size=3, bias=bias,padding=1),
            nn.BatchNorm2d(dim_list[i+1]),
            nn.ReLU(inplace=True)
            )
        layers.append(layer)

    if output_map:
        layer = nn.Conv2d(dim_list[-1], 1, kernel_size=1)
        layers.append(layer)

    ## with padding, doesn't change the resolution
    return nn.Sequential(*layers)

def Conv_Stage2(input_dim,dim_list, bias=True, output_map=False):
    num_layers = len(dim_list)
    dim_list = [input_dim] + dim_list

    layers = []
    for i in range(num_layers):
        layer = nn.Sequential(
            nn.Conv2d(dim_list[i], dim_list[i+1], kernel_size=3, bias=bias,padding=1),
            nn.BatchNorm2d(dim_list[i+1]),
            nn.ReLU(inplace=True)
            )
        layers.append(layer)

    if output_map:
        layer = nn.Conv2d(dim_list[-1], 5, kernel_size=1)
        layers.append(layer)

    ## with padding, doesn't change the resolution
    return nn.Sequential(*layers)

class ATT(nn.Module):

    def __init__(self):
        self.inplanes = 64
        super(ATT, self).__init__()

        # conv1 for boundary
        self.conv1_b = Conv_Stage(3, [8, 4, 16])
        self.conv2_b = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.soft_boundary=Conv_Stage2(16, [8, 8, 8, 8], output_map=True)

        ## init param
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):
        # extra branch conv1
        xf_1_b = self.conv1_b(x)  # (1, 16, 320, 320)
        xf_2_b = self.conv2_b(x)  # (1, 16, 320, 320)
        unet1 = torch.add(xf_1_b, xf_2_b)  # (1, 16, 320, 320)
        outs = self.soft_boundary(unet1)

        return outs

if __name__ == '__main__':
    model = ATT()
    dummy_input = torch.rand(1, 3, 320, 320)
    output = model(dummy_input)
    for out in output:
        print(out.size())