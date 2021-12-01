from torch.autograd.grad_mode import F
from torch.nn import Module, Upsample
from torch import Tensor, where
from CRAFT.craft_utils import getDetBoxes, adjustResultCoordinates
#from CRAFT.craft import CRAFT
from collections import OrderedDict
from pathlib import Path
from torch import device, load
#from CRAFT.refinenet import RefineNet
from torch.nn import Module, ModuleDict, Conv2d, ConvTranspose2d, Sigmoid, LeakyReLU, parameter, MaxPool2d, Linear, BatchNorm2d, Sequential, init, ReLU
from torch import cat, linspace, meshgrid, arange, cos, sin, exp, linalg, Tensor, flatten, randn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt

from collections import namedtuple

from torchvision import models
from torchvision.models.vgg import model_urls

class RefineNet(Module):
    def __init__(self):
        super(RefineNet, self).__init__()

        self.last_conv = Sequential(
            Conv2d(34, 64, kernel_size=3, padding=1), BatchNorm2d(64), ReLU(inplace=True),
            Conv2d(64, 64, kernel_size=3, padding=1), BatchNorm2d(64), ReLU(inplace=True),
            Conv2d(64, 64, kernel_size=3, padding=1), BatchNorm2d(64), ReLU(inplace=True)
        )

        self.aspp1 = Sequential(
            Conv2d(64, 128, kernel_size=3, dilation=6, padding=6), BatchNorm2d(128), ReLU(inplace=True),
            Conv2d(128, 128, kernel_size=1), BatchNorm2d(128), ReLU(inplace=True),
            Conv2d(128, 1, kernel_size=1)
        )

        self.aspp2 = Sequential(
            Conv2d(64, 128, kernel_size=3, dilation=12, padding=12), BatchNorm2d(128), ReLU(inplace=True),
            Conv2d(128, 128, kernel_size=1), BatchNorm2d(128), ReLU(inplace=True),
            Conv2d(128, 1, kernel_size=1)
        )

        self.aspp3 = Sequential(
            Conv2d(64, 128, kernel_size=3, dilation=18, padding=18), BatchNorm2d(128), ReLU(inplace=True),
            Conv2d(128, 128, kernel_size=1), BatchNorm2d(128), ReLU(inplace=True),
            Conv2d(128, 1, kernel_size=1)
        )

        self.aspp4 = Sequential(
            Conv2d(64, 128, kernel_size=3, dilation=24, padding=24), BatchNorm2d(128), ReLU(inplace=True),
            Conv2d(128, 128, kernel_size=1), BatchNorm2d(128), ReLU(inplace=True),
            Conv2d(128, 1, kernel_size=1)
        )

        init_weights(self.last_conv.modules())
        init_weights(self.aspp1.modules())
        init_weights(self.aspp2.modules())
        init_weights(self.aspp3.modules())
        init_weights(self.aspp4.modules())

    def forward(self, y, upconv4):
        refine = cat([y.permute(0,3,1,2), upconv4], dim=1)
        refine = self.last_conv(refine)

        aspp1 = self.aspp1(refine)
        aspp2 = self.aspp2(refine)
        aspp3 = self.aspp3(refine)
        aspp4 = self.aspp4(refine)

        #out = torch.add([aspp1, aspp2, aspp3, aspp4], dim=1)
        out = aspp1 + aspp2 + aspp3 + aspp4
        return out.permute(0, 2, 3, 1)  # , refine.permute(0,2,3,1)



class double_conv(Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = Sequential(
            Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
            BatchNorm2d(mid_ch),
            ReLU(inplace=True),
            Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            BatchNorm2d(out_ch),
            ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class CRAFT(Module):
    def __init__(self, pretrained=False, freeze=False):
        super(CRAFT, self).__init__()

        """ Base network """
        self.basenet = vgg16_bn(pretrained, freeze)

        """ U network """
        self.upconv1 = double_conv(1024, 512, 256)
        self.upconv2 = double_conv(512, 256, 128)
        self.upconv3 = double_conv(256, 128, 64)
        self.upconv4 = double_conv(128, 64, 32)

        num_class = 2
        self.conv_cls = Sequential(
            Conv2d(32, 32, kernel_size=3, padding=1), ReLU(inplace=True),
            Conv2d(32, 32, kernel_size=3, padding=1), ReLU(inplace=True),
            Conv2d(32, 16, kernel_size=3, padding=1), ReLU(inplace=True),
            Conv2d(16, 16, kernel_size=1), ReLU(inplace=True),
            Conv2d(16, num_class, kernel_size=1),
        )

        init_weights(self.upconv1.modules())
        init_weights(self.upconv2.modules())
        init_weights(self.upconv3.modules())
        init_weights(self.upconv4.modules())
        init_weights(self.conv_cls.modules())
        
    def forward(self, x):
        """ Base network """
        sources = self.basenet(x)

        """ U network """
        y = cat([sources[0], sources[1]], dim=1)
        y = self.upconv1(y)

        y = F.interpolate(y, size=sources[2].size()[2:], mode='bilinear', align_corners=False)
        y = cat([y, sources[2]], dim=1)
        y = self.upconv2(y)

        y = F.interpolate(y, size=sources[3].size()[2:], mode='bilinear', align_corners=False)
        y = cat([y, sources[3]], dim=1)
        y = self.upconv3(y)

        y = F.interpolate(y, size=sources[4].size()[2:], mode='bilinear', align_corners=False)
        y = cat([y, sources[4]], dim=1)
        feature = self.upconv4(y)

        y = self.conv_cls(feature)

        return y.permute(0,2,3,1), feature

if __name__ == '__main__':
    model = CRAFT(pretrained=True).cuda()
    output, _ = model(randn(1, 3, 768, 768).cuda())
    print(output.shape)

def init_weights(modules):
    for m in modules:
        if isinstance(m, Conv2d):
            init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()

class vgg16_bn(Module):
    def __init__(self, pretrained=True, freeze=True):
        super(vgg16_bn, self).__init__()
        model_urls['vgg16_bn'] = model_urls['vgg16_bn'].replace('https://', 'http://')
        vgg_pretrained_features = models.vgg16_bn(pretrained=pretrained).features
        self.slice1 = Sequential()
        self.slice2 = Sequential()
        self.slice3 = Sequential()
        self.slice4 = Sequential()
        self.slice5 = Sequential()
        for x in range(12):         # conv2_2
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 19):         # conv3_3
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(19, 29):         # conv4_3
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(29, 39):         # conv5_3
            self.slice4.add_module(str(x), vgg_pretrained_features[x])

        # fc6, fc7 without atrous conv
        self.slice5 = Sequential(
                MaxPool2d(kernel_size=3, stride=1, padding=1),
                Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
                Conv2d(1024, 1024, kernel_size=1)
        )

        if not pretrained:
            init_weights(self.slice1.modules())
            init_weights(self.slice2.modules())
            init_weights(self.slice3.modules())
            init_weights(self.slice4.modules())

        init_weights(self.slice5.modules())        # no pretrained model for fc6 and fc7

        if freeze:
            for param in self.slice1.parameters():      # only first conv
                param.requires_grad= False

    def forward(self, X):
        h = self.slice1(X)
        h_relu2_2 = h
        h = self.slice2(h)
        h_relu3_2 = h
        h = self.slice3(h)
        h_relu4_3 = h
        h = self.slice4(h)
        h_relu5_3 = h
        h = self.slice5(h)
        h_fc7 = h
        vgg_outputs = namedtuple("VggOutputs", ['fc7', 'relu5_3', 'relu4_3', 'relu3_2', 'relu2_2'])
        out = vgg_outputs(h_fc7, h_relu5_3, h_relu4_3, h_relu3_2, h_relu2_2)
        return out


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

class labelExtractor(Module):
    def __init__(self, savedPathDetection:Path, savedPathRefiner:Path, cudaDevice:device, textThreshold:float, linkThreshold:float, lowText:float) -> None:
        super().__init__()
        self.detectionModel = CRAFT()
        print(f'Loading weights from checkpoint {savedPathDetection}')   
        self.detectionModel.load_state_dict(copyStateDict(load(savedPathDetection)))
        self.detectionModel.to(cudaDevice)
        self.detectionModel.eval()

        self.refinerModel = RefineNet()
        print(f'Loading weights of refiner from checkpoint ({savedPathRefiner})')
        self.refinerModel.load_state_dict(copyStateDict(load(savedPathRefiner)))
        self.refinerModel.to(cudaDevice)
        self.refinerModel.eval()

        self.textThreshold = textThreshold
        self.linkThreshold = linkThreshold
        self.lowText = lowText

        self.Upsample = Upsample(scale_factor=2)

    def forward(self, thumbnail:Tensor) -> list:
        y, feature = self.detectionModel(thumbnail)
        # make score and link map
        y_refiner = self.refinerModel(y, feature)
        y_ = y_refiner[:,:,:,0].unsqueeze(0)
        # Post-processing
        boxes, _ = getDetBoxes(y[0,:,:,0].cpu().data.numpy(), y_refiner[0,:,:,0].cpu().data.numpy(), self.textThreshold, self.linkThreshold, self.lowText, False)
        # coordinate adjustment
        boxes = adjustResultCoordinates(boxes, 1, 1)
        return boxes, where(self.Upsample(y_)>0.1,1,0)

class down2d(Module):
    def __init__(self, inChannels:int, outChannels:int, filterSize:int):
        super(down2d, self).__init__()
        self.pooling2d = Conv2d(inChannels,  inChannels, 4, stride=2, padding=1)        
        self.conv1 = Conv2d(inChannels,  outChannels, filterSize, stride=1, padding=int((filterSize - 1) / 2))        
        self.conv2 = Conv2d(outChannels, outChannels, filterSize, stride=1, padding=int((filterSize - 1) / 2))
        self.lRelu = LeakyReLU(negative_slope = 0.1)
           
    def forward(self, x:Tensor) -> Tensor:
        x = self.lRelu(self.pooling2d(x))
        x = self.lRelu(self.conv1(x))
        x = self.lRelu(self.conv2(x))
        return x
    
class up2d(Module):
    def __init__(self, inChannels:int, outChannels:int):
        super(up2d, self).__init__()
        self.unpooling2d = ConvTranspose2d(inChannels, inChannels, 4, stride = 2, padding = 1)
        self.conv1 = Conv2d(inChannels,  outChannels, 3, stride=1, padding=1)
        self.conv2 = Conv2d(2 * outChannels, outChannels, 3, stride=1, padding=1)
        self.lRelu = LeakyReLU(negative_slope = 0.1)
           
    def forward(self, x:Tensor, skpCn:Tensor) -> Tensor:
        x = self.lRelu(self.unpooling2d(x))
        x = self.lRelu(self.conv1(x))
        x = self.lRelu(self.conv2(cat((x, skpCn), 1)))
        return x    

class UNet2d(Module):
    def __init__(self, inChannels:int, outChannels:int, ngf:int, fs:int):
        super(UNet2d, self).__init__()
        # Initialize neural network blocks.
        self.conv1 = Conv2d(inChannels, ngf, fs, stride=1, padding=int((fs - 1) / 2))
        self.conv2 = Conv2d(ngf, ngf, fs, stride=1, padding=int((fs - 1) / 2))
        self.down1 = down2d(ngf, 2*ngf, 5)
        self.down2 = down2d(2*ngf, 4*ngf, 3)
        self.down3 = down2d(4*ngf, 8*ngf, 3)
        self.down4 = down2d(8*ngf, 16*ngf, 3)
        self.down5 = down2d(16*ngf, 32*ngf, 3)
        self.down6 = down2d(32*ngf, 64*ngf, 3)
        self.down7 = down2d(64*ngf, 64*ngf, 3)
        self.up1   = up2d(64*ngf, 64*ngf)
        self.up2   = up2d(64*ngf, 32*ngf)
        self.up3   = up2d(32*ngf, 16*ngf)
        self.up4   = up2d(16*ngf, 8*ngf)
        self.up5   = up2d(8*ngf, 4*ngf)
        self.up6   = up2d(4*ngf, 2*ngf)
        self.up7   = up2d(2*ngf, ngf)
        self.conv3 = Conv2d(ngf, inChannels, 3, stride=1, padding=1)
        self.conv4 = Conv2d(2*inChannels, outChannels, 3, stride=1, padding=1)
        self.lRelu = LeakyReLU(negative_slope=0.1)
        self.sigmoid = Sigmoid()
        
    def forward(self, x):
        s0  = self.lRelu(self.conv1(x))
        s1 = self.lRelu(self.conv2(s0))
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        s5 = self.down4(s4)
        s6 = self.down5(s5)
        s7 = self.down6(s6)
        u0 = self.down7(s7)
        u1 = self.up1(u0, s7)
        u2 = self.up2(u1, s6)
        u3 = self.up3(u2, s5)
        u4 = self.up4(u3, s4)
        u5 = self.up5(u4, s3)
        u6 = self.up6(u5, s2)
        u7 = self.up7(u6, s1)
        y0 = self.lRelu(self.conv3(u7))
        y1 =self.sigmoid(self.conv4(cat((y0, x), 1)))
        return y1

class segmentationModel(Module):
    def __init__(self, nc=1, nGaborFilters=64, ngf=64, ncOut=2, supportSizes=[5,7,9,11]):
        super(segmentationModel, self).__init__()
        self.name = 'U_GEN'
        self.ngf = ngf
        self.supportSizes = supportSizes
        self.gaborFilters = ModuleDict({f'{supportSize}': Conv2d(nc, int(nGaborFilters/len(supportSizes)), supportSize, stride = 1, padding=int((supportSize-1)/2), padding_mode='reflect'  ) for supportSize in supportSizes})
        
        for param in self.gaborFilters.parameters():
            param.requires_grad = False
        self.setGaborfiltersValues()       
        
        self.unet = UNet2d(nGaborFilters, ncOut, ngf, 5)
        
    def setGaborfiltersValues(self, thetaRange = 180):
        thetas = linspace(0, thetaRange, int(self.ngf/len(self.supportSizes)))
        for supportSize in self.supportSizes:
            filters = gaborFilters(supportSize)
            for indextheta, theta in enumerate(thetas):
                self.gaborFilters[f'{supportSize}'].weight[indextheta][0] = parameter.Parameter(filters.getFilter(theta), requires_grad=False)

    def forward(self, x):
        c5  = self.gaborFilters['5'](x)
        c7  = self.gaborFilters['7'](x)
        c9  = self.gaborFilters['9'](x)
        c11 = self.gaborFilters['11'](x)
        y = cat((c5,c7,c9,c11),1)
        z = self.unet(y)
        return z

class gaborFilters():
    def __init__(self, supportSize):
        self.gridX, self.gridY = meshgrid(arange(-math.floor(supportSize/2),math.ceil(supportSize/2)), arange(-math.floor(supportSize/2),math.ceil(supportSize/2)))
        self.frequency = 1/8
        self.sigma = 3

    def getFilter(self, theta):
        Filter = cos(2*3.1415*self.frequency*(self.gridX*cos(theta) + self.gridY*sin(theta)))*exp(-(self.gridX*self.gridX+self.gridY*self.gridY)/(2*self.sigma*self.sigma))
        return Filter/linalg.norm(Filter)


class tilesClassifier(Module):
    def __init__(self, inChannels:int, outClasses:int, ngf:int, fs:int):
        super(tilesClassifier, self).__init__()
        # Initialize neural network blocks.
        self.conv1 = Conv2d(inChannels, ngf, fs, stride=1, padding=int((fs - 1) / 2))
        self.conv2 = Conv2d(ngf, ngf, fs, stride=1, padding=int((fs - 1) / 2))
        self.down1 = down2d(ngf, 2*ngf, 5)
        self.down2 = down2d(2*ngf, 4*ngf, 3)
        self.down3 = down2d(4*ngf, 8*ngf, 3)
        self.down4 = down2d(8*ngf, 16*ngf, 3)
        self.down5 = down2d(16*ngf, 32*ngf, 3)
        self.down6 = down2d(32*ngf, 64*ngf, 3)
        self.down7 = down2d(64*ngf, 64*ngf, 3)
        self.down8 = down2d(64*ngf, 64*ngf, 3)
        self.fc1 = Linear(256,128)
        self.fc2 = Linear(128,32)
        self.fc3 = Linear(32,outClasses)
        self.lRelu = LeakyReLU(negative_slope=0.1)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        s0  = self.lRelu(self.conv1(x))
        s1 = self.lRelu(self.conv2(s0))
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        s5 = self.down4(s4)
        s6 = self.down5(s5)
        s7 = self.down6(s6)
        s8 = self.down7(s7)
        u0 = self.down8(s8)
        y0 = flatten(u0, 1)
        y1 = self.lRelu(self.fc1(y0))
        y2 = self.lRelu(self.fc2(y1))
        z = self.fc3(y2)
        return z
