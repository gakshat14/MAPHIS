from torch.nn import Module, ModuleDict, Conv2d, ConvTranspose2d, Sigmoid, LeakyReLU, parameter, BatchNorm2d, Dropout, ReLU, init
from torch import cat, linspace, meshgrid, arange, cos, sin, exp, linalg, Tensor, device, load
from torch.optim import Adam
import numpy as np
import math

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
        
    def forward(self, x :Tensor):
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
    def __init__(self, parametersDict:dict):
        super(segmentationModel, self).__init__()
        self.name = 'U_GEN'
        ## Assert that all parameters are here:
        for paramKwd in ['ngf', 'ncIn', 'ncOut', 'nGaborFilters', 'supportSizes']:
            if not parametersDict[paramKwd]:raise KeyError (f'{paramKwd} is missing')
        self.ngf = parametersDict['ngf']
        self.supportSizes = parametersDict['supportSizes']
        self.gaborFilters = ModuleDict({f'{supportSize}': Conv2d(parametersDict['ncIn'], int(parametersDict['nGaborFilters']/len(self.supportSizes)), supportSize, stride = 1, padding=int((supportSize-1)/2), padding_mode='reflect'  ) for supportSize in self.supportSizes})
        
        for param in self.gaborFilters.parameters():
            param.requires_grad = False
        self.setGaborfiltersValues()       
        
        self.unet = UNet2d(parametersDict['nGaborFilters'], parametersDict['ncOut'], self.ngf, 5)
        
    def setGaborfiltersValues(self, thetaRange = 180):
        thetas = linspace(0, thetaRange, int(self.ngf/len(self.supportSizes)))
        for supportSize in self.supportSizes:
            filters = gaborFilters(supportSize)
            for indextheta, theta in enumerate(thetas):
                self.gaborFilters[f'{supportSize}'].weight[indextheta][0] = parameter.Parameter(filters.getFilter(theta), requires_grad=False)

    def forward(self, x:Tensor):
        c5  = self.gaborFilters['5'](x)
        c7  = self.gaborFilters['7'](x)
        c9  = self.gaborFilters['9'](x)
        c11 = self.gaborFilters['11'](x)
        y = cat((c5,c7,c9,c11),1)
        z = self.unet(y)
        return z

class gaborFilters():
    def __init__(self, supportSize:int):
        self.gridX, self.gridY = meshgrid(arange(-math.floor(supportSize/2),math.ceil(supportSize/2)), arange(-math.floor(supportSize/2),math.ceil(supportSize/2)))
        self.frequency = 1/8
        self.sigma = 3

    def getFilter(self, theta:int) -> np.float32:
        Filter = cos(2*3.1415*self.frequency*(self.gridX*cos(theta) + self.gridY*sin(theta)))*exp(-(self.gridX*self.gridX+self.gridY*self.gridY)/(2*self.sigma*self.sigma))
        return Filter/linalg.norm(Filter)


class generator(Module):
    def __init__(self, ngf):
        super(generator, self).__init__()
        self.name = 'genPaper'
        self.upconv_0 = ConvTranspose2d(100, 8*ngf, 4, 1, 0, bias=False)        
        4
        self.upconv_1 = ConvTranspose2d(8*ngf, 8*ngf, 4, 2, 1, bias=False)      
        8  
        self.upconv_2 = ConvTranspose2d(8*ngf, 4*ngf, 4, 2, 1, bias=False)  
        16      
        self.upconv_3 = ConvTranspose2d(4*ngf, 2*ngf, 4, 2, 1, bias=False) 
        32      
        self.upconv_4 = ConvTranspose2d(2*ngf, ngf, 4, 2, 1, bias=False)   
        64     
        self.upconv_5 = ConvTranspose2d(ngf, 1, 4, 2, 1, bias=False) 
        128       
 
        self.bn_4 = BatchNorm2d(1*ngf)
        self.bn_3 = BatchNorm2d(2*ngf)
        self.bn_2 = BatchNorm2d(4*ngf)
        self.bn_1 = BatchNorm2d(8*ngf)
        self.bn_0 = BatchNorm2d(8*ngf)
        
        self.dropout = Dropout()
        self.sigmoid = Sigmoid()
        self.relu = ReLU()

    def forward(self, y):
        x0 = self.relu(self.bn_0(self.upconv_0(y)))
        x1 = self.relu(self.bn_1(self.upconv_1(x0)))
        x2 = self.relu(self.bn_2(self.upconv_2(x1)))
        x3 = self.relu(self.bn_3(self.upconv_3(x2)))
        x4 = self.relu(self.bn_4(self.upconv_4(x3)))
        x = self.sigmoid(self.upconv_5(x4))
        return x

class discriminator(Module):
    def __init__(self, ngf):
        super(discriminator, self).__init__()
        
        self.conv_0 = Conv2d(1, ngf, 4, 2, 1, bias=False)
        #64
        self.conv_1 = Conv2d(ngf, ngf*2, 4, 2, 1, bias=False)
        #32
        self.conv_2 = Conv2d(ngf*2, ngf*4, 4, 2, 1, bias=False)
        #16
        self.conv_3 = Conv2d(ngf*4, 1, 3, 1, 1, bias=False)
        
        self.bn_1 = BatchNorm2d(2*ngf)
        self.bn_2 = BatchNorm2d(4*ngf)
        
        self.dropout = Dropout()
        self.sigmoid = Sigmoid()
        self.lRelu = LeakyReLU(0.2)

        
    def forward(self, cad):
        out_0 = self.lRelu(self.conv_0(cad))
        #out_0: [bs,  1*ngf, 64, 64]        
        out_1 = self.lRelu(self.bn_1(self.conv_1(out_0)))
        #out_1: [bs,  2*ngf, 32, 32]
        out_2 = self.lRelu(self.bn_2(self.conv_2(out_1)))
        #out_2: [bs,  4*ngf, 16, 16]
        return self.sigmoid(self.conv_3(out_2))
    

def initialiseWeights(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0)
    elif classname.find('ConvTranspose2d') != -1:
        init.normal_(m.weight.data, 0.01, 0.02)
    elif classname.find('Conv2d') != -1:
        init.normal_(m.weight.data, 0.01, 0.02)

class GAN(Module):
    def __init__(self, parametersDict:dict):
        ## Assert that all parameters are here:
        for paramKwd in ['nfGenerator', 'nfDetector']:
            if not parametersDict[paramKwd]:raise KeyError (f'{paramKwd} is missing')
        super(GAN, self).__init__()
        self.device = device
        self.networks =  ModuleDict()  
        self.networks['generator'] = generator(parametersDict['nfGenerator'])
        self.networks['discriminator'] = discriminator(parametersDict['nfDetector'])    
        
        if self.networks is not None:
            for network in self.networks.values():
                initialiseWeights(network)
               
    def load(self, loadPath):
        self.networks.load_state_dict(load(loadPath).state_dict())

    def setOptimisers(self, lr:float):
        self.optimisers = {}
        for key, value in self.networks.items():
            self.optimisers[key] = Adam(value.parameters(), lr=lr)

