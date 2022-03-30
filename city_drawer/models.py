import torch.nn as nn 
import torch
import numpy as np
import math

class down2d(nn.Module):
    def __init__(self, inChannels:int, outChannels:int, filterSize:int):
        """Down sampling unit of factor 2

        Args:
            inChannels (int): Number of input channels
            outChannels (int): Number of output channels
            filterSize (int): size of the filter of the conv layers that dont downsample, odd integer
        """
        super(down2d, self).__init__()
        self.pooling2d = nn.Conv2d(inChannels,  inChannels, 4, stride=2, padding=1)        
        self.conv1 = nn.Conv2d(inChannels,  outChannels, filterSize, stride=1, padding=int((filterSize - 1) / 2))        
        self.conv2 = nn.Conv2d(outChannels, outChannels, filterSize, stride=1, padding=int((filterSize - 1) / 2))
        self.lRelu = nn.LeakyReLU(negative_slope = 0.1)
           
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.lRelu(self.pooling2d(x))
        x = self.lRelu(self.conv1(x))
        x = self.lRelu(self.conv2(x))
        return x
    
class up2d(nn.Module):
    def __init__(self, inChannels:int, outChannels:int):
        """Up sampling unit of factor 2

        Args:
            inChannels (int): Number of input channels
            outChannels (int): Number of output channels
        """
        super(up2d, self).__init__()
        self.unpooling2d = nn.ConvTranspose2d(inChannels, inChannels, 4, stride = 2, padding = 1)
        self.conv1 = nn.Conv2d(inChannels,  outChannels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(2 * outChannels, outChannels, 3, stride=1, padding=1)
        self.lRelu = nn.LeakyReLU(negative_slope = 0.1)
           
    def forward(self, x:torch.Tensor, skpCn:torch.Tensor) -> torch.Tensor:
        x = self.lRelu(self.unpooling2d(x))
        x = self.lRelu(self.conv1(x))
        x = self.lRelu(self.conv2(torch.cat((x, skpCn), 1)))
        return x    

class UNet2d(nn.Module):
    def __init__(self, inChannels:int, outChannels:int, ngf:int, fs:int):
        super(UNet2d, self).__init__()
        # Initialize neural network blocks.
        self.conv1 = nn.Conv2d(inChannels, ngf, fs, stride=1, padding=int((fs - 1) / 2))
        self.conv2 = nn.Conv2d(ngf, ngf, fs, stride=1, padding=int((fs - 1) / 2))
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
        self.conv3 = nn.Conv2d(ngf, inChannels, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(inChannels, outChannels, 3, stride=1, padding=1)
        self.lRelu = nn.LeakyReLU(negative_slope=0.1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x :torch.Tensor):
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
        y1 =self.sigmoid(self.conv4(y0))
        return y1

class segmentationModel(nn.Module):
    def __init__(self, parametersDict:dict):
        super(segmentationModel, self).__init__()
        self.name = 'U_GEN'
        ## Assert that all parameters are here:
        for paramKwd in ['ngf', 'ncIn', 'ncOut', 'nGaborFilters', 'supportSizes']:
            if not parametersDict[paramKwd]:raise KeyError (f'{paramKwd} is missing')
        self.ngf = parametersDict['ngf']
        self.supportSizes = parametersDict['supportSizes']
        self.gaborFilters = nn.ModuleDict({f'{supportSize}': nn.Conv2d(parametersDict['ncIn'], int(parametersDict['nGaborFilters']/len(self.supportSizes)), supportSize, stride = 1, padding=int((supportSize-1)/2), padding_mode='reflect'  ) for supportSize in self.supportSizes})
        
        for param in self.gaborFilters.parameters():
            param.requires_grad = False
        self.setGaborfiltersValues()       
        
        self.unet = UNet2d(parametersDict['nGaborFilters'], parametersDict['ncOut'], self.ngf, 5)
        
    def setGaborfiltersValues(self, thetaRange = 180):
        """Set the gabor filters values of the nn.module dictionnary 

        Args:
            thetaRange (int, optional): Angles at which to instantiate the filters. Defaults to 180.
        """
        thetas = torch.linspace(0, thetaRange, int(self.ngf/len(self.supportSizes)))
        for supportSize in self.supportSizes:
            filters = gaborFilters(supportSize)
            for indextheta, theta in enumerate(thetas):
                self.gaborFilters[f'{supportSize}'].weight[indextheta][0] = nn.parameter.Parameter(filters.getFilter(theta), requires_grad=False)

    def forward(self, x:torch.Tensor):
        c5  = self.gaborFilters['5'](x)
        c7  = self.gaborFilters['7'](x)
        c9  = self.gaborFilters['9'](x)
        c11 = self.gaborFilters['11'](x)
        y = torch.cat((c5,c7,c9,c11),1)
        z = self.unet(y)
        return z

class gaborFilters():
    def __init__(self, supportSize:int, frequency=1/8, sigma=3) -> None:
        """Initialise Gabor Filters for fixed frequency and support size and sigma

        Args:
            supportSize (int): Size of the gabor filter, odd integer
            frequency (_type_, optional): Frequency of the Gabor Filter. Defaults to 1/8.
            sigma (int, optional): Deviation of the Gabor Filter. Defaults to 3.
        """
        self.gridX, self.gridY = torch.meshgrid(torch.arange(-math.floor(supportSize/2),math.ceil(supportSize/2)), torch.arange(-math.floor(supportSize/2),math.ceil(supportSize/2)))
        self.frequency = frequency
        self.sigma = sigma

    def getFilter(self, theta:float) -> np.float32:
        """Returns a (self.gridX.shape, self.gridY.shape) sized matrix containing the Gabor filter values for the and Theta

        Args:
            theta (float): angle, in radians, at which the filter is returned

        Returns:
            np.float32: The Gabor filter values
        """
        Filter = torch.cos(2*3.1415*self.frequency*(self.gridX*torch.cos(theta) + self.gridY*torch.sin(theta)))*torch.exp(-(self.gridX*self.gridX+self.gridY*self.gridY)/(2*self.sigma*self.sigma))
        return Filter/torch.linalg.norm(Filter)
