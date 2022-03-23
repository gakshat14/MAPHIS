from pathlib import Path, PurePath
import torch
import torch.nn as nn 
import torch.optim as optim
import torchvision
import argparse
import models
import matplotlib.pyplot as plt
import json
from pyprojroot import here
import numpy as np
from PIL import Image
import pandas as pd
import math
from generateMaps import makeBatch
import tensorboard

from datasetsFunctions import syntheticCity

FEATURENAMES = [ 'trees', 'buildings', 'labels' ]

def loadBackground(mapName='0105033010241', cityName='Luton'):
    return np.where(np.array(Image.open( here() / f'datasets/cities/{cityName}/500/tp_1/{mapName}.jpg').convert('L'), np.float32) <100, 0, 1)

def loadPatterns():
    sq2 = math.sqrt(2)
    sizes = [32,64,128,256,512]
    patternsDict = {}
    for featureName in FEATURENAMES:        
        patternsDict[featureName] = {}
        fullDf = pd.DataFrame(json.load(open(here() / f'datasets/layers/{featureName}/Luton/0105033010241.json'))[f'{featureName}']).transpose() 
        for size in sizes:
            sLow = size/2
            sHigh = size/sq2
            df1 = fullDf.query('@sLow<H<@sHigh & W<H ')
            df2 = fullDf.query('@sLow<W<@sHigh & H<W ')
            patternsDict[featureName][f'{size}'] = df1.append(df2)
    return patternsDict

def main():
    parser = argparse.ArgumentParser(description='Tree Generation')
    parser.add_argument('--batchSize', required=False, type=int, default = 4)
    parser.add_argument('--randomSeed', required=False, type=int, default = 753159)
    parser.add_argument('--datasetPath', required=False, type=PurePath, default = here().joinpath('datasets'))
    parser.add_argument('--imageSize', required=False, type=int, default = 512)
    parser.add_argument('--epochs', required=False, type=int, default = 3)
    parser.add_argument('--numWorkers', required=False, type=int, default = 0)
    parser.add_argument('--featureName', required=False, type=str, default = 'buildings')
    parser.add_argument('--process', required=False, type=str, default = 'segment')
    args = parser.parse_args()

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
        ])

    datasetPath = Path(args.datasetPath)
    background = loadBackground()
    patternsDict = loadPatterns()

    testBatch, testBatchMask = makeBatch(args.batchSize, patternsDict,  background)
    allMasks = sum(testBatchMask.values())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    modelSegmentParameters = {"ncIn":1, "nGaborFilters":64, "ngf":4, "ncOut":1, "supportSizes":[5,7,9,11]}
    modelSegment = models.segmentationModel(modelSegmentParameters)
    if not Path('saves').is_dir():
        Path('saves').mkdir(parents=True, exist_ok=True)
    with open(f'saves/{args.featureName}SegmentModelParameters.json', 'w') as saveFile:
        json.dump(modelSegmentParameters, saveFile)
    
    if Path(f'saves/{args.featureName}SegmentModelStateDict.pth').is_file():
        print(f"Loading from {Path(f'saves/{args.featureName}SegmentModelStateDict.pth')}")
        modelSegment.load_state_dict(torch.load(f'saves/{args.featureName}SegmentModelStateDict.pth'))
    
    modelSegment.to(device)

    optimizer = optim.Adam(modelSegment.unet.parameters(), lr=0.0001)

    criterion = nn.BCELoss()

    nSamples = 1000

    zeros = torch.zeros((args.batchSize,1,512,512), device=device)

    for epoch in range(args.epochs):
        running_loss = 0.0

        for i in range(nSamples):
            trainBatch, trainBatchMask = makeBatch(args.batchSize, patternsDict,  background)
            trainBatch = torch.from_numpy(trainBatch).float().to(device)
            if args.featureName =='':
                masks, masks_overlap = torch.from_numpy(sum(trainBatchMask.values())).float().to(device), zeros
            elif args.featureName == 'labels':
                masks, masks_overlap = torch.from_numpy(trainBatchMask[f'{args.featureName}']).float().to(device), torch.clamp(torch.from_numpy(trainBatchMask[f'trees']+trainBatchMask[f'buildings']),0,1).float().to(device)
            elif args.featureName == 'trees':
                masks, masks_overlap = torch.from_numpy(trainBatchMask[f'{args.featureName}']).float().to(device), torch.clamp(torch.from_numpy(trainBatchMask[f'labels']+trainBatchMask[f'buildings']),0,1).float().to(device)
            elif args.featureName == 'buildings':
                masks, masks_overlap = torch.from_numpy(trainBatchMask[f'{args.featureName}']).float().to(device), torch.clamp(torch.from_numpy(trainBatchMask[f'trees']+trainBatchMask[f'labels']),0,1).float().to(device)
            else:
                raise ValueError ("wrong FeatureName")
            optimizer.zero_grad()
            output = modelSegment(trainBatch)
            
            loss = criterion(output, masks) + criterion(output*masks_overlap, zeros)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print(f'[{i}] / [{int(nSamples)}] --> Item loss = {loss.item():.4f}')

            """if i%150==0:
                plt.imshow(output[0,0].detach().cpu())
                plt.title(f'Segmented Image')
                plt.show()
                plt.imshow((allMasks)[0,0].detach().cpu())
                plt.title(f'Mask')
                plt.show()"""

        torch.save(modelSegment.state_dict(), f'saves/{args.featureName}{args.process.capitalize()}ModelStateDict.pth')

if __name__ == '__main__':
    main()