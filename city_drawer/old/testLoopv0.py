import torch

import datasets
import argparse
import models

import matplotlib.pyplot as plt
import numpy as np
import cv2 
from pathlib import Path

import glob
from PIL import Image

def main(args):    
    if args.treatment == 'show':
        
        tileName = glob.glob(('maps/'+"*."+args.fileFormat))[args.imageIndex].split('\\')[-1].split('.')[0]
        originalImage = Image.open(f'maps/{tileName}.'+args.fileFormat)
        plt.matshow(originalImage)
        plt.show()
    
    elif args.treatment == 'process':
        Path('images').mkdir(parents=True, exist_ok=True)

        transform = datasets.unfold()
        trainSet = datasets.Maps(filepath=args.datasetPath, fileFormat=args.fileFormat, transform=transform)
        trainDataloader = torch.utils.data.DataLoader(trainSet, batch_size=args.batchSize,
                                                shuffle=True, num_workers=args.numWorkers)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        modelNames = ['Segment', 'treesClean', 'stripesClean']
        modelsDict = torch.nn.ModuleDict({})

        for modelName in modelNames:
            model = models.unet(1,64,1)
            model.load_state_dict(torch.load(f'saves/{modelName}ModelStateDict.pth'))
            model.to(device)
            model.eval()
            modelsDict[modelName] = model

        for i, data in enumerate(trainDataloader):
            print(f'Map {i} / {len(trainDataloader)}')
            map = data['map'].float().to(device)
            trees = np.zeros((7680,11520))
            stripes = np.zeros((7680,11520))
            mapWithoutTrees   = 1-data['map'][0,0].detach().numpy()
            mapWithoutStripes = 1-data['map'][0,0].detach().numpy()

            kS = 256
            nRows = 30
            with torch.no_grad():
                for i in range(nRows):
                    print(f'Row {i} / {nRows}')
                    for j in range(45):
                        thumbnail = map[:,:,kS*i:kS*(i+1), kS*j:kS*(j+1)]

                        segmented = modelsDict['Segment'](thumbnail)

                        cleanedTrees = modelsDict['treesClean'](segmented)[0,0].detach().cpu().numpy()
                        cleanedTrees = np.where(cleanedTrees > 0.5, 1, 0)
                        trees[kS*i:kS*(i+1), kS*j:kS*(j+1)] = cleanedTrees
                        mapWithoutTrees[kS*i:kS*(i+1), kS*j:kS*(j+1)] *=(1-cleanedTrees)

                        cleanedStripes = modelsDict['stripesClean'](segmented)[0,0].detach().cpu().numpy()
                        cleanedStripes = np.where(cleanedStripes > 0.5, 1, 0)
                        stripes[kS*i:kS*(i+1), kS*j:kS*(j+1)] = cleanedStripes
                        mapWithoutStripes[kS*i:kS*(i+1), kS*j:kS*(j+1)] *=(1-cleanedStripes)

                    if i%10==0:
                        plt.matshow(trees)
                        plt.title('trees')
                        plt.show()
                        plt.matshow(stripes)
                        plt.title('stripes')
                        plt.show()

            np.save(f'images/{data["mapName"][0].split(".")[0]}_trees.npy', trees)
            np.save(f'images/{data["mapName"][0].split(".")[0]}_withoutTrees.npy', mapWithoutTrees)
            np.save(f'images/{data["mapName"][0].split(".")[0]}_stripes.npy', stripes)
            np.save(f'images/{data["mapName"][0].split(".")[0]}_withoutStripes.npy', mapWithoutStripes)
    else:
        raise NotImplementedError ("Can only save or process the maps")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tree Generation')
    parser.add_argument('--batchSize', required=False, type=int, default = 1)
    parser.add_argument('--randomSeed', required=False, type=int, default = 753159)
    parser.add_argument('--numWorkers', required=False, type=int, default = 0)
    parser.add_argument('--datasetPath', required=False, type=str, default = 'maps/')
    parser.add_argument('--treatment', required=False, type=str, default = 'show')
    parser.add_argument('--fileFormat', required=False, type=str, default = 'jpg')
    parser.add_argument('--imageIndex', required=False, type=int, default = 0)
    args = parser.parse_args()

    main(args)

