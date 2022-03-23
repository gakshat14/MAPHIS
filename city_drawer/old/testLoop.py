import torch
from datasetsFunctions import Maps, pad
import argparse
from models import segmentationModel
import matplotlib.pyplot as plt
import numpy as np 
from pathlib import Path, PurePath
import json
from pyprojroot import here

def main():    
    parser = argparse.ArgumentParser(description='Tree Generation')
    parser.add_argument('--batchSize', required=False, type=int, default = 1)
    parser.add_argument('--datasetPath', required=False, type=PurePath, default = here().joinpath('datasets'))
    parser.add_argument('--fileFormat', required=False, type=str, default = '.jpg')
    parser.add_argument('--feature', required=False, type=str, default = '')
    parser.add_argument('--cityName', required=False, type=str, default = 'Luton')
    parser.add_argument('--numWorkers', required=False, type=int, default = '0')
    args = parser.parse_args()

    datasetPath = Path(args.datasetPath)
    Path('segmentedMaps').mkdir(parents=True, exist_ok=True)
    
    transform = pad()
    trainSet = Maps(datasetPath, args.cityName, fileFormat=args.fileFormat, transform=transform)
    trainDataloader = torch.utils.data.DataLoader(trainSet, batch_size=args.batchSize,
                                            shuffle=True, num_workers=args.numWorkers)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tilesSegmenterParameters = json.load(open(f'saves/SegmentModelParameters.json'))
    tilesSegmenter = segmentationModel(tilesSegmenterParameters)
    tilesSegmenter.load_state_dict(torch.load(f'saves/SegmentModelStateDict.pth'))
    tilesSegmenter.to(device)

    for i, data in enumerate(trainDataloader):
        print(f'Map {i} / {len(trainDataloader)}')
        map = data['map'][0].float().to(device)
        segmented = np.zeros((7680,11776))
        kS = 512
        nRows = 15
        nCCols= 23
        with torch.no_grad():
            for i in range(nRows):
                print(f'Row {i} / {nRows}')
                for j in range(nCCols):
                    thumbnail = map[:,:,kS*i:kS*(i+1), kS*j:kS*(j+1)]

                    segmented[kS*i:kS*(i+1), kS*j:kS*(j+1)] = tilesSegmenter(thumbnail).detach().cpu()

                if i%10==0:
                    plt.imshow(segmented)
                    plt.title('segmented')
                    plt.show()

        np.save(f'segmentedMaps/{data["mapName"][0].split(".")[0]}_segmented.npy', segmented)

if __name__ == '__main__':
    main()
    
