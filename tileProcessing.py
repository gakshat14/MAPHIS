from datasets.datasetsFunctions import Tiles, matchKeyToName
import argparse
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
from models import labelExtractor
import torch
from city_drawer.models import segmentationModel
from shapeExtraction import dilation, coloriseMap
import matplotlib.pyplot as plt

def main():
    parser =argparse.ArgumentParser(usage ='Argument Parser for tiling maps ')
    parser.add_argument('--datasetPath', type=str, required=False, default=r'C:\Users\hx21262\MAPHIS\datasets')
    parser.add_argument('--cityKey', type=str, required=False, default='36')
    parser.add_argument('--savedPathDetection', default='CRAFT/weights/craft_mlt_25k.pth', type=str, help='pretrained model for DETECTION')
    parser.add_argument('--savedPathRefiner', default='CRAFT/weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model for detection')
    parser.add_argument('--textThreshold', default=0.7, type=float, help='text confidence threshold')
    parser.add_argument('--lowText', default=0.4, type=float, help='text low-bound score')
    parser.add_argument('--linkThreshold', default=0.7, type=float, help='link confidence threshold')
    parser.add_argument('--canvas_size', default=512, type=int, help='image size for inference')
    parser.add_argument('--mag_ratio', default=1.0, type=float, help='image magnification ratio')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
    parser.add_argument('--mapFileExtension', type=str, default='.jpg', required=False)
    parser.add_argument('--fromCoordinates', type=bool, default=True, required=False)

    args = parser.parse_args()

    device = torch.device('cuda:0')
    labelExtractorModel = labelExtractor(args.savedPathDetection, args.savedPathRefiner, device, args.textThreshold, args.linkThreshold, args.lowText)

    treesSegmenterParameters = json.load(open(f'city_drawer/saves/treesSegmentModelParameters.json'))
    treesSegmenter = segmentationModel(treesSegmenterParameters)
    if Path(f'city_drawer/saves/treesSegmentModelStateDict.pth').is_file():
        treesSegmenter.load_state_dict(torch.load(f'city_drawer/saves/treesSegmentModelStateDict.pth'))
    else:
        raise FileNotFoundError ("There is no trained model")
    treesSegmenter.cuda(device)
    treesSegmenter.eval()

    stripesSegmenterParameters = json.load(open(f'city_drawer/saves/stripesSegmentModelParameters.json'))
    stripesSegmenter = segmentationModel(stripesSegmenterParameters)
    if Path(f'city_drawer/saves/stripesSegmentModelStateDict.pth').is_file():
        stripesSegmenter.load_state_dict(torch.load(f'city_drawer/saves/stripesSegmentModelStateDict.pth'))
    else:
        raise FileNotFoundError ("There is no trained model")
    stripesSegmenter.cuda(device)
    stripesSegmenter.eval()

    cityName = matchKeyToName(f'{args.datasetPath}/cityKey.json', args.cityKey)
    allTilesPaths = list(Path(f'{args.datasetPath}/cities/{cityName}').glob(f'*/*/0105033010241{args.mapFileExtension}'))
    
    labelSavePath = Path(f'datasets/labels/{cityName}')
    labelSavePath.mkdir(parents=True, exist_ok=True)

    for tilePath in allTilesPaths:
        print(f'Processing Tile {tilePath.stem}')
        tilesDataset = Tiles(Path(args.datasetPath), cityName, mapName=tilePath.stem, fromCoordinates=args.fromCoordinates)
        tileDataloader = DataLoader(tilesDataset, batch_size=args.batchSize, shuffle=True, num_workers=args.workers)
        westBoundary, northBoundary, xDiff, yDiff = tilesDataset.boundaries['westBoundary'], tilesDataset.boundaries['northBoundary'], tilesDataset.boundaries['xDiff'], tilesDataset.boundaries['yDiff']
        cleaned = np.zeros((tilesDataset.tilingParameters['height'], tilesDataset.tilingParameters['width']))
        labelDict = {'mapName':tilesDataset.mapName, 'labels':{}}
        nDetectedLabels = 0
        for i, data in enumerate(tileDataloader):
            tile, coords = data['tile'], data['coordDict']
            '''thumbnail = torch.cat([tile, tile, tile], dim = 1 ).cuda(device)
            bBoxes, blobs = labelExtractorModel(thumbnail)
            blobs = dilation(blobs[0,0].cpu().data.numpy(), 3)
            b = torch.from_numpy(blobs).unsqueeze(0).unsqueeze(0)
            clean_ = tile*(1-b) + b'''
            cleaned[coords['yLow']:coords['yHigh'], coords['xLow']:coords['xHigh']] += stripesSegmenter(tile.cuda(device))[0,0].cpu().data.numpy()
            
        '''    for bBox in bBoxes:
                minW = int(min(bBox, key=lambda x : x[0])[0])
                maxW = int(max(bBox, key=lambda x : x[0])[0])
                minH = int(min(bBox, key=lambda x : x[1])[1])
                maxH = int(max(bBox, key=lambda x : x[1])[1])
                W = maxW - minW
                H = maxH - minH
                x = westBoundary +(minW + coords['xLow'])*xDiff
                y = northBoundary+(minH + coords['yLow'])*yDiff
                labelDict['labels'][f'{nDetectedLabels}'] = {'x':x.item(), 'y':y.item(), 'xTile':coords['xLow'].item()+minW, 'yTile':coords['yLow'].item()+minH, 'H':H, 'W':W}
                nDetectedLabels +=1

        with open(labelSavePath / f'{tilesDataset.mapName}.json', 'w') as outfile:
            json.dump(labelDict, outfile)'''

        colorisedMap = coloriseMap(cleaned, trees=False)
        unpaddedColorisedMap = colorisedMap[tilesDataset.tilingParameters['paddingY']:tilesDataset.tilingParameters['paddingY']+7590, tilesDataset.tilingParameters['paddingX']:tilesDataset.tilingParameters['paddingX']+11400]
        np.save(f'datasets/coloredMaps/{cityName}/{tilesDataset.mapName}.npy', unpaddedColorisedMap)
        plt.matshow(unpaddedColorisedMap)
        plt.show()


if __name__=='__main__':
    main()

    
    

