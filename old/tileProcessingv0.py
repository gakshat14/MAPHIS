from datasets.datasetsFunctions import Tiles, pad, Maps, matchKeyToName
import argparse
from torch.utils.data import DataLoader
import numpy as np
import glob
from pathlib import Path
import json
import matplotlib.pyplot as plt
from models import labelExtractor
import torch
from imutils import grab_contours
from city_drawer.models import segmentationModel
from shapeExtraction import dilation

def main():
    parser =argparse.ArgumentParser(usage ='Argument Parser for tiling maps ')
    parser.add_argument('--datasetPath', type=str, required=False, default='datasets/cities')
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

    args = parser.parse_args()

    cityName = matchKeyToName('resources/cityKey.json', args.cityKey)
    folderPath = glob.glob(f'{args.datasetPath}/{cityName}/*/*/')[0]
    tilesDataset = Tiles(folderPath)
    tileDataloader = DataLoader(tilesDataset, batch_size=args.batchSize, shuffle=True, num_workers=args.workers)

    device = torch.device('cuda:0')
    labelExtractorModel = labelExtractor(args.savedPathDetection, args.savedPathRefiner, device, args.textThreshold, args.linkThreshold, args.lowText)

    tilesSegmenterParameters = json.load(open(f'city_drawer/saves/SegmentModelStateDict.pth'))
    tilesSegmenter = segmentationModel(tilesSegmenterParameters)
    if Path(f'city_drawer/saves/SegmentModelStateDict.pth').is_file():
        tilesSegmenter.load_state_dict(torch.load(f'city_drawer/saves/SegmentModelStateDict.pth'))
    else:
        raise FileNotFoundError ("There is no trained model")
    tilesSegmenter.cuda(device)
    tilesSegmenter.eval()

    westBoundary, northBoundary, xDiff, yDiff = tilesDataset.tilingDict['westBoundary'], tilesDataset.tilingDict['northBoundary'], tilesDataset.tilingDict['xDiff'], tilesDataset.tilingDict['yDiff']

    labelSavePath = Path(f'datasets/labels/{cityName}')
    labelSavePath.mkdir(parents=True, exist_ok=True)

    tilingParametersDict = json.load(open(f'datasets/tilingParameters.json'))

    cleaned = np.zeros((7590+2*tilingParametersDict['paddingY'],11400+2*tilingParametersDict['paddingX']))
    labelDict = {'mapName':tilesDataset.mapName, 'labels':{}}
    nDetectedLabels = 0
    for i, data in enumerate(tileDataloader):
        tile, coords = data[0], data[1]
        thumbnail = torch.cat([tile, tile, tile], dim = 1 ).cuda(device)
        bBoxes, blobs = labelExtractorModel(thumbnail)
        blobs = dilation(blobs[0,0].cpu().data.numpy(), 3)
        b = torch.from_numpy(blobs).unsqueeze(0).unsqueeze(0)
        clean_ = tile*(1-b) + b
        cleaned[coords['yLow']:coords['yHigh'], coords['xLow']:coords['xHigh']] += tilesSegmenter(clean_.cuda(device))[0,0].cpu().data.numpy()
        '''
        thumbnail = torch.cat([tile, tile, tile], dim = 1 ).cuda(device)
        bBoxes, blobs = labelExtractorModel(thumbnail)
        blobs = dilation(blobs[0,0].cpu().data.numpy(), 3)
        
        cleaned[coords['yLow']:coords['yHigh'], coords['xLow']:coords['xHigh']] = tile*(1-blobs) + blobs
        
        for bBox in bBoxes:
            minW = int(min(bBox, key=lambda x : x[0])[0])
            maxW = int(max(bBox, key=lambda x : x[0])[0])
            minH = int(min(bBox, key=lambda x : x[1])[1])
            maxH = int(max(bBox, key=lambda x : x[1])[1])
            W = maxW - minW
            H = maxH - minH
            x = westBoundary +(minW + coords['xLow'])*xDiff
            y = northBoundary+(minH + coords['yLow'])*yDiff
            labelDict['labels'][f'{nDetectedLabels}'] = {'x':x, 'y':y, 'xTile':coords['xLow'].item()+minW, 'yTile':coords['yLow'].item()+minH, 'H':H, 'W':W}
            nDetectedLabels +=1
            
    np.save(cleanedMapSavePath / f'{tilesDataset.mapName}', cleaned)
    with open(labelSavePath / f'{tilesDataset.mapName}.json', 'w') as outfile:
        json.dump(labelDict, outfile)
    '''
    if not Path(f'datasets/segmented/{cityName}').is_dir():
        Path(f'datasets/segmented/{cityName}').mkdir(parents=True, exist_ok=True)
    np.save(f'datasets/segmented/{cityName}/{tilesDataset.mapName}',cleaned)

if __name__=='__main__':
    main()

    
    

