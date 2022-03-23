from datasets.datasetsFunctions import Tiles, matchKeyToName
import argparse
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
from models import labelExtractor
import pathlib
import torch
from city_drawer.models import segmentationModel
from shapeExtraction import extractShapes
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
    parser.add_argument('--featureName', type=str, default='labels', required=False)
    parser.add_argument('--fromCoordinates', type=bool, default=True, required=False)

    args = parser.parse_args()

    device = torch.device('cuda:0')

    cityName = matchKeyToName(pathlib.Path(f'{args.datasetPath}/cityKey.json'), args.cityKey)
    allTilesPaths = list(Path(f'{args.datasetPath}/cities/{cityName}').glob(f'*/*/*{args.mapFileExtension}'))[1:2]

    #for featureName in ['trees', 'labels', 'buildings']:
    for featureName in ['buildings']:
        modelSegmentParameters= json.load(open(f'city_drawer/saves/{featureName}SegmentModelParameters.json'))
        modelSegment = segmentationModel(modelSegmentParameters)
        if Path(f'city_drawer/saves/{featureName}SegmentModelStateDict.pth').is_file():
            print('loading statedict')
            modelSegment.load_state_dict(torch.load(f'city_drawer/saves/{featureName}SegmentModelStateDict.pth'))
        modelSegment.cuda(device)
        modelSegment.eval()

        for tilePath in allTilesPaths:
            print(f'Processing Tile {tilePath.stem}')
            tilesDataset = Tiles(Path(args.datasetPath), cityName, mapName=tilePath.stem, fromCoordinates=args.fromCoordinates)
            tileDataloader = DataLoader(tilesDataset, batch_size=args.batchSize, shuffle=True, num_workers=args.workers)
            segmented = np.zeros((tilesDataset.tilingParameters['height'], tilesDataset.tilingParameters['width']))
            for i, data in enumerate(tileDataloader):
                tile, coords = data['tile'], data['coordDict']
                out = modelSegment(tile.float().cuda(device))
                segmented[coords['yLow']:coords['yHigh'], coords['xLow']:coords['xHigh']] += out[0,0].cpu().data.numpy()

            segmented = np.where(segmented>0.9,1,0)       
            plt.matshow(segmented)
            plt.show()
            np.save(f'datasets/layers/{featureName}/{cityName}/{tilePath.stem}_segmented.npy', np.uint8(segmented))

if __name__=='__main__':
    main()

    
    

