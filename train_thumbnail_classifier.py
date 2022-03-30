import pathlib
from datasets.datasetsFunctions import Thumbnails, matchKeyToName
import argparse
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
import torch
from city_drawer.models import segmentationModel
import matplotlib.pyplot as plt

def main():
    parser =argparse.ArgumentParser(usage ='Argument Parser for tiling maps ')
    parser.add_argument('--datasetPath', type=str, required=False, default=r'C:\Users\hx21262\MAPHIS\datasets')
    parser.add_argument('--cityKey', type=str, required=False, default='36')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
    parser.add_argument('--epochs', type=int, default=10, help='epochs for training')
    parser.add_argument('--mapFileExtension', type=str, default='.jpg', required=False)
    parser.add_argument('--featureName', type=str, required=True)
    parser.add_argument('--fromCoordinates', type=bool, default=True, required=False)

    args = parser.parse_args()

    device = torch.device('cuda:0')

    datasetPath = pathlib.Path(args.datasetPath)

    cityName = matchKeyToName(datasetPath.joinpath(f'cityKey.json'), args.cityKey)
    allTilesPaths = list(datasetPath.joinpath(f'cities/{cityName}').glob(f'*/*/*{args.mapFileExtension}'))

    criterion = torch.nn.BCELoss()

    modelSegmentParameters= json.load(open(f'city_drawer/saves/{args.featureName}SegmentModelParameters.json'))
    modelSegment = segmentationModel(modelSegmentParameters)
    if Path(f'city_drawer/saves/{args.featureName}SegmentModelStateDict.pth').is_file():
        modelSegment.load_state_dict(torch.load(f'city_drawer/saves/{args.featureName}SegmentModelStateDict.pth'))
    modelSegment.cuda(device)
    modelSegment.eval()

    zeros =  torch.zeros((args.batchSize,1,512,512)).float().to(device)

    optimiser = torch.optim.Adam(modelSegment.parameters(), 1e-4)

    tilesDataset = Thumbnails(Path(args.datasetPath), cityName, tileName=allTilesPaths[0].stem, featureName= args.featureName)
    tileDataloader = DataLoader(tilesDataset, batch_size=args.batchSize, shuffle=True, num_workers=0, drop_last=True)

    for epoch in range(args.epochs):
        print(f'{epoch} / {args.epochs}')

        for i, data in enumerate(tileDataloader):
            background, mask, overlap_mask = data['background'].float().cuda(device), data['mask'].float().cuda(device), data['overlap_mask'].float().cuda(device)
            optimiser.zero_grad()

            segmented = modelSegment(background)

            loss = criterion(segmented, mask) + criterion(segmented*overlap_mask,zeros) 
            loss.backward()
            optimiser.step()

        print(loss.item())
        plt.matshow(np.concatenate((background[0,0].detach().cpu(), segmented[0,0].detach().cpu(), mask[0,0].detach().cpu(), overlap_mask[0,0].detach().cpu()), 1))
        plt.show()

        torch.save(modelSegment.state_dict(), pathlib.Path(f'city_drawer/saves/{args.featureName}SegmentModelStateDict.pth'))


if __name__=='__main__':
    main()

    
    

