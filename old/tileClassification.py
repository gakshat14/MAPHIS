from datasets.datasetsFunctions import Tiles, matchKeyToName
import argparse
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from models import tilesClassifier
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

import torch

import matplotlib.pyplot as plt

def main():
    parser =argparse.ArgumentParser(usage ='Argument Parser for tiling maps ')
    parser.add_argument('--datasetPath', type=str, required=False, default='datasets')
    parser.add_argument('--cityKey', type=str, required=False, default='36')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batchSize', type=int, default=8, help='input batch size')
    parser.add_argument('--fromCoordinates', type=bool, default=True, required=False)

    args = parser.parse_args()

    cityName = matchKeyToName(f'{args.datasetPath}/cityKey.json', args.cityKey)
    tilesDataset = Tiles(Path(args.datasetPath), cityName, fromCoordinates=args.fromCoordinates, colored=True, mapfileFormat='.npy')
    tileDataloader = DataLoader(tilesDataset, batch_size=args.batchSize, shuffle=True, num_workers=args.workers)

    device = torch.device('cuda:0')
    model = tilesClassifier(3,5,1,5)
    model.cuda(device)
    epochs = 10

    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.0001)

    for epoch in range(epochs):
        for i, data in enumerate(tileDataloader):
            tile, coords, label = data[0].cuda(device), data[1], data[2].float().cuda(device)
            optimizer.zero_grad()
            outputs = model(tile)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            if i%30 == 0:
                print(loss.detach().cpu())

if __name__=='__main__':
    main()

    
    

