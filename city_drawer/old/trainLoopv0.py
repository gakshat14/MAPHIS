import pathlib
import torch
import torch.nn as nn 
import torch.optim as optim
import torchvision

import datasets
import argparse
import models

import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description='Tree Generation')
    parser.add_argument('--batchSize', required=False, type=int, default = 2)
    parser.add_argument('--randomSeed', required=False, type=int, default = 753159)
    parser.add_argument('--savePath', required=False, type=str, default = 'datasets')
    parser.add_argument('--imageSize', required=False, type=int, default = 512)
    parser.add_argument('--epochs', required=False, type=int, default = 1)
    parser.add_argument('--numWorkers', required=False, type=int, default = 0)
    parser.add_argument('--process', required=True, type=str, default = 'segment')
    parser.add_argument('--feature', required=False, type=str, default = '')
    args = parser.parse_args()

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
        ])

    trainSet = datasets.Trees(filepath='./datasets', train=True, transform=transform)
    trainDataloader = torch.utils.data.DataLoader(trainSet, batch_size=args.batchSize,
                                            shuffle=True, num_workers=args.numWorkers, drop_last=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    modelSegment = models.unet(1,64,1)
    
    modelsDict = torch.nn.ModuleDict({'segment':modelSegment})

    optimizersDict = {}

    if args.process == 'clean':

        if pathlib.Path('saves/SegmentModelStateDict.pth').is_file():
            modelsDict['segment'].load_state_dict(torch.load('saves/SegmentModelStateDict.pth'))
        else:
            raise ValueError (f'No saved file')

        modelsDict['segment'].eval()

        modelClean = models.unet(1,64,1)
        
        modelsDict[f'{args.process}{args.feature}'] = modelClean

        modelsDict[f'{args.process}{args.feature}'].load_state_dict(torch.load('saves/SegmentModelStateDict.pth'))
        modelsDict[f'{args.process}{args.feature}'].to(device)
        modelsDict[f'{args.process}{args.feature}'].train()   

        optimizersDict[args.feature]  = optim.Adam(modelsDict[f'{args.process}{args.feature}'].parameters(), lr=0.001)
        
    elif args.process =='segment':
        modelsDict[args.process].train()
        optimizer = optim.Adam(modelsDict[args.process].parameters(), lr=0.001)

    else:
        raise ValueError ('Wrong process argument : only clean or segment')
        
    modelsDict['segment'].to(device)

    criterion = nn.BCELoss()

    for epoch in range(args.epochs):
        running_loss = 0.0
        for i, data in enumerate(trainDataloader):
            inputImage, treesMask, stripesMask = data[0].float().to(device), data[1].float().to(device), data[2].float().to(device)
            maskDict = {'trees':treesMask, 'stripes':stripesMask}
            if args.process == 'segment':
                optimizer.zero_grad()
            output =  modelsDict['segment'](inputImage)
            if args.process == 'clean':
                optimizersDict[args.feature].zero_grad()

                outputClean = modelsDict[f'{args.process}{args.feature}'](output)

                loss = criterion(outputClean, maskDict[args.feature])

                loss.backward()

                optimizersDict[args.feature].step()

            else:   
                loss = criterion(output, inputImage*(treesMask+stripesMask))

            if args.process == 'segment':
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            print(f'[{i}] / [{int(3600/args.batchSize)}] --> Running Loss = {running_loss:.4f} ;  Item loss = {loss.item():.4f}')

            if i%200==0:
                plt.matshow(output[0,0].detach().cpu())
                plt.title(f'Segmented Image')
                plt.show()
                if args.process == 'clean':
                    plt.matshow(outputClean[0,0].detach().cpu())
                    plt.title(f'Cleaned  {args.feature}')
                    plt.show()

                torch.save(modelsDict[f'{args.process}{args.feature}'].state_dict(), f'saves/{args.feature}{args.process.capitalize()}ModelStateDict.pth')

if __name__ == '__main__':
    main()