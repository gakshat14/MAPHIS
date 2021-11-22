from pathlib import PurePath
from torchvision.transforms.transforms import ToTensor
from datasetsFunctions import TreePipeline, feature
import argparse
from pyprojroot import here
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms, utils

parser = argparse.ArgumentParser()
parser.add_argument('--datasetPath', type=PurePath, required=False, default=here().joinpath('datasets'))
parser.add_argument('--featureName', type=str, required=False, default='Trees')
parser.add_argument('--batchSize', type=int, required=False, default=1)
parser.add_argument('--numWorkers', type=int, required=False, default=0)
args = parser.parse_args()

transformations = transforms.Compose([TreePipeline(128), ToTensor()])

dataset = feature(args.datasetPath, args.featureName, True, transform=transformations)
trainDataloader = DataLoader(dataset, batch_size=args.batchSize,
                                            shuffle=True, num_workers=args.numWorkers, drop_last=False)

for data in trainDataloader:
    plt.matshow(data[0,0])
    plt.show()