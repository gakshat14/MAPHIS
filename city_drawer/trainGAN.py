from pathlib import PurePath
from torchvision.transforms.transforms import ToTensor
from datasetsFunctions import TreePipeline, feature
import argparse
from torch.nn import Sequential

parser = argparse.ArgumentParser()
parser.add_argument('--datasetPath', type=PurePath, required=False)
parser.add_argument('--featureName', type=str, default=True)
args = parser.parse_args()

transforms = Sequential(TreePipeline(128), ToTensor())

dataset = feature(args.datasetPath, args.featureName, True)