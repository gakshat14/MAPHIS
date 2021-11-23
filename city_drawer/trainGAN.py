from pathlib import PurePath, Path
from torchvision.transforms.transforms import ToTensor
from datasetsFunctions import TreePipeline, feature
import argparse
from pyprojroot import here
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms, utils
from models import GAN
from torch.nn import BCELoss
import torch
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasetPath', type=PurePath, required=False, default=here().joinpath('datasets'))
    parser.add_argument('--featureName', type=str, required=False, default='Trees')
    parser.add_argument('--batchSize', type=int, required=False, default=8)
    parser.add_argument('--numWorkers', type=int, required=False, default=0)
    parser.add_argument('--epochs', type=int, required=False, default=20000)
    parser.add_argument('--patchSize', type=int, required=False, default=16)
    parser.add_argument('--device', type=str, required=False, default='0')
    parser.add_argument('--savePath', type=Path, required=False, default=Path('saves'))
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device}')
    transformations = transforms.Compose([TreePipeline(128), ToTensor()])

    dataset = feature(args.datasetPath, args.featureName, True, transform=transformations)
    trainDataloader = DataLoader(dataset, batch_size=args.batchSize,
                                                shuffle=True, num_workers=args.numWorkers, drop_last=False)

    # Initialize BCELoss function
    criterion = BCELoss()
    # Establish convention for real and fake labels during training
    real_label = 0.9
    fake_label = 0.

    GANParameters = {"nfGenerator":16, "nfDetector":16}
    if not args.savePath.is_dir():
        args.savePath.mkdir(parents=True, exist_ok=True)
    with open(f'saves/{args.featureName}GAN.json', 'w') as saveFile:
        json.dump(GANParameters, saveFile)

    model = GAN(GANParameters)
    model.setOptimisers(1e-4)
    model.cuda(device)

    for epoch in range(args.epochs):
        for i, data in enumerate(trainDataloader): 
            data = data.cuda(device)   
            model.networks['discriminator'].zero_grad()
            # Format batch
            label = torch.full((args.batchSize,1,args.patchSize,args.patchSize), real_label, dtype=torch.float32, device=device)
            # Forward pass real batch through D

            #noise = torch.rand(args.batchSize, 1, 128, 128).cuda(device).float()
            output_patch = model.networks['discriminator'](data )
            # Calculate loss on all-real batch
            errD_real = criterion(output_patch, label)
            # Calculate gradients for D in backward pass
            errD_real.backward(retain_graph=True)
            Dx = output_patch.mean().item()
            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(args.batchSize, 100, 1, 1, device=device)
            # Generate fake image batch with G
            fake = model.networks['generator'](noise)

            label.fill_(fake_label)
            # Classify all fake batch with D
            output = model.networks['discriminator'](fake)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward(retain_graph=True)
            DGz1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            model.optimisers['discriminator'].step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            model.networks['generator'].zero_grad()
            #label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = model.networks['discriminator'](fake)
            # Calculate G's loss based on this output
            label = torch.full((2,1,args.patchSize,args.patchSize), real_label, dtype=torch.float32, device=device)
            topk_predictions_patch = torch.topk(output, 2, dim=0 )

            errG = criterion(topk_predictions_patch[0], label)
            #errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward(retain_graph=True)
            DGz2 = output.mean().item()
            # Update G
            model.optimisers['generator'].step()

            # Output training stats
        if epoch%1000 == 0:     
            plt.matshow(model.networks['discriminator'](fake)[0,0].detach().cpu())
            plt.show()                
            print(f'[{epoch}/{args.epochs}][{i}/{len(trainDataloader)}] \t lossD : {errD.item():.2f} \t lossG : {errG.item():.2f} \t D(x) : {Dx:.2f} \t D(G(z)) : {DGz1:.2f} / {DGz2:.2f}') 

        #torch.save(model.networks, args.savePath / f'{args.featureName}_GAN.pth')

if __name__ == '__main__':
    main()