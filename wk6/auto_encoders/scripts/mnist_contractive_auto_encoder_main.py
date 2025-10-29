#  -------------------------------------------------------------------------------------------------
#   Copyright (c) 2024.  SupportVectors AI Lab
#   This code is part of the training material, and therefore part of the intellectual property.
#   It may not be reused or shared without the explicit, written permission of SupportVectors.
#
#   Use is limited to the duration and purpose of the training at SupportVectors.
#
#   Author: SupportVectors AI Training Team
#
import os
import torch
from torch.utils.data import DataLoader
from svlearn_autoencoders import config

from svlearn_autoencoders.auto_encoders.vanilla_auto_encoder_mnist import AutoencoderMnist
from svlearn_autoencoders.auto_encoders.auto_encoder_single_channel_util import train_autoencoder

import torchvision.datasets as datasets

if __name__ == "__main__":
    mnist_data_path = config['mnist-classification']['data']
    model_path = config['mnist-classification']['results']
    
    mnist_trainset = datasets.MNIST(root=mnist_data_path, train=True, download=True, transform=None)

    train_dataset = mnist_trainset.data[:-10000].reshape(-1, 1, 28, 28) / 255.
    eval_dataset = mnist_trainset.data[-10000:].reshape(-1, 1, 28, 28) / 255.
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(eval_dataset, batch_size=128, shuffle=False)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "mps" if torch.backends.mps.is_available() else device
    model = AutoencoderMnist().to(device)
    
    checkpoint_file = f'{model_path}/MNIST/mnist_contractive_autoencoder.pt'
    
    starting_learning_rate = 1e-3
    
    if os.path.isfile(checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        results = checkpoint['results']
        if 'lr' in results:
            starting_learning_rate = results['lr'][-1]
            
    train_autoencoder(model=model, 
                      train_loader=train_loader, 
                      val_loader=val_loader, 
                      num_epochs=50, 
                      device=device,  
                      checkpoint_file=checkpoint_file,
                      learning_rate=starting_learning_rate,
                      mode='contractive')