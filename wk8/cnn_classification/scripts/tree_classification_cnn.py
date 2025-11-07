#  -------------------------------------------------------------------------------------------------
#   Copyright (c) 2016-2025.  SupportVectors AI Lab
#   This code is part of the training material and, therefore, part of the intellectual property.
#   It may not be reused or shared without the explicit, written permission of SupportVectors.
#
#   Use is limited to the duration and purpose of the training at SupportVectors.
#
#   Author: SupportVectors AI Training Team
#  -------------------------------------------------------------------------------------------------

from svlearn_cnn.trees.tree_dataset import TreeDataset
from svlearn_cnn.trees.preprocess import Preprocessor
from svlearn_cnn import config
from svlearn_cnn.train.simple_trainer import train_simple_network
from sklearn.metrics import accuracy_score
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torchvision.transforms import v2
import pandas as pd
num_classes = 2

model = nn.Sequential(
        # ----------------------------------------------------------------------------------------------------------------------------

        # Convolution Block 1
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, padding=0),     # ( B , 3 , 224 , 224 ) ->  ( B , 6 , 220 , 220 )
            nn.BatchNorm2d(num_features=6),                                     
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2),                                  # ( B , 6 , 220 , 220 ) ->  ( B , 6 , 110 , 110 )

        # ----------------------------------------------------------------------------------------------------------------------------
        # Convolution Block 2
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=0),    # ( B , 6 , 110 , 110 ) ->  ( B , 16 , 106 , 106 )
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                                   # ( B , 16 , 106 , 106 ) ->  ( B , 16 , 53 , 53 )

        # ----------------------------------------------------------------------------------------------------------------------------
        # Convolution Block 3
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4),              # ( B , 16 , 53 , 53 ) ->  ( B , 32 , 50 , 50 )                           
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                                   # ( B , 32 , 50 , 50 )   ->  ( B , 32 , 25 , 25 ) 

        # ----------------------------------------------------------------------------------------------------------------------------
            nn.Flatten(), # Change from 2D image to 1D tensor to be able to pass inputs to linear layer
        # ----------------------------------------------------------------------------------------------------------------------------
    
        # Linear Block 1
            nn.Linear(in_features=32 * 25 * 25, out_features=180),
            nn.ReLU(),

        # ----------------------------------------------------------------------------------------------------------------------------
        # Linear block 2
            nn.Linear(in_features=180, out_features=84),
            nn.ReLU(),

        # ----------------------------------------------------------------------------------------------------------------------------
            nn.Linear(in_features=84, out_features=num_classes)
        # ----------------------------------------------------------------------------------------------------------------------------
        )

if __name__ == "__main__":
    data_dir = config['tree-classification']['data']
    results_dir = config['tree-classification']['results']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps") if torch.backends.mps.is_available() else device
    
    preprocessor = Preprocessor()
    train_df, val_df, label_encoder = preprocessor.preprocess(data_dir)
    # 
    train_transform = v2.Compose([
        v2.ToImage(), 
        v2.RandomResizedCrop(224 , scale = (0.5, 1)), # Randomly crop and resize to 224x224
        v2.RandomHorizontalFlip(p=0.5),       # Randomly flip the image horizontally with a 50% chance
        v2.ColorJitter(brightness=0.4 , contrast=0.4, saturation=0.4), # randomly change the brightness , contrast and saturation of images
        v2.ToDtype(torch.float32, scale=True), # ensure te tensor is of float datatype
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # normalize tensor 
        
    ])

    test_transform = v2.Compose([
        v2.ToImage(), 
        v2.Resize(size=(224 , 224)),  # resize all images to a standard size suitable for the cnn model
        v2.ToDtype(torch.float32, scale=True), # ensure te tensor is of float datatype
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # normalize tensor 
    ])

    train_dataset = TreeDataset(train_df, transform=train_transform)
    val_dataset = TreeDataset(val_df, transform=test_transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False)   

    optimizer = AdamW(model.parameters(), lr = 0.001)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.5)

    results = train_simple_network(
                        model=model,
                        optimizer=optimizer,
                        lr_scheduler=scheduler,
                        loss_func=nn.CrossEntropyLoss(),
                        train_loader=train_loader,
                        test_loader=val_loader,
                        device=device,
                        epochs=10,
                        score_funcs={'accuracy': accuracy_score},
                        classify=True,
                        checkpoint_file=f"{results_dir}/cnn-model-trial-2.pt")

    pd.DataFrame(results).to_csv(f"{results_dir}/cnn-model-trial-2.csv", index=False)