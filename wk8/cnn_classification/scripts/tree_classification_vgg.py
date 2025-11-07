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
from torchvision.transforms import v2
import pandas as pd
import torchvision.models as models

if __name__ == "__main__":

    data_dir = config['tree-classification']['data']
    results_dir = config['tree-classification']['results']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps") if torch.backends.mps.is_available() else device
    
    preprocessor = Preprocessor()
    train_df, val_df, label_encoder = preprocessor.preprocess(data_dir)
    
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

    # Load the VGG16 model
    vgg_model = models.vgg16(pretrained=True)

    # Freeze the feature extraction layers
    for param in vgg_model.parameters():
        param.requires_grad = False

    # Modify the classifier for 2 classes
    vgg_model.classifier[6] = nn.Linear(4096, 2) 

    optimizer = torch.optim.Adam(vgg_model.classifier[6].parameters(), lr=0.001)

    result = train_simple_network(
                            model=vgg_model,
                            optimizer=optimizer,
                            loss_func=nn.CrossEntropyLoss(),
                            train_loader=train_loader,
                            test_loader=val_loader,
                            epochs=10,
                            score_funcs={'accuracy': accuracy_score},
                            classify=True,
                            device=device,
                            checkpoint_file=f"{results_dir}/vgg-model-01-2.pt")

    pd.DataFrame(result).to_csv(f"{results_dir}/vgg-model-01-2.csv", index=False)