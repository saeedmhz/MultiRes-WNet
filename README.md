# MultiRes-WNet

This repository contains the pytorch implementation of MultiRes-WNet, a convolutional neural network architecture used for image-to-image mapping in mechanics-based metamodeling in ''Predicting Mechanically Driven Full Field Quantities of Interest with Deep Learning-Based Metamodels''.

![MultiRes-WNet](https://user-images.githubusercontent.com/54042195/127224632-7df3a99d-4408-42a7-a824-d97799ae0492.png)

# Links

Mechanical MNIST datasets collection: http://hdl.handle.net/2144/39371

Mechanical MNIST Crack Path Github: https://github.com/saeedmhz/Mechanical-MNIST-Crack-Path

Pytorch: https://pytorch.org/get-started/locally/

Manuscript: 

## Running the model

Train a model:

    python main.py --lr=learning rate --d=learning rate decay --s=number of epochs before learning rate reduction \
        --e=total epochs --b=batch size --c=first layer channels --m=save model? --mn=model name --tr=True -ev=False
        
Test a model:

    python main.py --c=first layer channels --mn=model name --tr=False -ev=True
 
# This repository contains the following:

## The code used to create the metamodels in the paper

*Network/BuildingBlocks.py* -- contains building blocks used to create MultiRes-WNet and the Autoencoder used in the paper

*Network/MultiResUNet.py* -- contains the model for MultiRes-UNet using BuildingBlocks

*Network/Autoencoder.py* -- contains the model for Autoencoder using BuildingBlocks

*dataLoader.py* -- a pytorch function to load data from a folder

*main.py* -- the script used to train MultiResWNet on the training data and evaluate it on the test data

*trained-models* -- the folder contains best performing models trained on the Mechanical Fashion MNIST
