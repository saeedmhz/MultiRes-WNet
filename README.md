# MultiRes-WNet

This repository contains the pytorch implementation of MultiRes-WNet, a convolutional neural network architecute used in ''Predicting Mechanically Driven Full Field Quantities of Interest with Deep Learning-Based Metamodels'' for image-to-image mapping in mechanics-based metamodeling.

![MultiRes-WNet](https://user-images.githubusercontent.com/54042195/127224448-7d267e8f-0d6b-4b08-9335-87b72172cd2e.png)


# Links

Mechanical MNIST datasets collection: http://hdl.handle.net/2144/39371

Mechanical MNIST Crack Path Github: https://github.com/saeedmhz/Mechanical-MNIST-Crack-Path

Pytorch: https://pytorch.org/get-started/locally/

Manuscript: 

# This repository contains the following:

## The code used to create the metamodels in the paper

*Network/BuildingBlocks.py* -- contains building blocks used to create MultiRes-WNet and the Autoencoder used in the paper

*Network/MultiResUNet.py* -- contains the model for MultiRes-UNet using BuildingBlocks

*Network/Autoencoder.py* -- contains the model for Autoencoder using BuildingBlocks

*dataLoader.py* -- a pytorch function to load data from a folder

*main.py* -- the script used to train MultiResWNet on the training data and evaluate it on the test data
