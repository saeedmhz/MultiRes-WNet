# MultiRes-WNet

This repository contains the pytorch implementation of MultiRes-WNet, a convolutional neural network architecture used for image-to-image mapping in mechanics-based metamodeling in ``Predicting Mechanically Driven Full Field Quantities of Interest with Deep Learning-Based Metamodels''.

![MultiRes-WNet](https://user-images.githubusercontent.com/54042195/127224632-7df3a99d-4408-42a7-a824-d97799ae0492.png)

# Links

Mechanical MNIST datasets collection: http://hdl.handle.net/2144/39371

Mechanical MNIST Crack Path Github: https://github.com/saeedmhz/Mechanical-MNIST-Crack-Path

Pytorch: https://pytorch.org/get-started/locally/

Manuscript: https://arxiv.org/abs/2108.03995

Trained Models: https://drive.google.com/drive/folders/1CJngW03f6tHqDoukvGUWNcaWF4QGKTn6?usp=sharing

## Running the model

Train a dicplacement predictor model:

    python ./displacement-prediction/train.py --lr="learning rate" --d="learning rate decay' --s="reduce learning rate after s epochs" --e="total epochs" --b="batch size" --c="first lavel channels" --m="model name" --n="network_name" --ch="continue training from a previous checkpoint"

Train a strain predictor model:

    python ./strain-prediction/train.py --lr="learning rate" --d="learning rate decay' --s="reduce learning rate after s epochs" --e="total epochs" --b="batch size" --c="first lavel channels" --m="model name" --ch="continue training from a previous checkpoint"

Train a crack path predictor model:

    7z x ./crack-path-prediction/sample-dataset/damage.7z
    7z x ./crack-path-prediction/sample-dataset/material.7z
    python ./crack-path-prediction/encoder.py
    python ./crack-path-prediction/train.py --lr="learning rate" --d="learning rate decay' --s="reduce learning rate after s epochs" --e="total epochs" --b="batch size" --c="first lavel channels" --m="model name" --ch="continue training from a previous checkpoint"

Predicting crack paths with a trained model:

    python ./crack-path-prediction/prediction.py --c="first lavel channels" --m="model name"
    python ./crack-path-prediction/decoder.py

### This repository contains code and sample data used to generte the results in the manuscript of ``Predicting Mechanically Driven Full Field Quantities of Interest with Deep Learning-Based Metamodels''.
