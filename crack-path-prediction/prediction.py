"""

Options
----------
channels: int
    Number of channels in the first layer of the network
model_name: str
    Model name for saving and loading a trained model

Returns
-------
    
"""
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import argparse
import shutil
import os

from networks.MultiResUNet import MRUNet
from dataLoader import data_load

# randomness
sd = 987
np.random.seed(sd)
torch.manual_seed(sd)

#############################################
# Loading the model from a saved checkpoint #
#############################################
def load_model_ckp(checkpoint_fpath, model):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])

    return model

###########
# Options #
###########
def parseArgs():
    """ A list of options and hyperparameters can be passed through bash commands while running this script
    """
    parser = argparse.ArgumentParser(description='List of script options and network hyperparameters')
    parser.add_argument('-c', '--channels', metavar='', type=int, default=32, help='approximate number of first level channels in the model - default value = 32')
    parser.add_argument('-mn', '--model_name', metavar='', type=str, default="model", help='save the trained model into or load a trained model from "model_name.pt" - defaul name = "model"')

    return parser.parse_args()


######################
# Creating the Model #
######################
class Model(nn.Module):
    def __init__(self, c):
        super(Model, self).__init__()
        self.U1 = MRUNet(c, 3, 1)
        self.U2 = MRUNet(c, 3, 1)


    def forward(self, x):
        out1 = self.U1(x)
        out2 = self.U2(out1)

        return [out1, out2]


######################
# Training the Model #
######################
def predicting(args):
    """runs a full training of the network over the loaded dataset and saves the model
    """
    channels = args.channels
    model_name = args.model_name

    # initialize pytorch data loaders:
    data_loader = data_load(path_input='sample-dataset/material/mat', path_output='sample-dataset/material/mat', batch_size=10, dataset_size=1000, shuffle=True)

    # GPU
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    model = Model(channels) # creating an instance of the MultiResWNet
    model = model.double()
    model = load_model_ckp('best_model/'+model_name+'.pt', model)
    model = nn.DataParallel(model, device_ids=[0])
    model.to(dev)

    addr_output = 'sample-dataset/predicted-encoded-damage/'
    if not os.path.exists(addr_output):
        os.mkdir(addr_output)
    with torch.no_grad():
        num = 0
        for x_batch, _ in data_loader:
            x_batch = x_batch.to(dev)
            out = model(x_batch)[-1]
            output = out.cpu().detach().numpy()
            for i in range(len(output)):
                np.savetxt(addr_output+'dmg'+str(num)+'.txt', output[i].reshape(64,64), fmt='%.4e')
                num += 1


if __name__ == "__main__":
    args = parseArgs()
    predicting(args)
