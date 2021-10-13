from torch import nn
import numpy as np
import torch
import os

from networks.Autoencoder import Encoder, Decoder
from dataLoader import data_load


######################
# Create the Model   #
######################
class Model(nn.Module):
    def __init__(self, c):
        super(Model, self).__init__()
        self.encoder = Encoder(c, 3, 1)
        self.decoder = Decoder(c, 3, 1)

    def forward(self, x):
        encoded_image = self.encoder(x)

        return encoded_image


######################
# decoding           #
######################
def encoding():

    # GPU
    dev = 'cuda:0'

    data_loader = data_load(path_input='sample-dataset/damage/dmg', path_output='sample-dataset/damage/dmg', batch_size=10, dataset_size=1000, shuffle=False)

    model = Model(32)
    model = model.double()
    model.load_state_dict(torch.load("autoencoder.pt"))
    model = nn.DataParallel(model, device_ids=[0])
    model.to(dev)
    model.eval()
    
    addr_output = 'sample-dataset/encoded-damage/'
    if not os.path.exists(addr_output):
        os.mkdir(addr_output)
    with torch.no_grad():
        num = 0
        for x_batch, _ in data_loader:
            x_batch = x_batch.to(dev)
            out = model(x_batch)
            output = out.cpu().detach().numpy()
            for i in range(len(output)):
                np.savetxt(addr_output+'dmg'+str(num)+'.txt', output[i].reshape(64,64), fmt='%.4e')
                num += 1


if __name__ == '__main__':
    encoding()