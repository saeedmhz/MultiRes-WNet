""" Load the dataset, train or evaluate the MultiResWNet model on the dataset

Options
----------
learning_rate : float
    The initial learning rate of the optimization
decay_rate: float
    The decay rate of the learning rate
decay_step: int
    After "decay_step" epochs the scheduler reduces the learning rate by decay_rate
num_epoch: int
    Total number of training epochs
batch_size: int
    Training batch size
channels: int
    Number of channels in the first layer of the network
model_name: str
    Model name for saving and loading a trained model

Returns
-------
    during the training call the trained model is saved
    during the evaluation call the predicted output is saved
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

#####################################
# Saving best model and checkpoints # 
#####################################
def save_ckp(state, is_best, checkpoint_dir, best_model_dir):
    torch.save(state, checkpoint_dir)
    if is_best:
        shutil.copyfile(checkpoint_dir, best_model_dir)


#############################################
# Loading the model from a saved checkpoint #
#############################################
def load_model_ckp(checkpoint_fpath, model):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])

    return model, checkpoint['epoch']


#################################################
# Loading the optimizer from a saved checkpoint #
#################################################
def load_optim_ckp(checkpoint_fpath, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    optimizer.load_state_dict(checkpoint['optimizer'])

    return optimizer


###########
# Options #
###########
def parseArgs():
    """ A list of options and hyperparameters can be passed through bash commands while running this script
    """
    parser = argparse.ArgumentParser(description='List of script options and network hyperparameters')
    parser.add_argument('-lr', '--learning_rate', metavar='', type=float, default=1e-3, help='learning rate - default value = 1e-3')
    parser.add_argument('-d', '--decay_rate', metavar='', type=float, default=0.1, help='decay rate - default value = 0.1')
    parser.add_argument('-s', '--decay_step', metavar='', type=int, default=10, help='decay will be applied to the learning rate after s epochs - default value = 30')
    parser.add_argument('-e', '--num_epoch', metavar='', type=int, default=30, help='number of epochs - default value = 60')
    parser.add_argument('-b', '--batch_size', metavar='', type=int, default=16, help='number of data points in each mini batch - default value = 128')
    parser.add_argument('-c', '--channels', metavar='', type=int, default=32, help='approximate number of first level channels in the model - default value = 32')
    parser.add_argument('-m', '--model_name', metavar='', type=str, default="model", help='save the trained model into or load a trained model from "model_name.pt" - defaul name = "model"')
    parser.add_argument('-ch', '--checkpoint', metavar='', type=int, default=0, help='set to "1" to continue training from a previous checkpoint- default name = 0')

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
# training function
def make_train_step(model, loss_fn, optimizer):
    def train_step(x, y):
        """ A full training step over the current data batch
            parameters
            ----------
                x: input batch to the network
                y: true output batch
            returns
            -------
                error of the current batch
        """
        model.train()

        out = model(x)
        
        loss = loss_fn(out[-1], y)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        return loss

    return train_step


def training(args):
    """runs a full training of the network over the loaded dataset and saves the model
    """
    # training parameters
    epochs = args.num_epoch
    learning_rate = args.learning_rate
    decay_rate = args.decay_rate
    decay_step = args.decay_step

    channels = args.channels
    batch_size = args.batch_size
    model_name = args.model_name
    ckp = args.checkpoint

    # initialize pytorch data loaders:
    train_loader = data_load(path_input='sample-dataset/material/mat', path_output='sample-dataset/encoded-damage/dmg', batch_size=batch_size, dataset_size=1000, shuffle=True)
    test_loader = data_load(path_input='sample-dataset/material/mat', path_output='sample-dataset/encoded-damage/dmg', batch_size=batch_size, dataset_size=1000, shuffle=False)

    # GPU
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    model = Model(channels) # creating an instance of the MultiResWNet
    model = model.double()
    start_epoch = 0
    min_test_loss = np.inf
    if ckp:
        model, start_epoch, min_test_loss = load_model_ckp('checkpoint/'+model_name+'.pt', model)
    model = nn.DataParallel(model, device_ids=[0])
    model.to(dev)

    loss_fn = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_step, gamma=decay_rate)
    if ckp:
        optimizer = load_optim_ckp('best_model/'+model_name+'.pt', optimizer)

    train_step = make_train_step(model, loss_fn, optimizer)

    train_loss = [] # recording training error after each epoch
    test_loss = [] # recording test error after each epoch
    for epoch in range(start_epoch, epochs):
        loss = 0
        n_batches = 0
        # A training epoch
        for x_batch, y_batch in train_loader:
            n_batches += 1
            x_batch = x_batch.to(dev)
            y_batch = y_batch.to(dev)
            l = train_step(x_batch, y_batch)
            if n_batches % 20 == 0:
                print(f"epoch {epoch+1}/{epochs} | batch {n_batches} | batch loss: {l.item()}")
            loss += l
        scheduler.step()
        train_loss = np.append(train_loss, loss.item()/n_batches)

        # Test error calculation
        with torch.no_grad():
            loss_test = 0
            n_batches_test = 0
            for x_batch_test, y_batch_test in test_loader:
                n_batches_test += 1
                x_batch_test = x_batch_test.to(dev)
                y_batch_test = y_batch_test.to(dev)
                model.eval()
                test_out = model(x_batch_test)[-1]
                loss_test += loss_fn(y_batch_test, test_out)

        test_loss = np.append(test_loss, loss_test.item()/n_batches_test)
        print('epoch: ', epoch+1, '  |  loss: ', loss.item()/n_batches, '  |  loss (test): ', loss_test.item()/n_batches_test)

        checkpoint = {
            'min_test_loss': min_test_loss,
            'epoch': epoch+1,
            'state_dict': model.module.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        is_best = False
        if min_test_loss > test_loss[-1]:
            is_best = True
            min_test_loss = test_loss[-1]

        adr = './'
        if not os.path.exists('./checkpoint'):
            os.mkdir('./checkpoint')
        if not os.path.exists('./best_model'):
            os.mkdir('./best_model')
        save_ckp(checkpoint, is_best, adr+'/checkpoint/'+model_name+'.pt', adr+'/best_model/'+model_name+'.pt')

    with open('error-hist-'+str(model_name)+'.npy', 'wb') as file:
        np.save(file, train_loss)
        np.save(file, test_loss)


if __name__ == "__main__":
    args = parseArgs()
    training(args)
