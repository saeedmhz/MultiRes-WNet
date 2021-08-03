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
save_model: str
    Set to "true" if you want to save a trained model
model_name: str
    Model name for saving and loading a trained model
train: str
    Set to "true" if you want to train a model
eval: str
    Set to "true" if you want to evaluate a trained model

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

from Network.MultiResUNet import UNet

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
    parser.add_argument('-mn', '--model_name', metavar='', type=str, default="model", help='save the trained model into or load a trained model from "model_name.pt" - defaul name = "model"')
    parser.add_argument('-m', '--save_model', metavar='', type=str, default='false' , help='Set to true if you want to save your model - default value = False')
    parser.add_argument('-tr', '--train', metavar='', type=str, default='true', help='set to true if you are training the model - defaul value = True')
    parser.add_argument('-ev', '--eval', metavar='', type=str, default='false', help='Set to true if you are evaluating a model - default value = False')

    return parser.parse_args()


#############################
# Create pytorch dataloader #
#############################
def data_init(batch_size):
    train_input = np.loadtxt('./sample-dataset/sample_input.txt').reshape(-1,1,28,28)
    train_output = np.loadtxt('./sample-dataset/sample_output_y.txt').reshape(-1,1,28,28)
    test_input = np.loadtxt('./sample-dataset/sample_input.txt').reshape(-1,1,28,28)
    test_output = np.loadtxt('./sample-dataset/sample_output_y.txt').reshape(-1,1,28,28)
    
    # normalize inputs
    mu = 33.318421449829934 #based on the mnist input dataset
    std = 78.56748998339798 #based on the mnist input dataset
    train_input = (train_input-mu) / std
    test_input = (test_input-mu) / std

    # convert numpy array data to torch tensor
    train_input = torch.tensor(train_input, dtype=float)
    train_output = torch.tensor(train_output, dtype=float)
    test_input = torch.tensor(test_input, dtype=float)
    test_output = torch.tensor(test_output, dtype=float)

    # set up torch dataloader
    train_loader = DataLoader(TensorDataset(train_input, train_output), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_input, test_output), batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


######################
# Creating the Model #
######################
class Model(nn.Module):
    def __init__(self, c):
        super(Model, self).__init__()
        self.U1 = UNet(c, 3, 1)
        self.U2 = UNet(c, 3, 1)


    def forward(self, x):
        out1 = self.U1(x)
        out2 = self.U2(out1)

        return [out1, out2]


########################
# Evaluating the Model #
########################
def evaluation(args):
    # args
    channels = args.channels
    model_name = args.model_name

    # load the dataset
    train_loader, test_loader = data_init(batch_size)
    
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    model = Model(channels)
    model.load_state_dict(torch.load(model_name+'.pt'))
    #model, _ = load_model_ckp('./trained-models/'+model_name+'.pt', model)
    model = model.double()
    # uncomment in case of using multiple gpus
    #model = nn.DataParallel(model, device_ids=[0])
    model.to(dev)

    addr = './predicted_output'
    if not os.path.exists(addr):
        os.mkdir(addr)
    with torch.no_grad():
        model.eval()
        pred = np.zeros((0,28,28))
        for x_batch, _ in test_loader:
            x_batch = x_batch.to(dev)
            out = model(x_batch)
            output = out[-1].cpu().detach().numpy().reshape(-1,28,28)
            pred = np.concatenate([pred, output], axis=0)
        np.savetxt(addr+model_name+'.txt', pred.reshape(-1,28*28), fmt='%.5f)


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

        loss = loss_fn(y, out[-1])
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
    ckp = False

    # initialize pytorch data loaders:
    train_loader, test_loader = data_init(batch_size)

    # GPU
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    model = Model(channels) # creating an instance of the MultiResWNet
    model = model.double()
    if ckp:
        model, start_epoch = load_model_ckp('best_model/'+model_name+'.pt', model)
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
    min_test_loss = np.inf
    for epoch in range(epochs):
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
        if epoch % 5 == 0:
            with torch.no_grad():
                loss_test = 0
                n_batches_test = 0
                for x_batch_test, y_batch_test in test_loader:
                    n_batches_test += 1
                    x_batch_test = x_batch_test.to(dev)
                    y_batch_test = y_batch_test.to(dev)
                    model.eval()
                    test_out = model(x_batch_test)
                    loss_test += loss_fn(y_batch_test, test_out[-1])

            test_loss = np.append(test_loss, loss_test.item()/n_batches_test)
            print('epoch: ', epoch+1, '  |  loss: ', loss.item()/n_batches, '  |  loss (test): ', loss_test.item()/n_batches_test)
        else:
            print('epoch: ', epoch+1, '  |  loss: ', loss.item()/n_batches)

        checkpoint = {
            'epoch': epoch,
            'state_dict': model.module.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        is_best = False
        if min_test_loss > test_loss[-1]:
            is_best = True
            min_test_loss = test_loss

        adr = './'
        if not os.path.exists('./checkpoint'):
            os.mkdir('./checkpoint)
        if not os.path.exists('./best_model'):
            os.mkdir('./best_model)
        save_ckp(checkpoint, is_best, adr+'/checkpoint/ckp_'+model_name+'.pt', adr+'/best_model/'+model_name+'.pt')

    with open('error_history.npy', 'wb') as file:
        np.save(file, train_loss)
        np.save(file, test_loss)


if __name__ == "__main__":
    args = parseArgs()
    if args.train == 'true':
        training(args)
    if args.eval == 'true':
        evaluation(args)
