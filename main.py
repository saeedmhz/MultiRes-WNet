""" Load the dataset, train or evaluate the MultiResWNet model on the dataset

Parameters
----------
file_loc : str
    The file location of the spreadsheet
print_cols : bool, optional
    A flag used to print the columns to the console (default is False)

Returns
-------
    trained model will be saved in
"""
import torch
from torch import nn
import numpy as np
import argparse
import shutil

from Network.MultiResUNet import UNet
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
    parser.add_argument('-mn', '--model_name', metavar='', type=str, default="model", help='save the trained model into or load a trained model from "model_name.pt" - defaul name = "model"')
    parser.add_argument('-m', '--save_model', metavar='', type=str, default='false' , help='Set to true if you want to save your model - default value = False')
    parser.add_argument('-tr', '--train', metavar='', type=str, default='true', help='set to true if you are training the model - defaul value = True')
    parser.add_argument('-ev', '--eval', metavar='', type=str, default='false', help='Set to true if you are evaluating a model - default value = False')

    return parser.parse_args()


#############################
# Create pytorch dataloader #
#############################
def data_init(batch_size):
    path_train_input = 'D:/phase-field/data/train/material/mat'
    path_train_output = 'D:/test/train/autoencoder_10000x8-weightdecay-1e-4/encoded/enc'
    train_loader = data_load(path_train_input, path_train_output, batch_size, 1000, True)

    path_test_input = 'D:/phase-field/data/train/material/mat'
    path_test_output = 'D:/test/train/autoencoder_10000x8-weightdecay-1e-4/encoded/enc'
    test_loader = data_load(path_test_input, path_test_output, 8, 8, False)

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

    # GPU
    dev = 'cuda:0' #if torch.cuda.is_available() else 'cpu'

    model = Model(channels)
    model.load_state_dict(torch.load('wnet-c64-b32-wd0_auto-c32-b8-wd1e-4.pt'))
    model = model.double()
    model = nn.DataParallel(model, device_ids=[0,1])#,2,3])
    model.to(dev)

    addr = '/projectnb/lejlab2/saeed/model-training/crackpath/pred'
    with torch.no_grad():
        num = 0
        for x_batch, _ in train_loader:
            x_batch = x_batch.to(dev)
            out = model(x_batch)
            for i in range(len(out[-1])):
                np.save(addr+str(num)+'.npy', out[-1][i][0].cpu().detach().numpy())
                num += 1


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
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)#, weight_decay=1e-2)
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

        adr = 'C:/Users/saeed/OneDrive/دسکتاپ/Codes/Github/MultiRes-WNet'
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