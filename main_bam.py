import torch
import torch.autograd
from torch import Tensor
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt

from utils.loss_funcs import *
from utils.BAM_motion_3d import BAM_Motion3D

def train(dataset, vald_dataset, model):
    train_loss = []
    val_loss = []
    val_loss_best = 1000
    print('>>> Training dataset length: {:d}'.format(dataset.__len__()))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    #vald_dataset = datasets.Datasets(path,input_n,output_n,skip_rate, split=1)
    print('>>> Validation dataset length: {:d}'.format(vald_dataset.__len__()))
    vald_loader = DataLoader(vald_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    for epoch in range(n_epochs-1):
        running_loss = 0
        n = 0
        model.train()
        for cnt, batch in enumerate(data_loader):
            batch = batch.float().to(device)
            batch_dim = batch.shape[0]
            n += batch_dim

            sequences_train = batch[:, :input_n].view(-1, input_n, joints_to_consider, 3).permute(0, 3, 1, 2)
            sequences_gt = batch[:, input_n:input_n+output_n].view(-1, output_n, joints_to_consider, 3)

            optimizer.zero_grad()
            print(sequences_train.shape)
            sequences_predict = model(sequences_train).permute(0, 1, 3, 2)
            loss = mpjpe_error(sequences_predict, sequences_gt)
            if cnt % 200 == 0:
                print('[%d, %5d]  training loss: %.3f' %(epoch + 1, cnt + 1, loss.item()))
            loss.backward()
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            running_loss += loss * batch_dim

        train_loss.append(running_loss.detach().cpu() / n)  
        model.eval()
        with torch.no_grad():
            running_loss = 0 
            n = 0
            for cnt, batch in enumerate(vald_loader):
                batch = batch.float().to(device)
                batch_dim = batch.shape[0]
                n += batch_dim
                
                sequences_train = batch[:, :input_n].view(-1, input_n, joints_to_consider, 3).permute(0, 3, 1, 2)
                sequences_gt = batch[:, input_n:input_n+output_n].view(-1, output_n, joints_to_consider, 3)

                sequences_predict = model(sequences_train).permute(0, 1, 3, 2)
                loss = mpjpe_error(sequences_predict,sequences_gt)
                if cnt % 200 == 0:
                    print('[%d, %5d]  validation loss: %.3f' %(epoch + 1, cnt + 1, loss.item())) 
                running_loss += loss * batch_dim
            
            val_loss.append(running_loss.detach().cpu() / n)
            
            if running_loss / n < val_loss_best:
                val_loss_best = running_loss / n
                torch.save(model.state_dict(), f"{model_path}{model_name}_STS_best")

        if use_scheduler:
            scheduler.step()

        if (epoch + 1) % 10 == 0:
            print('----saving model-----')
            torch.save(model.state_dict(), f"{model_path}{model_name}_STS")
            plt.figure(1)
            plt.plot(train_loss, 'r', label='Train loss')
            plt.plot(val_loss, 'g', label='Val loss')
            plt.legend()
            plt.savefig(f"loss_curve_epoch_{epoch+1}.png")
        
        if (epoch + 1) == n_epochs:
            print('----saving model-----')

            torch.save(model.state_dict(), f"{model_path}{model_name}_STS")


datas = 'BAM'
path = '/home/ssg1002/Datasets/BAM'

input_n = 10 # number of frames to train on(default=10)
output_n = 25 # number of frames to predict on
input_dim = 3 # dimensions of the input coordinates(default=3)
skip_rate = 1 # skip rate of frames for H3.6M (default=1) 
joints_to_consider = 17  #joints

# FLAGS FOR THE MODEL
tcnn_layers = 4 # number of layers for the Temporal Convolution of the Decoder (default=4)
tcnn_kernel_size = [3, 3] # kernel for the T-CNN layers (default=[3,3])
input_dim = 3 # dimensions of the input coordinates(default=3)
st_gcnn_dropout = 0.1 # (default=0.1)
tcnn_dropout = 0.0  # (default=0.0)

n_epochs = 61
batch_size = 256
batch_size_test = 8
lr = 1e-01 # learning rate
use_scheduler = True # use MultiStepLR scheduler
milestones = [10, 25, 30, 37] # the epochs after which the learning rate is adjusted by gamma ########### SOTA [25,30,37] 

gamma = 0.1 # gamma correction to the learning rate, after reaching the milestone epochs
clip_grad = None # select max norm to clip gradients
model_path = '/home/ssg1002/cvg_hiwi/CHICO-PoseForecasting/checkpoints/' # path to the model checkpoint file
model_name = f"{datas}_3d_{output_n}_frames_ckpt" #the model name to save/load

from models.SeSGCN_teacher import Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

model = Model(input_dim, input_n, output_n, st_gcnn_dropout, joints_to_consider, tcnn_layers, tcnn_kernel_size, tcnn_dropout).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-05)
if use_scheduler:
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
print('Total number of parameters of the network is: ' + str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

# Load Data
dataset = BAM_Motion3D(path, input_n, output_n, 'train')
vald_dataset = BAM_Motion3D(path, input_n, output_n, 'validation')

train(dataset, vald_dataset, model)