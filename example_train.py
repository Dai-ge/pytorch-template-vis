import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import torch.optim as optim
import time
from example_model import *
from data_utils import *
from skimage.io import imsave
from skimage.io import imread
from skimage import data,img_as_float,img_as_int

from pyevtk.hl import imageToVTK
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def trainNet(model,args,dataset):
    #init and parse the settings
    all_setting = args
    mode = args['mode']
    model_setting = args['model_setting']
    train_setting = args['train_setting']
    num_epochs = train_setting['num_epochs']
    checkpoints = train_setting['checkpoints']
    data_setting = args['data_setting']
    model_name = args['model_name']
    dataset_name = data_setting['dataset']+'-{}'.format(data_setting['var'])
    logging_baseDir = train_setting['log_dir']
    

    

    #optimizer and loss init
    optimizer = optim.Adam(model.parameters(), lr=train_setting['lr'],betas=(0.9,0.999),weight_decay=1e-6)
    criterion = nn.MSELoss()
    train_loader = dataset.GetTrainingData()
    t = 0
    for itera in range(1,num_epochs+1):
        #open loss file
        loss_widget = open(logging_baseDir+'Log/'+f'{dataset_name}-{model_name}-loss.txt','a+')

        
        x = time.time()
        #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,last_epoch=-1)
        print('======='+str(itera)+'========')
        loss_mse = 0
        loss_grad = 0
        idx = 0
        for batch_idx, (coord,v) in enumerate(train_loader):
            t1 = time.time()
            idx += 1
            coord = coord.to(device)
            v = v.to(device)
            optimizer.zero_grad()
            v_pred = model(coord)
            mse = criterion(v_pred.view(-1),v.view(-1))
            mse.backward()
            loss_mse += mse.mean().item()
            optimizer.step()
            #print(time.time()-t1)
        
        y = time.time()
        t += y-x
        print(y-x)
        print("Epochs "+str(itera)+": loss = "+str(loss_mse/idx))
        loss_widget.write("Epochs "+str(itera)+": loss = "+str(loss_mse/idx))
        loss_widget.write('\n')

        #save the model at checkpoints
        if itera%checkpoints == 0 or itera == 1:
            torch.save(model.state_dict(),logging_baseDir+'Log/'+'STCoordNet'+'-'+str(model_setting['init'])+'init'+'-'+str(model_setting['num_res'])+'res'+'-'+str(itera)+'.pth')

        # loss_widget.write("Time = "+str(t))
        loss_widget.write('\n')
        loss_widget.close()

def inf(model,args,dataset):
    #init and parse the settings
    all_setting = args
    mode = args['mode']
    model_setting = args['model_setting']
    train_setting = args['train_setting']
    num_epochs = train_setting['num_epochs']
    checkpoints = train_setting['checkpoints']
    data_setting = args['data_setting']
    model_name = args['model_name']
    dataset_name = data_setting['dataset']+'-{}'.format(data_setting['var'])
    logging_baseDir = train_setting['log_dir']


    model.load_state_dict(torch.load(logging_baseDir+'Log/'+'STCoordNet'+'-'+str(model_setting['init'])+'init'+'-'+str(model_setting['num_res'])+'res'+'-'+str(num_epochs)+'.pth'))
    
    #model inf code
    
    inf_dataset = dataset.GetTestingData()
    for t_index,t_coords in enumerate(inf_dataset):
        t_index = t_index + 1 #the time slot starts from 1
        data_loader = DataLoader(dataset = torch.FloatTensor(t_coords),batch_size=data_setting['batch_size'],shuffle=False)
        v_res = []
        for batch_idx,coord in enumerate(data_loader):
            coord = coord.cuda()
            with torch.no_grad():
                v_pred = model(coord)
                v_res += list(v_pred.view(-1).detach().cpu().numpy())
        v_res = np.asarray(v_res,dtype='<f')
        save_file_name = f'{dataset_name}-{t_index:4d}'
        save_file_path = os.path.join(train_setting['log_dir']+'Results',save_file_name)
        v_res=v_res.reshape((640,240,80))
        imageToVTK(save_file_path,pointData = {'sf':v_res})
            