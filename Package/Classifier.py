import numpy as np
import gzip, os, torch
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
from Utils import extract_data, extract_labels, Encoder,MyDataset
from sklearn.metrics import roc_auc_score

CHECKPOINT_PATH = '/usr/local/data/bbera/Challenge/checkpoint'
LEARNING_RATE = 3e-4
EPOCHS = 100
DEVICE = 'cuda:0'
BATCH_SIZE = 32
PERC_SUPERVISION = 10


def train_classifier(perc_supervision,batch_size,epochs,device,checkpoint_path,data_path,verbose):
    CHECKPOINT_PATH = checkpoint_path
    LEARNING_RATE = 3e-4
    EPOCHS = epochs
    DEVICE = device
    BATCH_SIZE = batch_size
    PERC_SUPERVISION = perc_supervision
    DATAPATH = data_path
    assert perc_supervision<=10, f'Only up to 10% of labeled data are allowed'

    train_data,train_labels = get_data(DATAPATH,flag_labels=True)

    X_train,X_val,y_train,y_val = train_test_split(train_data,
                                                    train_labels,
                                                    test_size=0.2,
                                                    random_state=13)

    #Only 10% of the dataset is available with label.
    num_label_data_available = train_labels.shape[0]*(PERC_SUPERVISION/10)
    n_sample_train = int(num_label_data_available*0.8)
    n_sample_val = int(num_label_data_available-n_sample_train)

    idx_train = np.random.choice(range(X_train.shape[0]),n_sample_train,replace=False)
    X_train_classifier = X_train[idx_train,...]
    idx_val = np.random.choice(range(X_val.shape[0]),n_sample_val,replace=False)
    X_val_classifier = X_val[idx_val,...]

    encoder = Encoder(input_channels=1, num_classes=10)
    checkpoint = torch.load(os.path.join(CHECKPOINT_PATH,'checkpoint_autoencoder.pth.tar'),map_location={'cuda:0':'cpu'})
    encoder.load_state_dict(checkpoint['encoder'])

    train_dataset = MyDataset(X_train,y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataset = MyDataset(X_val,y_val)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    CE_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(encoder.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        print(f'Epoch:{epoch}')
        
        for name, param in encoder.named_parameters():
            if 'fc3' not in name:
                param.requires_grad = False
        
        encoder = encoder.to(DEVICE)
        
        for images_train,train_label in tqdm(train_loader):
            
            images_train = images_train.to(DEVICE)
            train_label = train_label.to(DEVICE)

            optimizer.zero_grad()
            
            output = encoder(images_train)
            one_hot_labels = F.one_hot(train_label, 10)
            
            loss = CE_loss(output,one_hot_labels.float())
            
            loss.backward()
            optimizer.step()
        
        preds_list,true_list = [],[]
        for data_val,label_val in val_loader:
            
            encoder.eval()
            data_val = data_val.to(DEVICE)
            label_val = label_val.to(DEVICE)
            
            output = encoder(data_val)
            output = F.softmax(output)
            one_hot_labels = F.one_hot(label_val, 10)
            
            preds_list.append(output.detach().cpu().numpy())
            true_list.append(one_hot_labels.cpu().numpy())
        
        print(roc_auc_score(np.vstack(true_list),np.vstack(preds_list)))