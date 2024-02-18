import numpy as np
import gzip, os, torch
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
from Utils import Encoder, Decoder, get_data



def train_autoencoder(batch_size,epochs,device,checkpoint_path,data_path,verbose):
    BATCH_SIZE = batch_size
    EPOCHS = epochs
    LEARNING_RATE = 3e-4
    DEVICE = device
    CHECKPOINT_PATH = checkpoint_path
    DATAPATH = data_path
    OUTPUT_SIZE= (32, 32)

    train_data = get_data(DATAPATH)

    X_train,X_val = train_test_split(train_data,
                                    test_size=0.2,
                                    random_state=13)

    encoder = Encoder(input_channels=1, num_classes=10)
    encoder = encoder.to(DEVICE)

    decoder = Decoder(input_channels=64, num_classes=10, out_channels=1)
    decoder = decoder.to(DEVICE)

    # train_dataset = MyDataset(X_train)
    train_loader = DataLoader(X_train, batch_size=BATCH_SIZE, shuffle=True)
    # val_dataset = MyDataset(X_val,y_val)
    val_loader = DataLoader(X_val, batch_size=BATCH_SIZE, shuffle=True)

    mse_loss = nn.MSELoss()
    parameters = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(parameters, lr=LEARNING_RATE)

    mse_score_best = 100
    for epoch in range(EPOCHS):
        print(f'Epoch:{epoch}')
        encoder.train()
        decoder.train()
        encoder = encoder.to(DEVICE)
        decoder = decoder.to(DEVICE)

        for images_train in tqdm(train_loader):
            
            images_train = images_train.to(DEVICE)

            optimizer.zero_grad()

            output = encoder(images_train)
            output = decoder(output)

            # Use interpolate to resize the tensor
            images_train = F.interpolate(images_train, size=OUTPUT_SIZE, mode='bilinear', align_corners=False)
            loss = mse_loss(output, images_train)

            loss.backward()
            optimizer.step()

        list_loss_val = []
        for data_val,label_val in val_loader:
            encoder.eval()
            decoder.eval()
            
            data_val = data_val.to(DEVICE)
            label_val = label_val.to(DEVICE)
            
            output = encoder(data_val)
            output = decoder(output)
            
            output_size = (32, 32)
            data_val = F.interpolate(data_val, size=output_size, mode='bilinear', align_corners=False)
            list_loss_val.append(mse_loss(output, data_val).item())
        
        if np.mean(list_loss_val)<mse_score_best:
            mse_score_best = np.mean(list_loss_val)
            if verbose:
                print('The val loss decreased... saving model')
                print(f'New val loss: {mse_score_best}')
            torch.save({
                        'epoch':epoch,
                        'encoder':encoder.cpu().state_dict(),
                        'decoder':decoder.cpu().state_dict(),
                        'opt':optimizer.state_dict(),
                        },os.path.join(CHECKPOINT_PATH,'checkpoint_autoencoder.pth.tar'))
        
        
        
