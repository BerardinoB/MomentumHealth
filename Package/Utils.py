import numpy as np
import gzip, os, torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch.optim as optim
from tqdm import tqdm

def extract_data(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(28 * 28 * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, 28,28)
        return data
    
def extract_labels(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        return labels

def flatten(x):
    """Flattens a tensor."""
    return x.view(x.size(0), -1)


class Encoder(nn.Module):

    def __init__(self, input_channels, num_classes=10):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=(5, 5), bias=False)
        self.max1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.batch1 = nn.BatchNorm2d(num_features=64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(5, 5), bias=False)
        self.batch2 = nn.BatchNorm2d(num_features=64)
        self.max2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.flat = flatten
        self.fc1 = nn.Linear(1024, 384, bias=True)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(384, 192, bias=True)
        self.fc3 = nn.Linear(192, num_classes, bias=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.max1(out)
        out = self.batch1(out)
        out = self.conv2(out)
        out = self.batch2(out)
        out = self.max2(out)
        out = self.flat(out)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

class Decoder(nn.Module):

    def __init__(self, input_channels, num_classes=10, out_channels=1):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(num_classes, 192, bias=True)
        self.fc2 = nn.Linear(192, 384, bias=True)
        self.fc3 = nn.Linear(384, 1600, bias=True)
        self.conv1 = nn.ConvTranspose2d(input_channels, out_channels=64, kernel_size=(5, 5), stride=2, padding=2,
                                        output_padding=1)
        self.batch1 = nn.BatchNorm2d(num_features=64)
        self.conv2 = nn.ConvTranspose2d(input_channels, out_channels=64, kernel_size=(5, 5), stride=1, padding=0,
                                        output_padding=0)
        self.conv3 = nn.ConvTranspose2d(input_channels, out_channels=64, kernel_size=(5, 5), stride=2, padding=2,
                                        output_padding=1)
        self.conv4 = nn.ConvTranspose2d(input_channels, out_channels=out_channels, kernel_size=(5, 5), stride=1,
                                        padding=0, output_padding=0)
        self.sigmoid = nn.Tanh()

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        out = out.view(-1, 64, 5, 5)
        out = self.conv1(out)
        out = self.batch1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.sigmoid(out)
        return out

class MyDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label
    
def get_data(basepath,flag_test=False,flag_label=False):
    train_data = extract_data(os.path.join(basepath,'train-images-idx3-ubyte.gz'), 60000)
    test_data = extract_data(os.path.join(basepath,'t10k-images-idx3-ubyte.gz'), 10000)

    train_labels = extract_labels(os.path.join(basepath,'train-labels-idx1-ubyte.gz'),60000)
    test_labels = extract_labels(os.path.join(basepath,'t10k-labels-idx1-ubyte.gz'),10000)

    train_data = train_data.reshape(-1, 1, 28,28)
    test_data = test_data.reshape(-1, 1, 28,28)
    train_data = train_data / np.max(train_data)
    test_data = test_data / np.max(test_data)
    
    if flag_test:
        if flag_label:
            return test_data,test_labels
        return test_data
    if flag_label:
        return train_data,train_labels
    return train_data

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

    train_loader = DataLoader(X_train, batch_size=BATCH_SIZE, shuffle=True)
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
        for data_val in val_loader:
            encoder.eval()
            decoder.eval()
            
            data_val = data_val.to(DEVICE)
            
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

   
def train_classifier(perc_supervision,batch_size,epochs,device,checkpoint_path,data_path,verbose):
    CHECKPOINT_PATH = checkpoint_path
    LEARNING_RATE = 3e-4
    EPOCHS = epochs
    DEVICE = device
    BATCH_SIZE = batch_size
    PERC_SUPERVISION = perc_supervision
    DATAPATH = data_path
    assert perc_supervision<=10 and perc_supervision>0, f'Only up to 10% of labeled data are allowed'

    train_data,train_labels = get_data(DATAPATH,flag_label=True)

    X_train,X_val,y_train,y_val = train_test_split(train_data,
                                                    train_labels,
                                                    test_size=0.2,
                                                    random_state=13)

    #Only up to 10% of the label data is allowed .
    num_label_data_available = train_labels.shape[0]*(PERC_SUPERVISION/100)
    n_sample_train = int(num_label_data_available*0.8)
    n_sample_val = int(num_label_data_available-n_sample_train)

    idx_train = np.random.choice(range(X_train.shape[0]),n_sample_train,replace=False)
    X_train_classifier = X_train[idx_train,...]
    y_train_classifier = y_train[idx_train]
    idx_val = np.random.choice(range(X_val.shape[0]),n_sample_val,replace=False)
    X_val_classifier = X_val[idx_val,...]
    y_val_classifier = y_val[idx_val]

    encoder = Encoder(input_channels=1, num_classes=10)
    checkpoint = torch.load(os.path.join(CHECKPOINT_PATH,'checkpoint_autoencoder.pth.tar'),map_location={'cuda:0':'cpu'})
    encoder.load_state_dict(checkpoint['encoder'])

    train_dataset = MyDataset(X_train_classifier,y_train_classifier)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataset = MyDataset(X_val_classifier,y_val_classifier)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    CE_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(encoder.parameters(), lr=LEARNING_RATE)

    auc_best = 0.5
    for epoch in range(EPOCHS):
        print(f'Epoch:{epoch}')
        
        encoder.train()
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
            output = F.softmax(output,dim=1)
            one_hot_labels = F.one_hot(label_val, 10)
            
            preds_list.append(output.detach().cpu().numpy())
            true_list.append(one_hot_labels.cpu().numpy())
        
        auc_score = roc_auc_score(np.vstack(true_list),np.vstack(preds_list))

        if auc_score>auc_best:
            auc_best = auc_score
            if verbose:
                print('Accuracy increased... saving model')
                print(f'New AUC score: {auc_score}')
            torch.save({
                        'epoch':epoch,
                        'encoder':encoder.cpu().state_dict(),
                        'opt':optimizer.state_dict(),
                        },os.path.join(CHECKPOINT_PATH,'checkpoint_classifier.pth.tar'))
