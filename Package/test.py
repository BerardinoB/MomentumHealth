from Utils import Encoder,get_data
import argparse,torch,os
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import numpy as np

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Testing SSL Model")
    parser.add_argument("--input_path", help="Path to the input data folder", default='./fashion-mnist/data/fashion')
    parser.add_argument("--checkpoint_path", help="Path to save checkpoints", default='./checkpoint')
    parser.add_argument("--device", help="devise for testing", default='cuda:0')
    
    args = parser.parse_args()
    
    input_path = args.input_path
    checkpoint_path = args.checkpoint_path
    device = args.device
    
    checkpoint = torch.load(os.path.join(checkpoint_path,'checkpoint_classifier.pth.tar'),map_location={'cuda:0':'cpu'})
    encoder = Encoder(input_channels=1, num_classes=10)
    encoder.load_state_dict(checkpoint['encoder'])
    
    encoder = encoder.to(device)
    encoder.eval()
    test_data,test_labels = get_data(input_path,flag_label=True,flag_test=True)
    data_test = torch.from_numpy(test_data).to(device)
    label_test = torch.from_numpy(test_labels).to(device)
    output = encoder(data_test)
    output = F.softmax(output,dim=1)
    one_hot_labels = F.one_hot(label_test, 10)
    
    preds_list = output.detach().cpu().numpy()
    true_list = one_hot_labels.cpu().numpy()
    
    auc_score = roc_auc_score(np.vstack(true_list),np.vstack(preds_list))
    
    print(f'Test AUC score: {auc_score}')
    
