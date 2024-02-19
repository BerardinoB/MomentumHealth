import argparse
from Utils import train_autoencoder,train_classifier


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Training SSL Model")
    
    parser.add_argument("--input_path", help="Path to the input data folder", default='./fashion-mnist/data/fashion')
    parser.add_argument("--checkpoint_path", help="Path to save checkpoints", default='./checkpoint')
    parser.add_argument("--epochs", help="N epochs", default=200)
    parser.add_argument("--device", help="devise for training", default='cuda:0')
    parser.add_argument("--batch_size", help="batch size for training", default=32)
    parser.add_argument("--supervision", help="Percentage of labeled data (1 to 10)", default=10)
    parser.add_argument("--verbose", action="store_true", help="Enable verbose mode")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the parsed arguments
    input_path = args.input_path
    checkpoint_path = args.checkpoint_path
    epochs = int(args.epochs)
    device = args.device
    batch_size = int(args.batch_size)
    perc_supervision = int(args.supervision)
    verbose = args.verbose
    
    train_autoencoder(batch_size,epochs,device,checkpoint_path,input_path,verbose)
    train_classifier(perc_supervision,batch_size,epochs,device,checkpoint_path,input_path,verbose)

    print('End of Training!!!!')
