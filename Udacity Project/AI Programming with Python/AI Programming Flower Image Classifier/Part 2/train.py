import argparse
from torch import nn
from torch import optim
from model import get_input_args, data_transforms, cuda_available, choose_arch, load_checkpoint, train_network, test_network, save_checkpoint

def main():   
    in_arg = get_input_args()
 
    # Prepare training, validation and testing datasets
    trainloader, validloader, testloader, train_dataset = data_transforms()
    
    # Use CUDA if available
    device = cuda_available(in_arg.gpu)
    
    # Choose architecture
    model = choose_arch(in_arg.arch, in_arg.hidden_units1, in_arg.hidden_units2, in_arg.hidden_units3)
    
    # Initialize epochs
    epochs_start = 0
    
    # Load checkpoint if provided
    if in_arg.load_checkpoint != 'None':
        model, epochs_start = load_checkpoint(in_arg.load_checkpoint)
        print('Checkpoint successfully loaded..')
        
    # Define criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=in_arg.learning_rate)    
    print(f'Learning rate: {in_arg.learning_rate}')
    print(f'Data Directory: {in_arg.data_dir}')
    
    model.to(device);
    
    # Train the network
    epochs_done = train_network(in_arg.epochs, trainloader, validloader, model, device, optimizer, criterion, epochs_start)
    
    # Test the network
    test_network(testloader, model, device, optimizer, criterion)
    
    # Save the checkpoint
    save_checkpoint(in_arg.arch, in_arg.save_dir, model, optimizer, train_dataset, epochs_done)
    
if __name__ == '__main__':
    main()