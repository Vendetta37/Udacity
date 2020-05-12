import argparse
import torch
from torch import nn
from torchvision import datasets, transforms, models

def get_input_args():
    
    #Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_dir', type=str, default='flowers',
                        help='path to the folder of flowers images')
                        
    parser.add_argument('--gpu', type=int, default=1,
                        help='use CUDA if GPU is available')
    
    parser.add_argument('--arch', type=str, default='None',
                        help='choose architecture')
    
    parser.add_argument('--hidden_units1', type=int, default=3000,
                        help='Set the number of hidden units in hidden layer 1')
    
    parser.add_argument('--hidden_units2', type=int, default=1000,
                        help='Set the number of hidden units in hidden layer 2')
    
    parser.add_argument('--hidden_units3', type=int, default=400,
                        help='Set the number of hidden units in hidden layer 2')
    
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='decide learning rate')
    
    parser.add_argument('--epochs', type=int, default=5,
                        help='decide epochs')
    
    parser.add_argument('--save_dir', type=str, default='checkpoint',
                        help='Set directory to save checkpoint')
    
    parser.add_argument('--load_checkpoint', type=str, default='None',
                        help='input checkpoint to load')
    
 
    return parser.parse_args()

# Prepare training, validation and testing datasets
def data_transforms():
    in_arg = get_input_args()
    
    # Path of the datasets
    data_dir = in_arg.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Transforms for the training, validation and testing datasets
    train_transforms = transforms.Compose([transforms.RandomRotation(50),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])]) 
    # Load the datasets with ImageFolder
    test_transforms = transforms.Compose([transforms.Resize(225),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=test_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)

    # Use the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

    return trainloader, validloader, testloader, train_dataset

# Use CUDA if available
def cuda_available(n):
    if n == 1:
        if torch.cuda.is_available() == True:
            device = torch.device('cuda')
            print('Mode: GPU')
            
        else:
            device = torch.device('cpu')
            print('CUDA is not available, swtich back to CPU mode',)
            print('Mode: CPU')
            
    elif n == 0:
        device = torch.device('cpu')
        print('Mode: CPU')
        
    else:
        print('Wrong GPU input: ON:1, OFF:0')

    return device

# Choose architecture
def choose_arch(arch, hidden_units1, hidden_units2, hidden_units3):
    if arch == 'None':
        print('Error: Please select one of these architecture: vgg11, alexnet, densenet121')
    
    if arch == 'vgg11':
        print('Architecture: VGG11')
        model = models.vgg11(pretrained=True)  
        
        for param in model.parameters():
            param.requires_grad = False 
            
        if hidden_units1 >= 25088:
            hidden_units1 = 3000
        if hidden_units2 >= hidden_units1:
            hidden_units2 = 1000 
        if hidden_units3 >= hidden_units2:
            hidden_units3 = 400 
        
        model.classifier = nn.Sequential(nn.Linear(25088, hidden_units1),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.2),
                                        nn.Linear(hidden_units1, hidden_units2),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.2),
                                        nn.Linear(hidden_units2, hidden_units3),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.2),
                                        nn.Linear(hidden_units3, 102),
                                        nn.LogSoftmax(dim=1))
        
    elif arch == 'alexnet':
        print('Architecture: ALEXNET')
        model = models.alexnet(pretrained=True)
        
        for param in model.parameters():
            param.requires_grad = False 
            
        if hidden_units1 >= 9126:
            hidden_units1 = 2000
        if hidden_units2 >= hidden_units1:
            hidden_units2 = 800
        if hidden_units3 >= hidden_units2:
            hidden_units3 = 400    
        
        model.classifier = nn.Sequential(nn.Linear(9216, hidden_units1),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.2),
                                        nn.Linear(hidden_units1, hidden_units2),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.2),
                                        nn.Linear(hidden_units2, hidden_units3),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.2),
                                        nn.Linear(hidden_units3, 102),
                                        nn.LogSoftmax(dim=1))
    
    elif arch == 'densenet121':
        print('Architecture: DENSENET121')
        model = models.densenet121(pretrained=True)
        
        for param in model.parameters():
            param.requires_grad = False 
        
        if hidden_units1 >= 1024:
            hidden_units1 = 600
        if hidden_units2 >= hidden_units1:
            hidden_units2 = 300
        
        model.classifier = nn.Sequential(nn.Linear(1024, hidden_units1),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.2),
                                        nn.Linear(hidden_units1, hidden_units2),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.2),
                                        nn.Linear(hidden_units2, 102),
                                        nn.LogSoftmax(dim=1))            
        
    else:
        print('Error: provided architecture is not available') 
    
    return model    

# Load checkpoint if provided
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    
    if checkpoint['arch'] == 'vgg11':
        model = models.vgg11(pretrained=True)  
        
    elif checkpoint['arch'] == 'alexnet':
        model = models.alexnet(pretrained=True)
        
    elif checkpoint['arch'] == 'densenet121':
        model = models.densenet121(pretrained=True)
        
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = checkpoint['classifier']
    model.classifier.load_state_dict = checkpoint['classifier_state_dict']
    model.class_to_idx = checkpoint['class_to_idx']
    
    epochs_start = checkpoint['epochs_done']
    return model, epochs_start
    
# Train the network
def train_network(epochs, trainloader, validloader, model, device, optimizer, criterion, epochs_start):
    epochs_done = epochs_start + epochs
    running_loss = 0
    steps = 0
    print_every = 50
    
    for epoch in range(epochs_start, epochs_start + epochs):
        for images, labels in trainloader:
            model.train()
            steps += 1
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for images, labels in validloader:
                        images, labels = images.to(device), labels.to(device)
                        log_ps = model(images)
                        valid_loss += criterion(log_ps, labels)

                        ps = torch.exp(model(images))
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor))

                print(f"Epoch: {epoch+1}",
                      f"Accuracy: {(accuracy/len(validloader))*100:.3f}%",
                      f"Train_loss: {running_loss/print_every:.3f}",
                      f"Valid_loss: {valid_loss/len(validloader):.3f}")

                running_loss = 0
                model.train()
    return epochs_done   

# Test the network
def test_network(testloader, model, device, optimizer, criterion):
    test_loss = 0
    accuracy = 0
    model.eval()

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            log_ps = model(images)
            test_loss += criterion(log_ps, labels)

            ps = torch.exp(model(images))
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))

        print('Test result:',
              f"Accuracy: {(accuracy/len(testloader))*100:.3f}%",
              f"Test_loss: {test_loss/len(testloader):.3f}")
        
# Save checkpoint
def save_checkpoint(arch, save_dir, model, optimizer, train_dataset, epochs_done):
    checkpoint = {'arch': arch,
                  'model_state_dict': model.state_dict(),
                  'classifier': model.classifier,
                  'classifier_state_dict': model.classifier.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict,
                  'class_to_idx': train_dataset.class_to_idx,
                  'epochs_done': epochs_done
                 }

    torch.save(checkpoint, str(save_dir) + '/' + str(arch) + '_checkpoint.pth')
    print('Checkpoint save as ' + str(save_dir) + '/' + str(arch) + '_checkpoint.pth')