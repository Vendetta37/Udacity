import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, n_layers=1):
        super(DecoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.output_size = vocab_size
        self.embed_size = embed_size
#         self.batch_size = batch_size
#         self.hidden = (torch.zeros(n_layers, batch_size, hidden_size).cuda(),
#                        torch.zeros(n_layers, batch_size, hidden_size).cuda())
        
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, n_layers, dropout=0.5, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        batch_size = features.size(0)
        captions = captions[:,:-1]
        
        e_out = self.embed(captions)
        e_out = torch.cat((features.unsqueeze(1), e_out), dim=1)
        r_out, _ = self.lstm(e_out)
        out = r_out.contiguous().view(-1, self.hidden_size)
        out = self.fc(out)
        out = out.view(batch_size, -1, self.output_size)
        
        return out

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        output = []
        caption = []
        caption.append(0)
        [caption.append(i) for i in output]
        caption = torch.Tensor(caption).long().cuda()
        
        for _ in range(max_len):
            batch_size = inputs.size(0)
            
            e_out = self.embed(caption)
            e_out = torch.cat((inputs, e_out.unsqueeze(0)), dim=1)
            r_out, _ = self.lstm(e_out, states)
            out = r_out.contiguous().view(-1, self.hidden_size)
            out = self.fc(out)
            out = out.view(batch_size, -1, self.output_size)            
            
            _, words = out.max(2)
            caption = words.squeeze(0)
            
            if caption[-1] == 1:
                break
                
        return caption.tolist()