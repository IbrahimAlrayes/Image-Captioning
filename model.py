import torch
import torch.nn as nn
import torchvision.models as models
from torch import optim 


class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_cnn=False):
        super(EncoderCNN, self).__init__()

        self.train_cnn = train_cnn

        self.efficientnet =  models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
        self.efficientnet.classifier[1] = nn.Linear(self.efficientnet.classifier[1].in_features, embed_size)
        
        # for param in self.efficientnet.parameters():
        #     param.requires_grad = False
        # self.efficientnet.classifier.requires_grad = True
        # print(self.efficientnet.classifier.weight)
        # print(self.efficientnet.classifier.bias)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, images):
        features = self.efficientnet(images)

        # for name, param in self.efficientnet.parameters():
        #     if "classifier.weight" in name or "classifier.bias" in name:
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = self.train_cnn



        return self.dropout(self.relu(features))
    


class Decoder_RNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocabulary_size, n_layers):
        super(Decoder_RNN, self).__init__()

        self.embedding = nn.Embedding(vocabulary_size, embed_size) 
        self.rnn = nn.LSTM(embed_size, hidden_size, n_layers)
        self.linear = nn.Linear(hidden_size, vocabulary_size)
        # self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        embeddings = self.embedding(captions)
        embeddings = torch.concat((features.unsqueeze(0), embeddings), dim=0)

        hiddens, cells = self.rnn(embeddings)

        outputs = self.linear(hiddens)
        return outputs
    

class CNNtoRNN(nn.Module):
    def __init__(self,embed_size, hidden_size, vocabulary_size, n_layers,train_cnn=False):
        super(CNNtoRNN, self).__init__()

        self.CNN_Encoder = EncoderCNN(embed_size, train_cnn)
        self.RNN_Decoder = Decoder_RNN(embed_size, hidden_size, vocabulary_size, n_layers)

    def forward(self, images, captions):
        features = self.CNN_Encoder(images)
        outputs = self.RNN_Decoder(features,captions)
        self.device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return outputs
    
    def caption_image(self, image, vocab, max_length=50):
        result = []

        with torch.no_grad():
            image = image.to(self.device).unsqueeze(0)
            x = self.CNN_Encoder(image)
            states = None
            
            for i in range(max_length):
                hiddens , states = self.RNN_Decoder.rnn(x, states)
                output = self.RNN_Decoder.linear(hiddens.squeeze(0))
                prediction = output.argmax()
                result.append(prediction.item())
                x = self.RNN_Decoder.embedding(prediction).unsqueeze(0)
                if vocab.itos[prediction.item()] == "<EOS>":
                    break
            return [vocab.itos[idx] for idx in result]


# model = CNNtoRNN(embed_size=200, n_layers=1, vocabulary_size=1000, hidden_size=256)
# optimizer = optim.Adam(model.parameters(), lr=0.0001)
