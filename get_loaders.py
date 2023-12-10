import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from torchvision.transforms import transforms
# import spacy
# import en_core_web_sm
from collections import Counter
from nltk.tokenize import word_tokenize
import pandas as pd
import os

# spacy_eng = en_core_web_sm.load()

class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}

        self.stoi = {tok: num for num,tok in self.itos.items()}

        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)
    
    @staticmethod
    def tokenize(text):
        return [token.lower() for token in word_tokenize(text)]

    
    def build_vocab(self, sentence_list):
        frequencies = Counter()

        idx = 4

        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                frequencies[word] += 1
                if frequencies[word] == self.freq_threshold:
                    self.itos[idx] = word
                    self.stoi[word] = idx
                    idx += 1
    
    def encode(self, text):
        tokenized_text = self.tokenize(text)
        return [self.stoi[word] if word in self.stoi else self.stoi['<UNK>'] for word in tokenized_text]


#testing the vicab class 
v = Vocabulary(freq_threshold=1)
sentence = ["This is a good place to find a city"]
v.build_vocab(sentence)
print(v.stoi)
print(v.encode("This is a good place to find a city here!!"))
                
               




class Flicker_Dataset(Dataset):
    def __init__(self, root, annotations, transform, freq_threshold=5):
        self.root = root
        self.df = pd.read_csv(annotations)
        self.transform = transform

        self.imgs = self.df['image']
        self.captions = self.df['caption']

        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocab(self.captions.tolist())


    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        caption = self.captions[index]
        img_name = self.imgs[index]
        img_location = os.path.join(self.root, img_name)

        img = Image.open(img_location).convert("RGB")

        # if self.transform is not None:
        img = self.transform(img)

        caption_encoding = []
        caption_encoding += [self.vocab.stoi["<SOS>"]]
        caption_encoding += self.vocab.encode(caption)
        caption_encoding += [self.vocab.stoi["<EOS>"]]

        # caption_encoding = []
        # caption_encoding.append(self.vocab.stoi["<SOS>"])
        # caption_encoding.append(self.vocab.encode(caption))
        # caption_encoding.append(self.vocab.stoi["<EOS>"])

        return img, torch.tensor(caption_encoding)

class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx
    
    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)

        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)

        return imgs, targets


data_location = "/Flicker8k"
# def get_dataloader(data_location,
#                    transform,
#                    batch_size=32,
#                    shuffle=True,
#                    pin_memory=True):


#     dataset = dataset = Flicker_Dataset(
#         root_dir = data_location+"/Images",
#         captions_file = data_location+"/captions.txt",
#         transform=transform
#     )

#     pad_idx = dataset.vocab.stoi["<PAD>"]

#     loader = DataLoader(
#         dataset=dataset,
#         batch_size=batch_size,
#         shuffle=shuffle,
#         pin_memory=pin_memory,
#         collate_fn=MyCollate(pad_idx)
#     )
#     return loader


