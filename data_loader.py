# -*- coding: utf-8 -*-
import numpy as np
import cv2
import os
import torch
import pickle
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random

class dataset(Dataset):
    def __init__(self, label_path, vocab_path, dict_path, is_train=True, train_percent=0.7):
        real_labels = open(label_path, encoding='utf-8').readlines()
        with open(dict_path, 'rb') as f:
            label_sequence = pickle.load(f)
        vocab = open(vocab_path, encoding='utf-8').readlines()[0]
        radical_vocab_size = len(vocab)
        # cal the max len of all captions
        max_len = 0
        for v in label_sequence.values():
            if len(v) > max_len:
                max_len = len(v)
        # print(max_len)
        # label max len = max len + 2
        label_max_len = max_len + 2


        train_valid_split = int((len(real_labels)*train_percent))
        random.shuffle(real_labels)
        if is_train:
            self.labels = real_labels[:train_valid_split]
        else:
            self.labels = real_labels[train_valid_split:]
        self.label_sequence = label_sequence
        self.radical_vocab_size = radical_vocab_size
        self.label_max_len = label_max_len

    def __getitem__(self, index):
        label = self.labels[index]
        real_label = label.replace('\n', '').split(' ')[-1]
        image_path = label.replace('\n', '').split(' ')[0]

        image = cv2.resize(cv2.imread(image_path), (224, 224))
        image = image / 255.
        image = image.reshape(3, 224, 224)
        # process label
        label = self.label_sequence[real_label]
        # labels_len = actual label length + <start> + <end> =  len(label) + 2
        label_len = len(label) + 2
        # padding label with <start>, <end> and <pad>
        # label index = actual label index + 3
        label = [int(c) for c in label]
        # add <start> : 473  at begining
        label = [self.radical_vocab_size] + label
        # add <end> : 474 at end
        label = label + [self.radical_vocab_size + 1]
        # add <pad> : 475
        while len(label) < self.label_max_len:
            label.append(self.radical_vocab_size + 2)

        return image, np.array(label), np.array(label_len)

    def __len__(self):
        return len(self.labels)


if __name__ == "__main__":
    train_data = dataset('data/labels.txt', 'radical_alphabet.txt', 'char2seq_dict.pkl')
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    image, label, label_len = train_loader.__iter__().__next__()
    print(label_len)
    print(label_len.size())
































