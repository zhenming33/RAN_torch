import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
from torch import nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence
from model import Model
from tensorboardX import SummaryWriter
import numpy as np
from tqdm import tqdm
from data_loader import dataset
import Levenshtein
from utils import *
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_steps = 50
train_steps_size = 1000
batch_size = 256
lr = 1e-3
image_channel = 3
alpha_c = 1.
grad_clip = 5.
valid_interval = 50
valid_steps = 1
save_interval = 1000

label_path = 'labels.txt'
vocab_path = 'radical_alphabet.txt'
dict_path = 'char2seq_dict.pkl'
word_map = open(vocab_path, encoding='utf-8').readlines()[0]
word_map = word_map + 'sep'
vocab_size = len(word_map)
save_dir = 'weights'
log_dir = 'logs/lr1e-3+batch256+edropout0.5+xvaier+data_shuff+grad_clip+lrdeacy'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# train step
def train_step(model, criterion, optimizer,images, encoded_captions, caption_lengths):
    model.train()
    optimizer.zero_grad()
    scores, caps_sorted, decode_lengths, alphas, sort_ind = model(images, encoded_captions, caption_lengths)
    targets = caps_sorted
    scores_pad, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
    targets_pad, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)
    loss = criterion(scores_pad, targets_pad)
    loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
    loss.backward()
    clip_gradient(model, grad_clip)
    # clip_gradient(optimizer, grad_clip)
    optimizer.step()

    return loss

def valid_step(model, criterion, images, encoded_captions, caption_lengths):
    with torch.no_grad():
        model.eval()
        scores, caps_sorted, decode_lengths, alphas, sort_ind = model(images, encoded_captions, caption_lengths)
        targets = caps_sorted
        scores_pad, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets_pad, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)
        loss = criterion(scores_pad, targets_pad)
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        pred_index = torch.argmax(scores[0], 1).cpu().numpy()
        preds = [word_map[c] for c in pred_index]
        label_index = caps_sorted[0, :max(decode_lengths)].cpu().numpy()
        labels = [word_map[c] for c in label_index]
        preds = ''.join(preds)
        labels = ''.join(labels)
        return loss, alphas, preds, labels, sort_ind



if __name__ == "__main__":
    # add records
    writer = SummaryWriter(log_dir)

    dataloader = data_gen(batch_size, dataset, label_path, vocab_path, dict_path, train_percent=0.7, num_workers=1)
    model = Model(image_channel, vocab_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss().to(device)
    global_step = 0

    # record_graph(writer, model)

    for i in range(train_steps):

        print('train steps ' + str(i))

        for k in tqdm(range(train_steps_size)):

            if global_step == 3000:
                adjust_learning_rate(optimizer, 0.1)
                lr = lr * 0.1

            record_scale(writer, lr, global_step, 'lr')

            images, encoded_captions, caption_lengths = dataloader.train()

            loss = train_step(model, criterion, optimizer, images, encoded_captions, caption_lengths)

            record_scale(writer, loss, global_step, 'train/loss')

            if global_step % valid_interval == 0:
                images, encoded_captions, caption_lengths = dataloader.valid()
                loss, alphas, preds, labels, sort_ind = valid_step(model, criterion, images, encoded_captions, caption_lengths)
                images = images[sort_ind]
                record_scale(writer, loss, global_step, 'valid/loss')
                # record_images(writer, images, global_step)
                # for t in range(max(caption_lengths).item()):
                #     record_attention(writer, alphas, t, global_step)
                # record_text(writer, preds, global_step, 'valid/preds')
                # record_text(writer, labels, global_step, 'valid/labels')
                edit_distance = Levenshtein.distance(preds,labels)
                normalized_edit_distance = edit_distance / max(len(preds), len(labels))
                record_scale(writer, normalized_edit_distance, global_step, 'valid/N.E.D')

            if global_step % save_interval == 0:
                torch.save(model, save_dir + '/' + str(global_step) + '.pth')

            global_step = global_step + 1

    writer.close()









