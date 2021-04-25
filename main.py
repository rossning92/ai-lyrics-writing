#!/usr/bin/env python3

import datetime
import glob
import os
import sys
from pprint import pprint

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter

debug = False

embed_size = 128
hidden_size = 1024
lr = 0.001
lstm_layers = 2
batch_size = 32
epochs = 15
seq_len = 48

if debug:
    batch_size = 2
    epochs = 1000


class LyricsDataset(Dataset):
    def __init__(self, seq_len, file="data/lyrics.txt"):
        SOS = 0  # start of song
        EOS = 1  # end of song

        self.seq_len = seq_len
        with open(file, encoding="utf-8") as f:
            lines = f.read().splitlines()

        self.word2index = {"<SOS>": SOS, "<EOS>": EOS}

        # Convert words to indices
        indices = []
        num_words = 0
        for line in lines:
            indices.append(SOS)
            for word in line:
                if word not in self.word2index:
                    self.word2index[word] = num_words
                    num_words += 1
                indices.append(self.word2index[word])
            indices.append(EOS)

        self.index2word = {v: k for k, v in self.word2index.items()}
        self.data = np.array(indices, dtype=np.int64)

    def __len__(self):
        return (len(self.data) - 1) // self.seq_len

    def __getitem__(self, i):
        start = i * self.seq_len
        end = start + self.seq_len
        return (
            torch.as_tensor(self.data[start:end]),  # input
            torch.as_tensor(self.data[start + 1 : end + 1]),  # output
        )


class LyricsNet(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, lstm_layers):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, lstm_layers, batch_first=True)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, vocab_size)

    def forward(self, word_ids, lstm_hidden=None):
        # Embed word ids into vectors
        embedded = self.embedding(word_ids)

        # Forward propagate LSTM
        lstm_out, lstm_hidden = self.lstm(embedded, lstm_hidden)

        # Forward propagate linear layer
        out = self.h2h(lstm_out)

        # Decode hidden states to one-hot encoded words
        out = self.h2o(out)

        return out, lstm_hidden


def accuracy(output, target):
    """Compute the accuracy between model output and ground truth.

    Args:
        output: (batch_size, seq_len, vocab_size)
        target: (batch_size, seq_len)

    Returns:
        float: accuracy value between 0 and 1
    """
    output = output.reshape(-1, vocab_size)
    target = target.flatten()

    a = output.topk(1).indices.flatten()
    b = target
    return a.eq(b).sum().item() / len(a)


def generate(start_phrases):
    # Convert to a list of start words.
    # i.e. '宁可/无法' => ['宁可', '无法']
    start_phrases = start_phrases.split("/")

    hidden = None

    def next_word(input_word):
        nonlocal hidden

        input_word_index = dataset.word2index[input_word]
        input_ = torch.Tensor([[input_word_index]]).long().to(device)
        output, hidden = model(input_, hidden)
        top_word_index = output[0].topk(1).indices.item()
        return dataset.index2word[top_word_index]

    result = []  # a list of output words
    cur_word = "/"

    for i in range(seq_len):
        if cur_word == "/":  # end of a sentence
            result.append(cur_word)
            next_word(cur_word)

            if len(start_phrases) == 0:
                break

            for w in start_phrases.pop(0):
                result.append(w)
                cur_word = next_word(w)

        else:
            result.append(cur_word)
            cur_word = next_word(cur_word)

    # Convert a list of generated words to a string
    result = "".join(result)
    result = result.strip("/")  # remove trailing slashes
    return result


def training_step():
    for i, (input_, target) in enumerate(train_loader):
        model.train()

        input_, target = input_.to(device), target.to(device)

        output, _ = model(input_)
        loss = F.cross_entropy(output.reshape(-1, vocab_size), target.flatten())

        optimizer.zero_grad()  # Make sure gradient does not accumulate
        loss.backward()  # Compute gradient
        optimizer.step()  # Update NN weights

        acc = accuracy(output, target)

        print(
            "Training: Epoch=%d, Batch=%d/%d, Loss=%.4f, Accuracy=%.4f"
            % (epoch, i, len(train_loader), loss.item(), acc)
        )

        if not debug:
            step = epoch * len(train_loader) + i
            writer.add_scalar("loss/training", loss.item(), step)
            writer.add_scalar("accuracy/training", acc, step)

            if i % 50 == 0:
                generated_lyrics = generate("深/度/学/习")
                writer.add_text("generated_lyrics", generated_lyrics, i)
                writer.flush()


def evaluation_step():
    model.eval()

    epoch_loss = 0
    epoch_acc = 0

    with torch.no_grad():
        for data in test_loader:
            input_, target = data[0].to(device), data[1].to(device)

            output, _ = model(input_)
            loss = F.cross_entropy(output.reshape(-1, vocab_size), target.flatten())

            epoch_acc += accuracy(output, target)

            epoch_loss += loss.item()

    epoch_loss /= len(test_loader)
    epoch_acc /= len(test_loader)
    print(
        "Validation: Epoch=%d, Loss=%.4f, Accuracy=%.4f"
        % (epoch, epoch_loss, epoch_acc)
    )

    if not debug:
        writer.add_scalar("loss/validation", epoch_loss, epoch)
        writer.add_scalar("accuracy/validation", epoch_acc, epoch)
        writer.flush()


def save_checkpoint():
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        "checkpoint-%s.pth" % datetime.datetime.now().strftime("%y%m%d-%H%M%S"),
    )


def load_checkpoint(file):
    global epoch

    ckpt = torch.load(file)

    print("Loading checkpoint from %s." % file)

    model.load_state_dict(ckpt["model_state_dict"])

    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    epoch = ckpt["epoch"]


if __name__ == "__main__":
    # Create cuda device to train model on GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define dataset
    dataset = LyricsDataset(seq_len=seq_len)

    # Split dataset into training and validation
    data_length = len(dataset)
    lengths = [int(data_length - 1000), 1000]
    train_data, test_data = random_split(dataset, lengths)

    # Create data loader
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=True, num_workers=0
    )
    if debug:
        train_loader = [next(iter(train_loader))]
        test_loader = [next(iter(test_loader))]

    # Sanity check: view training data
    if False:
        i = 0
        for data in train_loader:
            if i >= 10:
                break
            input_batch, _ = data
            first_sample = input_batch[0]

            pprint("".join([dataset.index2word[x.item()] for x in first_sample]))
            i += 1

    # Create NN model
    vocab_size = len(dataset.word2index)
    model = LyricsNet(
        vocab_size=vocab_size,
        embed_size=embed_size,
        hidden_size=hidden_size,
        lstm_layers=lstm_layers,
    )
    model = model.to(device)

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Load checkpoint
    checkpoint_files = glob.glob("checkpoint-*.pth")
    if (
        not debug
        and len(checkpoint_files) > 0
        and input("Enter y to load %s: " % checkpoint_files[-1]) == "y"
    ):
        load_checkpoint(checkpoint_files[-1])
    else:
        epoch = 0

    if (
        input("Enter y to enter inference mode, anything else to enter training mode: ")
        == "y"
    ):
        # Inference loop
        while True:
            start_words = input("Enter start-words divided by '/' (e.g. '深/度/学/习'): ")
            if not start_words:
                break

            print(generate(start_words))

    else:
        if not debug:
            writer = SummaryWriter()

        # Optimization loop
        while epoch < epochs:
            training_step()
            evaluation_step()

            if not debug:
                save_checkpoint()

            epoch += 1
