import torch
import json
import random
import argparse
import numpy as np
import os
from data_io.data_SST_2 import DataIOSST2
from seq_indexers.seq_indexer_embedding_base import SeqIndexerBaseEmbeddings
from seq_indexers.seq_indexer_base import SeqIndexerBase
from models.DAN import MLP
from models.TextCNN import TextCNN
from tqdm import tqdm

parser = argparse.ArgumentParser()

# Training parameters
parser.add_argument("--data_dir", "-dd", type=str, default='data/SST-2')
parser.add_argument("--save_dir", '-sd', type=str, default='save')
parser.add_argument('--load', type=str, default="")
parser.add_argument('--embedding_dir', "-ed", type=str, default='data/glove/glove.6B.100d.txt')
parser.add_argument('--embedding_dim', type=int, default=100)
parser.add_argument('--random_state', '-rs', type=int, default=0)
parser.add_argument('--num_epoch', '-ne', type=int, default=100)
parser.add_argument('--batch_size', '-bs', type=int, default=16)
parser.add_argument('--dropout-rate', '-dr', type=float, default=0.4)
parser.add_argument('--learning-rate', '-lr', type=float, default=0.001)
parser.add_argument('--gpu', type=int, default=1)
# model parameters

if __name__ == "__main__":
    args = parser.parse_args()

    # Fix the random seed of package random.
    random.seed(args.random_state)
    np.random.seed(args.random_state)

    # Fix the random seed of Pytorch when using GPU.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_state)
        torch.cuda.manual_seed(args.random_state)

    # Fix the random seed of Pytorch when using CPU.
    torch.manual_seed(args.random_state)
    torch.random.manual_seed(args.random_state)

    dataset = DataIOSST2(args)
    train_loader, dev_loader = dataset.get_data_loader("1")
    train_word, train_label, dev_word, dev_label, test_word = dataset.read_train_dev_test()

    seq_indexer = SeqIndexerBaseEmbeddings("glove", args.embedding_dir, args.embedding_dim, ' ')
    seq_indexer.load_embeddings_from_file()

    label_indexer = SeqIndexerBase("label", False, False)
    label_indexer.add_instance(train_label)

    # model = MLP(embedding_indexer=seq_indexer, gpu=args.gpu, feat_num=label_indexer.__len__())
    model = TextCNN(embedding_indexer=seq_indexer, gpu=args.gpu, feat_num=label_indexer.__len__(),filter_window_sizes=[2,3,4])
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                                 amsgrad=False)

    for epoch in range(args.num_epoch):
        train_loss = 0.0
        for x, y in tqdm(train_loader):
            padded_text, [sorted_label], seq_lens = dataset.add_padding(
                x, [(y, False)]
            )
            ## Transfer the texts and the labels --->index---->tensor
            padded_text = seq_indexer.get_index(padded_text)
            sorted_label = label_indexer.get_index(sorted_label)
            padded_text = torch.LongTensor(padded_text)
            sorted_label = torch.LongTensor(sorted_label)
            seq_lens = torch.LongTensor(seq_lens)

            if (args.gpu > 0):
                padded_text = padded_text.cuda()
                sorted_label = sorted_label.cuda()
                seq_lens = seq_lens.cuda()

            # y = model(padded_text, seq_lens)
            y = model(padded_text)
            loss = criterion(y, sorted_label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            optimizer.zero_grad()

        print(': ', train_loss / (len(train_word) / args.batch_size))
        allNum = 0
        correct = 0
        for x, y in dev_loader:
            padded_text, [sorted_label], seq_lens = dataset.add_padding(
                x, [(y, False)]
            )
            with torch.no_grad():
                padded_text = seq_indexer.get_index(padded_text)
                sorted_label = label_indexer.get_index(sorted_label)
                padded_text = torch.LongTensor(padded_text)
                sorted_label = torch.LongTensor(sorted_label)
                if (args.gpu > 0):
                    padded_text = padded_text.cuda()
                    sorted_label = sorted_label.cuda()
                y = model(padded_text, sorted_label)
                _, predict = torch.max(y.data, dim=1)
                allNum += y.size(0)
                correct += (predict == sorted_label).sum().item()

        print('accuracy ', 1.0 * correct / allNum)
