import numpy as np
from data_io.data_SST_2 import DataIOSST2
import argparse
import torch
from models.TextCNN import TextCNN
from layers.layer_word_cnn import  LayerWordCNN
# values = ['5','1.1','3.4',' ','6.6','9.44','3.44','0']
# emb_vector = list(map(lambda t: float(t),filter(lambda n: n and not n.isspace(), values[1:])))
# print(emb_vector)
#
# rand = np.random.uniform(-np.sqrt(3.0 / 4), np.sqrt(3.0 / 4), 4).tolist()
# print(rand)
# Training parameters
parser = argparse.ArgumentParser()
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
parser.add_argument('--gpu', type=int, default=-1)
args = parser.parse_args()
sentence1 = [[3,7,3,3],[6,2,5,4],[3,4,2,3]]
sentence2 = [[1,2,3,2],[1,9,3,2],[7,4,8,2]]

sentences = []
sentences.append(sentence1)
sentences.append(sentence2)
sentence1 = torch.FloatTensor(sentence1)
sentences = torch.FloatTensor(sentences)

batch_size = 2
filter_num = 3
leng_size = 4
zeros = torch.zeros(batch_size,filter_num,leng_size)
print(sentences)
sentences = sentences.view(batch_size,-1)
print(sentences)






# WordCNNLayer = LayerWordCNN(gpu=-1,word_embeddings_dim=4,filter_num=2,word_window_size=2)
#
# y  = WordCNNLayer(sentences)



