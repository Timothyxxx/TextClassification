import torch
import torch.nn.functional as F
from seq_indexers.seq_indexer_embedding_base import SeqIndexerBaseEmbeddings
from layers.layer_word_embeddings import LayerWordEmbeddings


class MLP(torch.nn.Module):
    def __init__(self, embedding_indexer: SeqIndexerBaseEmbeddings, gpu, feat_num):
        super(MLP, self).__init__()
        self.embeding = LayerWordEmbeddings(embedding_indexer, gpu)
        self.linear1 = torch.nn.Linear(embedding_indexer.emb_dim, 50)
        self.linear2 = torch.nn.Linear(50, feat_num)
        self.act_func = torch.nn.LeakyReLU()

        if(gpu >=0):
            self.cuda(device=gpu)

    def forward(self, words, lens : torch.Tensor):
        words = self.embeding(words)
        words = torch.sum(words, dim=1, keepdim=False) / lens.unsqueeze(-1)
        words = self.linear1(words)
        words = self.act_func(words)
        words = self.linear2(words)
        words = self.act_func(words)
        return words
