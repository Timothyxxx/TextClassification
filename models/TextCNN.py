import torch
import torch.nn.functional as F
from seq_indexers.seq_indexer_embedding_base import SeqIndexerBaseEmbeddings
from layers.layer_word_embeddings import LayerWordEmbeddings
from layers.layer_word_cnn import  LayerWordCNN

class TextCNN(torch.nn.Module):
    def __init__(self, embedding_indexer: SeqIndexerBaseEmbeddings, gpu, feat_num, filter_window_sizes):
        """
        :param embedding_indexer: the seq_indexer which could give the index /embedding vectors of a word
        :param gpu: whether to work on gpu for higher speed
        :param feat_num: the numbers of different classification you want to have.
        :param filer_window_sizes: a list contains the size of the different filters.
        """

        super(TextCNN, self).__init__()
        self.gpu = gpu
        self.filter_window_sizes = filter_window_sizes
        self.filter_groups = len(filter_window_sizes)
        self.embeding = LayerWordEmbeddings(embedding_indexer, gpu)
        self.filter_num = 2


        self.convs = []
        for filter_window_size in filter_window_sizes:
            self.convs.append(LayerWordCNN(gpu = self.gpu,word_embeddings_dim = embedding_indexer.emb_dim,filter_num=self.filter_num,
                                           word_window_size=filter_window_size))

        self.linear1 = torch.nn.Linear(self.filter_num * self.filter_groups, 50)
        self.linear2 = torch.nn.Linear(50, feat_num)
        self.act_func = torch.nn.LeakyReLU()


        if(gpu >=0):
            self.cuda(device=gpu)

    def forward(self, words):
        batch_size, max_sequence_num = words.shape
        if (self.gpu > 0):
            padded_text = padded_text.cuda()
            sorted_label = sorted_label.cuda()

        words = self.embeding(words)
        words_re = torch.zeros(batch_size ,self.filter_groups,self.filter_num)

        i = 0
        for conv in self.convs:
            words_re[:,i,:] = conv(words)
            i+=1

        words = words_re.view(batch_size,-1)

        words = self.linear1(words)
        words = self.act_func(words)
        words = self.linear2(words)
        words = self.act_func(words)
        return words