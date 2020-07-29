"""class implements word-level convolutional 1D layer"""
import torch
import torch.nn as nn
from layers.layer_base import LayerBase

class LayerWordCNN(LayerBase):
    """LayerWordCNN implements word-level convolutional 1D layer.
    (1d conv is often used in the text classification)"""
    def __init__(self, gpu, word_embeddings_dim, filter_num, word_window_size):
        super(LayerWordCNN, self).__init__(gpu)
        self.word_embeddings_dim = word_embeddings_dim
        self.word_cnn_filter_num = filter_num
        self.word_window_size = word_window_size


        self.output_dim = word_embeddings_dim * filter_num
        self.conv1d = nn.Conv1d(in_channels = word_embeddings_dim,
                                out_channels = filter_num,
                                kernel_size = word_window_size)

    def is_cuda(self):
        return self.conv1d.weight.is_cuda

    def forward(self, word_embeddings_feature):  # batch_num x max_seq_len x word_embeddings_dim
        batch_num, max_seq_len, word_embeddings_dim = word_embeddings_feature.shape
        word_embeddings_feature = word_embeddings_feature.permute(0,2,1)
        max_pooling_out , _ = torch.max(self.conv1d(word_embeddings_feature), dim=2)
        return max_pooling_out  # shape: batch_num x 1 x filter_num
