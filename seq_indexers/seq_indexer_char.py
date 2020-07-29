import torch
import numpy as np
from seq_indexers.seq_indexer_embedding_base import SeqIndexerBaseEmbeddings



class SeqIndexerBaseChar(SeqIndexerBaseEmbeddings):
    """SeqIndexerBaseChar converts list of lists of characters to list of lists of integer indices and back."""
    def __init__(self, name ,embedding_path, emb_dim, emb_delimiter):
        SeqIndexerBaseEmbeddings.__init__(self,name = name,embedding_path = embedding_path, emb_dim = emb_dim, emb_delimiter = emb_delimiter)

    def add_char(self, c):
        if isinstance(c, (list, tuple)):
            return [self.add_char(elem) for elem in c]
        ## Add the char to the instance(also a random generated emd_vector) if it hasn't.
        if c not in self.__index2instance:
            self.add_instance(c)
            self.add_emb_vector(self.generate_random_emb_vector())

    def get_loaded_embeddings_tensor(self):
        return torch.FloatTensor(np.asarray(self.embedding_vectors_list))
