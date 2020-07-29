from seq_indexers.seq_indexer_base import SeqIndexerBase
import numpy as np
import torch


class SeqIndexerBaseEmbeddings(SeqIndexerBase):
    def __init__(self, name, embedding_path, emb_dim, emb_delimiter):
        super(SeqIndexerBaseEmbeddings, self).__init__(name=name, if_use_pad=True, if_use_unk=True)
        self.path = embedding_path
        self.embedding_vectors_list = list()
        self.emb_dim = emb_dim
        self.emb_delimiter = emb_delimiter

        ## Notice here that due to we set the if_use_pad and if_use_unk to true which means we added their
        ## signal to the instances ,so we have to also add their embeddings to the embedding vectors'list to
        # maintain the equality of the total-number between them
        #
        self.add_emb_vector(self.generate_zero_emb_vector())
        self.add_emb_vector(self.generate_random_emb_vector())

    def load_embeddings_from_file(self):
        """
        load embedding vectors from the file.

        :return:
        """
        for k, line in enumerate(open(self.path,encoding='utf-8')):
            values = line.split(self.emb_delimiter)
            self.add_instance(values[0])
            emb_vector = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), values[1:])))
            self.add_emb_vector(emb_vector)
            if(k%250000==0):
                print("read " + str(k) + " words")

    def generate_zero_emb_vector(self):
        if self.emb_dim == 0:
            raise ValueError('embeddings_dim is not known.')
        return [0 for _ in range(self.emb_dim)]

    def generate_random_emb_vector(self):
        if self.emb_dim == 0:
            raise ValueError('embeddings_dim is not known.')
        return np.random.uniform(-np.sqrt(3.0 / self.emb_dim), np.sqrt(3.0 / self.emb_dim),
                                 self.emb_dim).tolist()

    def add_emb_vector(self, emb_vector):
        self.embedding_vectors_list.append(emb_vector)

    def get_loaded_embeddings_tensor(self):
        return torch.FloatTensor(np.asarray(self.embedding_vectors_list))
