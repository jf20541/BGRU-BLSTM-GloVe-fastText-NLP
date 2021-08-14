import numpy as np
from keras.preprocessing.text import Tokenizer


class GloVeEmbedding:
    def __init__(self, dataframe, glove_params):
        self.dataframe = dataframe
        self.glove = glove_params

    def create_embedding_matrix(self, word_index, embedding_dict=None, d_model=100):
        """Creates the embedding matrix save in numpy array
        Args:
            word_index (dict): dictionary with tokens
            embedding_dict (dict, optional): dict with word embedding
            d_model (int): dimension of word pretrained embedding (Defaults to 100) Glove embedding is 100
        Returns:
            [array]: array with embedding vectors for all known words
        """
        embedding_matrix = np.zeros((len(word_index) + 1, d_model))
        for word, index in word_index.items():
            if word in embedding_dict:
                embedding_matrix[index] = embedding_dict[word]
        return embedding_matrix

    def embedding_matrix(self, glove_embedding):
        # tokenize review words
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(self.dataframe.values.tolist())
        return self.create_embedding_matrix(
            tokenizer.word_index, embedding_dict=glove_embedding
        )
