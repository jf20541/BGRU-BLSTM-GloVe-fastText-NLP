import numpy as np
from keras.preprocessing.text import Tokenizer


class GloVeEmbedding:
    def __init__(self, dataframe, glove_params):
        self.dataframe = dataframe
        self.glove = glove_params

    def create_embedding_matrix(self, word_index, embedding_dict=None, d_model=100):
        embedding_matrix = np.zeros((len(word_index) + 1, d_model))
        for word, index in word_index.items():
            if word in embedding_dict:
                embedding_matrix[index] = embedding_dict[word]
        return embedding_matrix

    def embedding_matrix(self, glove_embedding):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(self.dataframe.values.tolist())
        return self.create_embedding_matrix(
            tokenizer.word_index, embedding_dict=glove_embedding
        )
