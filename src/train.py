import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
from sklearn.model_selection import train_test_split
import torch
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from dataset import IMDBDataset
from bilstm_model import BILSTM
from bigru_model import BIGRU
from engine import Engine
from embeddings import GloVeEmbedding
import config

# use device as global variable
device = torch.device("cuda")


def train():
    df = pd.read_csv(config.TRAINING_FILE_CLEAN)
    # class allows to vectorize a text corpus
    tokenizer = Tokenizer()
    # updates internal vocabulary based on a list of sequences
    tokenizer.fit_on_texts(df.review.values.tolist())
    # read the vector representations for words
    glove = pd.read_csv(
        config.GLOVE_PARAMS, sep=" ", quoting=3, header=None, index_col=0
    )
    # load and access a word vectors
    glove_embedding = {key: val.values for key, val in glove.T.items()}

    # define features and target values
    targets = df[["sentiment"]]
    features = df["review"]

    # hold-out based validation 80% training and 20% testing set
    x_train, x_test, y_train, y_test = train_test_split(
        features, targets, test_size=0.2, stratify=targets
    )
    # only most frequent words will be taken into account
    x_train = tokenizer.texts_to_sequences(x_train.values)
    x_test = tokenizer.texts_to_sequences(x_test.values)

    # transforms a list of sequencesinto a 2D Numpy array of shape (num_samples, maxlen)
    x_train = tf.keras.preprocessing.sequence.pad_sequences(
        x_train, maxlen=config.MAX_LENGTH
    )
    x_test = tf.keras.preprocessing.sequence.pad_sequences(
        x_test, maxlen=config.MAX_LENGTH
    )
    # define target values in arrays
    y_train = y_train.sentiment.values
    y_test = y_test.sentiment.values

    # initialize custom dataset
    train_dataset = IMDBDataset(x_train, y_train)
    test_dataset = IMDBDataset(x_test, y_test)

    # initialize dataloader from custom dataset and defined batch size for training/testing set
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.TEST_BATCH_SIZE
    )

    # initialize GloVeEmbedding Class
    emb = GloVeEmbedding(df.review, glove)
    embedding_matrix = emb.embedding_matrix(glove_embedding)

    # initialize GRU model with defined parameters
    # embedding_matrix (rows, dims), hidden size, num of layers, and dropout respectivaly
    model = BILSTM(
        embedding_matrix,
        embedding_matrix.shape[0],
        embedding_matrix.shape[1],
        128,
        1,
        2,
        0.2,
    )

    model.to(device)

    # initialize Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    # adjust the learning rate based on the number of epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.3, verbose=True
    )
    # initialize Engine class with model, optimizer, and device
    eng = Engine(model, optimizer, device)

    for epochs in range(config.EPOCHS):
        # initiating training and evaluation function
        train_targets, train_outputs = eng.train_fn(train_loader, scheduler)
        eval_targets, eval_outputs = eng.eval_fn(test_loader)

        # binary classifier
        eval_outputs = np.array(eval_outputs) >= 0.5
        train_outputs = np.array(train_outputs) >= 0.5

        # calculating accuracy score and precision score
        train_metric = accuracy_score(train_targets, train_outputs)
        eval_metric = accuracy_score(eval_targets, eval_outputs)
        prec_score = precision_score(eval_targets, eval_outputs)
        print(
            f"Epoch:{epochs+1}/{config.EPOCHS}, Train Accuracy: {train_metric:.2f}%, Eval Accuracy: {eval_metric:.2f}%, Eval Precision: {prec_score:.4f}"
        )
        print(confusion_matrix(eval_targets, eval_outputs))
        # save Bi-GRU's parameters
        torch.save(model.state_dict(), config.MODEL_PATH)


if __name__ == "__main__":
    train()
