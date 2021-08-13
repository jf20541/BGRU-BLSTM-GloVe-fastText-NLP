import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from dataset import IMDBDataset
from model import GRU
from engine import Engine
from embeddings import GloVeEmbedding
import config

device = torch.device("cuda")


def train():
    df = pd.read_csv(config.TRAINING_FILE_CLEAN)
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(df.review.values.tolist())
    glove = pd.read_csv(
        config.GLOVE_PARAMS, sep=" ", quoting=3, header=None, index_col=0
    )
    glove_embedding = {key: val.values for key, val in glove.T.items()}

    targets = df[["sentiment"]]
    features = df["review"]

    x_train, x_test, y_train, y_test = train_test_split(
        features, targets, test_size=0.2
    )

    x_train = tokenizer.texts_to_sequences(x_train.values)
    x_test = tokenizer.texts_to_sequences(x_test.values)
    x_train = tf.keras.preprocessing.sequence.pad_sequences(
        x_train, maxlen=config.MAX_LENGTH
    )
    x_test = tf.keras.preprocessing.sequence.pad_sequences(
        x_test, maxlen=config.MAX_LENGTH
    )

    y_train = y_train.sentiment.values
    y_test = y_test.sentiment.values

    train_dataset = IMDBDataset(x_train, y_train)
    test_dataset = IMDBDataset(x_test, y_test)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.TEST_BATCH_SIZE
    )

    emb = GloVeEmbedding(df.review, glove)
    embedding_matrix = emb.embedding_matrix(glove_embedding)

    global model

    model = GRU(
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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # reduce learning rate when a metric has stopped improving.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.3, verbose=True
    )

    eng = Engine(model, optimizer, device)

    for epochs in range(config.EPOCHS):
        # initiating training and evaluation function
        train_targets, train_outputs = eng.train_fn(train_loader)

        eval_targets, eval_outputs = eng.eval_fn(test_loader)

        eval_outputs = np.array(eval_outputs) >= 0.5
        train_outputs = np.array(train_outputs) >= 0.5

        # calculating accuracy score
        train_metric = accuracy_score(train_targets, train_outputs)
        scheduler.step(train_metric)
        eval_metric = accuracy_score(eval_targets, eval_outputs)
        prec_score = precision_score(eval_targets, eval_outputs)
        print(
            f"Epoch:{epochs+1}/{config.EPOCHS}, Train Accuracy: {train_metric:.2f}%, Eval Accuracy: {eval_metric:.2f}%, Eval Precision: {prec_score:.4f}"
        )
        print(confusion_matrix(eval_targets, eval_outputs))
        torch.save(model.state_dict(), config.MODEL_PATH)


if __name__ == "__main__":
    train()
