from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
from bigru_model import BIGRU
from bilstm_model import BILSTM
from embeddings import GloVeEmbedding
import pandas as pd
import config
from keras.preprocessing.text import Tokenizer

app = Flask(__name__)
MODEL = None


def sentence_prediction(sentence, model):
    review = str(sentence)
    reviews = torch.tensor(review[idx, :], dtype=torch.long).unsqueeze(0)
    reviews = reviews.to(config.DEVICE, dtype=torch.long)
    outputs = model(reviews).cpu().detach().numpy().tolist()
    return outputs


@app.route("/predict")
def predict():
    sentence = request.args.get("sentence")
    positive_prediction = sentence_prediction(sentence, model=MODEL)
    negative_prediction = 1 - positive_prediction

    response = {}
    response["response"] = {
        "positive": str(positive_prediction),
        "negative": str(negative_prediction),
        "sentence": str(sentence),
    }
    return jsonify(response)


# @app.route("/", methods=["GET", "POST"])
# def predict():
#     return render_template("index.html")

sentence = "I love this movie"
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentence.values.tolist())

glove = pd.read_csv(config.GLOVE_PARAMS, sep=" ", quoting=3, header=None, index_col=0)
glove_embedding = {key: val.values for key, val in glove.T.items()}

emb = GloVeEmbedding(sentence, glove)
embedding_matrix = emb.embedding_matrix(glove_embedding)


if __name__ == "__main__":
    MODEL = BIGRU(
        embedding_matrix,
        embedding_matrix.shape[0],
        embedding_matrix.shape[1],
        128,
        1,
        2,
        0.2,
    )
    MODEL.load_state_dict(torch.load(config.MODEL_PATH))
    MODEL.to(config.DEVICE)
    MODEL.eval()
    app.run()
