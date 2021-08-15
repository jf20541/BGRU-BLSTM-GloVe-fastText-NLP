from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn


# function to make sentiment prediction
# find a way to save the sentence
# show the models result

app = Flask(__name__)
MODEL = None
DEVICE = 'cuda'

def sentence_prediction(sentence):
    review = str(sentence)
    
    
    # convert each features, sentiment to tensors

    reviews =  torch.tensor(review[idx, :], dtype=torch.long).unsqueeze(0)
    reviews = reviews.to(DEVICE, dtype=torch.long)
    targets = targets.to(DEVICE, dtype=torch.float)
    outputs = MODEL(reviews)
    return outputs

@app.route("/predict")
def predict():
    sentence = request.args.get('sentence')
    positive_prediction = sentence_prediction(sentence)
    negative_prediction = 1 - positive_prediction
    
    response = {}
    response['response'] = {
        'positive': positive_prediction,
        'negative': negative_prediction,
        'sentence': sentence
    }
    return jsonify(response)

# @app.route("/", methods=["GET", "POST"])
# def predict():
#     return render_template("index.html")


# class GRU(nn.Module):
#     def __init__(
#         self,
#         embedding_matrix,
#         vocab_size,
#         embedding_dim,
#         hidden_dim,
#         output_dim,
#         n_layers,
#         dropout,
#     ):
#         super(GRU, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, embedding_dim)
#         self.embedding.weight = nn.Parameter(
#             torch.tensor(embedding_matrix, dtype=torch.float32)
#         )
#         self.embedding.weight.requires_grad = False
#         self.lstm = nn.GRU(
#             embedding_dim,
#             hidden_dim,
#             n_layers,
#             dropout=dropout,
#             bidirectional=True,
#             batch_first=True,
#         )
#         self.out = nn.Linear(hidden_dim * 4, output_dim)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         x = self.embedding(x)
#         h0, _ = self.lstm(x)
#         avg_pool = torch.mean(h0, 1)
#         max_pool, _ = torch.max(h0, 1)
#         out = torch.cat((avg_pool, max_pool), 1)
#         out = self.out(out)
#         return self.sigmoid(out)


if __name__ == "__main__":
    MODEL = 
    
    app.run(port=12000, debug=True)
