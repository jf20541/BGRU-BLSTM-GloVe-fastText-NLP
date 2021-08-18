# BGRU-BLSTM-Sentiment-NLP

## Objective
Predict a binary NLP sentiment classification for the IMDB dataset with 50,000 reviews with an evenly distributed target values **[1:Positive & 2:Negative]** using a Bidirectional **Gated Recurrent Unit** and Bidirectional **Long-Short-Term Memory**. Feature Engineer the reviews by cleaning, removing stop-words, tokenizing before obtaining a vector representation for each token using **GloVe pre-trained word embeddings**. Measure GRU performance with **accuracy score** since the target values are evenly distributed. Bi-GRU outperformed Bi-LSTM with 90.99% and 86.11%, respectively. 

## Output 
**Bi-Directional GRU**
```
                               ....                   ....                    ....
Epoch:15/20, Train Accuracy: 93.20%, Eval Accuracy: 88.00%, Eval Precision: 0.8831
Epoch:16/20, Train Accuracy: 93.39%, Eval Accuracy: 88.07%, Eval Precision: 0.8850
Epoch:17/20, Train Accuracy: 93.51%, Eval Accuracy: 88.10%, Eval Precision: 0.8846
Epoch:18/20, Train Accuracy: 93.65%, Eval Accuracy: 89.82%, Eval Precision: 0.8885
Epoch:19/20, Train Accuracy: 93.74%, Eval Accuracy: 89.87%, Eval Precision: 0.8989
Epoch:20/20, Train Accuracy: 94.96%, Eval Accuracy: 90.99%, Eval Precision: 0.8970
```

**Bi-Directional LSTM**
```
                               ....                   ....                    ....
Epoch:15/20, Train Accuracy: 88.42%, Eval Accuracy: 85.98%, Eval Precision: 0.8911
Epoch:16/20, Train Accuracy: 88.54%, Eval Accuracy: 86.13%, Eval Precision: 0.8897
Epoch:17/20, Train Accuracy: 88.64%, Eval Accuracy: 86.09%, Eval Precision: 0.8909
Epoch:18/20, Train Accuracy: 88.85%, Eval Accuracy: 86.23%, Eval Precision: 0.8838

Epoch    19: reducing learning rate of group 0 to 2.7000e-05.

Epoch:19/20, Train Accuracy: 88.95%, Eval Accuracy: 86.30%, Eval Precision: 0.8864
Epoch:20/20, Train Accuracy: 89.11%, Eval Accuracy: 86.11%, Eval Precision: 0.8933
```
## Repository File Structure
    ├── src          
    │   ├── train.py             # Training Bidirectional GRU and evaluating metric (accuracy & precision) 
    │   ├── bigru_model.py       # Bidirectional Gated Recurrent Unit (GRU) architecture, inherits nn.Module
    │   ├── bilstm_model.py      # Bidirectional Long-Short Term Memory (LSTM) architecture, inherits nn.Module
    │   ├── engine.py            # Class Engine for Training, Evaluation, and Loss function 
    │   ├── dataset.py           # Custom Dataset that return a paris of [input, label] as tensors
    │   ├── embeddings.py        # GloVe Embeddings
    │   ├── data.py              # Cleaning dataset by removing stopwords and special characters, labeled target values
    │   └── config.py            # Define path as global variable
    ├── inputs
    │   ├── clean_train.csv      # Cleaned Data and Featured Engineered with re
    │   └── train.csv            # Kaggle IMDB Dataset 
    ├── models
    │   └── imdb_model.bin       # GRU parameters saved into model.bin 
    ├── static
    │   └── style.css            #  
    ├── templates
    │   └── index.html           # 
    ├── requierments.txt         # Packages used for project
    └── README.md

## Model's Architecture
```
BIGRU(
  (embedding): Embedding(180446, 100)
  (gru): BIGRU(100, 128, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
  (out): Linear(in_features=512, out_features=1, bias=True)
  (sigmoid): Sigmoid()
)



BILSTM(
  (embedding): Embedding(180446, 100)
  (lstm): BILSTM(100, 128, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
  (out): Linear(in_features=512, out_features=1, bias=True)
  (sigmoid): Sigmoid()
)
```  

## GPU's Menu Accelerator
```
Sat Aug 14 00:31:06 2021       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.42.01    Driver Version: 460.32.03    CUDA Version: 11.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla P100-PCIE...  Off  | 00000000:00:04.0 Off |                    0 |
| N/A   39C    P0    33W / 250W |   2047MiB / 16280MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
```
