# BiGRU-Sentiment-NLP

## Objective
Predict a binary NLP sentiment classification for the IMDB dataset with 50,000 reviews with an evenly distributed target values **[1:Positive & 2:Negative]** using a **Gated Recurrent Unit**. Feature Engineer the reviews by cleaning, removing stop-words, tokenizing before obtaining a vector representation for each token using **GloVe pre-trained word embeddings**. Measure GRU performance with **accuracy score** since the target values are evenly distributed. 

## Output 
**Bi-Direction GRU**
```
                               ....                   ....                    ....
Epoch:15/20, Train Accuracy: 93.20%, Eval Accuracy: 88.00%, Eval Precision: 0.8831
Epoch:16/20, Train Accuracy: 93.39%, Eval Accuracy: 88.07%, Eval Precision: 0.8850
Epoch:17/20, Train Accuracy: 93.51%, Eval Accuracy: 88.10%, Eval Precision: 0.8846
Epoch:18/20, Train Accuracy: 93.65%, Eval Accuracy: 89.82%, Eval Precision: 0.8885
Epoch:19/20, Train Accuracy: 93.74%, Eval Accuracy: 89.87%, Eval Precision: 0.8989
Epoch:20/20, Train Accuracy: 94.96%, Eval Accuracy: 90.99%, Eval Precision: 0.8970
```
**Logistic Regression with Bag of Words**
```
Fold: 0, Accuracy = 88.49%
Fold: 1, Accuracy = 89.09%
Fold: 2, Accuracy = 88.38%
Fold: 3, Accuracy = 89.53%
Fold: 4, Accuracy = 89.31%
Accuracy Mean = 88.96%

```
**Logistic Regression with Bag of Words**
```
Fold: 0, Accuracy = 84.81%
Fold: 1, Accuracy = 84.81%
Fold: 2, Accuracy = 84.51%
Fold: 3, Accuracy = 84.96%
Fold: 4, Accuracy = 84.80%
Accuracy Mean = 84.77%
```

## Repository File Structure
    ├── src          
    │   ├── train.py             # Training Bidirectional GRU and evaluating metric (accuracy & precision) 
    │   ├── model.py             # Bidirectional Gated Recurrent Unit (GRU) architecture, inherits nn.Module
    │   ├── engine.py            # Class Engine for Training, Evaluation, and Loss function 
    │   ├── dataset.py           # Custom Dataset that return a paris of [input, label] as tensors
    │   ├── embeddings.py        # GloVe Embeddings
    │   ├── data.py              # Cleaning dataset by removing stopwords and special characters, labeled target values
    │   ├── logistic_reg.py      # Initialized Logistic Regression with Bag of Words and evaluated metric
    │   ├── naivebayes.py        # Initialized Naive Bayes Classifier for multinomial models with Bag of Words and evaluated metric
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
GRU(
  (embedding): Embedding(180446, 100)
  (lstm): GRU(100, 128, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
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
