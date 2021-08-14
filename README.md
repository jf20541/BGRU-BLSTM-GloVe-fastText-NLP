# GRU-Sentiment-NLP


## Objective


## Output 
```bash
                               ....                   ....                    ....
Epoch:15/20, Train Accuracy: 93.20%, Eval Accuracy: 88.00%, Eval Precision: 0.8831
Epoch:16/20, Train Accuracy: 93.39%, Eval Accuracy: 88.07%, Eval Precision: 0.8850
Epoch:17/20, Train Accuracy: 93.51%, Eval Accuracy: 88.10%, Eval Precision: 0.8846
Epoch:18/20, Train Accuracy: 93.65%, Eval Accuracy: 89.82%, Eval Precision: 0.8885
Epoch:19/20, Train Accuracy: 93.74%, Eval Accuracy: 89.87%, Eval Precision: 0.8989
Epoch:20/20, Train Accuracy: 94.96%, Eval Accuracy: 90.99%, Eval Precision: 0.8970
```

## Repository File Structure
    ├── src          
    │   ├── train.py             # Training Bidirectional GRU and evulating metrics 
    │   ├── model.py             # Bidirectional Gated Recurrent Unit (GRU) architecture, inherits nn.Module
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
GRU(
  (embedding): Embedding(180446, 100)
  (lstm): GRU(100, 128, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
  (out): Linear(in_features=512, out_features=1, bias=True)
  (sigmoid): Sigmoid()
)
```  
