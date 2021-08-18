import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import config

if __name__ == "__main__":
    df = pd.read_csv(config.TRAINING_FILE_CLEAN_FOLDS)
    for fold in range(5):
        train_df = df[df.kfold != fold].reset_index(drop=True)
        test_df = df[df.kfold == fold].reset_index(drop=True)

        # Convert a collection of text documents to a matrix of token counts (Bag of Words) and fit
        count_vec = CountVectorizer(tokenizer=word_tokenize, token_pattern=None)
        count_vec.fit(train_df.review)

        # transform train/valid data
        xtrain = count_vec.transform(train_df.review)
        xtest = count_vec.transform(test_df.review)

        # initialize Logistic Regression model and fit
        model = LogisticRegression()
        model.fit(xtrain, train_df.sentiment)
        pred = model.predict(xtest)

        # calculate accuracy score for each fold
        acc = accuracy_score(test_df.sentiment, pred)
        print(f"Fold: {fold}, Accuracy = {acc:0.2f}")
