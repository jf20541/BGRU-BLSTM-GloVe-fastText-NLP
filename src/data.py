import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
import config

tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words("english")


def remove_special_characters(text):
    """Remove unwanted characters [!@#$%^&*()]
    Args:
        text (str): description review
    Returns:
        [str]: cleaned reviews
    """
    soup = BeautifulSoup(text, "html.parser")
    review = soup.get_text()
    review = r"[^a-zA-z0-9\s]"
    review = re.sub(review, "", text)
    return review.lower()


def remove_stopwords(text):
    """Removing stopwords (common words) to reduce computation, noise, improve performance
    Args:
        text (str): description review
    Returns:
        [str]: more relevant retrieval for each review
    """
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    filtered_text = " ".join(filtered_tokens)
    return filtered_text


if __name__ == "__main__":
    df = pd.read_csv(config.TRAINING_FILE)
    # label encode sentiment values (1=positive, 0=negative)
    df.sentiment = [1 if each == "positive" else 0 for each in df.sentiment]
    # apply remove_stopwords and remove_special_characters functions
    df["review"] = df["review"].apply(remove_special_characters)
    df["review"] = df["review"].apply(remove_stopwords)
    df.to_csv(config.TRAINING_FILE_CLEAN, index=False)
