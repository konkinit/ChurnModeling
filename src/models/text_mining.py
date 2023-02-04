
import re
from numpy import max, log
from pandas import DataFrame, concat, get_dummies
from typing import List

from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

class TextMining:
    def __init__(self, 
                train_data: DataFrame,
                valid_data: DataFrame) -> None:
        self.train_verbatims = train_data
        self.valid_verbatims = train_data
            
    
    def train_BERTopic_model(self):
        # Train a BERTopic model
        topic_model = BERTopic()
        topics, probs = topic_model.fit_transform(self.train_verbatims)

        # Fine-tune topic representations after training BERTopic
        vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, 3))
        topic_model.update_topics(self.train_verbatims, vectorizer_model=vectorizer_model)