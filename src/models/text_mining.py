from pandas import DataFrame
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer


class TextMining:
    def __init__(self, train_data: DataFrame, valid_data: DataFrame) -> None:
        self.train_verbatims = train_data
        self.valid_verbatims = train_data
        self.train_docs = train_data.verbatims.to_list()
        self.valid_docs = valid_data.verbatims.to_list()

    def BERTopic_model(self, n_topics: int):
        """
        Train a BERTopic model
        """
        topic_model = BERTopic()
        topics, probs = topic_model.fit_transform(self.train_docs)
        vectorizer_model = CountVectorizer(stop_words="english")
        topic_model.update_topics(
                            self.train_docs,
                            n_gram_range=(1, 3),
                            vectorizer_model=vectorizer_model)
        topic_model.reduce_topics(self.train_docs, nr_topics=n_topics)
        topic_model._map_predictions(self.valid_docs)
