
import re
from numpy import max, log
from pandas import DataFrame, concat, get_dummies
from typing import List

import gensim
from gensim.utils import simple_preprocess
from gensim.corpora import Dictionary
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

class TextMining:
    def text_processing(self):
        """
        to remove ponctuation and to convert text to lowercase
        """ 
        self.data["verbatims_processed"] = (self.data["verbatims"]
                                                .apply(lambda x : re.sub('[,\.!?]', '', x))
                                                .apply(lambda x : x.lower())
                                            )


    @staticmethod
    def sentence_to_words(sentences):
        for sentence in sentences:
            # deacc=True removes punctuations
            yield(simple_preprocess(str(sentence), deacc=True))


    @staticmethod
    def remove_stop_edundant_words(texts):
        stop_words = stopwords.words('english')
        return [[word for word in simple_preprocess(str(doc)) 
                    if word not in stop_words] for doc in texts]


    @staticmethod
    def remove_redundant_words(docs):
        id2word = Dictionary(docs)
        id2word.dfs 
        return id2word

    
    @staticmethod
    def id2words_corpus(docs):
        id2word = Dictionary(docs)
        corpus = [id2word.doc2bow(text) for text in docs]
        return id2word, corpus


    @staticmethod
    def lda_modeling(
                corpus, 
                id2word, 
                num_topics=15):
        """
        build latent Dirichlet Allocation model
        """
        lda_model = gensim.models.LdaMulticore(
                                        corpus=corpus,
                                        id2word=id2word,
                                        num_topics=num_topics)
        return lda_model


    @staticmethod
    def extract_topic(
                ldamodel, 
                id2word, 
                topic):
        topic_word = [id2word[term[0]] 
                        for term in ldamodel.get_topic_terms(topic)]     
        return '"'+','.join(topic_word)+'"'


    def text_mining(self):
        df_text = self.data["verbatims"].to_frame()
        data = df_text.verbatims_process.values.tolist()
        self.text_processing()
        data_words = self.remove_stopwords(list(self.sentence_to_words(data)))
        id2word, corpus = self.id2words_corpus(data_words)
        lda_model = self.lda_modeling(corpus, id2word)
        list_topics = [self.extract_topic(lda_model, id2word, i) for i in range(10)]