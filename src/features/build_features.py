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


class DataProcessing:
    def __init__(self, raw_data: DataFrame) -> None:
        self.data = raw_data

    def useless_feature(self) -> None:
        """
        remove useless features like zip_code, ...
        """
        self.data.drop(["verbatims", "upsell_xsell", "issue_level2", 
                    "resolution", "city", "city_lat", "city_long", 
                    "data_usage_amt", "mou_onnet_6m_normal", "mou_roam_6m_normal", 
                    "region_lat", "region_long", "state_lat", 
                    "state_long", "tweedie_adjusted"], 
                    axis=1, inplace=True)


    def decode_char(self) -> None:
        """
        decode str to UTF-8
        """
        list_vars_object = list(self.data.select_dtypes(
                                exclude=['int64', 'float64']
                                ).columns)
        for var in list_vars_object:
            self.data[var] = self.data[var].apply(lambda x : x.decode("utf-8"))


    def lower_limit(self) -> None:
        """
        some variables intuitively cannot have negative values, 
        replace negative values with 0 
        """
        list_ = ["tot_mb_data_roam_curr", "seconds_of_data_norm", "lifetime_value", 
                "bill_data_usg_m03", "bill_data_usg_m06", "voice_tot_bill_mou_curr",
                "tot_mb_data_curr", "mb_data_usg_roamm01", "mb_data_usg_roamm02", 
                "mb_data_usg_roamm03", "mb_data_usg_m01", "mb_data_usg_m02", 
                "mb_data_usg_m03", "calls_total", "calls_in_pk", "calls_out_pk", 
                "calls_in_offpk", "calls_out_offpk", "mb_data_ndist_mo6m", "data_device_age",
                "mou_onnet_pct_MOM", "mou_total_pct_MOM"]
        for var in list_:
            self.data[var] = self.data[var].apply(lambda x : max(x, 0))
        return self.data


    def log_transform(self) -> None:
        """
        handling the high skewness of the variables MB_Data_Usg_M by applying log transformation
        """
        for i in range(4, 10):
            self.data[f"log_MB_Data_Usg_M0{str(i)}"] = self.data[f"MB_Data_Usg_M0{str(i)}"].apply(lambda x: log(1+x))
            self.data.drop(columns=[f"MB_Data_Usg_M0{str(i)}"], inplace=True)


    def label_encode_variable(self) -> None:
        """
        verbatims is the only tet variable to let in the dataframe for text_mining
        """
        df__ = self.data.select_dtypes(exclude='float64')
        self.data.drop(list(df__.columns), axis=1, inplace=True)
        df_ = get_dummies(df__, prefix_sep="_", drop_first=True)
        self.data = concat([self.data, df_], axis=1)


    def missing_var(self) -> None:
        """
        retrieving the list of variables having missing values
        """
        df_missing = self.data.isnull().sum().to_frame().reset_index()
        df_missing.columns = ["variable", "missing_nb"]
        df_missing = df_missing[df_missing["missing_nb"] > 0].reset_index(drop=True)
        df_missing = df_missing.sort_values('missing_nb', ascending=False).reset_index(drop=True)
        return list(df_missing["variable"])

    
    def add_missing_indicator(self) -> None:
        """
        create missing indicator for features with missingness
        """
        list_missing_var = self.missing_var()
        df_missing_indicator = self.data[list_missing_var].isnull().astype(int).add_suffix("_MI")
        self.data = concat([self.data, df_missing_indicator], axis=1)


    def imputation(self):
        """
        impute missing values with right method
        """
        list_missing_var = self.missing_var()
        for var in list_missing_var:
            if len(self.data[var].unique()) > 50:
                """
                condition that a variable is continious
                """
                self.data[var].fillna(self.data[var].mean(), inplace=True)
            else :
                self.data[var].fillna(
                    self.data[var].value_counts(ascending=False
                    ).to_frame().reset_index().iloc[0, 0], inplace=True)


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
    def remove_stopwords(texts):
        stop_words = stopwords.words('english')
        return [[word for word in simple_preprocess(str(doc)) 
                    if word not in stop_words] for doc in texts]


    @staticmethod
    def id2words_corpus(docs):
        id2word = Dictionary(docs)
        corpus = [id2word.doc2bow(text) for text in docs]
        return id2word, corpus


    @staticmethod
    def lda_modeling(corpus, id2word, num_topics=15):
        """
        build latent Dirichlet Allocation model
        """
        lda_model = gensim.models.LdaMulticore(
                                        corpus=corpus,
                                        id2word=id2word,
                                        num_topics=num_topics)
        return lda_model


    @staticmethod
    def extract_topic(ldamodel, topic):
        topic_word = [id2word[term[0]] for term in ldamodel.get_topic_terms(topic)]     
        return '"'+','.join(topic_word)+'"'

    def text_mining(self):
        df_text = self.data["verbatims"].to_frame()
        data = df_text.verbatims_process.values.tolist()
        self.text_processing()
        data_words = self.remove_stopwords(list(self.sentence_to_words(data)))
        id2word, corpus = self.id2words_corpus(data_words)
        lda_model = self.lda_modeling(corpus, id2word)
        list_topics = [self.extract_topic(lda_model, i) for i in range(10)]
