import pandas as pd
import numpy as np
import re

import gensim
from gensim.utils import simple_preprocess
import gensim.corpora as corpora
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

def useless_feature(df):
    df_copy = df.copy()
    df_copy.drop(["verbatims", "upsell_xsell", "issue_level2", "resolution", "city", "city_lat", "city_long", "data_usage_amt", "mou_onnet_6m_normal", "mou_roam_6m_normal", "region_lat", "region_long", "state_lat", "state_long", "tweedie_adjusted"], axis=1, inplace=True)
    return df_copy

def decode_char(df):
    df_copy = df.copy()
    list_vars_object = list(df_copy.select_dtypes(exclude=['int64', 'float64']).columns)
    for var in list_vars_object:
        df_copy[var] = df_copy[var].apply(lambda x : x.decode("utf-8"))
    return df_copy

# some variables intuitively cannot have negative values, let replace negative value by 0 
def lower_limit(df):
    df_copy = df.copy()
    list_ = ["tot_mb_data_roam_curr", "seconds_of_data_norm", "lifetime_value", "bill_data_usg_m03", "bill_data_usg_m06", "voice_tot_bill_mou_curr",
            "tot_mb_data_curr", "mb_data_usg_roamm01", "mb_data_usg_roamm02", "mb_data_usg_roamm03", "mb_data_usg_m01", "mb_data_usg_m02", "mb_data_usg_m03",
            "calls_total", "calls_in_pk", "calls_out_pk", "calls_in_offpk", "calls_out_offpk", "mb_data_ndist_mo6m", "data_device_age",
            "mou_onnet_pct_MOM", "mou_total_pct_MOM"]
    for var in list_:
        df_copy[var] = df_copy[var].apply(lambda x : max(x, 0))
    return df_copy

# handling the high skewness of the variables MB_Data_Usg_M by applying log transformation
def log_transform(df):
    df_copy = df.copy()
    for i in range(4, 10):
        df_copy["log_MB_Data_Usg_M0"+str(i)] = df_copy["MB_Data_Usg_M0"+str(i)].apply(lambda x: np.log(1+x))
        df_copy.drop(columns=["MB_Data_Usg_M0"+str(i)])
    return df_copy

# label variable encoding
def label_encode_variable(df):
    df_copy = df.copy()
    df__ = df_copy.select_dtypes(exclude='float64')
    # verbatims is the only tet variable to let in the df for text_mining
    #df__.drop("verbatims", axis=1)
    df_copy.drop(list(df__.columns), axis=1, inplace=True)
    #df_ = pd.get_dummies(df__.drop("verbatims", axis=1), prefix_sep="_", drop_first=True)
    df_ = pd.get_dummies(df__, prefix_sep="_", drop_first=True)
    df_copy = pd.concat([df_copy, df_], axis=1)
    return df_copy

# retrieving the list of variables having missing values
def missing_var(df):
    df_missing = df.isnull().sum().to_frame().reset_index()
    df_missing.columns = ["variable", "missing_nb"]
    df_missing = df_missing[df_missing["missing_nb"] > 0].reset_index(drop=True)
    # df_missing['missing_pct'] = round(100 * df_missing['missing_nb'] / df.shape[0], 2)
    df_missing = df_missing.sort_values('missing_nb', ascending=False).reset_index(drop=True)
    return list(df_missing["variable"])

# missing indicator
def add_missing_indicator(df):
    df_copy = df.copy()
    list_missing_var = missing_var(df_copy)
    df_ = df_copy[list_missing_var].isnull().astype(int).add_suffix("_MI")
    df_copy = pd.concat([df_copy, df_], axis=1)
    return df_copy

# imputation
def imputation(df):
    df_copy = df.copy()
    list_missing_var = missing_var(df_copy)
    for var in list_missing_var:
        if len(df_copy[var].unique()) > 50:
            df_copy[var].fillna(df_copy[var].mean(), inplace=True)
        else :
            df_copy[var].fillna(df_copy[var].value_counts(ascending=False).to_frame().reset_index().iloc[0, 0], inplace=True)
    return df_copy

# text mining : verbatims variable
def text_processing(df):
    # function to remove ponctuation
    df["verbatims_processed"] = df["verbatims"].apply(lambda x : re.sub('[,\.!?]', '', x))
    # function to convert text to lowercase
    df["verbatims_processed"] = df["verbatims_processed"].apply(lambda x : x.lower())

def sentence_to_words(dsentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(simple_preprocess(str(sentence), deacc=True))

def remove_stopwords(texts):
    stop_words = stopwords.words('english')
    # stop_words.extend([])
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def id2words_corpus(docs):
    id2word = corpora.Dictionary(docs)
    # tem document frequency
    corpus = [id2word.doc2bow(text) for text in docs]
    return id2word, corpus

# latent dirichlet allocation
def lda_modeling(corpus, id2word, num_topics=15):
    # Build LDA model
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                    id2word=id2word,
                                    num_topics=num_topics)
    return lda_model

def extract_topic(ldamodel, topic):
    topic_word = [id2word[term[0]] for term in ldamodel.get_topic_terms(topic)]     
    return '"'+','.join(topic_word)+'"'

def text_mining(df):
    df_text = df["verbatims"].to_frame()
    data = df_text.verbatims_process.values.tolist()
    text_processing(df_text)
    data_words = remove_stopwords(list(sentence_to_words(data)))
    id2word, corpus = id2words_corpus(data_words)
    lda_model = lda_modeling(corpus, id2word)
    list_topics = [extract_topic(lda_model, i) for i in range(10)]
