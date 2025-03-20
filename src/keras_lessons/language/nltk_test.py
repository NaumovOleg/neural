import nltk
import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, LancasterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sentence_transformers import SentenceTransformer, util
from prepare_tokens import tokenize_text, prepare_sentences
from calculate_tfidf import calculate_tfidf

tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
lemmatizer = WordNetLemmatizer()
stemmer = LancasterStemmer()
vectorizer = CountVectorizer(
    analyzer="word",
    lowercase=True,
    tokenizer=None,
    preprocessor=None,
    stop_words="english",
    max_features=5000,
)


input_data = "Order details confirmed. Prepare ingredients. Ingredients ready"
corpus = "As a baker, when order details are confirmed, I want to prepare ingredients, so that I can have the ingredients ready"


input_token = tokenize_text(input_data)
output_token = tokenize_text(corpus)

input_sentence = " ".join(input_token)
output_sentence = " ".join(output_token)

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode([input_sentence, output_sentence], convert_to_tensor=True)
similarity_score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()

print(similarity_score, end="\n")
# print(input_sentence, end="\n")
# print(output_sentence, end="\n")

tf_idf, vectorizer = calculate_tfidf(prepare_sentences(corpus))
# print(vectorizer.get_feature_names_out(), end="\n\n\n")
# print(tf_idf, end="\n\n\n")


df_tfidf = pd.DataFrame(tf_idf.toarray(), columns=vectorizer.get_feature_names_out())


def filter_words_with_score(df):
    sentences = []
    for column in df.columns:
        sentence = df[column][df[column] > 0].index.tolist()
        sentences.append(" ".join(sentence))  # Join words into a sentence
    return sentences


sentences = filter_words_with_score(df=df_tfidf.T)


print(sentences, end="\n\n\n")
print(output_sentence)
