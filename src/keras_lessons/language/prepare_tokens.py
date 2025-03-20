import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, LancasterStemmer

unique_stops = set(stopwords.words("english"))

lemmatizer = WordNetLemmatizer()
stemmer = LancasterStemmer()


# nltk.download("punkt")
# nltk.download("punkt_tab")
# nltk.download("stopwords")


unique_stops = set(stopwords.words("english"))


def tokenize_sentences(sentences: list[str]):
    words = np.array([])
    for sentence in sentences:
        tokenized = np.array(np.char.lower(word_tokenize(sentence)))
        words = np.concatenate((words, tokenized))


def prepare_tokens(text: str):
    word_list = np.array([])
    sentences = np.array(sent_tokenize(text))

    for sentence in sentences:
        words = np.array(np.char.lower(word_tokenize(sentence)))
        word_list = np.concatenate((word_list, words))

    word_list = word_list[~np.isin(word_list, list(unique_stops))]
    is_alpha = np.vectorize(str.isalpha)
    return list(word_list[is_alpha(word_list)])


def lemmatize_words(words: list[str]):
    return [lemmatizer.lemmatize(word) for word in words]


def stemming(words: list[str]):
    return [stemmer.stem(word) for word in words]


def tokenize_text(text: str):
    unique = prepare_tokens(text)
    lemmatized = lemmatize_words(unique)
    return stemming(lemmatized)


def prepare_sentences(text: str):
    sentences = np.array(sent_tokenize(text))
    tokenized = []
    for sentence in sentences:
        words = np.array((np.array(np.char.lower(word_tokenize(sentence)))))
        words = list(words[~np.isin(words, list(unique_stops))])
        words = lemmatize_words(words)
        words = np.array(stemming(words))
        print(words, end="\n\n\n")
        tokenized.append(words)
    return tokenized
