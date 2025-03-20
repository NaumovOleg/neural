from sklearn.feature_extraction.text import TfidfVectorizer


def calculate_tfidf(sentences):
    sentences_str = [" ".join(sentence) for sentence in sentences]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences_str)

    return (tfidf_matrix, vectorizer)
