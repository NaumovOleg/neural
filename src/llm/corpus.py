with open("data_sets/shakespere.txt", "r", encoding="utf-8") as f:
    corpus = f.readlines()
    corpus[0] = corpus[0].replace("\ufeff", "")
