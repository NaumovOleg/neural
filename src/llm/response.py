from tokenizer import tokenizer
from prepare_training import pad_sequences, maxlen, vocab_size
import numpy as np


def generate_response(model, prompt, max_tokens=10, temperature=1.0):
    for _ in range(max_tokens):
        sequence = tokenizer.texts_to_sequences([prompt])[0]
        sequence = pad_sequences([sequence], maxlen=maxlen - 1, padding="pre")
        preds = model.predict(sequence, verbose=0)[0]
        preds = np.log(preds + 1e-8) / temperature
        probs = np.exp(preds) / np.sum(np.exp(preds))
        next_idx = np.random.choice(range(vocab_size), p=probs)
        next_word = tokenizer.index_word.get(next_idx, "")
        prompt += " " + next_word
        if next_word == "<eos>":
            break
    return prompt
