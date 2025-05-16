import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ==== Параметры ====
max_len = 40  # длина входной последовательности
vocab = sorted(set("абвгдеёжзийклмнопрстуфхцчшщьыъэюя .,!?-:\nМ"))  # можно расширить
vocab_size = len(vocab)
embed_dim = 64
num_heads = 2
ff_dim = 128
epochs = 20

# ==== Токенизация ====
char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = {i: u for u, i in char2idx.items()}


def encode(s):
    return [char2idx[c] for c in s if c in char2idx]


def decode(tokens):
    return "".join(idx2char[t] for t in tokens)


with open("data_sets/text-good.txt", "r", encoding="utf-8") as f:
    text = f.readlines()

text = " ".join(text)

# ==== Данные ====
# text = (
#     "Мама мыла раму. "
#     "Папа читал газету. "
#     "Бабушка вязала носки. "
#     "Собака лаяла. "
#     "Окно блестело на солнце."
# )

print(text)
data = encode(text)


def create_dataset(data, seq_len):
    inputs, targets = [], []
    for i in range(len(data) - seq_len):
        inputs.append(data[i : i + seq_len])
        targets.append(data[i + 1 : i + seq_len + 1])
    return np.array(inputs), np.array(targets)


X, y = create_dataset(data, max_len)


# ==== Модель ====
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=sequence_length, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


def build_model():
    inputs = keras.Input(shape=(max_len,))
    x = PositionalEmbedding(max_len, vocab_size, embed_dim)(inputs)
    x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)
    x = layers.Dense(vocab_size)(x)
    return keras.Model(inputs=inputs, outputs=x)


model = build_model()
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer="adam"
)

# ==== Обучение ====
model.fit(X, y, epochs=epochs, batch_size=16)


# ==== Генерация ====
def generate_text(model, start_text, max_gen=100):
    input_ids = encode(start_text)
    input_ids = input_ids[:max_len]
    input_ids = input_ids + [0] * (max_len - len(input_ids))
    input_ids = np.array([input_ids])

    generated = encode(start_text)

    for _ in range(max_gen):
        preds = model.predict(input_ids, verbose=0)
        next_token_logits = preds[0, -1]
        next_token = np.argmax(next_token_logits)
        generated.append(next_token)
        input_ids = np.roll(input_ids, -1, axis=1)
        input_ids[0, -1] = next_token

    return decode(generated)


# ==== Результат ====
print("Сгенерированный текст:")
print(generate_text(model, "Не просто мечтай "))
