from build_model import build_model
from prepare_training import maxlen, vocab_size, X, y
from response import generate_response
import matplotlib.pyplot as plt

model = build_model(maxlen, vocab_size)
model.summary()


history = model.fit(X, y, epochs=200)
model.save("./saved_models/savedmodel1.h5")
print("hi there-------->", generate_response(model, "hi there", temperature=0.8))
print("How are you-------->", generate_response(model, "How are you", temperature=0.8))
print(
    "Tell me about your mood----------->",
    generate_response(model, "Tell me about your mood", temperature=0.8),
)


def show_training_history():
    plt.plot(history.history["loss"])
    plt.title("Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train"], loc="upper left")
    plt.show()


show_training_history()
