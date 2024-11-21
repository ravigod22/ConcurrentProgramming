import numpy as np
from rnn import SimpleRNN
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import nltk

# Descargar recursos de NLTK si es necesario
nltk.download("punkt")
nltk.download("stopwords")


def tonkenize_text(text):
    tokens = word_tokenize(text.lower())  # Tokenización básica
    tokens = [word for word in tokens if word.isalnum()]  # Eliminar puntuación
    stop_words = set(stopwords.words("english"))  # Stopwords en inglés
    tokens = [word for word in tokens if word not in stop_words]  # Eliminar stopwords
    return tokens

def train_model(text):
    tokens = tonkenize_text(text)
    vocab = {word: i for i, word in enumerate(set(tokens))}
    inv_vocab = {i: word for word, i in vocab.items()}
    
    input_sequences = []
    target_sequences = []

    for i in range(len(tokens) - 1):
        x = np.zeros((len(vocab), 1))
        y = np.zeros((len(vocab), 1))

        x[vocab[tokens[i]]] = 1
        y[vocab[tokens[i + 1]]] = 1

        input_sequences.append(x)
        target_sequences.append(y)

    input_sequences = [input_sequences]
    target_sequences = [target_sequences]

    rnn = SimpleRNN(input_size=len(vocab), hidden_size=10, output_size=len(vocab), learning_rate=0.01)
    rnn.train(input_sequences, target_sequences, epochs=50)

    # Devolver solo los pesos y vocabulario
    return {
        "weights": {
            "Wx": rnn.Wx,
            "Wh": rnn.Wh,
            "Wy": rnn.Wy,
            "bh": rnn.bh,
            "by": rnn.by
        },
        "vocab": vocab,
        "inv_vocab": inv_vocab
    }
  

def combine_models(models):
    combined_vocab = {}
    current_index = 0
    for model in models:
        for word in model["vocab"]:
            if word not in combined_vocab:
                combined_vocab[word] = current_index
                current_index += 1

    combined_inv_vocab = {v: k for k, v in combined_vocab.items()}
    hidden_size = models[0]["weights"]["Wx"].shape[0]
    combined_size = len(combined_vocab)

    combined_weights = {
        "Wx": np.zeros((hidden_size, combined_size)),
        "Wh": np.zeros((hidden_size, hidden_size)),
        "Wy": np.zeros((combined_size, hidden_size)),
        "bh": np.zeros((hidden_size, 1)),
        "by": np.zeros((combined_size, 1)),
    }

    for model in models:
        weights = model["weights"]
        for word, idx in model["vocab"].items():
            combined_idx = combined_vocab[word]
            combined_weights["Wx"][:, combined_idx] += weights["Wx"][:, idx]
            combined_weights["Wy"][combined_idx, :] += weights["Wy"][idx, :]
            combined_weights["by"][combined_idx, :] += weights["by"][idx, :]

        combined_weights["Wh"] += weights["Wh"]
        combined_weights["bh"] += weights["bh"]

    num_models = len(models)
    for key in combined_weights:
        combined_weights[key] /= num_models

    return {
        "weights": combined_weights,
        "vocab": combined_vocab,
        "inv_vocab": combined_inv_vocab,
    }


def predict(sequence, combined_model):
    vocab = combined_model["vocab"]
    inv_vocab = combined_model["inv_vocab"]
    weights = combined_model["weights"]

    # Reconstruir el modelo
    rnn = SimpleRNN(input_size=weights["Wx"].shape[1],
                    hidden_size=weights["Wx"].shape[0],
                    output_size=weights["Wy"].shape[0])
    rnn.Wx = weights["Wx"]
    rnn.Wh = weights["Wh"]
    rnn.Wy = weights["Wy"]
    rnn.bh = weights["bh"]
    rnn.by = weights["by"]

    # Convertir la secuencia a one-hot encoding
    input_seq = np.zeros((len(vocab), 1))
    input_seq[vocab[sequence]] = 1

    predicted_idx = rnn.predict([input_seq])
    return inv_vocab[predicted_idx]

