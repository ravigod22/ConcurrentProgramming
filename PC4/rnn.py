import numpy as np

class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Pesos de la red
        self.Wx = np.random.randn(hidden_size, input_size) * 0.01  # Pesos de entrada a estado oculto
        self.Wh = np.random.randn(hidden_size, hidden_size) * 0.01  # Pesos de estado oculto recurrente
        self.Wy = np.random.randn(output_size, hidden_size) * 0.01  # Pesos de estado oculto a salida
        self.bh = np.zeros((hidden_size, 1))  # Sesgo del estado oculto
        self.by = np.zeros((output_size, 1))  # Sesgo de salida

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x, axis=0, keepdims=True)

    def forward(self, inputs, hidden_state):
        h_states = []
        y_outputs = []

        for t in range(len(inputs)):
            hidden_state = np.tanh(np.dot(self.Wx, inputs[t]) + np.dot(self.Wh, hidden_state) + self.bh)
            # print(f"Dim Wx: {self.Wx.shape}, Dim input: {inputs[t].shape}, Dim hidden: {hidden_state.shape}")
            h_states.append(hidden_state)
            y = np.dot(self.Wy, hidden_state) + self.by
            y_outputs.append(self.softmax(y))

        return h_states, y_outputs

    def backward(self, inputs, targets, h_states, y_outputs):
        dWx = np.zeros_like(self.Wx)
        dWh = np.zeros_like(self.Wh)
        dWy = np.zeros_like(self.Wy)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)

        dh_next = np.zeros_like(h_states[0])

        for t in reversed(range(len(inputs))):
            dy = y_outputs[t] - targets[t]
            dWy += np.dot(dy, h_states[t].T)
            dby += dy

            dh = np.dot(self.Wy.T, dy) + dh_next
            dh_raw = (1 - h_states[t] ** 2) * dh

            dWx += np.dot(dh_raw, inputs[t].T)
            dWh += np.dot(dh_raw, h_states[t-1].T) if t > 0 else 0
            dbh += dh_raw

            dh_next = np.dot(self.Wh.T, dh_raw)

        for param, dparam in zip([self.Wx, self.Wh, self.Wy, self.bh, self.by],
                                 [dWx, dWh, dWy, dbh, dby]):
            param -= self.learning_rate * dparam

    def train(self, inputs, targets, epochs=100):
        hidden_state = np.zeros((self.hidden_size, 1))

        for epoch in range(epochs):
            total_loss = 0

            for seq_idx in range(len(inputs)):
                h_states, y_outputs = self.forward(inputs[seq_idx], hidden_state)
                loss = -np.sum(np.log(y_outputs[-1]) * targets[seq_idx])
                total_loss += loss
                self.backward(inputs[seq_idx], targets[seq_idx], h_states, y_outputs)

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

    def predict(self, input_seq):
        hidden_state = np.zeros((self.hidden_size, 1))
        _, y_outputs = self.forward(input_seq, hidden_state)
        predicted_idx = np.argmax(y_outputs[-1])
        return predicted_idx
