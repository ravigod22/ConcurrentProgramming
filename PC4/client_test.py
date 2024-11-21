import socket
import pickle

def send_sentence_for_testing(server_address, sentence):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect(server_address)
        task = {"action": "test", "sentence": sentence}
        sock.sendall(pickle.dumps(task))
        try:
            response = pickle.loads(sock.recv(4096))
            print(f"Sentence: {sentence} -> Prediction: {response}")
        except Exception as e:
            print(f"Error receiving prediction for sentence '{sentence}': {e}")


if __name__ == "__main__":
    server_address = ("127.0.0.1", 5000)
    with open("test.txt", "r") as file:
        for line in file:
            sentence = line.strip()
            if sentence:
                send_sentence_for_testing(server_address, sentence)
