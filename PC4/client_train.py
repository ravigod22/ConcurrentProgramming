import socket
import pickle

def send_file_content_for_training(server_address, file_path):
    """Lee el contenido del archivo y lo envía al servidor para entrenamiento."""
    with open(file_path, "r") as file:
        content = file.read()

    print(f"Contenido del archivo leído: {content}")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect(server_address)
        task = {"action": "train", "tokens": content}  # Envía el texto completo al servidor
        sock.sendall(pickle.dumps(task))
        response = pickle.loads(sock.recv(4096))
        print(f"Pesos recibidos del servidor: {response['weights']}")

if __name__ == "__main__":
    server_address = ("127.0.0.1", 5000)
    file_path = "book.txt"  # Ruta del archivo
    send_file_content_for_training(server_address, file_path)
