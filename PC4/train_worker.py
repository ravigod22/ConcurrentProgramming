import socket
import pickle
from model import train_model

class TrainWorker:
    def __init__(self, host, server_address):
        self.host = host
        self.port = self.get_free_port()  # Puerto dinámico
        self.server_address = server_address

    def get_free_port(self):
        """Encuentra un puerto libre dinámicamente."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as temp_socket:
            temp_socket.bind((self.host, 0))
            return temp_socket.getsockname()[1]

    def register_with_server(self):
        """Registra el worker con el servidor."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.connect(self.server_address)
                task = {"action": "register", "type": "train", "port": self.port}
                sock.sendall(pickle.dumps(task))
                response = sock.recv(4096)
                print(response.decode())
        except Exception as e:
            print(f"Failed to register with server: {e}")

    def handle_task(self, conn):
        """Procesa la tarea recibida del servidor."""
        print("Recibido")
        try:
            data = pickle.loads(conn.recv(4096))
            print(f"Received task: {data}")
            if data["action"] == "train":
                print("Training model...")
                trained_model = train_model(data["tokens"])
                conn.sendall(pickle.dumps(trained_model))
        except Exception as e:
            print(f"Error handling task: {e}")
        finally:
            conn.close()

    def start(self):
        """Inicia el worker y escucha conexiones."""
        self.register_with_server()
        worker_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        worker_socket.bind((self.host, self.port))
        worker_socket.listen(5)
        print(f"Train Worker listening on {self.host}:{self.port}")
        while True:
            conn, _ = worker_socket.accept()
            self.handle_task(conn)


if __name__ == "__main__":
    worker = TrainWorker("127.0.0.1", ("127.0.0.1", 5000))
    worker.start()

