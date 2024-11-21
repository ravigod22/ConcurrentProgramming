import socket
import pickle
from model import predict


class PredictWorker:
    def __init__(self, host, server_address):
        self.host = host
        self.port = self.get_free_port()
        self.server_address = server_address
        self.model = None
    
    def get_free_port(self):
        """Encuentra un puerto libre dinámicamente."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as temp_socket:
            temp_socket.bind((self.host, 0))
            return temp_socket.getsockname()[1]
        
    def register_with_server(self):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.connect(self.server_address)
                task = {"action": "register", "type": "predict", "port": self.port}
                sock.sendall(pickle.dumps(task))
                response = sock.recv(4096)
                print(response.decode())
        except Exception as e:
            print(f"Failed to register with server: {e}")

    def handle_task(self, conn):
        try:
            data = pickle.loads(conn.recv(4096))
            if data["action"] == "store_model":
                print("Storing model...")
                self.model = data["model"]
                print("Model update successfully")
                conn.sendall(b"Model stored.")
            elif data["action"] == "predict":
                print("Making predictions...")
                predictions = predict(data["sentence"], self.model)
                conn.sendall(pickle.dumps(predictions))
        except Exception as e:
            print(f"Error handling task: {e}")
        finally:
            conn.close()

    def start(self):
        worker_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        worker_socket.bind((self.host, self.port))
        worker_socket.listen(5)
        self.register_with_server()
        print(f"Predict Worker listening on {self.host}:{self.port}")
        while True:
            conn, _ = worker_socket.accept()
            self.handle_task(conn)


if __name__ == "__main__":
    worker = PredictWorker("127.0.0.1", ("127.0.0.1", 5000))
    worker.start()
