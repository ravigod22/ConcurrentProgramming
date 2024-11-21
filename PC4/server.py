import socket
import threading
import pickle
from collections import Counter
import random
from consensus import Consensus
from model import combine_models, predict
from threading import Lock


def split_text_equally(text, num_workers):
    """Divide el texto equitativamente entre los workers."""
    words = text.split()  # Divide el texto en palabras
    chunk_size = len(words) // num_workers  # Tamaño base de cada lote
    remainder = len(words) % num_workers   # Palabras sobrantes

    batches = []
    start = 0
    for i in range(num_workers):
        # Asignar el tamaño base más palabras adicionales si hay remanente
        end = start + chunk_size + (1 if i < remainder else 0)
        batch = " ".join(words[start:end])  # Reunir palabras en una cadena
        batches.append(batch)
        start = end

    return batches

class Server:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.train_workers = []
        self.predict_workers = []
        self.current_leader = None
        self.model = None
        self.lock = Lock()

    def handle_client(self, conn, addr):

        try:
            data = pickle.loads(conn.recv(4096))
            if data["action"] == "register":
                self.register_worker((addr[0], data["port"]), data["type"])
                conn.sendall(b"Worker registered dynamically.")
            elif data["action"] == "train":
                print(f"Received book for training from {addr}")
                self.distribute_training(data["tokens"])
                conn.sendall(b"Training completed and model distributed to predict workers.")
            elif data["action"] == "test":
                print(f"Received testing request from {addr}")
                prediction = self.distribute_testing(data["sentence"])
                print(prediction)
                conn.sendall(pickle.dumps(prediction))
            else:
                raise ValueError(f"Unknown action: {data['action']}")
        except Exception as e:
            print(f"Error handling client {addr}: {e}")
        finally:
            conn.close()

    def register_worker(self, worker_address, worker_type):

        with self.lock:
            if worker_type == "train" and worker_address not in self.train_workers:
                self.train_workers.append(worker_address)
                print(f"Train Worker dynamically registered: {worker_address}")
                
            elif worker_type == "predict" and worker_address not in self.predict_workers:
                self.predict_workers.append(worker_address)
                print(f"Predict Worker dynamically registered: {worker_address}")
                
                # Enviar el modelo al nuevo Predict Worker si está disponible
                if self.model:
                    try:
                        print(f"Sending model to new Predict Worker {worker_address}")
                        # print(self.model)
                        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                            sock.connect(worker_address)
                            task = {"action": "store_model", "model": self.model}
                            print("Enviando modelo")
                            sock.sendall(pickle.dumps(task))
                    except Exception as e:
                        print(f"Failed to send model to Predict Worker {worker_address}: {e}")
    
    def elect_leader(self):
        
        with self.lock:
            print("Starting leader election via voting...")
            self.current_leader = random.choice(self.predict_workers)
            print(f"New leader elected: {self.current_leader}")
    
    def distribute_training(self, tokens):
        """Distribuye tareas de entrenamiento equitativamente entre los workers."""
        num_workers = len(self.train_workers)
        if num_workers == 0:
            print("No train workers available for training.")
            return

        print("Distribuyendo tokens equitativamente...")
        batches = split_text_equally(tokens, num_workers)  # Dividir tokens equitativamente
        #print(batches)
        trained_models = []

        for worker, batch in zip(self.train_workers, batches):
            print(f"Enviando lote de tamaño {len(batch)} a worker {worker}")
            trained_model = self.send_to_worker(worker, {"action": "train", "tokens": batch})
            if trained_model:
                trained_models.append(trained_model)

        print("Modelos entrenados recibidos")
        self.model = combine_models(trained_models)  # Combinar modelos entrenados
        print("Modelo combinado")

    def distribute_testing(self, task):
        with self.lock:
            print(self.predict_workers)
            self.elect_leader()
            print(self.current_leader)
            
            chosen_worker = self.current_leader

        try:
            # Enviar tarea al líder
            print(f"Assigning task to leader {chosen_worker}...")
            response = self.send_to_worker(chosen_worker, {"action": "predict", "sentence": task})
            print(f"Response from leader {chosen_worker}: {response}")
            return response
        except Exception as e:
            print(f"Failed to assign task to leader {chosen_worker}: {e}")
            self.predict_workers[chosen_worker] = "free"  # Liberar líder si ocurre un error
            return None

        
    def send_to_worker(self, worker_address, task):
        """Envía una tarea a un worker."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.connect(worker_address)
                sock.sendall(pickle.dumps(task))
                response = pickle.loads(sock.recv(4096))
                return response
        except Exception as e:
            print(f"Failed to communicate with worker {worker_address}: {e}")
            return None

    def start(self):
        """Inicia el servidor."""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((self.host, self.port))
        server_socket.listen(5)
        print("Server listening...")
        while True:
            conn, addr = server_socket.accept()
            threading.Thread(target=self.handle_client, args=(conn, addr)).start()


if __name__ == "__main__":
    server = Server("127.0.0.1", 5000)
    server.start()

