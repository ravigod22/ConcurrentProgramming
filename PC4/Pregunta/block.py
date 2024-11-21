import hashlib
import time
import threading
import random

class Node:
    def __init__(self, node_id):
        self.node_id = node_id
        self.estado = "Seguidor"  
        self.votar_por = None  


class Block:
    def __init__(self, index, data, previous_hash):
        self.index = index
        self.data = data
        self.previous_hash = previous_hash
        self.timestamp = time.time()
        self.hash = self.calculate_hash()
        
    def calculate_hash(self):
        block_data = f"{self.index}{self.data}{self.previous_hash}{self.timestamp}"
        return hashlib.sha256(block_data.encode()).hexdigest()


class BlockchainConsensus:
    def __init__(self, node_count):
        self.nodes = [Node(i) for i in range(node_count)]
        self.leader = None
        self.chain = []
        self.lock = threading.Lock()
    
    def elegir_lider(self):
        with self.lock:
            
            for node in self.nodes:
                node.estado = "Seguidor"
                node.votar_por = None
            
            
            candidate = random.choice(self.nodes)
            candidate.estado = "Candidato"
            candidate.votar_por = candidate.node_id
            votes = 1  
            
           
            for node in self.nodes:
                if node != candidate and node.votar_por is None:
                    node.votar_por = candidate.node_id
                    votes += 1
            
            if votes > len(self.nodes) // 2:
                candidate.estado = "Lider"
                self.leader = candidate
                print(f"El líder elegido fue el nodo {candidate.node_id}")
            else:
                print("No se pudo elegir un líder, repitiendo elección...")
                self.elegir_lider()  # Reinicia el proceso de elección
    
    def generar_bloques(self):
        if not self.leader:
            return
        
        for i in range(5):
            previous_hash = self.chain[-1].hash if self.chain else "0"
            block_data = f"Bloque {len(self.chain)} generado por el nodo {self.leader.node_id}"
            new_block = Block(len(self.chain), block_data, previous_hash)
            self.chain.append(new_block)
            print(f"Bloque generado {new_block.index}: {new_block.hash}")
            time.sleep(0.1)  
    
    def run_consensus(self):
        while True:
            self.elegir_lider()
            self.generar_bloques()
            print("Iniciando nueva elección...\n")
            time.sleep(0.5) 


if __name__ == "__main__":
    consensus = BlockchainConsensus(node_count=5)
    consensus_thread = threading.Thread(target=consensus.run_consensus)
    consensus_thread.start()
