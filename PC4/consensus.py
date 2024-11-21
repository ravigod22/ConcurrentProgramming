import random


class Consensus:
    def __init__(self, workers):
        self.workers = workers

    def get_leader(self):
        return random.randint(0, len(self.workers) - 1)
