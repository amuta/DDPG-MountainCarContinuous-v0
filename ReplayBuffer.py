import random
from collections import namedtuple, deque
import numpy as np
import copy

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size: maximum size of buffer
            batch_size: size of each training batch
        """
        self.buffer_size = buffer_size
        self.memory = deque(maxlen=self.buffer_size)  # internal memory (deque)
        self.temp = []
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.temp.append(e)
    
    def reset(self):
        self.memory = deque(maxlen=self.buffer_size)

    def update(self, done):
        if done:
            self.memory.extendleft(self.temp)
        self.temp = []
        
    def sample(self, batch_size=64):
        """sample a batch of experiences from memory using priorities weight as probs"""
        sorted_memory = sorted(self.memory, key=lambda r: abs(r[2]), reverse=True)
        probs = [0.95 ** i for i in range(len(sorted_memory))]
        sum_p = sum(probs)
        probs = [p / sum_p for p in probs]
        ids = np.random.choice(range(len(self.memory)), replace=False, size=batch_size, p=probs)
        return np.array(sorted_memory)[ids]
        # return np.array(random.sample(self.memory, k=self.batch_size))


    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)