import hashlib
import json
import time
from cryptography.fernet import Fernet
import pandas as pd
import SEllC
import base64

# Define the Block class
class Block:
    def __init__(self, index, previous_hash, timestamp, data, merkle_root, hash):
        self.index = index
        self.previous_hash = previous_hash
        self.timestamp = timestamp
        self.data = data  # Store the base64-encoded string
        self.merkle_root = merkle_root
        self.hash = hash


# Define the MerkleNode class for the Merkle Tree
class MerkleNode:
    def __init__(self, left=None, right=None, data=None):
        self.left = left
        self.right = right
        self.data = data

    def hash(self):
        if self.data is None:
            return ""
        elif isinstance(self.data, int):
            data_str = str(self.data)
        else:
            data_str = self.data
        
        left_hash = self.left.hash() if self.left else ""
        right_hash = self.right.hash() if self.right else ""
        
        return hashlib.sha256((left_hash + right_hash + data_str).encode()).hexdigest()


class Blockchain:
    def __init__(self):
        self.chain = []

    def create_genesis_block(self):
        # Create the genesis block with arbitrary data
        genesis_block = Block(0, "0", int(time.time()), "Genesis Block", "", self.hash_block(None))
        self.chain.append(genesis_block)

    def calculate_merkle_root(self, data_blocks):
        # Construct the Merkle Tree and calculate the Merkle Root hash
        if not data_blocks:
            return ""
        leaf_nodes = [MerkleNode(data=block) for block in data_blocks]
        while len(leaf_nodes) > 1:
            parent_nodes = []
            for i in range(0, len(leaf_nodes), 2):
                left = leaf_nodes[i]
                right = leaf_nodes[i + 1] if i + 1 < len(leaf_nodes) else None
                parent = MerkleNode(left=left, right=right)
                parent_nodes.append(parent)
            leaf_nodes = parent_nodes
        return leaf_nodes[0].hash()

    def hash_block(self, block):
        # Hash a block using SHA-256
        if block is None:
            # Handle the case of the genesis block
            block_string = json.dumps("0", sort_keys=True)
        else:
            block_string = json.dumps(block.__dict__, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()

    def add_block(self, data):
        # Add a new block to the blockchain
        previous_block = self.chain[-1] if self.chain else None
        new_index = len(self.chain)
        new_timestamp = int(time.time())
        merkle_root = self.calculate_merkle_root(data)  # Calculate Merkle Root for the new block       
        new_block = Block(new_index, previous_block.hash if previous_block else "0", new_timestamp, data, str(merkle_root), self.hash_block(previous_block))

        self.chain.append(new_block)


