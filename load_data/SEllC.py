from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import io
import os
import pandas as pd
import secrets  

def SEIICEncrypt(record_data):

    # Generate an ECC private key
    private_key = ec.generate_private_key(ec.SECP256R1(), default_backend())
    
    # Serialize the private key 
    private_key_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )
    
    # Deserialize the private key (for decryption)
    loaded_private_key = serialization.load_pem_private_key(private_key_bytes, password=None, backend=default_backend())
    
    # Generate a shared secret using ECDH
    public_key = private_key.public_key()
    shared_key = private_key.exchange(ec.ECDH(), public_key)

    record_data = pd.DataFrame(record_data, index=['1'])
    # df = pd.read_json(record_data, orient='split')
    df = pd.DataFrame(record_data)
    byte_buffer = io.BytesIO()
    df.to_pickle(byte_buffer)
    byte_data = byte_buffer.getvalue()
    data_to_encrypt = byte_data
    
    # Derive a key for encryption
    salt = os.urandom(16)  # Generate a random 16-byte salt
    iterations = 100000
    key_length = 32  # 256 bits
    
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        salt=salt,
        iterations=iterations,
        length=key_length,
        backend=default_backend()
    )
    key = kdf.derive(shared_key)
    
    # Generate a random IV for encryption
    iv = secrets.token_bytes(16)  # 16 bytes for AES-128, adjust as needed
    
    # Use the derived key to encrypt data
    cipher = Cipher(algorithms.AES(key), modes.CFB(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    padder = padding.PKCS7(algorithms.AES.block_size).padder()
    padded_data = padder.update(data_to_encrypt) + padder.finalize()
    encrypted_data = encryptor.update(padded_data) + encryptor.finalize()


    return encrypted_data,key,iv

def SEIICDecrypt(encrypted_data,key,iv):
    # Decrypt the data using the shared key
    cipher = Cipher(algorithms.AES(key), modes.CFB(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
    decrypted_data = decryptor.update(encrypted_data) + decryptor.finalize()
    original_data = unpadder.update(decrypted_data) + unpadder.finalize()
    
    # Deserialize the DataFrame
    byte_buffer = io.BytesIO(original_data)
    df = pd.read_pickle(byte_buffer)

    return df