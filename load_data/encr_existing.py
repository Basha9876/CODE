from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding

'''
RSA ENCRYPTION
'''

def RSA_Encrypt(data):
    # Generate RSA keys
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048
    )
    
    # Serialize the private key to PEM format (you can save it to a file)
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )
    
    # Generate the corresponding public key
    public_key = private_key.public_key()
    
    # Serialize the public key to PEM format (you can share it)
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    
    # data = {
    #     'duration': 0,
    #     'protocol_type': 'tcp',
    #     'service': 'http',
    #     'flag': 'SF',
    #     'src_bytes': 229,
    #     'dst_bytes': 934,
    #     'land': 0,
    #     # ... (rest of the dictionary)
    # }
    
    data_str = str(data)
    try:
        ciphertext = public_key.encrypt(
            data_str.encode(),
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        print("Encrypted data:", ciphertext.hex())
    
    except Exception as e:
        print("Encryption failed:", str(e))
    
    
from cryptography.hazmat.primitives.asymmetric import padding

def RSA_decrypt(ciphertext_hex,private_key):
    ciphertext_hex = "hjkutkylyi;lyi;yiolyi8o"  # Paste the ciphertext here
    ciphertext_bytes = bytes.fromhex(ciphertext_hex)
    try:
        decrypted_data_bytes = private_key.decrypt(
            ciphertext_bytes,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        decrypted_data_str = decrypted_data_bytes.decode()
        decrypted_data = eval(decrypted_data_str)

        print("Decrypted data:", decrypted_data)
    
    except Exception as e:
        print("Decryption failed:", str(e))
        
        
        
'''
BLOW FISH ENCRYPTION
'''

from Crypto.Cipher import Blowfish
from Crypto import Random
import json
import base64

# Create a Blowfish cipher object
def create_cipher(key):
    return Blowfish.new(key, Blowfish.MODE_ECB)

# Pad the data to a multiple of 8 bytes
def pad_data(data):
    block_size = 8
    padding = block_size - len(data) % block_size
    return data + bytes([padding] * padding)

# Remove padding from data
def unpad_data(data):
    padding = data[-1]
    return data[:-padding]

# Encrypt data with Blowfish
def encrypt_data(data, key):
    cipher = create_cipher(key)
    data = pad_data(data.encode())
    ciphertext = cipher.encrypt(data)
    return base64.b64encode(ciphertext).decode()

# Decrypt data with Blowfish
def decrypt_data(encrypted_data, key):
    cipher = create_cipher(key)
    ciphertext = base64.b64decode(encrypted_data.encode())
    data = cipher.decrypt(ciphertext)
    return unpad_data(data).decode()

# Main program
if __name__ == "__main__":
    # Secret key (must be between 4 and 56 bytes)
    key = b'secretk'

    data_json = json.dumps(record_data)
    encrypted_data = encrypt_data(data_json, key)
    decrypted_data_json = decrypt_data(encrypted_data, key)
    decrypted_data_dict = json.loads(decrypted_data_json)

    print("Original data (dictionary):", record_data)
    print("Encrypted data:", encrypted_data)
    print("Decrypted data (dictionary):", decrypted_data_dict)
    
    
'''
AES ENCRYPTION
'''
import json
from cryptography.fernet import Fernet

def AES_encr(record_data):
    # Generate a random AES key
    key = Fernet.generate_key()
    cipher_suite = Fernet(key)
    data_json = json.dumps(record_data).encode('utf-8')
    encrypted_data = cipher_suite.encrypt(data_json)
    decrypted_data_json = cipher_suite.decrypt(encrypted_data)
    decrypted_data_dict = json.loads(decrypted_data_json.decode('utf-8'))
    print("Original Data (Dictionary):", record_data)
    print("encryptedData (Dictionary):", encrypted_data)
    print("Decrypted Data (Dictionary):", decrypted_data_dict)
    
    
'''
PAILLIER ENCRYPTION
'''
import random
import math

# Paillier key generation
def generate_paillier_keypair(bit_length):
    p, q = generate_large_primes(bit_length)
    n = p * q
    g = n + 1
    lambda_n = least_common_multiple(p - 1, q - 1)
    mu = mod_inverse(L(g, lambda_n), n)
    public_key = (n, g)
    private_key = (lambda_n, mu)
    return public_key, private_key

def generate_large_primes(bit_length):
    p = q = 0
    while p == q:
        p = generate_prime(bit_length)
        q = generate_prime(bit_length)
    return p, q

def generate_prime(bit_length):
    prime_candidate = random.getrandbits(bit_length)
    while not is_prime(prime_candidate):
        prime_candidate = random.getrandbits(bit_length)
    return prime_candidate

def is_prime(n, k=5):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0:
        return False
    r, s = 0, n - 1
    while s % 2 == 0:
        r += 1
        s //= 2
    for _ in range(k):
        a = random.randrange(2, n - 1)
        x = pow(a, s, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True

def least_common_multiple(a, b):
    return abs(a * b) // math.gcd(a, b)

def mod_inverse(a, m):
    m0, x0, x1 = m, 0, 1
    while a > 1:
        q = a // m
        m, a = a % m, m
        x0, x1 = x1 - q * x0, x0
    return x1 + m0 if x1 < 0 else x1

# Encryption and Decryption
def encrypt(public_key, plaintext):
    n, g = public_key
    r = random.randint(1, n - 1)
    ciphertext = (pow(g, plaintext, n ** 2) * pow(r, n, n ** 2)) % (n ** 2)
    return ciphertext

def decrypt(private_key, public_key, ciphertext):
    n, _ = public_key
    lambda_n, mu = private_key
    L_c = L(pow(ciphertext, lambda_n, n ** 2), lambda_n)
    plaintext = (L_c * mu) % n
    return plaintext

def L(x, n):
    return (x - 1) // n
def encrypt_string(public_key, plaintext):
    plaintext_json = json.dumps(plaintext)
    ciphertext = encrypt(public_key, int(plaintext_json))
    return ciphertext

# Decrypt the ciphertext and deserialize back to a string
def decrypt_string(private_key, public_key, ciphertext):
    plaintext_json_int = decrypt(private_key, public_key, ciphertext)
    plaintext_json = str(plaintext_json_int)
    plaintext = json.loads(plaintext_json)
    return plaintext

# Example usage
bit_length = 128
public_key, private_key = generate_paillier_keypair(bit_length)
plaintext = "Hello, Paillier!"
ciphertext = encrypt_string(public_key, plaintext)
decrypted_plaintext = decrypt_string(private_key, public_key, ciphertext)

print("Original String:", plaintext)
print("Encrypted Data:", ciphertext)
print("Decrypted String:", decrypted_plaintext)

# Example usage
bit_length = 128
public_key, private_key = generate_paillier_keypair(bit_length)
plaintext = 'JGFVUHDIYHZSCKSHDVK JGUKS'
ciphertext = encrypt(public_key, plaintext)
decrypted_plaintext = decrypt(private_key, public_key, ciphertext)

print("Original Data:", plaintext)
print("Encrypted Data:", ciphertext)
print("Decrypted Data:", decrypted_plaintext)
