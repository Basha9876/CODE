import pickle
import socket
from pathlib import Path
import requests
from retry import retry
import urllib.request

sock = socket.socket()
host = socket.gethostname()
sock.bind(('', 0))
port=sock.getsockname()[1]



"""---------------------------------code to connect ipfs-------------------------------------------"""


def add_to_ipfs(filepath):
           
           
        with Path(filepath).open("rb") as fp:
            binary=fp.read()   ## converting the file to binary  
        """Saving encrpted data to ipfs url"""
        
        url="http://127.0.0.1:5001/api/v0/add" ## global gateway for ipfs
        
        response = requests.post(url, files={"file":binary}) ###generating  a request 
                                                             ## and uploading to  ipfs
        
        ipfs_hash=response.json()["Hash"] ## getting the hash to retrive the data
        filename=filepath.split("/")[-1:][0]
        file_url=f"https://ipfs.io/ipfs/{ipfs_hash}?filename={filename}"        
        
        return file_url

def retieve_data(ipfs_hash):
    
    with open('load_data/encrypt_datas.pkl', 'rb') as file:
        encrypted_datas = pickle.load(file)
        
    return encrypted_datas

    

