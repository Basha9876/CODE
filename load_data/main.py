import warnings
import tensorflow as tf
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
import pandas as pd
import block_chain
import SEllC
import base64
import IP_FS
import time
import psutil
import pickle

def main():
    
    
    cpu_before = psutil.cpu_percent()
    '''
    read dataset
    ''' 
    data = pd.read_csv('nsl-kdd/KDDTrain.csv')
    nsl_kdd_record = data[:10]
    
    '''
    load block chain
    '''
    start_time = time.time()
    # Initialize the blockchain
    blockchain = block_chain.Blockchain()
    blockchain.create_genesis_block()
    encrypted_datas=[]
    # Iterate through the DataFrame and add each record as a new block
    for _, row in nsl_kdd_record.iterrows():
        record_data = row.to_dict()
        
        '''
        encrypt data using Secret Elliptic curve cryptography  
        '''
        
        encr_data,sec_key,iv = SEllC.SEIICEncrypt(record_data)
        encrypted_datas.append([encr_data,sec_key,iv])
        # Convert the encrypted data to a base64-encoded string
        encr_data_str = base64.b64encode(encr_data).decode('utf-8')
        blockchain.add_block(encr_data_str)
        
    '''
    store encrypted data to Interplanetary File System 
    '''
    
    
    file_name = 'load_data/encrypt_datas.pkl'
    with open(file_name, 'wb') as file:
        pickle.dump(encrypted_datas, file)
        
    # Save the encrypted data to IPFS
    ipfs_hash = IP_FS.add_to_ipfs(file_name)
    print("\nsave data to ipfs successfully...\n")
    print(f"IPFS Hash: {ipfs_hash}")
    # write ipfs hash for retrieving contents
    file_path = "load_data/ipfs_hash.txt"
    with open(file_path, 'w') as file:
        file.write(ipfs_hash)
        
    '''
    calculate encryption decryption time , computational cost and network throughput  
    '''
        
    end_time = time.time()
    encryption_time = end_time - start_time
    cpu_after = psutil.cpu_percent()
    computational_cost = cpu_after - cpu_before
    throughput = len(encr_data_str) / encryption_time  # Bytes per second
    
    np.save('perf_vals.npy',[encryption_time,computational_cost,throughput])
        
    # Print the blockchain
    for block in blockchain.chain[:2]:
        print(f"Block {block.index}")
        print(f"Timestamp: {block.timestamp}")
        print(f"Data: {block.data}")
        print(f"Previous Hash: {block.previous_hash}")
        print(f"Hash: {block.hash}")
        print("\n")
        
#%%

'''
retrieve data from ipfs and decrypt data for classification
'''
with open("load_data/ipfs_hash.txt", 'r') as file:
        ipfs_hash = file.read()
        
print("\nRetrieve data from IPFS...\n")

encrypted_datas = IP_FS.retieve_data(ipfs_hash)


print("\nDecrypting datas...\n")


start_time = time.time()
decrypted_datas=[]

for i in encrypted_datas:
    decrypted_data = SEllC.SEIICDecrypt(i[0],i[1],i[2])
    decrypted_datas.append(decrypted_data)
    
final_data = pd.concat(decrypted_datas, ignore_index=True)
end_time = time.time()
decryption_time = end_time - start_time


#%%

'''
prepare data for classification
'''

import prep_data

feature,label = prep_data.prepare_data(final_data)

# split data for training and testing
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(np.array(feature), np.array(label), test_size=0.2, random_state=42,stratify=label)

'''
classification
'''
# Attention Bidirectional Gated unit assisted Residual network model 
import model
from keras.models import load_model
print("\nTesting...\n")
saved_model = load_model('Model')
clf = model.Att_BGR(X_train,y_train,X_test,y_test)
pred_vals = clf.predict(saved_model)

#%%

'''
calculate performance metrix and comparison
'''

import performance
    
cnf_matrix, Accuracy,precision,f1_score,recall,encryption_time,computational_cost,throughput = performance.Performancecalc(y_test,pred_vals)

print('\n_____Proposed Performance_____\n')
print('*** Att-BGR ***\n')
print('Accuracy  :',Accuracy)
print('Precision :',precision)
print('F1_score  :',f1_score)
print('Recall    :',recall)
print('Encryption Time    :',encryption_time)
print('Computational Cost :',computational_cost)
print('Network Throughput :',throughput)
print('Decryption Time    :',decryption_time)

import plot

plot.cnf_metx(cnf_matrix)
with open('load_data/history.pkl', 'rb') as file:
    loaded_history = pickle.load(file)
plot.ACCURACY_LOSS(loaded_history)
plot.plot_comparison(Accuracy,precision,f1_score,recall,encryption_time,computational_cost,throughput)
    
    
