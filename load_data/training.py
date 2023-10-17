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

### split data for training and testing ###
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(np.array(feature), np.array(label), test_size=0.2, random_state=42,stratify=label)

'''
classification
'''
#### Attention Bidirectional Gated unit assisted Residual network model ###
import model

print("\nStart training...\n")

clf = model.Att_BGR(X_train,y_train,X_test,y_test)
model = clf.train_model()
### save model ###
model.save('Model')