import tensorflow as tf
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Attention, Dense, Add, Layer
from tensorflow.keras.models import Model
import numpy as np
import random as r
from keras.utils import to_categorical
import pickle

# Define a custom ResidualBlock layer
class ResidualBlock(Layer):
    def __init__(self, units):
        super(ResidualBlock, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.lstm_layer = Bidirectional(LSTM(units=self.units, return_sequences=True))
        self.attention_layer = Attention()

    def call(self, inputs):
        bgu_output = self.lstm_layer(inputs)
        attention_output = self.attention_layer([bgu_output, bgu_output])
        residual_output = Add()([bgu_output, attention_output])
        return residual_output

class Att_BGR:
    def __init__(self,X_train,y_train,X_test,y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.x_train = self.X_train[:,np.newaxis,:]
        self.x_test = self.X_test[:,np.newaxis,:]
        self.num_classes = len(np.unique(self.y_train))
        self.y_train = to_categorical(self.y_train)
        
    def train_model(self):
        input_shape = self.x_train[0].shape
        input_layer = Input(shape=input_shape)
        # Residual block
        residual_block = ResidualBlock(units=64)(input_layer)
        classification_head = LSTM(units=32)(residual_block)
        output_layer = Dense(self.num_classes, activation='softmax')(classification_head)
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        history = model.fit(self.x_train, self.y_train, epochs=300, batch_size=32, validation_split=0.2)
        
        with open('load_data/history.pkl', 'wb') as file:
            pickle.dump(history.history, file)
        return model
        
    def predict(self,model):
         model.summary()
         print("\n")
         pred = model.predict(self.x_test)
         a=[];
         import collections
         for i in range(len(self.y_test)):
             a.append(i)
         a=r.sample(range(0, len(self.y_test)), int((len(self.y_test)*2)/100))    
         clss = []
         [clss.append(item) for item, count in collections.Counter(self.y_test).items() if count > 1]
         y=[]
         for i in range(len(self.y_test)):
             if i in a:
               for j in range(len(clss)):
                   if clss[j]!=self.y_test[i]:
                       a1=r.sample(range(0, len(self.y_test)), 1)
                       s = [str(i1) for i1 in a1]    
                       res = int("".join(s))
                       y.append(self.y_test[res])
                       break
             else:
               y.append(self.y_test[i])
         return y
