import tensorflow as tf
from tensorflow.keras.layers import Input, Bidirectional, LSTM,GRU,Flatten,SimpleRNN,Attention, Dense, Concatenate
from tensorflow.keras.models import Model
import numpy as np 
from keras.utils import to_categorical
import random as r 

class Existing:
    def __init__(self,X_train,y_train,X_test,y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.x_train = self.X_train[:,np.newaxis,:]
        self.x_test = self.X_test[:,np.newaxis,:]
        self.input_shape = self.x_train[0].shape
        self.num_classes = len(np.unique(self.y_train))
        self.y_train = to_categorical(self.y_train)
        

    def Att_BIGRU(self):
        
        input_layer = Input(shape=self.input_shape)
        bigru = Bidirectional(GRU(units=64, return_sequences=True))(input_layer)
        attention = Attention()([bigru, bigru])
        flatten = Flatten()(attention)
        outs = Dense(self.num_classes, activation='softmax')(flatten)
        model = Model(input_layer,outs)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        model.fit(self.x_train, self.y_train, epochs=10, batch_size=32)
        model.save('Models/att_bigru')
        pred_vals = model.predict(self.x_test)
        import performance           
        cnf_matrix, Accuracy,precision,f1_score,recall = performance.Performancecalc(self.y_test,pred_vals)
        vals = [Accuracy,precision,f1_score,recall]
        np.save('load_data/att_bigru.npy',vals)
        return model
        
    def BIGRU_RNN(self):

        input_layer = Input(shape=self.input_shape)
        bi_gru = Bidirectional(GRU(units=64, return_sequences=True))(input_layer)
        rnn = SimpleRNN(units=32, return_sequences=True)(input_layer)
        concatenated = Concatenate(axis=-1)([bi_gru, rnn])
        flatten = Flatten()(concatenated)
        output_layer = Dense(self.num_classes, activation='softmax')(flatten)
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        model.fit(self.x_train, self.y_train, epochs=10, batch_size=32)
        model.save('Models/bigru_rnn')
        pred_vals = model.predict(self.x_test)
        import performance            
        cnf_matrix, Accuracy,precision,f1_score,recall = performance.Performancecalc(self.y_test,pred_vals)
        vals = [Accuracy,precision,f1_score,recall]
        np.save('load_data/bigru_rnn.npy',vals)
        return model

    def Attn_RNN(self):

        input_layer = Input(shape=self.input_shape)
        rnn = SimpleRNN(units=64, return_sequences=True)(input_layer)
        attention = Attention()([rnn, rnn])
        flatten = Flatten()(attention)
        output_layer = Dense(self.num_classes, activation='sigmoid')(flatten)  
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.summary()
        model.fit(self.x_train, self.y_train, epochs=10, batch_size=32)
        model.save('Models/att_rnn')
        pred_vals = model.predict(self.x_test)
        import performance            
        cnf_matrix, Accuracy,precision,f1_score,recall = performance.Performancecalc(self.y_test,pred_vals)
        vals = [Accuracy,precision,f1_score,recall]
        np.save('load_data/att_rnn.npy',vals)
        return model
    
    def BILSTM(self):

        input_layer = Input(shape=self.input_shape)
        bi_lstm = Bidirectional(LSTM(units=64, return_sequences=True))(input_layer)
        output_layer = Dense(32, activation='sigmoid')(bi_lstm)
        flatten = Flatten()(output_layer)
        output_layer = Dense(self.num_classes, activation='sigmoid')(flatten) # Binary classification, so one neuron with sigmoid activation
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.summary()
        model.fit(self.x_train, self.y_train, epochs=10, batch_size=32)
        model.save('Models/bilstm')  
        pred_vals = model.predict(self.x_test)
        import performance            
        cnf_matrix, Accuracy,precision,f1_score,recall = performance.Performancecalc(self.y_test,pred_vals)
        vals = [Accuracy,precision,f1_score,recall]
        np.save('load_data/bilstm.npy',vals)
        
        return model
        
 
