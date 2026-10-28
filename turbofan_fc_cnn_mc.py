
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, Reshape, LeakyReLU, BatchNormalization, Dropout,ReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt

class fc_cnn_mc:
    def __init__(self, x_train, y_train, x_test = None, y_test = None):

        # Declare input variables as instance variables
        self.x_train = x_train
        self.y_train = y_train
        self.x_test  = x_test
        self.y_test  = y_test
        
    def build_model(self, print_summary = False, lr = 0.0005, myalpha = 0.2):
        alpha_test = 0.2
        n_channels = self.y_train.shape[-1]
        self.model = Sequential([
            # 1st hidden layer
            Dense(256, input_dim = self.x_train.shape[1]), 
            #LeakyReLU(alpha=myalpha),
            #BatchNormalization(),             
            Dense(128), 
            #LeakyReLU(alpha=myalpha),
            #BatchNormalization(), 
            # 2nd hidden layer
            Dense(64),
            #LeakyReLU(alpha=myalpha),
            #BatchNormalization(), 
            # reshape layer
            Reshape((8, 8, 1)), 
   

            #0th conv layer
            Conv2DTranspose(256, kernel_size = (3, 3), strides = (1, 1), padding = 'same'),  
            LeakyReLU(alpha=myalpha),
            #BatchNormalization(),

            # 1st conv layer
            Conv2DTranspose(256, kernel_size = (3, 3), strides = (1, 1), padding = 'same'), 
            LeakyReLU(alpha=myalpha),
            #BatchNormalization(),
            # 2nd conv layer
            Conv2DTranspose(128, kernel_size = (3, 3), strides = (2, 2), padding = 'same'), 
            LeakyReLU(alpha=myalpha),
            #BatchNormalization(),
            # 3rd conv layer
            Conv2DTranspose(64, kernel_size = (3, 3), strides = (1, 1), padding = 'same'), 
            LeakyReLU(alpha=myalpha),
            #BatchNormalization(),
            # 4th conv layer
            Conv2DTranspose(32, kernel_size = (3, 3), strides = (2, 2), padding = 'same'), 
            LeakyReLU(alpha=myalpha),
            #BatchNormalization(),
            # 5th conv layer
            Conv2DTranspose(32, kernel_size = (3, 3), strides = (1, 1), padding = 'same'), 
            LeakyReLU(alpha=myalpha),
            #BatchNormalization(),
            # 6th conv layer
            Conv2DTranspose(16, kernel_size = (3, 3), strides = (2, 2), padding = 'same'),  
            LeakyReLU(alpha=myalpha),
            #BatchNormalization(),
            # 7th conv layer
            Conv2DTranspose(16,  kernel_size = (5, 5), strides = (1, 1), padding = 'same'), 
            LeakyReLU(alpha=myalpha),
            #BatchNormalization(),
            # output layer
            Conv2DTranspose(n_channels,  kernel_size = (3, 3), strides = (1, 1), padding = 'same')
        ])

        if type(lr) == list:
            lr_schedule = keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=lr[0],
                decay_steps=lr[1],
                decay_rate=lr[2])
            self.model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr_schedule) , metrics = ['mae'])
        else:
            self.model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr) , metrics = ['mae'])
        if print_summary == True:
            self.model.summary()
        return self.model

    
    def fit_model(self,ep = 3000, p = 70, bs = 128, ts = 0.15):
        es = EarlyStopping(monitor = 'val_loss', 
               mode = 'min', 
               verbose = 1, 
               patience = p)
        if self.x_test is not None and self.y_test is not None:
            new_X_test = self.x_test
            new_Y_test = self.y_test
        else:
            new_X_train, new_X_test, new_Y_train, new_Y_test = train_test_split(self.x_train,self.y_train, 
                                                                                test_size = ts,shuffle=True)
        self.model_history = self.model.fit(self.x_train, self.y_train,
                                      epochs = ep, 
                                      batch_size = bs, 
                                      #validation_data = (self.x_test,self.y_test),
                                      validation_data = (new_X_test,new_Y_test),      
                                      shuffle=True,
                                      verbose = 1, 
                                      callbacks = [es])
        
    def plot_loss(self): 
        plt.plot(self.model_history.history['loss'])
        plt.plot(self.model_history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper right')
        plt.show()

        plt.plot(self.model_history.history['mae'])
        plt.plot(self.model_history.history['val_mae'])
        plt.title('MAE')
        plt.ylabel('MAE')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper right')
        plt.show()        