#Importing Libraries
import pandas as pd
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint


#Loading training set
train = pd.read_csv("../input/train.csv")
print(train.shape)
train.head()

#Loading Test Set
test= pd.read_csv("../input/test.csv")
print(test.shape)
test.head()

#Instantiating callbacks to objects
earlystopper = EarlyStopping(monitor='val_loss', min_delta=0,
                             patience=1, verbose=1, mode='auto')
checkpointer = ModelCheckpoint(filepath="/tmp/best.hdf5",
                               verbose=1, save_best_only=True)

#Splitting train from lables and scaling it
#Scaling test set
X_train = (train.ix[:,1:].values).astype('float32') # all pixel values
y_train = train.ix[:,0].values.astype('int32') # only labels i.e targets digits
X_test = test.values.astype('float32')

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

#converting label to 10 categories
y_train_cat = to_categorical(y_train, 10)

#Checking the status of what ever we have done
X_train.shape
X_test.shape
y_train_cat

#Model building
model = Sequential()

#Model Summary
'''
________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 26, 26, 32)        320       
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 24, 24, 32)        9248      
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 12, 12, 32)        0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 12, 12, 32)        0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 4608)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 128)               589952    
_________________________________________________________________
dropout_2 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 10)                1290      
=================================================================
Total params: 600,810
Trainable params: 600,810
Non-trainable params: 0
_________________________________________________________________
'''

#Fitting the model
model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
model.summary()
model.fit(X_train, y_train_cat, batch_size=32,epochs=15, verbose=1, validation_split=0.3, callbacks=[checkpointer,earlystopper])
predictions = model.predict_classes(X_test)


#Final output
submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),"Label": predictions})
submissions.to_csv("Mnist.csv", index=False, header=True)
