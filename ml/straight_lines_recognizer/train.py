import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dense, Convolution2D, MaxPooling2D, Dropout, Flatten, ZeroPadding2D
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint

import data_gen
import validation_data

training_data, target_data, target_data_simple = data_gen.generate_samples(2000)

training_data = training_data.reshape(training_data.shape[0], 6, 6, 1)
target_data = target_data.reshape(target_data.shape[0], 6, 6, 1)
input_shape = (6, 6, 1)

model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=input_shape, activation='relu', border_mode='same'))
model.add(Dropout(0.25))
model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))
model.add(Dropout(0.25))
model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
model.add(Dropout(0.25))
model.add(Convolution2D(1, 1, 1, activation='sigmoid', border_mode='same'))

model.summary()

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model_checkpoint = ModelCheckpoint(filepath='saved_models/weights-loss_{val_loss:.2f}-acc_{val_acc:.2f}.hdf5')

(val_input_data, val_target_data, _) = validation_data.get_validation_data_tuple()
val_input_data = val_input_data.reshape(val_input_data.shape[0], 6, 6, 1)
val_target_data = val_target_data.reshape(val_target_data.shape[0], 6, 6, 1)

#model.load_weights("saved_models/same_shape_model_3/weights-loss_0.00-acc_1.00.hdf5")

history = model.fit(training_data,
                    target_data,
                    nb_epoch=1000,
                    batch_size=32,
                    shuffle='batch',
                    verbose=2,
                    validation_data = (val_input_data, val_target_data),
                    callbacks=[model_checkpoint]
                    )


np.set_printoptions(threshold=np.inf)

prediction = model.predict(val_input_data);
print "prediction"
print(prediction)
print "rounded prediction"
print(prediction.round())
