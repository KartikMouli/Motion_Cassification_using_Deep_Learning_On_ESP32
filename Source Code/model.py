import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Inputlayer, Dropout, ConvlD, Conv2D, Flatten, Reshape, MaxPoolinglD, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
sys.path .append('./resources/libraries')

# model architecture
model = Sequential()
model.add(Dense(20, activation='relu',
                activity_regularizer=tf.keras.regularizers.l1(0.00001)))
model.add(Dense(10, activation='relu',
                activity_regularizer=tf.keras.regularizers.l1(0.00001)))
model.add(Dense(classes, activation='softmax', name='y_pred'))

# this controls the learning rate
opt = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999)

# this controls the batch size,or you can manipulate the tf.data"
BATCH_SIZE = 32
train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=False)
validation_dataset = validation_dataset.batch(BATCH_SIZE, drop_remainder=False)
callbacks.append(BatchloggerCallback(BATCH_SIZE, train_sample_count))

# train the neural net1ork
model.compile(loss='categorical_crossentropy',
              optimizer=opt, metrics=['accuracy'])
model.fit(train_dataset, epochs=30, validation_data=validation_dataset,
          verbose=2, callbacks=callbacks)
