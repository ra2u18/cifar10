from constants import config
import tensorflow as tf

from tensorflow.keras import Model 
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Dropout, Flatten,GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau

from utils import train_gen, valid_gen, test_gen
from constants import batch_size

import pickle

model_path = './saved_model/my_model4'

def resnet_model():
    ResNet_model = tf.keras.applications.ResNet152V2(weights='imagenet', include_top=False, input_shape=(200, 200, 3))

    # The last 15 layers fine tune
    for layer in ResNet_model.layers[:-15]:
        layer.trainable = False

    x = ResNet_model.output
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    x = Dense(units=512, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(units=512, activation='relu')(x)
    x = Dropout(0.3)(x)
    output  = Dense(units=15, activation='softmax')(x)
    model = Model(ResNet_model.input, output)

    print(model.summary())

    return model

def train():
    model = resnet_model()

    loss = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    # Compile model
    model.compile(optimizer=optimizer, loss=loss, metrics= ['accuracy'])

    lrr = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.4, min_lr=0.0001)
    # Create a callback that saves the model's weights every 5 epochs
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='./checkpoints', verbose=1, save_weights_only=True, save_freq=5*batch_size)

    callbacks = [lrr, cp_callback]

    # model fit_generator
    STEP_SIZE_TRAIN=train_gen.n//train_gen.batch_size
    STEP_SIZE_VALID=valid_gen.n//valid_gen.batch_size

    transfer_learning_history = model.fit_generator(generator=train_gen,steps_per_epoch=STEP_SIZE_TRAIN,
                        validation_data=valid_gen,validation_steps=STEP_SIZE_VALID,epochs=35,callbacks=callbacks)

    model.save(model_path)

    # save history of the model as a dictionary and plot loss, val_loss later on  
    with open('/tmp/trained-history', 'wb') as file_pi:
        pickle.dump(transfer_learning_history.history, file_pi)
        # history = pickle.load(open('/trainHistoryDict'), "rb") to load the history

    # model evaluate with validation set
    new_model = tf.keras.models.load_model(model_path)
    new_model.evaluate(valid_gen, steps=STEP_SIZE_VALID,verbose=1)