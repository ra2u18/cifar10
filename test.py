import os
import pandas as pd
import tensorflow as tf
from constants import config

import numpy as np

from utils import test_gen, train_gen, valid_gen

allowed_extensions = ['jpg', 'jpeg', 'png']

def extract_image_names(filenames):
    image_names = []

    for f in filenames:
        _, image_name = f.split('/')
        if image_name.split('.')[1] in allowed_extensions: # check for validity of the filename
            image_names.append(image_name)
    
    return image_names


def test(validate=None):
    model = tf.keras.models.load_model('saved_model/my_model4')

    '''
    # first evaluate on the validation set
    if validate != None:
        print('Start validating the model on the validation set...')
        model.evaluate(valid_gen, steps=None,verbose=1)
        print('End validating the model on the validation set...')
    '''

    # Reset the classes of subfolders and ignore them
    test_gen.reset()
    print('Start predicting the model on the test set...')
    pred = model.predict(test_gen, steps=None, verbose=1)
    print('End predicting the model on the test set...')
    predicted_class_indices=np.argmax(pred,axis=1)

    labels = train_gen.class_indices
    labels = dict((v,k) for k,v in labels.items())
    predictions = [k for k in predicted_class_indices]

    # Check if we are within the write boundaries [0, 14]
    max_value = max(predicted_class_indices)
    min_value = min(predicted_class_indices)
    print(f'Test prediction boundaries ({min_value}, {max_value})')

    # Map categorical labels to original labels
    original_labels = [labels[p] for p in predictions]
    image_names = extract_image_names(test_gen.filenames)

    data_meta = {'test_data': image_names, 'category': original_labels}
    output = pd.DataFrame(data=data_meta)

    # Output the results to txt file
    if not os.path.isfile('run3.txt'):
        print('File doesn\'t exist, create run3.txt...')
        output.to_csv(r'run3.txt', header=None, index=None, sep=' ', mode='a')
    else:
        print('File run3.txt already exists')