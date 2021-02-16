from tensorflow.keras.preprocessing.image import ImageDataGenerator
from constants import config, classes, batch_size, resized_output, SEED

# ImageDataGenerator (in-place augmentation)
train_data_gen = ImageDataGenerator(rotation_range=50,width_shift_range=0.2,height_shift_range=0.2,zoom_range=0.3,
            horizontal_flip=True,vertical_flip=True,fill_mode='constant',cval=0,rescale=1./255)
valid_data_gen = ImageDataGenerator(rotation_range=45,width_shift_range=0.2,height_shift_range=0.2, zoom_range=0.3,
            horizontal_flip=True,vertical_flip=True,fill_mode='constant', cval=0, rescale=1./255)
test_data_gen = ImageDataGenerator(rescale=1./255)

train_gen = train_data_gen.flow_from_directory(config['train_path'], target_size=resized_output, batch_size=batch_size, classes=classes, class_mode='categorical',shuffle=True, seed=SEED)  
valid_gen = valid_data_gen.flow_from_directory(config['valid_path'],target_size=resized_output,batch_size=batch_size, classes=classes, class_mode='categorical', shuffle=False, seed=SEED)
test_gen = test_data_gen.flow_from_directory(config['test_path'],target_size=resized_output, batch_size=batch_size+2, shuffle=False,seed=SEED,class_mode=None,)

