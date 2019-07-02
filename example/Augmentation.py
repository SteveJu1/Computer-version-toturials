# Augmentation 数据增强
# imageDataGenerator
# Cats v Dogs classifier
#
import os
import zipfile
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
impot random
from shutil import copyfile

#########
!wget --no-check-certificate \
    "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip" \
    -O "/tmp/cats-and-dogs.zip"
local_zip = '/tmp/cats_and_dogs.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp')
zip_ref.close()
#########
print(len(os.listdir('/tmp/PetImages/Cat/')))
print(len(os.listdir('/tmp/PetImages/Dog/')))
########
try:
    os.mkdir('/tmp/cats-v-dogs')
    os.mkdir('/tmp/cats-v-dogs/training')
    os.mkdir('/tmp/cats-v-dogs/testing')
    os.mkdir('/tmp/cats-v-dogs/training/cats')
    os.mkdir('/tmp/cats-v-dogs/training/dogs')
    os.mkdir('/tmp/cats-v-dogs/testing/cats')
    os.mkdir('/tmp/cats-v-dogs/testing/dogs')
except OSError:
    pass

def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    files=[]
    for filename in os.listdir(SOURCE):
        file=SOURCE+filename      #字符串相加
        if os.path.getsize(file)>0：
            files.append(filename)
        else:
            print(filename + " is zero length, so ignoring.")
    #######        
     training_length = int(len(files)*SPLIT_SIZE)
     testing_length = int(len(files) - training_length)   
     random_set = random.sample(files, len(files))   
     training_set =random_set[0:training_length]
     testing_set =random_set[:testing_length]
        
     for filename in traning_set:
        this_file=SOURCE + filename
        destination=TRAINING + filename
        
        copyfile(this_file, destination)
CAT_SOURCE_DIR = "/tmp/PetImages/Cat/"
TRAINING_CATS_DIR = "/tmp/cats-v-dogs/training/cats/"
TESTING_CATS_DIR = "/tmp/cats-v-dogs/testing/cats/"
DOG_SOURCE_DIR = "/tmp/PetImages/Dog/"
TRAINING_DOGS_DIR = "/tmp/cats-v-dogs/training/dogs/"
TESTING_DOGS_DIR = "/tmp/cats-v-dogs/testing/dogs/"

split_size = .9
split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, TESTING_CATS_DIR, split_size)
split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, TESTING_DOGS_DIR, split_size)        
        
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=1e-4),
              metrics=['acc'])

```
#rotation_range is a value in degrees (0–180), a range within which to randomly rotate pictures.
#width_shift and height_shift are ranges (as a fraction of total width or height) within which to randomly translate pictures vertically or horizontally.
#shear_range is for randomly applying shearing transformations.
#zoom_range is for randomly zooming inside pictures.
``` 
##########              
train_datagen = ImageDataGenerator(rescale=1./255
,rotation_range=50,
 width_shift_range=0.2,
height_shift_range=0.2,
shear_range=0.2
zoom_range=0.2
horizontal_filp=True,  #horizontal_flip is for randomly flipping half of the images horizontally.
fill_mode='nearest' # filling in newly created pixels, which can appear after a rotation or a width/height shift.    
)

TRAINING_DIR = "/tmp/cats-v-dogs/training/"
train_generator = train_datagen.flow_from_directory(
        TRAINING_DIR,  # This is the source directory for training images
        target_size=(150, 150),  # All images will be resized to 150x150
        batch_size=20,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')
test_datagen = ImageDataGenerator(rescale=1./255)

VALIDATION_DIR = "/tmp/cats-v-dogs/testing/"
validation_generator = test_datagen.flow_from_directory(
        VALIDATION_DIR,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary') 
        
history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,  # 2000 images = batch_size * steps     steps_per_epoch可以不用写
      epochs=100,
      validation_data=validation_generator,
      validation_steps=50,  # 1000 images = batch_size * steps  validation_steps也可以不用写
      verbose=2)

import matplotlib.pyplot as plt
acc=history.history['acc']
val_acc = history.history['val_acc']
epochs = range(len(acc)) #可以直接给50
plt.plot(epochs,acc,'bo',label='Training accuracy')   #plt.plot(x轴，y轴，用点画，label标签) label要用 plt.legend()

plt.figure()
plt.show()


########### upload  your data
import numpy as np
from google.colab import files
from keras.preprocessing import image

uploaded = files.upload()

for fn in uploaded.keys():
 
  # predicting images
  path = '/content/' + fn
  img = image.load_img(path, target_size=(150, 150))     #改变加载的图片大小
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)

  images = np.vstack([x])
  classes = model.predict(images, batch_size=10)
  print(classes[0])
  if classes[0]>0.5:
    print(fn + " is a dog")
  else:
    print(fn + " is a cat")


