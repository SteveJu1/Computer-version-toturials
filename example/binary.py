``` run envoriment:google colab```
import tensorflow as tf
import os
import zipfile


!wget --no-check-certificate \
    "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/happy-or-sad.zip" \
    -O "/tmp/happy-or-sad.zip"
zip_ref = zipfile.ZipFile("/tmp/happy-or-sad.zip", 'r')
zip_ref.extractall("/tmp/h-or-s")
zip_ref.close()

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self,epoch,logs={}):
    if(logs.get('acc')>0.999):
      print('end')
      self.model.stop_training=True
callbacks = myCallback()

# This Code Block should Define and Compile the Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16,(3,3),input_shape=(150,150,3),activation='relu'),
                                    tf.keras.layers.MaxPooling2D(3,3),
                                    ####second conv
                                    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
                                    tf.keras.layers.MaxPooling2D(3,3),
                                    ####thrid conv
                                    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
                                    tf.keras.layers.MaxPooling2D(3,3),                                     
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(256,activation='relu'),
                                    tf.keras.layers.Dense(1,activation='sigmoid')
])
model.summary()
from tensorflow.keras.optimizers import RMSprop
model.compile(optimizer=RMSprop(lr=0.001),loss='binary_crossentropy',metrics=['acc'])

from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1/255)
train_generator = train_datagen.flow_from_directory(
        "/tmp/h-or-s",  
        target_size=(150, 150),        # 将加载的图像变成150*150
        batch_size=10,
        class_mode='binary')           #二分类

# This code block should call model.fit_generator and train for a number of epochs. 
history = model.fit_generator(
                            train_generator,
                            steps_per_epoch=2,
                            epochs=15,
                            verbose=1,
                            callbacks=[callbacks]
)
