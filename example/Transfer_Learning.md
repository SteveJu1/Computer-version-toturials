[一文概览Inception家族的「奋斗史」](http://baijiahao.baidu.com/s?id=1601882944953788623&wfr=spider&for=pc)
```python

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import model
from tensorflow.keras.optimizers import RMSprop

# inception就是同时
from tensorflow.kera.appalication.inception_v3 import InceptionV3       #因为下载文件的是inception v3的weight，所以导出InceptionV3这个模块
local_weights_file='inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
pre_trained_model=InceptionV3(i)(input_shape=(150,150,3),
                 include_top=False,
                 weights=None)
                 
pre_trained_model.load_weights(local_weights_file)                 
# Make all the layers in the pre-trained model non-trainable
for layer in pre_trained_model.layers:
  layer.trainable = False
last_layer=pre_trained_model.get_layer('mixed7')

last_output=last_layer.output

# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(1024, activation='relu')(x)
# Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)                  
# Add a final sigmoid layer for classification
x = layers.Dense  (1, activation='sigmoid')(x)           

model = Model( pre_trained_model.input, x)          # .input
model.compile(optimizer = RMSprop(lr=0.0001), 
              loss = 'binary_crossentropy', 
              metrics = ['acc'])

model.summary()


```
