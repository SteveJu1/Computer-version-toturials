```
在神经网络的基础上加了 Convolution /Maxpoling
```
import tensorflow as tf
import numpy as np
mnist=tf.keras.datasets.mnist

num=np.array(training_images)
print(num.shape)

(train_set,train_label),(test_set,test_label)=mnist.load_data()
train_set=train_set.reshape(60000,28,28,1)         #将数据放在一个Tensor里，才能 convolution
train_set=train_set/255.0
test_set=test_set.reshape(60000,28,28,1) 
test_set=test_set/255.0

model=tf.keras.models.Sequential([
tf.keras.layers.Conv2D(64,(3,3),activation='relu', input_shape=(28, 28, 1)) , # 卷积层，64个3*3的卷积核，relu激活函数，input_shape可不加
tf.keras.layers.MaxPoling2D(2,2),                          # 2*2的最大池化
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(128,activation='relu')
tf.keras.layers.Dense(10,activation='softmax')
])

model.compile(optimizer='Adam',loss='sparse_categorical_crossentropy',metrics=['accuracy')]# metrics衡量指标 accuracy加[]
model.summary()              # 统计模型的信息 input_shape=(28, 28, 1)必须要
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 26, 26, 64)        640       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 13, 13, 64)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 11, 11, 64)        36928     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 5, 5, 64)          0         
_________________________________________________________________
flatten (Flatten)            (None, 1600)              0         
_________________________________________________________________
dense (Dense)                (None, 128)               204928    
_________________________________________________________________
dense_1 (Dense)              (None, 10)                1290      
=================================================================
```
####### 显示卷积层，池化层的输出
model.fit(train_set,train_label,epochs=5)
test_loss,test_acc=model.evaluate(test_set)     # valuate	英[ɪˈvæljueɪt]    model.evaluate会返回两个值，若只有一个变量接受会返回矩阵
                 
import matplotlib.pyplot as plt
f, axarr = plt.subplots(3,4)
FIRST_IMAGE=0
SECOND_IMAGE=23
THIRD_IMAGE=55
CONVOLUTION_NUMBER =4
from tensorflow.keras import models
layer_outputs = [layer.output for layer in model.layers]
activation_model = tf.keras.models.Model(inputs = model.input, outputs = layer_outputs)
for x in range(0,4):
  f1 = activation_model.predict(test_images[FIRST_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[0,x].imshow(f1[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[0,x].grid(False)
  f2 = activation_model.predict(test_images[SECOND_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[1,x].imshow(f2[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[1,x].grid(False)
  f3 = activation_model.predict(test_images[THIRD_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[2,x].imshow(f3[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[2,x].grid(False)
