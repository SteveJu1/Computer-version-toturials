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

model.compile(optimizer='Adam',loss='sparse_categorical_crossentropy',metrics='accurary') # metrics衡量指标accurary（反映数据的拟合效果）
model.summary()              # 统计模型的信息
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

model.fit(train_set,train_label,epochs=5)
test_loss,test_acc=model.evaluate(test_set)     # valuate	英[ɪˈvæljueɪt]    model.evaluate会返回两个值，若只有一个变量接受会返回矩阵
