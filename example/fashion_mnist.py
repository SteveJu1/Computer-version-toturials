```
tesorflow 1.14.0 
run in colab.
dataset:fashion_mnist 
consisting of a training set of 60,000 examples and a test set of 10,000 examples.
Each example is a 28x28 grayscale image, associated with a label from 10 classes. 
```
import tesorflow as tf
print(tf.__version__)                             # tensorflow版本信息

mnist=tf.keras.datasets.fashion_mnist             #定义keras下的fashion_mnist数据集
(train_set,train_label),(test_set,test_label)=mnist.load_data()    #加载数据 可以用[] ,分为train_set 和 test_set

import matplotlib.pyplot as plt
plt.imshow(train_set[59999])               #显示图片 
print(train_set[59999])                    #显示矩阵数值
print(train_label[59999])

train_set=train_set/255.0              # ormalizing（归一化加快收敛）
test_set=test_set/255.0                #model.fit() 必须是浮点数
 


```
简单分为三层（有多少tf.keras.layers就有多少层）
Sequential: That defines a SEQUENCE of layers in the neural network
Flatten: Remember earlier where our images were a square, when you printed them out? 
Flatten just takes that square and turns it into a 1 dimensional set.
```
model=tf.keras.models.Squential( [tf.keras.layers.Flatten(),                           # 第1层
                                tf.keras.layers.Dense(256,activation='relu')           # 第2层 Dense表示神经元节点数，有256个。激活函数relu
                                tf.keras.layers.Dense(10,activation='softmax') ]       # 第3层 10对应10种类别，
``` 设置训练数据时的梯度下降函数，损失函数，迭代次数```
model.compile(optimizer=tf.train.AdaOptimizer(),
             loss='sparse_categorical_crossentropy'
             metrics=['accuracy']              #[]一定要为list
model.fit(train_set,train_label,epochs=5)

model.evaluate(test_set,test_label)    #test_set 精度

classifications = model.predict(test_images)  #所以testset的分类
print(classifications[0])
#第一个test集数据的分类结果，代表每个类别的概率 
[1.5650018e-07 6.9283632e-09 7.0401005e-09 1.1662392e-09 2.3912292e-09
 2.3528349e-03 2.7860409e-07 1.9346710e-02 9.2136581e-08 9.7829986e-01]
 ```
 




