```手写数字识别```
#########简单神经网络版###########
import tensorflow as tf
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self,epoch,log ={}):                        # on_epoch_end 
    if (log.get('acc')>0.99):                                  #log.get('')  从log里提取
      print('Reached 99% accuracy so cancelling training!')
      self.model.stop_training = True      
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train,x_test=x_train/255.0,x_test/255.0   
callbacks=myCallback()    
model = tf.keras.models.Sequential([
tf.keras.layers.Flatten(),                               #也可以写成 tf.keras.layers.Flatten(input_shape=(28,28))
tf.keras.layers.Dense(512,activation=tf.nn.relu),        #activation=tf.nn.relu  调TensorFlow包
tf.keras.layers.Dense(10,activation=tf.nn.softmax)     
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train,y_train,epochs=5,callbacks=[callbacks])
model.evaluate(x_test,y_test)
classfication=model.predict(x_test)                  # predict测试数据
print(classfication[1])                              # 测试数据第一个结果
print(y_test[1])

#########卷积神经网络版###########
import tensorflow as tf
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self,epoch,log={}):
    if (log.get('acc')>0.998):
      print('\nReached 99.8% accuracy so cancelling training!')
      self.model.stop_training=Ture

mnist = tf.keras.datasets.mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images=training_images.reshape(60000,28,28,1)
test_images=test_images.reshape(10000,28,28,1)
training_images,test_images=training_images/255.0,test_images/255.0
call=myCallback()
model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(32,(3,3),activation='relu', input_shape=(28, 28, 1)),
                                    tf.keras.layers.MaxPooling2D(2,2),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128,activation='relu'),
                                    tf.keras.layers.Dense(10,activation='softmax')
                                   ])

model.compile(optimizer='Adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.summary()

model.fit(training_images,training_labels,epochs=10, callbacks=[call])

model.evaluate(test_images)
classfication=model.predict(test_images)
