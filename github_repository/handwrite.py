from cgi import MiniFieldStorage
from tkinter.messagebox import NO
from tokenize import Triple
import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data
import pylab


minisit = input_data.read_data_sets("./tensor/minist",one_hot=True)

x = tf.placeholder(tf.float32,[None,784])

y = tf.placeholder(tf.float32,[None,10])#标签

#定义权重和偏置

w= tf.Variable(tf.random_normal([784,10]))
b = tf.Variable(tf.zeros([10]))

#构建模型，计算结果

pred_y  = tf.nn.softmax(tf.matmul(x,w)+b)
#损失函数

cross_entropy = -tf.reduce_sum(y*tf.log(pred_y),reduction_indices=1)
cost = tf.reduce_mean(cross_entropy)

optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

batch_size =100#每批读取100个样本
saver = tf.train.Saver()
model_path = './tensor/minist/minist_model.ckpt'

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #开始训练
    for epoch in range(200):
        total_batch = int(minisit.train.num_examples/batch_size)
        avg_cost = 0.0 #接收损失值
        for i in range(total_batch):
            #从训练集读取一个批次的样本
            batch_xs,batch_ys = minisit.train.next_batch(batch_size)
            params = {x:batch_xs,y:batch_ys}
            o,c = sess.run([optimizer,cost],feed_dict = params)
            avg_cost += (c/total_batch)#计算平均损失值
        print("epoch:%d, cost=%.9f"%(epoch+1,avg_cost))
    print("训练结束")
    #从测试集中取数据进行模型评估
    correct_pred = tf.equal(tf.argmax(pred_y,1),#预测结果中最大索引
                            tf.argmax(y,1))#真实结果最大索引
    #correct_pred返回的bool值
    accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
    print("accuracy:",accuracy.eval({x:minisit.test.images,
                                    y:minisit.test.labels}))
    save_path = saver.save(sess,model_path)
    print("模型以保存:",save_path)

#从测试集随即读取2张图像进行预测
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess,model_path)
    batch_xs,batch_ys = minisit.test.next_batch(2)
    output = tf.argmax(pred_y,1)
    output_val, predv = sess.run([output,pred_y],feed_dict={x:batch_xs})
    print("预测结果：\n",output_val,"\n")
    print("真是结果：\n",batch_ys,"\n")
    print("预测概率：\n",predv,"\n")
    #显示图片
    im = batch_xs[0]#第一个测试样本
    im = im.reshape(-1,28)
    pylab.imshow(im)
    pylab.show()

    im = batch_xs[1]#第2个测试样本
    im = im.reshape(-1,28)
    pylab.imshow(im)
    pylab.show()

