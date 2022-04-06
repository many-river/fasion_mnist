import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
class FasionMnist():
    out_feature1 = 12
    out_feature2 = 24 #第二层卷积层输出通道数量（卷积核数）
    con_neuros = 512#全连接曾神经元数量
    def __init__(self,path):
        '''
        :param  path:指定数据集目录
        '''
        self.sess = tf.Session()
        self.data = read_data_sets(path,one_hot=True)
    def init_weight_variable(sef,shape):
        '''
        根据指定形状初始化权重
        shape：要初始化的变量的形状
        '''
        initial = tf.truncated_normal(shape,stddev=0.1)#截尾正态分布，数据分布不超过两个标准差范围
        return tf.Variable(initial)
    def init_bias_variable(self,shape):
        initial = tf.constant(1.0,shape = shape)
        return tf.Variable(initial)

    def conv2d(sef,x,w):
        '''
        二维卷积方法
        x:原始数据
        2：卷积核
        return 返回卷积运算结果

        '''
        #卷积核：【高度，宽度，输入通道，输出通道]
        return tf.nn.conv2d(x,w,strides=[1,1,1,1],#个纬度上的步长值
                            padding = 'SAME')#输入矩阵大小和输出一样
    def max_pool_2x2(self,x):
        '''
        定义池化方法
        '''
        return tf.nn.max_pool(x,ksize=[1,2,2,1],#池化区域大小
                        strides= [1,2,2,1],
                        padding='SAME')
    def create_conv_pool_layer(self,input,input_features,out_features):
        '''
        定义卷积，激活，池化层
        input原始数据
        input_feature输入特征数量
        return 卷积激活 池化计算结果
        '''
        filter = self.init_weight_variable([5,5,input_features,out_features])
        b_conv = self.init_bias_variable([out_features])
        h_conv = tf.nn.relu(self.conv2d(input,filter) + b_conv)#卷积激活计算
        h_pool = self.max_pool_2x2(h_conv)#对卷积结果池化
        return h_pool

    def creat_fc_layer(self,h_pool_flat,input_features,con_neurons):
        '''
         h_pool_flat:输入数据，经过拉伸后的一维张量
         input_features:输入特征数量
         con_neurons 神经元数量
         return 全链接计算后的结果
         '''
        w_fc = self.init_weight_variable([input_features,con_neurons])
        b_fc = self.init_bias_variable([con_neurons])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat,w_fc)+b_fc)
        return h_fc1

    def build(self):
        '''
        组建神经网络模型
        '''
        #定义输入数据、标签数据的占位符
        self.x = tf.placeholder(tf.float32,shape=[None,784])
        x_image = tf.reshape(self.x,[-1,28,28,1])#变纬成28*28单通道数据
        self.y = tf.placeholder(tf.float32,shape=[None,10])
        #第一组卷积池化
        h_pool1 = self.create_conv_pool_layer(x_image,1,self.out_feature1)
        #第二层
        h_pool2 = self.create_conv_pool_layer(h_pool1,self.out_feature1,#输入特征数量为上一层输出数量
                                            self.out_feature2)#输出特征数量
        #全链接
        h_pool2_flat_features = 7*7*self.out_feature2#计算特征点数量
        h_pool2_flat = tf.reshape(h_pool2,[-1,h_pool2_flat_features])#拉伸成一维
        h_fc = self.creat_fc_layer(h_pool2_flat,h_pool2_flat_features,self.con_neuros)
        #dropout(通过随即丢弃一定比例的神经元的更新，防止过拟合）
        self.keep_prob = tf.placeholder("float")#保存率
        h_fc1_drop = tf.nn.dropout(h_fc,self.keep_prob)
        #输出层
        w_fc = self.init_weight_variable([self.con_neuros,10])#512行10列
        b_fc = self.init_bias_variable([10])#10个偏置
        y_conv = tf.matmul(h_fc1_drop,w_fc) + b_fc#计算wx+b
        #计算准确率
        correct_prediction = tf.equal(tf.argmax(y_conv,1),
                                        tf.argmax(self.y,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        #损失函数
        loos_func = tf.nn.softmax_cross_entropy_with_logits(labels=self.y,
                                                            logits=y_conv)
        cross_entropy = tf.reduce_mean(loos_func)

        #优化
        optimizer = tf.train.AdamOptimizer(0.001)
        self.train_step = optimizer.minimize(cross_entropy)
    def train(self):
        self.sess.run(tf.global_variables_initializer())
        batch_size =100
        saver = tf.train.Saver()
        model_path = './tensor/fasion_mnist/fasion_mnist_model.ckpt'
        print("begin trainning")
        for i in range(10):
            total_batch = int(self.data.train.num_examples/batch_size)
            for j in range(total_batch):
                batch = self.data.train.next_batch(batch_size)
                params = {self.x:batch[0],
                self.y:batch[1],
                self.keep_prob:0.5}
                t, acc = self.sess.run([self.train_step,self.accuracy],params)
                if j%100 == 0:
                    print("i:%d,j:%d acc:%f"%(i,j,acc))
        save_path = saver.save(self.sess,model_path)
        print("模型以保存:",save_path)
    def eval(self,x,y,keep_prob):
        params = {self.x:x,self.y:y,self.keep_prob:0.5}
        test_acc = self.sess.run(self.accuracy,params)
        print("test accuracy:%f"%test_acc)
        
    def close(self):
        self.sess.close()
if __name__ == '__main__':
    mnist = FasionMnist("./fashion_mnist/")
    mnist.build()
    mnist.train()
    #评估
    xs, ys = mnist.data.test.next_batch(100)
    mnist.eval(xs,ys,0.5)
    mnist.close()

        
