# Package: Basic NN model wrapper based on TensorFlow
# Author : Baatarsuren Munkhdorj
# Contact: munkhdorj@data-artist.com 
#
# Copyright 2016 The DANN Authors. All Rights Reserved.
#
# Licensed under the Data-artist License, Version 1.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.data-artist.com/licenses/LICENSE-1.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf
import numpy as np 
import inspect

class NN():
    
    def __init__(self, layer_size=[16,128,64,16], activations=['relu', 'softsign', 'relu'], softmax_flag=False, model_type='tf', debug_flag='sqrt'):
        """
        Initializes a NN class.
        input:
            layer_size  : architecture of the model, [size of input, size of layer1,...,layerN, size of output](1d int list)
            activations : activation functions for each layer(list of string)
            softmax_flag: control flag for the usage of softmax in the output layer(boolean)
            model_type  : type of model used when saving/loading the model('numpy' or 'tf')
            debug_flag  : additional flag for development(string)
        """
        self.min_loss       = 99999
        self.sess           = tf.InteractiveSession()
        self.softmax_flag   = softmax_flag
        self.activations    = activations
        self.model_type     = model_type
        self.debug_flag     = debug_flag
        self.layers         = self.set_layers(layer_size)
        self.saver          = tf.train.Saver()
        self.x              = tf.placeholder('float',[None, layer_size[0]])
        self.y              = tf.placeholder('float',[None, layer_size[-1]])
        self.activation = activations
    def set_layers(self, layer_size):
        """
        Initializes, defines each layer of the model.
        input:
            layer_size: number of layers(int)
        """
        layers = []
        for num,size in enumerate(layer_size[1:], 1):
            #print "yyyyyyyyyyyyyyyyyyyy"
            #print layer_size[num-1],size
            layers.append({
                'weight'     : tf.Variable(tf.zeros([layer_size[num-1], size])),
                'bias'       : tf.Variable(tf.random_normal([size])),
                'activation' : getattr(tf.nn, self.activations[num-1])})

        return layers

    #def feed_nn(self,x):
        
    def feed_nn(self, data):
        """
        Feeds the model with data(2d numpy array).
        input: 
            data: input vectors(2d numpy array)
        """
        res = data
        #
        for layer in self.layers[:-1]:
            #print "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
            res = tf.add(tf.matmul(res, layer['weight']), layer['bias'])
            res = layer['activation'](res)
        res = tf.add(tf.matmul(res, self.layers[-1]['weight']), self.layers[-1]['bias'])
        if self.activation[-1] == "sigmoid":
            return tf.sigmoid(res)
        else:
            return tf.nn.softmax(res) if self.softmax_flag else res
    
    def train(self, dataset, learning_rate, batch_size, label_size, test_size, dir_name, model_name, epoch_size, optimizer='AdamOptimizer'):
        """
        Trains the model.
        input :  
            dataset       : training set(a 2d numpy array)
            learning_rate : learning_rate(float) 
            batch_size    : the precentage of batch size against the size of training set(float)
            label_size    : size of the label. First <label_size> elements of an input vector is assumed as the label vector(int)
            test_size     : the precentage of the size of test set against the size of training set(float)
            dir_name      : name of directory where the model will be saved(string)
            model_name    : name of the model(string)
            epoch_size    : epoch_size(int)
            optimizer     : the name of optimizer function(string)
        """
        #self.batch_size     = int(dataset.shape[0] * batch_size)
        self.batch_size =    batch_size
        self.train_lab      = dataset[:,0:label_size]
        self.train_data     = np.array(dataset[:,label_size:])
        self.learning_rate  = learning_rate
        self.optimizer      = getattr(tf.train, optimizer)
        
        self.define_loss()
        optimizer  = self.optimizer(learning_rate=self.learning_rate).minimize(self.loss)
        self.sess.run(tf.initialize_all_variables())

        for epochs in range(epoch_size):
            epoch_loss, i = 0.0, 0
            while i < len(self.train_data):
                batch_x, batch_y = self.mini_batch(self.train_data, self.train_lab, i)
                #print(batch_x.shape)
                _, c             = self.sess.run([optimizer, self.loss], feed_dict = {self.x:batch_x, self.y:batch_y})
                epoch_loss      += c
                i               += self.batch_size
            print ('Epoch', epochs , 'loss:' , epoch_loss)    
            if epoch_loss < self.min_loss: 
                self.min_loss  = epoch_loss
                self.save_params(dir_name, model_name, epoch_loss)

    def train_with_valid(self, dataset, testset,learning_rate, batch_size, label_size, test_size, dir_name, model_name, epoch_size, optimizer='AdamOptimizer'):
        '''
        for layer in self.layers[:-1]:
            print layer
            print "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        
        '''
        #print "layers "
        #print (self.layers[:-1])
        #print len(self.layers[:-1])
        """
        Trains the model.
        input :  
            dataset       : training set(a 2d numpy array)
            learning_rate : learning_rate(float) 
            batch_size    : the precentage of batch size against the size of training set(float)
            label_size    : size of the label. First <label_size> elements of an input vector is assumed as the label vector(int)
            test_size     : the precentage of the size of test set against the size of training set(float)
            dir_name      : name of directory where the model will be saved(string)
            model_name    : name of the model(string)
            epoch_size    : epoch_size(int)
            optimizer     : the name of optimizer function(string)
        """
        #self.batch_size     = int(dataset.shape[0] * batch_size)
        self.batch_size = batch_size
        self.train_lab      = dataset[:,0:label_size]
        self.train_data     = np.array(dataset[:,label_size:])
        self.test_data = testset[:,label_size:]
        self.test_lab = testset[:,0:label_size]
        self.learning_rate  = learning_rate
        self.optimizer      = getattr(tf.train, optimizer)
        
        self.define_loss()
        optimizer  = self.optimizer(learning_rate=self.learning_rate).minimize(self.loss)
        prediction = self.feed_nn(self.x)
        #correct_prediction = tf.equal(tf.argmax(prediction,1),tf.argmax(self.y,1))
        #accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
        
        #y_p = tf.argmax(prediction,1)

        
        self.sess.run(tf.initialize_all_variables())

        for epochs in range(epoch_size):
            epoch_loss, i = 0.0, 0
            while i < len(self.train_data):
                batch_x, batch_y = self.mini_batch(self.train_data, self.train_lab, i)
                #print(batch_x.shape)
                _, c             = self.sess.run([optimizer, self.loss], feed_dict = {self.x:batch_x, self.y:batch_y})
                epoch_loss      += c
                i               += self.batch_size
            print ('Epoch', epochs , 'loss:' , epoch_loss)
            
            if epoch_loss < self.min_loss: 
                self.min_loss  = epoch_loss
                #self.save_params(dir_name, model_name, epoch_loss)
        #argmz,acc = self.sess.run([y_p,accuracy],feed_dict={self.x:self.test_data,self.y:self.test_lab})
        #print argmz,acc
        
        a = self.sess.run(prediction,feed_dict={self.x:self.test_data})

        return a
        
    def define_loss(self):
        """
        Defines loss function.
        output:
            loss function
        """
        y = self.feed_nn(self.x)
        if self.debug_flag == 'sqrt':
            self.loss = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(self.y, y))))
        else:
            #cross_entropy = tf.nn.softmax_cross_entropy_with_logits(y, self.y, name='xentropy')
            cross = (self.y*tf.log(y))
            #cost = -tf.reduce_sum(y*tf.log(prediction))
            #self.loss  = -tf.reduce_sum(cross_entropy, name='xentropy_mean')
            #cost = -tf.reduce_sum(y*tf.log(prediction))
            self.loss = -tf.reduce_sum(cross)
    def save_params(self, model_dir, model_name, epoch_loss):
        """
        Saves the model identified by <directory name>/<model name>.
        input :  
            model_dir  : directory where the model saved.
            model_name : name of the model.
            epoch_loss : train loss of the time being.
        """
        save_path   = self.get_path(model_dir, model_name)
        if self.is_numpy():
            params  = []
            for layer in self.layers:
                params.append({
                    'weight'     : self.sess.run(layer['weight']),
                    'bias'       : self.sess.run(layer['bias']),
                    'activation' : layer['activation']})
            np.save(save_path, params)
        else:
            save_path = self.saver.save(self.sess, save_path)
        print ("Parameter saved", 'loss', epoch_loss, 'path', save_path)

    def load_model(self, model_dir, model_name):
        """
        Loads the model identified by <directory name>/<model name>.
        input :  
            model_dir  : directory where the model saved.
            model_name : name of the model.
        """
        if self.is_numpy():
            self.model = np.load(self.get_path(model_dir, model_name))
        else:
            self.saver.restore(self.sess, self.get_path(model_dir, model_name))
        print ("Model loaded", self.get_path(model_dir, model_name))

    def predict_numpy(self, data):
        """
        Returns the prediction result for multiple input vectors by using a numpy  model.
        input :  2d numpy array
        output:  2d numpy array
        """
        res = data
        for layer in self.model[:-1]:
            res = tf.add(tf.matmul(res, layer['weight']), layer['bias'])
            res = layer['activation'](res)
        res = tf.add(tf.matmul(res, self.model[-1]['weight']), self.model[-1]['bias'])
        return tf.nn.softmax(res) if self.softmax_flag else res

    def predict_one(self, x):
        """
        Returns the prediction result for single input vector by using the loaded model.
        input :  1d numpy array
        output:  1d numpy array
        """
        x   = np.array(x, dtype='float32')
        res = self.predict_numpy(np.append(x, x, axis=0)) if self.is_numpy() else self.feed_nn(np.append(x, x , axis=0))
        return res[0].eval()

    def predict_all(self, data):
        """
        Returns the prediction result for multiple input vectors by using the loaded model.
        input :  
            data: input vectors(2d numpy array)
        output:  
            result vectors(2d numpy array)
        """
        data = np.array(data, dtype='float32')
        res  = self.predict_numpy(data) if self.is_numpy() else self.feed_nn(data)
        return res.eval()

    def show_methods(self):
        """
        Prints the documentation for each method.
        """
        methods = inspect.getmembers(self, predicate=inspect.ismethod)
        for method in methods:
            #if method[0].endswith('__') or method[0].startswith('__'): continue
            self.show_help(method[0])

    def show_help(self, name):
        """
        Prints the documentation for the method.
        input:
            name: name of the method(string)
        """
        x = getattr(self, name)
        print(name + '()')
        print(x.__doc__)

    def mini_batch(self, data, label, i):
        """
        Produces mini batch from the training set.
        input: 
            data : training set(2d numpy array)
            label: label for the training set(2d numpy array)
            i    : starting index(int)
        output:
            mini batch(2d numpy array) 
        """
        index = (i* self.batch_size) % data.shape[0]
        return data[index:index + self.batch_size], label[index:index+self.batch_size] 

    def get_path(self, model_dir, model_name):
        """
        Retruns the path where the model saved/loaded.
        input: 
            model_dir : name of the directory(string)
            model_name: name of the model(string)
        output:
            path for the model(string)
        """
        dir_name    = model_dir if model_dir[-1] == '/' else model_dir + '/'
        return dir_name + model_name + '.npy' if self.is_numpy() else dir_name + model_name + '.ckpt'
    
    def is_numpy(self):
        """
        Checks if the model is saved/loaded as numpy array.
        output:
            answer(boolean)
        """
        return True if self.model_type == 'numpy' else False

    def variable_summaries(var):
        """
        Attach a lot of summaries to a Tensor (for TensorBoard visualization).
        """
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)
