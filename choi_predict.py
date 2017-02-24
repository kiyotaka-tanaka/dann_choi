from dann import NN
import numpy as np

import argparse
import tensorflow as tf
import sys
parser = argparse.ArgumentParser()

parser.add_argument("--filename",help = "input filename: txt,numpy,tsv",type=str,default="data_use.txt")
parser.add_argument("--shuffle",help = "shuffles data when argument nonzero",type = int,default =0)
#parser.add_argument("--outputfile",type=str, default="result.txt")
parser.add_argument("--layer",help = "layers of model",type = int ,nargs ='*', default = [223,200,100,20,8])
parser.add_argument("--activation",help ="activations between layers",type = str, nargs = '*',default = ['relu','relu', 'relu', 'softmax'])
parser.add_argument("--model_name",help = "full path to model",type = str, default = "./models/test2.ckpt")
parser.add_argument("--label_size",help = "length of label",type = int , default = 8)

args = parser.parse_args()


def predicts(array):
    array1 = np.around(array,decimals=0)
    return  array1

def compute_accuracy(label,prediction):
    length = min(len(label),len(prediction))
    acc =0
    for i in range(length):
        if np.array_equal(label[i],prediction[i]):
            acc = acc +1

    acc = float(acc)/float(length)
    return acc



if args.filename.endswith(".txt") or args.filename.endswith(".tsv"):
    dataset = np.loadtxt(args.filename)
elif args.filename.endwith(".npy"):
    dataset = np.load(args.filename)
else:
    print "input file's type not right"
    sys.exit(1)
#dataset = np.load("data_use.npy")

label_size = args.layer[-1]

data = np.array(dataset[:,label_size:])



data_label = np.array(dataset[:,0:label_size])



x  = tf.placeholder('float',[None,args.layer[0]])





sess = tf.Session()



my_nn = NN(layer_size=args.layer, model_type='tf', activations= args.activation, softmax_flag=True,debug_flag = 'cross_entropy')

saver = tf.train.Saver()
saver.restore(sess,args.model_name)

predict = my_nn.feed_nn(x)


prediction = sess.run(predict,feed_dict = {x:data})




prediction = predicts(prediction)
if args.label_size == 0:
    print compute_accuracy(data_label,prediction)

#print prediction



