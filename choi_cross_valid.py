
# -*- coding: utf-8 -*-
from dann import NN
import numpy as np
import argparse
import sys
from dann import nn1

'''
argument filename is input text file
it shuffles data when shuffle is 1

python choi_cross_valid.py --filename train_06_2.txt --shuffle 1
'''
parser = argparse.ArgumentParser()

parser.add_argument("--filename",help="input file: numpy txt tsv",type=str,default="data_use_2000.txt")
parser.add_argument("--shuffle",help = "shuffles training data when argument is nonzero",type = int,default =0)
parser.add_argument("--outputfile",help="output file's name : out.txt",type=str, default="result.txt")
parser.add_argument("--layer",help= "layers of NN : input_size,layers[i]... output_size",type = int ,nargs ='*', default = [294,100,20,10,8])
parser.add_argument("--activation",help = "activations between layers : size can't be smaller than number of the layers",type = str, nargs = '*',default = ['relu','relu', 'relu', 'sigmoid'])
parser.add_argument("--learningrate",help = "learning rate : type float",type = float,default = 0.0003)
parser.add_argument("--model_dir",help = "saves model to this directory",type = str, default = "models")
parser.add_argument("--max_epochs",help ="maximum number of epochs for each train",type = int,default=500)
parser.add_argument("--batch_size",help="size of batch :",type = float, default=40)
#parser.add_argument("--remove_count",type = int, default = 350)
parser.add_argument("--label_size",help = "size of label: equals to layer[-1]",type = int , default = 8)
parser.add_argument("--loss",help ="type of loss : cross_entropy or sqrt",type = str, default="sqrt")
args = parser.parse_args()

'''
filename 学習するテクストファイルの指定
shuffle 1の時データを　shuffle
outputfile 出力ファイルを指定
layer レイヤを指定、input_size,layer1,layer2 ... output_size
activation must be layerの数　relu..., last one should be  softmax
learningrate 学習率
model_dir モデル保存するフォルダー(特に必要ない、モデル保存しない)
max_epochs   epochs
 


'''
    

'''
n cross_valid    
'''

def predict(array):
        array1 = np.around(array,decimals =0)
        return array1


cross_valid = 5

if args.filename.endswith(".txt") or args.filename.endswith(".tsv"):
        data_set = np.loadtxt(args.filename)
elif args.filename.endswith(".npy"):
        data_set = np.load(args.filename)

else:
        print "Input file type is wrong "
        sys.exit(1)
#data_set = np.load("data_use_2000.npy")

'''
特徴量の　8以降を　10 で　掛け算する
'''

length = len(data_set)/cross_valid


def compute_accuracy(label,prediction):

        #print len(label),len(prediction)
        length = min(len(label),len(prediction))
        acc =0
        #print "right predictions [prediction][label]"
        for i in range(length):
                
                if np.array_equal(label[i],prediction[i]):
                        acc = acc +1
                        #print "right prediction right prediction"
                        #print prediction[i],label[i]
                        
                        
                else:
                        
                        print prediction[i],label[i]
        acc = float(acc)/float(length)
        return acc





if args.shuffle:
        # shuffles data                                                                                                                                                                                         
        data_set = data_set[np.random.permutation(data_set.shape[0]),:]
        print "shuffled shuffled shuffled shuffled shuffled"
    

f = open(args.outputfile,"w")

text = "run number %d accuracy is %f"


for i in range(cross_valid):
        testset = data_set[i*length:(i+1)*length]


        index = [j for j in range(i*length,(i+1)*length)]


        
        datasetUse = np.delete(data_set,index,axis = 0)

    

        my_nn = NN(layer_size=args.layer, model_type='numpy', activations= args.activation, softmax_flag=True,debug_flag = args.loss)

        label = testset[:,0:args.label_size]



        a=my_nn.train_with_valid(datasetUse,testset, learning_rate=args.learningrate, batch_size= args.batch_size, label_size=args.label_size, test_size=0, dir_name="models", model_name='test2', epoch_size=args.max_epochs, optimizer='AdamOptimizer')
    
      

        if args.activation[-1] == 'sigmoid':
                
                a = predict(a)
      
        accuracy = compute_accuracy(label,a)
        print "run number %d is accuracy is %f" % (i,accuracy)
    
        text_write = text %(i+1, accuracy)
        f.write(text_write)
        f.write("\n")
    

    
    
    
