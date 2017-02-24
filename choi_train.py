from dann import NN
import numpy as np
import argparse
import sys

'''
argument filename is input text file
i1;4205;0ct shuffles data when shuffle is 1

python choi_cross_valid.py --filename train_06_2.txt --shuffle 1
'''
parser = argparse.ArgumentParser()

parser.add_argument("--filename",help = "input file name: txt,tsv,numpy",type=str,default="data_use.txt")
parser.add_argument("--shuffle",help = "when 1 shuffles dataset (when 0 does not shuffle):",type = int,default =0)
#parser.add_argument("--outputfile",type=str, default="result.txt")
parser.add_argument("--layer",help = "NN size: input_size,layer1,layer2,.. layern,output",type = int ,nargs ='*', default = [300,100,20,10,2])
parser.add_argument("--activation",type = str, help="activations between layers: cannot be smaller than layers",nargs = '*',default = ['relu','relu', 'relu', 'softmax'])
parser.add_argument("--learningrate",help = "learning rate type is float" ,type = float,default = 0.00005)
parser.add_argument("--model_name",help = "model will be saved under this name",type = str, default = "test2")
parser.add_argument("--model_dir",help = "model will be saved to this folder",type = str, default = "models")
parser.add_argument("--max_epochs",help = "Number of Maximum Epoch:" ,type = int , default = 500)
parser.add_argument("--batch_size",type = float, default = 20)
#parser.add_argument("--remove_count",type = int, default = 300)
parser.add_argument("--loss",help = "Loss type:sqrt or cross entropy ",type=str , default = "sqrt")
parser.add_argument("--label_size",help = "size of labels : type int",type= int , default = 8)
args = parser.parse_args()


if args.filename.endswith(".txt") or args.filename.endswith(".tsv"):
    data_set = np.loadtxt(args.filename)
elif args.filename.endswith(".npy"):
    data_set = np.load(args.filename)

else:
    print "input file type is wrong "
    sys.exit(1)
#data_set = np.load("data_use.npy")



my_nn = NN(layer_size=args.layer, model_type='tf', activations= args.activation, softmax_flag=True,debug_flag = args.loss)
my_nn.train(data_set, learning_rate=args.learningrate, batch_size=args.batch_size, label_size=args.label_size, test_size=0, dir_name=args.model_dir, model_name=args.model_name, epoch_size=args.max_epochs, optimizer='AdamOptimizer')





