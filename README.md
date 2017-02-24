REQUIREMENTS:

python2.7
tensorfllow-gpu=0.12.0-rc0



コードの説明:


　　choi_train.py 学習用なテクストファイルと学習パラメータをargumentで渡すと自動的に学習するプログラムである。
学習データの形：
　txt,tsv,or numpy:
    ラベルが先に入って、データが繋いでいる形です。間には,(”\t”) tabキーで離れています。
　　 label[0] + tab + label[1] …+ tab + data[0] + … data[n]　+ “\n”
次のデータの間では、newline 

ARGUMENTの説明：

 python choi_train.py -h でargumentの説明を表示する。
 
  -h, --help            show this help message and exit
  
  --filename FILENAME   input file name: txt,tsv,numpy
  
  --shuffle SHUFFLE     when 1 shuffles dataset (when 0 does not shuffle):
  
  --layer 　NN layers size: input_size,layer[1],layer[2],.. layer[n],output
　　　　(学習ネットワーク層を定義する)
    
  --activation    activations between layers: cannot be smaller than number  of  layers
  
  
  --learningrate   learning rate type is float（学習率）
  
  --model_name             model will be saved under this name（保存されるモデルの名前定義）
  
  --model_dir              model will be saved to this folder（モデルの保存されるフォルダーを定義）
  
  --max_epochs MAX_EPOCHS    Number of Maximum Epoch:（エポック数の定義）
  
  --batch_size バッチ 数
  
  --loss            Loss type:sqrt or cross entropy(学習に使うロス関数を定義する)
  --label_size 　size of labels : type int (ラベルサイズです。)

使い方の例：
python choi_train.py --layer 223 200 100 20 8 --activation relu relu relu relu softmax  --model_name new_model --learningrate 0.00001 --max_epochs 2000 –-filename data_use.txt --label_size 8


choi_cross_valid.py の説明：
	
	データの５等分に分けて、4等分で学習、１分でvalidationを5回行うプログラムである。

ARGUMENTの説明：

　  -h, --help            show this help message and exit
   
  --filename   input filename: numpy txt tsv
  
  --shuffle      shuffles training data when argument is nonzero
  
  --outputfile   output file's name : out.txt
  
  --layer    layers of NN : input_size,layers[i]... output_size
  
  --activation 　 activations between layers : size can't be smaller　than number of the layers
  
  --learningrate   learning rate : type float
  
  --model_dir   saves model to this directory
  
  --max_epochs   maximum number of epochs for each train
  
  
  --batch_size    size of batch :
  
  
  --label_size     size of label: equals to layer[-1]
  
  --loss          type of loss : cross_entropy or sqrt	
  
使い方：

python choi_cross_valid.py --layer 432 100 50 20 8  --activation  relu relu relu sigmoid --batch_size 100 --learningrate 0.0001 --shuffle 1 --max_epochs 20 --filename train8.tsv

choi_predict.pyの説明です。

optional arguments:

  -h, --help            show this help message and exit
  
  --filename FILENAME   input filename: txt,numpy,tsv
  
  --shuffle SHUFFLE     shuffles data when argument nonzero
  
  --layer  layers of model 
  
  --activation   activations between layers type = array of string
  
  --model_name  full path to saved model
  
  --label_size      length of label type int

使い方の例：
python choi_predict.py --layer 223 200 100 20 8 --activation relu relu relu relu sigmoid --model_name ./models/new_model.ckpt



