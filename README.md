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
  --layer 　NN layers size: input_size,layer1,layer2,.. layern,output
　　　　(学習ネットワーク層を定義する)
  --activation    activations between layers: cannot be smaller than
                number  of  layers
  --learningrate LEARNINGRATE
                        learning rate type is float（学習列）
  --model_name MODEL_NAME
              model will be saved under this name（保存されるモデルの名前定義）
  --model_dir MODEL_DIR
               model will be saved to this folder（モデルの保存されるフォルダーを定義）
  --max_epochs MAX_EPOCHS    Number of Maximum Epoch:（エポック数の定義）
  --batch_size BATCH_SIZE
  --loss LOSS           Loss type:sqrt or cross entropy(学習に使うロス関数を定義する)
  --label_size LABEL_SIZE
                        size of labels : type int (ラベルサイズです。)

使い方の例：
python choi_train.py --layer 223 200 100 20 8 --activation relu relu relu relu softmax  --model_name new_model --learningrate 0.00001 --max_epochs 2000 –-filename data_use.txt --label_size 8


choi_cross_valid.py の説明：
	データの５等分に分けて、4等分で学習、１分でvalidationを5回行うプログラムである。
ARGUMENTの説明：
　  -h, --help            show this help message and exit
  --filename FILENAME   input file: numpy txt tsv
  --shuffle SHUFFLE     shuffles training data when argument is nonzero
  --outputfile OUTPUTFILE
                        output file's name : out.txt
  --layer [LAYER [LAYER ...]]
                        layers of NN : input_size,layers[i]... output_size
  --activation [ACTIVATION [ACTIVATION ...]]
                        activations between layers : size can't be smaller
                        than number of the layers
  --learningrate LEARNINGRATE
                        learning rate : type float
  --model_dir MODEL_DIR
                        saves model to this directory
  --max_epochs MAX_EPOCHS
                        maximum number of epochs for each train
  --batch_size BATCH_SIZE
                        size of batch :
  --label_size LABEL_SIZE
                        size of label: equals to layer[-1]
  --loss LOSS           type of loss : cross_entropy or sqrt	
　

