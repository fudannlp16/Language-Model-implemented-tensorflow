#English and Chinese Language Model implemented tensorflow
The implementation contrains:
Recurrent Neural Network(LSTM)
Word-level English Language Model
Character-level Chinese Language Model

#Prerequistes
Python 2.7
Tensorflow 0.12
Tensorlayer(sudo pip install tensorlayer)

#Usage
To train a model with ptb dataset:
'''shell
python train.py --data_path=ptb_data
'''

To train a model with chinese dataset:
'''shell
python train.py --data_path=zh_data
'''

#Note
You can put the Chinese dataset in the zh_data,and split it to
train.txt,valid.txt and test.txt

#Author
fudannlp16 