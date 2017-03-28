# -*- coding: utf-8 -*-
import os
import re
import codecs
import collections
import cPickle
import numpy as np
import tensorlayer as tl
import time

PAD = "_PAD"
GO = "_GO"
EOS = "_EOS"
UNK = "_UNK"
SPACE = " "
NEW_LINE = "\n"
W= "w"
UNK_ID = 3
START_VOCAB = [PAD, GO, EOS, UNK, SPACE, NEW_LINE,W]

def normalize_unicodes(text):
  text = normalize_punctuation(text)
  text = "".join([Q2B(c) for c in list(text)])
  return text


def replace_all(repls, text):
  # return re.sub('|'.join(repls.keys()), lambda k: repls[k.group(0)], text)
  return re.sub(u'|'.join(re.escape(key) for key in repls.keys()),
                lambda k: repls[k.group(0)], text)


def normalize_punctuation(text):
  cpun = [['	'],
          [u'﹗'],
          [u'“', u'゛', u'〃', u'′'],
          [u'”'],
          [u'´', u'‘', u'’'],
          [u'；', u'﹔'],
          [u'《', u'〈', u'＜'],
          [u'》', u'〉', u'＞'],
          [u'﹑'],
          [u'【', u'『', u'〔', u'﹝', u'｢', u'﹁'],
          [u'】', u'』', u'〕', u'﹞', u'｣', u'﹂'],
          [u'（', u'「'],
          [u'）', u'」'],
          [u'﹖'],
          [u'︰', u'﹕'],
          [u'・', u'．', u'·', u'‧', u'°'],
          [u'●', u'○', u'▲', u'◎', u'◇', u'■', u'□', u'※', u'◆'],
          [u'〜', u'～', u'∼'],
          [u'︱', u'│', u'┼'],
          [u'╱'],
          [u'╲'],
          [u'—', u'ー', u'―', u'‐', u'−', u'─', u'﹣', u'–', u'ㄧ']]
  epun = [u' ', u'！', u'"', u'"', u'\'', u';', u'<', u'>', u'、', u'[', u']', u'(', u')', u'？', u'：', u'･', u'•', u'~', u'|', u'/', u'\\', u'-']
  repls = {}

  for i in xrange(len(cpun)):
    for j in xrange(len(cpun[i])):
      repls[cpun[i][j]] = epun[i]

  return replace_all(repls, text)


def Q2B(uchar):
  """全角转半角"""
  inside_code = ord(uchar)
  if inside_code == 0x3000:
    inside_code = 0x0020
  else:
    inside_code -= 0xfee0
  #转完之后不是半角字符返回原来的字符
  if inside_code < 0x0020 or inside_code > 0x7e:
    return uchar
  return unichr(inside_code)

def read_words(filename):
    data=u''
    with codecs.open(filename,'r','utf-8') as f:
        # c=0
        # last_time=time.time()
        # for line in f:
        #     c+=1
        #     if c%1000==0:
        #         print c
        #         print time.time()-last_time
        #         last_time=time.time()
        #     line=normalize_unicodes(line)
        #     line=re.sub(r'\w+',W,line)
        #     line=re.sub(r' ','',line)
        #     print line
        #     data+=line
        data=f.read()
        data=normalize_unicodes(data)
        data=re.sub(r'\w+',W,data)
        data=line=re.sub(r' ','',data)
    return data

def bulid_vocab(data,threshold=0):    
    counter=collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    chars, counts = zip(*count_pairs)
    vocab=START_VOCAB+[c for i, c in enumerate(chars) if c not in START_VOCAB and counts[i] > threshold]
    word2idx=dict(zip(vocab,range(len(vocab))))
    return word2idx,vocab

def generate_data(filename,word2idx):
    data=read_words(filename)
    return tl.nlp.words_to_word_ids(data,word2idx,unk_key=UNK)

def ptb_iterator(raw_data, batch_size, num_steps):

    raw_data = np.array(raw_data, dtype=np.int32)

    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

    epoch_size = (batch_len - 1) // num_steps

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = data[:, i*num_steps:(i+1)*num_steps]
        y = data[:, i*num_steps+1:(i+1)*num_steps+1]
        yield (i,epoch_size,x, y)

class Reader(object):
    def __init__(self,data_path):
        self.train_file = os.path.join(data_path, 'train.txt')
        self.valid_file = os.path.join(data_path, 'valid.txt')
        self.test_file = os.path.join(data_path, 'test.txt') 

        data=read_words(self.train_file)
        self.word2idx,self.vocab=bulid_vocab(data)

        self.train_data=tl.nlp.words_to_word_ids(data,self.word2idx)
        self.valid_data=generate_data(self.valid_file,self.word2idx)
        self.test_data= generate_data(self.test_file,self.word2idx)

        print "len(Train_data):%d" % len(self.train_data)
        print "len(Valid_data):%d" % len(self.valid_data)
        print "len(Test_data):%d" % len(self.test_data)
        print "vocab_size:%d" % len(self.vocab)
        print "-------------------------------------------------------"

     


    def getVocabSize(self):
        return len(self.vocab)
    
    def yieldSpliceBatch(self,tag,batch_size,step_size):

        if tag=='Train':
            data=self.train_data
        elif tag=='Valid':
            data=self.valid_data
        else:
            data=self.test_data
            
        return ptb_iterator(data,batch_size,step_size)
if __name__ == '__main__':
    reader=Reader('zh_data')
    idx2word=tl.nlp.build_reverse_dictionary(reader.word2idx)
    print ' '.join([idx2word[x] for x in reader.test_data])



