import tensorflow as tf
import numpy as np
import time

class LSTMLM(object):
    def __init__(self, config, mode):

        self.config = config
        self.mode = mode

        if mode == "Train":
            self.is_training = True
            self.batch_size = self.config.train_batch_size
            self.step_size = self.config.train_step_size

        elif mode == "Valid":
            self.is_training = False
            self.batch_size = self.config.valid_batch_size
            self.step_size = self.config.valid_step_size 
        else:
            self.is_training = False
            self.batch_size = self.config.test_batch_size
            self.step_size = self.config.test_step_size 

        vocab_size = config.vocab_size
        embed_dim = config.word_embedding_dim     
        lstm_size = config.lstm_size              
        lstm_layers = config.lstm_layers          
        lstm_forget_bias = config.lstm_forget_bias
        batch_size = self.batch_size
        step_size = self.step_size

            
        # INPUTS and TARGETS
        self.inputs  = tf.placeholder(tf.int32, [batch_size, step_size]) 
        self.targets = tf.placeholder(tf.int32, [batch_size, step_size])
        
        with tf.device("/cpu:0"):
            # word_embedding
            self.word_embedding = tf.get_variable("word_embedding", [
                vocab_size, embed_dim])
            inputs = tf.nn.embedding_lookup(self.word_embedding, self.inputs)

        # INPUT DROPOUT 
        if self.is_training and self.config.dropout_prob < 1:
            inputs = tf.nn.dropout(inputs, keep_prob=config.dropout_prob)

        # Multi RNNCell
        lstm_cell=tf.nn.rnn_cell.BasicLSTMCell(lstm_size, forget_bias=lstm_forget_bias, state_is_tuple=True)
        if self.is_training and config.dropout_prob < 1:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell,output_keep_prob=config.dropout_prob)          
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell]*lstm_layers,state_is_tuple=True)
            
        self.initial_state=cell.zero_state(batch_size,dtype=tf.float32)
        
        #RNN
        inputs=tf.unstack(inputs,step_size,axis=1)
        outputs,state=tf.nn.rnn(cell,inputs,self.initial_state)

        
        # Softmax & loss
        output=tf.reshape(tf.concat(1,outputs),[-1,lstm_size])
        softmax_w = tf.get_variable("softmax_w", [lstm_size, vocab_size],dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [vocab_size],dtype=tf.float32)
        logits=tf.matmul(output,softmax_w)+softmax_b
        loss=tf.nn.seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(self.targets,[-1])],
            [tf.ones([batch_size*step_size],dtype=tf.float32)]
        )   
        self.cost=cost=tf.reduce_sum(loss/batch_size)
        self.final_state=state

        if self.is_training:
            self.lr = tf.Variable(0.0, trainable=False)
            self.global_step=tf.Variable(0.0,trainable=False)
            optimizer = tf.train.GradientDescentOptimizer(self.lr)
            tvars = tf.trainable_variables()
            grads = tf.gradients(cost,tvars)
            grads,_= tf.clip_by_global_norm(grads,config.max_grad_norm)
            self.eval_op = optimizer.apply_gradients(zip(grads, tvars),global_step=self.global_step)
        else:
            self.eval_op = tf.no_op()

    def update_lr(self, session, learning_rate):
        if self.is_training:
            session.run(tf.assign(self.lr, learning_rate))

    def run(self,session,reader,verbose=True):
        start_time=time.time()
        costs = 0.0
        iters = 0
        state=session.run(self.initial_state)

        fetches=[self.cost,self.final_state,self.eval_op]
        
        for batch in reader.yieldSpliceBatch(self.mode,self.batch_size,self.step_size):
            batch_id,batch_num,x,y= batch
            feed_dict={
                self.inputs:x,
                self.targets:y,
                self.initial_state:state
            }
            cost,state,_=session.run(fetches,feed_dict)
            costs+=cost
            iters+=self.step_size
              
            if verbose and (batch_id % max(10, batch_num // 10)) == 0:
                print "%.3f perplexity: %.3f speed: %.0f wps" % (batch_id*1.0/batch_num,np.exp(costs / iters),
                batch_id*self.batch_size/(time.time()-start_time))
                
        return costs,np.exp(costs/iters)
