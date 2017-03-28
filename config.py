class SmallConfig(object):
    init_scale = 0.1
    learning_rate = 1.0
    decay = 0.5
    max_grad_norm = 5
    lstm_layers = 2
    lstm_size = 200
    word_embedding_dim =200
    lstm_forget_bias = 0.0
    max_epoch=4
    epoch_num =13
    dropout_prob = 1.0
    vocab_size=10004

    train_batch_size = 20
    train_step_size = 20
    valid_batch_size = 20
    valid_step_size = 20
    test_batch_size = 20
    test_step_size = 20

class MediumConfig(object):
    init_scale = 0.05
    learning_rate = 1.0
    decay = 0.8
    max_grad_norm = 5
    lstm_layers = 2
    lstm_size = 650
    word_embedding_dim =650
    lstm_forget_bias = 0.0
    max_epoch=6
    epoch_num =39
    dropout_prob = 0.5
    vocab_size=10004

    train_batch_size = 20
    train_step_size = 35
    valid_batch_size = 20
    valid_step_size = 35
    test_batch_size = 20
    test_step_size = 35

class LargeConfig(object):
    init_scale = 0.04
    learning_rate = 1.0
    decay = 0.8
    max_grad_norm = 10
    lstm_layers = 2
    lstm_size = 1500
    word_embedding_dim =1500
    lstm_forget_bias = 0.0
    max_epoch=14
    epoch_num =55
    dropout_prob = 0.35
    vocab_size=10004

    train_batch_size = 20
    train_step_size = 35
    valid_batch_size = 20
    valid_step_size = 35
    test_batch_size = 20
    test_step_size = 35
    
    
    
    
    

    