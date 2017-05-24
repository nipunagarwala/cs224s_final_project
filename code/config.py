# Configuration class for SimpleEmgNN

class Config(object):
    ##########################
    # Checkpointing configuration
    ##########################
    checkpoint_dir = "checkpoints"
    tensorboard_dir = "tensorboard"
    steps_per_checkpoint = 1
    freq_of_longterm_checkpoint = 0.5     # in hours
    
    ##########################
    # Data
    ##########################
    train_path = "sample-data/train"
    test_path = "sample-data/train" # "sample-data/test"
    
    # What type of features to extract from the data
    # Valid feature_types: "wand_lda", "wand", "spectrogram"
    feature_type = "wand_lda"
       
    ##########################
    # Model architecture
    ##########################
    # Size of hidden layers in model
    hidden_size = 256
    
    # Number of hidden layers in model
    num_layers = 3
    
    # Cell type to use
    # Valid cell types: 'lstm', 'rnn', 'gru'
    cell_type = 'lstm' 
    
    
    ##########################
    # Training options
    ##########################
    # Number of passes to make over entire dataset
    num_epochs = 50
    # Number of examples in each batch (should be as 
    # large as successfully fits in memory)
    batch_size = 2 #32
    
    # Initial step size for Adam optimizer
    learning_rate = 1e-3
    # Weight on regularization cost
    l2_lambda = 1e-7
    
    # We clip by global norm, such that the 
    # maximum norm globally is scaled to this value
    max_norm = 10 
    
    # Number of predictions to generate
    beam_size = 10
