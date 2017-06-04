# Configuration class for SimpleEmgNN

class Config(object):
    ##########################
    # Checkpointing configuration
    ##########################
    checkpoint_dir = "checkpoints"
    tensorboard_dir = "tensorboard"
    tensorboard_prefix = "my_run"
    steps_per_checkpoint = 25
    freq_of_longterm_checkpoint = 0.5     # in hours
    
    ##########################
    # Reporting configuration
    ##########################
    # Frequency with which to print quant & qual 
    # monitoring information to stdout and tensorboard 
    steps_per_train_report = 1
    steps_per_dev_report = 10
    
    ##########################
    # Data
    ##########################
    train_path = "data/train"
    dev_path =   "data/dev"
    test_path =  "data/test"
    
    # What mode to use
    # Valid modes: None (meaning all), "audible", "whispered", "silent"
    mode = None
    
    # What type of features to extract from the data
    # Valid feature_types: 
    # - "wand": stacked time-domain features derived from DFT re: Wand papers
    # - "wand_lda": wand limited to the 12-dim subspace that best represents all triphone labels
    # - "wand_ldaa": wand limited to the 12-dim subspace that best represents *audible* triphone labels
    # - "spectrogram": discrete Fourier transform of each frame
    feature_type = "wand_lda"
    
    # What additional dummies to include:
    # None (no dummies), or a list with any of: 
    # ["speakerId", "speakerSess", "gender", "mode"]
    dummies = None#["speakerId", "speakerSess", "gender", "mode"]
    
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
    num_epochs = 1000
    # Number of examples in each batch (should be as 
    # large as successfully fits in memory)
    batch_size = 32
    
    # Initial step size for Adam optimizer
    learning_rate = 1e-3
    # Weight on regularization cost
    l2_lambda = 1e-7
    
    # We clip by global norm, such that the 
    # maximum norm globally is scaled to this value
    max_norm = 10 
    
    # Beam search: number of predictions to generate
    top_paths = 5

    # Beam search: beam width
    beam_width = 100
