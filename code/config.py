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
    num_classes = 28 #Can change depending on the dataset
    num_features = 100 #TO FIX!!!!
    
    ##########################
    # Model architecture
    ##########################
    hidden_size = 256
    num_layers = 3
    
    ##########################
    # Training options
    ##########################
    num_epochs = 1 #50
    batch_size = 2 #32
    lr = 1e-3
    l2_lambda = 1e-7
    
    # We clip by global norm, such that the 
    # maximum norm globally is scaled to this value
    max_norm = 10 

