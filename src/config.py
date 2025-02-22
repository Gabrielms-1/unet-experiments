import time

# Data paths
root_dir = "data/cityviews/preprocessed"

# Checkpoint directory
checkpoint_dir = 'data/model/checkpoints/'

# Current timestamp for saving models and checkpoints
current_time = time.strftime("%Y-%m-%d_%H-%M-%S")

# Model and training parameters
num_classes = 8
learning_rate = 0.001
epochs = 10 # Default value, can be overridden by command line arguments
batch_size = 64 # Default value, can be overridden by command line arguments

# Wandb settings
wandb_project = "cityviews-segmentation" 