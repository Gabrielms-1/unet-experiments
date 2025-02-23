import time

# Data paths
root_dir = "data/camvid/preprocessed"
device = "cpu"
# Checkpoint directory
checkpoint_dir = 'data/model/checkpoints/'
img_res = 720
# Current timestamp for saving models and checkpoints
current_time = time.strftime("%Y-%m-%d_%H-%M-%S")

# Model and training parameters
num_classes = 32
learning_rate = 0.001
epochs = 20 # Default value, can be overridden by command line arguments
batch_size = 4 # Default value, can be overridden by command line arguments

# Wandb settings
wandb_project = "camvid-segmentation" 