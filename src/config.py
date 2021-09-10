import torch

TRAINING_FILE = "../inputs/train.csv"
TRAINING_FILE_CLEAN = "../inputs/clean_train.csv"
MODEL_PATH = "../models/imdb_model.bin"
GLOVE_PARAMS = "../models/glove.6B.100d.txt"
FASTTEXT_PARAMS = "../models/wiki.simple.vec"
TRAIN_BATCH_SIZE = 75
TEST_BATCH_SIZE = 50
MAX_LENGTH = 256
EPOCHS = 20
LEARNING_RATE = 0.0003
DEVICE = torch.device("cuda")
