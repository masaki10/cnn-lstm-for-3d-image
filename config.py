INPUT_SHAPE = (32, 32, 32, 1)

IS_PRETRAINED = False

ROOT_DIR = "" # don't use now

EPOCH = 500

BATCH = 8

DATA_NUM = 0 # don't use now

FRAME = 5

TEST_RATE_FOR_ALL = 0.2
VAL_RATE_FOR_TRAIN = 0.2

TRAIN_DATA_CSV_PATH = "./csv_data/train_0.csv"
VAL_DATA_CSV_PATH = "./csv_data/val_0.csv"
TEST_DATA_CSV_PATH = "./csv_data/test_0.csv"
# TEST_DATA_CSV_PATH = "./csv_data/train_0.csv"

LOG_PATH = "./checkpoint/log.csv"

EARLY_STOPPING_MONITOR = "val_loss"

EARLY_STOPPING_PATIENCE = 500

CHECKPOINT_DIR = "./checkpoint"

CHECKPOINT_MONITOR = "val_loss"

CHECKPOINT_PERIOD = 1

# CNN_MODEL = "VGG16"
# CNN_MODEL = "VGG19"
CNN_MODEL = "ResNet50"
# CNN_MODEL = "ResNet101"
# CNN_MODEL = "EfficientNet"

LOSS_FUNCTION = "mean_absolute_error"

CONV_LSTM_FILTER_NUM = 128
CONV_LSTM_KERNEL_SIZE = 3

LEARNING_RATE = 0.001

FROM_CHECKPOINT = False
FROM_CHECKPOINT_PATH = "set this when want to train from checkpoint"

MODEL_PATH_FOR_PREDICT = "C:/Users/masuda/Downloads/result_by_2205_data_3/result_by_2205_data_3/resnet50/1e-3/checkpoint_0/weights.20-12.69.h5"
CSV_PATH_FOR_PREDICT = "C:/Users/masuda/Downloads/result_by_2205_data_3/result_by_2205_data_3/resnet50/1e-3/checkpoint_0/predict_train.csv"