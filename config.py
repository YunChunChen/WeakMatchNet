IMAGE_WIDTH = 240
IMAGE_HEIGHT = 240

TPS_GRID_SIZE = 3
TPS_REG_FACTOR = 0.2

NUM_OF_COORD = 100

MEAN = [0.485, 0.456, 0.406]
STDDEV = [0.229, 0.224, 0.225]

DATASET = 'PPM'

if DATASET == 'pf-pascal':
    TRAIN_DATA = 'train_data.csv'
    VAL_DATA = 'val_data.csv'
    TEST_DATA = 'test_data.csv'
    DATASET_PATH = 'path/to/dataset/root/directory'
    CSV_PATH = 'path/to/dataset/script/directory'
    NUM_OF_CLASS = 20

elif DATASET == 'pf-willow':
    TEST_DATA = 'test_data.csv'
    DATASET_PATH = 'path/to/dataset/root/directory'
    CSV_PATH = 'path/to/dataset/script/directory'
    NUM_OF_CLASS = 10
