from yacs.config import CfgNode as CN

_C = CN()

_C.NAME = "TG"
_C.DEBUG = True
_C.SEED = 42
_C.DEVICE = "cpu"
_C.LOG_DIR = "logs/"

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.NAME = "vit"
_C.MODEL.PROJ_DIM = 128
_C.MODEL.NUM_CLASSES = 2
_C.MODEL.IN_CHANNELS = 1
_C.MODEL.PATCH_SIZE = 2
_C.MODEL.INPUT_SIZE = 4
_C.MODEL.FRAMES = 200
_C.MODEL.FRAME_PATCH_SIZE = 4
_C.MODEL.EMB_DROPOUT = 0.5
_C.MODEL.OUTPUT_SIZE = 16

_C.TRANSFORMER = CN()
_C.TRANSFORMER.NUM_LAYERS = 4
_C.TRANSFORMER.D_MODEL = 128
_C.TRANSFORMER.NUM_HEADS = 4
_C.TRANSFORMER.DIM_HEAD = 32
_C.TRANSFORMER.MLP_DIM = 128
_C.TRANSFORMER.ATT_DROPOUT = 0.25
_C.TRANSFORMER.POOL = "mean"

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATA = CN()
_C.DATA.NAME = "emgleap"
_C.DATA.PATH = "dataset/emgleap"
_C.DATA.SEGMENT_LENGTH = 200
_C.DATA.STRIDE = 1
_C.DATA.LABEL_COLUMNS = []

_C.DATA.NORMALIZE = True
_C.DATA.ICA = False
_C.DATA.EXP_SETUP = 'exp1'
_C.DATA.EXP_SETUP_PATH = 'experiments.json'

# augmentations
_C.DATA.JITTER_SCALE = 1.1
_C.DATA.FREQ_PERTUB_RATIO = 0.1


# EMG
_C.DATA.EMG = CN()
_C.DATA.EMG.SAMPLING_RATE = 250
_C.DATA.EMG.NUM_CHANNELS = 16
_C.DATA.EMG.FEATURE_EXTRACTOR = "RMS"
_C.DATA.EMG.WINDOW_SIZE = 100
_C.DATA.EMG.WINDOW_STRIDE = 50
_C.DATA.EMG.HIGH_FREQ = 400
_C.DATA.EMG.LOW_FREQ = 20
_C.DATA.EMG.NORMALIZATION = "max"
_C.DATA.EMG.NOTCH_FREQ = 50
_C.DATA.EMG.BUFF_LEN = 0
_C.DATA.EMG.Q = 30

# -----------------------------------------------------------------------------
# Loss
# -----------------------------------------------------------------------------
_C.LOSS = CN()
_C.LOSS.TEMPRATURE = 1
_C.LOSS.USE_COSINE = True
_C.LOSS.METRIC = 'MAE'
_C.LOSS.LAMBDA = 0.25

# -----------------------------------------------------------------------------
# Solver
# -----------------------------------------------------------------------------
_C.SOLVER = CN()
#  optimizer
_C.SOLVER.OPTIMIZER = "adam"
_C.SOLVER.LR = 1e-3
_C.SOLVER.WEIGHT_DECAY = 0.0001

# scheduler
_C.SOLVER.PATIENCE = 10

_C.SOLVER.BATCH_SIZE = 128
_C.SOLVER.NUM_EPOCHS = 40
_C.SOLVER.NUM_WORKERS = 1
_C.SOLVER.CHECKPOINT = "checkpoint/"
_C.SOLVER.LOG_DIR = "logs/"
_C.SOLVER.SAVE_DIR = "output/"
_C.SOLVER.PRETRAINED_PATH = "output/exp1"
_C.SOLVER.RESUME = False


# -----------------------------------------------------------------------------
#  Visualization
# -----------------------------------------------------------------------------
_C.VIS = CN()
_C.VIS.PORT = 9000
_C.VIS.SLEEP_TIME = 0.1
_C.VIS.SAVE_TEST_SET = False

