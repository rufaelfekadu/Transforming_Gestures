from yacs.config import CfgNode as CN

_C = CN()

_C.NAME = "TG"
_C.DEBUG = True
_C.SEED = 42


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.NAME = "vit"
_C.MODEL.NUM_CLASSES = 2
_C.MODEL.IN_CHANNELS = 1
_C.MODEL.PATCH_SIZE = 2
_C.MODEL.INPUT_SIZE = 4
_C.MODEL.FRAMES = 150
_C.MODEL.FRAME_PATCHES = 4

_C.MODEL.TRANSFORMER = CN()
_C.MODEL.TRANSFORMER.N_LAYERS = 4
_C.MODEL.TRANSFORMER.D_MODEL = 128
_C.MODEL.TRANSFORMER.N_HEADS = 4
_C.MODEL.TRANSFORMER.D_FF = 2048
_C.MODEL.TRANSFORMER.DROPOUT = 0.25
_C.MODEL.TRANSFORMER.EMB_DROPOUT = 0.5
_C.MODEL.TRANSFORMER.POOL = "mean"
_C.MODEL.TRANSFORMER.CHANNELS = 1


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATA = CN()
_C.DATA.NAME = "emgleap"
_C.DATA.PATH = "data/"
_C.DATA.SEGMENT_LENGTH = 150
_C.DATA.STRIDE = 1
_C.DATA.LABEL_COLUMNS = []

_C.DATA.NORMALIZE = True
_C.DATA.ICA = False
_C.DATA.EXP_SETUP = 'exp1'
_C.DATA.EXP_SETUP_PATH = 'experiments.json'

# EMG
_C.DATA.EMG = CN()
_C.DATA.EMG.SAMPLING_RATE = 150
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
# Solver
# -----------------------------------------------------------------------------
_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER = "adam"
_C.SOLVER.LR = 1e-3
_C.SOLVER.WEIGHT_DECAY = 0
_C.SOLVER.BATCH_SIZE = 32
_C.SOLVER.NUM_EPOCHS = 10
_C.SOLVER.NUM_WORKERS = 4
_C.SOLVER.CHECKPOINT = "checkpoint/"
_C.SOLVER.LOG_DIR = "logs/"
_C.SOLVER.SAVE_DIR = "output/"
_C.SOLVER.RESUME = False


