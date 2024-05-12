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
_C.MODEL.FRAME_PATCHES = 75

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
_C.DATA.DATA_DIR = "data/"
_C.DATA.SEGMENT_LENGTH = 150
_C.DATA.STRIDE = 1
_C.DATA.LABEL_COLUMNS = []

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


