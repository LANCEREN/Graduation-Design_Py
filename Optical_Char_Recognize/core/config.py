from pathlib2 import Path
from global_var import globalVars
from easydict import EasyDict as edict


__C                           = edict()
# Consumers can get config by: from config import cfg

cfg                           = __C

# Common options
__C.COMMON                    = edict()

__C.COMMON.T_DIR_PATH       = globalVars.projectPath / Path('Optical_Char_Recognize', 'data', 'dataset')
__C.COMMON.F_DIR_PATH       = globalVars.projectPath / Path('Optical_Char_Recognize', 'data', 'dataset')
__C.COMMON.CH_DIR_PATH      = globalVars.projectPath / Path('Optical_Char_Recognize', 'data', 'dataset')
__C.COMMON.MODEL_DIR_PATH     = globalVars.projectPath / Path('Optical_Char_Recognize', 'data', 'model')
__C.COMMON.DETECTION_DIR_PATH     = globalVars.projectPath / Path('Optical_Char_Recognize', 'data', 'detection')