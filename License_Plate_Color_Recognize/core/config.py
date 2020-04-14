from pathlib2 import Path
from global_var import globalVars
from easydict import EasyDict as edict


__C                           = edict()
# Consumers can get config by: from config import cfg

cfg                           = __C

# Common options
__C.COMMON                    = edict()

__C.COMMON.MODEL_DIR_PATH     = globalVars.projectPath / Path('License_Plate_Color_Recognize', 'data', 'model')