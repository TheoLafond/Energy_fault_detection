from pathlib import Path
import sys
import torch
from tqdm import tqdm
from scipy.io.wavfile import write

CUR_DIR_PATH = Path(__file__).resolve()
ROOT = CUR_DIR_PATH.parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))# add ROOT to PATH

from Config.test import cfg

from Data.Datamodule.Datamodule import MyDataModule

datamodule = MyDataModule(8,2)

datamodule.setup("train")
