from pathlib import Path
import sys
import torch
from tqdm import tqdm
from scipy.io.wavfile import write

CUR_DIR_PATH = Path(__file__).resolve()
ROOT = CUR_DIR_PATH.parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))# add ROOT to PATH



from Data.Datamodule.Dataset import DatasetLibrispeech

# dataloader = torch.utils.data.DataLoader(
#     DatasetLibrispeech,
#     batch_size=cfg["training"]["batch_size"],
#     shuffle=True,
#     num_workers=cfg["training"]["num_workers"]
#     )


generator = torch.Generator().manual_seed(42)
dataset = DatasetLibrispeech()
train_set, test_set = torch.utils.data.random_split(dataset, [0.8, 0.2], generator=generator)
n = 501
print(train_set[n].keys())
print(train_set[n]["stft"].shape)