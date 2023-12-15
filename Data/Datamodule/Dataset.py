from torchaudio.datasets import LIBRISPEECH
import torch
from class_transform import transforms



class DatasetLibrispeech():
    def __init__(self) -> None:
        self.LibriSpeech = LIBRISPEECH('./Data/Raw', 'dev-clean',download=True)
    def __len__(self):
        return len(self.LibriSpeech)
    def __getitem__(self,num):
        item = self.LibriSpeech[num]
        sample = {
            "Waveform":item[0],
            "Sample_rate":item[1],
            "Transcript":item[2],
            "Speaker_ID":item[3],
            "Chapter_ID":item[4],
            "Utterance_ID":item[5],
            }
        return transforms(sample)