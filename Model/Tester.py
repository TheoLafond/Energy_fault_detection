import torch
from class_transform import retour
from scipy.io.wavfile import write
import os
import numpy as np

class tester(object):
    def __init__(self, log_path,save = False):
        self.log_path = log_path
        self.save = save
    def __call__(self,batch,mask_out):
        snr = torch.zeros(batch["Waveform"].shape[0])
        for i_batch in range(batch["Waveform"].shape[0]):
            curr_path = os.path.join(self.log_path,f"{batch['Speaker_ID'][i_batch]}_{batch['Chapter_ID'][i_batch]}_{batch['Utterance_ID'][i_batch]}")
            output = batch["stft"][i_batch]*(mask_out[i_batch]>0.5)
            output = retour(output,len(batch["Waveform"][i_batch]))
            if self.save:
                sr = batch["Sample_rate"][i_batch]
                write(curr_path +'Waveform.wav',sr,batch["Waveform"][i_batch].numpy())
                write(curr_path +'Noised_Waveform.wav',sr,batch["Noised_Waveform"][i_batch].numpy())
                write(curr_path +'Output.wav',sr,output[0])
            snr[i_batch] = 10*np.log10((batch["Waveform"][i_batch].cpu()**2).sum()/((batch["Waveform"][i_batch].cpu()-output)**2).sum())
        return snr

class valider(object):
    def __call__(self,batch,mask_out):
        snr = torch.zeros(batch["Waveform"].shape[0])
        for i_batch in range(batch["Waveform"].shape[0]):
            curr_path = os.path.join(self.log_path,f"{batch['Speaker_ID'][i_batch]}_{batch['Chapter_ID'][i_batch]}_{batch['Utterance_ID'][i_batch]}")
            output = batch["stft"][i_batch]*(mask_out[i_batch]>0.5)
            output = retour(output,len(batch["Waveform"][i_batch]))
            snr[i_batch] = 10*np.log10((batch["Waveform"][i_batch]**2).sum()/((batch["Waveform"][i_batch]-output)**2).sum())
        return snr

if __name__ == "__main__" :
    pass