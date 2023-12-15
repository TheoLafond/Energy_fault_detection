import torch

class CNN(torch.nn.Module):

    def __init__(self,stft_frequency_size):
        super(CNN, self).__init__()
        #1 channel input; 4 channels output; kernel on all frequencies stft_frequency_size over 3 temporal samples, no stride but padding horizontal 
        self.conv1 = torch.nn.Conv2d(1, 4, 3,stride=1,padding=1)
        #1 channel input; 8 channels output; kernel on all frequencies stft_frequency_size over 3 temporal samples, no stride but padding horizontal 
        self.conv2 = torch.nn.Conv2d(4, 8, 3,stride=1,padding=1)
        self.conv3 = torch.nn.Conv2d(8, 1, 3,stride=1,padding=1)
        self.relu = torch.nn.ReLU()
        self.sig = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.sig(self.conv3(x))
        return x

if __name__=="__main__":

    my_tensor = torch.rand(1, 513, 313)
    mon_modele=CNN(513)
    mon_modele(my_tensor)
