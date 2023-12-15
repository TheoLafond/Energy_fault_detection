from torch import nn
import torch

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.relu = nn.ReLU()
        # Encoder
        # In the encoder, convolutional layers with the Conv2d function are used to extract features from the input image. 
        # Each block in the encoder consists of two convolutional layers followed by a max-pooling layer, with the exception of the last block which does not include a max-pooling layer.
        # -------
        # input: ~~314*513
        self.e11 = nn.Conv2d(1, 8, 3, padding=1) # output: [8, 513, 313]
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output: [8, 256, 156]

        # input: 
        self.e21 = nn.Conv2d(8, 16, kernel_size=3, padding=1) # output: [16, 256, 156]
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output: [16, 128, 78]

        # input: 
        self.e31 = nn.Conv2d(16, 32, kernel_size=3, padding=1) # output:[32, 128, 78]
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # output: [32, 64, 39]

        # input: 
        self.e41 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # output: [64, 64, 39]


        # Decoder
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(32, 32, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2 ,output_padding=(0,1))
        self.d21 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(16, 16, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2,output_padding=(1,0))
        self.d31 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(8, 8, kernel_size=3, padding=1)

        # Output layer
        self.outconv = nn.Conv2d(8, 1, kernel_size=1)
        self.sig = nn.Sigmoid()
    
    def forward(self, x):
        # Encoder
        xe1 = self.relu(self.e11(x))
        xp1 = self.pool1(xe1)

        xe2 = self.relu(self.e21(xp1))
        xp2 = self.pool2(xe2)

        xe3 = self.relu(self.e31(xp2))
        xp3 = self.pool3(xe3)

        xe4 = self.relu(self.e41(xp3))
        
        # Decoder
        xu1 = self.upconv1(xe4)
        xu11 = torch.cat([xu1, xe3], dim=-3)
        xd11 = self.relu(self.d11(xu11))
        xd12 = self.relu(self.d12(xd11))

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe2], dim=-3)
        xd21 = self.relu(self.d21(xu22))
        xd22 = self.relu(self.d22(xd21))

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe1], dim=-3)
        xd31 = self.relu(self.d31(xu33))
        xd32 = self.relu(self.d32(xd31))

        # Output layer
        out = self.sig(self.outconv(xd32))

        return out

if __name__=="__main__":

    my_tensor = torch.rand(1, 513, 314)
    mon_modele=UNet()
    mon_modele(my_tensor)