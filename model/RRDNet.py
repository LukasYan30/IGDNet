import torch
import torch.nn as nn

class RRDNet(nn.Module):
    def __init__(self):
        super(RRDNet, self).__init__()
        class DenseBlock(nn.Module):
            def __init__(self, in_channels, growth_rate, num_layers):
                super(DenseBlock, self).__init__()
                self.layers = nn.ModuleList([nn.Conv2d(in_channels + i * growth_rate, growth_rate, kernel_size=3, padding=1) for i in range(num_layers)])

            def forward(self, x):
                features = [x]
                for layer in self.layers:
                    out = layer(torch.cat(features, 1))
                    features.append(out)
                return torch.cat(features, 1)

        self.illumination_reflectance_net = nn.Sequential(
            nn.Conv2d(4, 16, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 4, 3, 1, 1),)
        
        self.illumination_net = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, 1, 1),
        )

        self.reflectance_net = nn.Sequential(
            DenseBlock(3, 16, 4),
            nn.LeakyReLU(),
            nn.Conv2d(67, 3, 3, 1, 1),
        )

        self.noise_net = nn.Sequential(
            DenseBlock(3, 16, 4),
            nn.LeakyReLU(),
            nn.Conv2d(67, 3, 3, 1, 1),
        )

    def concat(layers, axis=3):
        return torch.cat(layers, dim=axis)    
    
    def forward(self, input):

        illumination_0 , _= torch.max(input, 1) 
        illumination_0 = illumination_0.unsqueeze(1)

        input_illu = torch.cat((input,illumination_0),1) 
 
        illumination_reflectance = torch.sigmoid(self.illumination_reflectance_net(input_illu))
        reflectance_noise = illumination_reflectance[:,0:3,:,:]  

        illumination = torch.sigmoid(self.illumination_net(input))
        noise = torch.tanh(self.noise_net(reflectance_noise))   
        reflectance_clean = torch.sigmoid(self.reflectance_net(reflectance_noise))

        return illumination, reflectance_clean, noise 
