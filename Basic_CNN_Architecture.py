import torch
import torch.nn as nn

class BasicCNN(nn.Module):
    def __init__(self, 
            in_channels: int, 
            enc_channels: int, 
            out_channels: int,
            lin_channels: int, 
            num_classes : int, 
            kernel_size: int,
            stride: int,
            dropout = None,
            activation = nn.ReLU(inplace = False), 
            ):
        super(BasicCNN, self).__init__()
        self.conv1 = nn.Conv2d(
                    in_channels = in_channels, 
                    out_channels = enc_channels,
                    kernel_size = kernel_size, 
                    stride = stride
                    )
        self.conv2 = nn.Conv2d(
                    in_channels = enc_channels, 
                    out_channels = out_channels,
                    kernel_size = kernel_size, 
                    stride = stride
                    )
        self.pool1 = nn.MaxPool2d(enc_channels)
        self.pool2 = nn.MaxPool2d(out_channels)

        self.fc1 = nn.Linear(in_features = out_channels, out_features = lin_channels)
        self.fc2 = nn.Linear(in_features = lin_channels, out_features = num_classes)

        if dropout is not None:
            if (dropout > 1 or dropout < 0) or type(dropout) is not float:
                raise ValueError("Give Dropout Probability between 0 and 1")    
            else:
                self.dropout1 = nn.Dropout(p = dropout, inplace = False)
                self.dropout2 = nn.Dropout(p = dropout, inplace = False)


        self.activation = activation


    def forward(self, x):
        out = self.conv1(x)
        out = self.activation(out)
        out = self.pool1(out)\
        if self.dropout1 is not None:
            out = self.dropout2(out) 
        out = self.conv2(out)
        out = self.activation(out)
        out = self.pool2(out)
        if self.dropout2 is not None:
            out = self.dropout2(out) 
        out = self.fc1(out)
        out = self.fc2(out)

        return out