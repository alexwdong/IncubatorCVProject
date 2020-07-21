import torch
import torch.nn as nn


class TwoLayerCNN_w_features_64x64(nn.Module):
    def __init__(self,
                 num_classes,
                 with_features=False,
                 num_features=0,
                 dropout=None,
                 use_batchnorm=False,
                 ):
        super(TwoLayerCNN_w_features_64x64, self).__init__()
        #### Define the CNN parameters
        in_channels= 1  
        conv1_out_channels= 128 
        conv2_out_channels = 128
        lin1_out_channels= 64
        kernel_size = 5
        stride = 2
        padding = 2
        self.dropout = dropout
        activation = nn.ReLU(inplace = False)
        ### End define CNN Parameters
        ### Define some parameters used for managing using extra features
        ### dropout, and batch normalization
        self.with_features=with_features
        self.use_batchnorm = use_batchnorm
        self.activation = activation
        
        ### Define Layers that are always used
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=conv1_out_channels,
            kernel_size=kernel_size, 
            stride=stride,
            padding=padding,
            padding_mode='zeros'
            )
        self.conv2 = nn.Conv2d(
            in_channels=conv1_out_channels, 
            out_channels=conv2_out_channels,
            kernel_size=kernel_size, 
            stride=stride,
            padding=padding,
            padding_mode='zeros'
            )
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        
        self.flatten = nn.Flatten()
        if with_features:
            self.fc1 = nn.Linear(in_features=4*4*conv2_out_channels + num_features,
                             out_features=lin1_out_channels)
        else:
            self.fc1 = nn.Linear(in_features=4*4*conv2_out_channels,
                             out_features=lin1_out_channels)
        
        self.fc2 = nn.Linear(in_features=lin1_out_channels,
                             out_features=num_classes)
        self.softmax = nn.Softmax(dim=1)

        ### Now define optional layers
        if use_batchnorm:
            self.batchnorm_conv1 = nn.BatchNorm2d(num_features=conv1_out_channels)
            self.batchnorm_conv2 = nn.BatchNorm2d(num_features=conv2_out_channels)
            self.batchnorm_linear1 = nn.BatchNorm1d(num_features=lin1_out_channels)
        else: 
            self.batch_norm_conv1 = nn.Identity()
            self.batch_norm_conv2 = nn.Identity()
            self.batchnorm_linear1 = nn.Identity()

        if self.dropout is not None:
            if (self.dropout > 1 or self.dropout < 0) or type(self.dropout) is not float:

                raise ValueError("Give Dropout Probability between 0 and 1")    
            else:
                self.dropout1 = nn.Dropout(p = dropout, inplace = False)
                self.dropout2 = nn.Dropout(p = dropout, inplace = False)
        else:
            self.dropout1 = None
            self.dropout2 = None
        
        self.softmax = nn.Softmax(dim=1)
        

    def forward(self,image,features=None):
        
        out = self.conv1(image)
        #print(1,out.size())
        out = self.activation(out)
        #print(2,out.size())
        out = self.pool1(out)
        out = self.batchnorm_conv1(out)
        #print(3,out.size())
        if self.dropout1 is not None:
            out = self.dropout1(out) 
        out = self.conv2(out)
        #print(4,out.size())
        out = self.activation(out)
        #print(5,out.size())
        out = self.pool2(out)
        out = self.batchnorm_conv2(out)
        #print(6,out.size())
        if self.dropout2 is not None:
            out = self.dropout2(out)
        out = self.flatten(out)
        if features is not None:
            out = torch.cat((out,features),1)
        else: 
            pass
        out = self.fc1(out)
        out = self.activation(out)
        out = self.batchnorm_linear1(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out
