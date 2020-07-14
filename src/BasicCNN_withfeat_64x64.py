import torch
import torch.nn as nn

class BasicCNN_w_features_64x64(nn.Module):
    def __init__(self,num_classes):
        super(BasicCNN_w_features_64x64, self).__init__()

        in_channels= 3  
        conv1_out_channels= 8 
        conv2_out_channels = 8
        lin1_out_channels= 64
        kernel_size = 5
        stride = 2
        padding = 2
        dropout = None
        activation = nn.ReLU(inplace = False)
        
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
        self.fc1 = nn.Linear(in_features=4*4*conv2_out_channels + 10222*300,
                             out_features=lin1_out_channels)
        # infeat_size = (featmap_size + num_img * num_eigenvecs)
        self.fc2 = nn.Linear(in_features=lin1_out_channels,
                             out_features=num_classes)
        
        if dropout is not None:
            if (dropout > 1 or dropout < 0) or type(dropout) is not float:
                raise ValueError("Give Dropout Probability between 0 and 1")    
            else:
                self.dropout1 = nn.Dropout(p = dropout, inplace = False)
                self.dropout2 = nn.Dropout(p = dropout, inplace = False)
        else:
            self.dropout1 = None
            self.dropout2 = None
        
        self.activation = activation
        self.softmax = nn.Softmax(dim=1)


    def forward(self,image,features):
        
        out = self.conv1(image)
        #print(1,out.size())
        out = self.activation(out)
        #print(2,out.size())
        out = self.pool1(out)
        #print(3,out.size())
        if self.dropout1 is not None:
            out = self.dropout2(out) 
        out = self.conv2(out)
        #print(4,out.size())
        out = self.activation(out)
        #print(5,out.size())
        out = self.pool2(out)
        #print(6,out.size())
        if self.dropout2 is not None:
            out = self.dropout2(out)
        out = self.flatten(out)
        out = torch.cat((out,features),1)
        out = self.fc1(out)
        out = self.activation(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out
