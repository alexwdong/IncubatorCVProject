import torch
import torch.nn as nn

class BasicCNN_128x128(nn.Module):
    def __init__(self,num_classes):
        super(BasicCNN_128x128, self).__init__()
         
        in_channels= 3  
        conv1_out_channels= 16 
        conv2_out_channels = 16
        conv3_out_channels = 16
        conv4_out_channels = 16
        lin1_out_channels= 256
        lin2_out_channels = 128
        lin3_out_channels = 64
        lin4_out_channels = 32
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
        self.bn1 = nn.BatchNorm2d(conv1_out_channels)
        self.conv2 = nn.Conv2d(
            in_channels=conv1_out_channels, 
            out_channels=conv2_out_channels,
            kernel_size=kernel_size, 
            stride=stride,
            padding=padding,
            padding_mode='zeros'
            )

        self.bn2 = nn.BatchNorm2d(conv2_out_channels)
        self.pool1 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(
            in_channels=conv2_out_channels, 
            out_channels=conv3_out_channels,
            kernel_size=kernel_size, 
            stride=stride,
            padding=padding,
            padding_mode='zeros'
            )
        self.bn3 = nn.BatchNorm2d(conv3_out_channels)
        self.conv4 = nn.Conv2d(
            in_channels=conv3_out_channels, 
            out_channels=conv4_out_channels,
            kernel_size=kernel_size, 
            stride=stride,
            padding=padding,
            padding_mode='zeros'
            )
        self.bn4 = nn.BatchNorm2d(conv4_out_channels)
        self.pool2 = nn.MaxPool2d(2)
        
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features=16*16*conv4_out_channels,
                             out_features=lin1_out_channels)
        self.fc2 = nn.Linear(in_features=lin1_out_channels,
                             out_features=lin2_out_channels)
        self.fc3 = nn.Linear(in_features=lin2_out_channels,
                             out_features=lin3_out_channels)
        self.fc4 = nn.Linear(in_features=lin3_out_channels,
                             out_features=lin4_out_channels)
        self.fc5 = nn.Linear(in_features=lin4_out_channels,
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

    def forward(self, image):
        print('start')
        print(0,image.size())
        out = self.conv1(image)
        out =  self.bn1(out)
        out = self.conv2(out)
        out =  self.bn2(out)
        print(1,out.size())
        out = self.activation(out)
        print(2,out.size())
        out = self.pool1(out)
        #print(3,out.size())
        if self.dropout1 is not None:
            out = self.dropout2(out) 
        out = self.conv3(out)
        out =  self.bn3(out)
        out = self.conv4(out)
        out =  self.bn4(out)
        print(4,out.size())
        out = self.activation(out)
        print(5,out.size())
        out = self.pool2(out)
        print(6,out.size())
        if self.dropout2 is not None:
            out = self.dropout2(out)
        out = self.flatten(out)
        print(7,out.size())
        out = self.fc1(out)
        out = self.activation(out)
        out = self.fc2(out)
        #print(8,out.size())
        out = self.activation(out)
        out = self.fc3(out)
        out = self.activation(out)
        out = self.fc4(out)
        out = self.activation(out)
        out = self.fc5(out)
        out = self.softmax(out)
        return out 
    
