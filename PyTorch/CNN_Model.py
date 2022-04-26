# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 15:12:23 2022

@author: Monster
"""

# Veri Seti: LSI Far Infrared Pedestrian Dataset - https://e-archivo.uc3m.es/handle/10016/17370

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import time   # to calculate execution time

#%% Device Configuration 

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device: ', device)

#%%

def read_images(path, num_img):
    array = np.zeros([num_img, 64 * 32])
    
    i = 0
    for img in os.listdir(path):
        img_path = path + '\\' + img
        
        img = Image.open(img_path, mode = 'r')
        
        data = np.asarray(img, dtype = 'uint8')
        data = data.flatten()
        
        array[i, :] = data
        i += 1
        
    return array

#%% Read Train Images

train_negative_path = r'D:/dev/python/IR_Pedestrian/LSIFIR/Classification/Train/neg'
num_train_negative_image = 43390

# Read train negatives
train_negative_array = read_images(train_negative_path, num_train_negative_image)


# Numpy array to PyTorch array Conversion
x_train_negative_tensor = torch.from_numpy(train_negative_array)
print('x_train_negative_tensor: ', x_train_negative_tensor.size())

# Numpy array to PyTorch array Conversion
y_train_negative_tensor = torch.zeros(num_train_negative_image, dtype = torch.long)
print('y_train_negative_tensor: ', y_train_negative_tensor.size())


train_positive_path = r'D:/dev/python/IR_Pedestrian/LSIFIR/Classification/Train/pos'
num_train_positive_image = 10208

# Read train positives
train_positive_array = read_images(train_positive_path, num_train_positive_image)

# Numpy array to PyTorch array
x_train_positive_tensor = torch.from_numpy(train_positive_array)
print('x_train_positive_tensor: ', x_train_positive_tensor.size())

y_train_positive_tensor = torch.ones(num_train_positive_image, dtype = torch.long)
print('y_train_positive_tensor: ', y_train_positive_tensor.size())


# Concat Train Set
x_train = torch.cat((x_train_negative_tensor, x_train_positive_tensor), 0)  # 0, yukarıdan aşağı doğru birleştirme yapılacağını belirtir.
y_train = torch.cat((y_train_negative_tensor, y_train_positive_tensor), 0)

print('x_train size: ', x_train.size())
print('y_train size: ', y_train.size())


#%% Read Test Images


test_negative_path = r'D:/dev/python/IR_Pedestrian/LSIFIR/Classification/Test/neg'
num_test_negative_image = 22050

# Read train negatives
test_negative_array = read_images(test_negative_path, num_test_negative_image)


# Numpy array to PyTorch array Conversion
x_test_negative_tensor = torch.from_numpy(test_negative_array)
print('x_test_negative_tensor: ', x_train_negative_tensor.size())

# Numpy array to PyTorch array Conversion
y_test_negative_tensor = torch.zeros(num_test_negative_image, dtype = torch.long)
print('y_test_negative_tensor: ', y_train_negative_tensor.size())


test_positive_path = r'D:/dev/python/IR_Pedestrian/LSIFIR/Classification/Test/pos'
num_test_positive_image = 5944

# Read train positives
test_positive_array = read_images(test_positive_path, num_test_positive_image)

# Numpy array to PyTorch array
x_test_positive_tensor = torch.from_numpy(test_positive_array)
print('x_test_positive_tensor: ', x_train_positive_tensor.size())

y_test_positive_tensor = torch.ones(num_test_positive_image, dtype = torch.long)
print('y_test_positive_tensor: ', y_train_positive_tensor.size())


# Concat Train Set
x_test = torch.cat((x_test_negative_tensor, x_test_positive_tensor), 0)  # 0, yukarıdan aşağı doğru birleştirme yapılacağını belirtir.
y_test = torch.cat((y_test_negative_tensor, y_test_positive_tensor), 0)

print('x_test size: ', x_test.size())
print('y_test size: ', y_test.size())


#%% Visualize

# Karşılaşılan bi hata için
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

plt.imshow(x_train[45001, :].reshape(64, 32), cmap = 'gray')
plt.show()


#%% CNN model

# Hyperparameter
epochs = 5000
num_classes = 2
batch_size = 8933
learning_rate = 0.00001


class Net(nn.Module):   # nn.Module'dan inherit edilir
    def __init__(self):
        super(Net, self).__init__()
        
        # Parameters: in_channels, out_channels, kernel_size
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 16, 5)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(16 * 13 * 5, 520)
        self.fc2 = nn.Linear(530, 130)
        self.fc3 = nn.Linear(130, num_classes)
    
    def forward(self, x):
        
        # Convolution blogu + relu activation + maxpool
        a = self.pool1(F.relu(self.conv1(x)))
        b = self.pool1(F.relu(self.conv1(a)))
        
        # Flatten
        c = b.view(-1, 16 * 13 * 5)
        
        d = F.relu(self.fc1(c))
        e = F.relu(self.fc2(d))
        f = self.fc3(e)
        
        return f
        
        
#%%

# Modele vermek için TensorDataset oluşturuldu (bağımlı ve bağımsız değişkenleri birleştirir.)
train = torch.utils.data.TensorDataset(x_train, y_train)
train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = True)


test = torch.utils.data.TensorDataset(x_test, y_test)
test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = True)
        

#net = Net()                # CPU ile eğitim
net = Net().to(device)      # GPU kullanırken ekran kartına gönderilmesi gerekiyor. CPU için gerek yok    
    

#%% Loss and Optimizer

# Loss function
criterion = nn.CrossEntropyLoss()

# optimizer
import torch.optim as optim
optimizer = optim.SGD(net.parameters(), lr = learning_rate, momentum = 0.8)


#%% Training

start = time.time()

train_accuracy = []
test_accuracy = []
loss_list = []

use_gpu = True  # True  # GPU kullanılmayacaksa False

# train
for epoch in range(epochs):
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        
        # parameter 1, number of channel
        inputs = inputs.view(batch_size, 1, 64, 32) # reshape
        inputs = inputs.float()                     # float
        
        # use_gpu
        if use_gpu:
            if torch.cuda.is_available():
                inputs, labels = inputs.to(device), labels.to(device)   # inputs ve labels gpu'ya gönderilir.
        
    
        # zero gradient
        optimizer.zero_grad()

        # forward
        outputs = net(inputs)
        
        # loss
        loss = criterion(outputs, labels)
        
        # back
        loss.backward()
        
        # update weights
        optimizer.step()
        
    # Test
    correct = 0     # correct prediction number
    total = 0       # total number
    with torch.no_grad():   # backpropagation kapatılır, epoch sonu için.
        for data in test_loader:
            images, labels = data
            
            images = images.view(batch_size, 1, 64, 32)
            images = images.float()
            
            
            # use_gpu
            if use_gpu:
                if torch.cuda.is_available():
                    images, labels = images.to(device), labels.to(device)   # inputs ve labels gpu'ya gönderilir.
        
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            
    acc1 = 100 * correct / total
    print('Accuracy test: ', acc1)
    test_accuracy.append(acc1)
    
    
    # Train
    correct = 0     # correct prediction number
    total = 0       # total number
    with torch.no_grad():   # backpropagation kapatılır, epoch sonu için.
        for data in train_loader:
            images, labels = data
            
            images = images.view(batch_size, 1, 64, 32)
            images = images.float()
            
            s
            # use_gpu
            if use_gpu:
                if torch.cuda.is_available():
                    images, labels = images.to(device), labels.to(device)   # inputs ve labels gpu'ya gönderilir.
        
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            
    acc2 = 100 * correct / total
    print('Accuracy train: ', acc2)
    train_accuracy.append(acc2)
    
print('Train is Done')

end = time.time()
process_time = (end - start) / 60
print('Process Time: ', process_time)



#%%



























