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


#%%







#%%



























