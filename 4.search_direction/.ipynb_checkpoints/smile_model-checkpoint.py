# 导入相关包
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.autograd import Variable

import transforms as transforms
from skimage import io
from skimage.transform import resize
from models import *


# 加载模型
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

net = VGG('VGG19')
checkpoint = torch.load(os.path.join('checkpoint/', 'PrivateTest_model.t7'))
net.load_state_dict(checkpoint['net'])
net.cuda()
net.eval()


transform_test = transforms.Compose([
    transforms.TenCrop(44),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])

def predictExpression(image_np):
    # raw_img = io.imread(image_file)
    gray = np.dot(image_np[...,:3], [0.299, 0.587, 0.114])
    gray = resize(gray, (48,48), mode='symmetric').astype(np.uint8)
    img = gray[:, :, np.newaxis]

    img = np.concatenate((img, img, img), axis=2)
    img = Image.fromarray(img)
    inputs = transform_test(img)

    ncrops, c, h, w = np.shape(inputs)

    inputs = inputs.view(-1, c, h, w)
    inputs = inputs.cuda()

    outputs = net(inputs)

    outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops
    _, predicted = torch.max(outputs_avg.data, 0)
    
    label_id = int(predicted.cpu().numpy())
    
    score = F.softmax(outputs_avg,dim=0)
    
    score = score[label_id].cpu().item()
    
    
    label = class_names[label_id]
    
    return label,score
