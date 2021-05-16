# -*- coding: utf-8 -*-
"""
Created on Fri May 14 00:38:20 2021

@author: priya
"""

import os
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


app = Flask(__name__)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(
            self, in_channels=1, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return torch.sigmoid(self.final_conv(x))

model = UNET(1, 1, [64, 128, 256, 512])
checkpoint = torch.load('C:/Users/priya/OneDrive/Desktop/SGMOD/my_checkpoint.tar', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['state_dict'])

app.secret_key = "super secret key"
BASE_DIR = 'C:/Users/priya/OneDrive/Desktop/SGMOD/'
UPLOAD_FOLDER = 'C:/Users/priya/OneDrive/Desktop/SGMOD/static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'image' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['image']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            image_location = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_location)
            test_img = Image.open(image_location).convert('L')
            test_img = np.array(test_img).reshape(1,1,test_img.size[0],test_img.size[1])
            pred = model(torch.tensor(test_img, dtype=torch.float32))
            pred = (pred>0.5).float().detach().numpy()
            pred = pred.reshape(101, 101)
            pred = Image.fromarray(pred).convert('L')
            plt.imshow(pred, cmap='binary')
            plt.savefig(os.path.join(BASE_DIR+'static/predictions', filename))
            return render_template('home.html', image_loc=filename)
    return render_template('home.html', image_loc=None)

if __name__=="__main__":
    app.run(port=12000, debug=True)