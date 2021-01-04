# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 11:59:07 2020

@author: Andrea
@description: Autoencoder for MRI volume dataset
"""

import os
import numpy as np
from matplotlib import pyplot as plt
import SimpleITK as sitk

import glob

from keras.layers import Dense, Input, Conv3D,Conv2D, Conv3DTranspose, Conv2DTranspose, BatchNormalization, Activation, AveragePooling3D, MaxPooling3D, MaxPooling2D, Flatten, UpSampling3D, Concatenate#FILL ME# Which layer
from keras.models import Model, Sequential
from keras.losses import binary_crossentropy
from skimage.transform import rescale, resize


if __name__=="__main__":
    file = os.path.abspath('')#Put the urrent path
    AD_files = '\**\*AD*.nii'#find all nifti files with AD in the name
    AD_path = file + AD_files
    file = os.path.abspath('')
    AD_files = '\**\*CTRL*.nii'
    CTRL_path = file + AD_files
    CTRL_images = []
    AD_images = []
    CTRL_data = []
    AD_data = []
    
    for image in glob.glob(os.path.normpath(AD_path), recursive=True):#for loop in order to create a list of images
        AD_images.append(sitk.ReadImage(image, imageIO = "NiftiImageIO"))
    for image in glob.glob(os.path.normpath(CTRL_path), recursive=True):
        CTRL_images.append(sitk.ReadImage(image, imageIO = "NiftiImageIO"))
    CTRL_data = [sitk.GetArrayViewFromImage(x) for x in CTRL_images]
    CTRL_data_resized = np.array([resize(image, (128, 128,128),anti_aliasing=True) for image in CTRL_data])#resize images for Autoencoder
    AD_data = [sitk.GetArrayViewFromImage(x) for x in AD_images]
    AD_data_resized = np.array([resize(image, (128, 128,128),anti_aliasing=True) for image in AD_data])
#%%MAking labels
    zeros = np.array([0]*len(CTRL_data_resized))
    ones = np.asarray([1]*len(AD_data_resized))
    data_resized = np.append(CTRL_data_resized, AD_data_resized, axis = 0)
    labels = np.append(zeros, ones, axis = 0)
    data_resized, labels = data_resized[..., np.newaxis], labels[..., np.newaxis]
    #one way to permutate data but i can do it better with train_test_split
    #p = np.random.permutation(data_resized.shape[0])#shuffle AD cases with CTRL
    #data_resized, labels = data_resized[p,:,:,:], labels[p]
#%%
    x1 = np.linspace(0, CTRL_data_resized[0].shape[0]-1, CTRL_data_resized[0].shape[0])  
    y1 = np.linspace(0, CTRL_data_resized[0].shape[1]-1, CTRL_data_resized[0].shape[1])  
    z1 = np.linspace(0, CTRL_data_resized[0].shape[2]-1, CTRL_data_resized[0].shape[2])
    X, Y, Z = np.meshgrid(x1, y1, z1)#creating grid matrix

    #%%just to see if the resize is doing well
    import plotly
    import plotly.graph_objs as go
    from plotly.offline import download_plotlyjs, init_notebook_mode#these two lines are needed for graphic creation. (plotly works on ipython so we need to emulate it)
    init_notebook_mode()
    data_ein=np.einsum('ijk->jki', CTRL_data_resized[0])#here i swap the two directions "x" and "z" in order to rotate the image
    fig = go.Figure(data=go.Volume(
    x=X.flatten(),
    y=Y.flatten(),
    z=Z.flatten(),
    value=data_ein.flatten(),
    isomin=CTRL_data_resized[0].max()/2,#min value of isosurface ATTENTION: A bigger calcoulation time could bring the rendering to a runtime  error if we use the browser option
    isomax=CTRL_data_resized[0].max(),
    opacity=0.2, # needs to be small to see through all surfaces
    surface_count=15, # needs to be a large number for good volume rendering
    caps=dict(x_show=False, y_show=False, z_show=False)
    ))

    fig.show(renderer="browser") 

#%% Try edge detection for mask
    #FIRST of ALL: it takes directly the image
    threshold_filters= sitk.LaplacianSharpeningImageFilter()#first selection of the zones
    
    thresh_img = threshold_filters.Execute(CTRL_images[0])

    threshold_filters= sitk.UnsharpMaskImageFilter() #enhancement of the edges in order to set a more accurate threshold
    thresh_img = threshold_filters.Execute(thresh_img)
    #threshold_filters= sitk.YenThresholdImageFilter() #this is a good threshold too but it's a little blurry
    threshold_filters= sitk.RenyiEntropyThresholdImageFilter() # best threshold i could find
    threshold_filters.SetInsideValue(0)#
    threshold_filters.SetOutsideValue(1)#binomial I/O
    thresh_img = threshold_filters.Execute(thresh_img)
    fig, ax = plt.subplots()
    ax.imshow(sitk.GetArrayViewFromImage(CTRL_images[0])[:,50,:], cmap = 'Greys_r')
    ax.imshow(sitk.GetArrayViewFromImage(thresh_img)[:,50,:], alpha = 0.6, cmap='RdGy_r')
    data = sitk.GetArrayFromImage(thresh_img)

#%%Simple U-net model. The real U-net select a part of the layer in order to have the same dimension in "concatenation". I'll just use "same" in order to have always the dimension diviseble by 2
    input_shape = (128,128,128,1)
    inputs = Input(shape=input_shape)
    conv = Conv3D(32, kernel_size=(3,3,3),padding = 'same')(inputs)
    b_norm = BatchNormalization()(conv)
    act = Activation('relu')(b_norm)
    conv1 = Conv3D(64, kernel_size=(3,3,3),padding = 'same')(act)
    b_norm = BatchNormalization()(conv1)
    act = Activation('relu')(b_norm)
    pool = MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2))(act)
    
    conv = Conv3D(64, kernel_size=(3,3,3),padding = 'same')(pool)
    b_norm = BatchNormalization()(conv)
    act = Activation('relu')(b_norm)
    conv2 = Conv3D(128, kernel_size=(3,3,3),padding = 'same')(act)
    b_norm = BatchNormalization()(conv2)
    act = Activation('relu')(b_norm)
    pool = MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2))(act)
    
    conv = Conv3D(128, kernel_size=(3,3,3),padding = 'same')(pool)
    b_norm = BatchNormalization()(conv)
    act = Activation('relu')(b_norm)
    conv3 = Conv3D(256, kernel_size=(3,3,3),padding = 'same')(act)
    b_norm = BatchNormalization()(conv3)
    act = Activation('relu')(b_norm)
    pool = MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2))(act)
    
    conv = Conv3D(256,(3,3,3),padding = 'same')(pool)
    b_norm = BatchNormalization()(conv)
    act = Activation('relu')(b_norm)
    conv = Conv3D(512,(3,3,3),padding = 'same')(act)
    b_norm = BatchNormalization()(conv)
    act = Activation('relu')(b_norm)
    
    up = UpSampling3D(size=(2,2,2))(act)
    concat = Concatenate()([conv3, up])
    conv = Conv3D(256, kernel_size=(3,3,3),padding = 'same')(concat)
    b_norm = BatchNormalization()(conv)
    act = Activation('relu')(b_norm)
    conv = Conv3D(256, kernel_size=(3,3,3),padding = 'same')(act)
    b_norm = BatchNormalization()(conv)
    act = Activation('relu')(b_norm)
    
    up = UpSampling3D(size=(2,2,2))(act)
    concat = Concatenate()([conv2, up])
    conv = Conv3D(128, kernel_size=(3,3,3),padding = 'same')(concat)
    b_norm = BatchNormalization()(conv)
    act = Activation('relu')(b_norm)
    conv = Conv3D(128, kernel_size=(3,3,3),padding = 'same')(act)
    b_norm = BatchNormalization()(conv)
    act = Activation('relu')(b_norm)
    
    up = UpSampling3D(size=(2,2,2))(act)
    concat = Concatenate()([conv1, up])
    conv = Conv3D(64, kernel_size=(3,3,3),padding = 'same')(concat)
    b_norm = BatchNormalization()(conv)
    act = Activation('relu')(b_norm)
    conv = Conv3D(64, kernel_size=(3,3,3),padding = 'same')(act)
    b_norm = BatchNormalization()(conv)
    act = Activation('relu')(b_norm)
    outputs = Conv3D(1, kernel_size=(3,3,3),padding = 'same',activation='sigmoid')(act)
    
    model = Model(inputs=inputs, outputs=outputs)
    loss="binary_crossentropy"
    model.compile(loss=loss, optimizer='adam',metrics=['accuracy'])
    model.build(input_shape)
    model.summary()
    
    #%%Normal Conv-autoecoder
    input_shape = (128,128,1)
    input_tensor = Input(shape=input_shape)
    x = Conv2D(32, (5, 5), strides=2, padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling3D(pool_size=(2,2), strides=(2,2))(x)
    x = Conv2D(64, (3,3), strides=2,  padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling3D(pool_size=(2,2), strides=(2,2))(x)
    x = Conv2D(128, (3,3), strides=2, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling3D(pool_size=(2,2), strides=(2,2))(x)

    x = Conv2DTranspose(64, (3,3), strides=2,  padding='same', activation='relu')(x)
    x = Conv2DTranspose(32, (3,3), strides=2, padding='same',activation='relu')(x)
    x = Conv2DTranspose(32, (3,3), strides=2, padding='same',activation='relu')(x)
    out = Conv2D(1, (5,5), padding='same',activation='sigmoid')(x)
    model = Model(input_tensor, out)
    loss="binary_crossentropy"
    model.compile(loss=loss, optimizer='adam',metrics=['accuracy'])
    model.build(input_shape)
    model.summary()
    #%%Fitting the model
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(data_resized, labels, test_size=0.2, random_state=42)
    history = model.fit(X_train,X_train, validation_split=0.2, epochs=2)