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
    file = os.path.abspath('')#Put the current path
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
    # CTRL_data = [sitk.GetArrayViewFromImage(x) for x in CTRL_images]
    # CTRL_data_resized = np.array([resize(image, (128, 128,128),anti_aliasing=True) for image in CTRL_data])#resize images for Autoencoder
    # AD_data = [sitk.GetArrayViewFromImage(x) for x in AD_images]
    # AD_data_resized = np.array([resize(image, (128, 128,128),anti_aliasing=True) for image in AD_data])
#%%MAking labels
    
    #one way to permutate data but i can do it better with train_test_split
    #p = np.random.permutation(data_resized.shape[0])#shuffle AD cases with CTRL
    #data_resized, labels = data_resized[p,:,:,:], labels[p]
# #%%
#     x1 = np.linspace(0, CTRL_data_resized[0].shape[0]-1, CTRL_data_resized[0].shape[0])  
#     y1 = np.linspace(0, CTRL_data_resized[0].shape[1]-1, CTRL_data_resized[0].shape[1])  
#     z1 = np.linspace(0, CTRL_data_resized[0].shape[2]-1, CTRL_data_resized[0].shape[2])
#     X, Y, Z = np.meshgrid(x1, y1, z1)#creating grid matrix

    # #%%just to see if the resize is doing well
    # import plotly
    # import plotly.graph_objs as go
    # from plotly.offline import download_plotlyjs, init_notebook_mode#these two lines are needed for graphic creation. (plotly works on ipython so we need to emulate it)
    # init_notebook_mode()
    # data_ein=np.einsum('ijk->jki', CTRL_data_resized[0])#here i swap the two directions "x" and "z" in order to rotate the image
    # fig = go.Figure(data=go.Volume(
    # x=X.flatten(),
    # y=Y.flatten(),
    # z=Z.flatten(),
    # value=data_ein.flatten(),
    # isomin=CTRL_data_resized[0].max()/2,#min value of isosurface ATTENTION: A bigger calcoulation time could bring the rendering to a runtime  error if we use the browser option
    # isomax=CTRL_data_resized[0].max(),
    # opacity=0.5, # needs to be small to see through all surfaces
    # surface_count=15, # needs to be a large number for good volume rendering
    # caps=dict(x_show=False, y_show=False, z_show=False)
    # ))

    # fig.show(renderer="browser") 

#%% Try edge detection for mask
    #FIRST of ALL: it takes directly the image
    CTRL_GM = []
    AD_GM = []
    for x in CTRL_images:
        threshold_filters= sitk.LaplacianSharpeningImageFilter()#first selection of the zones
        
        thresh_img = threshold_filters.Execute(x)
    
        threshold_filters= sitk.UnsharpMaskImageFilter() #enhancement of the edges in order to set a more accurate threshold
        thresh_img = threshold_filters.Execute(thresh_img)
        #threshold_filters= sitk.YenThresholdImageFilter() #this is a good threshold too but it's a little blurry
        threshold_filters= sitk.RenyiEntropyThresholdImageFilter() # best threshold i could find
        threshold_filters.SetInsideValue(0)#
        threshold_filters.SetOutsideValue(1)#binomial I/O
        thresh_img = threshold_filters.Execute(thresh_img)
        data = sitk.GetArrayFromImage(thresh_img)
        #Taking GM elements
        filtered_img = np.where(data == 1, sitk.GetArrayViewFromImage(x), data)
        CTRL_GM.append(filtered_img.flatten())
    for x in AD_images:
        threshold_filters= sitk.LaplacianSharpeningImageFilter()#first selection of the zones
        
        thresh_img = threshold_filters.Execute(x)
    
        threshold_filters= sitk.UnsharpMaskImageFilter() #enhancement of the edges in order to set a more accurate threshold
        thresh_img = threshold_filters.Execute(thresh_img)
        #threshold_filters= sitk.YenThresholdImageFilter() #this is a good threshold too but it's a little blurry
        threshold_filters= sitk.RenyiEntropyThresholdImageFilter() # best threshold i could find
        threshold_filters.SetInsideValue(0)#
        threshold_filters.SetOutsideValue(1)#binomial I/O
        thresh_img = threshold_filters.Execute(thresh_img)
        data = sitk.GetArrayFromImage(thresh_img)
        #Taking GM elements
        filtered_img = np.where(data == 1, sitk.GetArrayViewFromImage(x), data)
        AD_GM.append(filtered_img.flatten())
#%% Making labels
    import pandas as pd
    dataset = []
    zeros = np.array([-1]*len(CTRL_images))
    ones = np.asarray([1]*len(AD_images))
    dataset.extend(CTRL_GM)
    dataset.extend(AD_GM)
    dataset = np.array(dataset)
    labels = np.append(zeros, ones, axis = 0).tolist()
    # df = pd.DataFrame()
    # df['Data'] = dataset
    # df['Labels'] = labels
    #datase, labels = dataset[..., np.newaxis], labels[..., np.newaxis]
#%% Now try a SVM-RFE
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_curve, auc
    from numpy import interp
    from sklearn.preprocessing import RobustScaler, StandardScaler
    from sklearn.pipeline import Pipeline
    #%%
    train_set_data, test_set_data, train_set_lab, test_set_lab = train_test_split(dataset, labels, test_size = 0.3,random_state=42)
    classifier = SVC(kernel='linear', probability=True)
    classifier = classifier.fit(train_set_data, train_set_lab)
    coef_vect = classifier.coef_ #vettore dei pesi
    classifier = classifier.fit(train_set_data, train_set_lab, np.abs(coef_vect))
    coef_vect = classifier.coef_ #vettore dei pesi
    #%%Resume of above
    print(classifier)
    print(classifier.score(test_set_data, test_set_lab))
#%% # create pipeline
        #Try RFE
    from sklearn.feature_selection import RFE
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import RepeatedStratifiedKFold

    X, y = dataset, np.array(labels)
    n_splits = 20#secondo articolo(mi sembra) 
    #%%
    rfe = RFE(estimator=SVC(kernel='linear', probability=True), n_features_to_select=1000)
    model = SVC(kernel='linear', probability=True)
    pipeline = Pipeline(steps=[('s',rfe),('m',model)])
    # evaluate model
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=1, random_state=1)
    n_scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    # report performance
    print('Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))


    #%%#PARTE COPIATA DALLA RETICO 
     
    def plot_cv_roc(X, y, classifier, n_splits, scaler=None):
        if scaler:
            model = Pipeline([('scaler', scaler()),
                    ('classifier', classifier)])
        else:
            model = classifier

        try:
            y = y.to_numpy()
            X = X.to_numpy()
        except AttributeError:
            pass
    
        cv = StratifiedKFold(n_splits)
    
        tprs = [] #True positive rate
        aucs = [] #Area under the ROC Curve
        interp_fpr = np.linspace(0, 1, 100)
        plt.figure()
        i = 0
        for train, test in cv.split(X, y):
          probas_ = model.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area under the curve
          fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    #      print(f"{fpr} - {tpr} - {thresholds}\n")
          interp_tpr = interp(interp_fpr, fpr, tpr)
          tprs.append(interp_tpr)
        
          roc_auc = auc(fpr, tpr)
          aucs.append(roc_auc)
          plt.plot(fpr, tpr, lw=1, alpha=0.3,
                  label=f'ROC fold {i} (AUC = {roc_auc:.2f})')
          i += 1
        plt.legend()
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.show()
    
        plt.figure()
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
              label='Chance', alpha=.8)
    
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(interp_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(interp_fpr, mean_tpr, color='b',
                label=f'Mean ROC (AUC = {mean_auc:.2f} $\pm$ {std_auc:.2f})',
                lw=2, alpha=.8)
    
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(interp_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                        label=r'$\pm$ 1 std. dev.')
    
        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])
        plt.xlabel('False Positive Rate',fontsize=18)
        plt.ylabel('True Positive Rate',fontsize=18)
        plt.title('Cross-Validation ROC of SVM',fontsize=18)
        plt.legend(loc="lower right", prop={'size': 15})
        plt.show()
    
    plot_cv_roc(X,y, classifier, n_splits, scaler=None)
    #%%Resume of above
    print(classifier)
    print(classifier.score(StandardScaler().fit_transform(test_set_data), test_set_lab))

    
    
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