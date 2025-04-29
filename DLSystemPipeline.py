import segmentation_models
import numpy as np
import pandas as pd
import cv2
import os
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Activation
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D, Activation, BatchNormalization,ReLU
import glob
from skimage.transform import resize
import SimpleITK as sitk
from segmentation_models.losses import dice_loss
from segmentation_models.metrics import iou_score
from tensorflow.keras.losses import binary_crossentropy
import time
from segmentation_models import Unet

############################################################
def loadDataDicom(X):
    CXRs = []
    for i in range(len(X)):
        print(i)
        img = sitk.ReadImage(X[i])
        img = sitk.GetArrayFromImage(img).astype(np.float32)
        img = img[0, ...]
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
        img = (img - img.mean()) / (img.std())
        CXRs.append(img)
    return np.array(CXRs)

def rgb(x):
    X=[]
    for i in range(len(x)):
        im = x[i,...]
        im = np.expand_dims(im, axis=-1)
        im = np.concatenate((im,im,im),axis=-1)
        X.append(im)
    return np.array(X)

def pre_processing_ThoraxSeg(imgs_paths):
    cxrs = []
    for n in imgs_paths:
        im = cv2.imread(n)
        im = im[..., 0].astype(np.float32)
        im = cv2.resize(im, (512, 512), interpolation=cv2.INTER_AREA)
        im = (im - im.mean()) / im.std()
        cxrs.append(im)
    cxrs = np.array(cxrs)
    cxrs = rgb(cxrs)
    return cxrs

def pre_processing_Classification(imgs_paths):
    cxrs = []
    for n in imgs_paths:
        im = cv2.imread(n)
        im = im[..., 0].astype(np.float32)
        im = resize(im, (512, 512), preserve_range=True,anti_aliasing=False,order=0)
        #im = cv2.resize(im, (512, 512), interpolation=cv2.INTER_AREA)
        im = (im - im.mean()) / (im.std()*3)
        cxrs.append(im)
    cxrs = np.array(cxrs)
    cxrs = rgb(cxrs)
    return cxrs

######## Lung Segmentation ##########################################
def dsc(y_true, y_pred):
    smooth = 0.0001
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

def custom_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

BACKBONE = 'inceptionresnetv2'
seg_model = Unet(backbone_name=BACKBONE, input_shape=(512, 512, 3), classes=1,
            activation='sigmoid', weights=None, encoder_weights='imagenet',
            encoder_freeze=False, encoder_features='default', decoder_block_type='upsampling',
            decoder_filters=(512, 256, 128, 64, 32), decoder_use_batchnorm=True)

def multiTaskModel_LungSeg(lr=0.0001, base_model = seg_model, gamma=0.05):
    model = Model(inputs=base_model.inputs, outputs=[base_model.get_layer('final_conv').output])
    reconstruct_output = Activation('tanh', name='reconstruct_output')(model.layers[-1].output)

    seg_output = Activation('sigmoid', name='seg_output')(model.layers[-1].output)

    model = Model(inputs=model.inputs, outputs=[seg_output, reconstruct_output])

    opt = keras.optimizers.Adam(learning_rate=lr)

    model.compile(optimizer=opt,
                  loss={'seg_output':custom_loss,
                        'reconstruct_output': keras.losses.MeanSquaredError()},
                  loss_weights={'seg_output': 1 - gamma,
                                'reconstruct_output': gamma},
                  metrics={'seg_output':[dsc, iou_score],
                           'reconstruct_output': ['MeanSquaredError']})

    return  model

######## Classification ###############################################
BACKBONE = 'inceptionresnetv2'

seg_model = Unet(backbone_name=BACKBONE, input_shape=(512, 512, 3), classes=1,
                 activation='sigmoid', weights=None, encoder_weights='imagenet',
                 encoder_freeze=False, encoder_features='default', decoder_block_type='upsampling',
                 decoder_filters=(512, 256, 128, 64, 32), decoder_use_batchnorm=True)

def multiTaskModel(lr=0.0001, base_model=seg_model, gamma=0.1):
    model = Model(inputs=base_model.inputs, outputs=[base_model.get_layer('final_conv').output])
    reconstruct_output = Activation('tanh', name='reconstruct_output')(model.layers[-1].output)

    seg_output = Activation('sigmoid', name='seg_output')(model.layers[-1].output)

    model = Model(inputs=model.inputs, outputs=[seg_output, reconstruct_output])

    opt = keras.optimizers.Adam(learning_rate=lr, decay=0.0004, beta_1=0.9, beta_2=0.999, amsgrad=False)

    model.compile(optimizer=opt,
                  loss={'seg_output': keras.losses.binary_crossentropy,
                        'reconstruct_output': keras.losses.MeanSquaredError()},
                  loss_weights={'seg_output': 1 - gamma,
                                 'reconstruct_output': gamma},
                  metrics={'seg_output': [dsc],
                           'reconstruct_output': ['MeanSquaredError']})

    return model

preSeg_model = multiTaskModel()

filepath_preSeg = 'TL_infection_seg.h5'

preSeg_model.load_weights(filepath_preSeg)

def seg_cls(lr=0.0001, base_model=preSeg_model):
        model = Model(inputs=base_model.inputs, outputs=[base_model.get_layer('conv_7b_ac').output])
        x = tf.keras.layers.GlobalAveragePooling2D()(model.output)

        x = Dense(256)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dropout(0.2)(x)

        output = Dense(1, activation='sigmoid')(x)
        # define new model
        model = Model(inputs=model.inputs, outputs=output)

        opt = keras.optimizers.Adam(learning_rate=lr, decay=0.0004, beta_1=0.9, beta_2=0.999, amsgrad=False)

        model.compile(
            optimizer=opt,
            loss=keras.losses.binary_crossentropy,
            metrics=['accuracy'],
        )

        return model

################################################################
LungSeg_model = multiTaskModel_LungSeg()

filepath_lungSeg = 'MT_ThoraxSeg.h5'

LungSeg_model.load_weights(filepath_lungSeg)

#####
cls1 = seg_cls()
cls1_path = 'Classification_step1.h5'
cls1.load_weights(cls1_path)

##################################
imgs_ext = np.array(glob.glob(r"your_data_path\*.png")) # type of image .png, .jpeg, ...

cxrs = pre_processing_ThoraxSeg(imgs_ext)

cxrs_cls = pre_processing_Classification(imgs_ext)

#######################################
start_time = time.time()

lungs_prob, _ = LungSeg_model.predict(cxrs, batch_size= 1)

lungs_pred = np.where(lungs_prob > 0.5, 1, 0).astype(np.float32)

cxrs_cls = cxrs_cls * lungs_pred

########## 2 step classification ##################################################
# Step 1: Classify Normal vs Pneumonia
# for multi-source external validation thr1 = 0.75 and for single-source external valiation thr = 0.3

step1_pred = cls1.predict(cxrs_cls, batch_size=4)

thr1 = 0.75
step1_pred_labels = np.where(step1_pred > thr1, 1, 0)

# Indices of pneumonia predictions
pneumonia_idx = np.where(step1_pred_labels == 1)[0]

# Step 2: Classify pneumonia into COVID-19 vs other
# for multi-source external validation thr1 = 0.75 and for single-source external valiation thr = 0.2

cls2 = seg_cls()
cls2_path = 'Classification_step2.h5'
cls2.load_weights(cls2_path)

x_pneumonia = cxrs_cls[pneumonia_idx]
step2_pred = cls2.predict(x_pneumonia, batch_size=4)

thr2 = 0.3
step2_pred_labels = np.where(step2_pred > thr2, 1, 0)

# Construct final predictions
final_pred = np.zeros_like(step1_pred_labels[:,0])

# Insert predictions from step 2
final_pred[pneumonia_idx] = step2_pred_labels[:,0] + 1

## Evaluation metrics
# print(confusion_matrix(y_true, final_pred))
# acc = accuracy_score(y_true, final_pred)
# print('Accuracy:', acc)
# rec = recall_score(y_true, final_pred)
# print('Recall:', rec)
# prec = precision_score(y_true, final_pred)
# print('Precision:', prec)

##################### Sverity Scoring #################################
def seg_sev(lr=0.0001, base_model=preSeg_model):
    model = Model(inputs=base_model.inputs, outputs=[base_model.get_layer('conv_7b_ac').output])
    x = tf.keras.layers.GlobalAveragePooling2D()(model.output)

    output = Dense(1, activation='linear')(x)
    # define new model
    model = Model(inputs=model.inputs, outputs=output)

    opt = keras.optimizers.Adam(learning_rate=lr, decay=0.0004, beta_1=0.9, beta_2=0.999, amsgrad=False)

    model.compile(
        optimizer=opt,
        loss=keras.losses.mean_absolute_error,
        metrics=['mean_absolute_error'],
    )
    return model

Sev_test = seg_sev()

filepath = 'SeverityScoring.h5' # severity scoring weghts

Sev_test.load_weights(filepath)

covid_idx = np.where(step2_pred_labels == 1)[0]

x_covid = x_pneumonia[pneumonia_idx]

y_pred = Sev_test.predict(x_covid, batch_size=4)


# import sklearn

# print(sklearn.metrics.mean_absolute_error(y_pred[:,0] , y_truth))

# tst = (np.absolute(y_pred[:,0] - y_truth)/(y_truth+1))
# print(tst.mean(), tst.std())



