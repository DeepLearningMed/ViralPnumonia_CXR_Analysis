import numpy as np
import SimpleITK as sitk
import tensorflow as tf
from skimage.transform import resize
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation
from tensorflow import keras
from scipy.stats import pearsonr
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score,confusion_matrix, mean_absolute_error
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import segmentation_models
from segmentation_models import Unet
from segmentation_models.utils import set_trainable
 
### utility functions ########################################
def dsc(y_true, y_pred):
    smooth = 0.001
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score
 
def loadData(X):
    CXRs = []
    for i in range(len(X)):
        print(i)
        img = sitk.ReadImage(X[i])
        img = sitk.GetArrayFromImage(img)
        im = img[0, ...]
        im = resize(im, (512, 512), preserve_range=True,anti_aliasing=False,order=0)
        im = (im - im.mean())/(im.std()*3)
        CXRs.append(im)
    return np.array(CXRs)
 
def pre3d(X):
    x = []
    for i in range(len(X)):
        im = X[i,...]
        im = np.expand_dims(im, axis=-1)
        im = np.concatenate((im, im, im), axis=-1, dtype='float32')
        x.append(im)
    return np.array(x)
 
### thorax region segmentation ###############################
BACKBONE = 'inceptionresnetv2'
thorax_seg_model = Unet(backbone_name=BACKBONE, input_shape=(512, 512, 3), classes=1,
            activation='sigmoid', weights=None, encoder_weights='imagenet',
            encoder_freeze=False, encoder_features='default', decoder_block_type='upsampling',
            decoder_filters=(512, 256, 128, 64, 32), decoder_use_batchnorm=True)
 
def multiTaskModel(lr=0.0001, base_model = thorax_seg_model, gamma=0.1):
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
                  metrics={'seg_output':[dsc],
                           'reconstruct_output': ['MeanSquaredError']})
 
    return  model
 
 
 
MT_model = multiTaskModel()
 
es = EarlyStopping(monitor='val_seg_output_dsc', mode='max', verbose=1, patience=10)
 
filepath ='./thorax_segmentation_weights.h5'
 
checkpoint = ModelCheckpoint(filepath,
                             monitor='val_seg_output_dsc',
                             verbose=2,
                             save_best_only=True,
                             mode='max')
 
callbacks_list = [checkpoint, es]
 
batch_size = 1
 
epochs = 200
 
steps = int(X_train.shape[0] / batch_size)
 
history1 = MT_model.fit(X_train, [Y_train, X_train[...,:1]], epochs=epochs, # one forward/backward pass of training data
                    steps_per_epoch=steps, # number of images comprising of one epoch
                    validation_data=(X_validation, [Y_validation, X_validation[...,:1]]), # data for validation
                    callbacks=callbacks_list,
                    verbose=2
                    )
 
##### Transfer learning: lesion segmentation ####################################################
#Step1 : Public dataset
pretrained_lesion_seg_model1 = segmentation_models.Unet(backbone_name='inceptionresnetv2', input_shape=(512, 512, 3), classes=1,
                            activation='sigmoid', weights=None, encoder_weights='imagenet',
                            encoder_freeze=False, encoder_features='default',
                            decoder_block_type='upsampling',
                            decoder_filters=(512, 256, 128, 64, 32),
                            decoder_use_batchnorm=True)
 
opt = tf.keras.optimizers.Adam(learning_rate=0.001, decay= 1e-6, beta_1=0.9, beta_2=0.999, amsgrad=False)
 
pretrained_lesion_seg_model1.compile(optimizer= opt , loss= 'binary_crossentropy', metrics=['accuracy', dsc])
 
#
es = EarlyStopping(monitor='val_dsc', mode='max', verbose=1, patience=20)
 
filepath_inf_Seg_1 = './inf_Seg_1.h5'
 
checkpoint = ModelCheckpoint(filepath_inf_Seg_1,
                             monitor='val_dsc',
                             verbose=2,
                             save_best_only=True,
                             mode='max')
 
callbacks_list = [checkpoint, es]
 
#
batch_size = 1
 
datagen = ImageDataGenerator(
        rotation_range=5,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.02,
        zoom_range=0.05,
        fill_mode='nearest')
 
data_gen = datagen.flow(X_train, Y_train, batch_size=batch_size)
 
#
epochs = 200
 
steps = int(X_train.shape[0] / batch_size)
 
history2 = pretrained_lesion_seg_model1.fit_generator(data_gen, epochs=epochs,
                                          steps_per_epoch=steps,
                                          validation_data=(X_validation, Y_validation),
                                          callbacks=callbacks_list,
                                          verbose=2)
########################################
#Step2 : Inhouse lesion segmentation data
 
pretrained_lesion_seg_model2 = segmentation_models.Unet(backbone_name='inceptionresnetv2', input_shape=(512, 512, 3), classes=1,
                            activation='sigmoid', weights=None, encoder_weights='imagenet',
                            encoder_freeze=False, encoder_features='default',
                            decoder_block_type='upsampling',
                            decoder_filters=(512, 256, 128, 64, 32),
                            decoder_use_batchnorm=True)
 
pretrained_lesion_seg_model2.load_weights(filepath_inf_Seg_1)
 
# the multitask network
MT_pretrained_lesion_seg_model =multiTaskModel(lr=0.00001, base_model=pretrained_lesion_seg_model2)
 
 
es = EarlyStopping(monitor='val_seg_output_dsc', mode='max', verbose=1, patience=5)
 
filepath_inf_seg_2 = './inf_seg_2.h5'
 
checkpoint = ModelCheckpoint(filepath_inf_seg_2,
                             monitor='val_seg_output_dsc',
                             verbose=2,
                             save_best_only=True,
                             mode='max')
 
callbacks_list = [checkpoint, es]
 
batch_size = 1
 
epochs = 15
 
steps = int(X_train.shape[0]/ batch_size)
 
history3 = MT_pretrained_lesion_seg_model.fit(X_train, [Y_train, X_train[...,:1]], epochs=epochs,
                                   steps_per_epoch=steps,
                                   validation_data=(X_validation, [Y_validation, X_validation[...,:1]]),
                                   callbacks=callbacks_list,
                                   verbose=2)
 
###### Diagnosis module ######################################
def diagnosis(lr=0.0001, base_model = MT_pretrained_lesion_seg_model):
 
    model = Model(inputs=base_model.inputs, outputs=[base_model.get_layer('conv_7b_ac').output])
    x = tf.keras.layers.GlobalAveragePooling2D()(model.layers[-1].output)
    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
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
 
diagnosis_model = diagnosis()
 
#
es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=20)
 
filepath_diagnosis = 'diagnosis.h5'
 
checkpoint = ModelCheckpoint(filepath_diagnosis,
                             monitor='val_accuracy',
                             verbose=2,
                             save_best_only=True,
                             mode='max')
 
callbacks_list = [checkpoint, es]
 
#
batch_size = 4
 
datagen = ImageDataGenerator(
       rotation_range=5,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.02,
        zoom_range=0.05,
        fill_mode='nearest')
 
data_gen = datagen.flow(X_train, Y_train, batch_size=batch_size)
 
#
epochs = 200
 
steps = int(X_train.shape[0] / batch_size)
 
history4 = diagnosis_model.fit_generator(data_gen, epochs=epochs,
                    steps_per_epoch=steps,
                    validation_data=(X_validation, Y_validation),
                    callbacks=callbacks_list,
                    verbose=2)
 
######### Sevrity score prediction module ###################################################
def Severity_predictor(lr=0.0001, base_model = MT_pretrained_lesion_seg_model):
 
    model = Model(inputs=base_model.inputs, outputs=[base_model.get_layer('conv_7b_ac').output])
    x = tf.keras.layers.GlobalAveragePooling2D()(model.layers[-1].output)
 
    output = Dense(1, activation='relu')(x)
    # define new model
    model = Model(inputs=model.inputs, outputs=output)
 
    opt = keras.optimizers.Adam(learning_rate=lr, decay=0.0004, beta_1=0.9, beta_2=0.999, amsgrad=False)
 
    model.compile(
        optimizer=opt,
        loss=keras.losses.mean_absolute_error,
        metrics=['mean_absolute_error'],
    )
    return  model
 
Severity_model = Severity_predictor()
 
#
es = EarlyStopping(monitor='val_mean_absolute_error', mode='min', verbose=1, patience=30)
 
filepath_Severity = 'Severity.h5'
 
checkpoint = ModelCheckpoint(filepath_Severity,
                             monitor='val_mean_absolute_error',
                             verbose=2,
                             save_best_only=True,
                             mode='min')
 
callbacks_list = [checkpoint, es]
 
 
batch_size = 4
 
datagen = ImageDataGenerator( rotation_range=5,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.02,
        zoom_range=0.05,
        fill_mode='nearest')
 
data_gen = datagen.flow(X_train, Y_train, batch_size=batch_size)
 
#
epochs = 200
 
steps = int(X_train.shape[0] / batch_size)
 
history5 = Severity_model.fit_generator(data_gen, epochs=epochs,
                    steps_per_epoch=steps,
                    validation_data=(X_validation, Y_validation),
                    callbacks=callbacks_list,
                    verbose=2
                    )
 
###############################################
# data split
from sklearn.model_selection import StratifiedKFold
 
fold_var = 1
 
# Create StratifiedKFold object.
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
 
# x: cxr images - y: labels
 
for train_index, test_index in skf.split(x, y):
 
    tr_indx_cls = np.array([train_index])
    tst_indx_cls = np.array([test_index])
 
    np.save('./tr_indx' + str(fold_var) + '.npy', tr_indx_cls)
    np.save('./tst_indx' + str(fold_var) + '.npy', tst_indx_cls)
 
    fold_var = fold_var + 1
