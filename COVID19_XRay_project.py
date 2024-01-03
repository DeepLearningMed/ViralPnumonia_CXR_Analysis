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

    return  model



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
es = EarlyStopping(monitor='val_dsc', mode='max', verbose=1, patience=40)

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
        rotation_range=2,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
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

MT_pretrained_lesion_seg_model =multiTaskModel(base_model=pretrained_lesion_seg_model2)

es = EarlyStopping(monitor='val_seg_output_dsc', mode='max', verbose=1, patience=10)

filepath_inf_seg_2 = './inf_seg_2.h5'

checkpoint = ModelCheckpoint(filepath_inf_seg_2,
                             monitor='val_seg_output_dsc',
                             verbose=2,
                             save_best_only=True,
                             mode='max')

callbacks_list = [checkpoint, es]

batch_size = 1

epochs = 200

steps = int(X_train.shape[0]/ batch_size)

history3 = MT_pretrained_lesion_seg_model.fit(X_train, [Y_train, X_train[...,:1]], epochs=epochs,
                                   steps_per_epoch=steps,
                                   validation_data=(X_validation, [Y_validation, X_validation[...,:1]]),
                                   callbacks=callbacks_list,
                                   verbose=2)

###### Diagnosis model ######################################
def diagnosis(lr=0.0001, base_model = MT_pretrained_lesion_seg_model):

    model = Model(inputs=base_model.inputs, outputs=[base_model.get_layer('conv_7b_ac').output])
    x = tf.keras.layers.GlobalMaxPooling2D()(model.layers[-1].output)

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
es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=12)

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
        rotation_range=3,
        width_shift_range=0.02,
        height_shift_range=0.02,
        zoom_range=0.02,
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

######### Sevrity score prediction model ###################################################
def Severity_predictor(lr=0.0001, base_model = MT_pretrained_lesion_seg_model):

    model = Model(inputs=base_model.inputs, outputs=[base_model.get_layer('conv_7b_ac').output])
    x = tf.keras.layers.GlobalMaxPooling2D()(model.layers[-1].output)

    output = Dense(1, activation='linear')(x)
    # define new model
    model = Model(inputs=model.inputs, outputs=output)

    opt = keras.optimizers.Adam(learning_rate=lr, decay=0.0004, beta_1=0.9, beta_2=0.999, amsgrad=False)

    model.compile(
        optimizer=opt,
        loss=keras.losses.mean_absolute_error,
        metrics=['mean_absolute_error'],
    )
    return  model

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

datagen = ImageDataGenerator(width_shift_range=0.05,
                             height_shift_range=0.05,
                             shear_range=0.02,
                             zoom_range=0.05,
                             fill_mode='nearest')

data_gen = datagen.flow(X_train, Y_train, batch_size=batch_size)

#
epochs = 300
batch_size=4
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

fold_var = 1

tr_indx = np.load('./tr_indx' + str(fold_var) + '.npy')
tst_indx = np.load('./tst_indx' + str(fold_var) + '.npy')

tr_indx = tr_indx[0,:]
tst_indx = tst_indx[0,:]

l1 = int(len(tr_indx) * 0.87)

X_tr, X_val, X_tst = x[tr_indx[:l1]], x[tr_indx[l1:]], x[tst_indx]
Y_tr, Y_val, Y_tst = y[tr_indx[:l1]], y[tr_indx[l1:]], y[tst_indx]

# predictions: 2 step classificartion

X_tst = x[tst_indx]

Y_tst = y[tst_indx]
Y_tst012 = np.where(Y_tst==2,1,Y_tst)

batch_size = 4
y_d012= diagnosis_model1.predict(X_tst, batch_size=batch_size)

thr1 = 0.55

pred012 = np.where(y_d012>thr1, 1, 0)

XL1 = []
YL1 = []
XL0 = []
YL0 = []
for i in range(len(X_tst)):
    if pred012[i]==1:
        XL1.append(X_tst[i, ...])
        YL1.append(Y_tst[i])
    elif pred012[i]==0:
        XL0.append(X_tst[i, ...])
        YL0.append(Y_tst[i])

XL1 = np.array(XL1)
YL1 = np.array(YL1)

XL0 = np.array(XL0)
YL0 = np.array(YL0)

y_d12 = diagnosis_model2.predict(XL1, batch_size=batch_size)

thr2 = 0.4

predL1 = np.where(y_d12>thr2,1,0)

X_gt = np.concatenate((XL0, XL1), axis=0)
Y_gt = np.concatenate((YL0, YL1), axis=0)
y_pred = np.concatenate((np.zeros(len(YL0)), predL1[:,0] + 1), axis=0)

acc = accuracy_score(Y_gt, y_pred)
print('Accuracy:', acc)
f1012 = f1_score(Y_gt, y_pred)
print('f1_score:', f1012)
rec = recall_score(Y_gt, y_pred, average=None)
print('Recall:', rec)
prec = precision_score(Y_gt, y_pred, average=None)
print('Precision:', prec)
confusion_matrix(Y_gt, y_pred)

# predictions: Severity score
batch_size = 4

y_pred = Severity_model.predict(X_tst, batch_size=batch_size)


print(mean_absolute_error(y_pred[:,0] , Y_tst))

tst = (np.absolute(y_pred[:,0] - Y_tst))
print(tst.std())

corr, _ = pearsonr(y_pred[:,0], Y_tst)
print('Pearsons correlation: %.3f' % corr)


