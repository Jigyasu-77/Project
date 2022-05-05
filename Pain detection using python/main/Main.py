#imports for defined utilities and variable configuration file
import utils.data_prc as dp 
import utils.build_network as bn
import utils.compilation_opt as cpo
import utils.config as conf
from keras.models import load_model
%pylab inline


#pickling datasets: Emotion,pain and GSR
Xtrain_pain, Xtest_pain, Xval_pain = dp.dataset_pickle_pain(conf.DATASET_NAME_PAIN)
dp.dataset_pickle_emotions(conf.DATASET_NAME_EMOTION, Xtrain_pain, Xtest_pain, Xval_pain)
dp.dataset_pickle_sr(conf.DATASET_NAME_SR)
dp.dataset_pickle_sr_crossVal(conf.DATASET_NAME_SR)


#loading dataset: Emotion,pain and GSR
X_train_E, y_train_E, X_test_E, y_test_E, X_val_E, y_val_E = dp.dataset_loading(conf.DATASET_NAME_EMOTION)
X_train, y_train, X_test, y_test, X_val, y_val = dp.dataset_loading(conf.DATASET_NAME_PAIN)
X_train_g, y_train_g, X_test_g, y_test_g, X_val_g, y_val_g = dp.dataset_loading(conf.DATASET_NAME_SR+'_noCrossVal')
X_train_gs, y_train_gs, X_test_gs, y_test_gs = dp.load_sr_crossVal(conf.DATASET_NAME_SR)

#printing dataset's shapes
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape, X_val.shape, y_val.shape)
print(X_train_E.shape, y_train_E.shape, X_test_E.shape, y_test_E.shape, X_val_E.shape, y_val_E.shape)
print(X_train_g.shape, y_train_g.shape, X_test_g.shape, y_test_g.shape,  X_val_g.shape, y_val_g.shape)
print(X_train_gs.shape, y_train_gs.shape, X_test_gs.shape, y_test_gs.shape)

#printing a sample on Emotion dataset
num_classes = len(conf.emotion)
samples_per_class = 7

for y, cls in conf.emotion.items():
    idxs = np.flatnonzero(y_train_E == y)
    idxs = np.random.choice(idxs, samples_per_class, replace=True)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(reshape(X_train_E[idx],(48,48)))
        plt.axis('off')
        if i == 0:
            plt.title(cls)
plt.show()

#Labels from values to categorical forms
y_train_E, y_test_E, y_val_E= dp.y_to_categorical(y_train_E, y_test_E, conf.NUM_CLASSES_EMOTION, y_val_E)
y_train, y_test, y_val = dp.y_to_categorical(y_train, y_test, conf.NUM_CLASSES_PAIN, y_val)
y_train_g, y_test_g, y_val_g = dp.y_to_categorical(y_train_g, y_test_g, conf.NUM_CLASSES_SR, y_val_g)
y_train_gs, y_test_gs = dp.y_to_categorical(y_train_gs, y_test_gs, conf.NUM_CLASSES_SR)

#Preparing model
#Creation of the Optimizer
    #Build the Model
    #Compilation of the model with the optimizer
	
opt = cpo.rmsPropOpt()
#model = bn.inception_v3()
model_1 = bn.build_pain_model(X_train)
#model = bn.build_model(X_train_E)
#model_1 = bn.build_gsr_model()
model_1.summary()
model_1 = cpo.compiling(model_1, opt)
#model = cpo.compiling(model, opt)
 
#train the model 
#loading a model if needed Training the model with loaded dataset and ploting history of training graph
#model = load_model('model_emo_InceptionV3_nogcloud_30ep.h5')
#model_1 = load_model('model_pain_nogcloud_adam_120ep.h5')

epochs = 120
batch_size = 32

print(X_train.shape, y_train.shape)

#model_1 = cpo.training_cross_valid(model_1, batch_size, epochs, X_train_gs, y_train_gs)
#print (X_train_E.shape, y_train_E.shape)
#model, history = cpo.training(model, batch_size, epochs, X_train_E, y_train_E, X_val_E, y_val_E, "../../../history_csv/adam_inceptionV3_emo")
model_1, history = cpo.training(model_1, batch_size, epochs, X_train, y_train, X_val, y_val, "../../../history_csv/rms_plateau_240_pain.csv")
cpo.plot(model_1, history, epochs)

#Saving File
model_1.save('model_pain_rmsprop_120ep_3rdtrial_plateau.h5')  # creates a HDF5 file '.h5'
#del model  # deletes the existing model


#Second model created, trained and history ploted with new optimizer
opt1 = cpo.adamOpt() 
model_2 = bn.build_pain_model(X_train)
model_2 = cpo.compiling(model_2, opt1)
model_2, history2 = cpo.training(model_2, batch_size, epochs, X_train, y_train, X_val, y_val, "../../../history_csv/adam_plateau_120_emo.csv")
cpo.plot(model_2, history2, epochs)

scores = model_2.evaluate(X_test, y_test, batch_size = 32,verbose=0)
print(scores)


#Plotting models' history simultaneously
epochs_array = np.arange(epochs)
plt.figure(1)
plt.title('Loss in %d epochs' %(epochs))
plt.plot(epochs_array, np.asarray(history['loss']), 'b', label='RMSProp')
plt.plot(epochs_array, np.asarray(history2['loss']), 'r', label='ADAM')
plt.legend()
    
plt.figure(2)
plt.title('Accuracy in %d epochs' %(epochs))
plt.plot(epochs_array, np.asarray(history['acc']), 'b', label='RMSProp')
plt.plot(epochs_array, np.asarray(history2['acc']), 'r', label='ADAM')
plt.legend()    
    
plt.figure(3)
plt.title('Val_Loss in %d epochs' %(epochs))
plt.plot(epochs_array, np.asarray(history['val_loss']), 'b', label='RMSProp')
plt.plot(epochs_array, np.asarray(history2['val_loss']), 'r', label='ADAM')
plt.legend()

plt.figure(4)
plt.title('Val_Accuracy in %d epochs' %(epochs))
plt.plot(epochs_array, np.asarray(history['val_acc']), 'b', label='RMSProp')
plt.plot(epochs_array, np.asarray(history2['val_acc']), 'r', label='ADAM')
plt.legend()
