import os, os.path
import pickle
import numpy as np
import pandas as pd
import keras
from keras.utils import np_utils
import matplotlib.pyplot as plt
from keras.preprocessing import image as image_utils
from PIL import Image
from scipy.misc import imread, imsave
import config as conf


emotion_dataset = np.zeros((conf.DATASET_SIZE_EMOTION,3))

def dataset_pickle_pain(filename):
    
    pickle_file = conf.DATASET_PATH_PAIN + filename+'.pickle'
    if(os.path.exists(pickle_file)):
        print '%s already exists. Skipping pickling.' % pickle_file
        return None, None, None
    
    else:
        X_train = np.empty([conf.TRAIN_SIZE_PAIN_GEATER_2, conf.PICTURE_DIM_PAIN_H, conf.PICTURE_DIM_PAIN_W,3])
        y_train = np.empty([conf.TRAIN_SIZE_PAIN_GEATER_2,])
        X_val = np.empty([conf.VALIDATION_SIZE_PAIN_GEATER_2, conf.PICTURE_DIM_PAIN_H, conf.PICTURE_DIM_PAIN_W,3])
        y_val = np.empty([conf.VALIDATION_SIZE_PAIN_GEATER_2,])
        X_test = np.empty([conf.TEST_SIZE_PAIN_GEATER_2, conf.PICTURE_DIM_PAIN_H, conf.PICTURE_DIM_PAIN_W,3])
        y_test = np.empty([conf.TEST_SIZE_PAIN_GEATER_2,])
        
        X_train_gray = np.empty([conf.TRAIN_SIZE_PAIN_GEATER_2, conf.PICTURE_DIM_EMOTION, conf.PICTURE_DIM_EMOTION,1])
        X_val_gray = np.empty([conf.VALIDATION_SIZE_PAIN_GEATER_2, conf.PICTURE_DIM_EMOTION, conf.PICTURE_DIM_EMOTION,1])
        X_test_gray = np.empty([conf.TEST_SIZE_PAIN_GEATER_2, conf.PICTURE_DIM_EMOTION, conf.PICTURE_DIM_EMOTION,1])
        

        train_indx = 0
        val_indx = 0 
        test_indx = 0
        for label,folder_name in enumerate(os.listdir(conf.DATASET_PATH_PAIN)):
            folder = os.path.join(conf.DATASET_PATH_PAIN,folder_name)
            print(label,folder_name)
            for lbl, subfolder_name in enumerate(os.listdir(folder)):
                subfolder = os.path.join(folder, subfolder_name)
                if (int(subfolder_name) > 2):
                    for f in os.listdir(subfolder):
                        fileName = os.path.join(subfolder, f)
                        image = image_utils.load_img(fileName, target_size=(conf.PICTURE_DIM_PAIN_H, conf.PICTURE_DIM_PAIN_W))
                        image = image_utils.img_to_array(image).astype(np.float32)
                        image = image/ 255.

                        gray_image = image_utils.load_img(fileName, target_size=(conf.PICTURE_DIM_EMOTION, conf.PICTURE_DIM_EMOTION))
                        gray_image = image_utils.img_to_array(gray_image).astype(np.float32)
                        gray_image = gray_image/ 255.
                        gray_image = np.dot(gray_image[..., :3], [0.299, 0.587, 0.114])                                             
                        gray_image = np.reshape(gray_image, (48, 48, 1))                                         
                        
                        if (folder_name == 'train'):
                            X_train[train_indx] = image
                            y_train[train_indx] = int(subfolder_name)
                            X_train_gray[train_indx] = gray_image
                            train_indx +=1
                        elif(folder_name == 'test'):
                            X_test[test_indx] = image
                            y_test[test_indx] = int(subfolder_name)
                            X_test_gray[test_indx] = gray_image
                            test_indx +=1
                        else:
                            X_val[val_indx] = image
                            y_val[val_indx] = int(subfolder_name)
                            X_val_gray[val_indx] = gray_image
                            val_indx +=1


        rand_train =  np.arange(conf.TRAIN_SIZE_PAIN_GEATER_2)
        np.random.shuffle(rand_train)
        y_train = y_train[rand_train]
        X_train = X_train[rand_train]
        X_train_gray = X_train_gray[rand_train]

        rand_test =  np.arange(conf.TEST_SIZE_PAIN_GEATER_2)
        np.random.shuffle(rand_test)
        y_test = y_test[rand_test]
        X_test = X_test[rand_test]
        X_test_gray = X_test_gray[rand_test]

        rand_val =  np.arange(conf.VALIDATION_SIZE_PAIN_GEATER_2)
        np.random.shuffle(rand_val)
        y_val = y_val[rand_val]
        X_val = X_val[rand_val]
        X_val_gray = X_val_gray[rand_val]
        
        print ('Pickling', pickle_file, '...')

        with open(pickle_file, 'wb') as picklefile:
            save = {
                'dataset_Xtrain': X_train,
                'dataset_ytrain': y_train-3,
                'dataset_Xtest': X_test,
                'dataset_ytest': y_test-3,
                'dataset_Xval': X_val,
                'dataset_yval': y_val-3
            }
            pickle.dump(save, picklefile, pickle.HIGHEST_PROTOCOL)
            print pickle_file, 'pickled successfully!'
            
        return X_train_gray, X_test_gray, X_val_gray

def normalize(x):
    mean = np.mean(x, axis=0)
    sigma = np.std(x, axis= 0)
    X = (x - mean)/sigma
    return X

def dataset_pickle_sr_crossVal(filename):
    
    pickle_file  = conf.DATASET_PATH_PAIN_SR_HR + filename + '.pickle'
    
    if(os.path.exists(pickle_file)):
        print '%s already exists. Skipping pickling.' % pickle_file
    else:
        gsr_dataset = pd.read_csv(conf.DATASET_PATH_PAIN_SR_HR + filename +'.csv')
        X_train =  np.array(gsr_dataset[0: conf.TRAIN_SIZE_SR_HR+conf.VALIDATION_SIZE_SR_HR])
        y_train = X_train [:, 0]
        X_train = np.delete(X_train, 0, 1)
        
        X_test =  np.array(gsr_dataset[conf.TRAIN_SIZE_SR_HR+conf.VALIDATION_SIZE_SR_HR:conf.DATASET_SIZE_SR_HR])
        y_test = X_test [:, 0]
        X_test = np.delete(X_test, 0, 1)
      
        print 'Pickling', pickle_file, '...'

        with open(pickle_file, 'wb') as picklefile:
            save = {
                'dataset_Xtrain': X_train,
                'dataset_ytrain': y_train,
               
                'dataset_Xtest': X_test,
                'dataset_ytest': y_test                
            }
            pickle.dump(save, picklefile, pickle.HIGHEST_PROTOCOL)
            print pickle_file, 'pickled successfully!'
            

def dataset_pickle_sr(filename):
    filename = conf.DATASET_PATH_PAIN_SR_HR + filename + '.csv'

    pickle_file  = os.path.splitext(filename)[0] + '_noCrossVal' + '.pickle'
    if(os.path.exists(pickle_file)):
        print '%s already exists. Skipping pickling.' % pickle_file
    else:
        gsr_dataset = pd.read_csv(filename )
        X_train =  np.array(gsr_dataset[0:conf.TRAIN_SIZE_SR_HR])
        y_train = X_train [:, 0]
        X_train = np.delete(X_train, 0, 1)
        
        X_val =  np.array(gsr_dataset[conf.TRAIN_SIZE_SR_HR:conf.TRAIN_SIZE_SR_HR+conf.VALIDATION_SIZE_SR_HR])
        y_val = X_val [:, 0]
        X_val = np.delete(X_val, 0, 1)
        
        X_test =  np.array(gsr_dataset[conf.TRAIN_SIZE_SR_HR+conf.VALIDATION_SIZE_SR_HR:conf.DATASET_SIZE_SR_HR])
        y_test = X_test [:, 0]
        X_test = np.delete(X_test, 0, 1)
       
        
        print 'Pickling', pickle_file, '...'

        with open(pickle_file, 'wb') as picklefile:
            save = {
                'dataset_Xtrain': X_train,
                'dataset_ytrain': y_train,
                'dataset_Xval': X_val,
                'dataset_yval': y_val,
                'dataset_Xtest': X_test,
                'dataset_ytest': y_test                
            }
            pickle.dump(save, picklefile, pickle.HIGHEST_PROTOCOL)
            print pickle_file, 'pickled successfully!'

def load_sr_crossVal(filename):
    filename = conf.DATASET_PATH_PAIN_SR_HR + filename + '.pickle'

    with open(filename, 'rb') as picklefile:
            save = pickle.load(picklefile)

            X_train = save['dataset_Xtrain']

            y_train = save['dataset_ytrain']

            X_test = save['dataset_Xtest']

            y_test = save['dataset_ytest']


            return X_train, y_train, X_test, y_test

def dataset_loading(filename):
    if (filename == 'All_features_noCrossVal' or filename == 'GSR_ds_noCrossVal'):
        filename = conf.DATASET_PATH_PAIN_SR_HR + filename + '.pickle'
 
    elif (filename == 'fer2013'):
        filename = conf.DATASET_PATH_EMOTION + filename + '.pickle'
    else:
        filename = conf.DATASET_PATH_PAIN + filename + '.pickle'

    with open(filename, 'rb') as picklefile:
            save = pickle.load(picklefile)

            X_train = save['dataset_Xtrain']

            y_train = save['dataset_ytrain']

            X_test = save['dataset_Xtest']

            y_test = save['dataset_ytest']

            X_val = save['dataset_Xval']

            y_val = save['dataset_yval']

            return X_train, y_train, X_test, y_test, X_val, y_val

def remove_disgust(emotion_dataset):
    emotion = emotion_dataset.pop('emotion')
    print "Changing Disgust to Anger"
    
    for i in range(emotion.size):
        if(emotion[i] == 0 or emotion[i] == 1):
            emotion[i] = 0
        else:
            emotion[i] -= 1
    
    emotion_dataset['emotion'] = emotion
    return emotion_dataset
    

def dataset_pickle_emotions(filename, X_train_pain, X_test_pain, X_val_pain, force=False):

    filename = conf.DATASET_PATH_EMOTION + filename + '.csv'

    pickle_file  = os.path.splitext(filename)[0] + '.pickle'

    global emotion_dataset
    if(os.path.exists(pickle_file) and not force):
        print '%s already exists. Skipping pickling.' % pickle_file
    else:
        with open(filename, 'rb') :
            emotion_dataset = pd.read_csv(filename)
            X_train = emotion_dataset.pixels[0:conf.TRAIN_SIZE_EMOTION]
            y_train = emotion_dataset.emotion[0:conf.TRAIN_SIZE_EMOTION]
            
            X_val = emotion_dataset.pixels[conf.TRAIN_SIZE_EMOTION:conf.TRAIN_SIZE_EMOTION+conf.VALIDATION_SIZE_EMOTION]
            y_val = emotion_dataset.emotion[conf.TRAIN_SIZE_EMOTION:conf.TRAIN_SIZE_EMOTION+conf.VALIDATION_SIZE_EMOTION]

            X_test = emotion_dataset.pixels[conf.TRAIN_SIZE_EMOTION+conf.VALIDATION_SIZE_EMOTION:conf.DATASET_SIZE_EMOTION]
            y_test = emotion_dataset.emotion[conf.TRAIN_SIZE_EMOTION+conf.VALIDATION_SIZE_EMOTION:conf.DATASET_SIZE_EMOTION]

        
        X_train = np.array(list(map(lambda arr: np.fromiter(list(map(lambda str: int(str),
                     arr)), dtype= np.int), list(map(lambda str: str.split(),
                      X_train)))))

        y_train = np.fromiter(list(map(int, y_train)), dtype=np.int)
        
        X_val = np.array(list(map(lambda arr: np.fromiter(list(map(lambda str: int(str),
                     arr)), dtype= np.int), list(map(lambda str: str.split(),
                      X_val)))))

        y_val = np.fromiter(list(map(int, y_val)), dtype=np.int)

        X_test = np.array(list(map(lambda arr: np.fromiter(list(map(lambda str: int(str),
                     arr)), dtype= np.int), list(map(lambda str: str.split(),
                      X_test)))))
        y_test = np.fromiter(list(map(int, y_test)), dtype=np.int)
        
        
        X_train, X_test, X_val = prepare_emotions_examples(X_train, X_test, X_val)
        
        X_train = np.concatenate((X_train, X_train_pain), axis=0)
        X_test = np.concatenate((X_test, X_test_pain), axis=0)
        X_val = np.concatenate((X_val, X_val_pain), axis=0)
        y_train = np.concatenate((y_train, np.ones(conf.TRAIN_SIZE_PAIN_GEATER_2)*7), axis=0)
        y_val = np.concatenate((y_val, np.ones(conf.VALIDATION_SIZE_PAIN_GEATER_2)*7), axis=0)
        y_test = np.concatenate((y_test, np.ones(conf.TEST_SIZE_PAIN_GEATER_2)*7), axis=0)
        
        rand_train =  np.arange(conf.FIN_TRAIN_SIZE_EMOTION)
        np.random.shuffle(rand_train)
        y_train = y_train[rand_train]
        X_train = X_train[rand_train]

        rand_test =  np.arange(conf.FIN_TEST_SIZE_EMOTION)
        np.random.shuffle(rand_test)
        y_test = y_test[rand_test]
        X_test = X_test[rand_test]

        rand_val =  np.arange(conf.FIN_VALIDATION_SIZE_EMOTION)
        np.random.shuffle(rand_val)
        y_val = y_val[rand_val]
        X_val = X_val[rand_val]
        
        
        print 'Pickling', pickle_file, '...'

        with open(pickle_file, 'wb') as picklefile:
            save = {
                'dataset_Xtrain': X_train,
                'dataset_ytrain': y_train,
                'dataset_Xtest': X_test,
                'dataset_ytest': y_test,
                'dataset_Xval': X_val,
                'dataset_yval': y_val
            }
            pickle.dump(save, picklefile, pickle.HIGHEST_PROTOCOL)
            print pickle_file, 'pickled successfully!'


def prepare_emotions_examples(x_train, x_test, x_val):
    
    x_train, x_test, x_val =  np.reshape(x_train,(x_train.shape[0], conf.PICTURE_DIM_EMOTION, conf.PICTURE_DIM_EMOTION,1)),np.reshape(x_test,(x_test.shape[0], conf.PICTURE_DIM_EMOTION, conf.PICTURE_DIM_EMOTION,1)),np.reshape(x_val,(x_val.shape[0], conf.PICTURE_DIM_EMOTION, conf.PICTURE_DIM_EMOTION,1))
    
    if (x_train.shape[1]==160):
        x_train = x_train.astype('uint8')
        x_val = x_val.astype('uint8')
        x_test = x_test.astype('uint8')
    
    else:
        x_train = x_train.astype('float32')
        x_train/=255

        x_val = x_val.astype('float32')
        x_val/=255

        x_test = x_test.astype('float32')
        x_test/=255
    
    return x_train, x_test, x_val

def y_to_categorical(y_train, y_test, num_of_classes, y_val=None):
    
    if (y_val is None):
        return keras.utils.to_categorical(y_train, num_of_classes), keras.utils.to_categorical(y_test, num_of_classes)
    else:
        return keras.utils.to_categorical(y_train, num_of_classes), keras.utils.to_categorical(y_test, num_of_classes),keras.utils.to_categorical(y_val, num_of_classes) 