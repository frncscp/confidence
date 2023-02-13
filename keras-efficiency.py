import os
import cv2
import tensorflow as tf
import numpy as np
from keras.models import load_model

#Model dir = /path/to/model.h5
#Source = /path/to/dataset
#Dataset Class = either True if all images on the dataset are from the trained class, or False if all the images are anomalous
#Score bias = S' on the formula, it adds weight to the score variable
#Average Prediction bias = P' on the formula, it adds weight to the average prediction variable
#Score threshold = from which percentage a prediction counts as a positive class
#Score step = how much unities will added or substracted for each prediction
#Image Normalization = if True, normalizes all values by dividing them by 255

class Metrics():
    
    def __init__(self, model_dir, source, dataset_class = bool, image_height = 224, image_width = 224, score_bias = 1, avg_prediction_bias = 1, score_threshold = .5, score_step = 1, image_normalization = True):       
        MB_CONVERSION = 1024**2 
        self.dataset_class = dataset_class
        self.stats = {}
        self.stats['score'] = 0
        self.stats['avg_prediction'] = 0
        self.PLACEHOLDER_FOR_EXPAND_DIMS = 0
        self.model_dir = model_dir
        self.source = source
        self.image_height = image_height
        self.image_width = image_width
        self.score_bias = score_bias
        self.avg_prediction_bias = avg_prediction_bias
        self.score_threshold = score_threshold
        self.score_step = score_step
        self.image_normalization = 255 if image_normalization == True else 1
        self.source = source #prueba_DIR
        self.listdir = os.listdir(self.source) #pruebas
        self.maximum_score = len(self.listdir)
        self.minimum_score = -len(self.listdir)
        self.model = load_model(model_dir)


    def get_stats(self):
        i = 0
        self.non_read_images = 0
        for image in self.listdir:
            print(f'Doing and saving all predictions: {round((i/len(self.listdir))*100,2)}%', end = '\r')
            i+=1
            try:
                image_dir = os.path.join(self.source, image)
                image = cv2.imread(image_dir)
                resize = tf.image.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB),(self.image_width, self.image_height))
            except:
                self.non_read_images += 1
                continue        
            y_gorrito = float(self.model.predict(np.expand_dims(resize/self.image_normalization, self.PLACEHOLDER_FOR_EXPAND_DIMS), verbose = 0))
            if self.dataset_class:    
                if y_gorrito > self.score_threshold:
                    self.stats['score'] += self.score_step
                else:
                    self.stats['score'] -= self.score_step
            else:
                if y_gorrito > self.score_threshold:
                    self.stats['score'] -= self.score_step
                else:
                    self.stats['score'] += self.score_step
            self.stats['avg_prediction'] += y_gorrito
        self.maximum_score -= self.non_read_images
        self.minimum_score += self.non_read_images
        self.stats['avg_prediction'] /= (len(self.listdir) - self.non_read_images)
        self.stats['normalized_score'] = (self.stats['score']-self.minimum_score)/(self.maximum_score-self.minimum_score)
        if self.dataset_class:
            self.stats['efficiency'] = ((self.score_bias*self.stats['normalized_score'])+(self.avg_prediction_bias*self.stats['avg_prediction']))/(self.score_bias + self.avg_prediction_bias)
        else:
            self.stats['efficiency'] = ((self.score_bias*self.stats['normalized_score'])+(self.avg_prediction_bias*(1-self.stats['avg_prediction'])))/(self.score_bias + self.avg_prediction_bias)
        #self.stats['efficiency per MB'] = self.stats['efficiency']/self.stats['model size']
        return self.stats
  #It returns a dictionary with this order:
  #1. Folder
  #2. Metrics
