import os
import tensorflow as tf
from tensorflow.keras.models import load_model

#Model dir = /path/to/model.h5, always inside a list
#Source = /path/to/dataset
#Dataset Class = either True if all images on the dataset are from the trained class, or False if all the images are anomalous
#Score bias = S' on the formula, it adds weight to the score variable
#Average Prediction bias = P' on the formula, it adds weight to the average prediction variable
#Score threshold = from which percentage a prediction counts as a positive class
#Score step = how much unities will added or substracted for each prediction
#Image Normalization = if True, normalizes all pixel values on the image by dividing them by 255

class Metrics():
    
    def __init__(self, model_dir, source, dataset_class = bool, image_height = 224, image_width = 224, score_bias = 1, avg_prediction_bias = 1, score_threshold = .5, score_step = 1, image_normalization = True, ensemble = False):       
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

    @tf.function #this decorator allows the predictions to be much faster
    def predict(self, model_list, img, weights = None): 
        #the model and weight's lists have to be in the same order. i.e: [model_1, model_2], [weight_1, weight_2] 
        y_gorrito = 0
        if weights is None:
            weights = [1] * len(model_list)
        for model, weight in zip(model_list, weights):
            y_gorrito += tf.cast(model(tf.expand_dims(img/255., 0)), dtype=tf.float32)*weight
        return y_gorrito / sum(weights)
    
    @tf.function
    def preprocess(self, image, size):
        height, width = size
        image_dir = os.path.join(self.source, image)
        img_file = tf.io.read_file(image_dir)
        img_decode = tf.image.decode_image(img_file, channels=3)
        resize = tf.image.resize(img_decode,(height, width))
        return resize

    def get_stats(self, weights = []):
        i = 0
        self.non_read_images = 0    
        ensemble = [load_model(model) for model in self.model_dir] #loads all models ready to predict onto a list
        
        for image in self.listdir: #iterate through all the folder
            print(f'Doing and saving all predictions: {round((i/len(self.listdir))*100,2)}%', end = '\r')
            i+=1
            try: #preprocess the image
                resize = preprocess(image, (self.image_height, self.image_width))
            except: #you can call later object.non_read_images to know how much data was unused, but the average will take this into account so no worries
                self.non_read_images += 1
                continue            
            y_gorrito = float(self.predict(ensemble, resize, weights))
            if self.dataset_class:  #if the data contained in the folder is of the positive class
                if y_gorrito >= self.score_threshold:
                    self.stats['score'] += self.score_step
                else:
                    self.stats['score'] -= self.score_step
            else: #if the data is of the negative class
                if y_gorrito >= self.score_threshold:
                    self.stats['score'] -= self.score_step
                else:
                    self.stats['score'] += self.score_step
            self.stats['avg_prediction'] += y_gorrito
        
        #here is where non_read_images is used, so the formula is not altered
        self.maximum_score -= self.non_read_images
        self.minimum_score += self.non_read_images
        self.stats['avg_prediction'] /= (len(self.listdir) - self.non_read_images)
        self.stats['normalized_score'] = (self.stats['score']-self.minimum_score)/(self.maximum_score-self.minimum_score)
        
        #there are two formulas: one for the positive class, and one for the negative one
        if self.dataset_class:
            self.stats['efficiency'] = ((self.score_bias*self.stats['normalized_score'])+(self.avg_prediction_bias*self.stats['avg_prediction']))/(self.score_bias + self.avg_prediction_bias)
        else:
            self.stats['efficiency'] = ((self.score_bias*self.stats['normalized_score'])+(self.avg_prediction_bias*(1-self.stats['avg_prediction'])))/(self.score_bias + self.avg_prediction_bias)
        return self.stats
    
  #self.stats returns a dictionary nested with this order:
  #1. Model
  #2. Folder
  #3. Metrics

#Some examples on how to use the class:

#model = Metrics(['D:\ptctrn\models\ptctrn_v1.4.h5', 'D:\ptctrn\models\ptctrn_v1.5.h5', 'D:\ptctrn\models\ptctrn_v1.6.h5', 'D:\ptctrn\models\ptctrn_v1.12.h5', 'D:\ptctrn\models\ptctrn_v1.12.1.h5'], r"D:\ptctrn\new_dataset\Cuasi-Patacon", dataset_class= False, score_threshold= .8)
#model = Metrics(["D:\ptctrn\models\ptctrn_v1.12.2.h5"], r"C:\Users\franc\Downloads\fotogramas", dataset_class= True, score_threshold= .65)
#result_dict = (model.get_stats())
#print(model.non_read_images)
#print(model.model_dir)
