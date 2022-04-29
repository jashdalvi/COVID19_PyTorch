import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.applications.resnet50 import ResNet50
import numpy as np
from faster import hdf5DatasetWriter,StoreFeatures
from imutils import paths




model = ResNet50(weights = "imagenet",include_top = False,input_shape = (224, 224, 3))
base_dir = "/home/jash/Desktop/JashWork/Covid19CT/Dataset"
label_mapping = {"NonCOVID":0,"COVID":1}

store_features = StoreFeatures(base_dir,model,label_mapping)

hdf5_obj = hdf5DatasetWriter("/home/jash/Desktop/JashWork/Covid19CT/features.hdf5",store_features.imagepaths,buffer_size = 64)

store_features.storefeatures(hdf5_obj,batch_size = 16)