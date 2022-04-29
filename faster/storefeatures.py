import numpy as np
from imutils import paths
import os
from tensorflow.keras.preprocessing.image import img_to_array,load_img
from tensorflow.keras.applications.resnet50 import preprocess_input

class StoreFeatures:
    def __init__(self,base_dir,model,label_mapping):
        self.base_dir = base_dir
        self.imagepaths = list(paths.list_images(self.base_dir))
        self.labels = [x.split(os.path.sep)[-2] for x in self.imagepaths]
        self.model = model
        self.label_mapping = label_mapping


    def storefeatures(self,hdf5,batch_size = 16):

        for i in range(0,len(self.imagepaths),batch_size):
            if len(self.imagepaths) - i >= batch_size:
                batchpaths = self.imagepaths[i:i+batch_size]
                batchlabels = self.labels[i:i+batch_size]
            else:
                batchpaths = self.imagepaths[i:]
                batchlabels = self.labels[i:]

            if i % 128 == 0:
                print("Processed {} images".format(i))

            batchimages = []
            batch_final_labels = []


            for imagepath,label in zip(batchpaths,batchlabels):
                img = load_img(imagepath,target_size = (224,224))
                img = img_to_array(img)
                img = np.expand_dims(img,axis = 0)
                img = preprocess_input(img)
                
                batchimages.append(img)
                batch_final_labels.append(self.label_mapping[label])


            batchimages = np.vstack(batchimages)
            features = self.model.predict(batchimages).reshape(len(batchimages),-1)
            batch_final_labels = np.array(batch_final_labels,dtype = "int")

            hdf5.add(features,batch_final_labels)

        hdf5.close()


        

