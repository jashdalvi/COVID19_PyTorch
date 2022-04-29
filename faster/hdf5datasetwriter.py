import h5py
import numpy as np

class hdf5DatasetWriter:
    def __init__(self,filename,imagepaths,buffer_size = 1000):

        self.file = h5py.File(filename,"w")
        len_imagepaths = len(imagepaths)
        self.features = self.file.create_dataset("features", (len_imagepaths,100352), dtype='f')
        self.labels = self.file.create_dataset("labels",(len_imagepaths,),dtype = 'i')
        self.buffer_size = buffer_size

        self.buffer = {"features":[],"labels":[]}
        self.idx = 0

    def add(self,features,labels):

        self.buffer["features"].extend(features)
        self.buffer["labels"].extend(labels)
        if len(self.buffer["features"]) >= self.buffer_size:
            self.flush()
            self.buffer = {"features":[],"labels":[]}

    
    def flush(self):
        actual_buffer_size = len(self.buffer["features"])
        self.features[self.idx:self.idx + actual_buffer_size] = np.array(self.buffer["features"],dtype = "float")
        self.labels[self.idx:self.idx + actual_buffer_size] = np.array(self.buffer["labels"],dtype = "int")
        self.idx = self.idx + actual_buffer_size


    def close(self):
        if len(self.buffer["features"]) > 0:
            self.flush()

        self.file.close()

        