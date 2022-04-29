import pickle 
from sklearn.model_selection import GridSearchCV,StratifiedKFold,train_test_split,cross_val_score
from sklearn.linear_model import LogisticRegression
import h5py
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix,roc_curve,classification_report,accuracy_score
import matplotlib.pyplot as plt


db = h5py.File("/home/jash/Desktop/JashWork/Covid19CT/features.hdf5","r")
X = np.array(db["features"])
y = np.array(db["labels"])

clf = LogisticRegression(C = 0.001)

clf.fit(X,y)

with open("/home/jash/Desktop/JashWork/Covid19CT/models/head_model.pkl","wb") as f:
    pickle.dump(clf,f)