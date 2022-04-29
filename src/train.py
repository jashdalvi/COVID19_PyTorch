import pickle 
from sklearn.model_selection import GridSearchCV,StratifiedKFold,train_test_split,cross_val_score
from sklearn.linear_model import LogisticRegression
import h5py
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix,roc_curve,classification_report,accuracy_score
import matplotlib.pyplot as plt
from sklearn.svm import SVC


db = h5py.File("/home/jash/Desktop/JashWork/Covid19CT/features.hdf5","r")
X = np.array(db["features"])
y = np.array(db["labels"])
skf = StratifiedKFold()

#Cross validating over different values of C - [0.001,0.01,0.1,1]

# c_values = [0.001,0.01,0.1,1]
# accuracy_values = {}
# for c in c_values:
#     accuracy = cross_val_score(LogisticRegression(C = c,max_iter=500),X,y,scoring = "accuracy",cv = skf,n_jobs = -1)
#     accuracy_values[c] = np.array(accuracy).mean()
#     print("Training done for Logistic Regression model with {} C parameter".format(c))

# print(accuracy_values)

#Training model on svc with default parameters

accuracy = cross_val_score(SVC(kernel = "linear"),X,y,scoring = "accuracy",cv = skf,n_jobs = -1)
print(accuracy)
print(np.array(accuracy).mean())