# -*- coding: utf-8 -*-

"""
Created on Fri Mar  3 10:52:38 2023

@author: Prasun Tripathi
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, matthews_corrcoef
from kale.pipeline.mpca_trainer import MPCATrainer



# Load Preprocessed Four Chamber data
x_train1=np.load('fc128train.npy', allow_pickle=True)
x_test1=np.load('fc128test.npy', allow_pickle=True)

# Load Preprocessed Short-axis data
x_train2=np.load('sx128train.npy', allow_pickle=True)
x_test2=np.load('sx128test.npy', allow_pickle=True)


# Perform early fusion using FC and SX
x_train=np.concatenate([x_train1,x_train2],axis=1)
x_test=np.concatenate([x_test1,x_test2],axis=1)



# Load train and test labels 

train_labels=np.load('train_labels.npy', allow_pickle=True)
test_labels=np.load('test_labels.npy', allow_pickle=True)


x_train = np.concatenate([x_train[i].reshape((1,) + x_train[i].shape) for i in range(len(x_train))], axis=0)
x_test = np.concatenate([x_test[i].reshape((1,) + x_test[i].shape) for i in range(len(x_test))], axis=0)


#x_train2 = np.concatenate([x_train2[i].reshape((1,) + x_train2[i].shape) for i in range(len(x_train2))], axis=0)
#x_test2 = np.concatenate([x_test2[i].reshape((1,) + x_test2[i].shape) for i in range(len(x_test2))], axis=0)




trainer = MPCATrainer(classifier="linear_svc",classifier_param_grid={"C": [0.00001,0.0001, 0.001, 0.01]}, n_features=210)


#cv_results = cross_validate(trainer, x_train, train_labels, cv=10, scoring=["accuracy", "roc_auc",], n_jobs=1)

trainer.fit(x_train, train_labels)

y_pred = trainer.predict(x_test)
y_score = trainer.decision_function(x_test1)

#f1=f1_score(test_labels, y_pred,average = 'weighted')

#mcc1= matthews_corrcoef(test_labels, y_pred)
#cm1 = metrics.confusion_matrix(test_labels, y_pred)

#Sensitivity = cm1[1,1]/(cm1[1,0]+cm1[1,1])

#Accuracy = (cm1[0,0]+cm1[1,1])/total1
#Specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])
#Sensitivity = cm1[1,1]/(cm1[1,0]+cm1[1,1])
 
fpr, tpr, thresholds = roc_curve(test_labels, y_score)
lw=2
plt.figure(figsize=(5,5),dpi=100)
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_score(test_labels, y_score))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([-0.05, 1.05])
plt.show()

print("Accuracy- {:.4f}" .format(accuracy_score(test_labels, y_pred)))
print("AUC- {:.4f}" .format(roc_auc_score(test_labels, y_score)))
print("MCC Score-{:.4f}" .format(matthews_corrcoef(test_labels, y_pred)))



