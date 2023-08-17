# -*- coding: utf-8 -*-

"""
Created on Fri Mar  3 10:52:38 2023

@author: Prasun Tripathi
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, matthews_corrcoef
from kale.pipeline.mpca_trainer import MPCATrainer

#late fusion implementation of short-axis and four-chamber

# Load Preprocessed Four Chamber data
x_train1=np.load('fc128train.npy', allow_pickle=True)
x_test1=np.load('fc128test.npy', allow_pickle=True)

# Load Preprocessed Short-axis data
x_train2=np.load('sx128train.npy', allow_pickle=True)
x_test2=np.load('sx128test.npy', allow_pickle=True)


# Load train and test labels 

train_labels=np.load('train_labels.npy', allow_pickle=True)
test_labels=np.load('test_labels.npy', allow_pickle=True)

#train the classifier on both the modalities

trainer1 = MPCATrainer(classifier="linear_svc",classifier_param_grid={"C": [0.00001,0.0001, 0.001, 0.01]}, n_features=210)

trainer1.fit(x_train1, train_labels)

y_pred1 = trainer1.predict(x_test1)
y1_score = trainer1.decision_function(x_test1)


trainer2 = MPCATrainer(classifier="linear_svc",classifier_param_grid={"C": [0.00001,0.0001, 0.001, 0.01]}, n_features=210)

trainer2.fit(x_train2, train_labels)

y_pred2 = trainer2.predict(x_test2)
y2_score = trainer2.decision_function(x_test2)



#normalize decision scores for both the modalities

x_norm1 = (y1_score-np.min(y1_score))/(np.max(y1_score)-np.min(y1_score))

x_norm2 = (y2_score-np.min(y2_score))/(np.max(y2_score)-np.min(y2_score))

#late fusion of the decision scores

final_score=0.5*x_norm1+0.5*x_norm2

#thresholding the final score

y_pred=np.zeros(len(y_pred1))

for i in range(len(final_score)):
    if final_score[i]>=0.5:
        y_pred[i]=1
    else:
        y_pred[i]=0

#plotting the ROC curve
 
fpr, tpr, thresholds = roc_curve(test_labels, final_score)
lw=2
plt.figure(figsize=(5,5),dpi=100)
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_score(test_labels, final_score))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([-0.05, 1.05])
plt.show()

print("Accuracy- {:.4f}" .format(accuracy_score(test_labels, y_pred)))
print("AUC- {:.4f}" .format(roc_auc_score(test_labels, final_score)))
print("MCC Score-{:.4f}" .format(matthews_corrcoef(test_labels, y_pred)))