import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

plt.style.use('ggplot')
plt.rcParams["figure.figsize"] = (16,9)
#%matplotlib inline

score = np.array([0.8, 0.6, 0.4, 0.2])
y = np.array([1,0,1,0])

# false positive rate
FPR = []
# true positive rate
TPR = []
# Iterate thresholds from 0.0 to 1.0
thresholds = np.arange(0.0, 1.01, 0.2)
# array([0. , 0.2, 0.4, 0.6, 0.8, 1. ])

# get number of positive and negative examples in the dataset
P = sum(y)
N = len(y) - P

# iterate through all thresholds and determine fraction of true positives
# and false positives found at this threshold
for thresh in thresholds:
    FP=0
    TP=0
    thresh = round(thresh,2) #Limiting floats to two decimal points, or threshold 0.6 will be 0.6000000000000001 which gives FP=0
    for i in range(len(score)):
        if (score[i] >= thresh):
            if y[i] == 1:
                TP = TP + 1
            if y[i] == 0:
                FP = FP + 1
    FPR.append(FP/N)
    TPR.append(TP/P)
    
# FPR [1.0, 1.0, 0.5, 0.5, 0.0, 0.0]
# TPR [1.0, 1.0, 1.0, 0.5, 0.5, 0.0]
    

# This is the AUC
auc = -1 * np.trapz(TPR, FPR)

plt.plot(FPR, TPR, linestyle='--', marker='o', color='darkorange', lw = 2, label='ROC curve', clip_on=False)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve, AUC = %.2f'%auc)
plt.legend(loc="lower right")
plt.savefig('AUC_example.png')
plt.show()

#Sklearn approach
scores = np.array([0.8, 0.6, 0.4, 0.2])
y = np.array([1,0,1,0])

#thresholds : array, shape = [n_thresholds] Decreasing thresholds on the decision function used to compute fpr and tpr. 
#thresholds[0] represents no instances being predicted and is arbitrarily set to max(y_score) + 1
fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=1)
#thresholds: array([1.8, 0.8, 0.6, 0.4, 0.2])
#tpr: array([0. , 0.5, 0.5, 1. , 1. ])
#fpr: array([0. , 0. , 0.5, 0.5, 1. ])
metrics.auc(fpr, tpr)
#0.75