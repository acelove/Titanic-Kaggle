#!/usr/bin/env python
# coding=utf-8
from utils import *
from feats import *
from model import *
from sklearn.utils import shuffle
import sys
def evaluate(data):
    data = shuffle(data)
    data_8 = data[0:int(data.shape[0]*0.8)]
    data_2 = data[int(data.shape[0]*0.8):]

    train_X, evaluate_X = get_feats(data_8,data_2)
  

    train_y = data_8['Survived']
    evaluate_y = data_2['Survived']

    classifier = RF_model()
    classifier.fit(train_X,train_y)

    return classifier.score(evaluate_X,evaluate_y)

if __name__ == "__main__":
    times = 10
    if len(sys.argv)>1:
        times = int(sys.argv[1])
    lst = []
    data = pd.read_csv('../input/train.csv')
    for i in xrange(times):
        score = evaluate(data)
        print "Evaluate%d's score is %.6f." % (i+1,score)
        lst.append(score)
    print "After %d times evaluation, mean score is %.6f." % (times,np.mean(np.array(lst)))
