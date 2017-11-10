#!/usr/bin/env python
# coding=utf-8
from utils import *
from feats import *
from model import *

if __name__ == "__main__":

    data_train = pd.read_csv('../input/train.csv')
    data_test = pd.read_csv('../input/test.csv')
    train_y = data_train['Survived']
    ID = data_test['PassengerId'].values
    train_X ,test_X = get_feats(data_train,data_test)
    print train_X

    #data_test = pd.read_csv('../input/test.csv')
    #test_X= get_feats(data_test)


    classifier = RF_model()
    #classifier = GBC_model()
    #classifier = XGB_model()
    classifier.fit(train_X,train_y)

    predictions = classifier.predict(test_X)
    gen_ans_file(predictions,ID,"../output/rf_1200.csv")

