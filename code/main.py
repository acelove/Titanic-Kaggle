#!/usr/bin/env python
# coding=utf-8
from utils import *
from feats import *

if __name__ == "__main__":

    data_train = pd.read_csv('../input/train.csv')
    train_X = get_feats(data_train)
    train_y = data_train['Survived']

    data_test = pd.read_csv('../input/test.csv')
    test_X= get_feats(data_test)


    classifier = RandomForestClassifier(n_estimators=4000)
    classifier.fit(train_X,train_y)

    predictions = classifier.predict(test_X)
    gen_ans_file(predictions,data_test,"../output/rf_4000.csv")

