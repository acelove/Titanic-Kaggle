#!/usr/bin/env python
# coding=utf-8
import pandas as pd
import numpy as np
#特征无关的函数方法

def gen_ans_file(predictions,ID,output_file):
  result =pd.DataFrame({'PassengerId':ID,'Survived':predictions.astype(np.int32)})
  result.to_csv(output_file,index=False)


def get_dummies(train,test,prefix):
    data = pd.concat([train,test],axis=0)
    dummies_data = pd.get_dummies(data,prefix=prefix)
    d_train = dummies_data[0:train.shape[0]]
    d_test  = dummies_data[train.shape[0]:]
    return d_train, d_test

def drop(data_list, feat_list):
    for data in data_list:
        data.drop(feat_list,axis=1,inplace=True)
