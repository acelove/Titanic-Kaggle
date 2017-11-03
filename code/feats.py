#!/usr/bin/env python
# coding=utf-8
from utils import *


def get_feats(data):

  data, rfr = set_missing_ages(data)

  data = set_Cabin_type(data)

  dummies_Cabin = pd.get_dummies(data['Cabin'], prefix= 'Cabin')

  dummies_Embarked = pd.get_dummies(data['Embarked'], prefix= 'Embarked')

  dummies_Sex = pd.get_dummies(data['Sex'], prefix= 'Sex')

  dummies_Pclass = pd.get_dummies(data['Pclass'], prefix= 'Pclass')

  df = pd.concat([data, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
  df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)


  scaler = preprocessing.StandardScaler()
  age_scale_param = scaler.fit(df['Age'].values.reshape(-1,1))
  df['Age_scaled'] = scaler.fit_transform(df['Age'].values.reshape(-1,1), age_scale_param)
  fare_scale_param = scaler.fit(df['Fare'].values.reshape(-1,1))
  df['Fare_scaled'] = scaler.fit_transform(df['Fare'].values.reshape(-1,1), fare_scale_param)
  df.drop(['Age','Fare'],axis=1,inplace=True)

  # 用正则取出我们要的属性值
  X = df.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*').as_matrix()
  return X
