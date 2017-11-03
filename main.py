#!/usr/bin/env python
# coding=utf-8
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import sklearn.preprocessing as preprocessing
from sklearn.ensemble import RandomForestClassifier
import numpy as np
### 使用 RandomForestClassifier 填补缺失的年龄属性
def set_missing_ages(df,pre_rfr=None):
    print pre_rfr
    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    df.loc[ (df.Fare.isnull()),'Fare'] = 0
    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]

    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()

    # y即目标年龄
    y = known_age[:, 0]

    # X即特征属性值
    X = known_age[:, 1:]

    predictedAges = None
    if pre_rfr is not None:
      predictedAges = pre_rfr.predict(unknown_age[:, 1::])
      rfr = pre_rfr
    else:
      # fit到RandomForestRegressor之中
      rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
      rfr.fit(X, y)

      # 用得到的模型进行未知年龄结果预测
      predictedAges = rfr.predict(unknown_age[:, 1::])

    # 用得到的预测结果填补原缺失数据
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges 

    return df, rfr

def set_Cabin_type(df):
    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = "Yes"
    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "No"
    return df


def pre(data,isTrain=True,rfr=None):
  rfr = rfr
  if not isTrain:
    data, rfr = set_missing_ages(data,rfr)
  else:
    data, rfr = set_missing_ages(data)
  data = set_Cabin_type(data)

  dummies_Cabin = pd.get_dummies(data['Cabin'], prefix= 'Cabin')

  dummies_Embarked = pd.get_dummies(data['Embarked'], prefix= 'Embarked')

  dummies_Sex = pd.get_dummies(data['Sex'], prefix= 'Sex')

  dummies_Pclass = pd.get_dummies(data['Pclass'], prefix= 'Pclass')

  df = pd.concat([data, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
  df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)



  scaler = preprocessing.StandardScaler()
  age_scale_param = scaler.fit(df['Age'])
  df['Age_scaled'] = scaler.fit_transform(df['Age'], age_scale_param)
  fare_scale_param = scaler.fit(df['Fare'])
  df['Fare_scaled'] = scaler.fit_transform(df['Fare'], fare_scale_param)
  df.drop(['Age','Fare'],axis=1,inplace=True)

  # 用正则取出我们要的属性值
  train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
  train_np = train_df.as_matrix()
  return train_np,rfr

data_train = pd.read_csv('data/train.csv')
data_test = pd.read_csv('data/test.csv')
train_np,rfr = pre(data_train)
test_X ,rfr= pre(data_test,False,rfr)
# y即Survival结果
y = train_np[:, 0]

# X即特征属性值
X = train_np[:, 1:]

print X
print y


c = RandomForestClassifier(n_estimators=4000)
c.fit(X,y)

print c
predictions = c.predict(test_X)

#predictions = clf.predict(test)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})
result.to_csv("rf_predictions.csv", index=False)


