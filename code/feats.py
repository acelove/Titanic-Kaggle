#!/usr/bin/env python
# coding=utf-8
from utils import *
import sklearn.preprocessing as preprocessing

def deal_Sex(train,test):

    for data in [train,test]:
        data['Sex'] = data['Sex'].apply(lambda x:1 if x=="male" else 0)
    
    dummies_Sex_train, dummies_Sex_test = get_dummies(train['Sex'],test['Sex'],prefix='Sex')
    train = pd.concat([train,dummies_Sex_train],axis=1)
    test = pd.concat([test, dummies_Sex_test],axis=1)

    drop([train,test],['Sex'])

    return train, test
  
def Name_Title_Code(x):

    if x=="Mr.":
        return 1
    if x=="Mrs." or x=="Ms."   or x == 'Mlle.' or x =='Mme':
        return 2
    if x=="Miss." or x=="Lady.":
        return 3
    if x=="Rev.":
        return 4
    return 5

def deal_Name(train,test):

    for data in [train,test]:
        data['Name_Title'] = data['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0]).apply(Name_Title_Code)
    
    dummies_Name_Title_train, dummies_Name_Title_test = get_dummies(train['Name_Title'],test['Name_Title'],prefix='Name_Title')
    train = pd.concat([train, dummies_Name_Title_train],axis=1)
    test = pd.concat([test, dummies_Name_Title_test],axis=1)

    drop([train,test],['Name'])

    return train, test

def deal_Age(train, test):

    train['Age'] = train.groupby(['Name_Title','Pclass'])['Age'].transform(lambda x : x.fillna(x.mean()))
    test['Age'] = train.groupby(['Name_Title','Pclass'])['Age'].transform(lambda x : x.fillna(x.mean()))

    drop([train,test],['Name_Title'])
    return train,test
  
def deal_SibSp_and_Parch(train,test):

    for data in [train,test]:
        data['Fam_Size'] = data['SibSp'] + data['Parch']
        data['Fam_Size'] = data['Fam_Size'].apply(lambda x: 'Solo' if x==0 else ('Normal' if x<=3 else ('Middle' if x<7 else 'Big')))

    dummies_Fam_Size_train, dummies_Fam_Size_test = get_dummies(train['Fam_Size'],test['Fam_Size'],prefix='Fam_Size')
    train = pd.concat([train, dummies_Fam_Size_train],axis=1)
    test = pd.concat([test, dummies_Fam_Size_test],axis=1)

    drop([train,test],['SibSp','Parch','Fam_Size'])

    return train, test

def deal_Ticket(train,test):

    for data in [train,test]:
        data['Ticket_Lett'] = data['Ticket'].apply(lambda x: str(str(x)[0]))
        data['Ticket_Lett'] = data['Ticket_Lett'].apply(lambda x : x if x in ['1', '2', '3', 'S', 'P', 'C', 'A'] else ('Low_ticket' if x in ['W', '4', '7', '6', 'L', '5', '8'] else 'Other_ticket'))
        data['Ticket_Len'] = data['Ticket'].apply(lambda x: len(x))

    dummies_Ticket_Lett_train, dummies_Ticket_Lett_test = get_dummies(train['Ticket_Lett'],test['Ticket_Lett'],prefix='Ticket_Lett')
    train = pd.concat([train, dummies_Ticket_Lett_train],axis=1)
    test = pd.concat([test, dummies_Ticket_Lett_test],axis=1)

    drop([train,test],['Ticket','Ticket_Lett'])

    return train, test


def deal_Cabin(train, test):

    for i in [train, test]:
        i['Cabin_num1'] = i['Cabin'].apply(lambda x: str(x).split(' ')[-1][1:] if not pd.isnull(x) else np.nan)
        i['Cabin_num1'] = i['Cabin_num1'].apply(lambda x: int(x) if not pd.isnull(x) and x != '' else np.nan)
        i['Cabin_num'] = pd.qcut(train['Cabin_num1'],3)
        i['Cabin_Letter'] = i['Cabin'].apply(lambda x: str(x)[0])

    dummies_Cabin_Letter_train, dummies_Cabin_Letter_test = get_dummies(train['Cabin_Letter'],test['Cabin_Letter'],prefix='Cabin_Letter')
    dummies_Cabin_num_train, dummies_Cabin_num_test = get_dummies(train['Cabin_num'],test['Cabin_num'],prefix='Cabin_num')
    train = pd.concat([train, dummies_Cabin_Letter_train, dummies_Cabin_num_train],axis=1)
    test = pd.concat([test, dummies_Cabin_Letter_test, dummies_Cabin_num_test],axis=1)

    drop([train,test],['Cabin_num1','Cabin','Cabin_num','Cabin_Letter'])

    return train, test

def deal_Embarked(train,test):
    
    dummies_Embarked_train, dummies_Embarked_test = get_dummies(train['Embarked'],test['Embarked'],prefix='Embarked')
    train = pd.concat([train, dummies_Embarked_train],axis=1)
    test = pd.concat([test, dummies_Embarked_test],axis=1)

    drop([train,test],['Embarked'])

    return train, test

def deal_Pclass(train,test):

    dummies_Pclass_train, dummies_Pclass_test = get_dummies(train['Pclass'],test['Pclass'],prefix='Pclass')

    train = pd.concat([train, dummies_Pclass_train],axis=1)
    test = pd.concat([test, dummies_Pclass_test],axis=1)

    drop([train,test],['Pclass'])

    return train, test

def get_feats(train,test):

    drop([train],['PassengerId','Survived'])
    drop([test],['PassengerId'])

    #Sex feature
    train, test = deal_Sex(train, test)
    
    #name title feature
    train, test = deal_Name(train, test)

    #age feature
    train,test = deal_Age(train, test)

    #Family feature
    train, test = deal_SibSp_and_Parch(train, test)

    #Ticket feature
    train,test = deal_Ticket(train, test)


    #Cabin feature
    train, test = deal_Cabin(train, test)

    #embark feature
    train, test = deal_Embarked(train, test)

    #Pclass feature
    train, test = deal_Pclass(train, test)

    test.loc[(test.Fare.isnull()),'Fare'] = test['Fare'].mean()


    return train,test
