#!/usr/bin/env python
# coding=utf-8
from utils import *

def deal_Sex(train,test):
    train['Sex'] = train['Sex'].apply(lambda x:1 if x=="male" else 0)
    test['Sex'] = test['Sex'].apply(lambda x:1 if x=="male" else 0)
    dummies_Sex_train, dummies_Sex_test = get_dummies(train['Sex'],test['Sex'],prefix='Sex')
    del train['Sex']
    del test['Sex']
    train = pd.concat([train,dummies_Sex_train],axis=1)
    test = pd.concat([test, dummies_Sex_test],axis=1)
    return train, test
  
def Name_Title_Code(x):
    if x=="Mr.":
        return 1
    if x=="Mrs." or x=="Ms." or x=="Lady."  or x == 'Mlle.' or x =='Mme':
        return 2
    if x=="Miss":
        return 3
    if x=="Rev.":
        return 4
    return 5

def deal_Name(train,test):
    train['Name_Title'] = train['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0]).apply(Name_Title_Code)
    test['Name_Title'] = test['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0]).apply(Name_Title_Code)
    dummies_Name_Title_train, dummies_Name_Title_test = get_dummies(train['Name_Title'],test['Name_Title'],prefix='Name_Title')
    train = pd.concat([train, dummies_Name_Title_train],axis=1)
    test = pd.concat([test, dummies_Name_Title_test],axis=1)
    del train['Name']
    del test['Name']
    return train, test

def deal_Age(train, test):
    train['Age'] = train.groupby(['Name_Title','Pclass'])['Age'].transform(lambda x : x.fillna(x.mean()))
    test['Age'] = train.groupby(['Name_Title','Pclass'])['Age'].transform(lambda x : x.fillna(x.mean()))
    del train['Name_Title']
    del test['Name_Title']
  
def deal_SibSp_and_Parch(train,test):
    train['Fam_Size'] = np.where((train['SibSp']+train['Parch']) == 0 , 'Solo',
                        np.where((train['SibSp']+train['Parch']) <= 3,'Nuclear',
                        'Big'))
    test['Fam_Size'] = np.where((test['SibSp']+test['Parch']) == 0 , 'Solo',
                        np.where((test['SibSp']+test['Parch']) <= 3,'Nuclear',
                        'Big'))
    dummies_Fam_Size_train, dummies_Fam_Size_test = get_dummies(train['Fam_Size'],test['Fam_Size'],prefix='Fam_Size')
    train = pd.concat([train, dummies_Fam_Size_train],axis=1)
    test = pd.concat([test, dummies_Fam_Size_test],axis=1)
    del train['SibSp']
    del train['Parch']
    del test['SibSp']
    del test['Parch']
    del train['Fam_Size']
    del test['Fam_Size']
    return train, test

def deal_Ticket(train,test):
    train['Ticket_Lett'] = train['Ticket'].apply(lambda x: str(str(x)[0]))
    train['Ticket_Lett'] = np.where((train['Ticket_Lett']).isin(['1', '2', '3', 'S', 'P', 'C', 'A']), train['Ticket_Lett'],
                                    np.where((train['Ticket_Lett']).isin(['W', '4', '7', '6', 'L', '5', '8']),
                                            'Low_ticket', 'Other_ticket'))
    train['Ticket_Len'] = train['Ticket'].apply(lambda x: len(x))

    test['Ticket_Lett'] = test['Ticket'].apply(lambda x: str(str(x)[0]))
    test['Ticket_Lett'] = np.where((test['Ticket_Lett']).isin(['1', '2', '3', 'S', 'P', 'C', 'A']), test['Ticket_Lett'],
                                    np.where((test['Ticket_Lett']).isin(['W', '4', '7', '6', 'L', '5', '8']),
                                            'Low_ticket', 'Other_ticket'))
    test['Ticket_Len'] = test['Ticket'].apply(lambda x: len(x))

    dummies_Ticket_Lett_train, dummies_Ticket_Lett_test = get_dummies(train['Ticket_Lett'],test['Ticket_Lett'],prefix='Ticket_Lett')
    train = pd.concat([train, dummies_Ticket_Lett_train],axis=1)
    test = pd.concat([test, dummies_Ticket_Lett_test],axis=1)
    del train['Ticket']
    del test['Ticket']
    del train['Ticket_Lett']
    del test['Ticket_Lett']
    return train, test


def deal_Cabin(train, test):
    for i in [train, test]:
        i['Cabin_num1'] = i['Cabin'].apply(lambda x: str(x).split(' ')[-1][1:])
        i['Cabin_num1'].replace('an', np.NaN, inplace = True)
        i['Cabin_num1'] = i['Cabin_num1'].apply(lambda x: int(x) if not pd.isnull(x) and x != '' else np.NaN)
        i['Cabin_num'] = pd.qcut(train['Cabin_num1'],3)
        i['Cabin_Letter'] = i['Cabin'].apply(lambda x: str(x)[0])

    dummies_Cabin_Letter_train, dummies_Cabin_Letter_test = get_dummies(train['Cabin_Letter'],test['Cabin_Letter'],prefix='Cabin_Letter')
    dummies_Cabin_num_train, dummies_Cabin_num_test = get_dummies(train['Cabin_num'],test['Cabin_num'],prefix='Cabin_num')
    train = pd.concat([train, dummies_Cabin_Letter_train, dummies_Cabin_num_train],axis=1)
    test = pd.concat([test, dummies_Cabin_Letter_test, dummies_Cabin_num_test],axis=1)
    del train['Cabin_num1']
    del train['Cabin']
    del test['Cabin_num1']
    del test['Cabin']
    del train['Cabin_num']
    del train['Cabin_Letter']
    del test['Cabin_num']
    del test['Cabin_Letter']
    return train, test

def deal_Embarked(train,test):
    train['Embarked'] = train['Embarked'].fillna('S')
    test['Embarked'] = test['Embarked'].fillna('S')
    dummies_Embarked_train, dummies_Embarked_test = get_dummies(train['Embarked'],test['Embarked'],prefix='Embarked')
    train = pd.concat([train, dummies_Embarked_train],axis=1)
    test = pd.concat([test, dummies_Embarked_test],axis=1)
    del train['Embarked']
    del test['Embarked']
    return train, test

def deal_Pclass(train,test):

    dummies_Pclass_train, dummies_Pclass_test = get_dummies(train['Pclass'],test['Pclass'],prefix='Pclass')

    train = pd.concat([train, dummies_Pclass_train],axis=1)
    test = pd.concat([test, dummies_Pclass_test],axis=1)

    del train['Pclass']
    del test['Pclass']

    return train, test

def get_feats(train,test):
    del train['PassengerId']
    del test['PassengerId']
    del train['Survived']

    #Sex feature
    train, test = deal_Sex(train, test)
    
    #name title feature
    train, test = deal_Name(train, test)

    #age feature
    deal_Age(train, test)

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

    test.loc[(test.Fare.isnull()),'Fare'] = train['Fare'].mean()

    return train.values,test.values
