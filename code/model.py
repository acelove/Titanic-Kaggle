#!/usr/bin/env python
# coding=utf-8
from utils import *
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
def RF_model():
    return RandomForestClassifier(criterion='gini', 
                             n_estimators=1200,
                             min_samples_split=16,
                             min_samples_leaf=1,
                             max_features='auto',
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1)

def GBC_model():
    return GradientBoostingClassifier( 
                             n_estimators=1200,
                             min_samples_split=16,
                             min_samples_leaf=1,
                             max_features='auto',
                             random_state=1,)

def XGB_model():
    return xgb.XGBRegressor(max_depth=10, 
                        learning_rate=0.04, 
                        n_estimators=1400, 
                        silent=False, 
                        objective='reg:linear', 
                        nthread=-1, 
                        gamma=0,
                        min_child_weight=1, 
                        max_delta_step=0, 
                        subsample=0.85, 
                        colsample_bytree=0.7, 
                        colsample_bylevel=1, 
                        reg_alpha=0, 
                        reg_lambda=1, 
                        scale_pos_weight=1, 
                        seed=1440, 
                        missing=None)
