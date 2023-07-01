### ΔG ML modelling
### JFCAETANO 2023
### MIT Licence

import csv, math, sys
import pandas as pd
import numpy as np
from numpy import mean
from numpy import std
from sklearn import ensemble
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from sklearn import svm
from sklearn.svm import SVR
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split



#Training - this outpus the performace of each model according with different descriptor groups

#select algorithms

my_model = ["RandomForestRegressor(n_estimators= 250, min_samples_split= 2, min_samples_leaf= 1, max_features='sqrt', max_depth= None, bootstrap= False)", "GradientBoostingRegressor(n_estimators= 250, min_samples_split= 5, min_samples_leaf= 5, max_features= 'sqrt', max_depth= None)"]

#descriptor groups
eletronic=['MaxAbsPartialCharge_slt','MaxAbsEStateIndex_slt','MinPartialCharge_slt','MinAbsEStateIndex_slt','MaxPartialCharge_slt','MaxAbsEStateIndex_slv','MaxAbsPartialCharge_slv','MinEStateIndex_slt','MinAbsPartialCharge_slt','MaxEStateIndex_slt','MaxEStateIndex_slv','MinAbsPartialCharge_slv','MinAbsEStateIndex_slv','MinEStateIndex_slv','MaxPartialCharge_slv','MinPartialCharge_slv','MolMR_slt','MolMR_slv']
VSA=['SlogP_VSA2_slt','VSA_EState2_slt','SlogP_VSA1_slt','EState_VSA1_slt','PEOE_VSA1_slt','VSA_EState8_slt','EState_VSA9_slt','VSA_EState6_slt','SMR_VSA3_slt','VSA_EState1_slt','VSA_EState3_slt','PEOE_VSA8_slt','SMR_VSA5_slt','EState_VSA9_slv','PEOE_VSA7_slt','SMR_VSA7_slt','VSA_EState9_slt','SMR_VSA6_slt','SMR_VSA1_slt','VSA_EState1_slv','PEOE_VSA3_slt','SMR_VSA10_slt','PEOE_VSA1_slv','SlogP_VSA5_slt','EState_VSA5_slt','PEOE_VSA2_slt','SlogP_VSA4_slt','SlogP_VSA6_slt','EState_VSA8_slv','PEOE_VSA12_slt','SMR_VSA10_slv','PEOE_VSA14_slt','VSA_EState10_slv','PEOE_VSA13_slt','VSA_EState8_slv','SlogP_VSA12_slt','PEOE_VSA6_slv','SMR_VSA5_slv','VSA_EState4_slt','PEOE_VSA9_slt','SlogP_VSA2_slv','SlogP_VSA10_slt','TPSA_slt','TPSA_slv']
structural=['Chi1_slt','Chi1v_slt','Chi1n_slt','Chi4v_slt','Chi2v_slt','Chi3v_slt','Chi4n_slt','Chi0n_slt','Chi0_slt','Kappa1_slt','Chi0v_slt','Chi0v_slv','Chi3n_slt','Kappa3_slt','Kappa2_slt','Chi2v_slv','Chi1v_slv','Chi0n_slv','Chi1n_slv','Chi1_slv','Kappa3_slv','Chi2n_slt','Kappa2_slv','Kappa1_slv','Chi2n_slv','Chi3v_slv','BalabanJ_slv','HallKierAlpha_slt','HallKierAlpha_slv','BalabanJ_slt','BertzCT_slt','BertzCT_slv','HeavyAtomCount_slt','HeavyAtomCount_slv','HeavyAtomMolWt_slt','HeavyAtomMolWt_slv','MolWt_slt','MolWt_slv','NHOHCount_slt','NHOHCount_slv','NOCount_slt','NOCount_slv','solvent_group','MolLogP_slt','MolLogP_slv','NumHDonors_slt','NumHeteroatoms_slt','NumHAcceptors_slt','NumRotatableBonds_slt','NumValenceElectrons_slt','NumAromaticCarbocycles_slt','NumAromaticRings_slt','NumAromaticHeterocycles_slt','NumSaturatedHeterocycles_slt']

my_group=['eletronic', 'VSA','structural']

#train:test:split
train_ratio = 0.80
validation_ratio = 0.10
test_ratio = 0.10

o = list()
#munber of trials
for group in my_group:
    data = pd.read_csv('Model_Start.csv')
    include_cols=eval(f"{group}")
    X_names=[x for x in data.columns if x in include_cols]

    X = data.loc[:,X_names]
    X = X.fillna(0)
    y = data.loc[:,"dG"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_ratio)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio)) 

    
    for mod in my_model:    
        model = eval(f"{mod}")
        model.fit(X_train, y_train)
        y_train_fitted=model.predict(X_train)
        y_test_fitted=model.predict(X_test)
        rsq_test = np.corrcoef(y_test,y_test_fitted)[0,1]**2
        Score_train = model.score(X_train, y_train)
        Score_test = model.score(X_test, y_test)
        MAE=mean_absolute_error(y_test,y_test_fitted)
        cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=47)
        n_scores = cross_val_score(model, X_test, y_test, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
        STD=std(n_scores)
        MSE = np.square(np.subtract(y_test,y_test_fitted)).mean() 
        RMSE = math.sqrt(MSE)
    
        nl = dict()
        nl[f"Group"]=f"{group}"
        nl[f"Algorithm"]=f"{mod}"
        nl[f"Score_test"]=Score_test
        nl[f"Score_train"]=Score_train
        nl[f"MAE"]=MAE
        nl[f"RMSE"]=RMSE
        nl[f"STD"]=STD
        nl[f"R2"]=rsq_test

        o.append(nl)

output_fn = 'desc_group_trial.csv'
with open(output_fn,'w',newline='') as fout:
    writer = csv.DictWriter(fout, fieldnames=o[0].keys())
    writer.writeheader()
    for new_row in o:
        writer.writerow(new_row)
