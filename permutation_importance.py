### ΔG ML modelling
### JFCAETANO 2023
### MIT Licence

import csv, math, sys, time
import pandas as pd
import numpy as np
from numpy import mean
from numpy import std
from sklearn import ensemble
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn import inspection


#ALL PERMUTATION IMPORTANCE

#train:test:split
train_ratio = 0.80
validation_ratio = 0.10
test_ratio = 0.10

def main(**args):
    debug=False
    if(debug): print(args)
    log=open(args['ofn'],'w',buffering=1)
    basename=args['ofn'][:-4]
    #Get indexes for train and test sets:

    data = pd.read_csv('Model_Start.csv')

    exclude_cols=['Solvent Type','Solute Type','Solvent SMILES','Solute SMILES', 'dG', 'Solvent Key', 'Solute Key']
    ycols=['dG', 'Solvent Type']

    X_names=[x for x in data.columns if x not in exclude_cols]
    y_names=[x for x in data.columns if x in ycols]
    
    X = data.loc[:,X_names]
    X = X.fillna(0)
    y = data.loc[:,'dG']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_ratio)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio)) 

    #select algorithms to test
    my_model = ["RandomForestRegressor(random_state=47)", "GradientBoostingRegressor(random_state=47)"]

    for mod in my_model:
        model = eval(f"{mod}")
        model.fit(X_train, y_train)
        # Variable Importance
        vn_len=1+max([len(X) for X in X_names])
        p_imp=inspection.permutation_importance(model,X_test,y_test,n_repeats=10)
        p_imp_av=p_imp['importances_mean']
        p_imp_sd=p_imp['importances_std']
        i_lst=list(zip(X_names,model.feature_importances_,100.0*(model.feature_importances_/np.sum(model.feature_importances_)),p_imp_av,p_imp_sd,100.0*(p_imp_av/sum(p_imp_av))))
        i_lst.sort(key=lambda x: x[3],reverse=True)
        log.write("-"*(vn_len+55)+'\n')
        log.write("-"*(vn_len+55)+'\n')
        log.write(f"{mod}\n")
        log.write("\nVariable Importance (n_repeats=10) \n")
        log.write("-"*(vn_len+55)+'\n')
        log.write(f"{'Variable':^{vn_len}s} {'Gini':^10s} {'%Gini':^10s} {'PI(mean)':^10s} {'PI(std)':^10s} {'%PI(mean)':^10s}\n")
        log.write("-"*(vn_len+55)+'\n')
        n99=0
        i_sum=0.0
        for l in i_lst:
            log.write(f"{l[0]:{vn_len}s} {l[1]:10.4g} {l[2]:10.1f} {l[3]:10.4g} {l[4]:10.4g} {l[5]:10.1f}\n")
            if(i_sum<=99.0):
                n99+=1
                i_sum += l[5]
        log.write("-"*(vn_len+55)+'\n\n')
        log.write(f"@ 99% of variable importance (by permutation) achieved by {n99} predictors ({i_sum:5.1f}%):\n")
        

if(__name__=='__main__'):
    opts={}
    n=1
    opts['ifn']=''
    opts['ofn']=''
    opts['target']=list()
    opts['exclude']=list()
    while(n<len(sys.argv)):
        if(sys.argv[n]=='--notes'):
            print(notes)
            sys.exit(0)
        elif(sys.argv[n]=='--help'):
            print(__doc__)
            sys.exit(0)
        elif(sys.argv[n]=='-t'):
            n+=1
            opts['target'].append(sys.argv[n])
        elif(sys.argv[n]=='-e'):
            n+=1
            opts['exclude'].append(sys.argv[n])
        elif(sys.argv[n]=='-o'):
            n+=1
            opts['ofn']=sys.argv[n]
        else: 
            opts['ifn']=sys.argv[n]
        n += 1
    if opts['ofn']=='':
        now=time.localtime()
        opts['ofn']=f"{now[0]-2000:02d}{now[1]:02d}{now[2]:02d}-{now[3]:02d}{now[4]:02d}-Best_Features_run_.log"
    main(**opts)
