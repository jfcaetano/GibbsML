### ΔG ML modelling
### JFCAETANO 2023
### MIT Licence

import csv, math
import pandas as pd
import numpy as np
from numpy import mean
from numpy import std
from sklearn import ensemble
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from sklearn import svm
from sklearn.svm import SVR
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold


#solvent identification by InChIKey
solvents = ['KBPLFHHGFOOTCA-UHFFFAOYSA-N', 'DCAYPVUWAIABOU-UHFFFAOYSA-N', 'YXFVVABEGXRONW-UHFFFAOYSA-N', 'LFQSCWFLJHTTHZ-UHFFFAOYSA-N', 'CSCPPACGZOOCGX-UHFFFAOYSA-N', 'WYURNTSHIVDZCO-UHFFFAOYSA-N', 'YMWUJEATGCHHMB-UHFFFAOYSA-N', 'IAZDPXIOMUYVGZ-UHFFFAOYSA-N', 'XLYOFNOQVPJJNP-UHFFFAOYSA-N']

#select desired algorithm
my_model = ["GradientBoostingRegressor(random_state=47)"]

o = list()

for solvent in solvents:

    data = pd.read_csv('Model_Start.csv')
    data_test=data[data['Solvent Key'] == (f"{solvent}")]
    data.drop(data[data['Solvent Key'] == (f"{solvent}")].index, inplace = True)
    data_train=data
    
    y_train=data_train.loc[:,"dG"]
    
    
    #best descriptors selction
    include_cols=['MolMR_slt','TPSA_slt','Chi1_slt','NOCount_slt','NHOHCount_slt','NumHDonors_slt','HeavyAtomMolWt_slt','BertzCT_slt','Chi1v_slt','Chi1n_slt','MolWt_slt','SlogP_VSA2_slt','MaxAbsPartialCharge_slt','Chi4v_slt','Chi2v_slt','Chi3v_slt','MaxAbsEStateIndex_slt','Chi4n_slt','NumHeteroatoms_slt','MinPartialCharge_slt','Chi0n_slt','Chi0_slt','MolLogP_slt','VSA_EState2_slt','MolLogP_slv','Kappa1_slt','NumHAcceptors_slt','BalabanJ_slv','HeavyAtomMolWt_slv','Chi0v_slt','MinAbsEStateIndex_slt','SlogP_VSA1_slt','Chi0v_slv','EState_VSA1_slt','MaxPartialCharge_slt','NHOHCount_slv','PEOE_VSA1_slt','HallKierAlpha_slt','VSA_EState8_slt','MaxAbsEStateIndex_slv','HeavyAtomCount_slt','EState_VSA9_slt','Chi3n_slt','MaxAbsPartialCharge_slv','TPSA_slv','MinEStateIndex_slt','MolWt_slv','MinAbsPartialCharge_slt','NumRotatableBonds_slt','MaxEStateIndex_slt','Kappa3_slt','VSA_EState6_slt','SMR_VSA3_slt','Kappa2_slt','MaxEStateIndex_slv','HallKierAlpha_slv','Chi2v_slv','BalabanJ_slt','VSA_EState1_slt','VSA_EState3_slt','NumValenceElectrons_slt','PEOE_VSA8_slt','SMR_VSA5_slt','NumAromaticCarbocycles_slt','Chi1v_slv','EState_VSA9_slv','PEOE_VSA7_slt','MinAbsPartialCharge_slv','SMR_VSA7_slt','NumAromaticRings_slt','VSA_EState9_slt','SMR_VSA6_slt','SMR_VSA1_slt','VSA_EState1_slv','PEOE_VSA3_slt','SMR_VSA10_slt','PEOE_VSA1_slv','Chi0n_slv','MinAbsEStateIndex_slv','SlogP_VSA5_slt','Chi1n_slv','Chi1_slv','EState_VSA5_slt','PEOE_VSA2_slt','SlogP_VSA4_slt','MinEStateIndex_slv','SlogP_VSA6_slt','Kappa3_slv','EState_VSA8_slv','PEOE_VSA12_slt','Chi2n_slt','HeavyAtomCount_slv','SMR_VSA10_slv','PEOE_VSA14_slt','VSA_EState10_slv','NumAromaticHeterocycles_slt','solvent_group','PEOE_VSA13_slt','VSA_EState8_slv','MaxPartialCharge_slv','NOCount_slv','MinPartialCharge_slv','NumSaturatedHeterocycles_slt','SlogP_VSA12_slt','MolMR_slv','PEOE_VSA6_slv','Kappa2_slv','SMR_VSA5_slv','Kappa1_slv','Chi2n_slv','VSA_EState4_slt','PEOE_VSA9_slt','SlogP_VSA2_slv','Chi3v_slv','BertzCT_slv','SlogP_VSA10_slt']
    X_names=[x for x in data_train.columns if x in include_cols]
    X_train = data_train.loc[:,X_names]
    X_train = X_train.replace(np.nan,0)

    y_test=data_test.loc[:,"dG"]
    y_test_key=data_test.loc[:,"Solvent Key"]
    X_names=[x for x in data_test.columns if x in include_cols]
    X_test = data_test.loc[:,X_names]
    X_test = X_test.replace(np.nan,0)

    for mod in my_model:
        
        model = eval(f"{mod}")
        model.fit(X_train, y_train)
        y_train_fitted=model.predict(X_train)
        y_test_fitted=model.predict(X_test)
        rsq_train = np.corrcoef(y_train,y_train_fitted)[0,1]**2
        rsq_test = np.corrcoef(y_test,y_test_fitted)[0,1]**2
        Score_train = model.score(X_train, y_train)
        Score_test = model.score(X_test, y_test)
        MSE = np.square(np.subtract(y_test,y_test_fitted)).mean() 
        RMSE = math.sqrt(MSE)
        MAE=mean_absolute_error(y_test,y_test_fitted)
        cv = RepeatedKFold(n_splits=10, n_repeats=1, random_state=47)
        n_scores = cross_val_score(model, X_test, y_test, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
        STD=std(n_scores)     
    
        nl = dict()
        
        nl[f"Solvent"]=f"{solvent}"
        nl[f"Algorithm"]=f"{mod}"
        nl[f"Score_test"]=Score_test
        nl[f"Score_train"]=Score_train
        nl[f"MAE"]=MAE
        nl[f"RMSE"]=RMSE
        nl[f"STD"]=STD
        nl[f"R2"]=rsq_test

        o.append(nl)

        #Routine to overlook prediction results
        d = {'y_test': y_test, 'y_test_pred': y_test_fitted, 'y_test_key': y_test_key}
        df = pd.DataFrame(d)
        frames = [df, X_test]
        result = pd.concat(frames, axis="columns")

        a=result.y_test
        b=result.y_test_pred

        ev=(abs((b-a)/b))*100
        ev = pd.DataFrame(ev)
        ev.rename(columns={0 :'eval'}, inplace=True)

        ev1=(abs((b-a)/b))
        ev1 = pd.DataFrame(ev1)
        ev1.rename(columns={0 :'eval1'}, inplace=True)

        frames = [result, ev, ev1]
        ev.colums = ['eval']
        full_test = pd.concat(frames, axis="columns")
        #file including performnace results
        full_test.to_csv('full_test.csv')
        

output_fn = 'training_eval_solvent.csv'
with open(output_fn,'w',newline='') as fout:
    writer = csv.DictWriter(fout, fieldnames=o[0].keys())
    writer.writeheader()
    for new_row in o:
        writer.writerow(new_row)
