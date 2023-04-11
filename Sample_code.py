#SAMPLE
#José Carlos Caetano 2013
#MIT Licence

import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn import inspection
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import sys, time
from sklearn import model_selection
from sklearn.model_selection import RandomizedSearchCV
import csv
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
import math



# Setup dataframe
data_filename = 'ML_Gibbs_DatabaseDesc_SI.csv'
output_fn = 'ML_Gibbs_DatabaseDesc_SI_x.csv'


my_descriptors = list()
for desc_name in dir(Descriptors):
    if desc_name in ['BalabanJ','BertzCT']: # Add other descriptor names
        my_descriptors.append(desc_name)


# Prepare calculations
f = open(data_filename,'r')
reader = csv.DictReader(f, delimiter=',')
#
o = list()
for row in reader:
    # Columns to maintain; Edit SI data file accordingly
    nl = dict()
    nl['Solvent Type']                           = row['Solvent Type']
    nl['Solute Type']                            = row['Solute Type']
    nl['Solvent SMILES']                         = row['Solvent SMILES']
    nl['Solute SMILES']                          = row['Solute SMILES']
    nl['Gibbs']                                  = row['Gibbs'] #value of solvation Gibbs energy 

    # Load Compound
    comp_fn = row['Solvent SMILES']
    comp_mol = Chem.MolFromSmiles(comp_fn, sanitize=True)
    # Calculate Catal Descriptors
    for desc in my_descriptors:
        nl[f"{desc}_slv"]=eval(f"Descriptors.{desc}(comp_mol)")      
    # Append nl to output list

    # Load Compound
    comp_fn1 = row['Solute SMILES']
    comp_mol1 = Chem.MolFromSmiles(comp_fn1, sanitize=True)
    # Calculate Catal Descriptors
    for desc in my_descriptors:
        nl[f"{desc}_slt"]=eval(f"Descriptors.{desc}(comp_mol1)")      
    # Append nl to output list
    
    o.append(nl)

with open(output_fn,'w',newline='') as fout:
    writer = csv.DictWriter(fout, fieldnames=o[0].keys())
    writer.writeheader()
    for new_row in o:
        writer.writerow(new_row)

# Clean up stuff
f.close()


data = pd.read_csv('ML_Gibbs_DatabaseDesc_SI_x.csv')

exclude_cols=['Solvent Type','Solute Type','Solvent SMILES','Solute SMILES', 'Gibbs']
ycols=['Gibbs']

X_names=[x for x in data.columns if x not in exclude_cols]
y_names=[x for x in data.columns if x in ycols]

X = data.loc[:,X_names]
y = data.loc[:,y_names]

train_ratio = 0.80
validation_ratio = 0.10
test_ratio = 0.10


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_ratio)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio)) 

X_train = X_train.loc[:,X_names]
X_test = X_test.loc[:,X_names]
X_val = X_val.loc[:,X_names]

X_train=X_train.fillna(0)
X_test=X_test.fillna(0)
X_val=X_val.fillna(0)


# Fit RF model with no HP

model = RandomForestRegressor() #Add hyperparameters accordingly


model.fit(X_train, y_train)
y_train_fitted=model.predict(X_train)
y_test_fitted=model.predict(X_test)
rsq_test = np.corrcoef(y_test,y_test_fitted)[0,1]**2
Score_train = model.score(X_train, y_train)
Score_test = model.score(X_test, y_test)
MSE = np.square(np.subtract(y_test,y_test_fitted)).mean() 
RMSE = math.sqrt(MSE)
S1=mean_absolute_error(y_test,y_test_fitted), RMSE, rsq_test, Score_test, Score_train
