#RDKIT CONVERSION
#JF Caetano May 2023
#MIT Licence

import rdkit, sys, time, csv
from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd
import numpy as np


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

