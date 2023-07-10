# GibbsML

This is the repository of the JCIM paper "Supervised Machine Learning Model to Predict Solvation Gibbs Energy".
The full database is available at Zenodo (https://doi.org/10.5281/zenodo.8121619)


# File Overview

Database/

ML_Gibbs_Full_Database.csv: Complete dataset with descriptor calculations

ML_Gibbs_Full_Database.xlsx: Raw complete dataset without descriptors


Scripts/

rdkit_conversion.py: Calculation of desired RdKit descriptors using the raw database

model_calculations.py: Model calculations using desired algorithms with all calculated descriptors

model_descriptor_groups.py: Model performance using only best descriptors for model optimization

permutation_importance.py: Routine to determine best descriptors using permuataion importance

solvent_holdout_tests.py: Routine to perform solvent holdout tests using best descriptors



Results/

ML_Gibbs_Full_Results_SI.xlsx: File including all model results presented in the paper (including permuation importance, model statistical performance, solvent holdout tests and descriptors group performance determinations

# Authorship
Code was written by José Ferraz-Caetano, under the supervision of Filipe Teixeira and Natália Cordeiro.

# Acknowledgements
This code was developed at the Univerisity of Porto and was supported by the "Fundação para a Ciência e Tecnologia" (FCT/MCTES) to LAQV-REQUIMTE Lab (UIDP/50006/2020). JFC’s PhD Fellowship is supported by the doctoral Grant (SFRH/BD/151159/2021) financed by FCT, with funds from the Portuguese State and EU Budget through the Social European Fund and Programa Por_Centro, under the MIT Portugal Program.

# BibTex
