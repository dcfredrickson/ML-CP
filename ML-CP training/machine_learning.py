
# Machine Learning Code, last edited March 12th, 2024

# import packages
import pandas as pd
import numpy as np
from tqdm import tqdm
import sklearn
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import pickle
print("""
  __  __   _         _____   _____
 |  \/  | | |       / ____| |  __ \\
 | \  / | | |      | |      | |__) |
 | |\/| | | |      | |      |  ___/
 | |  | | | |____  | |____  | |
 |_|  |_| |______|  \_____| |_|
 """)
print("\nRunning Scikit-learn version " + str(sklearn.__version__) + "\n")

# Read data from excel spreadsheets
X_df = pd.read_excel('contact_data_by_cp.xlsx')

# Pull out target values
y = X_df.pop('contact_CP')

# function that plots a parity plot
def parity_plot(x,y,title):
    pad = 0.001
    std = np.std(abs(y-x.to_numpy()))
    x1 = np.linspace(min(x)-pad,max(x)+pad)
    x2 = [list(x)[i] for i in range(len(x)) if list(abs(y-x.to_numpy()))[i] > 3*std]
    y2 = [list(y)[i] for i in range(len(x)) if list(abs(y-x.to_numpy()))[i] > 3*std]
    ymin,ymax = float(x.min())-pad,float(x.max())+pad
    return(std)

# function that adds elemental information to dataframe
def get_element_feature_df_vv_en(input_df):
    element_df = pd.read_excel('element_data.xlsx') # read in the element feature reference spreadsheet
    element_df.set_index('Symbol',inplace=True) # re-index the reference spreadsheet by element name
    element_df_names = list(element_df.columns) # create a list of names of all elements in the reference sheet
    column_names = [name + '_element1' for name in list(element_df.columns)]+[name + '_element2' for name in list(element_df.columns)]+[name + '_contact1' for name in list(element_df.columns)]+[name + '_contact2' for name in list(element_df.columns)]+['element1_n','element2_n','contact1_homo','contact1_hetero','contact2_homo','contact2_hetero','length','vorovol_en']
    feature_df = pd.DataFrame(columns = column_names) # create an empty dataframe with the appropriate column names
    for n in tqdm(range(len(input_df)), bar_format='{l_bar}{bar:73}{r_bar}{bar:-73b}'):
        in_list = input_df.iloc[n]
        element_list = [in_list[0],in_list[2],in_list[4],in_list[7]]
        # create list of the features for each element in the contact
        features = [element_df.loc[element].values for element in element_list]
        # write each row of the dataframe based on each entry
        l1 = list(features[0])+list(features[1])+list(features[2])+list(features[3])
        l2 = [in_list[1],in_list[3],in_list[5],in_list[6],in_list[8],in_list[9],in_list[10],in_list[14]]
        feature_df.loc[n] = l1 + l2
    return feature_df

# add descriptors to the DF based on element identity, CN, bond len., # of homo/heteroatomic contacts, and voronoi descriptors
X_final = get_element_feature_df_vv_en(X_df)

# Add new features
X = X_final.copy()
X['Sum Metalic Radii'] = (X['Metallic Radii_contact1']+X['Metallic Radii_contact2'])/100
X['Metalic Volume'] = (X['Metallic Radii_contact1']/100)**3+(X['Metallic Radii_contact2']/100)**3
X['Length Cubed'] = X['length']**3
X['Pauling EN Range'] = abs(X['Pauling Electronegativity_contact1']-X['Pauling Electronegativity_contact2'])
X['Diff Len'] = X['Sum Metalic Radii']-X['length']
X['Diff Vorovol'] = X['vorovol_en']-X['Length Cubed']
X['Diff Metvol'] = X['Metalic Volume']-X['Length Cubed']

# convert ["element1", "element2"] data to ["avg", "std_dev"] format. This allows for the addition of ternaries
# Also, weight the average and std_dev by the stoichiometry
cols = X.columns
for i in range(14):
    X[cols[i][:-8]+"composition_avg"] = (X[cols[i]]*X['element1_n'] + X[cols[i+14]]*X['element2_n'])/(X['element1_n'] + X['element2_n'])
    X[cols[i][:-8]+"composition_std"] = np.sqrt((X['element1_n']*(X[cols[i]] - X[cols[i][:-8]+"composition_avg"])**2 + (X['element2_n']*(X[cols[i+14]] - X[cols[i][:-8]+"composition_avg"])**2))/(X['element1_n'] + X['element2_n']))
X = X.iloc[:,28:]
cols = X.columns
for i in range(14):
    X[cols[i][:-8]+"contact_avg"] = (X[cols[i]] + X[cols[i+14]])/2
    X[cols[i][:-8]+"contact_std"] = abs(X[cols[i]] - X[cols[i+14]])/2
X = X.iloc[:,28:]
cols = X.columns
for i in range(2):
    X["contacts_avg_"+cols[2+i][9:]] = (X[cols[2+i]] + X[cols[4+i]])/2
    X["contacts_std_"+cols[2+i][9:]] = abs(X[cols[2+i]] - X[cols[4+i]])/2
    X = X.drop(columns=[cols[2+i], cols[4+i]])
X = X.iloc[:,2:]

# split data ordered by CP so that 8 data points go to training, one goes to testing, and one goes to validation.
X_train = pd.DataFrame(columns=X.columns)
X_test = pd.DataFrame(columns=X.columns)
X_val = pd.DataFrame(columns=X.columns)
y_train = pd.Series(dtype="float", name="contact_CP")
y_test = pd.Series(dtype="float", name="contact_CP")
y_val = pd.Series(dtype="float", name="contact_CP")

#"""
# This splits the data evenly, taking every 10n-th data point for validation, 10n-th+1 for testing, and the remaining 8 for training
for i in tqdm(range(len(X.index)), bar_format='{l_bar}{bar:73}{r_bar}{bar:-73b}'):
    x_row = X.iloc[i]
    y_row = y.iloc[i]
    if (i)%10 == 0: #0, 10, 20, 30, ...
        X_val.loc[len(X_test.index)] = x_row
        y_val.loc[len(y_test.index)] = y_row
    elif (i+1)%10 == 0: #9, 19, 29, 39, ...
        X_test.loc[len(X_val.index)] = x_row
        y_test.loc[len(y_val.index)] = y_row
    else: # 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, ...
        X_train.loc[len(X_train.index)] = x_row
        y_train.loc[len(y_train.index)] = y_row

"""
X_test = X.iloc[:294]
X_val = X.iloc[294:536]
X_train = X.iloc[536:]
y_test = y.iloc[:294]
y_val = y.iloc[294:536]
y_train = y.iloc[536:]
#"""

# prepare target dataframes
threshold1 =  0.0000 # define threshold (0 = split between overall + and - CPs)
threshold2 = -0.0035 # define threshold (0 = split between overall + and - CPs)
y_train_df,y_test_df,y_val_df = pd.DataFrame(y_train),pd.DataFrame(y_test),pd.DataFrame(y_val)

# create new columns in target dataframes for binary classification
y_train_df['over'] = np.where(y_train_df['contact_CP']>threshold1, 1, 0) + np.where(y_train_df['contact_CP']>threshold2, 1, 0)
y_test_df['over'] = np.where(y_test_df['contact_CP']>threshold1, 1, 0) + np.where(y_test_df['contact_CP']>threshold2, 1, 0)
y_val_df['over'] = np.where(y_val_df['contact_CP']>threshold1, 1, 0) + np.where(y_val_df['contact_CP']>threshold2, 1, 0)

# train and fit RFC model
rfc = RandomForestClassifier(
    n_estimators=300,
    max_features=20,
    random_state=0,
    max_depth=15
)
rfc.fit(X_train,y_train_df['over'])

# predict target values for validation set
y_pred_RFC = rfc.predict(X_test)
y_train_pred = rfc.predict(X_train)
y_val_pred = rfc.predict(X_val)

print('')
# calculate and print CV score, show confusion matrix
scores = cross_val_score(rfc,X_test,y_test_df['over'],cv=10)

# calculate and print CV score, show confusion matrix
scores2 = cross_val_score(rfc,X_val,y_val_df['over'],cv=10)

# save trained model in a pickle file
f = open('rfc.pickle', 'wb')
pickle.dump(rfc, f)
f.close()

# Identify the top X most important features
feat_import = rfc.feature_importances_
imp_df = pd.DataFrame(columns=X_train.columns)
imp_df.loc[0]=feat_import
imp_df = imp_df.transpose().sort_values(by=0,ascending=False).head(30).transpose()

# Add classifier predictions to the dataframe used for the regressor.
X_train_rfr = X_train.join(pd.DataFrame(rfc.predict(X_train),index=X_train.index,columns=["rfc_predictions"]))
X_test_rfr = X_test.join(pd.DataFrame(rfc.predict(X_test),index=X_test.index,columns=["rfc_predictions"]))
X_val_rfr = X_val.join(pd.DataFrame(rfc.predict(X_val),index=X_val.index,columns=["rfc_predictions"]))

#rfr = RandomForestRegressor(n_estimators=300,max_features=15,random_state=0) # define the parameters of the RFR model
rfr = ExtraTreesRegressor(
    n_estimators = 300,
    max_features = 30,
    random_state = 0,
    max_depth = 20,
)
rfr.fit(X_train_rfr,y_train) # fit the RFR model to the training data
y_pred = rfr.predict(X_test_rfr) # predict target values for the validation set
y_val_pred = rfr.predict(X_val_rfr) # predict target values for the validation set

#============================================================================================

# TESTING
mae = round(round(mean_absolute_error(y_test, y_pred)*100,3)/100,5) # calculate the MAE of the validation predictions
rmse = round(round(np.sqrt(mean_squared_error(y_test, y_pred))*100,3)/100,5) # calculate the RMSE of the validation predictions
r2 = round(r2_score(y_test, y_pred,),5) # calculate R2 score

confussion = [0,0,0,0]
for i in range(len(y_test.index)):
    if y_test.iloc[i] > 0 and y_pred[i] > 0: # guess positive, correct
        confussion[0]+=1
    elif y_test.iloc[i] < 0 and y_pred[i] < 0: # guess negative, correct
        confussion[2]+=1
    elif y_test.iloc[i] > 0 and y_pred[i] < 0: # guess negative, wrong
        confussion[1]+=1
    else: # guess positive, wrong
        confussion[3]+=1

# graph the parity plot
std = round(parity_plot(y_test,y_pred,'full feature set'), 5)
#print(confussion)

#============================================================================================

# VALIDATION
mae2 = round(round(mean_absolute_error(y_val, y_val_pred)*100,3)/100,5) # calculate the MAE of the validation predictions
rmse2 = round(round(np.sqrt(mean_squared_error(y_val, y_val_pred))*100,3)/100,5) # calculate the RMSE of the validation predictions
r22 = round(r2_score(y_val, y_val_pred,),5) # calculate R2 score

confussion = [0,0,0,0]
for i in range(len(y_val.index)):
    if y_val.iloc[i] > 0 and y_val_pred[i] > 0: # guess positive, correct
        confussion[0]+=1
    elif y_val.iloc[i] < 0 and y_val_pred[i] < 0: # guess negative, correct
        confussion[2]+=1
    elif y_val.iloc[i] > 0 and y_val_pred[i] < 0: # guess negative, wrong
        confussion[1]+=1
    else: # guess positive, wrong
        confussion[3]+=1

# graph the parity plot
std2 = round(parity_plot(y_val,y_val_pred,'full feature set'), 5)
#print(confussion)

f = open('rfr.pickle', 'wb')
pickle.dump(rfr, f)
f.close()

# Identify the top X most important features
feat_import = rfr.feature_importances_
imp_df = pd.DataFrame(columns=X_train_rfr.columns)
imp_df.loc[0]=feat_import
imp_df = imp_df.transpose().sort_values(by=0,ascending=False).head(30).transpose()

# Print Summary Data
print('')
print('\t\tTESTING\t\tVALIDATION')
print(f"Classifier:\t{round(scores.mean(),4)}\t\t{round(scores2.mean(),4)}")
print(f'MAE:\t\t{round(mae,4)}\t\t{round(mae2,4)}')
print(f'RMSE:\t\t{round(rmse,4)}\t\t{round(rmse2,4)}')
print(f'R2:\t\t{round(r2,4)}\t\t{round(r22,4)}')
print(f'STDDEV:\t\t{round(std,4)}\t\t{round(std2,4)}')

