import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns

def Get_Range_of_Normally_Distributed_data(DF, Col):
    '''Accept The DataFrame and Column Name
    and retuurn the Range of the data'''
    
    mean=DF[Col].mean()
    Std= DF[Col].std()
    
    
    return (mean - 1.96*Std,mean + 1.96*Std )


def clean_fit_linear_mod(df, response_col, droppingcol,test_size=.3, rand_state=42):
    '''
    INPUT:
    df - a dataframe holding all the variables of interest
    response_col - a string holding the name of the column
    test_size - a float between [0,1] about what proportion of data should be in the test dataset
    rand_state - an int that is provided as the random state for splitting the data into training and test

    OUTPUT:
    X - cleaned X matrix (dummy and mean imputation)
    y - cleaned response (just dropped na)
    test_score - float - r2 score on the test data
    train_score - float - r2 score on the test data
    lm_model - model object from sklearn
    X_train, X_test, y_train, y_test - output from sklearn train test split used for optimal model

    This function cleans the data and provides the necessary output for the rest of this notebook.
    '''
    #Dropping where the salary has missing values
    df  = df.dropna(subset=[response_col], axis=0)

    #Drop columns with all NaN values
    df = df.dropna(how='all', axis=1)

    #Pull a list of the column names of the categorical variables
    cat_df = df.select_dtypes(include=['object'])
    cat_cols = cat_df.columns

    #dummy all the cat_cols
    for col in  cat_cols:
    
            df = pd.concat([df.drop(col, axis=1), pd.get_dummies(df[col], prefix=col, prefix_sep='_', drop_first=True, dummy_na=False)], axis=1)


    # Mean function
    fill_mean = lambda col: col.fillna(col.mean())
    # Fill the mean
    df = df.apply(fill_mean, axis=0)

    #Split into explanatory and response variables
    X = df.drop(response_col, axis=1)
    y = df[response_col]
   # print(X.columns)

    #Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=rand_state)

    lm_model = LinearRegression(normalize=True) # Instantiate
    lm_model.fit(X_train, y_train) #Fit

    #Predict using your model
    y_test_preds = lm_model.predict(X_test)
    y_train_preds = lm_model.predict(X_train)

    #Score using your model
    test_score = r2_score(y_test, y_test_preds)
    train_score = r2_score(y_train, y_train_preds)
    

    return X, y, test_score, train_score, lm_model, X_train, X_test, y_train, y_test



def fillna(df, columns_name):
    '''df =>>pass the dataframe with na value
       Columns=>> which needs to retuen the value
       
       return dataFrame with the filled value with mean.
     
    
    '''
    
    
    for i in columns_name:
        df[i] =df[i].fillna(df[i].mean())
            
    return df


def splitting_amenities_to_columns(DF,col):
    
    '''Input : DataFrame and column name'''
    
    
    unique_value=[]
    
    for i in range(0,DF[col].shape[0]):
        DF[col][i]=DF[col][i].replace('"','')
        DF[col][i]=DF[col][i].replace('{','')
        DF[col][i]=DF[col][i].replace('}','')

    
    for i in range(0,DF[col].shape[0]):      
        rows=DF[col][i].split(",")
        for j in rows:
        
            if j not in unique_value:
                unique_value.append(j)
            
                    

    df_t = pd.DataFrame() # creating the new DataFrame

#inserting value to dataFrame
    for vlu in unique_value:
    
#    print(vlu)
        lst=[]
    
        for i in range(0,DF[col].shape[0]):   
            rows=DF[col][i].split(",")
            if vlu in rows:
                lst.append(True)
            else:
#            print(False)
                lst.append(False)
            
        df_t[vlu]=lst
            
   
#print(unique_value)

    df_t['no_of_amenities']=df_t.sum(axis=1)
    
    return df_t



def mean_bar(df_t,col_name,Flag,all_dataFrame):
    
    '''input: df_t> input categorical dataFrame
              colname:- y axis of the barplot.
              flag:- true/False 
              dataframe:- overall Dataframe'''
    
    True_dict={}
    ana_DF=pd.DataFrame()

    for i in list(df_t.columns):

        True_dict[i]=float(all_dataFrame[all_dataFrame[i]==Flag].groupby(by=[i]).mean()[col_name])
        
    
    ana_DF=ana_DF.append(True_dict, ignore_index=True)
    ana_DF=ana_DF.T
    
#print(float(tips))
    return ana_DF
    