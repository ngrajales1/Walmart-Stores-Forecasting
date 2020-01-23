import pandas as pd
import numpy
import sklearn 

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def clean_data(input_data, impute):
    '''
    inputs:
       input_data: 
           - the input pandas dataframe to manipulate 
       impute: 
           - Set impute if you would like to fill in missing data using MICE
    '''
    data = input_data.copy()
    
    # convert labels A, B, C to numeric 
    new_type = {'A': 1, 'B': 2, 'C': 3}
    data['Type'] = [new_type[item] for item in data['Type']]
    
    # Convert True/False to 1/0
    data['IsHoliday'] = data['IsHoliday'].astype(int)

    # Convert Date to Datetime
    data['Date'] = pd.to_datetime(data['Date'])

    # Grab the week number from the date values 
    data['Week_Number'] = data['Date'].dt.week

    # Drop the Date Column
    data = data.drop(columns=['Date'], axis =1)
    
    output_data = data
    
    if impute == True:
        
        imp_model = IterativeImputer(max_iter=10)

        imp_model.fit(data)

        output_data_impute = imp_model.transform(data)

        output_data_impute = pd.DataFrame(output_data_impute, columns= data.columns)
        
        return output_data_impute
        
    else:
        return output_data