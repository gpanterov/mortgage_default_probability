"""
George Panterov
Prepared for Serigne Diop (Vero Capital Management)

This script prepares, cleans the data for the multinomial model
and runs the model.
Dependenies
pandas and numpy
See http://code.google.com/p/pandas/
    and http://numpy.scipy.org/
The class that implements the multinomial model is in the multinomial_main.py file

Due to missingness of observations we drop months 6,7 and 8 from 2011.
These could be modified by chaninging the dropperiods list below.

We also drop several variables that are deemed "irrelevant" but could be
easily incorporated in the model by modifying the dropvars list below.


###################################
Instruction on running the model ##
###################################

1) Make sure all dependencies are installed (pandas and numpy)
and that path to data files is correct as well as the multinomial_main
file is in the proper folder for import. This code works with the loans.csv file!

2) Select which variables to include by modifying the modelvars list below.
(Pay attention to the dropped variables in dropvars list earlier. A dropped variable
cannot be included in the modevars list)

3) Select the macro variable to be included in the model (using macrovars.csv file)
They are slected in the column_stack routine by being appended onto the X matrix
If forexample, one wishes to run the model without HPA simply modify the line from
Z = np.column_stack((X, Xhpa, Xunem))
to
Z = np.column_stack((X, Xunem))

4) Select the conditioning state by modifying the given_state variable

5) Select the type of model to run by modifying me_type input parameter in the fit() method
'me' = maximum entropy
'gme' = generalized maximum entropy (noisy moments)

5) Run the script

"""

# organize the data for the markov_main model
from pandas import *
import numpy as np
import datetime
import multinomial_main as main_model

# Set the seed for replication purposes
np.random.seed(12345)

def map_states(val):
    """Maps states into integers"""
    states_map = {'C':0, '30':1, '60':2, '90':3, '120':4, 'REO':5 ,'F':6}
    return states_map[val]

def create_states_dict(Y):
    s = Y.value_counts()
    mydict = {}
    r=0
    for indx in s.index:
	mydict[indx] = r
	r += 1
    return mydict

def get_date(val):
    """Reads a string and transforms in datetime.datetime object"""
    val = val.split("_")[2]
    month = int(val[:2])
    year = int(val[2:])
    day = 30
    if month == 2:
        day = 29
    return datetime.datetime(day=day, month=month, year=year).date()

def get_date_macro(val):
    """Transform string to datetime in macro vars datafile"""
    month = int(val.split("/")[0])
    day = int(val.split("/")[1])
    year = int(val.split("/")[2])
    return datetime.datetime(day=day, month=month, year=year).date()

data = read_csv('../data/loans.csv')
# Indicate which variables should be dropped.
# These are "unimportant" and have many missing values
dropvars = ['TEASER', 'FIRST_RATE_CAP', 'RATE_CAP', 'RESET_FREQUENCY',
	    'LIFE_RATE_CAP', 'LIFE_FLOOR', 'ARM_INDEX']
# Drop periods. If there are many missing values
# there is no point in keeping them
dropperiods = ['DQ_STATUS_082011',
'DQ_STATUS_072011', 'DQ_STATUS_062011']
df = data.drop(dropvars + dropperiods, axis=1).dropna()	   
indx = np.where(df.columns == 'DQ_STATUS_062012')[0]
periods = df.columns[indx:]
states = df[periods]#.applymap(map_states)  # States data (transitions)
all_covars = df.ix[:, :indx.item()]  # Covariates (Conditional data)

nperiods = len(periods) 
nlns = len(df)

## Macro data
macrovars = read_csv('../data/macro_vars.csv')
macrovars['DATE'] = macrovars.DATA_DATE.map(get_date_macro)
macrovars = macrovars.set_index('DATE')
macrovars = macrovars.fillna(8.3)  # arbitray choosing unemployment for one missing value

hpa = DataFrame(index = range(1, nlns + 1))
unemployment = DataFrame(index = range(1, nlns + 1))

for t in states.columns[:-1]:
    date = get_date(t)
    hpa[str(date)] = macrovars.ix[date, 'HPA']
    unemployment[str(date)] = macrovars.ix[date, 'UNEMPLOY']



# Make date suitable for running
model_vars = ['ORIGINAL_BALANCE', 'ORIGINAL_COUPON', 'ORIGINAL_FICO',
	    'ORIGINAL_LTV']
covars = all_covars[model_vars]
Y = states.ix[:, :-1].values.T.flatten()  # Stack the states
Ylag = states.ix[:, 1:].values.T.flatten()
X = np.kron(np.ones((nperiods-1, 1)), covars.values)
Xhpa = hpa.values.T.flatten()
Xunem = unemployment.values.T.flatten()
Z = np.column_stack((X, Xhpa, Xunem))

df1 = DataFrame(Z, columns=model_vars + ['HPA', 'UNEMPLOY'])
df1['Y'] = Y
df1['Ylag'] = Ylag

given_state = 'F'
indx = np.where(df1.Ylag == given_state)[0]
model_data = df1.ix[indx, :]

states_map = create_states_dict(model_data.Y)
Y = model_data['Y'].map(states_map).values
Z = model_data[model_vars + ['HPA', 'UNEMPLOY']].values

# Normalize
mx = np.max(Z, axis=0)
Z = Z / mx
#Z = Z / len(Y)
print "Only the following states are present in this run: "
print model_data.Y.value_counts()
print "Starting to run the model "
mymodel = main_model.Multinomial(Y, Z, me_type='me')
lhat_new = mymodel.fit()
test_case_covars = np.array([  0.05384615,   0.75862069,   0.81343284,   0.12745098,
       -11.26821937,   0.74509804])
test_case_covars = np.reshape(test_case_covars, (1, len(test_case_covars)))
pprob = mymodel.compute_pprob(newdata=test_case_covars)
pprob = DataFrame(pprob, columns = model_data.Y.value_counts().index)
print "New Lagrange Multipliers \n", lhat_new
print "New probability estimates \n", pprob
print "Normalized entropy is: ", mymodel.norm_entropy()

#lhat = basic.optim_gme(Y, Z)
#P = basic.compute_p(lhat, Z)
#print "Old lagrange multiplisers: ", lhat
#print basic.norm_entropy(P)
