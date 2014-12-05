"""
George Panterov
Prepared for Serigne Diop (Vero Capital Management)

This script prepares, cleans and runs the markov chain model.
The script requires numpy and pandas to run.
See http://code.google.com/p/pandas/
    and http://numpy.scipy.org/
The class that implements the Markov Chain model is in markov_main.py
The module datetime 
Due to missingness of observations we drop months 6,7 and 8 from 2011.
These could be modified by chaninging the dropperiods list below.

We also drop several variables that are deemed "irrelevant" but could be
easily incorporated in the model by modifying the dropvars list below.

########################################
Instructions on how to run the model: ##
########################################

1) Make sure all dependencies are installed (pandas and numpy)
and that path to data files is correct as well as the markov_main.py
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

4) define prior probabilities if needed. If not the default priors are uniform
If the user chooses to define prior probabilities for the transition matrix they must
be in a numpy array of dimension nstates-by-nstates

5) Run the script


"""
# organize the data for the markov_main model
from pandas import *
import numpy as np
import datetime
from markov_main import *

# Set the seed number for replication purposes
np.random.seed(12345)

def map_states(val):
    """ Assigns an integer to each state"""
    states_map = {'C':0, '30':1, '60':2, '90':3, '120':4, 'REO':5 ,'F':6}
    return states_map[val]

def get_date(val):
    """Reads a string and transforms in datetime.datetime object
        Must be used with loans.csv file
    """
    val = val.split("_")[2]
    month = int(val[:2])
    year = int(val[2:])
    day = 30
    if month == 2:
        day = 29
    return datetime.datetime(day=day, month=month, year=year).date()

def get_date_macro(val):
    """Transform string to datetime in macro vars datafile
        Can't use get_date() due to the different format of the date in
        the two files.
        Must be used with macrovars.csv
    """
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
# Transform states into integers
states = df[periods].applymap(map_states)  # States data (transitions)
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

for t in states.columns:
    date = get_date(t)
    hpa[str(date)] = macrovars.ix[date, 'HPA']
    unemployment[str(date)] = macrovars.ix[date, 'UNEMPLOY']

# Select variables to be included in the model
model_vars = ['ORIGINAL_BALANCE', 'ORIGINAL_COUPON','ORIGINAL_FICO',
	    'ORIGINAL_LTV']
covars = all_covars[model_vars]

Y = states.values.T.flatten()  # Stack the states into 1D vector

# Transform the covariates in a suitable form
X = np.kron(np.ones((nperiods, 1)), covars.values)
Xhpa = hpa.values.T.flatten()
Xunem = unemployment.values.T.flatten()
Z = np.column_stack((X, Xhpa, Xunem))

# Normalize
# There is no theoretical reason for normalizing the values.
# However, without these normalizations the
# generic optimization algorithms in scipy don't converge
# Perhaps a better optimization software is required!
mx = np.max(Z, axis=0)
Z = Z / mx
Z = Z / len(Y)

# Models:
print "Starting to run the models"
print "%%%%\n" * 5
mymodel = Markov(Y, nperiods, Z)
lhat_new = mymodel.fit()
print "Lagrange multipliers for new(OOP) version:  \n", lhat_new
pprob = mymodel.compute_pprob()
print "Transition probabilities: \n", pprob
print "Normalized entropy ", mymodel.norm_entropy()

