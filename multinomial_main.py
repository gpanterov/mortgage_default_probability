"""
George Panterov
Prepared for Serigne Diop (Vero Capital Management)

This module contains the Multinomial class used for estimation of
the multinomial model within the maximum entropy framework.
The data contains N loans observed over T periods and each loan
can be in one of J possible states. The matrix of covariates Z
must be NxT-by-K where K is the number of variables.

Dependencies:
The module requires numpy (for linear algebra) and scipy (for optimization)
See http://numpy.scipy.org/

References
----------
[1] Golan A., Judge G., Perloff, J. (1996)" A maximum entropy approach to recovering
    information from multinomial respnse data." Journal of American Statistical
    Association 91(434), 841-853 
"""

import numpy as np
from scipy import optimize

class Multinomial(object):
    # Tested. Produces same results as multinomial.py omptim_me
    """A class for the estimation of conditional Markov models via Maximum Entropy
    Methods
    -------
    dual_gme: dual problem for the generalized maximum entropy
    
    dual_me: dual problem for the maximum entropy
    
    fit: fits the model specified in me_type
    
    compute_pprob: computes the probabilities for each state
        
    stack_y_bool: stacks the transition data and transforms it into boolean matrix
    
    norm_entropy: computes the normalized entropy S(p)

    """
    def __init__(self, raw_ydat, zdat, me_type='me', nstates=None, esupport=3):
		    
	""" Initialize the Markov class
	Parameters
	----------
	raw_ydat: array_like
            NxT-by-1 vector of raw transition data. Each state should be
            represented by an integer starting from 0.
	nperdios: integer
            Number of periods
	zdat: array_like
            2D array NxT-by-K containing the covariates
	me_type: string
            "me": Maximum entropy (default)
            "gme": Generalized maximum entropy    
	esupport: integer
            The length of the support vector for the error probabilities

	"""
	# setup the states data (transition data)
	self.raw_ydat =raw_ydat
	self.ydat = self.stack_y_bool(raw_ydat, nstates)
	self.nlns, self.nstates = np.shape(self.ydat)
	self.zdat = zdat
	self.nvars = np.shape(self.zdat)[1]

    	# Noise support
	self.m = esupport
	if esupport == 3: 
	    self.esupport = np.array([-1, 0, 1])
	elif esupport == 5:
	    self.esupport = np.array([-1, -0.5, 0, 0.5, 1])
	else:
	    raise ValueError("invalid length for support vector")
	self.me_type = me_type

    def dual_me(self, L):
	"""Dual problem for the Max Entropy problem
	Parameters
	----------
	L: 1D array of size = nvars * (nstates - 1)
	    Lagrange multipliers
	Returns
	-------
	Value of objective function
	"""
	# Create unit vectors usef for Kron products
	u_n = np.ones((self.nlns, ))
	u_j = np.ones((self.nstates, ))
	u_k = np.ones((self.nvars, ))
	
	# Add zeros in the first column. This is because of the req. that
	# all probabilities should sum up to one. This means that the first
	# column of lagrange multipliers is all zeros
	Lmat = np.reshape(L, (self.nvars, self.nstates - 1))
	Lmat = np.column_stack((np.zeros((self.nvars, 1)), Lmat))

	y = np.kron(self.ydat.flatten(), u_k)
	z = np.kron(u_j, self.zdat).flatten()
	l = np.kron(u_n, Lmat.T.flatten())
	A = -np.sum(y * l * z)
	O1 = np.exp(-np.dot(self.zdat, Lmat))
	Om_i = np.dot(O1, np.ones(shape=(self.nstates, 1)))
	Om = np.dot(np.ones(self.nlns), np.log(Om_i))	

	return -A + Om

    def dual_gme(self, L):
	"""Dual problem for the Generalized
	    Max Entropy problem
	Parameters
	----------
	L: 1D array of size = nvars * (nstates - 1)
	    Lagrange multipliers
	Returns
	-------
	Value of objective function
	"""

	Lmat = np.reshape(L, (self.nvars, self.nstates - 1))
	Lmat = np.column_stack((np.zeros((self.nvars, 1)), Lmat))


	Iv = np.identity(self.nstates)
	V = np.kron(Iv, np.ones((self.m, 1)))

	O = np.dot(self.zdat, Lmat)
	V1 = np.exp(-np.kron(O, self.esupport))
	phi_ij = np.log(np.dot(V1, V))
	phi_i = np.dot(phi_ij, np.ones(shape=(self.nstates, 1)))
	phi = np.dot(np.ones(self.nlns), phi_i)

	return self.dual_me(L) + phi
    
    def fit(self):
	"""Numerical optimization of the objective fnction for the problem
	Paramaters
	----------
	"""
	if self.me_type == 'me':
	    func = lambda x: self.dual_me(x)
	elif self.me_type =='gme':
	    func = lambda x: self.dual_gme(x)
	else:
	    raise ValueError('type should be me or gme')
	# starting value for the optimization
	# The vector must be 1-D
	
	L0 = np.random.normal(size= self.nvars * (self.nstates - 1))
	lhat = optimize.fmin(func, L0, maxiter=1e3, maxfun=1e3)
	lhat = np.reshape(lhat, (self.nvars, self.nstates - 1))
	self.lhat = np.column_stack((np.zeros((self.nvars, 1)), lhat)) 
	return self.lhat

    def compute_pprob(self, lhat=None, newdata=None):
	"""Compute the probabilities for each state
	Paramaters
	----------
	lhat: 2D array nvars-by-nstates
	    Lagrange multipliers (first column should be zero)
	newdata: array_like
            covariates for which the probabilities should be estimated
	Returns
	-------
	pprob: 2D array nstates-by-nstates
	    Transition probability matrix for the markov problem
	"""
        if newdata == None:
            newdata = self.zdat
	if lhat == None:
	    lhat = self.lhat	
	Num = np.exp(-np.dot(newdata, lhat))
	Om = np.dot(Num, np.ones(shape=(self.nstates, 1)))
	pprob = Num / Om
	self.pprob = pprob
	return pprob

    def stack_y_bool(self, Y, nstates=None):
	"""
	Transform the data Y into a boolean matrix

	Parameters
	----------
	Y: N-by-1 vector of transition data
	    Each element of Y indicates the state the loan is in.
	    States are represented as integers with the first state being 0.

	Returns
	-------
	Ya: N-by-J Boolean matrix (2D array) 
	    where J is the number of possible outcomes fo Y
	"""
	if nstates == None:
	    #J = len(np.unique(Y))
	    J = max(Y)+1
	else:
	    J = nstates
	N = np.shape(Y)[0]
	Ya = np.zeros(( N, J))
	for i in range(N):
	    y_mat = np.zeros((1, J))
	    y_mat[0,Y[i]] = 1
	    Ya[i, :] = y_mat
	return Ya

    def norm_entropy(self, P=None):
        """ Computes the normalized entropy S(p) """
	if P == None:
	    #P = self.pprob
	    P = self.compute_pprob()
	P = P.flatten()
	indx = np.where(np.isnan(P)==False)
	P = P[indx]
	S = np.dot(P, np.log(P))
	self.norm_ent = -S/(self.nlns * np.log(self.nstates))
	return self.norm_ent
