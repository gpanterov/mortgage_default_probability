"""
George Panterov
Prepared for Serigne Diop (Vero Capital Management)

This module contains the Markov class used for estimation of
the Markov Chain model within the maximum entropy framework.
The data contains N loans observed over T periods and each loan
can be in one of J possible states. The matrix of covariates Z
must be NxT-by-K where K is the number of variables.

Dependencies:
The module requires numpy (for linear algebra) and scipy (for optimization)
See http://numpy.scipy.org/

References
[1] Golan A. (2008), "Information and Entropy Econometrics - A review and synthesis"
        Foundations and Trends in Econometrics 2(1-2) 1-145
        http://147.9.1.186/cas/economics/pdf/upload/Golan-Review-Foundations-08.pdf

"""

import numpy as np
from scipy import optimize
from sys import stdout

class Markov(object):
    # Tested. Produces same results as markov_gce.py
    """A class for the estimation of conditional Markov models via Maximum Entropy
    Methods
    -------
    dual_gce: dual problem for the generalized cross entropy
    
    fit: fits the generalized cross entropy
    
    compute_pprob: computes the transition probability matrix
    
    compute_wprob: computes the probabilities over the error support
    
    stack_y_bool: stacks the transition data and transforms it into boolean matrix
    
    norm_entropy: computes the normalized entropy S(p)
    """
    def __init__(self, raw_ydat, nperiods, zdat, 
		    pprior=None, eprior=None, esupport=3):
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
	pprior: array_like
            J-by-J array containing the prior prbabilities on the transition matrix.
            Defaults to uniform priors (each entry in the transition matrix is equally likely)
	eprior: array_like
            Priors for the error
	esupport: integer
            The length of the support vector for the error probabilities
	"""
	# setup the states data (transition data)
	self.raw_ydat =raw_ydat
	self.nperiods = nperiods
	self.ydat = self.stack_y_bool(raw_ydat)
	self.N, self.nstates = np.shape(self.ydat)
	self.nlns = self.N / self.nperiods
	self.Ylhs = self.ydat[self.nlns:, :]  # Ylhs goes from t=2 to t=T
	self.Yrhs = self.ydat[:-self.nlns, :]  # Yrhs goes from t=1 to t=T-1

	# setup the covariates (conditioning data)
	self.zdat = zdat
	self.nvars = np.shape(self.zdat)[1]
	self.Zlhs = self.zdat[self.nlns:, :]
	self.Zrhs = self.zdat[:-self.nlns, :]

	# Noise support
	self.m = esupport
	if esupport == 3: 
	    self.esupport = np.array([-1, 0, 1])
	elif esupport == 5:
	    self.esupport = np.array([-1, -0.5, 0, 0.5, 1])
	else:
	    raise ValueError("invalid length for support vector")
	
	# setup the priors. If None they are uniform
	if pprior == None:
	    pprior = np.ones((self.nstates, self.nstates))  # uniform prior prob
	    self.pprior = pprior / np.reshape(np.sum(pprior, 
					axis=1), (self.nstates, 1))
	if eprior == None:
	    eprior = np.ones((self.N, self.m))  # uniform noise prior
	    eprior = eprior / np.reshape(np.sum(eprior, axis=1), (self.N, 1))
	    self.eprior = np.kron(np.ones((1, self.nstates)), eprior)
	    #self.eprior2 = np.kron(np.ones((self.nstates, 1)), eprior)
	    
    def dual_gce(self, L):
	"""Dual problem for the Generalized
	    Cross Entropy problem
	Parameters
	----------
	L: 1D array of size = nvars * (nstates - 1)
	    Lagrange multipliers
	Returns
	-------
	Value of objective function
	"""
	# Create unit vectors usef for Kron products
	u_n = np.ones((self.nlns * (self.nperiods - 1),))
	u_j = np.ones((self.nstates,))
	u_s = np.ones((self.nvars,))
	Lmat = np.reshape(L, (self.nvars, self.nstates - 1))
	# Add zeros in the first column. This is because of the req. that
	# all probabilities should sum up to one. This means that the first
	# column of lagrange multipliers is all zeros

	Lmat = np.column_stack((np.zeros((self.nvars, 1)), Lmat))
	yl = np.kron(self.Ylhs.flatten(), u_s)
	zl = np.kron(u_j , self.Zlhs).flatten()
	l = np.kron(u_n, Lmat.T.flatten())
	# First term in eq. (7.29?) p. 122 in [1]
	# (Tested)
	A = np.sum(yl * l * zl)  

	# Omega. Second term in eq.(7.29) p.122 in [1]
	# (Tested with loops but without the priors -- shouldn't matter)
	O = np.dot(self.Zrhs, Lmat)  # sum over s (see [1])
	Om_jj = np.exp(np.dot(self.Yrhs.T, O)) # sum over i, t (n = i*t)
	Om_pjj = self.pprior * Om_jj  # element-wise multiplication 
	assert np.shape(Om_pjj) == (self.nstates, self.nstates)
	Om_j = np.log(np.dot(Om_pjj, u_j))  # sum over j
	Om = np.dot(Om_j, u_j)  # Th

	# Phi. Third term in eq.
	# (Tested without priors)
	Iv = np.identity(self.nstates)
	V = np.kron(Iv, np.ones((self.m, 1)))
	eprior_r = self.eprior[:-self.nlns, :]
	V1 = np.exp(np.kron(O, self.esupport))
	V1e = eprior_r * V1 
	phi_ij = np.log(np.dot(V1e, V))
	phi_i = np.dot(phi_ij, np.ones(shape=(self.nstates, 1)))
	phi = np.dot(np.ones(self.nlns * (self.nperiods - 1)), phi_i)

	return A - Om - phi


    def fit(self):
	"""Numerical optimization of the objective fnction for the problem"""
	func = lambda x: -self.dual_gce(x)
	# starting value for the optimization
	# The vector must be 1-D
	
	L0 = np.random.normal(size= self.nvars * (self.nstates - 1))
	lhat = optimize.fmin(func, L0, maxiter=1e3, maxfun=1e3)
	lhat = np.reshape(lhat, (self.nvars, self.nstates - 1))
	self.lhat = np.column_stack((np.zeros((self.nvars, 1)), lhat)) 
	return self.lhat


    def compute_pprob(self, lhat=None):
	"""Compute the probability matrix P
	Paramaters
	----------
	lhat: 2D array nvars-by-nstates
	    Lagrange multipliers (first column should be zero)
	Returns
	-------
	pprob: 2D array nstates-by-nstates
	    Transition probability matrix for the markov problem
	"""
	if lhat == None:
	    lhat = self.lhat
	O = np.dot(self.Zrhs, lhat)       
	Om_kj = self.pprior * np.exp(np.dot(self.Yrhs.T, O)) 
	Om_k = np.dot(Om_kj, np.ones([self.nstates, 1]))
	self.pprob = Om_kj / np.reshape(Om_k, (self.nstates, 1))
	return Om_kj / np.reshape(Om_k, (self.nstates, 1))

    def compute_wprob(self, lhat = None):
	if lhat == None:
	    lhat = self.lhat

	O = np.dot(self.Zrhs, lhat)
	Iv = np.identity(self.nstates)
	V = np.kron(Iv, np.ones((self.m, 1)))
	eprior_r = self.eprior[:-self.nlns, :]
	eprior_r2 = self.eprior2[:-self.nlns * self.nstates, :]
    	#V1 = np.exp(np.kron(O, self.esupport))
	#V1e = eprior_r * V1 	    
	#phi_itj = np.kron(np.dot(V1e, V), np.ones((1, self.m))) # sum over m
	#wprob = V1e / phi_itj
	O2 = np.reshape(O.flatten(), [len(self.Zrhs) * self.nstates, 1]) 
	V1 = np.exp(np.kron(self.esupport, O2))
	V1e = eprior_r2 * V1
	phi_itj = np.dot(V1e, np.ones((self.m, 1)))
	wprob = V1e / np.reshape(phi_itj, (self.nlns * 
		(self.nperiods - 1) * self.nstates, 1))  
	# array is I * (T-1) * J -by- m
	return wprob



    def stack_y_bool(self, Y):
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
	J = len(np.unique(Y))

	#J = max(Y)+1
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
	Pp = self.pprior.flatten()
	Sdenom = np.dot(Pp, np.log(Pp)) 
	#self.norm_ent = -S/(self.nlns * np.log(self.nstates))
	self.norm_ent = S / Sdenom
	return self.norm_ent
