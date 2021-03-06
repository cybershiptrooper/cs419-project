import numpy as np
from arms import *

class sampler():
	"""Various Bandit Sampling algorithms"""
	def __init__(self, arg, armtype = "Stocks"):
		if(arg[-1]=="False" or arg[-1]==False): arg[-1] = True
		else: arg[-1] = False
		if armtype == "Stocks":
			self.arms = stockArms(allArms=arg[-1])
		else:
			self.arms = bernoulliArms(arg[0], arg[-1])
		self.algo = arg[1]
		self.seed = int(arg[2])
		self.eps = float(arg[3])
		self.gamma = 0.05
		self.alpha = 0.96
		self.prev_val = np.zeros(self.arms.k)+0.0

	def sample(self):
		'''choose algo'''
		if(self.algo == "epsilon-greedy"):return self.epsilonGreedy()
		if(self.algo == "ucb"):return self.ucb()
		if(self.algo == "kl-ucb"):return self.klUCB()
		if(self.algo == "thompson-sampling"):return self.thompson()
		if(self.algo == "cvar"): return self.ucbCVaR()
		else: raise Exception("Please select correct algorithm")

	#utils
	global argmax, kl, isclose
	def argmax(mat):
		optimal_arms = np.where(mat==np.max(mat))[0]
		argmax = np.random.choice(len(optimal_arms))
		arm = optimal_arms[argmax]
		return arm

	def kl(p,q):
		if(p == 0):
			return (1-p)*np.log((1-p)/(1-q))
		if p==1:
			return p*np.log(p/q)

		return p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q))

	def isclose(a, b, precision=1e-06):
		return (abs(a-b) <= precision) #and (b>a)

	def cvar(self, prev, n, alpha):
		beta = 1-alpha
		X = self.arms.get_result(self.arms.t)
		c_new = []
		for i in range(len(n)):
			X_new = self.arms.get_result(int(n[i]*beta))
			X_prev = self.arms.get_result(int((n[i]-1)*beta))
			c_new.append((prev[i])*(n[i]-1)/n[i] + (alpha/beta * X_prev[i])*(n[i]-1)/n[i] - alpha/beta * X_new[i] + X[i]/(n[i]*beta))
		return  np.array(c_new)

	#algos
	def roundRobin(self):
		'''Pull each arm one time'''
		for arm in range(self.arms.k):
			if(self.arms.armpulls[arm] == 0):
				return self.arms.pull(arm)
		return None

	def epsilonGreedy(self):
		s = np.random.uniform()
		if(s < self.eps):
			#choose random arm
			arm = np.random.choice(self.arms.k)
		else:
			#choose a random arm with max arms.Pavg
			arm = argmax(self.arms.Pavg)

		#return seeded reward
		return self.arms.pull(arm)
			
	def ucb(self):
		#do round robin if nobody sampled
		reward = self.roundRobin()
		if(not (reward is None)):
			return reward
		#calculata uta, ucb
		pulls = self.arms.armpulls * 1.0
		uta = np.ones_like(pulls)
		uta[:] *= ( ((2 * np.log(self.arms.totalPulls))) / pulls[:] )**0.5
		ucb = self.arms.Pavg + uta

		#sample max ucb
		arm = argmax(ucb)
		#return seeded reward
		return self.arms.pull(arm)

	def klUCB(self, c = 3, precision = 1e-06):
		#round robin
		reward = self.roundRobin()
		if(not (reward is None)): return reward

		klucb = np.zeros(self.arms.k)
		t = self.arms.totalPulls
		logt_term = np.log(t) + c*np.log(np.log(t))

		#make klucb matrix
		for i in range(self.arms.k):
			p = self.arms.Pavg[i]
			RHS = logt_term / self.arms.armpulls[i]

			#boundary
			if(p == 1 or RHS < 0):
				klucb[i] = p
				continue

			#binary search
			lb, ub = p, 1.0
			q = (ub + p)/2.0
			LHS = kl(p,q)
			#loop until within precision
			while(not isclose(LHS , RHS, precision)):
				if(LHS > RHS): ub = q
				elif(LHS < RHS):lb = q
				q = (ub + lb)/2.0
				LHS = kl(p,q)

			#update klucb
			klucb[i] = q
		#get arm to pull
		arm = argmax(klucb)
		
		#return reward
		return self.arms.pull(arm)

	def thompson(self):
		#create beta choice vector
		s = self.Psum; #Sum of rewards = number of success for bernoulli
		f = self.arms.armpulls - s
		beta = np.random.beta(s+1, f+1)
		#choose maximal beta as arm and pull
		arm = np.argmax(beta)
		return self.arms.pull(arm)

	def ucbCVaR(self):
		#do round robin if nobody sampled
		reward = self.roundRobin()
		if(not (reward is None)):
			return reward
		#calculata uta, ucb
		pulls = self.arms.armpulls * 1.0
		uta = np.ones_like(pulls)
		uta[:] *= ( ((2 * np.log(self.arms.totalPulls))) / pulls[:] )**0.5
		new_cvar = self.cvar(self.prev_val, pulls, self.alpha)
		self.prev_val = new_cvar
		new_cvar *= (self.gamma)
		ucb = self.arms.Pavg + uta + new_cvar
		self.prev_val = new_cvar
		#sample max ucb
		arm = argmax(ucb)
		#return seeded reward
		return self.arms.pull(arm)
