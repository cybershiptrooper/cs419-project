from tkinter import N
import numpy as np
import pandas as pd

class bernoulliArms():
	def __init__(self, file, allArms=False):
		f = open(file)
		instances = []
		for instance in f.readlines():
			instances.append(float(instance.rstrip()))
		self.__instances = np.array(instances)
		self.allArms = allArms
		
		k = len(instances) 
		self.k = k
		self.Pavg = np.zeros(k)
		self.Psum = np.zeros(k)
		self.armpulls = np.zeros(k)
		self.totalPulls = 0; #not essential-save time for np.sum(Psum)

	def pull(self, arm):
		rewards = np.random.binomial(1, self.__instances)
		self.updateArms(arm, rewards)
		return rewards[arm]

	def updateArms(self, arm, rewards):
		if(self.allArms):
			self.Psum+= rewards
			self.armpulls += 1
			self.Pavg = self.Psum/self.armpulls
			self.totalPulls += len(rewards)
		else:
			self.Psum[arm] += rewards[arm]
			self.armpulls[arm] += 1
			self.Pavg[arm] = self.Psum[arm]/self.armpulls[arm]
			self.totalPulls += 1

	def best(self):
		best = np.max(self.__instances)*self.totalPulls
		if(self.allArms): best /= self.k
		return best

class stockArms():
	def __init__(self, file="fortune500.csv", allArms = True):
		#stock vars
		self.stocks=['aapl','amzn','cost','gs','jpm','msft','tgt','wfc','wmt']
		self.data = pd.read_csv(file, sep=',')
		self.t = 0
	
		k = len(self.stocks)
		self.k = k
		self.Pavg = np.zeros(k)
		self.Psum = np.zeros(k)
		self.armpulls = np.zeros(k)
		self.totalPulls = 0; #not essential-save time for np.sum(Psum)
		self.bestPulls = 0
		self.allArms = allArms

	def pull(self, arm):
		rewards = self.reward()
		self.t +=1
		reward = np.array([rewards[arm]]) #change this for risk aware
		self.updateArms(rewards, arm)
		return reward
	
	def updateArms(self, rewards, arm):
		if(self.allArms):
			self.Psum += (rewards)
			self.armpulls += 1
			self.Pavg = self.Psum/self.armpulls
			self.totalPulls += 1
		else:
			self.Psum[arm] += rewards[arm]
			self.armpulls[arm] += 1
			self.Pavg[arm] = self.Psum[arm]/self.armpulls[arm]
			self.totalPulls += 1
		self.bestPulls += np.max(rewards)

	
	def stock_open_close(self, t = None):
		if(t is None): t = self.t
		return (
			self.data.filter(regex='-open').loc[t].to_numpy(),
			self.data.filter(regex='-close').loc[t].to_numpy())

	def reward(self, invest_amount=10, t = None):
		if(t is None): t = self.t
		prices= self.stock_open_close()
		shares_bought= invest_amount/prices[0]
		selling_price= shares_bought*prices[1]
		return selling_price-invest_amount

	def best(self):
		return self.bestPulls
	
	def get_result(self, t):
		return self.reward(t = t)