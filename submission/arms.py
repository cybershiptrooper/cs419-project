import numpy as np
import pandas as pd

class bernoulliArms():
	def __init__(self, file):
		f = open(file)
		instances = []
		for instance in f.readlines():
			instances.append(float(instance.rstrip()))
		self.__instances = np.array(instances)
		
		k = len(instances) 
		self.k = k
		self.Pavg = np.zeros(k)
		self.Psum = np.zeros(k)
		self.armpulls = np.zeros(k)
		self.totalPulls = 0; #not essential-save time for np.sum(Psum)

	def pull(self, arm, n=1):
		rewards = np.random.binomial(1, self.__instances[arm], n)
		self.updateArms(arm, rewards)
		return rewards

	def updateArms(self, arm, rewards):
		self.Psum[arm] += np.sum(rewards)
		self.armpulls[arm] += len(rewards)
		self.Pavg[arm] = self.Psum[arm]/self.armpulls[arm]
		self.totalPulls += len(rewards)

	def best(self):
		return np.max(self.__instances)*self.totalPulls

class stockArms():
	def __init__(self, file="fortune500.csv", riskAware = True):
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
		self.riskAware = riskAware

	def pull(self, arm, n=1):
		rewards = self.reward()
		self.t +=1
		reward = np.array([rewards[arm]]) #change this for risk aware
		self.updateArms(rewards, arm)
		return reward
	
	def updateArms(self, rewards, arm):
		if(self.riskAware):
			self.Psum += np.sum(rewards)
			self.armpulls += 1
			self.Pavg = self.Psum/self.armpulls
			self.totalPulls += 1
		else:
			self.Psum[arm] += rewards[arm]
			self.armpulls[arm] += 1
			self.Pavg[arm] = self.Psum[arm]/self.armpulls[arm]
			self.totalPulls += 1
		self.bestPulls += np.max(rewards)

	
	def stock_open_close(self):
		return (
			self.data.filter(regex='-open').loc[self.t].to_numpy(),
			self.data.filter(regex='-close').loc[self.t].to_numpy())

	def reward(self, invest_amount=1):
		prices= self.stock_open_close()
		shares_bought= invest_amount/prices[0]
		selling_price= shares_bought*prices[1]
		return selling_price-invest_amount

	def best(self):
		return self.bestPulls #need to change this