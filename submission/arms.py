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

	def optimalArm(self):
		return np.max(self.__instances)

class stockArms():
	def __init__(self, file="fortune500.csv"):
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

		#stock vars
		self.stocks=['aapl','amzn','cost','gs','jpm','msft','tgt','wfc','wmt']
		self.data = pd.read_csv(file, sep=',')
		self.t = 0


	def pull(self, arm, n=1):
		rewards = self.reward()
		self.t +=1
		rewards = rewards[arm] #change this for risk aware
		self.updateArms(arm, rewards)
		return rewards
	
	def updateArms(self, arm, rewards):
		self.Psum[arm] += np.sum(rewards)
		self.armpulls[arm] += len(rewards)
		self.Pavg[arm] = self.Psum[arm]/self.armpulls[arm]
		self.totalPulls += len(rewards)

	
	def stock_open_close(self):
		print((self.data.filter('-open')))
		return (self.data.filter('-open').loc[self.t],self.data.filter('-close').loc[self.t])

	def reward(self, invest_amount=1):
		prices= self.stock_open_close()
		shares_bought= invest_amount/prices[0]
		selling_price= shares_bought*prices[1]
		return selling_price-invest_amount

	def optimalArm(self):
		return np.max(self.__instances)