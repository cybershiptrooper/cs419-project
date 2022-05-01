import numpy as np

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

	def pull(self, arm, n_arms = 0, n=1, improve = False):
		rewards = np.random.binomial(1, self.__instances[arm], n)
		if improve:
			for i in range(n_arms):
				if not i==arm:
					rew = np.random.binomial(1, self.__instances[i], n)
					self.updateArms(i, rew)
		self.updateArms(arm, rewards)
		return rewards

	def updateArms(self, arm, rewards):
		self.Psum[arm] += np.sum(rewards)
		self.armpulls[arm] += len(rewards)
		self.Pavg[arm] = self.Psum[arm]/self.armpulls[arm]
		self.totalPulls += len(rewards)
		
	def optimalArm(self):
		return np.max(self.__instances)
