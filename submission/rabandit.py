#usr/bin/env/python3
import sys
from algorithms import sampler
import numpy as np

class bandit(sampler):
	"""
	input is a list containing the following:
	instance- (relative path of file)
	algo- string
	seed- int 0-49
	eps- [0,1]
	horizon- int

	outputs regret
	"""
	def __init__(self, arg):
		super().__init__(arg[:-1])
		self.hz = int(arg[4])

	def run(self, seeded = True):
		REW = 0.0
		if(seeded): np.random.seed(self.seed)
		for i in range(self.hz):
			rewards = self.sample()
			REW += np.sum(rewards)
		#regret = reward - expected_optimal 
		REG = self.arms.best() - REW
		return REG#, self.armpulls

def main():
	if(len(sys.argv) != 11):
		print("Please enter valid arguments")
		sys.exit()
	for arg in sys.argv[2::2]:
		print(arg, end = ", ")
	run=0
	for i in range(50):
		bandit_instance = bandit(sys.argv[2::2])
		bandit_instance.seed = i
		run += (bandit_instance.run())
	print(run/50)


if __name__ == '__main__':
	main()
