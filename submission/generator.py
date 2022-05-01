from bandit import bandit
import sys
import time 

methods = ['epsilon-greedy', 'ucb', 'kl-ucb', 'thompson-sampling', 'thompson-sampling-with-hint']
epsilon = 0.02 #0.999, 0.001
horizons = [102400]#[100, 400, 1600, 6400, 25600, 102400]
instances = ["../instances/i-1.txt","../instances/i-2.txt", "../instances/i-3.txt",]

def generate_graph(todo):
	start = time.time()
	#jeeeezzz
	inst = 0
	for i in instances:
		print(i)
		for algo in todo:
			#f = open("../data-for-graph/"+"i-"+i[-5]+"/"+str(algo)+".txt","a")
			#g = open("../data-for-graph/"+str(algo)+".txt","a")
			#g.write(i+"\n")
			print(algo)
			for hz in horizons:
				regret = 0.0
				for seed in range(50):
					args = [i, algo, seed, epsilon, hz]
					bandit_instance = bandit(args)
					REG = bandit_instance.run()
					regret += REG
					#write file
					#f.write(i+' '+algo+' '+str(hz)+' '+str(seed)+' '+str(REG)+'\n')
					#print progress
					sys.stdout.write("\rseed: %i, time elapsed %.2f" % (seed ,(time.time()-start)))
					sys.stdout.flush()
				regret /= 50.0
				print("\nhorizon:", hz, "Regret:", regret)
				#g.write(i+"  ----"+str(regret)+"\n")
			#f.close()
			#g.close()
	print("time taken:",time.time()-start)

if __name__ == '__main__':
	print(instances, horizons)
	generate_graph([methods[2]]) #the final boss
	