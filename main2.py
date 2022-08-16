
import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("MIT_class/Machine_Learning_with_Python/project3/netflix/netflix_incomplete.txt")

K = [1, 12]    # Clusters to try
seeds = [0, 1, 2, 3, 4]     # Seeds to try

# Costs for diff. seeds
costs_kMeans = [0, 0, 0, 0, 0]
costs_EM = [0, 0, 0, 0, 0]

# Best seed for cluster based on lowest costs 
best_seed_kMeans = [0, 0, 0, 0]
best_seed_EM = [0, 0, 0, 0]

# Mixtures for best seeds
mixtures_kMeans = [0, 0, 0, 0, 0]
mixtures_EM = [0, 0, 0, 0, 0]

# Posterior probs. for best seeds
posts_kMeans = [0, 0, 0, 0, 0]
posts_EM = [0, 0, 0, 0, 0]


print("=============== K is only 1 and 12 ===============")
for k in range(len(K)):
    for i in range(len(seeds)):
        
        # # Run kMeans
        # mixtures_kMeans[i], posts_kMeans[i], costs_kMeans[i] = \
        # kmeans.run(X, *common.init(X, K[k], seeds[i]))
        
        # Run Naive EM
        mixtures_EM[i], posts_EM[i], costs_EM[i] = \
        em.run(X, *common.init(X, K[k], seeds[i]))
    
    # Print lowest cost
    print("=============== Clusters:", K[k], "======================")
    # print("Lowest cost using kMeans is:", np.min(costs_kMeans))
    print("Highest log likelihood using EM is:", np.max(costs_EM))


