"""
AA228 Final Project Code:
"Modeling a Highway Merge as a POMDP Using Dynamic GPS Error"
- This file defines constants for the MDP
"""

module MergeMDPConfig

export output_file
export max_iter, belres, discount_factor
export taus, v1s, v2s, ds
export nA, accelerations, dt, accelSigma
export weightsMat
export penActions, penCollision, rewardGoodPos
export goodPosTol, timeGap, vrelTol, dTol


# Output file name
output_file = "q_merge_20000collisionPen.csv"

# Value iteration parameters
max_iter = 500
belres = 1
discount_factor = 0.99

# Set up state space
taus = [-0.8 -0.78 -0.76 -0.74 -0.72 -0.7 -0.68 -0.66 -0.64 -0.62 -0.6 -0.58 -0.56 -0.54 -0.52 -0.5 -0.4 -0.3 -0.2 -0.1 0 0.1 0.2 0.3 0.4 0.5 0.52 0.54 0.6 0.62 0.64 0.66 0.68 0.7 0.72 0.74 0.76 0.78 0.8 0.82 0.84 0.86 0.88 0.9 0.92 0.94][:]
v1s = [24 26 28 30 32][:]
v2s = [18 20 22 24 26 28 29 30 31 32 33 34 35 36][:]
ds = [0 1 2 3 5 10 15 20 25 30 40 50 60 70 80 90 100 110 120 130][:]

# Set up actions
nA = 5
accelerations = [-0.4 -0.15 0. 0.2 0.35]*9.81


dt = 1 #1 # Make decisions every 0.2 seconds # Assuming keep same acceleration for about a second

accelSigma = 0.5 # About 1 mph

accelWeights1 = [0.15 0.7 0.15] # Weight of off nominal accelerations for highway car
accelWeights2 = [0.05 0.1 0.7 0.1 0.05] # Weight of off nominal accelerations for merging car
weightsMat = zeros(length(accelWeights1),length(accelWeights2))
for i = 1:length(accelWeights1)
    for j = 1:length(accelWeights2)
        weightsMat[i,j] = accelWeights1[i]*accelWeights2[j]
    end
end

# Define rewards
penActions = [-50 -5 0 -5 -50]
# penCollision = -500
penCollision = -20000
rewardGoodPos = 10000

goodPosTol = 0.025
timeGap = 0.6

vrelTol = 1

dTol = 20

end # module MergeMDPConfig
