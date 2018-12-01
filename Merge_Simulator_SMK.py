import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation
import pandas as pd
import csv
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import norm
import math


###################################FUNCTIONS#########################################
#####################################################################################

def getNextStates(XREL, X1, V1, X2, V2, ACTION, D2, D1, M, A1, A2):
    '''
    if ACTION == D2:
        act = np.random.normal(ACTION, 0.55)
    elif ACTION == D1:
        act = np.random.normal(ACTION, 0.55)
    elif ACTION == M:
        act = np.random.normal(ACTION, 0.55)
    elif ACTION == A1:
        act = np.random.normal(ACTION, 0.55)
    elif ACTION == A2:
        act = np.random.normal(ACTION, 0.55)
    '''

    if ACTION == D2:
        act = np.random.normal(ACTION, 0.1)
    elif ACTION == D1:
        act = np.random.normal(ACTION, 0.1)
    elif ACTION == M:
        act = np.random.normal(ACTION, 0.1)
    elif ACTION == A1:
        act = np.random.normal(ACTION, 0.1)
    elif ACTION == A2:
        act = np.random.normal(ACTION, 0.1)


    # 5 HZ
    X1_next = X1 + V1*0.2  # velocity * 0.2 seconds
    V1_next = V1 + np.random.normal(0, 0.1)            # highway car has no acceleration

    # below assumes velocity only changes after 0.2 seconds
    X2_next = X2 + V2*0.2 + 0.5*act*0.2*0.2  # x = x_0 + velocity*0.2sec + 0.5*accel*0.2sec^2
    V2_next = V2 + act*0.2                       # v = v_0 + accel*0.2sec

    # XREL.append(X1[-1] - length1 - X2[-1])      # valid for all cases when length1 = length2
    XREL_next = X1 - X2      # valid for all cases when length1 = length2
    return X1_next, V1_next, X2_next, V2_next, XREL_next

def getMergeAction():

    # use scipy.interpolate.griddata() to interpolate Q table
    # With interpolated Q, iterate through possible actions and associated rewards to find best

    decel_2 = -3.92
    decel_1 = -1.47
    noChange = 0
    accel_1 = 1.96
    accel_2 = 3.43

    action = accel_1
    return action


def approx_gaussian(sigma1,sigma2):
    # First create discrete Gaussian for d
    min_val = -3*sigma2
    max_val = 3*sigma2
    vals = np.linspace(min_val,max_val,11,endpoint=True)
    d_gaussian = np.zeros(len(vals)-1)
    for i in range(len(d_gaussian)):
        d_gaussian[i] = norm.cdf(vals[i+1],loc=0,scale=sigma2) - norm.cdf(vals[i],loc=0,scale=sigma2)
    d_gaussian = d_gaussian/np.sum(d_gaussian)
    # Next, create one for tau
    sigma_tau = math.sqrt(sigma1**2+sigma2**2)
    min_val = -3*sigma_tau
    max_val = 3*sigma_tau
    vals = np.linspace(min_val,max_val,11,endpoint=True)
    tau_gaussian = np.zeros(len(vals)-1)
    for i in range(len(tau_gaussian)):
        tau_gaussian[i] = norm.cdf(vals[i+1],loc=0,scale=sigma_tau) - norm.cdf(vals[i],loc=0,scale=sigma_tau)
    tau_gaussian = tau_gaussian/np.sum(tau_gaussian)
    # Combine into one matrix
    g_approx = np.zeros((len(d_gaussian),len(tau_gaussian)))
    for i in range(len(d_gaussian)):
        for j in range(len(tau_gaussian)):
            g_approx[i][j] = d_gaussian[i]*tau_gaussian[j]
    return g_approx


def getMergeActionInterpBelief(state, interpolator_dict, nA, approx_g, sigma_d, sigma_tau):
    best_action = 0
    best_val = -10000000

    dm = state[3]
    taum = state[0]

    min_d = dm - 3*sigma_d
    max_d = dm + 3*sigma_d
    min_tau = taum - 3*sigma_tau
    max_tau = taum + 3*sigma_tau

    d = np.linspace(min_d,max_d,10)
    tau = np.linspace(min_tau,max_tau,10)

    for i in range(nA):
        q_sum = 0
        for j in range(len(d)):
            for k in range(len(tau)):
                curr_state = list(state)
                curr_state[3] = d[j]
                curr_state[0] = tau[k]
                curr_state = tuple(curr_state)
                interp_a = interpolator_dict[i+1] # Interpolators start at 1
                curr_val = interp_a(curr_state)
                q_sum += curr_val*approx_g[j][k]
        if q_sum > best_val:
            best_val = q_sum
            best_action = i+1

    d2 = -3.92
    d1 = -1.47
    m = 0
    a1 = 1.96
    a2 = 3.43

    # Select acceleration based on based action
    if best_action == 1:
        return d2
    elif best_action == 2:
        return d1
    elif best_action == 3:
        return m
    elif best_action == 4:
        return a1
    else:
        return a2


def getMergeActionInterp(state, interpolator_dict, nA):

    # use scipy.interpolate.griddata() to interpolate Q table
    # With interpolated Q, iterate through possible actions and associated rewards to find best

    best_action = 0
    best_val = -10000000

    for i in range(nA):
        interp_a = interpolator_dict[i+1] # Interpolators start at 1
        curr_val = interp_a(state)
        if curr_val > best_val:
            best_val = curr_val
            best_action = i+1

    d2 = -3.92
    d1 = -1.47
    m = 0
    a1 = 1.96
    a2 = 3.43

    # Select acceleration based on based action
    if best_action == 1:
        return d2
    elif best_action == 2:
        return d1
    elif best_action == 3:
        return m
    elif best_action == 4:
        return a1
    else:
        return a2

# time gap
def getTau(XREL, V1, V2):
    if XREL < 0: # Highway in back
        tau = XREL/V1
    else:
        tau = XREL/V2
    return tau

def simulate(startX1,startV1,startX2,startV2,startD,interpolator_dict,approx_g,sigma1,sigma2):
    nA = 5

    x1 = [startX1]
    x2 = [startX2]
    v1 = [startV1]
    v2 = [startV2]
    d = [startD]

    xRel = [x1[0] - x2[0]]
    tau = [getTau(xRel[0],v1[0],v2[0])]

    d2 = -3.92
    d1 = -1.47
    m = 0
    a1 = 1.96
    a2 = 3.43

    sigma_d = sigma2

    while d[-1] > 0:
        state = (tau[-1],v1[-1],v2[-1],d[-1])
        if tau[-1] < 0: # Highway car in back
            sigma_tau = math.sqrt(sigma1**2+sigma2**2)/v1[-1]
        else:
            sigma_tau = math.sqrt(sigma1**2+sigma2**2)/v2[-1]
        action = getMergeActionInterpBelief(state, interpolator_dict, nA,approx_g,sigma_d,sigma_tau)
        # print(action)
        X1_next, V1_next, X2_next, V2_next, XREL_next = getNextStates(xRel[-1],x1[-1],v1[-1],x2[-1],v2[-1],action,d2, d1, m, a1, a2)
        x1.append(X1_next)
        v1.append(V1_next)
        x2.append(X2_next)
        v2.append(V2_next)
        xRel.append(XREL_next)
        tau.append(getTau(XREL_next,V1_next,V2_next))
        d.append(d[-1] - (x2[-1]-x2[-2]))
    # print(tau[-1])

    return x1,x2,v1,v2,xRel,tau,d

def simulate_more(x1,x2,v1,v2,tau,xRel,d):
    x1more = x1
    x2more = x2
    v1more = v1
    v2more = v2
    taumore = tau
    xRelmore = xRel
    dmore = d

    d2 = -3.92
    d1 = -1.47
    m = 0
    a1 = 1.96
    a2 = 3.43

    for i in range(15):
        action = 0
        X1_next, V1_next, X2_next, V2_next, XREL_next = getNextStates(xRel[-1],x1[-1],v1[-1],x2[-1],v2[-1],action,d2, d1, m, a1, a2)
        x1more.append(X1_next)
        v1more.append(V1_next)
        x2more.append(X2_next)
        v2more.append(V2_next)
        xRelmore.append(XREL_next)
        taumore.append(getTau(XREL_next,V1_next,V2_next))
        dmore.append(dmore[-1] - (x2more[-1]-x2more[-2]))

    return x1more,x2more,v1more,v2more,xRelmore,taumore,dmore



###########################PLOTTING################################
###################################################################
def animate(x1,x2,tau,xRel,dmore):
    startX1 = x1[0]
    startX2 = x2[0]
    startY1 = 11
    startY2 = 7
    startD = 114.5

    startlong1 = -4.86  # starting longitudinal GPS error (m)
    startlat1 = 1.87

    startlong2 = -4.86  # starting longitudinal GPS error (m)
    startlat2 = 1.87    # starting lateral GPS error (m)

    fig = plt.figure()
    #plt.axis('equal')
    ax = fig.add_subplot(111)
    ax.set_xlim(-10, 200)
    ax.set_ylim(0, 30)
    ax.set_facecolor('silver')

    x = np.linspace(-10, 200)
    y = np.linspace(-10, 120)
    z = np.linspace(120, 200)
    laneWidth = 3.7  # meters

    plt.plot(x, x - x + 10 + 2*laneWidth, linestyle='-', color='gold', linewidth=2)
    plt.plot(x, x - x + 10 + laneWidth, linestyle='--', color='1')
    plt.plot(y, y - y + 10, linestyle='-', color='1')
    plt.plot(z, z - z + 10, linestyle='--', color='1')
    plt.plot(x, x - x + 10 - laneWidth, linestyle='-', color='1', linewidth=2)




    ###########################ANIMATION################################
    ###################################################################
    patch1 = patches.Rectangle((0, 0), 0, 0, fc='b')  # Highway Car
    patch2 = patches.Rectangle((0, 0), 0, 0, fc='r')  # Merge Car
    patch3 = patches.Rectangle((0, 0), 0, 0, fc='yellowgreen')  # grass
    patch4 = patches.Rectangle((0, 0), 0, 0, fc='yellowgreen')  # grass

    label1 = ax.text(110, 28, "", ha='center', va='center', fontsize=15)
    label2 = ax.text(110, 25, "", ha='center', va='center', fontsize=15)


    def init():
        ax.add_patch(patch1)
        ax.add_patch(patch2)
        ax.add_patch(patch3)
        ax.add_patch(patch4)
        return patch1, patch2, patch3, patch4

    def animate(i):
        patch1.set_width(startlong1)  # length of car
        patch1.set_height(startlat1)  # width of car
        patch1.set_xy([x1[i], startY1])  # location

        patch2.set_width(startlong2)  # length of car
        patch2.set_height(startlat2)  # width of car
        if dmore[i] > 0:
            patch2.set_xy([x2[i], startY2])  # location
        else:
            patch2.set_xy([x2[i], startY1])

        #Grass
        patch3.set_width(250)
        patch3.set_height(25)
        patch3.set_xy([-10, 17.51])

        patch4.set_width(250)
        patch4.set_height(7)
        patch4.set_xy([-10, -2])

        #Text
        label1.set_text("Time Gap: " + str(round(tau[i], 2)) + " seconds")
        label2.set_text("Relative Distance: " + str(round(xRel[i], 2)) + " meters")


        return patch2, patch1, patch3, patch4, label1, label2


    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=min(len(x1), len(x2)), interval=250, blit=True)

    fig.set_size_inches(12, 6, forward=True)
    plt.show()


###########################Import Q Table################################
#########################################################################
# Define state dimensions
ntau = 46
nv1 = 5
nv2 = 14
nd = 20
# Define number of actions
nA = 5
# Initialize table of zeros
import_dims = (ntau*nv1*nv2*nd,5)
q = np.zeros(import_dims)
with open('q_merge_10000collisionPen.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter = ',')
    r_idx = -1
    for row in csv_reader:
        r_idx += 1
        for i in range(nA):
            q[r_idx,i] = row[i]


###########################Set Things Up For Interpolation################################
##########################################################################################
# Reshape to 5D Array - (tau,v1,v2,d,action)
# q_dims = (ntau,nv1,nv2,nd,nA)
q_dims = (nA,ntau,nv1,nv2,nd) # Switched nA to the beginning
# q_r = q.reshape(q_dims)
# Reshape is giving me issues, so I am just going to write it myself
# Initialize to zeros
q_r = np.zeros(q_dims)
# Fill in q_r
q_idx = -1
for i in range(nd):
    for j in range(nv2):
        for k in range(nv1):
            for l in range(ntau):
                q_idx += 1
                for m in range(nA):
                    q_r[m,l,k,j,i] = q[q_idx,m]

# Define variable discretizations
taus = np.array([-0.8, -0.78, -0.76, -0.74, -0.72, -0.7, -0.68, -0.66, -0.64, -0.62, -0.6, -0.58, -0.56, -0.54, -0.52, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.52, 0.54, 0.6, 0.62, 0.64, 0.66, 0.68, 0.7, 0.72, 0.74, 0.76, 0.78, 0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.92, 0.94])
v1s = np.array([24, 26, 28, 30, 32])
v2s = np.array([18, 20, 22, 24, 26, 28, 29, 30, 31, 32, 33, 34, 35, 36])
ds = np.array([0, 1, 2, 3, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130])
# Define Interpolators
interp_d2 = RegularGridInterpolator((taus,v1s,v2s,ds),q_r[0,:,:,:,:], method='linear', bounds_error=False)
interp_d1 = RegularGridInterpolator((taus,v1s,v2s,ds),q_r[1,:,:,:,:], method='linear', bounds_error=False)
interp_m = RegularGridInterpolator((taus,v1s,v2s,ds),q_r[2,:,:,:,:], method='linear', bounds_error=False)
interp_a1 = RegularGridInterpolator((taus,v1s,v2s,ds),q_r[3,:,:,:,:], method='linear', bounds_error=False)
interp_a2 = RegularGridInterpolator((taus,v1s,v2s,ds),q_r[4,:,:,:,:], method='linear', bounds_error=False)
test_pt = np.array([-0.746,30.1,22.1,36])

# Make a dictionary of interpolators to pass into the getMergeActionInterp function
interpolator_dict = {1:interp_d2, 2:interp_d1, 3:interp_m, 4:interp_a1, 5:interp_a2}

# Get gaussian approximation
approx_g = approx_gaussian(100,1)

###########################Highway Vehicle Parameters################################
#####################################################################################
#Highway Car Start State
startX1 = 0         # starting x position
startY1 = 11        # starting y position
startlong1 = -4.86  # starting longitudinal GPS error (m)
startlat1 = 1.87    # starting lateral GPS error (m)
startV1 = 29 # 25 # 27        # starting velocity (m/s)
startD1 = 135       # starting distance from end (m)
length1 = 4.86      # length of vehicle 1



###########################Merge Vehicle Parameters################################
###################################################################################
#Merge Car Start State
startX2 = 20.5         # starting x position
startY2 = 7         # starting y position
startlong2 = -4.86  # starting longitudinal GPS error (m)
startlat2 = 1.87    # starting lateral GPS error (m)
startV2 = 18        # starting velocity (m/s)
startD2 = 114.5       # starting distance from end (m)
length2 = 4.86      # length of vehicle 2


###########################IMPORT GPS DATA################################
##########################################################################
file = "open_sky_driving"
df = pd.read_csv(file + ".csv")
#print(df.head())
#print("\n" + str(columns))
#print(df)
gpsError = {}
for i in range(len(df)):
    # dictionary[seconds] = [longitudinal error (m), lateral error (m)]
    gpsError[df.at[i, 'seconds from start']] = df.at[i, 'pl_e'] # , df.at[i, 'pl_n']]

sigma1 = gpsError[60*50]/5.5
sigma2 = gpsError[60*1]/5.5

'''
times = np.array([1,2,4,6,8,10,20,30,50])
# times = np.array([1,2,4])
mean_taus = np.zeros(len(times))
for i in range(len(times)):
    taus = np.zeros(10)
    sigma1 = gpsError[60*50]/5.5
    sigma2 = gpsError[60*times[i]]/5.5
    for j in range(10):
        x1,x2,v1,v2,xRel,tau,d = simulate(startX1,startV1,startX2,startV2,startD2,interpolator_dict,approx_g,sigma1,sigma2)
        taus[j] = tau[-1]
        print(j)
    mean_taus[i] = np.mean(taus)

plt.title("Final Merge " + '$\\tau$' + " vs Amount of Time Since Last Reset")
plt.ylabel("Final Merge " + '$\\tau$' + " (s)")
plt.xlabel("Time Since Last Reset (min)")
plt.plot(times, mean_taus, 'b-')
plt.show()

plt.title("Final Merge Time Gap vs Amount of Time Since Last Reset")
plt.ylabel("Final Merge Time Gap (s)")
plt.xlabel("Time Since Last Reset (min)")
plt.plot(times, mean_taus, '-bo')
plt.show()
'''


x1,x2,v1,v2,xRel,tau,d = simulate(startX1,startV1,startX2,startV2,startD2,interpolator_dict,approx_g,sigma1,sigma2)
x1more,x2more,v1more,v2more,xRelmore,taumore,dmore = simulate_more(x1,x2,v1,v2,tau,xRel,d)


animate(x1more,x2more,taumore,xRelmore,dmore)




# Plot velocity vs time for both vehicles
time = []
for i in range(min(len(x1), len(x2))):
    time.append(i/5.)
plt.title("Speed of Vehicles vs Time")
plt.ylabel("Speed (m/s)")
plt.xlabel("Time (s)")
plt.plot(time, v1, 'b-', label="Highway Vehicle")
plt.plot(time, v2, 'r-', label="Merge Vehicle")
plt.legend()
plt.show()

plt.title("Time Gap vs Time")
plt.ylabel("Time Gap (s)")
plt.xlabel("Time (s)")
plt.plot(time, tau, 'b-')
plt.show()

plt.title("Headway vs Time")
plt.ylabel("Headway (m)")
plt.xlabel("Time (s)")
plt.plot(time, xRel, 'b-')
plt.show()

plt.title("Position of Vehicles vs Time")
plt.ylabel("Position (m)")
plt.xlabel("Time (s)")
plt.plot(time, x1, 'b-', label="Highway Vehicle")
plt.plot(time, x2, 'r-', label="Merge Vehicle")
plt.legend()
plt.show()

