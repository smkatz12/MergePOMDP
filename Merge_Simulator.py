import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation
import pandas as pd


###################################FUNCTIONS#########################################
#####################################################################################

def getNextStates(XREL, X1, V1, X2, V2, ACTION):
    # 5 HZ
    X1.append(X1[-1] + V1[-1]*0.2)  # velocity * 0.2 seconds
    V1.append(V1[-1])               # highway car has no acceleration

    # below assumes velocity only changes after 0.2 seconds
    X2.append(X2[-1] + V2[-1]*0.2 + 0.5*ACTION*0.2*0.2)  # x = x_0 + velocity*0.2sec + 0.5*accel*0.2sec^2
    V2.append(V2[-1] + ACTION*0.2)                       # v = v_0 + accel*0.2sec

    XREL.append(X1[-1] - X2[-1])

def getMergeAction():

    # use scipy.interpolate.griddata() to interpolate Q table
    # With interpolated Q, iterate through possible actions and associated rewards to find best

    decel_2 = -3.92
    decel_1 = -1.47
    noChange = 0
    accel_1 = 1.96
    accel_2 = 3.43

    action = accel_2
    return action


# time gap
def getTau(XREL, V2, tau):
    tau.append(XREL[-1]/V2[-1])
    return tau[-1]




'''
###########################IMPORT GPS DATA################################
##########################################################################
file = "open sky driving"
df = pd.read_csv(file + ".csv")

#print(df.head())
#print("\n" + str(columns))
#print(df)
gpsError = {}
for i in range(len(df)):
    # dictionary[seconds] = [longitudinal error (m), lateral error (m)]
    gpsError[str(df.at[i, 'seconds from start'])] = [df.at[i, 'pl_e'], df.at[i, 'pl_n']]
'''



###########################Highway Vehicle Parameters################################
#####################################################################################
#Highway Car Start State
startX1 = 0         # starting x position
startY1 = 11        # starting y position
startlong1 = -4.86  # starting longitudinal GPS error (m)
startlat1 = 1.87    # starting lateral GPS error (m)
startV1 = 30        # starting velocity (m/s)
startD1 = 120       # starting distance from end (m)



###########################Merge Vehicle Parameters################################
###################################################################################
#Merge Car Start State
startX2 = 0         # starting x position
startY2 = 7         # starting y position
startlong2 = -4.86  # starting longitudinal GPS error (m)
startlat2 = 1.87    # starting lateral GPS error (m)
startV2 = 36        # starting velocity (m/s)
startD2 = 120       # starting distance from end (m)





##############################SIMIULATION###################################
############################################################################
x1 = [startX1]
v1 = [startV1]

x2 = [startX2]
v2 = [startV2]
D2 = startD1

xRel = [x1[0] - x2[0]]
tGap = [xRel[-1]/v2[-1]]

# simulate until reach end distance
while D2 > 0:
    action = getMergeAction()                    # Will get action from Q table interpolation. Simplified currently.
    getNextStates(xRel, x1, v1, x2, v2, action)  # updates states
    getTau(xRel, v2, tGap)
    D2 = 120 - x2[-1]                            # 120m - most recent x position of merge car




###########################PLOTTING################################
###################################################################
fig = plt.figure()
#plt.axis('equal')
ax = fig.add_subplot(111)
ax.set_xlim(-10, 140)
ax.set_ylim(0, 30)
ax.set_facecolor('silver')

x = np.linspace(-10, 140)
y = np.linspace(-10, 80)
z = np.linspace(80, 140)
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
    patch2.set_xy([x2[i], startY2])  # location

    #Grass
    patch3.set_width(200)
    patch3.set_height(25)
    patch3.set_xy([-10, 17.51])

    patch4.set_width(200)
    patch4.set_height(7)
    patch4.set_xy([-10, -2])

    #Text
    label1.set_text("Time Gap: " + str(round(tGap[i], 2)) + " seconds")
    label2.set_text("Relative Distance: " + str(round(xRel[i], 2)) + " meters")

    return patch2, patch1, patch3, patch4, label1, label2


anim = animation.FuncAnimation(fig, animate, init_func=init, frames=min(len(x1), len(x2)), interval=200, blit=True)

fig.set_size_inches(12, 6, forward=True)
plt.show()


