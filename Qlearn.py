import numpy as np
import matplotlib.pyplot as plt

reward = []

#set up the rewards
rewa = np.genfromtxt("qreward.csv", delimiter=',', dtype=float)
rewa[0][0] = 0

rewaf = np.genfromtxt("qrewardf.csv", delimiter=',', dtype=float)
rewaf[0][0] = 0

#planning horizon
time = 100

Q = np.full(shape=(10,10), fill_value=0, dtype=float)

#Qtable to contain 100 steps for 10 possible states
Qtable = []
for i in range(100):
    if i == 1:
        Q = np.full(shape=(10, 10), fill_value=0, dtype=float)
        # Q[0,1] = 1
    Qtable.append(Q)


#DISCOUNTING FACTOR PARAMETER (constant)
GAMMA = 1

#State is defined by tuple (t, s): t is time, s is state at that time
INITIAL_STATE = (0,9)

#Function to determine available actions at a specific state
def available_actions(state):
    if state[0] < 99:
        current_state_row = rewa[state[1],]
    elif state[0] == 99:
        current_state_row = rewaf[state[1],]
    available_act = np.where(current_state_row > -100)[0]
    return available_act


#Function to choose the next action
def choose_next_act(available_act):
    action = int(np.random.choice(available_act, 1))
    return action


#Function to update the Q table based on the current state, action and gamma parameter
def update(state, action, gamma):
    next_state_time = state[0]+1
    max_i = np.where(Qtable[next_state_time][action,] == np.max(Qtable[next_state_time][action,]))[0]
    if max_i.shape[0] > 1:
        max_i = int(np.random.choice(max_i, size=1))
    else:
        max_i = int(max_i)
    max_val = Qtable[next_state_time][action,max_i]

    if state[0] < 97:
        Qtable[state[0]][state[1], action] = rewa[state[1], action] + GAMMA * max_val
    elif state[0] == 98:
        Qtable[state[0]][state[1], action] = rewaf[state[1], action] + GAMMA * max_val



# TESTING

def take_action(cur_state, Qtable):

    #RETURNS ACTION (AS STATE TO LAND AT THE END OF THE YEAR) and QVALUE of that state(ACTION), and a str saying what to do
    cur_time = cur_state[0]
    cur_s = cur_state[1]

    if cur_s == 9:
        action = np.where(Qtable[cur_time][cur_s, cur_s - 1:] == np.max(Qtable[cur_time][cur_s, cur_s - 1:]))[0]
        # print(action)
        if len(action) > 1:
            action = int(np.random.choice(action, size=1))
        if action == 0:
            return 8, Qtable[cur_time][cur_s, 8], "do-nothing"
        elif action == 1:
            return 9, Qtable[cur_time][cur_s, 9], "do-rehab/recon"
    elif cur_s == 0:
        temp = [Qtable[cur_time][cur_s, 0], Qtable[cur_time][cur_s, 1], Qtable[cur_time][cur_s, 9]]
        action = np.where(temp == np.max(temp))[0]
        if len(action) > 1:
            action = int(np.random.choice(action, size=1))
        if action == 0:
            return 0, Qtable[cur_time][cur_s, 0], "do-nothing"
        if action == 1:
            return 1, Qtable[cur_time][cur_s, 1], "do-rehab"
        if action == 2:
            return 9, Qtable[cur_time][cur_s, 9], "do-recon"
    else:
        temp = [Qtable[cur_time][cur_s, cur_s-1], Qtable[cur_time][cur_s, cur_s+1], Qtable[cur_time][cur_s, 9]]
        action = np.where(temp == np.max(temp))[0]
        if len(action) > 1:
            action = int(np.random.choice(action, size=1))
        if action == 0:
            return cur_s-1, Qtable[cur_time][cur_s, cur_s-1], "do-nothing"
        if action == 1:
            return cur_s+1, Qtable[cur_time][cur_s, cur_s+1], "do-rehab"
        if action == 2:
            return 9, Qtable[cur_time][cur_s, 9], "do-recon"


#Runs a simulation with the current Qtable starting at a given state
def test(cur_state, Qtable):
    score = 0
    while cur_state[0] != 99:
        next_action = take_action(cur_state, Qtable)
        score += next_action[1]
        cur_state = (cur_state[0] + 1, next_action[0])
    return score








INITIAL_STATE = (0,9)

#TRAINING
scores = []
for i in range(100000):
    if(INITIAL_STATE[0] == 99):
        scores.append(test(cur_state=(0,9), Qtable=Qtable))
        INITIAL_STATE = (0,9)
    available_act = available_actions(INITIAL_STATE)
    action = choose_next_act(available_act)
    update(INITIAL_STATE, action, GAMMA)

    if(INITIAL_STATE[0] == 99):
        INITIAL_STATE = (0,9)
    else:
        INITIAL_STATE = (INITIAL_STATE[0]+1,action)




# print(Qtable)
#plot scores
print(scores)
plt.plot(scores)
plt.xlabel("Simulations")
plt.ylabel("Score")
plt.savefig("figure.png")
plt.show()
