import numpy as np
import matplotlib.pyplot as plt

reward = []

#set up the rewards
rewa = np.genfromtxt("qreward.csv", delimiter=',', dtype=float)
rewa[0][0] = 0

rewaf = np.genfromtxt("qrewardf.csv", delimiter=',', dtype=float)
rewaf[0][0] = 0

time = 100

# Q = []
Q = np.full(shape=(10,10), fill_value=0, dtype=float)

#Qtable to contain 100 steps for 10 states
Qtable = []
for i in range(100):
    if i == 1:
        Q = np.full(shape=(10, 10), fill_value=0, dtype=float)
        Q[0,1] = 1
    Qtable.append(Q)
# print(len(Qtable))



GAMMA = .999

#State is defined by tuple (t, s): t is time, s is state at that time
INITIAL_STATE = (0,9)

def available_actions(state):
    if state[0] < 99:
        current_state_row = rewa[state[1],]
        # print(current_state_row)
    elif state[0] == 99:
        current_state_row = rewaf[state[1],]
    available_act = np.where(current_state_row > -100)[0]
    return available_act



def choose_next_act(available_act):
    action = int(np.random.choice(available_act, 1))
    return action



#UP TO THIS POINT IS CORRECT

# #VARS FOR TESTING
# state = (0,9)
# action = 9#DELETE THISS AFTER TESTING
# next_state_time = state[0]+1
# print(next_state_time)
# #ENDOF VARS FOR TESTING

# max_i = np.where(Qtable[next_state_time][action,] == np.max(Qtable[next_state_time][action,]))[0]
# print(max_i)
#
# print(max_i.shape[0] > 1)
# if max_i.shape[0] > 1:
#     max_i = int(np.random.choice(max_i, size=1))
#     print(max_i)
# else:
#     max_i = int(max_i)
#     print(max_i)
# max_val = Qtable[next_state_time][action,max_i]
#
# # print(Qtable[state[0]][state[1], action])
# print(max_val)
# if state[0] < 99:
#     Qtable[state[0]][state[1], action] = rewa[state[1], action] + GAMMA * max_val
# elif state[0] == 100:
#     Qtable[state[0]][state[1], action] = rewaf[state[1], action] + GAMMA * max_val
#
# print(Qtable[state[0]])
# print(Qtable[next_state_time])


def update(state, action, gamma):
    next_state_time = state[0]+1
    # print(next_state_time)
    max_i = np.where(Qtable[next_state_time][action,] == np.max(Qtable[next_state_time][action,]))[0]
    # print(max_i)

    # print(max_i.shape[0] > 1)
    if max_i.shape[0] > 1:
        max_i = int(np.random.choice(max_i, size=1))
        # print(max_i)
    else:
        max_i = int(max_i)
        # print(max_i)
    max_val = Qtable[next_state_time][action,max_i]

    # print(Qtable[state[0]][state[1], action])
    # print(max_val)
    if state[0] < 97:
        Qtable[state[0]][state[1], action] = rewa[state[1], action] + GAMMA * max_val
    elif state[0] == 98:
        Qtable[state[0]][state[1], action] = rewaf[state[1], action] + GAMMA * max_val

# available_act = available_actions(INITIAL_STATE)
# print(available_act)
# action = choose_next_act(available_act)
# print(action)
# update(INITIAL_STATE, action, GAMMA)
# print(INITIAL_STATE, action, GAMMA)
# print(Qtable[0])
# print(Qtable[1])

# TESTING

def take_action(cur_state, Qtable):

    #RETURNS ACTION (AS STATE) and QVALUE of that state(ACTION), and a str saying what to do
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


def test(cur_state, Qtable):
    score = 0
    while cur_state[0] != 99:
        next_action = take_action(cur_state, Qtable)
        # print(cur_state)
        # print(Qtable[cur_state[0]][cur_state[1],])
        # print(next_action)
        score += next_action[1]
        # print("Score: " + str(score))
        cur_state = (cur_state[0] + 1, next_action[0])
    return score








INITIAL_STATE = (0,9)
# print(INITIAL_STATE)
#TRAINING
scores = []
for i in range(10000000):
    if(INITIAL_STATE[0] == 99):
        scores.append(test(cur_state=(0,9), Qtable=Qtable))
        INITIAL_STATE = (0,9)
    available_act = available_actions(INITIAL_STATE)
    action = choose_next_act(available_act)
    update(INITIAL_STATE, action, GAMMA)
    # print(INITIAL_STATE)
    # print(Qtable[INITIAL_STATE[0]])
    if(INITIAL_STATE[0] == 99):
        INITIAL_STATE = (0,9)
    else:
        INITIAL_STATE = (INITIAL_STATE[0]+1,action)




# print(Qtable)
print(scores)
plt.plot(scores)
plt.show()




    # if cur_s == 9:
    #     # pos actions 8, 9 or 0,1 which is 0 do nothing 1 do recon
    #     print(Qtable[cur_time][cur_s, ])
    #     action = np.where(Qtable[cur_time][cur_s, cur_s-1:] == np.max(Qtable[cur_time][cur_s, cur_s-1:]))[0]
    #     print(action)
    #     if len(action) > 1:
    #         action = int(np.random.choice(action, size=1))
    #     if action == 0:
    #         return 8, Qtable[cur_time][cur_s, 8], "do-nothing"
    #     elif action == 1:
    #         return 9, Qtable[cur_time][cur_s, 9], "do-rehab/recon"
    # elif cur_s == 0:
    #     #pos actions 0, 1 (state) or 0, 1 which is 0 do nothing 1 do recon
    #     pos_actions = [0, 1]
    #     print(Qtable[cur_time][cur_s, ])
    #     action = np.where(Qtable[cur_time][cur_s, :cur_s+1] == np.max(Qtable[cur_time][cur_s, :cur_s+1]))[0]
    #     print(action)
    #     if len(action) > 1:
    #         action = int(np.random.choice(action, size=1))
    #     if action == 0:
    #         return 0, Qtable[cur_time][cur_s, 0], "do-nothing"
    #     elif action == 1:
    #         return 1, Qtable[cur_time][cur_s, 1], "do-rehab"
    # else:
    #     pos_actions = [cur_s-1, cur_s, cur_s+1]
    #
    # # for a in pos_actions:


#TESTING




#
# cur_state = (0, 9)
# score = 0
#
#
# while cur_state[0] != 99:
#     next_action = take_action(cur_state, Qtable)
#     print(cur_state)
#     print(Qtable[cur_state[0]][cur_state[1],])
#     print(next_action)
#     score += next_action[1]
#     print("Score: "+ str(score))
#     cur_state = (cur_state[0]+1, next_action[0])

    # cur_state[0] += 1
    # cur_state[1] = next_action[0]


    #
    # print(cur_state)
    # print(Qtable[cur_state[0]][cur_state[1],])
    # print(take_action(cur_state, Qtable))
# print(next_step_i)




# max_val = Q[next_state_time][action, max_i]


# def update(cur_state, action, gamma):
#     next_state_time = cur_state[0]+1
#     max_index = np.where(Qtable[next_state_time][action,])

#
# for q in range(100):
#     lq = np.full(shape=(900, 3), fill_value=0, dtype=float)
#     Q.append(lq)

