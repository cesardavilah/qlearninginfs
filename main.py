import numpy as np

ALPHA = 1
def read_input(matrices):
    '''
    :param matrices: List of file names for the matrices
    :return: Returns the matrices as an numpy 2d array. Matrices are flipped to make state numbers coincide with indexes.
    Matrices of actions are read matrix[from,to] while cost matrix is read cost[action, state]
    '''
    cost = np.genfromtxt(matrices[0], delimiter=',')
    cost = np.fliplr(cost)
    donot = np.genfromtxt(matrices[1], delimiter=',')
    donot = np.fliplr(np.flipud(donot))
    rehab = np.genfromtxt(matrices[2], delimiter=',')
    rehab = np.fliplr(np.flipud(rehab))
    recon = np.genfromtxt(matrices[3], delimiter=',')
    recon = np.fliplr(np.flipud(recon))


    naction, states = cost.shape
    return cost, donot, rehab, recon, naction, states

def ret_min(t, state):
    '''
    :param t: Table containing all costs for all steps in time t
    :param state: Current state being evaluated
    :return: Returns tuple of (cost, action) of minimum action(mini) cost in time table 't' for state 'state'
    '''
    mini = t[state][3]
    return t[state][int(mini)], mini

def donothing(donot, steps, in_step, for_state):
    '''
    :param donot: Matrix of donothing transition probabilities
    :param steps: List of steps in planning horizon
    :param in_step: Current step in planning horizon
    :param for_state: Current state evaluated in 'in_step' of planning horizon
    :return: Returns the calculated cost of action 0(do nothing) in step 'in_step' for state 'for_state'.
    '''
    #Case for state 0 where only one transition probability exists
    if (for_state == 0):
        #Transition probability donot[from, to]
        transition = donot[for_state, for_state]
        tminus = steps[in_step-1]
        c = ALPHA*transition*ret_min(tminus, for_state)[0]
        return c
    #Case for state != 0 where two transition probabilities exists
    else:
        #transition probabilities donot[from, to]
        transition = donot[for_state, for_state], donot[for_state, for_state-1]
        #table t of previous step
        tminus = steps[in_step-1]
        min_for_state = ret_min(tminus, for_state)[0], ret_min(tminus, for_state-1)[0]
        c = ALPHA*((transition[0]*min_for_state[0]) + (transition[1]*min_for_state[1]))
        return c

def rehabilitation(rehab, costs, steps, in_step, for_state):
    '''
    :param rehab: Matrix of rehabilitation transition probabilities
    :param costs: Matrix of costs for action 0, 1 and 2
    :param steps: List of steps in planning horizon
    :param in_step: Current step in planning horizon
    :param for_state: Current state evaluated in 'in_step' of planning horizon
    :return: Returns the calculated cost of action 1(rehabilitation) in step 'in_step' for state 'for_state'.
    '''
    #Cost of action 1 in state 'for_state'
    cofa = cost[1, for_state]
    #Case for state 9 where transition probabilities includes no further improvement to better state, only current and worse
    if (for_state == 9):
        #Transition probabilities for rehab[from, to] [9,9] [9,8]
        transition = rehab[for_state, for_state], rehab[for_state, for_state-1]
        tminus = steps[in_step-1]
        min_for_state = ret_min(tminus, for_state)[0], ret_min(tminus, for_state-1)[0]
        c =  cofa + ALPHA*((transition[0]*min_for_state[0]) + (transition[1]*min_for_state[1]))
        return c
    #For all other cases where state != 9
    else:
        transition = rehab[for_state, for_state], rehab[for_state, for_state+1]
        tminus = steps[in_step-1]
        min_for_state = ret_min(tminus, for_state)[0], ret_min(tminus, for_state+1)[0]
        c = cofa + ALPHA*((transition[0]*min_for_state[0]) + (transition[1]*min_for_state[1]))
        return c

def reconstruction(recon, cost, steps, in_step, for_state):
    '''
    :param recon: Matrix of reconstruction transition probabilities
    :param costs: Matrix of costs for action 0, 1 and 2
    :param steps: List of steps in planning horizon
    :param in_step: Current step in planning horizon
    :param for_state: Current state evaluated in 'in_step' of planning horizon
    :return: Returns the calculated cost of action 2(reconstruction) in step 'in_step' for state 'for_state'.
    '''
    # Cost of action 2 in state 'for_state'
    cofa = cost[2, for_state]
    transition = recon[for_state, 9]
    tminus = steps[in_step-1]
    min_for_state = ret_min(tminus, 9)[0]
    c = cofa + ALPHA*(transition*min_for_state)
    return c

#MAIN
if __name__ == '__main__':
    cost, donot, rehab, recon, naction, states = read_input(["cost.csv", "0donothing.csv", "1rehab.csv", "2reconstruction.csv"])
    plan = 100

    steps = []

    for i in range(plan):
        #Each time or step is represented as a 10(states) by 4(3 actions costs + 1 slot for action taken (action with min cost)) 2d array
        t = np.full(shape=(10, 4), fill_value=-1, dtype=float)

        # First step "100" assignment of SV value
        if i==0:
            for j, state in enumerate(t):
                state[0] = 100
                state[3] = 0
                if (j>2):
                    state[0] = 0
            steps.append(t)
            continue
        else:
            # All other steps 99 assignment through calculation using matrices
            for j, state in enumerate(t):
                for k, action in enumerate(state):
                    #Action 0 in state j in step i
                    if (k==0):
                        state[0] = donothing(donot, steps, in_step=i, for_state=j)
                    # Action 1 in state j in step i
                    if (k==1):
                        state[1] = rehabilitation(rehab, cost, steps, in_step=i, for_state=j)
                    #Action 2 in state j in step i
                    if (k==2):
                        state[2] = reconstruction(recon, cost, steps, in_step=i, for_state=j)
                    #Selection of min cost to be put in slot 3 for step i and state j
                    #If reconstruction is minimum cost alongside other action, reconstruction is prefered.
                    if (k==3):
                        if (state[2] == min(state[:3])): state[3] = 2
                        else:
                            state[3] = state[:3].argmin()
            steps.append(t)


#Prints all time steps
for t, time in enumerate(steps):
    print("At time: "+ str(100-t) +" costs are")
    print(steps[t])






