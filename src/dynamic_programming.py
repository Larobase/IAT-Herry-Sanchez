import matplotlib.pyplot as plt
import time
class dp_agent( ):
    mdp=None 
    
    def __init__(self,mdp,b,t, epsilon=0.000001): #and here...
        self.mdp=mdp
        self.states = mdp.get_states()
        self.v = { s: 0.0 for s in self.states } 
        # tuples = [(3, 7), (9, 1), (0, 6), (7, 2), (8, 4), (5, 9), (2, 0), (6, 3), (4, 8), (1, 5), (2, 7), (8, 1), (7, 4), (4, 9), (3, 0), (9, 3), (0, 8), (5, 5), (6, 7), (7, 1), (4, 6), (9, 2), (0, 4), (3, 9), (8, 0), (1, 3), (2, 8), (5, 1), (4, 7), (9, 1), (0, 6), (3, 2), (8, 4), (1, 9), (2, 0), (5, 3), (6, 8), (7, 5), (4, 1), (9, 7), (0, 1), (3, 6), (8, 2), (1, 4), (2, 9), (5, 0), (6, 3), (7, 8)]
        tuples = b
        for i in tuples:
            self.v[i] = 0
        self.v_bis = { s: 0.0 for s in self.states}
        self.epsilon= epsilon
        self.policy = { s: None for s in self.states }
        self.vInit = []
        self.t = t


    def initPolicy(self):
        for state in self.mdp.get_states():
            if state == self.mdp.TERMINAL:
                continue
            best_action_val = -999
            best_action = None
            for action in self.mdp.get_actions():
                val = 0
                for outcome, p in self.mdp.get_transitions(state, action):
                    r = self.mdp.get_reward(state, action, outcome)
                    val += p * (r + self.mdp.get_discount_factor() * self.get_value(outcome))
                if val > best_action_val:
                    best_action_val = val
                    best_action = action
            self.policy[state] = best_action

    def select_action(self, state):
        return self.policy[state]

    def get_value(self,s):
        #return the value of a specific state s according to value function v
        return self.v[s]
        
    def get_width(self,v,v_bis):
        #return the absolute norm between two value functions
        tab = [ v[s]-v_bis[s] for s in self.states ]
        return max(tab)

    def solve(self):
        debut = time.time()
        # 1. Create an empty list to store the values
        #main solving loop
        i=0
        while i==0 or self.get_width(self.v, self.v_bis) > self.epsilon :
            i=1
            self.v_bis = self.v.copy()
            for s in self.states:    
                self.update(s)
            self.vInit.append(self.v[self.mdp.get_initial_state()])
        self.initPolicy()
        self.t.append(time.time()-debut)
        print("time",time.time()-debut)
        return self.v  
    

    def update(self,s):
        #updates the value of a specific state s
     
        maxi = -99999
        for a in self.mdp.get_actions(s) :
            var = 0
            for trans in self.mdp.get_transitions(s,a):
                s_suivant, proba = trans
                var +=  self.mdp.get_reward(s, a, s_suivant) * proba
                var += self.mdp.get_discount_factor() * proba * self.v_bis[s_suivant]
            maxi = max(maxi, var)
        self.v[s] = maxi

    def plotVInit (self) :
        plt.plot(self.vInit)
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.title('Value of initial state over time')
        plt.show()

    def plot_time(self):
        plt.plot(self.t)
        plt.xlabel('quantité cases bloquantes')
        plt.ylabel('Time')
        plt.title('Time DP')
        plt.show()