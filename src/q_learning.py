import random 
from matplotlib import pyplot as plt
from collections import defaultdict
import time 

class q_agent:
    mdp=None
    def __init__(self, mdp,t,ep=10000):# and here...
        self.mdp = mdp
        self.q_values = defaultdict(lambda: defaultdict(float))
        self.alpha = 0
        self.gamma = mdp.get_discount_factor()
        self.epsilon = 0.1
        self.vInitUP = []
        self.vInitDOWN = []
        self.vInitRIGHT = []
        self.vInitLEFT = []
        self.states = mdp.get_states()
        self.state_visits = { s: 0 for s in self.states } 
        self.t = t
        self.episode = ep


    def greedy(self, s):
        # greedy function returning the best action to pick in a state s
        actions = self.mdp.get_actions(s)

        if random.random() < self.epsilon:
            return random.choice(actions)
        else:
            return max(actions, key=lambda a: self.q_values[s][a])

    def solve(self):
        # main solving loop
        debut = time.time()
        for episode in range(self.episode):
            s = self.mdp.get_initial_state()
            print("episode",episode)
            while not self.mdp.is_terminal(s):
                self.state_visits[s] += 1 # Add this line
                a = self.greedy(s)
                next_s, r= self.mdp.execute(s, a) 
                delta = self.get_delta(r, self.q_values[s][a], s, next_s)
                self.q_values[s][a] += self.alpha * delta
                s = next_s
                # print("qval",self.q_values[s])
                #mdp.visualise_q_function(q_function)
            self.vInitUP.append(self.q_values[self.mdp.get_initial_state()][self.mdp.UP])
            self.vInitDOWN.append(self.q_values[self.mdp.get_initial_state()][self.mdp.DOWN])
            self.vInitRIGHT.append(self.q_values[self.mdp.get_initial_state()][self.mdp.RIGHT])
            self.vInitLEFT.append(self.q_values[self.mdp.get_initial_state()][self.mdp.LEFT])
        self.t.append(time.time()-debut)
        print("time",time.time()-debut)
            

    def get_delta(self, reward, q_value, state, next_state):
        # calculate the delta for the update
        if self.q_values[next_state]:
            max_q = max(self.q_values[next_state].values())
        else:
            max_q = 0  
        return reward + self.gamma * max_q - q_value

    def state_value(self, state):
        # get the value of a state
        return max(self.q_values[state].values())
        
    def get_value(self,s):
        #return the value of a specific state s according to value function v
        return self.state_visits[s]

    def get_q_value(self,state, action):
        return self.q_values[state][action]

    def get_policy(self, state):
        # get the policy for a state
        print(max(self.q_values[state], key=self.q_values[state].get))
        return max(self.q_values[state], key=self.q_values[state].get)

    def plotVInit (self,v,title) :
        plt.plot(v)
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.title(title)
        plt.show()

    def plot_state_visits(self):
        h = self.mdp.height
        w = self.mdp.width
        plt.figure(figsize=(w, h))
        plt.imshow(self.state_visits, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Count')
        plt.title("Compteur d'exploration pour : epsilon = {} et episodes = 10000".format(self.epsilon))
        plt.show()
        
    def plot_time(self):
        plt.plot(self.t)
        plt.xlabel('quantité cases bloquantes')
        plt.ylabel('Time')
        plt.yscale('lin')  # Set the scale of the y axis to logarithmic
        plt.title('Time DP')
        plt.show()
    
