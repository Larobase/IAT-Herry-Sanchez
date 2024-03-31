from gridworld import *
from dynamic_programming import *
from q_learning import *

b=[(3, 7), (9, 1), (0, 6), (7, 2), (8, 4), (5, 9), (2, 0), (6, 3), (4, 8), (1, 5), (2, 7), (8, 1),  (7, 4), (4, 9), (3, 0), (9, 3), (0, 8), (5, 5), (6, 7), (7, 1), (4, 6), (9, 2), (0, 4), (3, 9), (8, 0), (1, 3), (2, 8), (5, 1), (4, 7), (9, 1), (0, 6), (3, 2), (8, 4), (1, 9), (2, 0), (5, 3), (6, 8), (7, 5), (4, 1), (9, 7), (0, 1), (3, 6), (8, 2), (1, 4), (2, 9), (5, 0), (6, 3), (7, 8)]
# b=[(1,1)]
t = []
for i in range(len(b)):
    blocks = b[:i]
    mdp = GridWorld (blocked_states =blocks)
    agent = dp_agent(mdp=mdp,b=blocks,t=t)
    agent.solve()
agent.plot_time()


print (" states :" , mdp.get_states () )
print (" terminal states :" , mdp.get_goal_states() )
print (" actions :" , mdp.get_actions() )
print ( mdp.get_transitions( mdp.get_initial_state() , mdp.UP ) )
# agent = dp_agent(mdp)
# agent = q_agent(mdp)
# agent.solve()
# agent.plot_state_visits()
# mdp.visualise_q_function(agent)
# agent.plotVInit(agent.vInitUP, "UP" )
# agent.plotVInit(agent.vInitDOWN, "DOWN" )
# agent.plotVInit(agent.vInitRIGHT, "RIGHT")
# agent.plotVInit(agent.vInitLEFT, "LEFT")
# mdp.visualise_value_function_as_heatmap(agent)
# mdp.visualise_policy(agent)


def policy_custom ( state ) :
    return agent.select_action(state)

while (1) :
    state = mdp.get_initial_state()
    new_state , _ = mdp.execute(state,policy_custom(state))
    mdp.initial_state = new_state
    mdp.visualise ()