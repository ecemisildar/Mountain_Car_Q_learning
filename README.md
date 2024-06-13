### ARTIFICIAL INTELLIGENCE FOR ROBOTICS I: REINFORCEMENT LEARNING PROJECT
### Mountain Car Q-Learning
  
### Aim of the Project: 
Using Q-learning algorithm, teaching the agent how to the reach goal point in a mountain. 
Agent can takes position and veloicty values of the car from the environment, and tries to reach the goal point in a reinforcement learning based model.
Training consists of 10000 iterations, the learning and the exploration rates are decreased in each iteration in order to converge faster
The environment is continues so it needs to be discretized, in order to discretization process Kbins Discretization method is used.
Results are given not only by plotting reward and time relationship but also video clips for randomly chosen iterations using gym package. 
In order to obtain video results, a folder called MountainCar should be created in the users Pc.
It can be seen that the agent can reach the goal point by updating Q-table.
 
### Q-Learning:

Q-Learning is an off-policy value-based method that uses a TD approach to train its action-value function.
Q-Learning basically consists of 4 steps:
1. Initialize the Q-table
   We need to initialize the Q-table for each state-action pair. Most of the time, we initialize with values of 0.
2. Choose action using epsilon-greedy strategy
   Epsilon greedy strategy is a policy that handles the exploration/exploitation trade-off.
   The idea is that we define the initial epsilon ɛ = 1.0:
   With probability 1 — ɛ : we do exploitation (aka our agent selects the action with the highest state-action pair value).
   With probability ɛ: we do exploration (trying random action).
   At the beginning of the training, the probability of doing exploration will be huge since ɛ is very high, so most of the time, agent explores. But as the training goes on, and consequently our Q-table gets better and better in its estimations, we progressively reduce the epsilon value since we will need less and less exploration and more exploitation.
3. Perform action At, gets reward Rt+1 and next state St+1
4. Update Q(St, At)
   in TD Learning, we update our policy or value function (depending on the RL method we choose) after one step of the interaction.
 
