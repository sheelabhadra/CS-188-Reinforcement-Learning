# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections
import numpy as np

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.pis = {}
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        # "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        for state in states:
          self.pis[state] = None

        for it in range(self.iterations):
          Vs = []
          pis = []
          for state in states:
            maxQ, bestAction = float('-inf'), None
            for action in self.mdp.getPossibleActions(state):
              # get the Q(s,a)
              Q = self.getQValue(state, action)
              if Q > maxQ:
                maxQ = Q
                bestAction = action
            Vs.append(maxQ)
            pis.append(bestAction)

          for i, state in enumerate(states):
            if Vs[i] > float('-inf'):
              self.values[state] = Vs[i]
              self.pis[state] = pis[i]

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        # "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        q = 0.0
        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
          q += prob*(self.mdp.getReward(state, action, nextState) + self.discount*self.getValue(nextState))
        return q


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.f

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        # "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        if state == 'TERMINAL_STATE':
          return None
        return self.pis[state]
        

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        # "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        for state in states:
          self.pis[state] = None

        it = 0
        while it < self.iterations:
          for state in states:
            maxQ, bestAction = float('-inf'), None
            for action in self.mdp.getPossibleActions(state):
              # get the Q(s,a)
              Q = self.getQValue(state, action)
              if Q > maxQ:
                maxQ = Q
                bestAction = action
            if maxQ > float('-inf'):
              self.values[state] = maxQ
              self.pis[state] = bestAction
            it += 1
            if it == self.iterations:
              break
     

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        # "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        pq = util.PriorityQueue()
        for state in states:
          self.pis[state] = None

        # Commpute the predecessors for each state here
        ##### TODO #####

        # Step 1
        for state in states:
          maxQ, bestAction = float('-inf'), None
          for action in self.mdp.getPossibleActions(state):
            # get the Q(s,a)
            Q = self.getQValue(state, action)
            if Q > maxQ:
              maxQ = Q
              bestAction = action
          diff = abs(maxQ - self.values[state])
          pq.push(state, -diff)

        # Step 2
        for it in range(self.iterations):
          if pq.isEmpty():
            break
          
          state = pq.pop()
          if state != 'TERMINAL_STATE':
            self.values[state] += -priority

          # iterate over the predecessors of the state
          ##### TODO #####


          for state in states:
            maxQ, bestAction = float('-inf'), None
            for action in self.mdp.getPossibleActions(state):
              # get the Q(s,a)
              Q = self.getQValue(state, action)
              if Q > maxQ:
                maxQ = Q
                bestAction = action
            if maxQ > float('-inf'):
              self.values[state] = maxQ
              self.pis[state] = bestAction
            it += 1
            if it == self.iterations:
              break
     

