# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        
        # Create a dictionary of the current state values in (state, Q-value) pairs
        self.values = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        # Return Q-value of a given state and action
        if ((state, action) in self.values.keys()):
            return self.values[(state, action)]

        # Return 0.0 if we have never seen a state
        return 0.0

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        # Create a list of Q-values for each legal action at a given state 
        qValList = [self.getQValue(state, action) for action in self.getLegalActions(state)]
        
        # Return the maximum Q-value from the list
        if (len(qValList) > 0): 
            return max(qValList)

        # Return 0.0 if there are no legal actions
        return 0.0

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        # Get the maximum Q-value at a given state
        maxQval = self.getValue(state)

        # Create a list of actions with maximum Q-value for each legal action at a given state 
        actionsList = [action for action in self.getLegalActions(state) if (self.getQValue(state, action) == maxQval)]
        
        # Return a random action from the list
        if (len(actionsList) > 0): 
            return random.choice(actionsList)

        # Return None if there are no legal actions
        return None

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        
        # Return None if there are no legal actions
        if (len(legalActions) == 0):
            return action
        
        # Return a random action from the legal actions if the exploration probability is successful
        if (util.flipCoin(self.epsilon)):
            action = random.choice(legalActions)
        # Return the best action at a given state
        else:
            action = self.getPolicy(state)
        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        # Update Q-value based on the Q-learning rule: Q(s, a) ← (1 - α)Q(s, a) + (α)[r + γ*maxQ(s′, a′)]
        updateQval = (1 - self.alpha) * self.getQValue(state, action) + self.alpha * (reward + (self.discount * self.getValue(nextState)))
        self.values[(state, action)] = updateQval

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        # Return the Q-value of dot product between 02 dictionaries of feature weights and vectors
        return self.getWeights() * self.featExtractor.getFeatures(state, action)

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        # Compute the difference based on the formula: (r + γ*maxQ(s′, a′)) − Q(s, a)
        difference = (reward + (self.discount * self.getValue(nextState))) - self.getQValue(state, action)

        # Update the new weight for each feature based on the formula: w ← w + α*difference*f(s, a)
        featuresDict = self.featExtractor.getFeatures(state, action)
        for feature in featuresDict:
            self.weights[feature] = self.weights[feature] + (self.alpha * difference * featuresDict[feature])

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
