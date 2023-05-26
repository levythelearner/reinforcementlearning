# analysis.py
# -----------
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


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

def question2():
    answerDiscount = 0.9
    # We tried to reduce the noise from 0.2 to 0.02 and to 0.002, it worked at 0.002 (based on the autograder),
        # so we set the noise to 0.002 to prevent the agent from frequently ending up in unintended states
    answerNoise = 0.002
    return answerDiscount, answerNoise

def question3a():
    # The agent prefers the close exit, which means that it considers the immediate rewards instead of future rewards;
        # therefore, we lowered the discount parameter while leaving the others as 0.0
    # After trying to gradually decrease the discount from 0.9 to 0.3, and it worked (based on the autograder);
        # thus, we set the discount to 0.1 to make sure that the agent factors the short-term rewards
    answerDiscount = 0.1
    answerNoise = 0.0
    answerLivingReward = 0.0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3b():
    # We kept the discount parameter the same as the agent still prefers the close exit;
        # however, since it wants to avoid the cliff by traveling the longer path, 
        # it considers a higher chance of stepping into an unintended state, 
        # and a higher reward for taking more steps
    # So we tried to increase the noise and living reward from 0.0 to 0.1 and it worked
    answerDiscount = 0.1
    answerNoise = 0.1
    answerLivingReward = 0.1
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3c():
    # Opposite of question 3a, this agent prefers the distant exit, 
        # which means that it considers the future rewards instead of immediate rewards; therefore,
        # we tried to keep the default discount of 0.9 while leaving the other parameters as 0.0, and it worked
    answerDiscount = 0.9
    answerNoise = 0.0
    answerLivingReward = 0.0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3d():
    # Same question 3b, which also travels the longer path, 
        # we kept the discount parameter the same as the previous question 3c, 
        # whereas increase the noise and living reward from 0.0 to 0.1,
        # and it worked
    answerDiscount = 0.9
    answerNoise = 0.1
    answerLivingReward = 0.1
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3e():
    # We tried to set the maximum value of 1.0 for the discount and living reward parameter while leaving the noise as 0.0, 
        # and it worked because the agent will tend to avoid the terminal state to maximize the future rewards, 
        # as well as it is encouraged to keep going and take more steps with a reward for each step taken
    answerDiscount = 1.0
    answerNoise = 0.0
    answerLivingReward = 1.0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question6():
    # Epsilon is the exploration probability of selecting a random action instead of 
        # the action with highest expected reward based on the agent's current knowledge
    # We set Epsilon to 0.0 because with a limitation of 50 iterations, we prefer an optimal action instead of a random one
    # We set Learning Rate to 1.0 because a higher Learning Rate will lead to larger updates and faster learning;
        # thus, with the limitation of 50 episodes, we want a faster learning process
    # Our trials failed, so we believe that it is impossible for the optimal policy to be learned within 50 iterations
    answerEpsilon = 0.0
    answerLearningRate = 1.0
    # return answerEpsilon, answerLearningRate
    return 'NOT POSSIBLE'

if __name__ == '__main__':
    print('Answers to analysis questions:')
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print('  Question %s:\t%s' % (q, str(response)))
