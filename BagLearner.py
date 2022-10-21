import random

import numpy as np


class BagLearner(object):
    """
    :param verbose: If “verbose” is True, your code can print out information for debugging.
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.
    :type verbose: bool
    """

    def __init__(self, learner, kwargs, bags, boost, verbose):
        """
        Constructor method
        """
        self.learner = learner
        self.kwargs = kwargs
        self.bags = bags
        self.boost = boost
        self.verbose = verbose

    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "dcheung35"

    def add_evidence(self, data_x, data_y):
        """
        Add training data to learner

        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray
        :param data_y: The value we are attempting to predict given the X data
        :type data_y: numpy.ndarray
        """
        self.learners = []    # list to simulate all bags
        for i in range(0,self.bags):
            self.learners.append(self.learner(**self.kwargs))   # create learner object
            x_random = np.empty((0, data_x.shape[1]))      # select random data w/ replacement
            y_random = np.empty(0)
            for j in range(0,data_x.shape[0]):
                random_counter = random.randint(0,data_x.shape[0]-1)
                x_random = np.append(x_random,np.array([data_x[random_counter,:]]),axis=0)
                y_random = np.append(y_random, np.array([data_y[random_counter]]), axis=0)
            self.learners[i].add_evidence(x_random,y_random)             # input random data to learner

    def query(self, points):
        """
        Estimate a set of test points given the model we built.

        :param points: A numpy array with each row corresponding to a specific query.
        :type points: numpy.ndarray
        :return: The predicted result of the input data according to the trained model
        :rtype: numpy.ndarray
        """
        learner_results = np.empty((0,points.shape[0]))                 # get predictions from each learner
        for i in range(0,self.bags):
            learner_results = np.append(learner_results, np.array([self.learners[i].query(points)]), axis=0)
        final_results = np.empty(0)                                    # compile results
        for i in range(0,points.shape[0]):
            final_results = np.append(final_results,np.mean(learner_results[:,i]))
        return final_results