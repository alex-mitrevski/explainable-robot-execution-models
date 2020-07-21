import numpy as np

class RuleLearner(object):
    '''Implements a statistical symbolic learner; assumes discrete predicates.
    More details about this method can be found in the following papers:
    * S. Ekvall and D. Kragic, "Learning task models from multiple human demonstrations," in
      Robot and Human Interactive Communication, 2006. ROMAN 2006. 15th IEEE Int. Symp., pp. 358-363, Sept. 2006
    * N. Abdo et al., "Learning manipulation actions from a few demonstrations," in
      Robotics and Automation (ICRA), 2013 IEEE Int. Conf., pp. 1268-1275, May 2013.
    * R. Toris et al., "Unsupervised learning of multi-hypothesized pick-and-place task templates via crowdsourcing," in
      Robotics and Automation (ICRA), 2015 IEEE Int. Conf., pp. 4504-4510, May 2015.

    Author -- Alex Mitrevski
    '''
    @staticmethod
    def learn_rules(predicate_values, predicate_acceptance_threshold=0.1, debug=False):
        '''Extracts a precondition set from 'predicate_values'.

        Keyword arguments:
        predicate_values -- A 2D 'numpy' array of symbolic relations.
                            Each row of the array should be a separate training example
        acceptance_threshold -- Threshold that controls the acceptance of the
                                potential preconditions (default 0.1)

        Returns:
        precondition_vector -- A binary 'numpy' integer array of length 'data.shape[1]';
                               a value of 1 at a given index indicates that the predicate
                               is a precondition of the respective action
        '''
        precondition_vector = np.zeros(predicate_values.shape[1], dtype=int)
        precondition_values = np.zeros(predicate_values.shape[1], dtype=int)
        for i in range(predicate_values.shape[1]):
            predicate_value_prob = RuleLearner.calculate_prob(predicate_values[:, i])
            predicate_entropy = RuleLearner.entropy(predicate_value_prob)

            if debug:
                print(predicate_entropy)

            if predicate_entropy < predicate_acceptance_threshold:
                precondition_value = RuleLearner.max_prob_value(predicate_value_prob)
                precondition_vector[i] = 1
                precondition_values[i] = precondition_value
        return precondition_vector, precondition_values

    @staticmethod
    def calculate_prob(data):
        '''Calculates the probabilities of the individual values of a given predicate.

        Keyword arguments:
        data -- A one-dimensional 'numpy' array storing the observed values of a given predicate

        Returns:
        prob -- A dictionary in which the keys represent predicate values and
                the values are the probabilities of the predicate values
        '''
        prob = dict()
        values = dict()
        for d in data:
            if d in values:
                values[d] += 1
            else:
                values[d] = 1

        for k, v in values.items():
            prob[k] = v / (len(data) * 1.)

        return prob

    @staticmethod
    def entropy(value_prob):
        '''Calculates the entropy of the input probability distribution.

        Keyword arguments:
        value_prob -- A dictionary in which the keys represent predicate values and
                      the values are the probabilities of the predicate values
        '''
        entropy = -sum([value_prob[v] * np.log2(value_prob[v]) for v in value_prob])
        return entropy

    @staticmethod
    def max_prob_value(value_prob):
        '''Returns the key of the input dictionary with the maximum value.
        Keyword arguments:
        value_prob -- A dictionary in which the keys represent predicate values and
                      the values are the probabilities of the predicate values
        '''
        max_prob = -1.
        max_value = -1
        for v, prob in value_prob.items():
            if prob > max_prob:
                max_prob = prob
                max_value = v
        return max_value

    @staticmethod
    def extract_preconditions(precondition_vector, precondition_values, predicates):
        precondition_idx = np.where(precondition_vector == 1)[0]
        return [predicates[i] + (precondition_values[i],)
                for i in precondition_idx]
