import abc

import numpy as np
import scipy.special as sf
import scipy.stats as stats


class BaseProcess:
    @abc.abstractmethod
    def update(self):
        """
        Update in light of observations
        """
        pass


class BaseTest:
    def __init__(self, control, treatment):
        self.control = control
        self.treatment = treatment

    @abc.abstractmethod
    def prob(self):
        """
        Return the probabiliy that treatment is better
        """
        pass

    @abc.abstractmethod
    def cvar(self):
        """
        Return the expected shortfall (CVar) upon selecting treatment
        """
        pass


class NormalProcess(BaseProcess):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def update(self, obs):
        pass


class NormalTest(BaseTest):    
    def prob(self):
        std = np.sqrt(self.treatment.sigma**2 + self.control.sigma**2)
        probability = stats.norm(loc=self.control.mu, scale=std).cdf(self.treatment.mu)

        return probability

    def cvar(self):
        std = np.sqrt(self.treatment.sigma**2 + self.control.sigma**2)
        norm = stats.norm(loc=self.treatment.mu, scale=std)
        
        risk = (self.control.mu - self.treatment.mu) * norm.cdf(self.control.mu) + \
            std**2 * norm.pdf(self.control.mu)
        
        return risk


class BernoulliProcess(BaseProcess):
    """
    Parameters
    ----------
    H : int
    number of successes
    T : int
    number of failures
    """

    def __init__(self, H, T):
        self.H = H
        self.T = T

    def update(self, obs):
        """
        Parameters
        ----------
        obs : list
        list of 1s and 0s where 1 is H and 0 is T
        """

        sum_of_obs = np.sum(obs)

        self.H += sum_of_obs
        self.T += len(obs) - sum_of_obs

        return self


class BernoulliTest(BaseTest):
    """
    Parameters
    ----------
    control : BernoulliProcess
    first variant
    treatment : BernoulliProcess
    second variant
    """

    def prob(self):
        """
        Return the probabiliy that treatment is better
        Source: Evan Miller
        """

        total = 0.0
        for i in range(1, int(self.treatment.H) + 2):
            total += np.exp(sf.betaln(self.control.H + i, self.control.T + self.treatment.T + 2)
                - np.log(self.treatment.T + i) - sf.betaln(i, self.treatment.T + 1)
                - sf.betaln(self.control.H + 1, self.control.T + 1))

        return total

    def cvar(self):
        """
        Return the expected shortfall (CVar) upon selecting treatment
        """

        this = self.__class__

        risk = (self.control.H + 1) / (self.control.H + self.control.T + 2) \
                * this(BernoulliProcess(self.treatment.H, self.treatment.T),
                       BernoulliProcess(self.control.H + 1, self.control.T)).prob() \
            - (self.treatment.H + 1) / (self.treatment.H + self.treatment.T + 2) \
                * this(BernoulliProcess(self.treatment.H + 1, self.treatment.T),
                       BernoulliProcess(self.control.H, self.control.T)).prob()

        return risk


class PoissonProcess(BaseProcess):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
    
    def update(self, obs):
        self.alpha += np.sum(obs)
        self.beta += len(obs)

        return self


class PoissonTest(BaseTest):

    def prob(self):
        """
        Return the probabiliy that treatment is better
        Source: Evan Miller
        """

        total = 0.0
        for k in range(self.treatment.alpha):
            total += np.exp(k * np.log(self.treatment.beta) + self.control.alpha * np.log(self.control.beta) 
                    - (k + self.control.alpha) * np.log(self.treatment.beta + self.control.beta)
                    - np.log(k + self.control.alpha) - sf.betaln(k + 1, self.control.alpha))
            
        return total

    def cvar(self):
        """
        Return the expected shortfall (CVar) upon selecting treatment
        """

        this = self.__class__

        risk = self.control.alpha / self.control.beta \
                * this(PoissonProcess(self.treatment.alpha, self.treatment.beta),
                      PoissonProcess(self.control.alpha + 1, self.control.beta)).prob() \
            - self.treatment.alpha / self.treatment.beta \
                * this(PoissonProcess(self.treatment.alpha + 1, self.treatment.beta),
                       PoissonProcess(self.control.alpha, self.control.beta)).prob()

        return risk


class ExponentialProcess(BaseProcess):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
    
    def update(self, obs):
        self.alpha += len(obs)
        self.beta += np.sum(obs)

        return self


class ExponentialTest(BaseTest):

    def prob(self):
        """
        Return the probabiliy that treatment is better
        """
        
        probability = PoissonTest(self.treatment, self.control).prob()

        return probability

    def cvar(self):
        """
        Return the expected shortfall (CVar) upon selecting treatment
        """

        this = self.__class__

        risk = self.control.beta / (self.control.alpha - 1) \
                * this(ExponentialProcess(self.treatment.alpha, self.treatment.beta),
                       ExponentialProcess(self.control.alpha - 1, self.control.beta)).prob() \
            - self.treatment.beta / (self.treatment.alpha - 1) \
                * this(ExponentialProcess(self.treatment.alpha - 1, self.treatment.beta),
                       ExponentialProcess(self.control.alpha, self.control.beta)).prob()

        return risk
