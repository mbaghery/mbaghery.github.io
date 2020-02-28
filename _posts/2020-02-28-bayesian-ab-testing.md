---
layout: post
title:  "Bayesian A/B Testing"
date:   2020-02-28
categories: jekyll update
author: Mehrdad Baghery, Ramin Ghelichi
---
A/B testing, also known as bucket testing, is a type of randomized test where a random variable is monitored in two slightly different versions of the same environment, in order to determine which version performs better. It should be fairly straightforward to extend the results to situations where multiple versions are available.
Companies use a multitude of A/B tests every day to evaluate the impact of candidate decisions on the performance of the company. The exact definition of performance depends on what we are trying to improve as a result of the A/B test. If the goal is to get as many customers to click on a certain button as possible, performance is defined in terms of the click-through rate. But if the goal is to earn as much money as possible, performance is defined in terms of the average order value.
In Bayesian statistics, any unknown variable is represented by a probability distribution, the exact choice of which depends on the quantity of interest. We use the Bernoulli distribution for the click-through rate, and the exponential distribution for the average dollar value. We will derive closed-form analytical expressions for these two cases in this post. Furthermore, normal distribution approximations are obtained in the limit of large numbers of observations. The final results are confirmed by numerical simulations where basic familiarity with the PyMC3 package is assumed. The utility file abtest.py used repeatedly in this article is available to download here.

{% highlight python %}
import numpy as np
import pymc3 as pm
import scipy.stats as stats
from bokeh.plotting import figure, show, output_notebook
from bokeh.palettes import Category10

import abtest as ab

output_notebook()
{% endhighlight %}

Let's test $$\frac{1}{2x}$$.

