---
layout: post
title: "Bayesian A/B Testing"
tags:
    - Bayesian statistics
author: "Mehrdad Baghery, Ramin Ghelichi"
comments: true
---
{%- include mathjax.html -%}

A/B testing, also known as bucket testing, is a type of randomized test where a random variable is monitored in two slightly different versions of the same environment, in order to determine which version performs better. It should be fairly straightforward to extend the results to situations where multiple versions are available.
Companies use a multitude of A/B tests every day to evaluate the impact of candidate decisions on their performance. The exact definition of performance depends on what we are trying to improve as a result of the A/B test. If the goal is to get as many customers to click on a certain button as possible, performance is defined in terms of the click-through rate. But if the goal is to earn as much money as possible, performance is defined in terms of the average order value.
In Bayesian statistics, any unknown variable is represented by a probability distribution, the exact choice of which depends on the quantity of interest. We use the Bernoulli distribution for the click-through rate, and the exponential distribution for the average dollar value. We will derive closed-form analytical expressions for these two cases in this post. Furthermore, normal distribution approximations are obtained in the limit of large numbers of observations. The final results are confirmed by numerical simulations where basic familiarity with the PyMC3 package is assumed. The utility file `abtest.py` used repeatedly in this article is available to download [here](/assets/2020-02-28-bayesian-ab-testing/abtest.py).

{% highlight python %}
import numpy as np
import pymc3 as pm
import scipy.stats as stats
from bokeh.plotting import figure, show, output_notebook
from bokeh.palettes import Category10

import abtest as ab

output_notebook()
{% endhighlight %}


<h2>Maximising click-through rate (Bernoulli distribution)</h2>
Suppose we set out to find the website design, of two available options, that is more attractive to our customers. <em>Attractiveness</em> here is defined as the sheer ratio of clicks to impressions, aka click-through rate. The probability of getting $H$ clicks after $N$ impressions is given by the Bernoulli distribution,
\begin{equation}
    P(H,T|p) = p^H (1-p)^T \, \mathrm{,}
\end{equation}
where $T = N - H$, and $p$ is the probability of getting a click; this resembles an (unfair) coin toss experiment where the coin comes up heads $H$ times and tails $T$ times. Since the exact value of $p$ is not known, by way of Bayesian statistics it is expressed as a probability distribution, and our goal is to <em>infer</em> the parameters of this probability distribution as accurately as we can.

<h3>Conjugate prior</h3>
The conjugate prior to the Bernoulli distribution, i.e. the probability distribution describing the click-through rate, is the Beta distribution $p \sim \mathrm{Beta} (\alpha, \beta)$, which for the particular choice of $\alpha=1$ and $\beta=1$ is equal to the uninformative uniform distribution. Let's plot this distribution for a few different values of the input parameters.

{% highlight python %}
x = np.linspace(0, 1, 100)
alphas = [0.5, 1, 1, 5, 5]
betas = [0.5, 1, 5, 1, 5]

p = figure(plot_width=350, plot_height=350)

for alpha, beta, color in zip(alphas, betas, Category10[10]):
    y = stats.beta(alpha, beta).pdf(x)
    p.line(x, y, line_color=color, legend_label=f'alpha={alpha}, beta={beta}')

p.legend.background_fill_alpha = 0.5
p.legend.location = 'top_center'

show(p)
{% endhighlight %}

![Beta distribution](/assets/2020-02-28-bayesian-ab-testing/bokeh_plot.png){:height="50%" width="50%"}

If $\alpha$ is greater than $\beta$, like the red line above, the design is more likely to get a click than not. If $\alpha$ is less than $\beta$, like the green line above, the design is more likely to not get a click.

After collecting data, the prior can be updated through Bayes' theorem to yield what is known as the posterior distribution. In this particular case, the posterior is again a beta distribution but with updated input parameters,
\begin{equation}
    P(p|H,T) \sim \mathrm{Beta}(\alpha + H, \beta + T) \, \mathrm{.}
\end{equation}

<h3>Termination criteria</h3>
After having shown the two different designs to two different groups of customers (called control and treatment groups), we need a concrete method to decide which design performed better. To this end, we will define two quantities: 1) probability of winning, and 2) the expected shortfall. These quantities should be periodically monitored so that the test can be stopped once there is a clear decision. As we will soon see, it is important that these two quantities are considered simultaneously.

<h4>Probability of winning</h4>
If $p_c$ and $p_t$ represent the click-through rates of the control and treatment designs respectively, the probability that the treatment design attracts more clicks in the long run is given by
\begin{equation}
\begin{split}
    h(H_c,T_c,H_t,T_t) &\equiv P(p_t>p_c) \\\\ 
    &= \int_0^1 dp_c \int_{p_c}^1 dp_t\, P(p_c| H_c, T_c) \, P(p_t| H_t, T_t) \, \mathrm{,}
\end{split}
\end{equation}
where  $P(p_c| H_c, T_c)$ and $P(p_t| H_t, T_t)$ are the posteriors of the two designs. The integral above can be evaluated analytically but the derivation is very tedious and we will only quote the final expression here (see references at the bottom of the post):
\begin{equation}
   h(H_c,T_c,H_t,T_t) = \frac{1}{\mathrm{B}(H_c+1, T_c+1)} \sum_{i=1}^{H_t+1} \frac{\mathrm{B}(H_c+i, T_c+T_t+2)}{(T_t+i)\mathrm{B}(i,T_t+1)} \, \mathrm{,}
\end{equation}
where $\mathrm{B}(x,y)$ is the beta <em>function</em>, and uniform priors are assumed for both designs.

If it is observed at any point during the life of the A/B test that $P(p_t>p_c) > 1 - \epsilon$ for a small value of $\epsilon$, e.g. 0.05, the test can be stopped with the second design declared the winner. However, one drawback of only focusing on the probability of winning is that although the sheer probability for one group might be high, the anticipated gain could still be negligible (or equivalently, the expected loss of moving away from the competing design could be high). Therefore, this probability should always be monitored in conjunction with the quantity defined in the next section.


<h4>Conditional value at risk (expected shortfall)</h4>
In addition to the probability of winning, we might want to know the degree to which the winning design is better (or the losing design is worse). The expected gain of one design can be thought of as the expected loss of the other. The expected loss of the treatment design (equal to the expected gain of the control design) is defined as
\begin{equation}
\begin{split}
    \mathcal{L}\_\mathrm{Bernoulli} &\equiv E[p_c-p_t | p_c > p_t ] \\\\ 
    &= \int_0^1 \int_{p_t}^1 (p_c-p_t) P(p_c,p_t) \, dp_t\, dp_c \\\\ 
    &= \frac{H_c+1}{H_c + T_c + 2} h(H_t,T_t,H_c+1,T_c ) - \frac{H_t+1}{H_t + T_t + 2} h(H_t+1,T_t,H_c,T_c) \, \mathrm{.}
\end{split}
\end{equation}

During the course of the A/B test, we pay close attention to the losses of the two designs, and as soon as one of them falls below a predefined threshold $\epsilon$, that design may be declared the winner.

Note that it's best to consider this loss function in the light of the probability of winning defined in the previous section, as focusing only on the expected shortfall might neglect the fact that a variation could have a small chance of winning despite a low expected shortfall. See figure below for a rough guideline.

![Interplay of termination criteria](/assets/2020-02-28-bayesian-ab-testing/picture.png){:height="50%" width="50%"}

<h3>Numerical simulation</h3>
In this section we will simulate an A/B test by generating sythetic data (in real life we should collect this data on the website, app, etc.), and will then try to infer the click-through rates.

{% highlight python %}
# control group
p_control = 0.6
n_control = 2100

with pm.Model() as binomial_control_data:
    y_control = pm.Bernoulli('y', p=p_control)
    trace_control = pm.sample(n_control, progressbar=False, chains=1)

control_obs = pm.trace_to_dataframe(trace_control)['y']
{% endhighlight %}


{% highlight python %}
# treatment group
p_treatment = 0.62
n_treatment = 2000

with pm.Model() as binomial_treatment_data:
    y_treatment = pm.Bernoulli('y', p=p_treatment)
    trace_treatment = pm.sample(n_treatment, progressbar=False, chains=1)

treatment_obs = pm.trace_to_dataframe(trace_treatment)['y']
{% endhighlight %}

All we need to do now is <em>infer</em> the click-through rates of the control and treatment designs using Bayes' theorem. This can be easily done with PyMC3,
{% highlight python %}
n_inf_samples = 10000

with pm.Model() as lift_bayes:
    # control
    inf_p_control = pm.Beta('p_control', alpha=1, beta=1)
    y_control = pm.Bernoulli('y_control', p=inf_p_control, observed=control_obs)

    # treatment
    inf_p_treatment = pm.Beta('p_treatment', alpha=1, beta=1)
    y_treatment = pm.Bernoulli('y_treatment', p=inf_p_treatment, observed=treatment_obs)

    # difference
    p_diff = pm.Deterministic('p_diff', inf_p_control - inf_p_treatment)
    
    trace = pm.sample(n_inf_samples)

inf_data = pm.trace_to_dataframe(trace)
{% endhighlight %}
where we assume a prior of $\mathrm{Beta}(1,1)$ for both treatment and control. Let's evaluate the two termination criteria we defined earlier to compare the two variations,
{% highlight python %}
# monte carlo simulation results
prob = (inf_data['p_treatment'] > inf_data['p_control']).mean()
cvar = inf_data.loc[inf_data['p_treatment'] < inf_data['p_control'], 'p_diff'].sum() / inf_data.shape[0]

# set up the test object for analytical results
control = ab.BernoulliProcess(1, 1).update(control_obs)
treatment = ab.BernoulliProcess(1, 1).update(treatment_obs)
test = ab.BernoulliTest(control, treatment)
{% endhighlight %}


{% highlight python %}
print('Probability of winning for treatment')
print('Monte Carlo:', prob, end=', ')
print('Analytical:', test.prob())

print('----')

print('CVaR of treatment')
print('Monte Carlo:', cvar, end=', ')
print('Analytical:', test.cvar())
{% endhighlight %}
<samp>
Probability of winning for treatment <br />
Monte Carlo: 0.96805, Analytical: 0.96883 <br />
---- <br />
CVaR of treatment <br />
Monte Carlo: 0.00020, Analytical: 0.00018 <br />
</samp>
where we see that treatment has a high probability of winning (0.97), while the expected loss is very low (0.00019). Hence, the treatment variation may be given the green light.

<h3>Maximum Likelihood Estimation & Normal approximation</h3>
If the number of impressions is large enough, the beta distribution can be approximated by a normal distribution. A large number of impressions also means the posterior is mainly dictated by the likelihood, justifying the use of the maximum likelihood estimation (MLE). In this case, the mean and variance of the beta distribution are given by
\begin{equation}
\begin{split}
    \mu &\approx \frac{H}{N} \\\\ 
    \sigma^2 &\approx \frac{\mu (1-\mu)}{N} \, \mathrm{,}
\end{split}
\end{equation}

{% highlight python %}
mu_treatment = treatment_obs.mean()
sigma_treatment = np.sqrt(mu_treatment * (1 - mu_treatment) / n_treatment)
{% endhighlight %}


{% highlight python %}
mu_control = control_obs.mean()
sigma_control = np.sqrt(mu_control * (1 - mu_control) / n_control)
{% endhighlight %}
which can be used as input parameters to the two approximate normal distributions we will use to describe control and treatment. Similar termination criteria can be derived for two normal distributions (see appendix).

{% highlight python %}
control = ab.NormalProcess(mu_control, sigma_control)
treatment = ab.NormalProcess(mu_treatment, sigma_treatment)
test = ab.NormalTest(control, treatment)
{% endhighlight %}

{% highlight python %}
print('Probability of winning for treatment')
print('Normal approximation:', test.prob())

print('----')

print('CVaR of treatment')
print('Normal approximation:', test.cvar())
{% endhighlight %}
<samp>
Probability of winning for treatment <br />
Normal approximation: 0.96905 <br />
---- <br />
CVaR of treatment <br />
Normal approximation: 0.00018 <br />
</samp>
We see that these are in very good agreement with the numerical and analytical results evaluated earlier.


<h3>How many observations are needed?</h3>
A frequently asked question at the beginning of a test is ``how long should the test run for?''. While a traditional A/B test (aka the frequentist's A/B test) needs to run its course before it can be concluded, this is not the case for a Bayesian test, which can be stopped as soon as the termination criteria are met. Having said that, an estimate can be made for how long the test should be run in the worst case scenario, i.e. when the two groups behave very similarly. In this case we would require the standard deviation $\sigma$ of each posterior distribution divided by its mean $\mu$ to be smaller than a threshold $\epsilon=5\%$,
\begin{equation}
    \frac{\sigma^2}{\mu^2} \approx \frac{N-H}{N H} < \epsilon^2 \, \mathrm{,}
\end{equation}
where we have used the approximate relations introduced earlier. Assuming $H$ is much smaller than $N$ (click-through rates are usually on the order of a few percent), we get
\begin{equation}
    H > \frac{1}{\epsilon^2} \, \mathrm{.}
\end{equation}
According to this condition, if the click-through rate is expected to be around 2\%, we will need to have $H>400$ clicks, or equivalently, each variation should be shown to $N>20,000$ customers.



<h2>Maximising revenue [Exponential distribution]</h2>
Sometimes our goal is not to find the variation that merely attracts more clicks, but the one that generates more revenue. Order values (dollars spent on an order) follow an exponential distribution. Let's take a look at a few examples,
{% highlight python %}
x = np.linspace(0, 2, 100)

p = figure(plot_width=350, plot_height=350)

for lamb, color in zip([0.1, 0.5, 1, 2, 5], Category10[5]):
    y = stats.expon(scale=1/lamb).pdf(x)
    p.line(x, y, line_color=color, legend_label=f'lambda={lamb}')

p.legend.background_fill_alpha = 0.5

show(p)
{% endhighlight %}

![Interplay of termination criteria](/assets/2020-02-28-bayesian-ab-testing/bokeh_plot2.png){:height="50%" width="50%"}

where the smaller $\lambda$ is, the longer the tail is. If the horizontal axis represents the order value, a longer tail would correspond to having orders with higher values. Our objective is to find the variation with a longer tail. Furthermore, the mean of the exponential distribution, i.e. the average order value, is $1/\lambda$, which means a longer tail corresponds to a greater average order value, and vice versa.

<h3>Conjugate prior</h3>
The conjugate prior to an exponential distribution, that is the probability distribution of its parameter $\lambda$, is the gamma distribution $\lambda \sim \Gamma(\alpha, \beta)$ with two parameters, shape $\alpha$ and rate $\beta$. Let's see what the gamma distribution looks like for a few values of $\alpha$ and $\beta$.

{% highlight python %}
x = np.linspace(0, 20, 100)

p = figure(plot_width=350, plot_height=350, x_axis_label='lambda')

for alpha, beta, color in zip([1, 1, 2, 2], [1, 2, 1, 2], Category10[10]):
    y = stats.gamma(alpha, scale=beta).pdf(x)
    p.line(x, y, line_color=color, legend_label=f'alpha={alpha}, beta={beta}')

p.legend.background_fill_alpha = 0.5

show(p)
{% endhighlight %}

![Interplay of termination criteria](/assets/2020-02-28-bayesian-ab-testing/bokeh_plot3.png){:height="50%" width="50%"}

In the light of incoming data, the prior is updated as follows to yield the posterior
\begin{equation}
\begin{split}
    \alpha &\rightarrow \alpha + n \\\\ 
    \beta &\rightarrow \beta + \sum_{i=1}^{n} x_{i}\, \mathrm{,}
\end{split}
\end{equation}
where $x_i$ denote observations, and $n$ is the total number of them.


<h3>Termination criteria</h3>

<h4>Probability of winning</h4>
The probability that the treatment group has a longer tail (or equivalently a higher average order value) is given by
\begin{equation}
\begin{split}
    h(\alpha_c, \beta_c, \alpha_t, \beta_t) &\equiv P(\lambda_t < \lambda_c) \\\\ 
    &= \int_0^\infty d\lambda_t \, \Gamma(\lambda_t; \alpha_t, \beta_t) \int_{\lambda_t}^\infty d\lambda_c \, \Gamma(\lambda_c; \alpha_c, \beta_c) \\\\ 
    &= \sum_{k=0}^{\alpha_c-1} \frac{(\beta_t + \beta_c)^{-(k+\alpha_t)} \beta_c^k \beta_t^{\alpha_t}}{(k+\alpha_t) \mathrm{B}(k+1,\alpha_t)} \, \mathrm{,}
\end{split}
\end{equation}
where $\mathrm{B}(x,y)$ is the beta <em>function</em>.


<h4>Conditional value at risk (expected shortfall)</h4>
Given the mean of the exponential distribution, $\frac{1}{\lambda}$, a good choice of expected shortfall is the expected value of the difference of the treatment and control means $\frac{1}{\lambda_c} - \frac{1}{\lambda_t}$, provided that $\lambda_t > \lambda_c$:
\begin{equation}
\begin{split}
    \mathcal{L}\_\mathrm{Exp}^\mathrm{test} &\equiv E\left[\frac{1}{\lambda_c} - \frac{1}{\lambda_t} \middle| \lambda_t > \lambda_c \right] \\\\ 
    &= \int_0^{+\infty} d\lambda_c \, \Gamma(\lambda_c; \alpha_c, \beta_c) \int_{\lambda_c}^{+\infty} d\lambda_t \, \Gamma(\lambda_t; \alpha_t, \beta_t) \, \left( \frac{1}{\lambda_c} - \frac{1}{\lambda_t} \right) \\\\ 
    &= \frac{\beta_c}{\alpha_c - 1} h(\alpha_t, \beta_t, \alpha_c - 1, \beta_c) - \frac{\beta_t}{\alpha_t - 1} h(\alpha_t - 1, \beta_t, \alpha_c, \beta_c) \, \mathrm{.}
\end{split}
\end{equation}


<h3>Numerical simulation</h3>
Similarly to what was done for the Bernoulli distribution above, we will first generate synthetic data for a fake A/B test (in a real test this data would be collected on the website).


{% highlight python %}
# control group
lambda_control = 1 / 4.0 # average order value: 4.00 EUR
n_control = 420

with pm.Model() as binomial_treatment_data:
    y_treatment = pm.Exponential('y', lam=lambda_control)

    trace_control = pm.sample(n_control, progressbar=False, chains=1)

control_obs = pm.trace_to_dataframe(trace_control)['y']
{% endhighlight %}


{% highlight python %}
# treatment group
lambda_treatment = 1 / 3.6 # average order value: 3.60 EUR
n_treatment = 300

with pm.Model() as binomial_treatment_data:
    y_treatment = pm.Exponential('y', lam=lambda_treatment)

    trace_treatment = pm.sample(n_treatment, progressbar=False, chains=1)

treatment_obs = pm.trace_to_dataframe(trace_treatment)['y']
{% endhighlight %}

We may now proceed to infer $\lambda$ for control and treatment.

{% highlight python %}
n_inf_samples = 20000

with pm.Model() as EmailTest:
    # control
    lambda_control = pm.Gamma('lambda_control', alpha=1, beta=1)
    inf_control = pm.Exponential('y_control', lam=lambda_control, observed=control_obs)

    # treatment
    lambda_treatment = pm.Gamma('lambda_treatment', alpha=1, beta=1)
    inf_treatment = pm.Exponential('y_treatment', lam=lambda_treatment, observed=treatment_obs)

    # difference
    lambda_diff = pm.Deterministic('lambda_diff', 1/lambda_control - 1/lambda_treatment)

    trace = pm.sample(n_inf_samples, tune=2000)

inf_data = pm.trace_to_dataframe(trace)
{% endhighlight %}

Calculating the two termination criteria is just a matter of tinkering with the dataframes produced by PyMC3,
{% highlight python %}
# monte carlo simulation results
prob = (inf_data['lambda_treatment'] < inf_data['lambda_control']).mean()
cvar = inf_data.loc[inf_data['lambda_treatment'] >= inf_data['lambda_control'], 'lambda_diff'].sum() \
    / inf_data.shape[0]

# set up the test object for analytical results
control = ab.ExponentialProcess(1, 1).update(control_obs)
treatment = ab.ExponentialProcess(1, 1).update(treatment_obs)
test = ab.ExponentialTest(control, treatment)
{% endhighlight %}

{% highlight python %}
print('Probability of winning for treatment')
print('Monte Carlo:', prob, end=', ')
print('Analytical:', test.prob())

print('----')

print('CVaR of treatment')
print('Monte Carlo:', cvar, end=', ')
print('Analytical:', test.cvar())
{% endhighlight %}
<samp>
Probability of winning for treatment <br />
Monte Carlo: 0.16605, Analytical: 0.16488 <br />
---- <br />
CVaR of treatment <br />
Monte Carlo: 0.33600, Analytical: 0.33671 <br />
</samp>
where we see that the treatment group has a small chance of winning (0.16), with a relatively high expected shortfall (0.33). Note that this need not always be the case, that is, a low probability will not always translate into a high expected shortfall, or vice versa.


<h3>Maximum Likelihood Estimation & Normal approximation</h3>
Since the quantity of interest is in fact $1/\lambda$, we need to calculate the mean and variance of $1/\lambda$ (which are identical to the mean and variance of the inverse gamma distribution with the same parameter values as the posterior). These quantities can be readily looked up on Wikipedia,
\begin{equation}
\begin{split}
    \mu &\approx \frac{\beta}{\alpha} \\\\ 
    \sigma^2 &\approx \frac{\beta^2}{\alpha^3} \approx \frac{\mu^2}{n}\, \mathrm{,}
\end{split}
\end{equation}
where $n$ is the total number of observations.

{% highlight python %}
mu_control = control_obs.mean()
sigma_control = mu_control / np.sqrt(n_control)
{% endhighlight %}

{% highlight python %}
mu_treatment = treatment_obs.mean()
sigma_treatment = mu_treatment / np.sqrt(n_treatment)
{% endhighlight %}

As with the Bernoulli distribution above, we use these quantities as the input parameters to the approximate normal distributions,
{% highlight python %}
control = ab.NormalProcess(mu_control, sigma_control)
treatment = ab.NormalProcess(mu_treatment, sigma_treatment)
test = ab.NormalTest(control, treatment)
{% endhighlight %}

{% highlight python %}
print('Probability of winning for treatment')
print('Normal approximation:', test.prob())

print('----')

print('CVaR of treatment')
print('Normal approximation:', test.cvar())
{% endhighlight %}
<samp>
Probability of winning for treatment <br />
Normal approximation: 0.69995 <br />
---- <br />
CVaR of treatment <br />
Normal approximation: 0.05362 <br />
</samp>
where a prior of $\Gamma(1,1)$ is used. The calculated values are in excellent agreement with the analytical and numerical evaluations above.


<h3>How many observations are needed?</h3>
If we require the standard deviation $\sigma$ of the corresponding inverse gamma distribution divided by its mean $\mu$ to be smaller than $\epsilon$, we get
\begin{equation}
    \frac{\sigma^2}{\mu^2} \approx \frac{1}{N} < \epsilon^2 \, \mathrm{,}
\end{equation}
which when rearranged yields
\begin{equation}
    N > \frac{1}{\epsilon^2} \, \mathrm{,}
\end{equation}
meaning at least 400 samples are needed if $\epsilon=0.05$.


<h2>Conclusion</h2>
A/B tests are an essential tool for any company that wants to make decisions based on data. They provide a simple setting for comparing different possible variations of the same feature, e.g. a dark theme versus a light theme for an app.

We formalised A/B testing for two different governing probability distributions: the Bernoulli and exponential distributions. The Bernoulli distribution should be used when the sheer number of clicks, purchases, etc. is to be maximised, while the exponential distribution is used when the revenue made through those clicks is to be maximised.

We also derived conditions under which the test can be terminated (with a clear winner).

<h2>Acknowledgement</h2>
We'd like to thank Julius Monello for pre-reading the draft.


<h2>Appendices</h2>

<h3>Maximum likelihood estimation</h3>
In the main text we take advantage of the analytical form of the posterior to calculate its mean and standard deviation. But an analytical form for the posterior cannot always be found. If a large number of observations are available, the posterior is mostly dominated by the likelihood,
\begin{equation}
    p(\theta|D) = \frac{p(D|\theta) \, p(\theta)}{p(D)} \propto p(D|\theta) \, \mathrm{,}
\end{equation}
and the posterior can be approximated by a normal distribution,
\begin{equation}
p(\theta|D)=\frac{1}{\sqrt{2\pi \sigma^2}}\exp\left( -\frac{(\theta-\hat{\theta})^2}{2\sigma^2} \right) \, \mathrm{,}
\end{equation}
where $\hat{\theta}$ is the maximum likelihood estimate defined by $\partial_{\theta=\hat{\theta}} \log p(D|\theta)=0$. The standard deviation $\sigma$ can be approximated by noting that
\begin{equation}
    \partial_{\theta=\hat{\theta}}^2 \log p(\theta|D) \approx \partial_{\theta=\hat{\theta}}^2 \log p(D|\theta) \, \mathrm{,}
\end{equation}
where the left hand side yields
\begin{equation}
    \partial_{\theta=\hat{\theta}}^2 \log p(\theta|D) = - \frac{1}{\sigma^2} \, \mathrm{.}
\end{equation}
Hence
\begin{equation}
    \sigma^2 = - \frac{1}{\partial_{\theta=\hat{\theta}}^2 \log p(D|\theta)} \, \mathrm{.}
\end{equation}

In sum, if we are only interested in the behavior of the posterior in the presence of a large number of observations, we don't need a conjugate prior or an analytical expression for the posterior. The posterior can be approximated by a normal distribution whose parameters (mean and variance) are given by the equations above. Furthermore, if the posterior is going to be used in some sort of A/B test, we can use the termination criteria in the next section.


<h3>Termination criteria for normal distributions</h3>
<h4>Probability of winning</h4>
If we assume the posterior is a normal distribution, the probability of winning for the test group is given by
\begin{equation}
\begin{split}
    P(p_t > p_c) &= \int_{-\infty}^{+\infty} dp_t \int_{-\infty}^{p_t} dp_c \, N(p_t; \mu_t, \sigma_t) \, N(p_c; \mu_c, \sigma_c) \\\\ 
    &= \Phi\left(\mu_t; \mu_c, \sqrt{\sigma_t^2 + \sigma_c^2}\right) \, \mathrm{,}
\end{split}
\end{equation}
where $\Phi(x; \mu, \sigma)$ is the normal CDF (cumulative distribution function).


<h4>Conditional value at risk (expected shortfall)</h4>
Similarly, the expected shortfall is given by
\begin{equation}
\begin{split}
    \mathcal{L}\_\mathrm{norm}^\mathrm{test} &= \int_{-\infty}^{+\infty} dp_c \int_{-\infty}^{p_c} dp_t \, (p_c - p_t) \, N(p_c; \mu_c, \sigma_c) \, N(p_t; \mu_t, \sigma_t) \\\\ 
    &= (\mu_c - \mu_t) \, \Phi\left(\mu_c; \mu_t, \sqrt{\sigma_t^2 + \sigma_c^2}\right) + (\sigma_t^2 + \sigma_c^2) \, N\left(\mu_c; \mu_t, \sqrt{\sigma_t^2 + \sigma_c^2} \right) \, \mathrm{.}
\end{split}
\end{equation}

<h2>References</h2>
Evan Miller, [Formulas for Bayesian A/B Testing](https://www.evanmiller.org/bayesian-ab-testing.html)
<br />
Wikipedia, [List of integrals of Gaussian functions](https://en.wikipedia.org/wiki/List_of_integrals_of_Gaussian_functions)



{%- include disqus_comments.html -%}
