---
layout: post
title: "Whatever is gradient boosting?"
tags:
    - Machine learning
author: "Mehrdad Baghery"
comments: true
---
{%- include mathjax.html -%}

<h2>What is boosting?</h2>

Boosting is similar to bagging (aka bootstrapping) in that the final model is a weighted sum of several constituent submodels. While in bagging the same model is trained on random (with replacement) samples of data and the average of the predictions of those models is used as the final prediction, boosting is a different story.

In boosting, models are trained sequentially, and each model is trained on the errors of the previous model. If $D_1 = \{x_i, y_i\}$ is the training dataset used to train the first model and $p_i^{(1)}$ are the corresponding predictions, then the training set for the second model is
\begin{equation}
D_2 = \{x_i, w_2 (y_i - p_i^{(1)})\} ,
\end{equation}
where $w_2$ is a prefactor. The predictions of the combined model are now given by
\begin{equation}
p_i^{(1+2)} = p_i^{(1)} + \frac{p_i^{(2)}}{w_2} .
\end{equation}
Similarly, a third model could be trained on $D_3 = \{x_i, w_3 (y_i - p_i^{(1+2)})\}$, with overall predictions given by
\begin{equation}
p_i^{(f)} = \sum_i{\frac{1}{w_m p_i^{(m)}}} .
\end{equation}

<h2>Whatever is gradient boosting?</h2>
You have most likely heard of an optimisation algorithm called _gradient descent_. In a nutshell, it tries to find the minimum of a function $f(x)$ by taking repeated but small steps in the opposite direction to its gradient $\vec{g}=\nabla f$. That is,
\begin{equation}
\vec{x}_{\min} = \vec{x}_0 + \alpha_1 \vec{g}_1 + \alpha_2 \vec{g}_2 + \ldots
\end{equation}
where $\vec{x}_0$ is the initial guess as to where the minimum might be. Note that this starting position is quite arbitrary, and that all the other terms are successive gradients evaluated at the previous point, i.e. $\vec{g}_i = \nabla f \vert_{x_i-1}$.

Now although many machine learning (ML) algorithms are formulated in terms of a cost function that is usually a sum over individual prediction errors, for example,
\begin{equation}
J=\frac{1}{2}\sum (y_i - p_i)^2,
\end{equation}
mathematically speaking, the cost function is in fact a simplification of/approximation to a cost _functional_,
\begin{equation}
J[f]=\frac{1}{2} \int (y(x) - f(x))^2\, dx
\end{equation}
where $y(x)$ is the true function we're trying to find, and $f(x)$ is our estimate of the true function.

The goal of any ML algorithm is to find a function that minimises this cost functional. That is, we're trying to find an $f(x)$ that minimises $J[f]$. Different ML algorithms use different forms to express $f(x)$.

Now we can devise a similar procedure to gradient descent for functionals. Using variational calculus, we can calculate the gradient of the functional (watch this excellent [playlist]({​​https://www.youtube.com/playlist?list=PLdgVBOaXkb9CD8igcUr9Fmn5WXLpE8ZE_}) by Faculty of Khan for an introduction to variational calculus):
\begin{equation}
\frac{\delta J}{\delta f} = y(x) - f(x).
\end{equation}
Next, suppose we start with a first approximation $f^{(0)}(x)\equiv 0$ to the true function $y(x)$. We calculate the gradient at that function, i.e. $\frac{\delta J}{\delta f} \vert_{f^{(0)}(x)} = y(x) - f^{(0)}(x)=y(x)$. Now we fit $f^{(1)}(x)$ to the gradient using our algorithm of choice. Again we can calculate the gradient at this new function, $\frac{\delta J}{\delta f} \vert_{f^{(0)}(x)+f^{(1)}(x)} = y(x) - (f^{(0)}(x) + f^{(1)}(x))$. We then fit another function $f^{(2)}(x)$ to this dataset. Alright, I hope you can see where this is going.



<h2>Appendix: Variational Calculus</h2>
The main idea is that we would like to find the gradient of a functional at a certain function. That is, if the input function to the functional were to change a little bit, how would the output of the functional change. Mathematicians use a unique notation for this: $\frac{\delta J[f]}{\delta f}$ where $f$ is a function, and $J[f]$ is a functional of $f$.

Before proceeding with the solution, remember how we calculated the directional gradient of a function back in first semester calculus,
$\nabla_{\vec{v}}f(x) \equiv \vec{v} \cdot \nabla f(x)$
where $\nabla f(x)$ is the gradient, and $\vec{v}$ is the vector representing the direction in which we want to evaluate the directional gradient. Now remember the dot product between two functions is defined as $\int f(x) g(x)\, dx$. So you would naturally expect the directional gradient of the functional in the direction of $\mu(x)$ to be
$\int \frac{\delta J}{\delta f}(x) \mu(x) \, dx$. On the other hand, directional gradient is defined in calculus as $\lim_{\epsilon \rightarrow 0} \frac{J[f+\epsilon \mu]-J[f]}{\epsilon}=\frac{d}{d\epsilon} J[f+\epsilon \mu]\vert_{\epsilon=0}$. Equating the two, we have
\begin{equation}
\int \frac{\delta J}{\delta f}(x) \mu(x) \, dx = \lim_{\epsilon \rightarrow 0}\frac{d}{d\epsilon} J[f(x)+\epsilon \mu(x)] .
\end{equation}
Therefore, all we have to do to evaluate the functional gradient is calculate the right hand side, and compare it to the left hand side to find $\frac{\delta J}{\delta f}$.



{%- include disqus_comments.html -%}
