# What is boosting?

Boosting is similar to bagging (aka bootstrapping) in that the final model is a weighted sum of several constituent submodels. While in bagging the same model is trained on random (with replacement) samples of data and the average of the predictions of those models is used as the final prediction, boosting is a different story.

In boosting, models are trained sequentially, and each model is trained on the errors of the previous model. If $\mathcal{D}_1 = \{x_i, y_i\}$ is the training dataset used to train the first model and $p_i^{(1)}$ are the corresponding predictions, then the training set for the second model is
$$
\mathcal{D}_2 = \{x_i, w_2 (y_i - p_i^{(1)})\} ,
$$
where $w_2$ is a prefactor. The predictions of the combined model are now given by
$$
p_i^{(1+2)} = p_i^{(1)} + \frac{p_i^{(2)}}{w_2} .
$$
Similarly, a third model could be trained on $\mathcal{D}_3 = \{x_i, w_3 (y_i - p_i^{(1+2)})\}$, with overall predictions given by
$$
p_i^{(f)} = \sum_i{\frac{1}{w_m p_i^{(m)}}} .
$$

# Whatever is gradient boosting?
You have most likely heard of an optimisation algorithm called _gradient descent_. In a nutshell, it tries to find the minimum of a function $f(x)$ by taking repeated but small steps in the opposite direction to its gradient $\vec{g}=\nabla f$. That is,
$$
\vec{x}_{min} = \vec{x}_0 + \alpha_1 \vec{g}_1 + \alpha_2 \vec{g}_2 + \ldots
$$
where $\vec{x}_0$ is the initial guess as to where the minimum might be. Note that this starting position is quite arbitrary, and that all the other terms are successive gradients evaluated at the previous point, i.e. $\vec{g}_i = \nabla f |_{x_i-1}$.

Now although many machine learning (ML) algorithms are formulated in terms of a cost function that is usually a sum over individual prediction errors, for example,
$$
J=\frac{1}{2}\sum (y_i - p_i)^2,
$$
mathematically speaking, the cost function is in fact a simplification of/approximation to a cost _functional_,
$$
J[f]=\frac{1}{2} \int (y(x) - f(x))^2\, dx
$$
where $y(x)$ is the true function we're trying to find, and $f(x)$ is our estimate of the true function.

The goal of any ML algorithm is to find a function that minimises this cost functional. That is, we're trying to find an $f(x)$ that minimises $J[f]$. Different ML algorithms use different forms to express $f(x)$.

Now we can devise a similar procedure to gradient descent for functionals. Using variational calculus, we can calculate the gradient of the functional (watch this excellent [playlist](​​https://www.youtube.com/playlist?list=PLdgVBOaXkb9CD8igcUr9Fmn5WXLpE8ZE_) by Faculty of Khan for an introduction to variational calculus):
$$
\frac{\delta J}{\delta f} = y(x) - f(x).
$$
Next, suppose we start with a first approximation $f^{(0)}(x)\equiv 0$ to the true function $y(x)$. We calculate the gradient at that function, i.e. $\frac{\delta J}{\delta f} |_{f^{(0)}(x)} = y(x) - f^{(0)}(x)=y(x)$. Now we fit $f^{(1)}(x)$ to the gradient using our algorithm of choice. Again we can calculate the gradient at this new function, $\frac{\delta J}{\delta f} |_{f^{(0)}(x)+f^{(1)}(x)} = y(x) - (f^{(0)}(x) + f^{(1)}(x))$. We then fit another function $f^{(2)}(x)$ to this dataset. Alright, I hope you can see where this is going.



# Appendix: Variational Calculus
The main idea is that we would like to find the gradient of a functional at a certain function. That is, if the input function to the functional were to change a little bit, how would the output of the functional change. Mathematicians use a unique notation for this: $\frac{\delta J[f]}{\delta f}$ where $f$ is a function, and $J[f]$ is a functional of $f$.

Before proceeding with the solution, remember how we calculated the directional gradient of a function back in first semester calculus,
$\nabla_{\vec{v}}f(x) \equiv \vec{v} \cdot \nabla f(x)$
where $\nabla f(x)$ is the gradient, and $\vec{v}$ is the vector representing the direction in which we want to evaluate the directional gradient. Now remember the dot product between two functions is defined as $\int f(x) g(x)\, dx$. So you would naturally expect the directional gradient of the functional in the direction of $\mu(x)$ to be
$\int \frac{\delta J}{\delta f}(x) \mu(x) \, dx$. On the other hand, directional gradient is defined in calculus as $\lim_{\epsilon \rightarrow 0} \frac{J[f+\epsilon \mu]-J[f]}{\epsilon}=\frac{d}{d\epsilon} J[f+\epsilon \mu]|_{\epsilon=0}$. Equating the two, we have
$$
\int \frac{\delta J}{\delta f}(x) \mu(x) \, dx = \lim_{\epsilon \rightarrow 0}\frac{d}{d\epsilon} J[f(x)+\epsilon \mu(x)] .
$$
Therefore, all we have to do to evaluate the functional gradient is calculate the right hand side, and compare it to the left hand side to find $\frac{\delta J}{\delta f}$.
