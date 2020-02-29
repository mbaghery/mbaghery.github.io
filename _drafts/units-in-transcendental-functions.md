---
layout: post
title: "Why no units in arguments of transcendental functions?"
tags:
    - physics
author: "Mehrdad Baghery"
comments: true
---
{%- include mathjax.html -%}

Ever wondered why we never use unitted numbers in sin, cos, etc.? Well, I have. In fact, I went through a whole lot of education without knowing exactly why.

I remember someone telling me "if you Taylor expand $\sin(1m)$ you are gonna have to add up terms of inconsistent dimensions (for example $m, m^2, \ldots$), which is clearly meaningless".

But this turns out to be wrong. Why? Because each term in the Taylor expansion has a prefactor ...

Ok. If we are allowed to write $\sin(1m)$, what is $\sin(2m)$? This is easy:
\begin{equation}
    \sin(2m)=2 \sin(1m) \cos(1m) \, \mathrm{.}
\end{equation}
But wait, that didn't get any easier.
Exactly. What's worse, what are we gonna do with $\exp(1m)+\sin(x)$ should it arise somewhere? How do we define the necessary algebra?

Furthermore, we've never measured anything like $\sin(1m)$ before, although it is not mathematically wrong.

Is there any way we can avoid this whole mess?
That's where the Buckingham theorem comes into play. It turns out we can rewrite any physical differential equation such that it is completelyfree of any units.

<h2>Comments</h2>
{% if page.comments %}
<div id="disqus_thread"></div>
<script>
var disqus_config = function () {
    this.page.url = '{{ page.url | absolute_url }}';
    this.page.identifier = '{{ page.url | absolute_url }}';
};
(function() { // DON'T EDIT BELOW THIS LINE
var d = document, s = d.createElement('script');
s.src = 'https://goodstuffgithub.disqus.com/embed.js';
s.setAttribute('data-timestamp', +new Date());
(d.head || d.body).appendChild(s);
})();
</script>
<noscript>Please enable JavaScript to view the <a href="https://disqus.com/?ref_noscript">comments powered by Disqus.</a></noscript>
{% endif %}