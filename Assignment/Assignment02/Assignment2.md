# Assignment 02





## Question 1

(a) 

True.

(b) 

First, we need to marginalize $x_{c}$ . According to the marginal Gaussian distribution,  $p(x_{a}, x_{b})$ is a Gaussian distribution with arguments as follows:
$$
\mu = \left( \begin{array}{c} \mu_a\\ \mu_b\end{array} \right), \quad
\Sigma=\left( \begin{array}{c} \Sigma_{aa} & \Sigma_{ab} \\  
								\Sigma_{ba} & \Sigma_{bb} \end{array} \right)
$$
Then, according to the conditional Gaussian distributions, $p(x_{a}|x_{b})$ is a Faussian distribution with arguments as follows:
$$
\mu_{a|b}=\mu_{a}+\Sigma_{ab}\Sigma^{-1}_{bb}(x_{b}-\mu_{b}), \quad 
\Sigma_{a|b}=\Sigma_{aa}-\Sigma_{ab}\Sigma^{-1}_{bb}\Sigma_{ba}
$$




## Question 2

(a)

According to the marginal Gaussian distribution, $p(\mathbf{x})$ is a Gaussian distribution with arguments as follows:
$$
\mu_{\mathbf{x}} = \mu, \quad \Sigma_{\mathbf{x}}= \Sigma_{\mathbf{x}\mathbf{x}}= \Lambda^{-1}
$$
So that $p(\mathbf{x})=\mathcal{N}(\mathbf{x|}\mu\mathbf{, \Lambda^{-1}})$.

(b)

According to the conditional Gaussian distribution, $p(\mathbf{y|x})$ is a Gaussian distribution with arguments as follows:
$$
\mu_{\mathbf{y|x}}=\mu_{\mathbf{y}}+\Sigma_{\mathbf{yx}}\Sigma^{-1}_{\mathbf{yy}}(\mathbf{x}-\mu)
=\mathbf{A}\mu \mathbf{+b} + \mathbf{A}\Lambda^{-1}\Lambda(\mathbf{x}-\mu)
= \mathbf{A}\mathbf{x}+\mathbf{b}
\\\\
\Sigma_{\mathbf{y|x}}=\Sigma_{\mathbf{yy}}-\Sigma_{\mathbf{yx}}\Sigma^{-1}_{\mathbf{xx}}\Sigma_{\mathbf{xy}}
= \mathbf{L^{-1}+A\Lambda^{-1}A^\mathrm{T}} - \mathbf{A}\Lambda^{-1} \Lambda \mathbf{\Lambda^{-1}A^\mathrm{T}}
=\mathbf{L^{-1}}
$$


So that $p(\mathbf{y|x})=\mathcal{N}(\mathbf{y|Ax+b, L^{-1}})$.







## Question 3

For the term $(-\frac{N}{2}\mathrm{ln}|\Sigma|)$ :
$$
\frac{\partial}{\partial\Sigma}(-\frac{\partial}{\partial\Sigma} \frac{N}{2}\mathrm{ln}|\Sigma|)
=-\frac{N}{2}(\Sigma^{-1})^{\mathrm{T}}
=-\frac{N}{2}\Sigma^{-1}
$$


For the term $(-\frac{1}{2}\sum^N_{n=1}(\mathbf{x}_n-\mu)^\mathrm{T}\Sigma^{-1}(\mathbf{x}_n-\mu))$ :
$$
\text{take }\mathrm{S}=\frac{1}{N}\sum^N_{n=1}(\mathbf{x}_n-\mu)(\mathbf{x}_n-\mu)^\mathrm{T}
\\\\
\text{thus we have }\sum^N_{n=1}(\mathbf{x}_n-\mu)^\mathrm{T}\Sigma^{-1}(\mathbf{x}_n-\mu)=
N\mathrm{Tr}[\Sigma^{-1}\mathrm{S}]
\\\\
\frac{\partial}{\partial\Sigma_{ij}}N\mathrm{Tr}[\Sigma^{-1}\mathrm{S}]
=N\mathrm{Tr}[\frac{\partial}{\partial\Sigma_{ij}}\Sigma^{-1}\mathrm{S}]  \\
=N\mathrm{Tr}[\frac{\partial}{\partial\Sigma_{ij}}\Sigma^{-1}\mathrm{S}]  \\
=-N\mathrm{Tr}[\Sigma^{-1}\frac{\partial\Sigma}{\partial\Sigma_{ij}}\Sigma^{-1}\mathrm{S}]  \\
=-N\mathrm{Tr}[\frac{\partial\Sigma}{\partial\Sigma_{ij}}\Sigma^{-1}\mathrm{S}\Sigma^{-1}]  
\\\\
\text{As } \Sigma_{ij} = \Sigma_{ji},  \\

\frac{\partial}{\partial\Sigma_{ij}}N\mathrm{Tr}[\Sigma^{-1}\mathrm{S}]
=-N(\Sigma^{-1}\mathrm{S}\Sigma^{-1})_{ij}
$$
So that we have:
$$
\frac{\partial}{\partial\Sigma}(-\frac{1}{2}\sum^N_{n=1}(\mathbf{x}_n-\mu)^\mathrm{T}\Sigma^{-1}(\mathbf{x}_n-\mu))
=\frac{N}{2}(\Sigma^{-1}\mathrm{S}\Sigma^{-1})
$$
Summing up the two terms we have:
$$
\frac{\partial}{\partial\Sigma} \mathrm{ln}p(\mathbf{X}|\mu, \Sigma) = -\frac{N}{2}\Sigma^{-1} 
+\frac{N}{2}(\Sigma^{-1}\mathrm{S}\Sigma^{-1})
=0
\\
\frac{N}{2}(\Sigma^{-1}\mathrm{S}\Sigma^{-1})=\frac{N}{2}\Sigma^{-1} 
\\
\Sigma = S
$$


### Lack something







## Question 4

(a)

From the side of $\sigma^{2}_{N}$:
$$
\sigma^{2}_{N}
=\frac{1}{N}\sum^{N-1}_{n=1}(x_{n}-\mu)+\frac{(x_{N}-\mu)^2}{N}  \\
=\frac{N-1}{N}\sigma^{2}_{N-1} + \frac{(x_{N}-\mu)^2}{N}  \\
=\sigma^{2}_{N-1} - \frac{1}{N}\sigma^{2}_{N-1} + \frac{(x_{N}-\mu)^2}{N}  \\
=\sigma^{2}_{N-1} + \frac{1}{N}((x_{N}-\mu)^2 - \sigma^{2}_{N-1})
$$
From the side of Robbins-Monro:
$$
\sigma^{2}_{N}
=\sigma^{2}_{N-1} 
+a_{N-1}\frac{\partial}{\partial\sigma^{2}_{N-1}}
(-\frac{\mathrm{ln}\sigma^{2}_{N-1}}{2}-\frac{(x_{N}-\mu)^{2}}{2\sigma^{2}_{N-1}})  \\

=\sigma^{2}_{N-1}
+a_{N-1}
(-\frac{1}{2\sigma^{2}_{N-1}} + \frac{(x_{N}-\mu)^{2}}{2\sigma^{4}_{N-1}})  \\

=\sigma^{2}_{N-1}
+\frac{a_{N-1}}{2\sigma^{4}_{N-1}}
((x_{N}-\mu)^2 - \sigma^{2}_{N-1})
$$
Comparing the two formulas we can get know that:
$$
\frac{a_{N-1}}{2\sigma^{4}_{N-1}} = \frac{1}{N}
\\\\
a_{N-1}=\frac{2\sigma^{4}_{N-1}}{N}
$$


(b)

From the side of $\Sigma_{N}$:
$$
\Sigma_{N}
=\frac{1}{N}\sum^{N-1}_{n=1}(\mathbf{x}_{n}-\mu)(\mathbf{x}_{n}-\mu)^{T}
+\frac{1}{N}(\mathbf{x}_{N}-\mu)(\mathbf{x}_{N}-\mu)^{T}
\\
=\frac{N-1}{N}\Sigma_{N-1}
+\frac{1}{N}(\mathbf{x}_{N}-\mu)(\mathbf{x}_{N}-\mu)^{T}
\\
=\Sigma_{N-1}
+\frac{1}{N}((\mathbf{x}_{N}-\mu)(\mathbf{x}_{N}-\mu)^{T}-\Sigma_{N-1})
$$
From the side of Robbins-Monro:
$$
\Sigma_{N}=
\Sigma_{N-1}
+a_{N-1}\frac{\partial}{\partial\Sigma_{N-1}}
(\mathrm{ln}p(\mathbf{x}_{N}|\mu,\Sigma_{N-1}))
\\\\
\text{from Question3 we have: }\\
\frac{\partial}{\partial\Sigma} \mathrm{ln}p(\mathbf{x}|\mu, \Sigma) 
= -\frac{1}{2}\Sigma^{-1}+\frac{1}{2}(\Sigma^{-1}\mathrm{S}\Sigma^{-1})
= \frac{1}{2}\Sigma^{-1}(S-\Sigma)\Sigma^{-1}
\\\\
\text{So we get:}\\
\Sigma_{N}
=\Sigma_{N-1}
+\frac{a_{N-1}}{2}
\Sigma^{-1}_{N-1} ((\mathbf{x}_{N}-\mu)(\mathbf{x}_{N}-\mu)^{T}-\Sigma_{N-1}) \Sigma^{-1}_{N-1}
\\\\
\text{As $\Sigma^{-1}_{N-1}$ is diagnal, we can go further and get:}\\
\Sigma_{N}
=\Sigma_{N-1}
+a_{N-1}
\frac{1}{2} \Sigma^{-2}_{N-1} ((\mathbf{x}_{N}-\mu)(\mathbf{x}_{N}-\mu)^{T}-\Sigma_{N-1})
$$
Comparing the two formulas we can get know that:
$$
a_{N-1}=\frac{2}{N}\Sigma^{-2}_{N-1}
$$


## Question 5

The posterior distribution should be in the form as follow:
$$
p(\mu|\mathbf{X})\propto p(\mu) \prod^{N}_{n=1}p(x_{n}|\mu,\sigma^{2})
$$
As we have:
$$
p(\mu)=\frac{1}{\sqrt{2\pi}\sigma_{0}}exp\{-\frac{(\mu-\mu_{0})^{2}}{2\sigma^{2}_{0}}\}
\\\\
\prod^{N}_{n=1}p(x_{n}|\mu,\sigma^{2})
=\frac{1}{(2\pi\sigma^{2})^{\frac{N}{2}}}exp\{-\frac{1}{2\sigma^{2}}\sum^{N}_{n=1}(x_{n}-\mu)^{2}\}
$$
We focus on the exponential part of $p(\mu|\mathbf{X})$:
$$
-\frac{(\mu-\mu_{0})^{2}}{2\sigma^{2}_{0}} - \frac{1}{2\sigma^{2}}\sum^{N}_{n=1}(x_{n}-\mu)^{2}
= -\frac{1}{2}(\frac{1}{\sigma^{2}_{0}}+\frac{N}{\sigma^{2}})\mu^{2} 
+ (\frac{\mu_{0}}{\sigma^{2}_{0}}+\frac{1}{\sigma^{2}_{0}}\sum^{N}_{n=1}x_{n})\mu + C
$$
$C$ is independent from $\mu$, so we do not have to care about it.

From the coefficient of $\mu^{2}$, we can get know that:
$$
\frac{1}{\sigma^{2}_{N}} = \frac{1}{\sigma^{2}_{0}}+\frac{N}{\sigma^{2}}
$$
From the coefficient of $\mu$, we can get know that:
$$
\mu_{N}
= \sigma^{2}_{N}(\frac{\mu_{0}}{\sigma^{2}_{0}}+\frac{1}{\sigma^{2}_{0}}\sum^{N}_{n=1}x_{n})
= \frac{\sigma^{2}}{N\sigma^{2}_{0}+\sigma^{2}}\mu_{0} + \frac{N\sigma^{2}_{0}}{N\sigma^{2}_{0}+\sigma^{2}}\mu_{ML}
\\\\
\text{where }\mu_{ML} = \frac{1}{N}\sum^{N}_{n=1}x_{n}
$$



























