# Assignment 03

#### Q1

$$
\text{take }R=diag(r_{1},r_{2}, ... , r_{N}) \\

\text{So we turn the error function to }
E_{D}(\mathbf{w})
= \frac{1}{2}(\mathbf{t}-\mathbf{\Phi\mathbf{w}})^{\rm{T}} R
(\mathbf{t}-\mathbf{\Phi\mathbf{w}}) \\
\text{let } \frac{\partial E_{D}(\mathbf{w})}{\partial \mathbf{w}} = 0 \\
\text{we get } \mathbf{w}^{*}
= (\Phi^{\rm T} \mathbf{R} \Phi)^{-1}\Phi^{\rm T}\mathbf{R}\mathbf{t}
$$



Data dependent noise variance:

The factor $r_{n}$ can be regarded as the precision parameter of the data point $(\mathbf{x}_{n},t_{n})$,  which replaces $\beta$. The larger  $r_{n}$ means the less noise of the data point.

Replicated data point:

The factor $r_{n}$ can be regarded as the effective number of the data point $(\mathbf{x}_{n},t_{n})$, if the data point repeats in the dataset for multiple times, we can use $r_{n}$ to represent it.

#### Q2

$$
p(\mathbf{w}, \beta|\mathbf{t})
\varpropto p(\mathbf{w},\beta)p(\mathbf{t}|\mathbf{X},{\rm w},\beta) 
\\\\
\mathrm{ln}p(\mathbf{w}, \beta|\mathbf{t}) = 
-\frac{\beta}{2}(\mathbf{w}-\mathbf{m}_{0})^{\mathrm{T}}\mathbf{S^{-1}_{0}}(\mathbf{w}-\mathbf{m}_{0})
- b_{0}\beta + (a_{0}-1)\mathrm{ln}\beta 
+ \frac{N}{2}\mathrm{ln}\beta 
- \frac{\beta}{2} \sum^{N}_{n=1}(\mathbf{w}^{\mathrm{T}}\phi(\mathrm{x}_{n}) - t_{n})^{2}
+ const
$$



As we have $p(\mathbf{w}, \beta|\mathbf{t})\varpropto p(\mathbf{w}| \beta,\mathbf{t})p(\beta|\mathbf{t}) $, 

because only $p(\mathbf{w}| \beta,\mathbf{t})$ is dependent to $\mathbf{w}$:
$$
\mathrm{ln}p(\mathbf{w}| \beta,\mathbf{t}) = 
- \frac{\beta}{2}\mathbf{w}^{\mathrm{T}}(\mathbf{\Phi}^{\mathrm{T}}\mathbf{\Phi}+\mathbf{S^{-1}_{0}})\mathbf{w}
+ \mathbf{w}^{\mathrm{T}}(\beta\mathbf{S^{-1}_{0}}\mathbf{m}_{0} + \beta\mathbf{\Phi}^{\mathrm{T}}\mathbf{t})
+ const
$$


So $p(\mathbf{w}|\beta,\mathbf{t})$ is the Gaussian distribution part of $p(\mathbf{w}, \beta|\mathbf{t})$, and that
$$
\mathbf{S}_N = \mathbf{S^{-1}_{0}}\mathbf{m}_{0} + \mathbf{\Phi}^{\mathrm{T}}\mathbf{t}
\\\\
\mathbf{m}_N = \mathbf{S}_N (\mathbf{\Phi}^{\mathrm{T}}\mathbf{\Phi}+\mathbf{S^{-1}_{0}})
$$


And for $p(\beta|\mathbf{t})$:
$$
\mathrm{ln}p(\beta|\mathbf{t}) =
-\frac{\beta}{2}\mathbf{m}_{0}^{\mathrm{T}}\mathbf{S^{-1}_{0}}\mathbf{m}_{0}
+\frac{\beta}{2}\mathbf{m}_{N}^{\mathrm{T}}\mathbf{S^{-1}_{N}}\mathbf{m}_{N}
- b_{0}\beta + (a_{0}-1)\mathrm{ln}\beta + \frac{N}{2}\mathrm{ln}\beta
- \frac{\beta}{2} \sum^{N}_{n=1}t_{n}^{2}
+ const
$$


So $p(\beta|\mathbf{t})$ is the Gamma distribution part of $p(\mathbf{w}, \beta|\mathbf{t})$, and that:
$$
a_{N} = a_{0} + \frac{N}{2}
\\\\
b_{N} = a_{0} + \frac{1}{2}
(\mathbf{m}_{0}^{\mathrm{T}}\mathbf{S^{-1}_{0}}\mathbf{m}_{0} 
- \mathbf{m}_{N}^{\mathrm{T}}\mathbf{S^{-1}_{N}}\mathbf{m}_{N}
+ \sum^{N}_{n=1}t_{n}^{2})
$$

#### Q3

As we have (3.80):
$$
E(\mathbf{w})=
E(\mathbf{m}_{N})
+ \frac{1}{2}(\mathbf{w}-\mathbf{m}_{N})^{\mathrm{T}}\mathbf{A}(\mathbf{w}-\mathbf{m}_{N})
$$


The second part is actually the exp part of a Gaussian, so that:
$$
\int\exp\{-E(\mathbf{w})\}\mathrm{d}\mathbf{w}
= \exp\{-E(\mathbf{m}_{N})\} 
\int\exp \frac{1}{2}(\mathbf{w}-\mathbf{m}_{N})^{\mathrm{T}}\mathbf{A}(\mathbf{w}-\mathbf{m}_{N})\mathrm{d}\mathbf{w} \\
= \exp\{-E(\mathbf{m}_{N})\} (2\pi)^{\frac{M}{2}}|\mathbf{A}|^{-\frac{1}{2}}
$$


As we have (3.78):
$$
p(\mathbf{t}|\alpha,\beta) = 
(\frac{\beta}{2\pi})^{\frac{N}{2}} (\frac{\alpha}{2\pi})^{\frac{M}{2}}
\int\exp\{-E(\mathbf{w})\}\mathrm{d}\mathbf{w}
\\\\
\mathrm{ln}p(\mathbf{t}|\alpha,\beta) =
\frac{M}{2}\ln\alpha + \frac{N}{2}\ln\beta - E(\mathbf{m}_N) - \frac{1}{2}\ln|\mathbf{A}| - \frac{N}{2}\ln(2\pi)
$$


#### Q4

$$
\text{Log likelihood function: }
L(a) = \ln\prod^{n}_{i=1}p(Y_{i}|X_{i},a)
= -\frac{n}{2}\ln(2\pi) -n\ln\sigma
-\frac{1}{2\sigma^{2}}\sum^{n}_{i=1}(Y_{i}-aX_{i})^2
\\\\
\frac{\partial L(a)}{\partial a} =
-\frac{1}{2\sigma^{2}}\sum^{n}_{i=1}(2X_{i}^{2}a - 2X_{i}Y_{i}) = 0
\\\\
a_{ML} = \frac{\sum^{n}_{i=1}X_{i}Y_{i}}{\sum^{n}_{i=1}X_{i}^{2}}
$$



#### Q5

$$
\text{Log likelihood function: }
L(\theta) = \ln\prod^{n}_{i=1}p(y_{i}|\theta)
= \sum^{n}_{i=1}(y_{i}\ln\theta-\theta-\ln(y_{i}!))
$$



#### Q6

$$
\text{Log likelihood function: }
L(\lambda) = \ln\prod^{n}_{i=1}p(X_{i}|\lambda,a)
= \ln \prod^{n}_{i=1} \frac{1}{\Gamma(\alpha)}\lambda^{\alpha}X_{i}^{\alpha-1}e^{-\lambda X_{i}}
\\
=-n\ln\Gamma(\alpha) + n\alpha\ln\lambda + (\alpha-1)\sum^{n}_{i=1}\ln X_{i} - \lambda\sum^{n}_{i=1}X_{i}
\\\\
\frac{\partial L(\lambda)}{\partial\lambda} =
\frac{n\alpha}{\lambda} - \sum^{n}_{i=1}X_{i} = 0
\\\\
\lambda_{ML} = \frac{n\alpha}{\sum^{n}_{i=1}X_{i}}
$$

