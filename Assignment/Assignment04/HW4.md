# Homework Ⅳ

*Course: Machine Learning(CS405) - Professor: Qi Hao*

## Question 1

Show that maximization of the class separation criterion given by $m_2 - m_1 = \mathbf{w^T(m_2 - m_1)}$ with respect to $\mathbf w$, using a Lagrange multiplier to enforce the constraint $\mathbf{w^T w = 1}$, leads to the result that $\mathbf w \propto \mathbf{(m_2 - m_1)}$.

###### Solution:
$$
\text{maximize } \mathbf{w^T(m_2 - m_1)} \text{ with respect to } \mathbf w
\\
\text{subject to } \mathbf{w^T w = 1}
\\
\text{Lagrangian: } \mathcal L(\mathbf w, \lambda) = \mathbf{w^T(m_2 - m_1)} - \lambda(\mathbf{w^T w - 1})
\\
\frac{\partial \mathcal L}{\partial \mathbf w} = \mathbf{m_2 - m_1} - 2\lambda\mathbf w = 0 \to \mathbf w = \frac{1}{2\lambda}(\mathbf{m_2 - m_1}) \text{ which leads to} \mathbf{w} \propto \mathbf{(m_2 - m_1)}
$$

## Question 2

Show that the Fisher criterion

$$
\mathrm J(\mathbf w) = \frac{(m_2 - m_1)^2}{s_1^2 + s_2^2}
$$

can be written in the form

$$
\mathrm J(\mathbf w) = \mathbf{\frac{w^T S_B w}{w^T S_W w}}
$$

**Hint.**
$$
y = \mathbf{w^T x},\qquad
$$

$$
m_k = \mathbf{w^T m_k},\qquad
$$

$$
s_k^2 = \sum_{n\in\mathcal C_k}(y_n - m_k)^2
$$

###### Solution:

​	
$$
J(\mathbf w) = \frac{(m_2 - m_1)^2}{s_1^2 + s_2^2} = \frac{(\mathbf{w^T m_2 - w^T m_1})^2}{\sum_{n\in\mathcal C_1}\mathbf{(y_n - m_1)^2} + \sum_{n\in\mathcal C_2}\mathbf{(y_n - m_2)^2}} 
\\ = \frac{\mathbf{w^T(m_2 - m_1)(m_2 - m_1)^Tw}}{(\sum_{n\in\mathcal C_1}\mathbf{w^T(x_n - \mathbf{m_1})(x_n - \mathbf{m_1})^Tw}) + \sum_{n\in\mathcal C_2}\mathbf{w^T(x_n - \mathbf{m_1})(x_n - \mathbf{m_2})^Tw}}

\\ = \frac{\mathbf{w^T(m_2 - m_1)(m_2 - m_1)^Tw}}{\mathbf{w^T}(\sum_{n\in\mathcal C_1}\mathbf{(x_n - \mathbf{m_1})(x_n - \mathbf{m_1})^T}+ \sum_{n\in\mathcal C_2}\mathbf{(x_n - \mathbf{m_2})(x_n - \mathbf{m_2})^T}) \mathbf{w}}
\\ = \mathbf{\frac{w^T S_B w}{w^T S_W w}}
$$


## Question 3

Consider a generative classification model for $K$ classes defined by prior class probabilities $p(\mathcal C_k) = \pi_k$ and general class-conditional densities $p(\phi|\mathcal C_k)$ where $\phi$ is the input feature vector. Suppose we are given a training data set \{ $\phi_n, \mathbf t_n$ \} where $n = 1, ..., N$, and $\mathbf t_n$ is a binary target vector of length $K$ that uses the 1-of-K coding scheme, so that it has components $t_{nj} = I_{jk}$ if pattern $n$ is from class $\mathcal C_k$.

Assuming that the data points are drawn independently from this model, show that the maximum-likelihood solution for the prior probabilities is given by

$$
\pi_k = \frac{N_k}{N},
$$

where $N_k$ is the number of data points assigned to class $\mathcal C_k$.

###### Solution:
$$
\text{Maximum likelihood solution: } \prod_{n=1}^N p(\phi_n, \mathbf t_n) = \prod_{n=1}^N \prod_{k=1}^K p(\phi_n|\mathcal C_k)^{t_{nk}}p(\mathcal C_k)^{t_{nk}}
\\
\text{Take the logarithm of both sides: } 
\\ \ln \prod_{n=1}^N p(\phi_n, \mathbf t_n) = \sum_{n=1}^N \sum_{k=1}^K t_{nk}\ln p(\phi_n|\mathcal C_k) + \sum_{n=1}^N \sum_{k=1}^K t_{nk}\ln \pi_k
\\
\text{Take derivative with respect to } \pi_k \text{ and set it to zero: } \sum_{n=1}^N \frac{t_{nk}}{\pi_k} = 0
\\ \text{Constraint: } \sum_{k=1}^K \pi_k = 1
\\ \text{Lagrangian: } \mathcal L(\pi_k, \lambda) = \sum_{n=1}^N \sum_{k=1}^K t_{nk}\ln \pi_k + \lambda(\sum_{k=1}^K \pi_k - 1) + \sum_{n=1}^N \sum_{k=1}^K t_{nk}\ln p(\phi_n|\mathcal C_k)
\\
\frac{\partial \mathcal L}{\partial \pi_k} = \frac{N_k}{\pi_k} + \lambda = 0 \to \pi_k = \frac{N_k}{N}
$$


## Question 4

Verify the relation

$$
\frac{\mathrm d\sigma}{\mathrm da} = \sigma(1 - \sigma)
$$

for the derivative of the logistic sigmoid function defined by

$$
\sigma(a) = \frac{1}{1 + \mathrm{exp}(-a)}
$$

###### Solution:
$$
\frac{d \sigma}{da} = \frac{\exp(-a)}{(1+\exp (-a))^2}
\\
\sigma(1-\sigma) = \frac{1}{1+\exp (-a)}(1-\frac{1}{1+\exp (-a)}) = \frac{\exp(-a)}{(1+\exp (-a))^2} = \frac{d \sigma}{da}
$$

## Question 5

By making use of the result

$$
\frac{\mathrm d\sigma}{\mathrm da} = \sigma(1 - \sigma)
$$

for the derivative of the logistic sigmoid, show that the derivative of the error function for the logistic regression model is given by

$$
\nabla \mathbb E(\mathbf w) = \sum^N_{n=1}(y_n - t_n)\phi_n.
$$

**Hint.**

The error function for the logistic regression model is given by

$$
\mathbb E(\mathbf w) = -\mathrm{ln}p(\mathbf{t|w}) = -\sum^N_{n=1}\{t_n\mathrm{ln}y_n + (1 - t_n)\mathrm{ln}(1 - y_n)\}.
$$

###### Solution:
$$
\frac{\partial \mathbb{E}(\mathbf{w})}{y_n} = -\frac{t_n}{y_n} + \frac{1-t_n}{1-y_n} = \frac{t_n(y_n-1)+y_n(1-t_n)}{y_n(1-y_n)} = \frac{y_n-t_n}{y_n(1-y_n)}
\\
\text{Let } a_n = \mathbf{w^T \phi_n}, \text{ then } y_n = \sigma(a_n)
\\ 
\frac{\partial y_n}{\partial a_n} = \sigma(a_n)(1-\sigma(a_n))
\\
\frac{\partial a_n}{\partial \mathbf w} = \phi_n
\\
\frac{\partial \mathbb{E}}{\partial \mathbf w} = \sum_{n=1}^N \frac{\partial \mathbb{E}}{\partial y_n} \frac{\partial y_n}{\partial a_n} \frac{\partial a_n}{\partial \mathbf w} = \sum_{n=1}^N \frac{y_n-t_n}{y_n(1-y_n)} \sigma(a_n)(1-\sigma(a_n)) \phi_n 
\\ = \sum_{n=1}^N \frac{y_n-t_n}{y_n(1-y_n)} y_n(1-y_n) \phi_n
\\ = \sum_{n=1}^N (y_n-t_n) \phi_n
$$

## Question 6

There are several possible ways in which to generalize the concept of linear discriminant functions from two classes to $c$ classes. One possibility would be to use ( $c-1$ ) linear discriminant functions, such that $y_k(\mathbf x )>0$ for inputs $\mathbf{x}$ in class $C_k$ and $y_k(\mathbf{x})<0$ for inputs not in class $C_k$.

By drawing a simple example in two dimensions for $c = 3$, show that this approach can lead to regions of x-space for which the classification is ambiguous.

Another approach would be to use one discriminant function $y_{jk}(\mathbf{x})$ for each possible pair of classes $C_j$ and $C_k$ , such that $y_{jk}(\mathbf{x})>0$ for patterns in class $C_j$ and $y_{jk}(\mathbf{x})<0$ for patterns in class $C_k$. For $c$ classes, we would need $c(c-1)/2$ discriminant functions.

Again, by drawing a specific example in two dimensions for $c = 3$, show that this approach can also lead to ambiguous regions.

###### Solution:
The example in the textbook is shown below:
![1700548157999](image/HW4/1700548157999.png)

## Question 7

Given a set of data points { $\{\mathbf{x}^n\}$ } we can define the convex hull to be the set of points $\mathbf{x}$ given by

$$
\mathbf{x} = \sum_n\alpha_n\mathbf{x}^n
$$

where $\alpha_n>=0$ and $\sum_n\alpha_n=1$. Consider a second set of points $\{\mathbf{z}^m\}$ and its corresponding convex hull. The two sets of points will be linearly separable if there exists a vector $\hat{\mathbf{w}}$ and a scalar $w_0$ such that $\hat{\mathbf{w}}^T\mathbf{x}^n+w_0>0$ for all $\mathbf{x}^n$, and $\hat{\mathbf{w}}^T\mathbf{z}^m+w_0<0$ for all $\mathbf{z}^m$.

Show that, if their convex hulls intersect, the two sets of points cannot be linearly separable, and conversely that, if they are linearly separable, their convex hulls do not intersect.

###### Solution:
If their convex hulls intersect, then there exists $\mathbf{x_0} = \sum_n\alpha_n\mathbf{x}^n = \mathbf{z_0} = \sum_m\beta_m\mathbf{z}^m$, where $\alpha_n>=0$, $\sum_n\alpha_n=1$, $\beta_m>=0$, $\sum_m\beta_m=1$. 

$\forall \mathbf{\hat w}, w_0$, we have $\hat{\mathbf{w}}^T\mathbf{x_0}+w_0 = \hat{\mathbf{w}}^T\mathbf{z_0}+w_0$, which means $\hat{\mathbf{w}}^T\mathbf{x_0} +w_0$ and $\hat{\mathbf{w}}^T\mathbf{z_0}+w_0$ have the same sign. Without loss of generality, we assume $\hat{\mathbf{w}}^T\mathbf{x_0} +w_0 > 0$, then $\hat{\mathbf{w}}^T\mathbf{z_0}+w_0 > 0$, which contradicts with $\hat{\mathbf{w}}^T\mathbf{z_0}+w_0 < 0$.

Therefore, if their convex hulls intersect, the two sets of points cannot be linearly separable.