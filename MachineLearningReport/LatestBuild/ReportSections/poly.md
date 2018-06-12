# Linear Basis Polynomial Regression Model

\begin{table}[h]
\centering
\caption{Polynomial Model Error Scores With Dropped Low Scoring Features}
\label{tab:compdrop}
\begin{tabular}{|l|l|l|l|l|l|}
\hline
\rowcolor[HTML]{00171F}
\multicolumn{1}{|c|}{\cellcolor[HTML]{00171F}{\color[HTML]{FFFFFF} Features}} &
\multicolumn{1}{c|}{\cellcolor[HTML]{00171F}{\color[HTML]{FFFFFF} ${E}_{RMS}$}} & \multicolumn{1}{c|}{\cellcolor[HTML]{00171F}{\color[HTML]{FFFFFF} ${E}_{M}$}} & \multicolumn{1}{c|}{\cellcolor[HTML]{00171F}{\color[HTML]{FFFFFF} ${E}_{\tilde{x}}$}} & \multicolumn{1}{c|}{\cellcolor[HTML]{00171F}{\color[HTML]{FFFFFF} ${E}_{MP}$}} & \multicolumn{1}{c|}{\cellcolor[HTML]{00171F}{\color[HTML]{FFFFFF} ${\sigma}^{2}$}} \\ \hline
\textit{Dropped} &
0.6373                                                               & 0.5035                                                               & 0.4136                                                             & 8.9696                                                           & 0.4033                                                         \\ \hline
{\textit{All}} &
0.6544                                                               & 0.5171                                                               & 0.4353                                                             & 9.2190                                                           & 0.4258                                                           \\ \hline
\textit{\% Change} &
2.7\% &
2.7\%	&
5.2\%	&
2.8\%	&
5.6\% \\ \hline
\end{tabular}
\end{table}


\begin{align*}
\textit{Degree Factor} = 3,
\textit{$\lambda$} = 0    
\end{align*}


## Preliminary Linear Regression Analysis
\begin{figure}[H]
    \centering
    \includegraphics[trim = 0 0 0 0, clip, width=0.85\textwidth]{lin_reg.pdf}
    \caption{Comparison of correlation of each feature against wine quality}
    \label{fig:linearreg}
  \end{figure}
To gain a general understanding of the relationship between each feature and its effect on wine quality, scatter plots comparing each feature were created; providing a simple understanding of the data relationship. Through calculating the regression values between features and the label (wine quality), parameters which may distort a linear regression model were highlighted. Figure \ref{fig:linearreg} gives an overview of this some of the strongest and weakest relationships. It is worth noting that pair wise correlation cannot be visualised using this method, meaning simply dropping features of low correlation may lead to decreased model performance.

## Overview

A linear basis model is a method of supervised machine learning, that takes a series of features and assumes that a (basis) function can be applied to these features to predict a target \cite{bishop2006machine}. These basis functions are summed together to create a linear combination of function, given it the *linear* name. A linear model can be generalised as the following:

\begin{align}
    y\left( x \right) ={ w }_{ 0 }\quad +\quad { w }_{ 1 }{ \phi  }_{ 1 }\left( { x }_{ 1 } \right) \quad +\quad { w }_{ 2 }{ \phi  }_{ 2 }\left( { x }_{ 2 } \right) +...+{ w }_{ n }{ \phi  }_{ n }\left( { x }_{ n } \right) \\ where\quad { \phi  }_{ n }\left( { x }_{ n } \right) = n \text{ basis functions }
    \label{eq:linearbasis}
\end{align}

Equation \ref{eq:linearbasis} is a generalisation of linear regression that essentially replaces each input with a function of the input. In the case where the basis function is the identity matrix, the model becomes just linear regression. The type of basis functions (i.e. the function $\phi$) is chosen to model the non-linearity in the relationship between the inputs and targets \cite{MachineL13:online}.

For this model, a polynomial basis function evaluated, that can be defined as:

\begin{align}
    { \phi  }_{ j }\left( x \right) ={ x }_{ j }
    \label{eq:polybasisfunc}
\end{align}

Different polynomial models can be tested by varying the degree factor ($j)$. The degree factor could be optimised by looping through a range to find the factor which minimised ${E}_{RMS}$. It is worth noting that at a factor of 1, the polynomial model is evaluating a basic linear regression model. As a second factor to optimise, the model ridge regression was applied to the model's weights. The regularisation coefficient ($\lambda$) is used to penalise weights terms with large values, smoothing out the curve of by reducing peaks.

After running the model for all methods, some experimentation into dropping certain features was made to see if the prediction could be improved. Analysing figure \ref{fig:linearreg} the 3 features with the weakest relationship were dropped from the model.

## Method
- Use the `cross_validation` function to reserve 10% of the data for the validation phase. This 10% will validate the model's performance on unseen data after all training and testing is complete
- Split the remaining data across a series of 10 folds to cross-validate the model using the `cv_folds` function
- Each fold is run through the polynomial training model using the function `cv_evaluation_poly_model`, this extracts the data in each fold splitting into training, and testing sets runs the polynomial model and stores the results in a dictionary for later evaluation
- The `polynomial` function, takes training and testing data iterates through all degree factors within a range, and then each regression coefficient to find the optimum settings for the polynomial model. To find the ranges of both variables a variety of well-spread points were picked to test the model's sensitivity; refining these ranges to capture the minima in results. Ranges of degrees($1 - 5$) and $\lambda(0-50)$ were chosen. The `polynomial` function can be further broken down into the following steps:
  1. `expand_to_2Dmonomials` takes the input matrix and expands this expands out the columns of the input matrix multiplying by the factorial of the polynomial degree. i.e. for a degree factor of 2 the matrix gets expanded from $x$ to $1, x, x^2$
  2. The function `regularised_ml_weights` is then used to create a unique weight for every polynomial factor (column) in the matrix; i.e. a degree factor of 2 with 11 features would be converted into 23 different weights. The function also regularises the weights, subtracting $\lambda$ using equation \ref{eq:polybasisfunc}; helping smooth out the function. A value of $\lambda = 0$ is equivalent to no regularisation coefficient
  4. `construct_3dpoly` sets a prediction function taking the inputs of the regularised weights and the degree factors for the polynomial basis function
  5.  (`prediction_function`) is used to create a target prediction for a series of inputs based on the previous parameters of the model.  `expand_to_2Dmonomials`  is reused to multiply each value by the basis function $x^{j}$
  6. Finally `polynomial` collects the error values using the `error_score` function and the weights analysing the performance of the model
- For all data folds, repeat steps $1-6$. On completion, all the different cross-validation results are aggregated to produce a final error score for the model
- The weights corresponding to the optimum degree factor and $\lambda$ are aggregated to create a final model for validation.
- The final model is given new testing data (reserved at the start of the method) to validate the model efficacy

## Results

\begin{figure}[H]
\centering
\begin{minipage}{.49\textwidth}
  \centering
  \includegraphics[trim = 0 0 0 0, clip, width=1\textwidth]{RMSvsdeg.pdf}
 \caption{$E_{RMS}$ Variation with Change in Polynomial Degree, $\lambda = 0$ }
 \label{fig:degpoly}
\end{minipage}
\hfill
\begin{minipage}{.49\textwidth}
  \centering
   \includegraphics[trim = 0 0 0 0, clip, width=1\textwidth]{RMSvsreg.pdf}
   \caption{$E_{RMS}$ Variation with Regression Coefficient ($\lambda$) where $Degree = 3$}
  \label{fig:regpoly}
\end{minipage}
\vspace{-20pt}
\end{figure}

### Discussion
Figures \ref{fig:degpoly} and \ref{fig:regpoly} both describe a snapshot of the sensitivity which the polynomial model has to changes in the degree factor and $\lambda$; this is because the model iterates through every combination of degree and $\lambda$ to find the perfect combination. On first observation of the the performance of the Polynomial model, it was witnessed that there was a significant improvement in accuracy when using a polynomial basis function over a basic linear function. Figure \ref{fig:degpoly} shows a parabolic shape that reaches its minimum point at a degree factor of **3** with a slight increase in error for both 2 and 4. The model's sensitivity between 2 and 4 can also be observed when comparing the optimum models across different folds.  Both the model's variables alter these three values and their corresponding optimum regression coefficient. The models sensitivity therefore eludes to requiring more data required generates a stable prediction across the folds.

Through holding out data in the validation phase, overfitting could be avoided due to comparing the model to data which has been included in the training phase. The prediction made in during the validation test is slightly worse than the value observed during the validation phase. This again is likely due to the variance in the data.  

By using the regression relationship between a feature and its effect on quality highlighted in figure \ref{fig:linearreg}, the features residual sugar, free sulphur dioxide and pH were dropped all having an $\left| {r}^{2} \right| < 0.1$. Dropping more features appeared to have a negative effect on the models prediction. Originally, the polynomial model would select a value of $\lambda = 0.22$, suggesting that the model was overfitting to some of the data-points. By removing these features $\lambda$ dropped to 0, suggesting the model was naturally fitting the data more smoothly. Table \ref{tab:compdrop} shows the positive effect that dropping these features has on the model.


