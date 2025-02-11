

# Evaluation

\begin{table}[h]
\centering
\caption{Final Results Comparison}
\label{tab:comparison}
\begin{tabular}{llllll}
\rowcolor[HTML]{333333}
{\color[HTML]{FFFFFF} }                                   & {\color[HTML]{FFFFFF} ${E}_{RMS}$} & {\color[HTML]{FFFFFF} ${E}_{M}$} & {\color[HTML]{FFFFFF} ${E}_{MP}$} & {\color[HTML]{FFFFFF} ${E}_{\tilde{x}}$} & {\color[HTML]{FFFFFF} ${\sigma}^{2}$} \\
\cellcolor[HTML]{333333}{\color[HTML]{FFFFFF} kNN}        & 0.7933                      & 0.6289                    & 11.1712                     & 0.5                       & 0.6280                     \\
\cellcolor[HTML]{333333}{\color[HTML]{FFFFFF} Polynomial (Dropped)} &
\cellcolor[HTML]{9AFF99}0.6373   &
0.5035   &
8.9696  &
0.4136  &    
\cellcolor[HTML]{9AFF99}0.4033                      \\
\cellcolor[HTML]{333333}{\color[HTML]{FFFFFF} Polynomial (All)} &
0.6544  &
0.5171  &  
9.2190  &                                                     
0.4353 &  
0.4258                    \\
\cellcolor[HTML]{333333}{\color[HTML]{FFFFFF} Bayesian Polynomial}   &
 0.6423                      &
\cellcolor[HTML]{9AFF99} 0.4984                    &
\cellcolor[HTML]{9AFF99} 8.8329                      &
\cellcolor[HTML]{9AFF99} 0.4036                    &
0.4111                    
\end{tabular}
\end{table}

## Comparison of Models
To fairly compare the performance of the different model's Table \ref{tab:comparison} summarises each models score for each of the five evaluation metrics. The following conclusions can be drawn from this reports analysis:

A Polynomial Regression model is far superior to the kNN. A kNN model is simple to implement but does not return a model and is very resource intensive taking around twice the runtime, therefore were the dataset to grow in size, it may not be computationally feasible. As well as this the kNN model suffers from high variance issues, although it also has a low bias

By varying the degrees of the polynomial basis function, a broad range of functions can be formulated from the data to make predictions. If there is any curvature in the data (as opposed to a step-like relationship by kNN), then a polynomial regression model is better suited to capturing it than the kNN model. However, linear models can suffer from the over-fitting of data. Some methods to mitigate this effect include:

  - Removing features with minimal correlation to the target, improving the model by 2.7\%
  - Holding out data on which to validate the model,
  - applying a Bayesian approach by applying prior values for the mean weights and covariance.

  Applying a Bayesian approach prevents over-fitting by drawing the data back to the targets generated by the initial prior values. As such, bayesian approach performs best on our ${E}_{RMS}$ metric of all our models, as it prevents our prediction values from sticking too closely to the curve generated by the polynomial basis function. Another means of reducing this over-fitting is to reduce the number of features.
  As mentioned above, if this is implemented it increases the accuracy of the model beyond that of the Bayesian Polynomial model approach. This is conclusion is arrived at by using the ${E}_{RMS}$ value as our primary error metric, however if the percentage mean and median error rate are used then the Bayesian Polynomial remains the best model.

  From this we can deduce that although the Bayesian Polynomial regression model performs best when we include all the features within the dataset, if we strategically remove features from the Polynomial Regression model then that model achieves the best accuracy.


