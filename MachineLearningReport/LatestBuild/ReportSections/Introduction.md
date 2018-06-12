
# Introduction

This report evaluates the performance of different regression models at predicting the quality of a variety of Portuguese wines. A dataset has been used which describes red wine via 11 different features, including alcohol content, fixed acidity and quality. The efficacy of different machine learning models in predicting the quality of a wine given the other features has been assessed in this report.  

This report discusses three regression models that were created to predict wine quality. This includes two linear basis function models; polynomial and Bayesian as well as k-NN. These models were selected to sample a selection of some of the most popular Linear Regression models. By optimising each model, a fair comparison of each models efficacy was made using the same scoring criteria.

## Performance Evaluation

The metrics used to evaluate the performance of each models are \cite{Howtocom28:online}:

- Root mean square error (${E}_{RMS}$): this is a measure of prediction accuracy. ${E}_{RMS}$ disproportionately affects the points further from the actual results, therefore favouring predictions with small variance; ensuring that the metric is more sensitive to outlier values. ${E}_{RMS}$  was used as the primary metric to assess a models performance. (${E}_{RMS}$) is described by:

\begin{align}{ E }_{ RMS }=\sqrt { \frac{ 2E({ { \mathbf{w}} }^{ * })}{ N }}
\end{align}

- Mean absolute error (${ E }_{ M }$): this is a measure of the actual distance of the error from the prediction. This is less sensitive to large errors than   ${E}_{RMS }$ but gives a clear measure of how far predictions on average deviate from the target.
- Mean absolute percentage error (${ E }_{ M \% }$): presenting mean absolute error as a percentage can help gauge the performance of the model more conceptually, helping to gauge how big the error is
- Median absolute error (${ E }_{ \tilde { x }  }$): This is the middle error value within the range of predictions. This measure is useful as it is unaffected by outliers giving a picture of the general accuracy of the model
- Variance (${ \sigma }^{ 2 }$): this measure highlights the precision marked by the spread of predictions. A low variance indicates that predictions are all made in a similar region with fewer outliers. However a model could have some issues with its accuracy; variance, therefore, must be used in conjunction with a mean value  

A function named `error_score.py`, was created which analysed predictions, making sure all the models were evaluated equally; enabling direct comparisons to be made.

## Cross Validation

To maximise the effectiveness of the limited dataset models were trained, tested and validated using K-fold cross validation. This method separates the data into K separate folds, of which K-1 are training folds and one is the testing fold. The model is then trained and tested K times, each time using a different fold for testing. We then take the average error values of all the folds. This ensures we reduce our bias (more fitting data) and variance (more validation data) as all of the data becomes both training and testing data.

\begin{table}[h]
\centering
\caption{Example showing data splitting for 5-Fold cross validation}
\label{tab:validation}
\begin{tabular}{ccccccc}
\cline{1-6}
\multicolumn{1}{l}{\cellcolor[HTML]{00171F}{\color[HTML]{FFFFFF} }} & \cellcolor[HTML]{00171F}{\color[HTML]{FFFFFF} fold 1} & \cellcolor[HTML]{00171F}{\color[HTML]{FFFFFF} fold 2} & \cellcolor[HTML]{00171F}{\color[HTML]{FFFFFF} fold 3} & \cellcolor[HTML]{00171F}{\color[HTML]{FFFFFF} fold 4} & \cellcolor[HTML]{00171F}{\color[HTML]{FFFFFF} fold 5} & \multicolumn{1}{l}{} \\ \cline{1-6}
\multicolumn{1}{|c|}{\cellcolor[HTML]{00171F}{\color[HTML]{FFFFFF} iteration 1}} & \multicolumn{1}{c|}{\cellcolor[HTML]{FFCCC9}test} & \multicolumn{1}{c|}{\cellcolor[HTML]{9AFF99}train} & \multicolumn{1}{c|}{\cellcolor[HTML]{9AFF99}train} & \multicolumn{1}{c|}{\cellcolor[HTML]{9AFF99}train} & \multicolumn{1}{c|}{\cellcolor[HTML]{9AFF99}train} &  \\ \cline{1-6}
\multicolumn{1}{|c|}{\cellcolor[HTML]{00171F}{\color[HTML]{FFFFFF} iteration 2}} & \multicolumn{1}{c|}{\cellcolor[HTML]{9AFF99}train} & \multicolumn{1}{c|}{\cellcolor[HTML]{FFCCC9}test} & \multicolumn{1}{c|}{\cellcolor[HTML]{9AFF99}train} & \multicolumn{1}{c|}{\cellcolor[HTML]{9AFF99}train} & \multicolumn{1}{c|}{\cellcolor[HTML]{9AFF99}train} &  \\ \cline{1-6}
\multicolumn{1}{|c|}{\cellcolor[HTML]{00171F}{\color[HTML]{FFFFFF} iteration 3}} & \multicolumn{1}{c|}{\cellcolor[HTML]{9AFF99}train} & \multicolumn{1}{c|}{\cellcolor[HTML]{9AFF99}train} & \multicolumn{1}{c|}{\cellcolor[HTML]{FFCCC9}test} & \multicolumn{1}{c|}{\cellcolor[HTML]{9AFF99}train} & \multicolumn{1}{c|}{\cellcolor[HTML]{9AFF99}train} &  \\ \cline{1-6}
\multicolumn{1}{|c|}{\cellcolor[HTML]{00171F}{\color[HTML]{FFFFFF} iteration 4}} & \multicolumn{1}{c|}{\cellcolor[HTML]{9AFF99}train} & \multicolumn{1}{c|}{\cellcolor[HTML]{9AFF99}train} & \multicolumn{1}{c|}{\cellcolor[HTML]{9AFF99}train} & \multicolumn{1}{c|}{\cellcolor[HTML]{FFCCC9}test} & \multicolumn{1}{c|}{\cellcolor[HTML]{9AFF99}train} &  \\ \cline{1-6}
\multicolumn{1}{|c|}{\cellcolor[HTML]{00171F}{\color[HTML]{FFFFFF} iteration 5}} & \multicolumn{1}{c|}{\cellcolor[HTML]{9AFF99}train} & \multicolumn{1}{c|}{\cellcolor[HTML]{9AFF99}train} & \multicolumn{1}{c|}{\cellcolor[HTML]{9AFF99}train} & \multicolumn{1}{c|}{\cellcolor[HTML]{9AFF99}train} & \multicolumn{1}{c|}{\cellcolor[HTML]{FFCCC9}test} &  \\ \cline{1-6}
\multicolumn{1}{l}{} & \multicolumn{1}{l}{} & \multicolumn{1}{l}{} & \multicolumn{1}{l}{} & \multicolumn{1}{l}{} & \multicolumn{1}{l}{} & \multicolumn{1}{l}{} \\ \cline{1-1} \cline{7-7}
\multicolumn{1}{|l|}{\cellcolor[HTML]{333333}{\color[HTML]{FFFFFF} validation phase}} & \multicolumn{1}{l}{} & \multicolumn{1}{l}{} & \multicolumn{1}{l}{} & \multicolumn{1}{l}{} & \multicolumn{1}{l|}{} & \multicolumn{1}{l|}{\cellcolor[HTML]{68CBD0}validation} \\ \cline{1-1} \cline{7-7}
\end{tabular}
\end{table}

A initial validation dataset was held out from the cross-validation phase, containing 10% of the original data. This means K-fold cross validation was performed on the other 90% of the data. Table \ref{tab:validation} illustrates of how the data is split for a K of 5.

