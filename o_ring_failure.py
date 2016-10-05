"""
In 1986, the Challenger space shuttle blew up on launch. The problem was
traced to o-ring failures on the solid rocket booster engines. The engines
have three joints. At each joint there are two o-rings, a primary and
secondary o-ring. Failure of the o-rings at one of the joints resulted in
the Challenger tragedy. O-rings fail when two events occur, erosion and
blow-by. These parameters singal o-ring distress and are known to be a
function of temperature. In this analysis, we look at 23 flights preceeding
the Challenger accident and evaluate the probability of a single 
o-ring failure as a function of temperature. We will use a logit model with
temperature as the predictor. Lastly, we will look at the LOOCV error rate.

Prior analysis see:

http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.52.531

looked at predicting the number of o-rings to fail at a given temperature.
The dataset can be found here:

https://archive.ics.uci.edu/ml/machine-learning-databases/space-shuttle/

"""

import pandas as pd
import numpy as np
import statsmodels.api as sm

from scipy import stats
from matplotlib import pyplot as plt

plt.style.use('ggplot')

# Data Import #
###############
# import the erosion file
df = pd.read_csv('data/erosion_only.csv', names=['num_rings', 
                 'num_distressed_rings', 'launch_temp', 
                 'pressure', 'flight_order'], header=None, 
                 delim_whitespace=True)

# Logit Model #
############### 
# We will construct a logit model estimating P(distress|temp) where failure
# is any ring experiencing distress.
temps = sm.add_constant(df.launch_temp.values)
distresses = np.array(df.num_distressed_rings > 0, dtype=bool)

mod = sm.Logit(distresses, temps)
results = mod.fit()
print(results.summary())

# Get Model Confidence Intervals #
##################################
# Statsmodels does not provide the SE of the fit for the logistic model. We
# will need to do that by hand here. The approach is to get the linear
# prediction beta*X and compute the SE of the Linear Model then transform
# under the logit 'link' function to get the SE of the model.

# Create a temperature array to make predictions with
pred_temps = sm.add_constant(np.linspace(min(temps[:,1]), 
                                max(temps[:,1]),200))

std_err = np.array([])
# Recall that Var(y_0|x=x_0) = X_0.T*Sigma*X_0, where X_0=[1,x_0] and sigma 
# is the covariance matrix (see personal notes pg 82). So we compute for
# each temp this matrix product and take sqrt to get SE at y_0.
for temp in pred_temps[:,1]:
    #  Transpose Temp
    temp_arr = np.array([[1,temp]]).T
    # compute Sigma*Temp
    a = np.dot(results.cov_params(), temp_arr)
    # append to standard errors
    std_err = np.append(std_err,np.sqrt(np.dot(temp_arr.T, a)))

# Compute the critical value t_alpha/2,n-1 ~ alpha = 10%
crit_value = stats.t.isf(.1/2,len(df)-1)
# compute the confidnce interval width
widths = crit_value*std_err
# compute the linear fit y_hats by asking predict to return linear=True
linear_fit_vals = results.predict(pred_temps, linear=True)

# constuct upper and lower CIs
ui_linear = linear_fit_vals + widths
li_linear = linear_fit_vals - widths

# Transform the CIs under Logit 'link' function
ui = np.exp(ui_linear)/(1+np.exp(ui_linear))
li = np.exp(li_linear)/(1+np.exp(li_linear))

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(pred_temps[:,1],results.predict(pred_temps),color='k',label='Logit')
ax.plot(pred_temps[:,1], ui, color='r', linestyle='-.', label='90% CI')
ax.plot(pred_temps[:,1], li, color= 'r', linestyle='-.')

# also plot the actual data points from the 23 flights
ax.scatter(temps[:,1], distresses, marker='o', color='b', label='Data')
ax.set_xlabel('Temperature $^\circ$F', fontsize=20)
ax.set_ylabel('O-ring Probability Failure', fontsize=20)
plt.legend(loc='best', prop={'size':15})

# Perform LOOCV #
#################
# Since we have only 23 observations we can compute LOOCV in a loop, for
# larger data sets we should use sklearn.cross_validation
X = temps
y = distresses

# create arrays for polnomial order and error estimates
y_predictions = np.array([])

for obs in range(len(df)): 
    # use list slicing and concatenate to generate a list without obs
    X_train = np.concatenate((X[:obs,], X[obs+1:,]), axis=0)
    y_train = np.concatenate((y[:obs], y[obs+1:]),axis=0)

    # fit the model on the training observation
    result = sm.Logit(y_train, X_train).fit(disp=0)

    # predict distress for the left out obs and append
    y_predictions = np.append(y_predictions, result.predict(X[obs]))

# Compare the y_predictions with the actual distressed rings
y_predictions = (y_predictions > 0.5)
print('LOOCV Error Rate =', np.mean(y_predictions != y))

# place LOOCV rate into plot title
loocv= str(np.round(np.mean(y_predictions != y),3))
ax.annotate(('LOOCV = '+ loocv),(78,.8), fontsize=15)
plt.show()
