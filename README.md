# FX Reinforcement Learning Playground
Beat me in my own game! Develop your RL agent with provided FX environment.

This repository contains an open challenge for a Portfolio Balancing AI in Forex.

The state of the FX market is represented via 512 features in X_train and X_test.

Returns of each asset during training and test periods are in y_train and y_test.

**example.py** contains an implementation, which balances a long-short portfolio.

# Participation into the Forex RL Challenge
Up to 2x leverage is allowed. Your objective is to outperform following risk metrics.

Please send me your saved model so that I can test it on a blind set for the contest.

I will list results from challengers here by sorting them w.r.t. the industry-standard,

risk measures including (but not limited to) the Calmar, Sortino, Omega ratio(s), etc.

Why use my features as environment summary? because they're performing well! 

# Test results that I have obtained myself:
Max. Drawdown: 2.56% 
Sortino Ratio: 13.65x

Sharpe Ratio: 3.99x 
Stability: 96.35% 

Tail Ratio: 2.99x 
Value at Risk: -0.52%

# Annual Cumulative Return
![](annual_return.png)
# Weekly Portfolio Log Return
![](weekly_return.png)


# How you can get rich out of this?
If you obtain successful results and you want to use your RL model for live-trading,

you can contact me for subscribing to a real-time feed of the environment summary.

Your objective results here can also attract business opportunities such as job offers.


# If you are interested in academics:
Are you an academician? Use FX RL challenge to experiment with it in your publication.

You won't need to worry about the right feature extraction, as I already did that for you.

Fixed input for challengers will allow a clear benchmarking of RL methods developed.

Below is a survey paper about reinforcement learning applications in financial markets.

https://www.statistik.rw.fau.de/files/2018/10/12-2018.pdf

# Sponsorships and Hackhathons

Please, contact me if you would like to sponsor the FX RL challenge; or organize a local

meetup, workshop or Hackathon where RL practitioners can participate to this challenge. 

# Bonus: Example Implementation
PyTorch implementation of Multi-processed training of a shared model where batches

are divided across processes. Looking for a person to contribute a TensorFlow version,

as well as an Open AI Gym environment to realistically simulate live-trading conditions.
