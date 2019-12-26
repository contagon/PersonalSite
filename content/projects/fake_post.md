title: I'm a fake post
date: 2020-11-21
description: Placeholder Post
tags: personal

# Part 3

We also save the death rate and birth rate (you can see them plotted above). Note since we didn't calculate the birth rate for everyone, it's just assumed to be 0 outside of that range.


```python
b_rate = np.zeros(100)
b_rate[15:45] = birth_rate
np.save("birth_rate.npy", b_rate.reshape((1,100)))
d_rates = np.append(d_rates, 1)
np.save("death_rate.npy", d_rates.reshape((1,100)))
```
