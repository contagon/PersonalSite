title: Python Snippets
date: 2020-03-12
description: Small Python tricks that shouldn't be forgotten
tags:
  - programming

Every once in a while, I run across a little python code snippet that just makes my day. For future references and to spread the joy, I'm going to start collecting them here. Many of these may be common knowledge to others and there's likely _even better ways_ to do a lot of these things and I recognize that - if you know a better way, please share!

Many of these are also things I got sick of searching for a past file to copy and paste to my new one, so they find their way here instead!

### Making Plots
One of the things I despise most is when I need to label things in `matplotlib` and it takes 6 lines to do so. I made a function that handles it for me:

```python
def pltAttr(x='', y='', title=None, legend='best', save=None):
    plt.tight_layout()
    if legend is not None:
        plt.legend(loc=legend)
    plt.xlabel(x)
    plt.ylabel(y)
    if title is not None:
        plt.title(title)
    if save is not None:
        plt.savefig(save)
```

### Printing in Numpy
I can't stand printing full arrays in numpy, they always print with way too many significant digits, and my fix was always to use `np.round()`. Luckily, there's an easier way than that:

```python
np.set_printoptions(precision=3)
```
This will apply globally to your whole file/notebook. Bonus! If you want to ditch significant digits, use this:
```Python
np.set_printoptions(suppress=True)
```

## 3D Plots
This one isn't particularly helpful, it's 100% just to help me remember this syntax!
```Python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ax = fig.add_subplot(111, projection='3d')
```
## Animating Plots
See above reasoning.
```Python
from matplotlib.animation import FuncAnimation

dot   = ax.scatter(x_data, y_data)
line, = ax.plot(x_data, y_data)

def update(i):
    dot.set_offsets(x_data, y_data)
    line.set_data(x_data, y_data)

a = FuncAnimation(fig, update, frames=N, interval=50, repeat=False)
```
