import seaborn as sns
import time
import matplotlib.pyplot as plt
import numpy as np

def draw_blocking_prob(x1, x2, x1_label, x2_label, x_label, y_label, title='', save=True, name=None):
  sns.set()
  fig, ax = plt.subplots()
  fig.set_size_inches(10, 7)
  ax.plot(x1, color='r', label=x1_label)
  ax.plot(x2, color='b', label=x2_label)
  ax.set_ylabel(y_label)
  ax.set_xlabel(x_label)
  legend = ax.legend(loc='upper left', shadow=True)
  if title != '':
    ax.set_title(title)

  if save:
    image_name = name if name is not None else time.strftime('%a,%d-%b-%Y-%I:%M:%S')
    fig.savefig(f'./results/{image_name}.png')
  return ax, fig


def save_data(data, name):
  path = f'./results/{name}.dat'
  np.array(data).tofile(path)

