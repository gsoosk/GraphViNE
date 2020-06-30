import seaborn as sns
import time
import matplotlib.pyplot as plt
import numpy as np

def draw_blocking_prob(x, labels, colors, x_label, y_label, title='', save=True, name=None):
  sns.set()
  fig, ax = plt.subplots()
  fig.set_size_inches(10, 7)
  for i in range(len(x)):
    ax.plot(x[i], color=colors[i], label=labels[i])
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

