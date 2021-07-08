import torch
import torch.nn as nn
import torch.optim as optim
# import torch.nn.functional as F

from torch.autograd import Variable

import pandas as pd

def to_var(var, device='cuda:0'):
    if torch.is_tensor(var):
        var = Variable(var)
        if torch.cuda.is_available():
            var = var.cuda(device)
        return var
    if isinstance(var, int) or isinstance(var, float) or isinstance(var, str):
        return var
    if isinstance(var, dict):
        for key in var:
            var[key] = to_var(var[key], device)
        return var
    if isinstance(var, list):
        var = map(lambda x: to_var(x, device), var)
        return var

def stop_gradient(x):
    if isinstance(x, float):
        return x
    if isinstance(x, tuple):
        return tuple(map(lambda y: Variable(y.data), x))
    return Variable(x.data)

def zero_var(sz):
    x = Variable(torch.zeros(sz))
    if torch.cuda.is_available():
        x = x.cuda()
    return x

import plotly.graph_objs as go
from plotly.graph_objs import Scatter
from plotly.graph_objs.scatter import Line
from plotly.subplots import make_subplots
import plotly
import os
import numpy as np

# Plots min, max and mean + standard deviation bars of a population over time
def lineplot(xs, ys_population, title, path='', xaxis='episode', mode='lines'):
  max_colour, mean_colour, std_colour, transparent = 'rgb(0, 132, 180)', 'rgb(0, 172, 237)', 'rgba(29, 202, 255, 0.2)', 'rgba(0, 0, 0, 0)'

  if isinstance(ys_population[0], list) or isinstance(ys_population[0], tuple):
    ys = np.asarray(ys_population, dtype=np.float32)
    ys_min, ys_max, ys_mean, ys_std, ys_median = ys.min(1), ys.max(1), ys.mean(1), ys.std(1), np.median(ys, 1)
    ys_upper, ys_lower = ys_mean + ys_std, ys_mean - ys_std

    trace_max = Scatter(x=xs, y=ys_max, line=Line(color=max_colour, dash='dash'), name='Max')
    trace_upper = Scatter(x=xs, y=ys_upper, line=Line(color=transparent), name='+1 Std. Dev.', showlegend=False)
    trace_mean = Scatter(x=xs, y=ys_mean, fill='tonexty', fillcolor=std_colour, line=Line(color=mean_colour), name='Mean')
    trace_lower = Scatter(x=xs, y=ys_lower, fill='tonexty', fillcolor=std_colour, line=Line(color=transparent), name='-1 Std. Dev.', showlegend=False)
    trace_min = Scatter(x=xs, y=ys_min, line=Line(color=max_colour, dash='dash'), name='Min')
    trace_median = Scatter(x=xs, y=ys_median, line=Line(color=max_colour), name='Median')
    data = [trace_upper, trace_mean, trace_lower, trace_min, trace_max, trace_median]
  else:
    data = [Scatter(x=xs, y=ys_population, line=Line(color=mean_colour), mode=mode)]
  plotly.offline.plot({
    'data': data,
    'layout': dict(title=title, xaxis={'title': xaxis}, yaxis={'title': title})
  }, filename=os.path.join(path, title + '.html'), auto_open=False)


def linesplot(xs, ys_population, legends, title, path='', xaxis='episode', mode='lines', auto_open=False):
    max_colour, mean_colour, std_colour, transparent = 'rgb(0, 132, 180)', 'rgb(0, 172, 237)', 'rgba(29, 202, 255, 0.2)', 'rgba(0, 0, 0, 0)'

    data = []
    for idx, ys in enumerate(ys_population):
        data.append(Scatter(x=xs, y=ys, mode=mode, name=legends[idx]))

    plotly.offline.plot({
    'data': data,
    'layout': dict(title=title, xaxis={'title': xaxis}, yaxis={'title': title})
    }, filename=os.path.join(path, title + '.html'), auto_open=auto_open)

def linesubplot(xs, ys_list1, ys_list2=None, ys_list3=None, legends1=None, legends2=None, legends3=None, title='', subtitles='', rows=1, path='', xaxis='episode', mode='lines', auto_open=False):
    assert len(ys_list1) == len(ys_list2), 'linesubplot Error!'
    fig = make_subplots(rows=rows, cols=1, subplot_titles=subtitles)

    for idx in range(len(ys_list1)):
        fig.add_trace(Scatter(x=xs, y=ys_list1[idx], mode=mode, name=legends1[idx]), row=idx+1, col=1)
        if ys_list2 is not None:
            fig.add_trace(Scatter(x=xs, y=ys_list2[idx], mode=mode, name=legends2[idx]), row=idx+1, col=1)
        if ys_list3 is not None:
            fig.add_trace(Scatter(x=xs, y=ys_list3[idx], mode=mode, name=legends3[idx]), row=idx+1, col=1)

    plotly.offline.plot(fig, filename=os.path.join(path, title + '.html'), auto_open=auto_open)