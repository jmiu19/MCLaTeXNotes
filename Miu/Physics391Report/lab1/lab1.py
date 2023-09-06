import pandas as pd
import numpy as np
import os

data_df = pd.read_csv('data/StatisticsLabCollectedData.csv',header=0, index_col=0)
anal_data_df = pd.read_csv('data/StatisticsLabCollectedData.csv',header=0, index_col=0)

means = [np.mean(data_df[column_name].values) for column_name in data_df.columns]
stdev = [np.std(data_df[column_name].values) for column_name in data_df.columns]
vars = [np.var(data_df[column_name].values) for column_name in data_df.columns]
anal_data_df.loc['means'] = means
anal_data_df.loc['stdev'] = stdev
anal_data_df.loc['vars'] = vars

def varSampleMean(N, sig2):
  """
  N: integer, sample of means
  sig2: float, population variance
  """
  return (1/N)*sig2

def coinTheoMean(p, n):
  """
  n: integer, number of trial
  p: float, probability of head
  """
  return p*n

def dieTheoMean(n_f):
  """
  n_f: integer, number of faces on each die
  """
  return n_f-1*1.0

def coinTheoVar(p, n):
  """
  p: float, probability of head
  n: integer, number of trial
  """
  return p*(1-p)*n

def dieTheoVar(n_f):
  """
  n_f: integer, number of faces on each die
  """
  return (n_f*n_f - 1)/6

def coinVarOVar(sig2, N, p):
  """
  sig2: float, population variance
  N: integer, number of trials
  p: float, probability of tossing a head
  """
  return (2*(sig2**2))/(N-1)+((1-6*p*(1-p))*sig2)/N


def dieVarOVar(sig2, n_f, N):
  """
  n_f: integer, number of faces on each die
  sig2: float, population variance
  N: integer, number of trials
  """
  return (2*sig2*sig2)/(N-1) - ((n_f*n_f+1)*sig2)/(10*N)


def saltVarOVar(sig2, N):
  """
  sig2: float, population variance
  N: integer, number of trials
  """
  return (2*sig2*sig2)/(N-1) + (sig2/N)


diffMeansKCLsNACL = (anal_data_df.loc['means']['KCl']-anal_data_df.loc['means']['NaCl'])

def plotHist(x_name, bins_arr, xlabel, ylabel, tit, pos,
             Gau=False, GauTMean=0, GauTVar=0, GauRange=np.arange(0, 10, 0.1) ):
  """
  x_name: string, name of data to be plot
  bins_arr: array, bins numbers
  xlabel: string, x-axis label
  ylabel: string, y-axis label
  tit: string, title of the plot
  pos: float, horizontal position (in fraction) of the annotation
  Gau: boolean, wheter overplot the gaussian distribution
  """
  x = data_df[x_name]
  n, bins, patches = plt.hist(x,bins=bins_arr,histtype='bar', align='left')
  plt.xlabel(xlabel,fontsize='xx-large')
  plt.ylabel(ylabel, fontsize='xx-large')
  plt.title(tit, fontsize='xx-large')
  plt.annotate('mean=%.2f'%anal_data_df.loc['means'][x_name], (pos,0.8),
               xycoords='figure fraction', fontsize='x-large')
  plt.annotate('variance=%.2f'%anal_data_df.loc['vars'][x_name], (pos,0.75),
               xycoords='figure fraction', fontsize='x-large')
  if Gau==True:
    x_data = bins[0:len(bins)-1]
    x_data = x_data + 0.5*(x_data[1] - x_data[0])
    x_gaussian = GauRange
    sigma_gaussian_curve = np.sqrt(GauTVar)
    mean_gaussian_curve = GauTMean
    amplitude = np.max(n)
    params_histogram, params_covariance = opt.curve_fit(gaussian_model, x_data, n, p0=[5,4,10])
    plt.plot(x_gaussian, gaussian_model(x_gaussian, mean_gaussian_curve,sigma_gaussian_curve,amplitude),'r',ls=':')
    plt.show()


from prettytable import PrettyTable

data_Tab = PrettyTable(["Experiment", "Sample Mean",
                        "Sample Variance", "Population Mean",
                        "Population Variance"],
                       digits=3, round=True)
data_Tab.add_row(["Number of head in 10 flips",
                  anal_data_df.loc['means']['Coins'].round(2),
                  anal_data_df.loc['vars']['Coins'].round(2),
                  round(coinTheoMean(0.5, 10),2),
                  round(coinTheoVar(0.5,10),2)])
data_Tab.add_row(["Sum of two dice",
                 anal_data_df.loc['means']['Dice'].round(2),
                 anal_data_df.loc['vars']['Dice'].round(2),
                 round(dieTheoMean(6),2),
                 round(dieTheoVar(6),2)])
data_Tab.add_row(["Count for KCl",
                  anal_data_df.loc['means']['KCl'].round(2),
                  anal_data_df.loc['vars']['KCl'].round(2),'-','-'])
data_Tab.add_row(["Count for NaCl",
                  anal_data_df.loc['means']['NaCl'].round(2),
                  anal_data_df.loc['vars']['NaCl'].round(2),'-','-'])

data_var_Tab = PrettyTable(["Experiment",
                            "Variance of Sample Mean (using population variance)",
                            "Variance of Sample Variance (using population variance)",
                            "Variance of Sample Mean (using sample variance)",
                            "Variance of Sample Variance (using sample variance)", ], digits=3, round=True)
data_var_Tab.add_row(
        ["Number of head in 10 flips",
         round(varSampleMean(50,coinTheoVar(0.5,10)),2),
         round(coinVarOVar(coinTheoVar(0.5,10), 50, 0.5),2),
         round(varSampleMean(50,anal_data_df.loc['vars']['Coins']),2),
         round(coinVarOVar(anal_data_df.loc['vars']['Coins'], 50, 0.5),2),])
data_var_Tab.add_row(
        ["Sum of two dice",
         round(varSampleMean(50,dieTheoVar(6)),2),
         round(dieVarOVar(dieTheoVar(6), 6, 50),2),
         round(varSampleMean(50,anal_data_df.loc['vars']['Dice']),2),
         round(dieVarOVar(anal_data_df.loc['vars']['Dice'], 6, 50),2)])
data_var_Tab.add_row(
        ["Count for KCl",
         '-',
         '-',
         round(varSampleMean(50,anal_data_df.loc['vars']['KCl']),2),
         round(saltVarOVar(anal_data_df.loc['vars']['KCl'], 50),2)])
data_var_Tab.add_row(
        ["Count for NaCl",
         '-',
         '-',
         round(varSampleMean(50,anal_data_df.loc['vars']['NaCl']),2),
         round(saltVarOVar(anal_data_df.loc['vars']['NaCl'], 50),2)])
