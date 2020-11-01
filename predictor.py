import tensorflow as tf
import pandas as pd
import numpy as np
import os

def find(key,dataframe):
  key = key.lower().strip()
  ret = []
  if key[-1] == '*':
    key = key[:-1]
    for i in range(dataframe.shape[0]):
      if key == str(dataframe.iloc[i][0]).lower() or key == str(dataframe.iloc[i][1]).lower():
        ret.append(dataframe.iloc[i])
  else:
    for i in range(dataframe.shape[0]):
      if (key == str(dataframe.iloc[i][0]).lower()) or (key == str(dataframe.iloc[i][1]).lower() and dataframe.iloc[i][0]==0):
        ret.append(dataframe.iloc[i])
    if ret == []:
      for i in range(dataframe.shape[0]):
        if key == str(dataframe.iloc[i][1]).lower():
          ret.append(dataframe.iloc[i])
  return ret

def compute_dif(a):
  r = []
  for i in range(len(a) - 1):
    r.append(a[i + 1] - a[i])
  return np.asarray(r)

def build_orig(b, a):
  r = [b]
  for i in a:
    b += i[0]
    r.append(b)
  return np.asarray(r, int)

def standardize(a, d = None):
  m = a.mean()
  s = a.std()
  if not (m == 0 or s == 0):
    a = (a - m)/s
    if d != None:
      d = (d - m)/s
      return a, d, m, s
    return a, m, s
  else:
    if d != None:
      return a, d, m, s
    return a, m, s

def normalize(s, d = None, k = 0.15):
  f = np.amax(np.absolute(s))
  if d == None:
    r = s / (f + k)
    return r, f
  elif abs(d) > f:
    f = abs(d) + k
  r1 = s / f
  r2 = d / f
  return r1, r2, f

def expand(a):
  r = np.expand_dims(np.asarray(a), -1)
  return r

def choice(feed, model, choice):
  data, mean, std = standardize(compute_dif(feed))
  if choice == 'beta':
    data, factor = normalize(data)
  inp = np.expand_dims(expand(data), 0)
  pred = model.predict(inp)[0][0]
  ret = pred * std + mean + feed[-1]
  if choice == 'beta':
    ret = pred * factor * std + mean + feed[-1]
  return ret

options = {
    'std': 'https://github.com/PerceptronV/Covid-Predictor/raw/main/corona.h5',
    'beta': 'https://github.com/PerceptronV/Covid-Predictor/raw/main/corona_new.h5'
}

sub=os.getcwd()+'/runtime/'
if not os.path.exists(sub):
  os.mkdir(sub)

fname='corona-series.csv'
if os.path.exists(sub+fname):
  os.remove(sub+fname)
csv_path = tf.keras.utils.get_file(
    origin='https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv',
    fname=fname, cache_subdir=sub)
df = pd.read_csv(csv_path).fillna(0)

choice = input('Use beta model? [Y]/[n] ').strip()

if choice == 'Y':
  choice = 'beta'
else:
  choice = 'std'

fname='model.h5'
if os.path.exists(sub+fname):
  os.remove(sub+fname)
model_path = tf.keras.utils.get_file(
    origin=options[choice],
    fname=fname, cache_subdir=sub)
model = tf.keras.models.load_model(model_path)

print('Future predictions\n')
default=['hong kong','united kingdom','china','us']
for inp in default:
  try:
    print(inp)
    col=list(find(inp,df))
    try:
      for i in range(len(col)):
          col[i]=col[i].values[-9:]
      feed=col[0]
      if len(col)>1:
        for i in col[1:]:
          feed=feed+i
      print(feed)
      print(str(predict(feed, model, choice))+'\n')
    except:
      pass
  except:
    print('Error with default list')
while (1):
  inp=input('Place: ')
  if inp is not '':
    col=list(find(inp,df))
    try:
      for i in range(len(col)):
          col[i]=col[i].values[-9:]
      feed=col[0]
      if len(col)>1:
        for i in col[1:]:
          feed=feed+i
      print(feed)
      print(str(predict(feed,model,choice))+'\n')
    except:
      pass
  else:
    break
os.remove(csv_path)
os.remove(model_path)
print('Session terminated')