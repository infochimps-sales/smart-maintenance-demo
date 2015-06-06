#helper_functions.py

#get imports
import numpy as np
import pandas as pd
from sklearn.svm import SVC

def get_events(motors):
    events_df = pd.DataFrame()
    for m in motors:
        events_df = events_df.append(pd.DataFrame(m.events))
    return events_df.reset_index(drop=True)

def train_svm(motors, training_axes, prediction_axis):
    pd.set_option('display.expand_frame_repr', False)
    print '...training...this portion of code is inefficient, needs work'
    xy_train = pd.DataFrame()
    for m in motors: 
        xy_train = xy_train.append(m.get_training_dataframe())
    x_avg = {col:xy_train[col].mean() for col in training_axes}
    x_std = {col:xy_train[col].std()  for col in training_axes}
    for col in training_axes:
        xy_train[col + '_norm'] = (xy_train[col] - x_avg[col])/x_std[col]
    training_axes_norm = [col + '_norm' for col in training_axes]
    x_train = xy_train[training_axes_norm].values
    y_train = xy_train['time_to_fail'].values.astype(int)
    clf = SVC(kernel='rbf')
    clf.fit(x_train, y_train)
    print '...accuracy of SVM training = ', clf.score(x_train, y_train)
    for m in motors:
        #theres gotta be a better way than this...
        m.clf = clf
        m.x_avg = x_avg
        m.x_std = x_std
    return xy_train

def motor_stats(motors):
    events_df = get_events(motors)
    N = events_df.groupby(['maint_type', 'state']).count().unstack()['id'].reset_index()
    N.loc[N.maintenance.isnull(), 'maintenance'] = 0
    N['total'] = N.maintenance + N.operating + N.repair
    N['percent_maint'] = N.maintenance*1.0/N.total
    N['percent_operating'] = N.operating*1.0/N.total
    N['percent_repair'] = N.repair*1.0/N.total
    return N.sort('percent_repair', ascending=False)
