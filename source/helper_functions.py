#helper_functions.py

#
import numpy as np
import pandas as pd
from sklearn.svm import SVC

#matplotlib imports, to export plots to png images
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import *

def train_svm(motors, training_axes, prediction_axis):
    pd.set_option('display.expand_frame_repr', False)
    print 'training...'
    events_df = get_events(motors)
    events_op = events_df[events_df.state == 'operating'].reset_index(drop=True)
    events_op['time_to_fail'] = None
    i = 0
    while (i < len(events_op)):
        j = i + 1
        while (j < len(events_op)):
            if ((events_op.loc[j, 'Pressure'] == events_op.loc[i, 'Pressure']) and (events_op.loc[j, 'Temp'] == events_op.loc[i, 'Temp'])):
                j += 1
            else:
                break
        events_op.loc[i:j, 'time_to_fail'] = events_op.Time_since_repair[i:j].max()
        i = j
    events_uniq = events_op[training_axes + ['time_to_fail'] ].drop_duplicates()
    x_train = events_uniq[training_axes].values
    y_train = events_uniq['time_to_fail'].values.astype(int)
    clf = SVC(kernel='rbf')
    clf.fit(x_train, y_train)
    print 'accuracy of SVM training = ', clf.score(x_train, y_train)
    Pressure_avg = events_uniq.Pressure.mean()
    Pressure_std = events_uniq.Pressure.std()
    Temp_avg = events_uniq.Temp.mean()
    Temp_std = events_uniq.Temp.std()
    for m in motors:
        #theres gotta be a better way than this...
        m.clf = clf
        m.Pressure_avg = Pressure_avg
        m.Pressure_std = Pressure_std
        m.Temp_avg = Temp_avg
        m.Temp_std = Temp_std
    ##plot trained/predicted %failed versus fail_prob
    #fig = plt.figure(figsize=(8.0, 6.0))
    #ax = fig.add_subplot(1, 1, 1)
    #ax.set_xlabel('Temperature')
    #ax.set_ylabel('Pressure')
    #ax.set_title('%')
    #ax.scatter(x_train[:,0], x_train[:,1], c=-y_train)
    #ax.patch.set_facecolor('lightyellow')
    #ax.grid(True, linestyle=':', alpha=0.3)
    #plotfile = '../data/percent_fail.png'
    #fig.savefig(plotfile)
    #plt.close(fig) 
    #print 'completed plot ' + plotfile
    return x_train, y_train

def get_events(motors):
    events_df = pd.DataFrame()
    for m in motors:
        events_df = events_df.append(pd.DataFrame(m.events))
    return events_df.reset_index(drop=True)
    
def motor_stats(motors):
    events_df = get_events(motors)
    N = events_df.groupby(['maint_type', 'state']).count().unstack()['id'].reset_index()
    N.loc[N.maintenance.isnull(), 'maintenance'] = 0
    N['total'] = N.maintenance + N.operating + N.repair
    N['percent_maint'] = N.maintenance*1.0/N.total
    N['percent_operating'] = N.operating*1.0/N.total
    N['percent_repair'] = N.repair*1.0/N.total
    return N.sort('percent_repair', ascending=False)
