#smart_maint.py

#imports
import numpy as np
import pandas as pd
from motor import Motor

#parameters for scheduled maintenance
fail_prob_rate = 0.01
maint_interval = 10
maint_duration = 2
repair_duration = 5

#
rn_seed = 13
np.random.seed(rn_seed)

#create motor
maint_type = 'run-to-fail'
time_previous_maint = 0
motor_id = 1001
m = Motor(motor_id, maint_type, fail_prob_rate, maint_interval, maint_duration, repair_duration)

#run motor using scheduled maintenance
time_final = 2000
for t in np.arange(0, time_final, 1):
    m.operate(t)

#widen display for dataframes
pd.set_option('display.expand_frame_repr', False)

#run motor using predictive maintenance
m.maint_type = 'predictive'

#generate training data
events = m.events.copy(deep=True)
idx = events[ (events.state == 'failed') & (events.just_failed == True) ].index.max()
events = events[0:idx + 1]
events['time_to_fail'] = float(np.nan)
events.loc[events.state != 'operating', 'time_to_fail'] = -1
events_shift = events.shift(periods=-1)
while(events.time_to_fail.isnull().sum() > 0):
    idx = (events.state == 'operating') & (events_shift.just_failed == True)
    events.loc[idx, 'time_to_fail' ] = events_shift[idx].time - events[idx].time
    events_shift = events_shift.shift(periods=-1)

events = events[events.state == 'operating'].sort(columns='fail_prob')
x = events.fail_prob.values
x = x.reshape((len(x),1))
x_avg = x.mean()
x_std = x.std() 
x_train = (x - x_avg)/x_std
y_train = events.time_to_fail.values.astype(int)
from sklearn.svm import SVC
clf = SVC(kernel='poly', degree=3)
clf.fit(x_train, y_train)
print 'accuracy of SVM fit to training data = ', clf.score(x_train, y_train)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('fail_prob')
ax.set_ylabel('Time to fail')
ax.set_title('poly degree=3')
x_train_denormalized = x_train*x_std + x_avg
ax.plot(x_train_denormalized, y_train, marker='o', linestyle='None')
y_predict = clf.predict(x_train)
ax.plot(x_train_denormalized, y_predict)
plt.show(block=False)

