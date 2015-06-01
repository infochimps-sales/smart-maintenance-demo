#smart_maint.py

#imports
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from motor import Motor

#parameters
fail_prob_rate = 0.01
maint_interval = 6
maint_duration = 2
smart_maint_threshold = 1
repair_duration = 3
N_motors = 100
run_interval = 250
ran_num_seed = 13

#set random number seed
np.random.seed(ran_num_seed)

#set maintenance type to: 'run-to-fail', 'scheduled', or 'predictive'
maint_type = 'run-to-fail'

#create motors
Time_init = 0
motors = [ Motor(motor_id + 100, Time_init, maint_type, fail_prob_rate, maint_interval, 
    maint_duration, smart_maint_threshold, repair_duration) for motor_id in np.arange(N_motors) ]

#run motor using run-to-fail maintenance 
print motors[0].maint_type
Time_final = Time_init + run_interval
for t in np.arange(Time_init, Time_final):
    for m in motors:
        m.operate(t)

#run motor using scheduled maintenance
for m in motors: m.maint_type = 'scheduled'
print motors[0].maint_type
Time_init = Time_final
Time_final += run_interval
for t in np.arange(Time_init, Time_final):
    for m in motors:
        m.operate(t)

#train
pd.set_option('display.expand_frame_repr', False)
events_df = pd.DataFrame()
for m in motors: events_df = events_df.append(pd.DataFrame(m.events))
events_sched = events_df[(events_df.maint_type == 'scheduled')]
events_sched = events_df[(events_df.maint_type == 'scheduled') &
    (events_df.state != 'maintenance')]
events_sub = events_sched[['id', 'fail_prob', 'state']]
N = events_sub.groupby(['fail_prob', 'state']).count().unstack()['id'].reset_index()
N[N.isnull()] = 0.0
N['total'] = N.operating + N.repair
N['percent_failed'] = np.round(N.repair*100.0/N.total)
N['weight'] = N.total**(0.5)
print N
x = N.fail_prob.values
x_train = x.reshape((len(x), 1))
x_train_avg = x_train.mean()
x_train_std = x_train.std()
x_train_norm = (x_train - x_train_avg)/x_train_std
y_train = N.percent_failed.values.astype(int)
weight = N.weight.values
clf = SVC(kernel='rbf')
clf.fit(x_train_norm, y_train, sample_weight=weight)
print 'accuracy of SVM training = ', clf.score(x_train_norm, y_train)

#plot time-to-fail vs fail-probability
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('fail_prob')
ax.set_ylabel('% failures')
ax.set_title('SVM prediction')
#ax.plot(x_train, y_train, marker='o', linestyle='None', markersize=5)
ax.scatter(x_train, y_train, s=weight)
y_train_predicted = clf.predict(x_train_norm)
ax.plot(x_train, y_train_predicted)
plt.show(block=False)

#store events in this file
file = open('sm_events.json','w')
for m in motors:
    for d in m.events:
        file.write(str(d) + '\n')
        #print d
        
file.close()


##repair stats
#pd.set_option('display.expand_frame_repr', False)
#events_df = pd.DataFrame(m.events)
#r = events_df.groupby(['maint_type', 'state']).count().Time
#print r

