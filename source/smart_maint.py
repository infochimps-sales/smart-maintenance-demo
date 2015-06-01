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
run_interval = 500
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
for m in motors:
    events_df = events_df.append(pd.DataFrame(m.events))
events_sched = events_df[(events_df.maint_type == 'scheduled')]
events_sched.loc[events_sched.state == 'maintenance', 'state'] = 'operating'
events_sub = events_sched[['id', 'fail_prob', 'state']]
N = events_sub.groupby(['fail_prob', 'state']).count().unstack()['id'].reset_index()
N[N.isnull()] = 0.5
N['percent_failed'] = np.round(N.repair*100.0/(N.operating + N.repair))
print N
x = N.fail_prob.values
x = x.reshape((len(x), 1))
x_avg = x.mean()
x_std = x.std() 
x_train = (x - x_avg)/x_std
y_train = N.percent_failed.values.astype(int)
clf = SVC(kernel='poly', degree=3)
clf.fit(x_train, y_train)



#store events in this file
file = open('sm_events.json','w')
for m in motors:
    for d in m.events:
        file.write(str(d) + '\n')
        print d
file.close()



#
import sys
sys.exit()



#
import sys
sys.exit()


#run motor using scheduled maintenance
m.maint_type = 'scheduled'
print m.maint_type
Time_init = Time_final
Time_final += run_interval
for t in np.arange(Time_init, Time_final):
    m.operate(t)

#run motor using predictive maintenance
m.maint_type = 'predictive'
print m.maint_type
x_train, y_train = m.train()
Time_init = Time_final
Time_final += run_interval
for t in np.arange(Time_init, Time_final):
    m.operate(t)

##show all events
#for d in m.events:
#    print d

#repair stats
pd.set_option('display.expand_frame_repr', False)
events_df = pd.DataFrame(m.events)
r = events_df.groupby(['maint_type', 'state']).count().Time
print r

#plot time-to-fail vs fail-probability
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('fail_prob')
ax.set_ylabel('Time to fail')
ax.set_title('poly degree=3')
ax.plot(x_train, y_train, marker='o', linestyle='None', markersize=2)
x_train_norm = (x_train - m.x_avg)/m.x_std
y_predict = m.clf.predict(x_train_norm)
ax.plot(x_train, y_predict)
plt.show(block=False)

