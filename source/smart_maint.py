#smart_maint.py

#imports
import numpy as np
from motor import Motor

#parameters for scheduled maintenance
fail_prob_rate = 0.01
maint_interval = 5
maint_duration = 2
repair_duration = 5

#set random number seed
rn_seed = 13
np.random.seed(rn_seed)

#set maintenance type to: 'run-to-fail', 'scheduled', or 'predictive'
maint_type = 'run-to-fail'

#create motor
motor_id = 201
Time_init = 0
m = Motor(motor_id, Time_init, maint_type, fail_prob_rate, maint_interval, maint_duration, 
    repair_duration)

#run motor using run-to-fail maintenance
Time_final = 100
print m.maint_type
for t in np.arange(Time_init, Time_final):
    m.operate(t)

#run motor using scheduled maintenance
m.maint_type = 'scheduled'
print m.maint_type
Time_init = Time_final
Time_final += 100
for t in np.arange(Time_init, Time_final):
    m.operate(t)

#run motor using predictive maintenance
m.maint_type = 'predictive'
print m.maint_type
pd.set_option('display.expand_frame_repr', False)
Time_init = Time_final
Time_final += 100
for t in np.arange(Time_init, Time_final):
    m.operate(t)

#show all events
for d in m.events:
    print d



import sys
sys.exit()


import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('fail_prob')
ax.set_ylabel('Time to fail')
ax.set_title('poly degree=3')
x_train_denormalized = x_train*x_std + x_avg
ax.plot(x_train_denormalized, y_train, marker='o', linestyle='None', markersize=2)
y_predict = clf.predict(x_train)
ax.plot(x_train_denormalized, y_predict)
plt.show(block=False)

