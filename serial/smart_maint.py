#smart_maint.py
#
#    this is the serial (non-sparkified) version of the smart maintenance demo.
#
#    execute on a hadoop foyer node via:     python smart_maint.py

#imports
import numpy as np
import pandas as pd
from motor import *
from helper_functions import *

#matplotlib imports, to export plots to png images
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import *

#motor parameters
N_motors = 20#0
ran_num_seed = 1

#maintenance & repair parameters
maint_duration = 2
repair_duration = 3

#scheduled maintenance parameters
maint_interval = 5

#predictive maintenance parameters
training_axes = ['Pressure', 'Temp']
prediction_axis = 'Time_since_repair'
pred_maint_buffer_Time = 1

#motor failure parameters
fail_prob_rate = 0.015
Temp_0 =100.0
delta_Temp = 20.0
Pressure_0 = 50.0
delta_Pressure = 20.0

#runtime parameters
run_interval = 200
Time_start_runtofail = 0
Time_stop_runtofail = Time_start_runtofail + run_interval
Time_start_sched_maint = Time_stop_runtofail
Time_stop_sched_maint = Time_start_sched_maint + run_interval
Time_start_pred_maint = Time_stop_sched_maint
Time_stop_pred_maint = Time_start_pred_maint + 4*run_interval

#economic parameters
operating_earnings = 1000.0
maintenance_cost = 0.22*operating_earnings
repair_cost = 2.2*operating_earnings

##########################################################################################
#monitor execution time
import time
start_time_sec = time.clock()

#set random number seed
np.random.seed(ran_num_seed)

#set maintenance type to: 'run-to-fail', 'scheduled', or 'predictive'
maint_type = 'run-to-fail'

#create motors
motors = [ 
    Motor(motor_id + 100, Time_start_runtofail, maint_type, fail_prob_rate, Temp_0,
        delta_Temp, Pressure_0, delta_Pressure, maint_interval, maint_duration, 
        repair_duration, pred_maint_buffer_Time, training_axes, prediction_axis)
    for motor_id in np.arange(N_motors) ]

#run motor using run-to-fail maintenance 
print 'maintenance mode:', motors[0].maint_type
for t in np.arange(Time_start_runtofail, Time_stop_runtofail):
    for m in motors:
        m.operate()

#run motor using scheduled maintenance
for m in motors: m.maint_type = 'scheduled'
print 'maintenance mode:', motors[0].maint_type
for t in np.arange(Time_start_sched_maint, Time_stop_sched_maint):
    for m in motors:
        m.operate()

#train motor for predictive maintenance 
clf, x_avg, x_std, xy_train = train_svm(motors, training_axes, prediction_axis)
for m in motors: 
    m.maint_type = 'predictive'
    m.x_avg = x_avg
    m.x_std = x_std
    m.clf = clf

#run motors using predictive maintenance
print 'maintenance mode:', motors[0].maint_type
for t in np.arange(Time_start_pred_maint, Time_stop_pred_maint):
    for m in motors:
        m.operate()

#get operating stats
pd.set_option('display.expand_frame_repr', False)
N = motor_stats(motors)
print N

#store all events in file, for debugging
file = open('events.json','w')
for m in motors:
    for d in m.events:
        file.write(str(d) + '\n')
file.close()

#plot & report results
money, events = plot_results(motors, xy_train, operating_earnings, maintenance_cost, 
    repair_cost, run_interval)
print 'cumulative revenue at completion of run-to-fail               (M$) = ', \
    money[money.index  <= Time_stop_runtofail].cumulative_revenue.values[-1]/1.0e6
print 'cumulative revenue at completion of scheduled-maintenance     (M$) = ', \
    money[money.index  <= Time_stop_sched_maint].cumulative_revenue.values[-1]/1.0e6
print 'cumulative revenue at completion of predictive-maintenance    (M$) = ', \
    money[money.index  <= Time_stop_pred_maint].cumulative_revenue.values[-1]/1.0e6
print
print 'execution time (minutes) = ', (time.clock() - start_time_sec)/60.0
print 'number of failures during run-to-fail', len(xy_train)
print 'total number of motor events = ', len(get_events(motors))

cnts = events.groupby(['Time', 'id'])['Pressure', 'Temp'].count()
cnts['PT'] = cnts.Pressure + cnts.Temp
print cnts[cnts.PT > 2]
print np.unique(xy_train.isnull())
