#smart_maint.py

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
N_motors = 10#0
ran_num_seed = 13

#maintenance/repair parameters
maint_interval = 8
maint_duration = 2
repair_duration = 3
percent_failed_threshold = 22

#motor failure parameters
fail_prob_rate = 0.020
Temp_0 =115.0
delta_Temp = 20.0
Pressure_0 = 50.0
delta_Pressure = 20.0

#runtime parameters
run_interval = 150
Time_start_runtofail = 0
Time_stop_runtofail = Time_start_runtofail + run_interval
Time_start_sched_maint = Time_stop_runtofail
Time_stop_sched_maint = Time_start_sched_maint + run_interval
Time_start_pred_maint = Time_stop_sched_maint
Time_stop_pred_maint = Time_start_pred_maint + 2*run_interval

#economic parameters
operating_earnings = 1000.0
maintenance_cost = 0.25*operating_earnings
repair_cost = 3.2*operating_earnings

##########################################################################################
#set random number seed
np.random.seed(ran_num_seed)

#set maintenance type to: 'run-to-fail', 'scheduled', or 'predictive'
maint_type = 'run-to-fail'

#create motors
motors = [ 
    Motor(motor_id + 100, Time_start_runtofail, maint_type, fail_prob_rate, Temp_0,
        delta_Temp, Pressure_0, delta_Pressure, maint_interval, maint_duration, 
        percent_failed_threshold, repair_duration) 
    for motor_id in np.arange(N_motors) ]

#run motor using run-to-fail maintenance 
print motors[0].maint_type
for t in np.arange(Time_start_runtofail, Time_stop_runtofail):
    for m in motors:
        m.operate(t)

#run motor using scheduled maintenance
for m in motors: m.maint_type = 'scheduled'
print motors[0].maint_type
for t in np.arange(Time_start_sched_maint, Time_stop_sched_maint):
    for m in motors:
        m.operate(t)

#run motor using predictive maintenance
for m in motors: m.maint_type = 'predictive'
print motors[0].maint_type
training_axes = ['Pressure', 'Temp']
prediction_axis = 'Time_since_repair'
x_train, y_train = train_svm(motors, training_axes, prediction_axis)
for t in np.arange(Time_start_pred_maint, Time_stop_pred_maint):
    for m in motors:
        m.operate(t)

#store all events in this file, for debugging
file = open('../data/sm_events.json','w')
for m in motors:
    for d in m.events:
        file.write(str(d) + '\n')
file.close()

#get operating stats
pd.set_option('display.expand_frame_repr', False)
N = motor_stats(motors)
print N

#plot revenue over time
events_df = get_events(motors)
events_df['earnings'] = 0.0
events_df.loc[events_df.state == 'operating', 'earnings'] = operating_earnings
events_df['expenses'] = 0.0
events_df.loc[events_df.state == 'maintenance', 'expenses'] = maintenance_cost
events_df.loc[events_df.state == 'repair', 'expenses'] = repair_cost
money = events_df.groupby('Time').sum()[['earnings', 'expenses']]
money['revenue'] = money.earnings - money.expenses
money['cumulative_earnings'] = money.earnings.cumsum()
money['cumulative_expenses'] = money.expenses.cumsum()
money['cumulative_revenue'] = money.revenue.cumsum()
matplotlib.rcParams.update({'font.size': 14})
fig = plt.figure(figsize=(12.0, 8.0))
fig.subplots_adjust(hspace=0.35)

ax = fig.add_subplot(2, 1, 1)
ax.set_xlabel('Time')
ax.set_ylabel('Earnings    (M$)')
ax.set_title('Cumulative Earnings & Expenses')
ax.plot(money.index, money.cumulative_earnings/1.e6, color='blue', linewidth=4, alpha=0.7)
ax.plot(money.index, money.cumulative_expenses/1.e6, color='red', linewidth=4, alpha=0.7)
ax.add_patch(matplotlib.patches.Rectangle(
    (0,0), run_interval, ax.get_ylim()[1], color='lightsalmon', alpha=0.35))
ax.annotate('run-to-fail', xy=(12, 38), verticalalignment='top')                
ax.add_patch(matplotlib.patches.Rectangle(
    (run_interval, 0), run_interval, ax.get_ylim()[1], color='gold', alpha=0.35))
ax.annotate('scheduled\nmaintenance', xy=(162, 38), verticalalignment='top')                
ax.add_patch(matplotlib.patches.Rectangle(
    (2*run_interval, 0), 2*run_interval, ax.get_ylim()[1], color='darkseagreen', alpha=0.35))
ax.annotate('predictive\nmaintenance', xy=(312, 38), verticalalignment='top')                
ax.grid(True, linestyle=':', alpha=0.3)

ax = fig.add_subplot(2, 1, 2)
ax.set_xlabel('Time')
ax.set_ylabel('Revenue    (M$)')
ax.set_title('Cumulative Revenue')
ax.plot(money.index, money.cumulative_revenue/1.e6, color='green', linewidth=4)
ax.add_patch(matplotlib.patches.Rectangle(
    (0,ax.get_ylim()[0]), run_interval, ax.get_ylim()[1]- ax.get_ylim()[0], 
    color='lightsalmon', alpha=0.35))
ax.add_patch(matplotlib.patches.Rectangle(
    (run_interval, ax.get_ylim()[0]), run_interval, ax.get_ylim()[1] - ax.get_ylim()[0], 
    color='gold', alpha=0.35))
ax.add_patch(matplotlib.patches.Rectangle(
    (2*run_interval, ax.get_ylim()[0]), 2*run_interval, ax.get_ylim()[1] - ax.get_ylim()[0], 
    color='darkseagreen', alpha=0.35))
ax.grid(True, linestyle=':', alpha=0.3)

plotfile = '../data/revenue.png'
fig.savefig(plotfile)
plt.close(fig) 
print 'completed plot ' + plotfile
