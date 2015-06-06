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
N_motors = 100
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
        m.operate(t)

#run motor using scheduled maintenance
for m in motors: m.maint_type = 'scheduled'
print 'maintenance mode:', motors[0].maint_type
for t in np.arange(Time_start_sched_maint, Time_stop_sched_maint):
    for m in motors:
        m.operate(t)

#run motor using predictive maintenance
xy_train = train_svm(motors, training_axes, prediction_axis)
for m in motors: m.maint_type = 'predictive'
print 'maintenance mode:', motors[0].maint_type
for t in np.arange(Time_start_pred_maint, Time_stop_pred_maint):
    for m in motors:
        m.operate(t)

#store all events in this file, for debugging
file = open('data/sm_events.json','w')
for m in motors:
    for d in m.events:
        file.write(str(d) + '\n')
file.close()

#plot fail_factor(Pressure, Temp):
events = get_events(motors)
x = events.Temp.values
y = events.Pressure.values
clr = events.fail_factor.values
fig = plt.figure(figsize=(7.5, 6.5))
ax = fig.add_subplot(1, 1, 1)
ax.set_title('fail-factor')
ax.set_xlabel('Temperature')
ax.set_ylabel('Pressure')
scat = ax.scatter(x, y, c=np.sqrt(clr), linewidths=0, s=30.0, cmap='jet')
ax.patch.set_facecolor('lightyellow')
ax.grid(True, linestyle=':', alpha=0.3)
plt.colorbar(scat)
plotfile = 'data/fail_factor.png'
fig.savefig(plotfile)
plt.close(fig) 
print 'completed plot ' + plotfile

#plot predicted ttf vs fail_factor:
df = events[events.predicted_ttf.notnull()]
x = df.fail_factor.values
y = df.predicted_ttf.values
clr = events.fail_factor.values
fig = plt.figure(figsize=(7.5, 6.5))
ax = fig.add_subplot(1, 1, 1)
ax.set_title('Predicted Time-to-Fail vs fail-factor')
ax.set_xlabel('fail-factor')
ax.set_ylabel('Predicted Time to Fail')
ax.set_xlim(x.min()-0.5, x.max()+0.5)
ax.scatter(x, y)
ax.patch.set_facecolor('lightyellow')
ax.grid(True, linestyle=':', alpha=0.3)
plotfile = 'data/time_to_fail.png'
fig.savefig(plotfile)
plt.close(fig) 
print 'completed plot ' + plotfile

#plot decision surface

#get operating stats
pd.set_option('display.expand_frame_repr', False)
N = motor_stats(motors)
print N

#plot revenue over time
events['earnings'] = 0.0
events.loc[events.state == 'operating', 'earnings'] = operating_earnings
events['expenses'] = 0.0
events.loc[events.state == 'maintenance', 'expenses'] = maintenance_cost
events.loc[events.state == 'repair', 'expenses'] = repair_cost
money = events.groupby('Time').sum()[['earnings', 'expenses']]
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
ax.annotate('run-to-fail', xy=(17, 52), verticalalignment='top')                
ax.add_patch(matplotlib.patches.Rectangle(
    (run_interval, 0), run_interval, ax.get_ylim()[1], color='gold', alpha=0.35))
ax.annotate('scheduled\nmaintenance', xy=(217, 52), verticalalignment='top')                
ax.add_patch(matplotlib.patches.Rectangle(
    (2*run_interval, 0), 4*run_interval, ax.get_ylim()[1], color='darkseagreen', alpha=0.35))
ax.annotate('predictive\nmaintenance', xy=(417, 52), verticalalignment='top')                
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
    (2*run_interval, ax.get_ylim()[0]), 4*run_interval, ax.get_ylim()[1] - ax.get_ylim()[0], 
    color='darkseagreen', alpha=0.35))
ax.grid(True, linestyle=':', alpha=0.3)

plotfile = 'data/revenue.png'
fig.savefig(plotfile)
plt.close(fig) 
print 'completed plot ' + plotfile
