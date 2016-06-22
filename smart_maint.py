#smart_maint.py
#
#the smart maintenance demo
#by Joe Hahn, jhahn@infochimps.com, 22 June 2015 
#
#submit this job to Yarn using spark-submit:
#    PYSPARK_PYTHON=/home/$USER/anaconda/bin/python spark-submit smart_maint.py
#
#execution time = 2.5 minutes

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

#setup to submit spark job to YARN
from pyspark import SparkContext
sc = SparkContext(pyFiles=['helper_functions.py', 'motor.py'])

##uncomment the following to setup to execute in local mode on the hadoop foyer node
#from pyspark import SparkConf, SparkContext
#sc = SparkContext()

#motor parameters
N_motors = 200
ran_num_seed = 2

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

#advance this demo Nsteps before forcing spark's lazy execution trigger, but this number
#downwards when spark complaint about 'excessively deep recursion'
Nsteps = 50

##########################################################################################
#monitor execution time
import time
start_time_sec = time.clock()

#set random number seed
#np.random.seed(ran_num_seed)

#set maintenance type to: 'run-to-fail', 'scheduled', or 'predictive'
maint_type = 'run-to-fail'

import sys
sys.exit()

#create parallelized list of motors
#num_partitions = 3*7*1    #3datanotes*(7 of 8 vcpus on m3.2xl)*1partitions_per_cpu exec_time=155sec
num_partitions = 3*7*2     #3datanotes*(7 of 8 vcpus on m3.2xl)*2partitions_per_cpu exec_time=150sec
#num_partitions = 3*7*4    #3datanotes*(7 of 8 vcpus on m3.2xl)*4partitions_per_cpu exec_time=151sec
motors = sc.parallelize(
    [ Motor(motor_id + 100, Time_start_runtofail, maint_type, fail_prob_rate, 
        Temp_0, delta_Temp, Pressure_0, delta_Pressure, maint_interval, maint_duration, 
        repair_duration, pred_maint_buffer_Time, training_axes, prediction_axis)
    for motor_id in np.arange(N_motors) ], numSlices=num_partitions )

#run motors using run-to-fail maintenance 
print 'maintenance mode:', motors.first().maint_type
for t in np.arange(Time_start_runtofail, Time_stop_runtofail):
    motors = motors.map(lambda m: m.operate())

#run motors using scheduled maintenance
maint_type = 'scheduled'
motors = motors.map(lambda m: m.set_maint_type(maint_type))
print 'maintenance mode:', motors.first().maint_type
for t in np.arange(Time_start_sched_maint, Time_stop_sched_maint):
    motors = motors.map(lambda m: m.operate())
    #inelegant way to trigger lazy execution and avoid 'excessively deep recursion' error
    if (t%100 == 99): motors = motors.sortBy(lambda m: m.id)
    
#train SVM to do predictive maintenance 
motors_local = motors.collect()
clf, x_avg, x_std, xy_train = train_svm(motors_local, training_axes, prediction_axis)
motors = motors.map(lambda m: m.train_motors(clf, x_avg, x_std))

#run motors using predictive maintenance
maint_type = 'predictive'
motors = motors.map(lambda m: m.set_maint_type(maint_type))
print 'maintenance mode:', motors.first().maint_type
for t in np.arange(Time_start_pred_maint, Time_stop_pred_maint):
    motors = motors.map(lambda m: m.operate())
    #inelegant way to trigger lazy execution and avoid 'excessively deep recursion' error
    if (t%300 == 299): motors = motors.sortBy(lambda m: m.id)

#get operating stats
pd.set_option('display.expand_frame_repr', False)
motors_local = motors.collect()
N = motor_stats(motors_local)
print N

#store all events in this file, for debugging
file = open('events.json','w')
for m in motors_local:
    for d in m.events:
        file.write(str(d) + '\n')

file.close()

#plot & report results
money, events = plot_results(motors_local, xy_train, operating_earnings, maintenance_cost, 
    repair_cost, run_interval)
print 'cumulative revenue at completion of run-to-fail               (M$) = ', \
    money[money.index  <= Time_stop_runtofail].cumulative_revenue.values[-1]/1.0e6
print 'cumulative revenue at completion of scheduled-maintenance     (M$) = ', \
    money[money.index  <= Time_stop_sched_maint].cumulative_revenue.values[-1]/1.0e6
print 'cumulative revenue at completion of predictive-maintenance    (M$) = ', \
    money[money.index  <= Time_stop_pred_maint].cumulative_revenue.values[-1]/1.0e6
#print 'execution time (minutes) = ', (time.clock() - start_time_sec)/60.0
