#sm_spark.py
#
#the smart maintenance demo
#by Joe Hahn, jhahn@infochimps.com, 22 June 2015
#
#to execute using pyspark
#    IPYTHON=1 pyspark
#    %run sm_spark.py


#imports
import numpy as np
import pandas as pd
from motor import *
from helper_functions import *
#from sklearn.svm import SVC

#matplotlib imports, to export plots to png images
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import *

####for spark
###from pyspark import SparkConf, SparkContext
###conf = SparkConf().setMaster("yarn-client").setAppName("Smart Maintenance")
###sc = SparkContext(conf=conf, pyFiles=['helper_functions.py'])
###conf = SparkConf().setMaster("local[4]").setAppName("Smart Maintenance")
###sc = SparkContext(appName='Smart Maintenance', pyFiles=['helper_functions.py'],
###    master='local[4]')

#setup for calling spark
from pyspark import SparkContext
sc = SparkContext(master='yarn-client', pyFiles=['helper_functions.py', 'motor.py'],
    appName='Smart Maintenance')


#motor parameters
N_motors = 200
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

#create parallelized list of motors
motors = sc.parallelize(
    [ Motor(motor_id + 100, Time_start_runtofail, maint_type, fail_prob_rate, 
        Temp_0, delta_Temp, Pressure_0, delta_Pressure, maint_interval, maint_duration, 
        repair_duration, pred_maint_buffer_Time, training_axes, prediction_axis)
    for motor_id in np.arange(N_motors) ] )

#run motors using run-to-fail maintenance 
print 'maintenance mode:', motors.first().maint_type
for t in np.arange(Time_start_runtofail, Time_stop_runtofail):
    motors = motors.map(lambda m: m.operate())
    #trigger lazy execution to avoid complaints of 'excessively deep recursion'
    if (t%10 == 9): motors = motors.sortBy(lambda m: m.id)

#run motors using scheduled maintenance
maint_type = 'scheduled'
motors = motors.map(lambda m: m.set_maint_type(maint_type))
print 'maintenance mode:', motors.first().maint_type
for t in np.arange(Time_start_sched_maint, Time_stop_sched_maint):
    motors = motors.map(lambda m: m.operate())
    #trigger lazy execution to avoid complaints of 'excessively deep recursion'
    if (t%10 == 9): motors = motors.sortBy(lambda m: m.id)

#train SVM to do predictive maintenance 
motors_list = motors.collect()
clf, x_avg, x_std = train_svm(motors_list, training_axes, prediction_axis)
motors = motors.map(lambda m: m.train_motors(clf, x_avg, x_std))


m = motors.collect()[100]
print m.events
print m.Time
import sys
sys.exit()

#run motors using predictive maintenance
maint_type = 'predictive'
motors = motors.map(lambda m: m.set_maint_type(maint_type))
print 'maintenance mode:', motors.first().maint_type
for t in np.arange(Time_start_pred_maint, Time_stop_pred_maint):
    print 't = ', t
    motors = motors.map(lambda m: m.operate())
    #trigger lazy execution to avoid complaints of 'excessively deep recursion'
    if (t%10 == 9): motors = motors.sortBy(lambda m: m.id)
    motors = motors.sortBy(lambda m: m.id)




