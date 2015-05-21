#sm.py

#imports
import numpy as np
import pandas as pd
from motor import Motor

#parameters
time_final = 100
fail_prob_rate = 0.01
time_previous_maint = 0
maint_interval = 10
maint_duration = 2
repair_duration = 5

#
rn_seed = 1
np.random.seed(rn_seed)
motor_id = 1001
time_init = 0
events = pd.DataFrame()
maint_mode = 'scheduled'

#create motor
m = Motor(motor_id, time_init, maint_type, fail_prob_rate, time_previous_maint, maint_interval, 
    maint_duration, repair_duration)

#
for t in np.arange(time_init, time_final, 1):
    m.operate(t)
    events = events.append([m.status(t)], ignore_index=True)
    print m.status(t) 
