#sm.py

import numpy as np
import pandas as pd
from vehicle import Vehicle

id = 1001
time_init = 0
fail_prob_rate = 0.01
time_previous_maint = 0
maint_interval = 10
maint_duration = 2
repair_duration = 5
np.random.seed(42)
v = Vehicle(id, time_init, fail_prob_rate, time_previous_maint, maint_interval, maint_duration, 
    repair_duration)
for t in np.arange(time_init, 100, 1):
    v.operate(t)
    print v.status(t) 
