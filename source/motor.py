#motor.py

import numpy as np
import pandas as pd
from sklearn.svm import SVC

class Motor:

    def __init__(self, id, time, maint_type, fail_prob_rate, time_previous_maint, maint_interval, 
            maint_duration, repair_duration):
        self.id = id
        self.maint_type = maint_type
        self.fail_prob_rate = fail_prob_rate
        self.time_previous_maint = time_previous_maint
        self.maint_interval = maint_interval
        self.maint_duration = maint_duration
        self.repair_duration = repair_duration
        self.time_when_repaired = None
        self.time_next_maint = self.time_previous_maint + self.maint_interval
        self.state = 'operating'
        self.clf = SVC(kernel='poly', degree=3)
        
    def status(self, time):
        return {'time':time, 'id':self.id, 'state':self.state, 'maint_type': self.maint_type,
            'fail_prob':self.fail_prob(time)}

    def fail_prob(self, time):
        return self.fail_prob_rate*(time - self.time_previous_maint)

    def failure_check(self, time):
        failprob = self.fail_prob(time)
        rn = np.random.uniform(low=0.0, high=1.0, size=None)
        if (rn < failprob):
            self.state = 'failed'
            self.time_when_repaired = time + self.repair_duration

    def maint_check(self, time):
        if ((self.state == 'operating') and (time >= self.time_next_maint)):
            self.state = 'maintenance'
            self.time_when_repaired = time + self.maint_duration
        if ((self.state == 'maintenance') and (time >= self.time_when_repaired)):
            self.state = 'operating'
            self.time_previous_maint = time 
            self.time_next_maint = time + self.maint_interval
        
    def repair_check(self, time):
        if (time >= self.time_when_repaired):
            self.state = 'operating'
            self.time_previous_maint = time 
            self.time_next_maint = time + self.maint_interval

    def operate(self, time):
        if (self.state == 'operating'): 
            self.failure_check(time)
            self.maint_check(time)
        if (self.state == 'maintenance'): 
            self.maint_check(time)
        if (self.state == 'failed'): self.repair_check(time)
