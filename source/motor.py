#motor.py

import numpy as np
import pandas as pd
from sklearn.svm import SVC

class Motor:

    def __init__(self, id, maint_type, fail_prob_rate, maint_interval, maint_duration, 
            repair_duration):
        self.id = id
        self.maint_type = maint_type
        self.fail_prob_rate = fail_prob_rate
        self.time_previous_maint = 0.0
        self.maint_interval = maint_interval
        self.maint_duration = maint_duration
        self.repair_duration = repair_duration
        self.time_when_repaired = None
        self.just_failed = False
        self.time_next_maint = self.time_previous_maint + self.maint_interval
        self.state = 'operating'
        self.clf = SVC(kernel='poly', degree=3)
        self.events = pd.DataFrame()
        
    def status(self, time):
        return {'time':time, 'id':self.id, 'state':self.state, 'maint_type': self.maint_type,
            'time_previous_maint':self.time_previous_maint, 'fail_prob':self.fail_prob(time),
            'just_failed':self.just_failed}

    def fail_prob(self, time):
        return self.fail_prob_rate*(time - self.time_previous_maint)

    def failure_check(self, time):
        failprob = self.fail_prob(time)
        rn = np.random.uniform(low=0.0, high=1.0, size=None)
        if (rn < failprob):
            self.state = 'failed'
            self.just_failed = True
            self.time_when_repaired = time + self.repair_duration

    def maint_check(self, time):
        if ((self.state == 'maintenance') and (time >= self.time_when_repaired)):
            #motor transitions from maintenance to operating
            self.state = 'operating'
            self.time_previous_maint = time 
            self.time_next_maint = time + self.maint_interval
        if (self.maint_type == 'scheduled'):
            #scheduled maintenance is begun
            if ((self.state == 'operating') and (time >= self.time_next_maint)):
                self.state = 'maintenance'
                self.time_when_repaired = time + self.maint_duration
        if (self.maint_type == 'run-to-fail'):
            pass
 
    def repair_check(self, time):
        if (time >= self.time_when_repaired):
            self.state = 'operating'
            self.time_previous_maint = time 
            self.time_next_maint = time + self.maint_interval
        else:
            self.just_failed = False

    def operate(self, time):
        if (self.state == 'failed'): 
            self.repair_check(time) 
        if (self.state == 'operating'): 
            self.failure_check(time)
            self.maint_check(time)
        if (self.state == 'maintenance'): 
            self.maint_check(time)
        dict = self.status(time)
        self.events = self.events.append([dict], ignore_index=True)
