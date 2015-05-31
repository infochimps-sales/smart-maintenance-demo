#motor.py

import numpy as np
import pandas as pd
from sklearn.svm import SVC

class Motor:

    def __init__(self, idnum, Time, maint_type, fail_prob_rate, maint_interval, maint_duration, 
            smart_maint_threshold, repair_duration):
        self.id = idnum
        self.maint_type = maint_type
        self.fail_prob_rate = fail_prob_rate
        self.maint_interval = maint_interval
        self.maint_duration = maint_duration
        self.repair_duration = repair_duration
        self.Time_previous_maint = Time
        self.Time_next_maint = Time - (self.maint_interval + 1) 
        self.Time_to_next_repair = None
        self.maintenance(Time)
        self.state = 'operating'
        self.clf = SVC(kernel='poly', degree=3)
        self.Time_to_fail = None
        self.smart_maint_threshold = smart_maint_threshold
        self.events = []
        
    def status(self, Time):
        Time_to_previous_maint = None
        failprob = None
        if (self.state == 'operating'):
            Time_to_previous_maint = Time - self.Time_previous_maint
            failprob = self.fail_prob(Time)
        return { 'Time':Time, 'id':self.id, 'state':self.state, 'maint_type': self.maint_type,
            'fail_prob':failprob, 'Time_previous_maint':self.Time_previous_maint, 
            'Time_next_maint':self.Time_next_maint, 'Time_to_previous_maint':Time_to_previous_maint, 
            'Time_to_next_repair':self.Time_to_next_repair }

    def fail_prob(self, Time):
        return self.fail_prob_rate*(Time - self.Time_previous_maint)

    def operate(self, Time):
        if ((self.state == 'repair') or (self.state == 'maintenance')): 
            #check if motor repairs/maintenance is done
            if (Time >= self.Time_resume_operating):
                self.state = 'operating'
                self.Time_previous_maint = Time - 1
                self.Time_next_maint = None
                if (self.maint_type == 'scheduled'):
                    self.Time_next_maint = Time + self.maint_interval
                self.Time_to_next_repair = None
        if (self.state == 'operating'): 
            self.maint_check(Time)
            self.repair_check(Time)
        self.events.append(self.status(Time))       

    def maint_check(self, Time):
        if (self.maint_type == 'run-to-fail'):
            #no maintenance
            pass
        if (self.maint_type == 'scheduled'):
            if (Time >= self.Time_next_maint):
                #scheduled maintenance is triggered
                self.maintenance(Time)
        if (self.maint_type == 'predictive'):
            #go to maintenance if the predicted Time-to-fail is soon enough
            if (self.time_to_fail_predicted(Time) <= self.smart_maint_threshold):
                self.maintenance(Time)

    def repair_check(self, Time):
        failprob = self.fail_prob(Time)
        rn = np.random.uniform(low=0.0, high=1.0, size=None)
        if (rn < failprob):
            #the motor has just failed and goes to maintenance
            self.state = 'repair'
            self.Time_next_maint = None
            self.Time_previous_maint = None
            self.Time_resume_operating = Time + self.repair_duration
            for j in np.arange(len(self.events) -1, -1, -1):
                if (self.events[j]['state'] == 'operating'):
                    self.events[j]['Time_to_next_repair'] = Time - self.events[j]['Time']
                else:
                    break

    def maintenance(self, Time):
        self.state = 'maintenance'
        self.Time_next_maint = None  
        self.Time_previous_maintenance = None
        self.Time_resume_operating = Time + self.maint_duration

    def train(self):
        events_df = pd.DataFrame(self.events)
        events_rtf = events_df[(events_df.state == 'operating') & 
            (events_df.maint_type == 'run-to-fail') & 
            (events_df.Time_to_next_repair > 0)].sort(columns='fail_prob')
        x = events_rtf.fail_prob.values
        x = x.reshape((len(x),1))
        self.x_avg = x.mean()
        self.x_std = x.std() 
        x_train = (x - self.x_avg)/self.x_std
        y_train = events_rtf.Time_to_next_repair.values.astype(int)
        self.clf.fit(x_train, y_train)
        return x, y_train

    def time_to_fail_predicted(self, Time):
        x = self.fail_prob(Time)
        x_norm = (x - self.x_avg)/self.x_std
        return self.clf.predict(x_norm)[0]
