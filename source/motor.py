#motor.py

import numpy as np
import pandas as pd
from sklearn.svm import SVC

class Motor:

    def __init__(self, idnum, Time, maint_type, fail_prob_rate, maint_interval, maint_duration, 
            repair_duration):
        self.id = idnum
        self.maint_type = maint_type
        self.fail_prob_rate = fail_prob_rate
        self.maint_interval = maint_interval
        self.maint_duration = maint_duration
        self.repair_duration = repair_duration
        self.Time_previous_maint = Time
        self.Time_next_maint = Time + self.maint_interval
        self.Time_to_next_repair = None
        self.state = 'operating'
        self.clf = SVC(kernel='poly', degree=3)
        self.train_clf = False
        self.Time_to_fail = None
        self.ttf_threshold = -1
        self.events = []
        
    def status(self, Time):
        Time_to_previous_maint = None
        failprob = None
        if (self.state == 'operating'):
            Time_to_previous_maint = Time - self.Time_previous_maint
            failprob = self.fail_prob(Time)
        return {'Time':Time, 'id':self.id, 'state':self.state, 'maint_type': self.maint_type,
            'Time_previous_maint':self.Time_previous_maint, 
            'Time_next_maint':self.Time_next_maint, 
            'Time_to_previous_maint':Time_to_previous_maint, 
            'Time_to_next_repair':self.Time_to_next_repair,
            'fail_prob':failprob }

    def fail_prob(self, Time):
        return self.fail_prob_rate*(Time - self.Time_previous_maint)

    def operate(self, Time):
        if ((self.state == 'repair') or (self.state == 'maintenance')): 
            #check if motor repairs/maintenance is done
            if (Time >= self.Time_resume_operating):
                self.state = 'operating'
                self.Time_previous_maint = Time - 1
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
        #if (self.maint_type == 'predictive'):
        #    #go to maintenance if the predicted Time-to-fail is soon enough
        #    ttf_predicted = self.predict_ttf(Time)
        #    if (ttf_predicted <= self.ttf_threshold):
        #        print '    ttf_predicted = ', ttf_predicted
        #        self.maintenance(Time)
            return

    def repair_check(self, Time):
        failprob = self.fail_prob(Time)
        rn = np.random.uniform(low=0.0, high=1.0, size=None)
        if (rn < failprob):
            #the motor has just failed and goes to maintenance
            self.state = 'repair'
            self.Time_next_maint = None
            self.Time_previous_maint = None
            self.Time_resume_operating = Time + self.repair_duration
            print Time, rn, failprob
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

    def train_clf(self):
        events_df = pd.DataFrame(m.events)
        events_rtf = events_df[(events_df.state == 'operating') & 
            (events_df.maint_type == 'run-to-fail')].sort(columns='fail_prob')
#        x = events.fail_prob.values
#        x = x.reshape((len(x),1))
#        x_avg = x.mean()
#        x_std = x.std() 
#        x_train = (x - x_avg)/x_std
#        y_train = events.Time_to_fail.values.astype(int)
#        self.clf.fit(x_train, y_train)
        
        events_rtf=np.array([np.array(d.values()) for d in m.events])
        events_rtf =  [dict  for dict in m.events if (dict['maint_type'] == 'run-to-fail') ]

    def predict_ttf(self, Time):
        if (self.train_clr == True):

            


#        x = self.fail_prob(Time)
#        x_norm = (x - x_avg)/x_std
        return self.clf.predict(x_norm)[0]
