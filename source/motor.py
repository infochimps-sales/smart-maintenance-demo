#motor.py

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from helper_functions import *

class Motor:

    def __init__(self, idnum, Time, maint_type, fail_prob_rate, Temp_0,
            delta_Temp, Pressure_0, delta_Pressure, maint_interval, maint_duration, 
            repair_duration, pred_maint_buffer_Time, training_axes, prediction_axis):
        self.id = idnum
        self.maint_type = maint_type
        self.fail_prob_rate = fail_prob_rate
        self.maint_interval = maint_interval
        self.maint_duration = maint_duration
        self.repair_duration = repair_duration
        self.Time_since_repair = 0
        self.Time_next_maint = None
        self.fail_prob = 0.0
        self.Temp = None
        self.Temp_0 = Temp_0
        self.delta_Temp = delta_Temp
        self.Pressure = None
        self.Pressure_0 = Pressure_0
        self.delta_Pressure = delta_Pressure
        self.get_Temp_Pressure()
        self.Time_resume_operating = Time - self.maint_duration - 1
        self.maintenance(Time)
        self.clf = None
        self.Pressure_avg = None
        self.Pressure_std = None
        self.Temp_avg = None
        self.Temp_std = None
        self.pred_maint_buffer_Time = pred_maint_buffer_Time
        self.training_axes = training_axes
        self.prediction_axis = prediction_axis
        self.events = []
        
    def status(self, Time):
        predicted_ttf = None
        if (self.maint_type == 'predictive'):
            predicted_ttf = self.predicted_time_to_fail()
        return { 'Time':Time, 'id':self.id, 'state':self.state, 'Temp':self.Temp,
            'Pressure':self.Pressure, 'Time_since_repair':self.Time_since_repair, 
            'maint_type': self.maint_type, 'predicted_ttf':predicted_ttf,
            'fail_factor':self.fail_factor() }

    def operate(self, Time):
        if ((self.state == 'repair') or (self.state == 'maintenance')): 
            #check if motor repairs/maintenance is done
            if (Time >= self.Time_resume_operating):
                if (self.state == 'repair'):
                    self.get_Temp_Pressure()
                self.state = 'operating'
                self.Time_previous_maint = Time - 1
                self.Time_next_maint = None
                self.Time_since_repair = 0
                if (self.maint_type == 'scheduled'):
                    self.Time_next_maint = Time + self.maint_interval
                if (self.maint_type == 'predictive'):
                    self.get_Temp_Pressure()
        if (self.state == 'operating'): 
            self.time_operating = 1
            self.Time_since_repair += 1
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
            predicted_time_to_fail = self.predicted_time_to_fail()
            if (self.Time_since_repair  > self.predicted_time_to_fail() - self.pred_maint_buffer_Time):
                self.maintenance(Time)

    def repair_check(self, Time):
        self.fail_prob = self.get_fail_prob(Time)
        rn = np.random.uniform(low=0.0, high=1.0, size=None)
        if (rn < self.fail_prob):
            #the motor has just failed and goes to maintenance
            self.state = 'repair'
            self.time_maintained = 0
            self.time_repaired = 1
            self.time_operating = 0
            self.Time_next_maint = None
            self.Time_previous_maint = Time
            self.Time_resume_operating = Time + self.repair_duration
            self.Time_since_repair = 0

    def fail_factor(self):
        factor =  1.0 + (np.abs(self.Pressure - self.Pressure_0)/self.delta_Pressure)**(3.0)
        if (self.Temp > self.Temp_0):
            factor += ((self.Temp - self.Temp_0)/self.delta_Temp)**(2.0)
        return factor

    def get_fail_prob(self, Time):
        return self.fail_prob_rate*(Time - self.Time_previous_maint)*self.fail_factor()

    def get_Temp_Pressure(self):
        #this should instead return a dict(training_axes)
        self.Temp = np.random.uniform(low=50.0, high=150.0)
        self.Pressure = np.random.uniform(low=0.0, high=100.0)

    def maintenance(self, Time):
        self.state = 'maintenance'
        self.time_maintained = 1
        self.time_repaired = 0
        self.time_operating = 0
        self.Time_next_maint = None  
        self.Time_previous_maint = Time
        self.Time_resume_operating = Time + self.maint_duration

    def predicted_time_to_fail(self):
        x = {'Pressure':self.Pressure, 'Temp':self.Temp} # <= YUCK! eliminate references to Pressure & Temp
        x_norm = { col:(x[col] - self.x_avg[col])/self.x_std[col] for col in self.training_axes}
        return self.clf.predict( x_norm.values() )[0]
        
    def get_training_dataframe(self):
        events_df = pd.DataFrame(self.events)
        events_op = events_df[(events_df.state == 'operating') & 
            (events_df.maint_type == 'run-to-fail')].reset_index(drop=True)
        events_op['time_to_fail'] = None
        i = 0
        while (i < len(events_op)):
            j = i + 1
            while (j < len(events_op)):
                if ((events_op.loc[j, 'Pressure'] == events_op.loc[i, 'Pressure']) and
                        (events_op.loc[j, 'Temp'] == events_op.loc[i, 'Temp']) and 
                        (events_op.loc[j, 'id'] == events_op.loc[i, 'id'])):
                    j += 1
                else:
                    break
            events_op.loc[i:j, 'time_to_fail'] = events_op.Time_since_repair[i:j].max()
            i = j
        xy_train = events_op[self.training_axes + ['time_to_fail'] ].drop_duplicates()
        return xy_train
