#motor.py

from sklearn.svm import SVC
from helper_functions import *

class Motor:

    def __init__(self, idnum, Time, maint_type, fail_prob_rate, Temp_0,
            delta_Temp, Pressure_0, delta_Pressure, maint_interval, maint_duration, 
            percent_failed_threshold, repair_duration):
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
        self.Time_to_fail = None
        self.percent_failed_threshold = percent_failed_threshold
        self.events = []
        
    def status(self, Time):
        return { 'Time':Time, 'id':self.id, 'state':self.state, 'Temp':self.Temp,
            'Pressure':self.Pressure, 'Time_since_repair':self.Time_since_repair, 
            'fail_prob':self.fail_prob, 'maint_type': self.maint_type }

    def operate(self, Time):
        if ((self.state == 'repair') or (self.state == 'maintenance')): 
            #check if motor repairs/maintenance is done
            if (Time >= self.Time_resume_operating):
                if (self.state == 'repair'):
                    self.get_Temp_Pressure()
                self.state = 'operating'
                self.Time_previous_maint = Time - 1
                self.Time_next_maint = None
                if (self.maint_type == 'scheduled'):
                    self.Time_next_maint = Time + self.maint_interval
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
            if (self.percent_failed_predicted(Time) > self.percent_failed_threshold):
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

    def get_fail_prob(self, Time):
        PT_factor = 1.0 + ((self.Pressure - self.Pressure_0)/self.delta_Pressure)**2
        if (self.Temp > self.Temp_0):
            PT_factor += ((self.Temp - self.Temp_0)/self.delta_Temp)**(0.5)
        return self.fail_prob_rate*PT_factor*(Time - self.Time_previous_maint)

    def get_Temp_Pressure(self):
        self.Temp = np.round(np.random.uniform(low=50.0, high=150.0))
        self.Pressure = np.round(np.random.uniform(low=0.0, high=100.0))

    def maintenance(self, Time):
        self.state = 'maintenance'
        self.time_maintained = 1
        self.time_repaired = 0
        self.time_operating = 0
        self.Time_next_maint = None  
        self.Time_previous_maint = Time
        self.Time_resume_operating = Time + self.maint_duration

    def predicted_time_to_fail(self, Time):
        P_norm = (self.Pressure - self.Pressure_avg)/self.Pressure_std
        T_norm = (self.Temp - self.Temp_avg)/self.Temp_std
        return self.clf.predict(x_norm)[0]
