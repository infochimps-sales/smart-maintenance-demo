#motor.py

import numpy as np
import pandas as pd
from sklearn.svm import SVC

#matplotlib imports, to export plots to png images
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import *

#this helper function trains the SVM classifier to predict the percent likelihood
#that a motor will fail
def train_svm(motors, prediction_axes, repair_duration):
    pd.set_option('display.expand_frame_repr', False)
    print 'training: relies on an inefficient hack, needs more work...'
    events_df = get_events(motors)
    groupby_axes = prediction_axes + ['state']
    events_sched = events_df[events_df.maint_type == 'scheduled'][groupby_axes + ['id']].fillna(value=0.0)
    N = events_sched.groupby(groupby_axes).count().reset_index()
    Np = N.pivot_table(index=prediction_axes, columns=['state'], values=['id'], fill_value=0)\
        .rename(columns = {'id':'counts'}).counts
    for idx, row in Np.iterrows():
        (P, T, t) = idx
        [m, o, r] = row.values.tolist()
        print P, T, t, m, o, r

    Np['percent_failed'] = 0
    Np['Pressure'] = None
    Np['Temp'] = None
    Np['t'] = None
    for idx, row in Np.iterrows():
        #the following needs to be altered if the prediction_axes change...
        try:
            (P, T, t) = idx
            idx_repair = (P, T, 0)
            Np.loc[idx, 'percent_failed'] = int(np.round(
                Np.loc[idx_repair, 'id_counts'].repair*100.0/(Np.loc[idx, 'id_counts'].maintenance 
                + Np.loc[idx, 'id_counts'].operating + Np.loc[idx_repair, 'id_counts'].repair)))
            if (t > 0): Np.loc[idx, 'Pressure'] = [ P, T, t]
        except:
            pass

    Np['percent_failed'] = -1
    Np['percent_failed'] = Np.apply(lambda row: np.abs(row.percent_failed))

    N['total'] = N.operating + N.repair
    N['percent_failed'] = np.round(N.repair*100.0/N.total)
    N['weight'] = N.total**(0.5)
    N['Pressure_avg'] = N.Pressure.mean()
    N['Pressure_std'] = N.Pressure.std()
    N['Temp_avg'] = N.Temp.mean()
    N['Temp_std'] = N.Temp.std()
    N['Pressure_norm'] = (N.Pressure - N.Pressure_avg)/N.Pressure_std
    N['Temp_norm'] = (N.Temp - N.Temp_avg)/N.Temp_std
    x_train = N[['Pressure_norm', 'Temp_norm']].values
    y_train = N.percent_failed.values.astype(int)
    weight = N.weight.values
    #weight = weight/weight.min()
    clf = SVC(kernel='rbf')
    clf.fit(x_train, y_train, sample_weight=weight)
    print 'accuracy of SVM training = ', clf.score(x_train, y_train)
    for m in motors:
        #theres gotta be a better way than this...
        m.clf = clf
        m.Pressure_avg = N.Pressure_avg.values[0]
        m.Pressure_std = N.Pressure_std.values[0]
        m.Temp_avg = N.Temp_avg.values[0]
        m.Temp_std = N.Temp_std.values[0]
    ##plot trained/predicted %failed versus fail_prob
    #fig = plt.figure(figsize=(8.0, 6.0))
    #ax = fig.add_subplot(1, 1, 1)
    #ax.set_xlabel('Temperature')
    #ax.set_ylabel('Pressure')
    #ax.set_title('%')
    #ax.scatter(x_train[:,0], x_train[:,1], c=-y_train)
    #ax.patch.set_facecolor('lightyellow')
    #ax.grid(True, linestyle=':', alpha=0.3)
    #plotfile = '../data/percent_fail.png'
    #fig.savefig(plotfile)
    #plt.close(fig) 
    #print 'completed plot ' + plotfile
    return x_train, y_train, weight

def get_events(motors):
    events_df = pd.DataFrame()
    for m in motors:
        events_df = events_df.append(pd.DataFrame(m.events))
    return events_df
    
def motor_stats(motors):
    events_df = get_events(motors)
    N = events_df.groupby(['maint_type', 'state']).count().unstack()['id'].reset_index()
    N.loc[N.maintenance.isnull(), 'maintenance'] = 0
    N['total'] = N.maintenance + N.operating + N.repair
    N['percent_maint'] = N.maintenance*1.0/N.total
    N['percent_operating'] = N.operating*1.0/N.total
    N['percent_repair'] = N.repair*1.0/N.total
    return N.sort('percent_repair', ascending=False)

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
        self.Time_previous_maint = Time
        self.Time_next_maint = Time - (self.maint_interval + 1) 
        self.was_maintained = None
        self.was_repaired = None
        self.Time_to_next_repair = None
        self.fail_prob = None
        self.Temp = None
        self.Temp_0 = Temp_0
        self.delta_Temp = delta_Temp
        self.Pressure = None
        self.Pressure_0 = Pressure_0
        self.delta_Pressure = delta_Pressure
        self.get_Temp_Pressure()
        self.maintenance(Time)
        self.state = 'operating'
        self.clf = None
        self.Pressure_avg = None
        self.Pressure_std = None
        self.Temp_avg = None
        self.Temp_std = None
        self.Time_to_fail = None
        self.percent_failed_threshold = percent_failed_threshold
        self.events = []
        
    def status(self, Time):
        Time_to_previous_maint = 0
        if (self.state == 'operating'):
            Time_to_previous_maint = Time - self.Time_previous_maint
        return { 'Time':Time, 'id':self.id, 'state':self.state, 'maint_type': self.maint_type,
            'fail_prob':self.fail_prob, 'Time_previous_maint':self.Time_previous_maint, 
            'Time_next_maint':self.Time_next_maint, 'Time_to_previous_maint':Time_to_previous_maint, 
            'Time_to_next_repair':self.Time_to_next_repair, 'was_maintained': self.was_maintained,
            'was_repaired': self.was_repaired, 'Temp':self.Temp, 'Pressure':self.Pressure }

    def operate(self, Time):
        if ((self.state == 'repair') or (self.state == 'maintenance')): 
            #check if motor repairs/maintenance is done
            if (Time >= self.Time_resume_operating):
                self.state = 'operating'
                self.Time_previous_maint = Time - 1
                self.Time_next_maint = None
                self.get_Temp_Pressure()
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
            if (self.percent_failed_predicted(Time) > self.percent_failed_threshold):
                self.maintenance(Time)

    def repair_check(self, Time):
        self.fail_prob = self.get_fail_prob(Time)
        rn = np.random.uniform(low=0.0, high=1.0, size=None)
        if (rn < self.fail_prob):
            #the motor has just failed and goes to maintenance
            self.state = 'repair'
            self.was_maintained = 0
            self.was_repaired = 1
            self.Time_next_maint = None
            self.Time_previous_maint = Time
            self.Time_resume_operating = Time + self.repair_duration
            for j in np.arange(len(self.events) -1, -1, -1):
                if (self.events[j]['state'] == 'operating'):
                    self.events[j]['Time_to_next_repair'] = Time - self.events[j]['Time']
                else:
                    break

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
        self.was_maintained = 1
        self.was_repaired = 0
        self.Time_next_maint = None  
        self.Time_previous_maint = Time
        self.Time_resume_operating = Time + self.maint_duration

    def percent_failed_predicted(self, Time):
        x = self.get_fail_prob(Time)
        x_norm = (x - self.x_avg)/self.x_std
        return self.clf.predict(x_norm)[0]
