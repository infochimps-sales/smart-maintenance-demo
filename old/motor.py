#motor.py

import numpy as np
import pandas as pd
from sklearn.svm import SVC

#matplotlib imports, to export plots to png images
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import *

#this helper function trains the SVM classifier that predicts the % likelihood that
#a motor will fail
def train_svm(motors, svm_plot):
    events_df = get_events(motors)
    events_sched = events_df[(events_df.maint_type == 'scheduled')]
    events_sched = events_df[(events_df.maint_type == 'scheduled') &
        (events_df.state != 'maintenance')]
    events_sub = events_sched[['id', 'fail_prob', 'state']]
    N = events_sub.groupby(['fail_prob', 'state']).count().unstack()['id'].reset_index()
    N[N.isnull()] = 0.0
    N['total'] = N.operating + N.repair
    N['percent_failed'] = np.round(N.repair*100.0/N.total)
    N['weight'] = N.total**(0.5)
    x = N.fail_prob.values
    x_train = x.reshape((len(x), 1))
    x_train_avg = x_train.mean()
    x_train_std = x_train.std()
    x_train_norm = (x_train - x_train_avg)/x_train_std
    y_train = N.percent_failed.values.astype(int)
    weight = N.weight.values
    weight = weight/weight.min()
    clf = SVC(kernel='rbf')
    clf.fit(x_train_norm, y_train, sample_weight=weight)
    print 'accuracy of SVM training = ', clf.score(x_train_norm, y_train)
    for m in motors:
        #theres gotta be a better way than this...
        m.clf = clf
        m.x_avg = x_train_avg
        m.x_std = x_train_std

    #plot trained/predicted %failed versus fail_prob
    if (svm_plot == True):
        fig = plt.figure(figsize=(8.0, 6.0))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('fail_prob')
        ax.set_ylabel('% failures')
        ax.set_title('SVM prediction')
        ax.scatter(x_train, y_train, s=weight)
        y_train_predicted = clf.predict(x_train_norm)
        ax.plot(x_train, y_train_predicted)
        ax.patch.set_facecolor('lightyellow')
        ax.grid(True, linestyle=':', alpha=0.3)
        plotfile = '../data/percent_fail.png'
        fig.savefig(plotfile)
        plt.close(fig) 
        print 'completed plot ' + plotfile
        
    return x_train, x_train_norm, y_train, weight

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

    def __init__(self, idnum, Time, maint_type, fail_prob_rate, maint_interval, maint_duration, 
            percent_failed_threshold, repair_duration):
        self.id = idnum
        self.maint_type = maint_type
        self.fail_prob_rate = fail_prob_rate
        self.maint_interval = maint_interval
        self.maint_duration = maint_duration
        self.repair_duration = repair_duration
        self.Time_previous_maint = Time
        self.Time_next_maint = Time - (self.maint_interval + 1) 
        self.Time_to_next_repair = None
        self.fail_prob = None
        self.maintenance(Time)
        self.state = 'operating'
        self.clf = None
        self.x_avg = None
        self.x_std = None
        self.Time_to_fail = None
        self.percent_failed_threshold = percent_failed_threshold
        self.events = []
        
    def status(self, Time):
        Time_to_previous_maint = None
        if (self.state == 'operating'):
            Time_to_previous_maint = Time - self.Time_previous_maint
        return { 'Time':Time, 'id':self.id, 'state':self.state, 'maint_type': self.maint_type,
            'fail_prob':self.fail_prob, 'Time_previous_maint':self.Time_previous_maint, 
            'Time_next_maint':self.Time_next_maint, 'Time_to_previous_maint':Time_to_previous_maint, 
            'Time_to_next_repair':self.Time_to_next_repair }

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
            if (self.percent_failed_predicted(Time) > self.percent_failed_threshold):
                self.maintenance(Time)

    def repair_check(self, Time):
        self.fail_prob = self.get_fail_prob(Time)
        rn = np.random.uniform(low=0.0, high=1.0, size=None)
        if (rn < self.fail_prob):
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

    def get_fail_prob(self, Time):
        return self.fail_prob_rate*(Time - self.Time_previous_maint)

    def maintenance(self, Time):
        self.state = 'maintenance'
        self.Time_next_maint = None  
        self.Time_previous_maintenance = None
        self.Time_resume_operating = Time + self.maint_duration

    def percent_failed_predicted(self, Time):
        x = self.get_fail_prob(Time)
        x_norm = (x - self.x_avg)/self.x_std
        return self.clf.predict(x_norm)[0]
