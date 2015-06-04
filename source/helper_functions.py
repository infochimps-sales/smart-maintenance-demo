#helper_functions.py

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
