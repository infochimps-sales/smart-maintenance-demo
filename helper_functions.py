#helper_functions.py

#get imports 
import numpy as np
import pandas as pd
from sklearn.svm import SVC
#from sklearn.cross_validation import StratifiedShuffleSplit
#from sklearn.grid_search import GridSearchCV

#matplotlib imports, to export plots to png images
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import *

def get_events(motors):
    events_df = pd.DataFrame()
    for m in motors:
        events_df = events_df.append(pd.DataFrame(m.events))
    return events_df.sort(['Time', 'id']).reset_index(drop=True)

def train_svm(motors, training_axes, prediction_axis):
    pd.set_option('display.expand_frame_repr', False)
    print '...training SVM...this portion is computed serially on foyer node...'
    xy_train = pd.DataFrame()
    for m in motors: 
        xy_train = xy_train.append(m.get_training_dataframe())
    x_avg = {col:xy_train[col].mean() for col in training_axes}
    x_std = {col:xy_train[col].std()  for col in training_axes}
    for col in training_axes:
        xy_train[col + '_norm'] = (xy_train[col] - x_avg[col])/x_std[col]
    training_axes_norm = [col + '_norm' for col in training_axes]
    x_train = xy_train[training_axes_norm].values
    y_train = xy_train['time_to_fail'].values.astype(int)
    #The following notes results for when various SVM parameters are employed...
    #clf = SVC(kernel='rbf', C=10.0, gamma=10.0) #score=0.449 => very weird dec surface, $ is somewhat ok
    #clf = SVC(kernel='rbf', C=1.0, gamma=10.0) #score=0.337 => weird dec surface, $ ok otherwise
    #clf = SVC(kernel='rbf', C=0.1, gamma=10.0) #score=0.185  => not good
    #clf = SVC(kernel='rbf', C=10.0, gamma=1.0) #score=0.254 => weird dec surface, $ is ok
    #clf = SVC(kernel='rbf', C=1.0, gamma=1.0) #score=0.240 => good dec surface, $ is good #####
    #clf = SVC(kernel='rbf', C=0.1, gamma=1.0) #score=0.197  => not good
    #clf = SVC(kernel='rbf', C=10.0, gamma=0.1) #score=0.229 => ok dec surface, $ is very good ####
    #clf = SVC(kernel='rbf', C=1.0, gamma=0.1) #score=0.198 => bad dec surface, $ is bad
    #clf = SVC(kernel='rbf', C=0.1, gamma=0.1) #score=0.185  => bad dec surf, bad $
    clf = SVC(kernel='rbf', C=1.0, gamma=1.0)
    clf.fit(x_train, y_train)
    print '...accuracy of SVM training = ', clf.score(x_train, y_train)
    return clf, x_avg, x_std, xy_train

def motor_stats(motors):
    events_df = get_events(motors)
    N = events_df.groupby(['maint_type', 'state']).count().unstack()['id'].reset_index()
    N.loc[N.maintenance.isnull(), 'maintenance'] = 0
    N['total'] = N.maintenance + N.operating + N.repair
    N['percent_maint'] = N.maintenance*1.0/N.total
    N['percent_operating'] = N.operating*1.0/N.total
    N['percent_repair'] = N.repair*1.0/N.total
    return N.sort('percent_repair', ascending=False)

def plot_results(motors, xy_train, operating_earnings, maintenance_cost, repair_cost, run_interval):

    #contour fail_factor vs Temp & Pressure
    print '...generating output plots...'
    events = get_events(motors)
    T_axis = np.arange(50.0, 151.0, 0.5)
    P_axis = np.arange(0.0, 101.0)
    x, y = np.meshgrid(T_axis, P_axis)
    z = np.zeros((len(P_axis), len(T_axis)))
    import copy
    m = copy.deepcopy(motors[0])
    for p_idx in np.arange(len(P_axis)):
        for t_idx in np.arange(len(T_axis)):
            m.Temp = T_axis[t_idx]
            m.Pressure = P_axis[p_idx]
            z[p_idx, t_idx] = m.fail_factor()
    fig = plt.figure(figsize=(7.0, 7.0))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('Observed Motor Lifetime')
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Pressure')
    Ncolors = 256
    cf = ax.contourf(x, y, z, Ncolors, cmap='jet')
    dot_size = xy_train.time_to_fail**1.5
    ax.scatter(xy_train.Temp, xy_train.Pressure, marker='o', color='white', alpha=0.6, 
        s=dot_size.tolist())
    plotfile = 'figs/fail_factor.png'
    fig.savefig(plotfile)
    plt.close(fig) 
    print 'completed plot ' + plotfile

    #contour predicted_time_to_fail(Temp, Pressure):
    T_axis = np.arange(50.0, 151.0, 0.5)
    P_axis = np.arange(0.0, 101.0)
    x, y = np.meshgrid(T_axis, P_axis)
    z = np.zeros((len(P_axis), len(T_axis)))
    id = 0
    import copy
    m = copy.deepcopy(motors[0])
    for p_idx in np.arange(len(P_axis)):
        for t_idx in np.arange(len(T_axis)):
            m.Temp = T_axis[t_idx]
            m.Pressure = P_axis[p_idx]
            z[p_idx, t_idx] = m.predicted_time_to_fail()
    fig = plt.figure(figsize=(8.0, 6.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('Predicted Time to Fail')
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Pressure')
    contour_vals = np.unique(z)
    contour_vals = np.append(contour_vals, contour_vals.max() + 1) - 0.5
    cf = ax.contourf(x, y, z, contour_vals, cmap='afmhot')
    plt.colorbar(cf, ticks=np.unique(z))
    plotfile = 'figs/predicted_time_to_fail.png'
    fig.savefig(plotfile)
    plt.close(fig) 
    print 'completed plot ' + plotfile

    #plot revenue over time
    events = get_events(motors)
    events['earnings'] = 0.0
    events.loc[events.state == 'operating', 'earnings'] = operating_earnings
    events['expenses'] = 0.0
    events.loc[events.state == 'maintenance', 'expenses'] = maintenance_cost
    events.loc[events.state == 'repair', 'expenses'] = repair_cost
    money = events.groupby('Time').sum()[['earnings', 'expenses']]
    money['revenue'] = money.earnings - money.expenses
    money['cumulative_earnings'] = money.earnings.cumsum()
    money['cumulative_expenses'] = money.expenses.cumsum()
    money['cumulative_revenue'] = money.revenue.cumsum()
    matplotlib.rcParams.update({'font.size': 14})
    fig = plt.figure(figsize=(12.0, 8.0))
    fig.subplots_adjust(hspace=0.35)
    #
    ax = fig.add_subplot(2, 1, 1)
    ax.set_xlabel('Time')
    ax.set_ylabel('Earnings    (M$)')
    ax.set_title('Cumulative Earnings & Expenses')
    ax.plot(money.index, money.cumulative_earnings/1.e6, color='blue', linewidth=4, 
        alpha=0.7, label='earnings')
    ax.plot(money.index, money.cumulative_expenses/1.e6, color='red', linewidth=4, 
        alpha=0.7, label='expenses')
    ax.add_patch(matplotlib.patches.Rectangle(
        (0,0), run_interval, ax.get_ylim()[1], color='lightsalmon', alpha=0.35))
    ax.annotate('run-to-fail', xy=(19, 111), verticalalignment='top')                
    ax.add_patch(matplotlib.patches.Rectangle(
        (run_interval, 0), run_interval, ax.get_ylim()[1], color='gold', alpha=0.35))
    ax.annotate('scheduled\nmaintenance', xy=(219, 111), verticalalignment='top')                
    ax.add_patch(matplotlib.patches.Rectangle(
        (2*run_interval, 0), 4*run_interval, ax.get_ylim()[1], color='darkseagreen', alpha=0.35))
    ax.annotate('predictive\nmaintenance', xy=(419, 111), verticalalignment='top')                
    ax.grid(True, linestyle=':', alpha=0.3)
    ax.legend(loc='lower right', fontsize='small')
    #
    ax = fig.add_subplot(2, 1, 2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Revenue    (M$)')
    ax.set_title('Cumulative Revenue')
    ax.plot(money.index, money.cumulative_revenue/1.e6, color='green', linewidth=4)
    ax.plot(money.index, money.index*0, color='purple', linewidth=2, linestyle='--', alpha=0.5)
    ax.add_patch(matplotlib.patches.Rectangle(
        (0,ax.get_ylim()[0]), run_interval, ax.get_ylim()[1]- ax.get_ylim()[0], 
        color='lightsalmon', alpha=0.35))
    ax.add_patch(matplotlib.patches.Rectangle(
        (run_interval, ax.get_ylim()[0]), run_interval, ax.get_ylim()[1] - ax.get_ylim()[0], 
        color='gold', alpha=0.35))
    ax.add_patch(matplotlib.patches.Rectangle(
        (2*run_interval, ax.get_ylim()[0]), 4*run_interval, ax.get_ylim()[1] - ax.get_ylim()[0], 
        color='darkseagreen', alpha=0.35))
    ax.grid(True, linestyle=':', alpha=0.3)
    #
    plotfile = 'figs/revenue.png'
    fig.savefig(plotfile)
    plt.close(fig) 
    print 'completed plot ' + plotfile
    return money, events

def make_dashboard(motors, xy_train, operating_earnings, maintenance_cost, repair_cost, run_interval):

    #calculate revenue vs time dataframe 
    print '...generating dashboard...'
    events = get_events(motors)
    events['earnings'] = 0.0
    events.loc[events.state == 'operating', 'earnings'] = operating_earnings
    events['expenses'] = 0.0
    events.loc[events.state == 'maintenance', 'expenses'] = maintenance_cost
    events.loc[events.state == 'repair', 'expenses'] = repair_cost
    money = events.groupby('Time').sum()[['earnings', 'expenses']]
    money['revenue'] = money.earnings - money.expenses
    money['cumulative_earnings'] = money.earnings.cumsum()
    money['cumulative_expenses'] = money.expenses.cumsum()
    money['cumulative_revenue'] = money.revenue.cumsum()
    #map the (P,T) decision surface
    T_min = 50
    T_max = 150
    P_min = 0
    P_max = 100
    T_axis = np.arange(T_min, T_max, 0.5)
    P_axis = np.arange(P_min, P_max, 0.5)
    x, y = np.meshgrid(T_axis, P_axis)
    ttf = np.zeros((len(P_axis), len(T_axis)))
    import copy
    m = copy.deepcopy(motors[0])
    for p_idx in np.arange(len(P_axis)):
        for t_idx in np.arange(len(T_axis)):
            m.Temp = T_axis[t_idx]
            m.Pressure = P_axis[p_idx]
            ttf[p_idx, t_idx] = m.predicted_time_to_fail()

    #plot decision surface
    from bokeh.plotting import figure, show, output_file, ColumnDataSource, vplot
    from bokeh.models import HoverTool
    output_file('dashboard.html', title='Smart Maintenance Dashboard')
    source = ColumnDataSource(
        data=dict(
            x = xy_train.Temp,
            y = xy_train.Pressure,
            ttf = xy_train.time_to_fail,
            size = 0.6*xy_train.time_to_fail,
        )
    )    
    dec_fig = figure(x_range=[T_min, T_max], y_range=[P_min, P_max], title='SVM Decision Surface',
        x_axis_label='Temperature', y_axis_label='Pressure', tools='box_zoom,reset,hover', 
        width=600, plot_height=600)
    dec_fig.title_text_font_size = '18pt'
    dec_fig.xaxis.axis_label_text_font_size = '14pt'
    dec_fig.yaxis.axis_label_text_font_size = '14pt'
    dec_fig.image(image=[-ttf], x=[T_min], y=[P_min], dw=[T_max - T_min], dh=[P_max - P_min], 
        palette='RdYlGn8')
    dec_fig.x('x', 'y', size='size', source=source, fill_alpha=0.5, fill_color='navy', 
        line_color='navy', line_width=1, line_alpha=0.5)
    hover = dec_fig.select(dict(type=HoverTool))
    hover.tooltips = [
        ("Temperature", "@x"),
        ("Pressure", "@y"),
        ("measured lifetime", "@ttf"),
    ]

    #plot earnings vs time
    source = ColumnDataSource(
        data=dict(
            t = money.index,
            earnings = money.cumulative_earnings/1.e6,
            expenses = money.cumulative_expenses/1.e6,
            revenue  = money.cumulative_revenue/1.e6,
            zero = money.cumulative_revenue*0,
        )
    )
    earn_fig = figure(title='Cumulative Earnings & Expenses', x_axis_label='Time', 
        y_axis_label='Earnings & Expenses    (M$)', tools='box_zoom,reset,hover', 
        width=1000, plot_height=300, x_range=[0, 1200], y_range=[0, 120])
    earn_fig.title_text_font_size = '15pt'
    earn_fig.xaxis.axis_label_text_font_size = '11pt'
    earn_fig.yaxis.axis_label_text_font_size = '11pt'
    earn_fig.line('t', 'earnings', color='blue', source=source, line_width=5, legend='earnings')
    earn_fig.line('t', 'expenses', color='red', source=source, line_width=5, legend='expenses')
    earn_fig.patch([0, 200, 200, 0], [0, 0, 120, 120], color='lightsalmon', alpha=0.35, 
        line_width=0)
    earn_fig.patch([200, 400, 400, 200], [0, 0, 120, 120], color='gold', alpha=0.35, 
        line_width=0)
    earn_fig.patch([400, 1200, 1200, 400], [0, 0, 120, 120], color='darkseagreen', 
        alpha=0.35, line_width=0) 
    earn_fig.text([45], [101], ['run-to-fail'])
    earn_fig.text([245], [101], ['scheduled'])
    earn_fig.text([245], [91], ['maintenance'])
    earn_fig.text([445], [101], ['predictive'])
    earn_fig.text([445], [91], ['maintenance'])
    hover = earn_fig.select(dict(type=HoverTool))
    hover.tooltips = [
        ("         Time", "@t"),
        (" earning (M$)", "@earnings"),
        ("expenses (M$)", "@expenses"),
    ]

    #plot revenue vs time
    rev_fig = figure(title='Cumulative Revenue', x_axis_label='Time', 
        y_axis_label='Revenue    (M$)', tools='box_zoom,reset,hover', 
        width=1000, plot_height=300, x_range=[0, 1200], y_range=[-15, 10])
    rev_fig.title_text_font_size = '15pt'
    rev_fig.xaxis.axis_label_text_font_size = '11pt'
    rev_fig.yaxis.axis_label_text_font_size = '11pt'
    rev_fig.line('t', 'revenue', color='green', source=source, line_width=5, legend='revenue')
    rev_fig.line('t', 'zero', color='purple', source=source, line_width=3, alpha=0.5, 
        line_dash=[10, 5])
    rev_fig.patch([0, 200, 200, 0], [-15, -15, 10, 10], color='lightsalmon', alpha=0.35, 
        line_width=0)
    rev_fig.patch([200, 400, 400, 200], [-15, -15, 10, 10], color='gold', alpha=0.35, 
        line_width=0)
    rev_fig.patch([400, 1200, 1200, 400], [-15, -15, 10, 10], color='darkseagreen', 
        alpha=0.35, line_width=0)        
    hover = rev_fig.select(dict(type=HoverTool))
    hover.tooltips = [
        ("         Time", "@t"),
        (" revenue (M$)", "@revenue"),
    ]
    
    #export plot to html and return
    plot_grid = vplot(dec_fig, earn_fig, rev_fig)
    show(plot_grid, new='tab')
    return money, events
