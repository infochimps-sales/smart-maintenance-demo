## The Smart Maintenance Demo

by Joe Hahn,
joe.hahn@infochimps.com,
20 May 2015

This is the Github repository for the master branch of the Smart Maintenance Demo for Hadoop.
This demo uses the Support Vector Machines (SVM) algorithm to perform predictive
maintenance on 200 simulated motors, with most of the computations being done in
parallell across the Hadoop cluster's datanodes using Spark. Python's Bokeh library
is also used to generate an interactive dashboard that visualizes the results of this
simulation.

###To install:

First clone this github repo to your home directory on the hadoop foyer node:

    cd; git clone git@github.com:infochimps-sales/smart-maintenance-demo.git 
    
   
Then execute the installer, this will download and install some python libraries to all 
hadoop nodes, and is done in 5 minutes:

    cd smart-maintenance-demo
    ./install.sh


###To execute:

To submit this spark job to Yarn for execution:

    PYSPARK_PYTHON=/home/$USER/anaconda/bin/python spark-submit \
        --master yarn-client --num-executors 3 --executor-cores 6 --executor-memory 4G \
        --driver-java-options "-Dlog4j.configuration=file:///home/$USER/smart-maintenance-demo/log4j.warn-only.properties" \
        smart_maint.py


Monitor this job's progress using the Spark UI by browsing:

    Cloudera Manager -> Home -> Yarn -> Resource Manager UI -> application_ID# -> Application Master


After the spark job completes, execute this script to dashboard the results of the 
smart-maintenance simulation,

    /home/$USER/anaconda/bin/python dashboard.py > /dev/null


and then browse the resulting dashboard at

    http://cdh-foyer.platform.infochimps:12321/dashboard.html


###The demo's storyline:

This demo calculates the operational history of 200 simulated motors over time. Initially these
motors are evolved using a _run-to-fail_ maintenance strategy, and the motor data collected
during this period is shown below. Each motor has two knobs,
Pressure (P) and Temperature (T), and motors having P,T settings in the interval
40 < P < 60 and T < 100 are by design longest lived, while motors 
having P,T setting further from the
sweet spot at P~50 and T<100 are progressively shorter lived. This trend is also
indicated by the size of the crosses in the following scatterplot, which shows
how engine lifetime varies with P,T.

![](https://github.com/infochimps-sales/smart-maintenance-demo/blob/master/slides/decision_surface.png)

All plots in this demo's dashboard (see http://cdh-foyer.platform.infochimps:12321/dashboard.html)
are interactive; click-drag to zoom in on a region, and mouse-over individual data
to see their their values.

The demo evolves these motors in run-to-fail mode until time t=200, and then (just for kicks)
it switches to a _scheduled-maintenance_ strategy during times 200 < t < 400.
During scheduled-maintenance operation, every engine is sent to maintenance every 5 days,
this simply removes some cruft and temporarily reduces the likelihood of motor failure.
Meanwhile the SVM algorithm is trained on the run-to-fail data, which is simply the observed
engine lifetimes versus their (P,T) settings. Once trained, the SVM algorithm is now 
able to use an engine's (P,T) settings to predict that engine's lifetime ie its 
estimated time-to-fail. Thereafter (at times t > 400) the engines are evolved using
_predictive-maintenance_, which simply sends an engine into maintenance
when its predicted time-to-fail is one day hence. The coloring in the above plot
also shows the SVM's
so-called _prediction surface_, which map's the engines' predicted time-to-fail across the
P,T parameter space. Note that the SVM's predicted time-to-fail does indeed recover
the engines' sweet-spot at 40 < P < 60 and T < 100, though the edges of the predicted stable
zone are rather fluid.

Each operating engine also generate earnings at a rate of $1000/day, while engines that are
being maintained instead generate modest expenses (-$200/day), with failed engines generating
larger expenses (-$2000/day) while in the shop for repairs. The following plots show
that operating these engines in _run-to-fail_ mode is very expensive, resulting in
cumulative losses of -$13M by time t=200. This plot also shows that operating these
engines using a _scheduled-maintenance_ strategy is a wash, with earnings nearly balancing expenses.
But switching to a _predictive-maintenance_ strategy at t=400 then results in earnings that
exceeds expenses, so much so that the operators of these engines recover all lost earnings
by time t=870, and have earned $6M at the end of this simulation.

![](https://github.com/infochimps-sales/smart-maintenance-demo/blob/master/slides/revenue.png)

So this demo's main punchline is: _get Smart Maintenance on the BDPaas to optimize
equipment maintenance schedules and to  dramatically reduce expenses and grow earnings._

The following plot is merely some dashboard-fu. Click-drag to zoom in on region in the plot. 
Then click the _Box Select_ icon in the upper right and click-drag again. This plot is also
linked to the table below, and clicking on any column heading there will force the table
to update and display only data that is highlighted in the plot above.

![](https://github.com/infochimps-sales/smart-maintenance-demo/blob/master/slides/motors.png)



###Known issues:

If the dashboard is not visible in the browser, the webserver likely needs to be restarted:

    /home/$USER/anaconda/bin/python -m SimpleHTTPServer 12321 > /dev/null 2>&1 &


dashboard.py also launches an ELinks console that isn't needed, that code needs to be
tweaked so that doesn't happen.
 

###Debugging notes:
        

To benchmark spark's commandline settings:

    START=$(date +%s); \
    PYSPARK_PYTHON=/home/$USER/anaconda/bin/python spark-submit --master yarn-client \
        --num-executors 3 --executor-cores 6 --executor-memory 4G \
        --driver-java-options "-Dlog4j.configuration=file:///home/$USER/smart-maintenance-demo/log4j.warn-only.properties" \
        smart_maint.py; \
    echo "execution time (seconds) = "$(( $(date +%s) - $START ))


One can also execute this demo line-by-line at the python command line using pyspark,
this is useful when debugging code:

    PYSPARK_PYTHON=/home/$USER/anaconda/bin/python pyspark


Then copy-n-past each line from smart_maint_spark.py into the python command line, 
EXCEPT for line 25: sc = SparkContext(conf=conf... 

To get pyspark to use ipython (rather than python):

    sudo rm -f /usr/bin/python /usr/bin/ipython
    sudo ln -s /home/$USER/anaconda/bin/python /usr/bin/python
    sudo ln -s /home/$USER/anaconda/bin/ipython /usr/bin/ipython
    IPYTHON=1 pyspark


And to undo the above changes:
 
    sudo rm /usr/bin/python /usr/bin/ipython
    sudo ln -s /usr/bin/python2.6 /usr/bin/python


