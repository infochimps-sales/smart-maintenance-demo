## The Smart Maintenance Demo

by Joe Hahn,
joe.hahn@infochimps.com,
20 May 2015

This is the Github repository for the master branch of the Smart Maintenance Demo for Hadoop.
This demo uses the Support Vector Machines (SVM) algorithm to perform predictive
maintenance on 200 simulated motors, with most (but not all) of the computation being done in
parallell on the Hadoop cluster's datanodes via Spark.

###To install:

First clone this github repo to your home directory on the hadoop foyer node:

    cd; git clone git@github.com:infochimps-sales/spark-airline-demo.git 
    
   
Then execute the installer, this will download and install some python libraries to all 
hadoop nodes, and is done in 5 minutes:

    cd spark-airline-demo
    ./install.sh


###To execute:

To submit this spark job to Yarn for execution:

    PYSPARK_PYTHON=/home/$USER/anaconda/bin/python spark-submit smart_maint_spark.py
    

Monitor this job's progress using the Spark UI by browsing:

    Cloudera Manager -> Home -> Yarn -> Resource Manager UI -> application_ID# -> Application Master


The output of this spark job is 3 png images that can be viewed by browsing

    http://cdh-foyer.platform.infochimps:12321/figs
    

###The demo's storyline:

This demo simulations the repair history of 200 simulated motors over time. Initially the
motors are evolved using a 'run-to-fail' maintenance strategy.  
Each motor has two knobs, Pressure (P) and Temperature (T),
and the size of the dots in the following scatterplot
shows that the longest-lived motors have (P,T) settings in this interval: 40 < P < 60 and T < 100,
with motors being progressively shorter-lived the further away their P,T setting are from this
sweet spot at P~50 and T<100: 

![](https://github.com/infochimps-sales/smart-maintenance-demo/blob/master/figs/fail_factor.png)

The SVM algorithm is then trained on these date, namely, the observed engine lifetime versus
engine (P,T). The now-trained SVM algorithm is now able to use an engine's (P,T) settings to
predict that engine's lifetime ie its estimated time-to-fail. Thereafter (a times t > 600) the
engines are run in predictive-maintenance mode, which simply sends an engine into maintenance
when its estimated time-to-fail is one day hence. The following contour map shows the SVM's
so-called prediction surface map's the engine's predicted time-to-fail versus the engine's (P,T)
settings; see http://cdh-foyer.platform.infochimps:12321/figs/predicted_time_to_fail.png. Note
that SVM's predicted time-to-fail does indeed recover the engine's 'finger of stability'
shown above at 40 < P < 60 and T < 100.

![](https://github.com/infochimps-sales/smart-maintenance-demo/blob/master/figs/predicted_time_to_fail.png)

This diagram shows...

Motors generate earning while
running, and they accrue some expenses when being maintained and greater expenses while
they are being repaired after a failure. 
![](https://github.com/infochimps-sales/smart-maintenance-demo/blob/master/figs/revenue.png)


###Known Issues:


1 If the png images are not browse-able, restart the webserver on the hadoop foyer node:

    /home/$USER/anaconda/bin/python -m SimpleHTTPServer 12321 > /dev/null 2>&1 &


2 Close inspection of Spark's Application Master UI will show that this job is being executed
on only 2 of the 3 available datanodes, I have no idea why one datanode is not participating,
this needs to be debugged.

3 Spark's console output is *way* too verbose, I attempted to dial that down on foyer node via:

    sudo cp /opt/cloudera/parcels/CDH-5.3.0-1.cdh5.3.0.p0.30/etc/spark/conf.dist/log4j.properties.template \
        /opt/cloudera/parcels/CDH-5.3.0-1.cdh5.3.0.p0.30/etc/spark/conf.dist/log4j.properties


and in log4j.properties set

    log4j.rootCategory=WARN, console


but the above didn't help any...perhaps one must do this on all datanodes?


###Debugging Notes:
        
    
One can execute this demo line-by-line at the python command line using pyspark,
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
