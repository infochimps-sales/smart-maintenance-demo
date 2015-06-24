## The Smart Maintenance Demo

by Joe Hahn,
joe.hahn@infochimps.com,
20 May 2015

This is the Github repository for the master branch of the Smart Maintenance Demo for Hadoop.
This demo uses python's scikit-learn machine-learning algorithm to perform predictive
maintenance on 200 simulated motors, with much (but not all) of the computation being done in
parallell across the Hadoop cluster's datanodes using Spark.

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



###Known Issues:


1 If the png images are not browse-able, restart the webserver on the hadoop foyer node:

    /home/$USER/anaconda/bin/python -m SimpleHTTPServer 12321 > /dev/null 2>&1 &


2 Close inspection of Spark's Application Master UI will show that this job is being executed
on only 2 of the 3 available datanodes, I have no idea why 1 datanode is not participating,
this needs to be debugged.

3 Spark's console output is *way* too verbose, I attempted to dial that down on foyer node via:

    sudo cp /opt/cloudera/parcels/CDH-5.3.0-1.cdh5.3.0.p0.30/etc/spark/conf.dist/log4j.properties.template \
        /opt/cloudera/parcels/CDH-5.3.0-1.cdh5.3.0.p0.30/etc/spark/conf.dist/log4j.properties


and in log4j.properties set

    log4j.rootCategory=WARN, console


but the above didn't help any...maybe to this on all datanodes?


###Debugging Notes:
        
    
One can execute this demo line-by-line at the python command line, using pyspark
(this is useful for debugging):

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
