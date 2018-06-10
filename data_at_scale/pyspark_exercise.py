
# coding: utf-8

# # MapReduce using SPARK

# In[1]:


get_ipython().magic('pylab inline')
import pandas as pd
import seaborn as sns
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)


# # Table of Contents
# 
# * [SPARK](#SPARK)
#     * Installing Spark locally
# * [Spark Context](#Spark-Context)
#     * [Create A RDD](#Create-A-RDD)
#     * [Call `collect` on an RDD: Lazy Spark](#Call-collect-on-an-RDD:-Lazy-Spark)
#     * [Operations on RDDs](#Operations-on-RDDs)
#     * [Word Examples](#Word-Examples)
#     * [Key Value Pairs](#Key-Value-Pairs)
#     * [word count 1](#word-count-1)
#     * [word count 2:  `reduceByKey()`](#word-count-2:--reduceByKey%28%29)
#     * [Nested Syntax](#Nested-Syntax)
#     * [Using Cache](#Using-Cache)
#     * [Fun with words](#Fun-with-words)
#     * [DataFrames](#DataFrames)
#     * [Machine Learning](#Machine-Learning)
# 

# With shameless stealing of some code and text from:
# 
# - https://github.com/tdhopper/rta-pyspark-presentation/blob/master/slides.ipynb
# - Databricks and Berkeley Spark MOOC: https://www.edx.org/course/introduction-big-data-apache-spark-uc-berkeleyx-cs100-1x
# 
# which you should go check out.

# ## Installing Spark locally
# 
# 
# **Step 1: Install Apache Spark**
# 
# For example, for Mac users using Homebrew:
# 
# ```
# $ brew install apache-spark
# ```

# **Step 2: Install the Java SDK version 1.8 or above for your platform (not just the JRE runtime)**
# 
# Make sure you can access commands such as `java` on your command line.

# **Step 3: Install the latest findspark package using pip**
# 
# ```
# ➜  ~  pip install findspark
# Collecting findspark
#   Downloading findspark-0.0.5-py2.py3-none-any.whl
# Installing collected packages: findspark
# Successfully installed findspark-0.0.5
# ```

# # Spark Context
# 
# You can also use it directly from the notebook interface on the mac if you installed `apache-spark` using `brew` and also installed `findspark` above.

# In[2]:


import findspark
findspark.init()
import pyspark
# The following line is not needed because a SparkContext was already initiated
#   by running 'pyspark' in terminal in order to initiate this jupyter notebook instance.
# sc = pyspark.SparkContext()


# It also outputs a bunch of stuff on my terminal. This is because the entire java context is started up.
# 
# ```Using Spark's default log4j profile: org/apache/spark/log4j-defaults.properties
# 15/10/21 14:46:15 INFO SparkContext: Running Spark version 1.4.0
# 2015-10-21 14:46:15.774 java[30685:c003] Unable to load realm info from SCDynamicStore
# 15/10/21 14:46:15 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
# 15/10/21 14:46:15 INFO SecurityManager: Changing view acls to: rahul
# 15/10/21 14:46:15 INFO SecurityManager: Changing modify acls to: rahul
# 15/10/21 14:46:15 INFO SecurityManager: SecurityManager: authentication disabled; ui acls disabled; users with view permissions: Set(rahul); users with modify permissions: Set(rahul)
# 15/10/21 14:46:16 INFO Slf4jLogger: Slf4jLogger started
# 15/10/21 14:46:16 INFO Remoting: Starting remoting
# 15/10/21 14:46:16 INFO Remoting: Remoting started; listening on addresses :[akka.tcp://sparkDriver@10.251.101.163:64359]
# 15/10/21 14:46:16 INFO Utils: Successfully started service 'sparkDriver' on port 64359.
# 15/10/21 14:46:16 INFO SparkEnv: Registering MapOutputTracker
# 15/10/21 14:46:16 INFO SparkEnv: Registering BlockManagerMaster
# 15/10/21 14:46:16 INFO DiskBlockManager: Created local directory at /private/var/folders/_f/y76rs29s3c57ykwyz9c8z12c0000gn/T/spark-00a4e09e-e5db-485f-81dc-2e5016e9a27e/blockmgr-8966e07c-223b-4c38-9273-11543aa9d3c1
# 15/10/21 14:46:16 INFO MemoryStore: MemoryStore started with capacity 273.0 MB
# 15/10/21 14:46:16 INFO HttpFileServer: HTTP File server directory is /private/var/folders/_f/y76rs29s3c57ykwyz9c8z12c0000gn/T/spark-00a4e09e-e5db-485f-81dc-2e5016e9a27e/httpd-6af0a9e0-1cfe-42c4-a1bd-e01715b98436
# 15/10/21 14:46:16 INFO HttpServer: Starting HTTP Server
# 15/10/21 14:46:17 INFO Utils: Successfully started service 'HTTP file server' on port 64360.
# 15/10/21 14:46:17 INFO SparkEnv: Registering OutputCommitCoordinator
# 15/10/21 14:46:18 INFO Utils: Successfully started service 'SparkUI' on port 4040.
# 15/10/21 14:46:18 INFO SparkUI: Started SparkUI at http://10.251.101.163:4040
# 15/10/21 14:46:18 INFO Executor: Starting executor ID driver on host localhost
# 15/10/21 14:46:18 INFO Utils: Successfully started service 'org.apache.spark.network.netty.NettyBlockTransferService' on port 64361.
# 15/10/21 14:46:18 INFO NettyBlockTransferService: Server created on 64361
# 15/10/21 14:46:18 INFO BlockManagerMaster: Trying to register BlockManager
# 15/10/21 14:46:18 INFO BlockManagerMasterEndpoint: Registering block manager localhost:64361 with 273.0 MB RAM, BlockManagerId(driver, localhost, 64361)
# 15/10/21 14:46:18 INFO BlockManagerMaster: Registered BlockManager
# ```

# In[3]:


sc


# In[4]:


sc.parallelize([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).map(lambda x: x**2).sum()


# ### Create A RDD
# 

# In[5]:


wordsList = ['cat', 'elephant', 'rat', 'rat', 'cat']
wordsRDD = sc.parallelize(wordsList, 4)
# Print out the type of wordsRDD
print(type(wordsRDD))


# ### Call `collect` on an RDD: Lazy Spark

# Spark is lazy. Until you `collect`, nothing is actually run.
# 
# >Instead, they just remember the transformations applied to some base dataset (e.g. a file). The transformations are only computed when an action requires a result to be returned to the driver program.

# In[6]:


wordsRDD.collect()


# ```
# 15/10/21 14:59:59 INFO SparkContext: Starting job: collect at <ipython-input-6-dee494da0714>:1
# 15/10/21 14:59:59 INFO DAGScheduler: Got job 0 (collect at <ipython-input-6-dee494da0714>:1) with 4 output partitions (allowLocal=false)
# 15/10/21 14:59:59 INFO DAGScheduler: Final stage: ResultStage 0(collect at <ipython-input-6-dee494da0714>:1)
# 15/10/21 14:59:59 INFO DAGScheduler: Parents of final stage: List()
# 15/10/21 14:59:59 INFO DAGScheduler: Missing parents: List()
# 15/10/21 14:59:59 INFO DAGScheduler: Submitting ResultStage 0 (ParallelCollectionRDD[0] at parallelize at PythonRDD.scala:396), which has no missing parents
# 15/10/21 15:00:00 INFO MemoryStore: ensureFreeSpace(1224) called with curMem=0, maxMem=286300569
# 15/10/21 15:00:00 INFO MemoryStore: Block broadcast_0 stored as values in memory (estimated size 1224.0 B, free 273.0 MB)
# 15/10/21 15:00:00 INFO MemoryStore: ensureFreeSpace(777) called with curMem=1224, maxMem=286300569
# 15/10/21 15:00:00 INFO MemoryStore: Block broadcast_0_piece0 stored as bytes in memory (estimated size 777.0 B, free 273.0 MB)
# 15/10/21 15:00:00 INFO BlockManagerInfo: Added broadcast_0_piece0 in memory on localhost:64361 (size: 777.0 B, free: 273.0 MB)
# 15/10/21 15:00:00 INFO SparkContext: Created broadcast 0 from broadcast at DAGScheduler.scala:874
# 15/10/21 15:00:00 INFO DAGScheduler: Submitting 4 missing tasks from ResultStage 0 (ParallelCollectionRDD[0] at parallelize at PythonRDD.scala:396)
# 15/10/21 15:00:00 INFO TaskSchedulerImpl: Adding task set 0.0 with 4 tasks
# 15/10/21 15:00:00 INFO TaskSetManager: Starting task 0.0 in stage 0.0 (TID 0, localhost, PROCESS_LOCAL, 1379 bytes)
# 15/10/21 15:00:00 INFO TaskSetManager: Starting task 1.0 in stage 0.0 (TID 1, localhost, PROCESS_LOCAL, 1384 bytes)
# 15/10/21 15:00:00 INFO TaskSetManager: Starting task 2.0 in stage 0.0 (TID 2, localhost, PROCESS_LOCAL, 1379 bytes)
# 15/10/21 15:00:00 INFO TaskSetManager: Starting task 3.0 in stage 0.0 (TID 3, localhost, PROCESS_LOCAL, 1403 bytes)
# 15/10/21 15:00:00 INFO Executor: Running task 0.0 in stage 0.0 (TID 0)
# 15/10/21 15:00:00 INFO Executor: Running task 2.0 in stage 0.0 (TID 2)
# 15/10/21 15:00:00 INFO Executor: Running task 1.0 in stage 0.0 (TID 1)
# 15/10/21 15:00:00 INFO Executor: Running task 3.0 in stage 0.0 (TID 3)
# 15/10/21 15:00:00 INFO Executor: Finished task 1.0 in stage 0.0 (TID 1). 646 bytes result sent to driver
# 15/10/21 15:00:00 INFO Executor: Finished task 3.0 in stage 0.0 (TID 3). 665 bytes result sent to driver
# 15/10/21 15:00:00 INFO Executor: Finished task 0.0 in stage 0.0 (TID 0). 641 bytes result sent to driver
# 15/10/21 15:00:00 INFO Executor: Finished task 2.0 in stage 0.0 (TID 2). 641 bytes result sent to driver
# 15/10/21 15:00:00 INFO TaskSetManager: Finished task 1.0 in stage 0.0 (TID 1) in 61 ms on localhost (1/4)
# 15/10/21 15:00:00 INFO TaskSetManager: Finished task 3.0 in stage 0.0 (TID 3) in 60 ms on localhost (2/4)
# 15/10/21 15:00:00 INFO TaskSetManager: Finished task 0.0 in stage 0.0 (TID 0) in 96 ms on localhost (3/4)
# 15/10/21 15:00:00 INFO TaskSetManager: Finished task 2.0 in stage 0.0 (TID 2) in 63 ms on localhost (4/4)
# 15/10/21 15:00:00 INFO DAGScheduler: ResultStage 0 (collect at <ipython-input-6-dee494da0714>:1) finished in 0.120 s
# 15/10/21 15:00:00 INFO TaskSchedulerImpl: Removed TaskSet 0.0, whose tasks have all completed, from pool
# 15/10/21 15:00:00 INFO DAGScheduler: Job 0 finished: collect at <ipython-input-6-dee494da0714>:1, took 0.872367 s
# ```

# ### Operations on RDDs

# From the Spark Programming Guide:
# 
# >RDDs support two types of operations: transformations, which create a new dataset from an existing one, and actions, which return a value to the driver program after running a computation on the dataset. For example, map is a transformation that passes each dataset element through a function and returns a new RDD representing the results. On the other hand, reduce is an action that aggregates all the elements of the RDD using some function and returns the final result to the driver program (although there is also a parallel reduceByKey that returns a distributed dataset).

# ### Word Examples

# In[7]:


def makePlural(word):
    return word + 's'

print(makePlural('cat'))


# Transform one RDD into another.

# In[8]:


pluralRDD = wordsRDD.map(makePlural)
print(pluralRDD.first())
print(pluralRDD.take(2))


# In[9]:


pluralRDD.take(1)


# In[10]:


pluralRDD.collect()


# ### Key Value Pairs

# In[11]:


wordPairs = wordsRDD.map(lambda w: (w, 1))
print(wordPairs.collect())


# ```
# ➜  sparklect  ps auxwww | grep pyspark
# rahul           30685   0.4  0.8  3458120  68712 s012  S+    2:46PM   2:00.21 /System/Library/Java/JavaVirtualMachines/1.6.0.jdk/Contents/Home/bin/java -cp /usr/local/opt/apache-spark/libexec/conf/:/usr/local/opt/apache-spark/libexec/lib/spark-assembly-1.4.0-hadoop2.6.0.jar:/usr/local/opt/apache-spark/libexec/lib/datanucleus-api-jdo-3.2.6.jar:/usr/local/opt/apache-spark/libexec/lib/datanucleus-core-3.2.10.jar:/usr/local/opt/apache-spark/libexec/lib/datanucleus-rdbms-3.2.9.jar -Xms512m -Xmx512m -XX:MaxPermSize=128m org.apache.spark.deploy.SparkSubmit pyspark-shell
# rahul           31520   0.0  0.0  2432784    480 s011  R+    6:42PM   0:00.00 grep --color=auto --exclude-dir=.bzr --exclude-dir=.cvs --exclude-dir=.git --exclude-dir=.hg --exclude-dir=.svn pyspark
# rahul           31494   0.0  0.7  2548972  57288 s012  S     6:41PM   0:00.10 python -m pyspark.daemon
# rahul           31493   0.0  0.7  2548972  57308 s012  S     6:41PM   0:00.10 python -m pyspark.daemon
# rahul           31492   0.0  0.7  2548972  57288 s012  S     6:41PM   0:00.11 python -m pyspark.daemon
# rahul           31446   0.0  0.8  2548972  68460 s012  S     6:35PM   0:01.34 python -m pyspark.daemon
# ```

# ### WORD COUNT!
# 
# This little exercise shows how to use mapreduce to calculate the counts of individual words in a list.

# In[12]:


wordsList = ['cat', 'elephant', 'rat', 'rat', 'cat']
wordsRDD = sc.parallelize(wordsList, 4)
wordCountsCollected = (wordsRDD
                       .map(lambda w: (w, 1))
                       .reduceByKey(lambda x,y: x+y)
                       .collect())
print(wordCountsCollected)


# ![Tons of shuffling](https://databricks.gitbooks.io/databricks-spark-knowledge-base/content/images/reduce_by.png)

# In[13]:


print((wordsRDD
    .map(lambda w: (w, 1))
    .reduceByKey(lambda x,y: x+y)).toDebugString())


# ### Using Cache

# In[14]:


wordsList = ['cat', 'elephant', 'rat', 'rat', 'cat']
wordsRDD = sc.parallelize(wordsList, 4)
print(wordsRDD)
wordsRDD.count()


# Normally, every operation is run from the start. This may be inefficient in many cases. So when appropriate, we may want to cache the result the first time an operation is run on an RDD.

# In[15]:


#this is rerun from the start
wordsRDD.count()


# In[16]:


#default storage level (MEMORY_ONLY)
wordsRDD.cache()#nothing done this is still lazy


# In[17]:


#parallelize is rerun and cached because we told it to cache
wordsRDD.count()


# In[18]:


#this `sc.parallelize` is not rerun in this case
wordsRDD.count()


# Where is this useful: it is when you have branching parts or loops, so that you dont do things again and again. Spark, being "lazy" will rerun the chain again. So `cache` or `persist` serves as a checkpoint, breaking the RDD chain or the *lineage*.

# In[19]:


birdsList=['heron','owl']
animList=wordsList+birdsList
animaldict={}
for e in wordsList:
    animaldict[e]='mammal'
for e in birdsList:
    animaldict[e]='bird'
animaldict


# In[20]:


animsrdd = sc.parallelize(animList, 4)
animsrdd.cache()
#below runs the whole chain but causes cache to be populated
mammalcount=animsrdd.filter(lambda w: animaldict[w]=='mammal').count()
#now only the filter is carried out
birdcount=animsrdd.filter(lambda w: animaldict[w]=='bird').count()
print(mammalcount, birdcount)


# ### Exercises: Fun with MapReduce
# 
# Read http://spark.apache.org/docs/latest/programming-guide.html for some useful background and then try out the following exercises

# The file `./sparklect/english.stop.txt` contains a list of English stopwords, while the file `./sparklect/shakes/juliuscaesar.txt` contains the entire text of Shakespeare's 'Julius Caesar'.
# 
# * Load all of the stopwords into a Python list
# * Load the text of Julius Caesar into an RDD using the `sparkcontext.textfile()` method. Call it `juliusrdd`.

# In[21]:


# your turn

# Using .collect() creates an array
stopwords = sc.textFile('./sparklect/english.stop.txt').collect()
# Whereas not using it leaves the object as an RDD
juliusrdd = sc.textFile('./sparklect/shakes/juliuscaesar.txt')


# How many words does Julius Caesar have? *Hint: use `flatMap()`*.

# In[22]:


# your turn
# Delineate words via spaces
wordnumrdd = juliusrdd.flatMap(lambda x: (x.split(" ")))

# Import packages to remove punctation
from string import punctuation

# Also make all words have only lowercase letters
wordnumrdd = wordnumrdd.filter(lambda x: x not in punctuation)                         .map(lambda x: str.strip(str.strip(str.lower(x)),punctuation))

print(wordnumrdd.count())


# Now print the first 20 words of Julius Caesar as a Python list.

# In[23]:


# your turn

# The .take(n) method returns an array of the first n entries of the RDD
wordnumrdd.take(20)


# Now print the first 20 words of Julius Caesar, **after removing all the stopwords**. *Hint: use `filter()`*.

# In[24]:


# your turn

# remove the stopwords
wordfilteredrdd = wordnumrdd.filter(lambda x: x not in stopwords)

# first 20 words
wordfilteredrdd.take(20)


# Now, use the word counting MapReduce code you've seen before. Count the number of times each word occurs and print the top 20 results as a list of tuples of the form `(word, count)`. *Hint: use `takeOrdered()` instead of `take()`*

# In[25]:


# your turn

# takeOrdered naturally sorts in ascending order,
#    so we sort by their negatives to get the top values.
topwords = wordfilteredrdd.map(lambda x: (x,1))                             .reduceByKey(lambda x,y: x+y)                             .takeOrdered(20,key=lambda x: -x[1])
topwords


# Plot a bar graph. For each of the top 20 words on the X axis, represent the count on the Y axis.

# In[26]:


# your turn

import matplotlib.pyplot as plt

words = [x[0] for x in topwords]
frequency = [x[1] for x in topwords]

plt.figure(figsize=(20,10))
plt.bar(words, frequency)
plt.xlabel('Word')
plt.ylabel('Frequency')
plt.title('Most Common non-Stop Words in Julius Caesar')
plt.show()


# ### Using partitions for parallelization

# In order to make your code more efficient, you want to use all of the available processing power, even on a single laptop. If your machine has multiple cores, you can tune the number of partitions to use all of them! From http://www.stat.berkeley.edu/scf/paciorek-spark-2014.html:
# 
# >You want each partition to be able to fit in the memory availalbe on a node, and if you have multi-core nodes, you want that as many partitions as there are cores be able to fit in memory.
# 
# >For load-balancing you'll want at least as many partitions as total computational cores in your cluster and probably rather more partitions. The Spark documentation suggests 2-4 partitions (which they also seem to call slices) per CPU. Often there are 100-10,000 partitions. Another rule of thumb is that tasks should take at least 100 ms. If less than that, you may want to repartition to have fewer partitions.

# In[27]:


shakesrdd=sc.textFile("./sparklect/shakes/*.txt", minPartitions=4)


# In[28]:


shakesrdd.take(10)


# Now calculate the top 20 words in all of the files that you just read.

# In[29]:


# your turn

# Repeat the same process as above all in one step:
#   remove punctuation, remove white space, make everything lower case, and remove stop words.
wordsrdd = shakesrdd.flatMap(lambda x: (x.split(" ")))     .filter(lambda x: x not in punctuation)     .map(lambda x: str.strip(str.strip(str.lower(x)),punctuation))     .filter(lambda x: x not in stopwords)     .map(lambda x: (x,1))     .reduceByKey(lambda x,y: x+y)     .takeOrdered(20,key=lambda x: -x[1])
    
wordsrdd


# ## Optional topic 1: DataFrames
# 
# Pandas and Spark dataframes can be easily converted to each other, making it easier to work with different data formats. This section shows some examples of each.

# Convert Spark DataFrame to Pandas
# 
# `pandas_df = spark_df.toPandas()`
# 
# Create a Spark DataFrame from Pandas
# 
# `spark_df = context.createDataFrame(pandas_df)`
# 
# Must fit in memory.
# 
# ![](https://ogirardot.files.wordpress.com/2015/05/rdd-vs-dataframe.png?w=640&h=360)
# 
# VERY IMPORTANT: DataFrames in Spark are like RDD in the sense that they’re an immutable data structure.

# In[30]:


df=pd.read_csv("sparklect/01_heights_weights_genders.csv")
df.head()


# Convert this pandas dataframe to a Spark dataframe

# In[31]:


from pyspark.sql import SQLContext
sqlsc=SQLContext(sc)
sparkdf = sqlsc.createDataFrame(df)
sparkdf


# In[32]:


sparkdf.show(5)


# In[33]:


type(sparkdf.Gender)


# In[34]:


# You can't call .map() on a DataFrame directly - you first have to convert it into an RDD
temp = sparkdf.rdd.map(lambda r: r.Gender)
print(type(temp))
temp.take(10)


# ## Optional topic 2: Machine Learning using Spark
# 
# 

# In[35]:


from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.regression import LabeledPoint


# Now create a data set from the Spark dataframe

# In[36]:


data=sparkdf.rdd.map(lambda row: LabeledPoint(row.Gender=='Male',[row.Height, row.Weight]))
data.take(5)


# In[37]:


data2=sparkdf.rdd.map(lambda row: LabeledPoint(row[0]=='Male',row[1:]))
data2.take(1)[0].label, data2.take(1)[0].features


# Split the data set into training and test sets

# In[38]:


train, test = data.randomSplit([0.7,0.3])
train.cache()
test.cache()


# In[39]:


type(train)


# Train the logistic regression model using MLIB

# In[40]:


model = LogisticRegressionWithLBFGS.train(train)


# In[41]:


model.weights


# Run it on the test data

# In[42]:


results = test.map(lambda lp: (lp.label, float(model.predict(lp.features))))
print(results.take(10))
type(results)                       


# Measure accuracy and other metrics

# In[43]:


# As of Python 3.x, lambda functions no longer accept tuples as arguments.
# test_accuracy=results.filter(lambda (a,p): a==p).count()/float(results.count())
# test_accuracy


# In[44]:


from pyspark.mllib.evaluation import BinaryClassificationMetrics
metrics = BinaryClassificationMetrics(results)


# In[45]:


print(type(metrics))
metrics.areaUnderROC


# In[46]:


type(model)


# In[47]:


get_ipython().system('rm -rf mylogistic.model')


# In[48]:


model.save(sc, "mylogistic.model")


# The pipeline API automates a lot of this stuff, allowing us to work directly on dataframes.

# Also see:
# 
# - http://jordicasanellas.weebly.com/data-science-blog/machine-learning-with-spark
# - http://spark.apache.org/docs/latest/mllib-guide.html
# - http://www.techpoweredmath.com/spark-dataframes-mllib-tutorial/
# - http://spark.apache.org/docs/latest/api/python/
# - http://spark.apache.org/docs/latest/programming-guide.html

# `rdd.saveAsTextFile()` saves an RDD as a string.

# ## Optional Topic 3: Your Turn at Machine Learning! :)
# 
# For this exercise, we're going to use one of the datasets we've already worked with: the Boston House Prices dataset. We're going to try a couple of regression algorithms, but from the SparkML library this time.
# 
# Before you proceed, make sure to do an overview of the documentation: 
# http://spark.apache.org/docs/latest/api/python/pyspark.ml.html

# In[49]:


# All imports go here




# First, we have to load the dataset, which resides as a CSV file in the folder for this exercise.

# In[50]:


# Path: /sparklect/boston.csv


# Inspect the data to make sure everything is loaded properly.

# Now we'll need to create a train/test split.

# In[51]:


# We'll first have to vectorize the features


# As the next step, fit a Linear Regression model on the training set.

# Now validate the model on the test set, and check the Root Mean Squared Error.

# Let's compare Linear Regression with a more powerful algorithm - the Random Forest. As the Random Forest has several hyperparameters that can be tuned for maximum accuracy, we're going to need to use k-fold Cross Validation.

# First, set up a grid for the hyperparameter search.

# Now, with a Random Forest regressor using k-fold Cross Validation, and find the optimal combination of hyperparameters.

# Finally, validate the model on the test set and check the Root Mean Squared Error again.

# 
# 
# ## Optional Topic 4: Model Your Capstone Dataset
# 

# If you have time, load up the cleaned dataset from one of your capstone projects. Do you remember which algorithm and the accompanying combination of hyperparameters did the best job? For practice, try and implement the same model in SparkML.

# In[52]:


# Stop Spark
sc.stop()

