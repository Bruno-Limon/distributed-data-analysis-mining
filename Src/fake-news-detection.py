# %% [markdown]
# # **<font color="#ff006a">DISTRIBUTED DATA ANALYSIS AND MINING - PROJECT</font>**
#
# By:
# - Bruno Limón
# - Rebecca Saitta
# - Charlotte Brimont
# - Giovanni Battista D’Orsi
#
# ---
#
#

# %% [markdown]
# # **<font color="#34ebdb">IMPORTS</font>**

# %%
# For Colab
# Install packages:
# !pip install pyspark
# !pip install sparknlp

# Or, setup pyspark environment
!wget http://setup.johnsnowlabs.com/colab.sh -O - | bash

# %%
# General
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
np.set_printoptions(suppress=True) # to avoid scientific notation while printing

from wordcloud import WordCloud, STOPWORDS
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, precision_recall_curve, average_precision_score, roc_auc_score

# Spark
from pyspark import SparkContext
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.types import FloatType

# Preprocessing
from pyspark.sql.functions import *
from pyspark.ml.stat import Correlation
from pyspark.ml.functions import vector_to_array
from pyspark.ml.feature import CountVectorizer, VectorAssembler, StringIndexer, RegexTokenizer, StopWordsRemover, OneHotEncoder

# Clustering
from pyspark.ml.clustering import LDA, KMeans, BisectingKMeans
from pyspark.ml.feature import PCA as PCAml, HashingTF, IDF

# Classification
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, DecisionTreeClassifier, LinearSVC, LinearSVCSummary

# Evaluation
from pyspark.ml.evaluation import BinaryClassificationEvaluator, ClusteringEvaluator
from pyspark.mllib.evaluation import BinaryClassificationMetrics, MulticlassMetrics

# Tuning
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# Frequent Pattern Mining
from pyspark.ml.fpm import FPGrowth

# %%
# For Colab, mount drive, making sure to have the files as a shortcut in your main drive
from google.colab import drive
drive.mount('/content/drive')

# %% [markdown]
# # **<font color="#34ebdb">1.0 DATA UNDERSTANDING & PREPROCESSING</font>**

# %%
# Initialize a new Spark Session
sc = SparkContext(appName="DDAM_Group6_context", master="local[*]")

spark = SparkSession.builder \
     .master('local[*]') \
     .appName('DDAM_Group6_session') \
     .getOrCreate()

sqlContext = SQLContext(spark)

# %%
# Choose between colab or local path
path = '/content/drive/MyDrive/' # Colab
# path = 'Data/'                 # local

# %% [markdown]
# ## **<font color="#FBBF44">1.1 FAKE DATASET</font>**

# %%
# loading the fake dataset
fake_data = spark.read.format("csv") .options(header='true',
             multiLine = 'True',
             inferschema='true',
             treatEmptyValuesAsNulls='true',
             escape='\"').load(path + 'Fake.csv')

print(fake_data.printSchema())
print(fake_data.count())
fake_data.show(5)

# %%
#see if we have only unique titles and texts
print('nb unique title:', fake_data.select('title').distinct().count())
print('nb unique text:', fake_data.select('text').distinct().count())
print('nb rows:', fake_data.count())

# %%
# Looking at the distinct values in 'subject' from fake news
fake_data.select('subject').distinct().collect()

# %%
# Collecting contents of dataset to see a full example of the text column
print(fake_data.collect()[0].__getitem__('text'))

# %% [markdown]
# ## **<font color="#FBBF44">1.1 TRUE DATASET</font>**

# %%
# loading the true dataset
true_data = spark.read.format("csv") .options(header='true',
             multiLine = 'True',
             inferschema='true',
             treatEmptyValuesAsNulls='true',
             escape='\"').load(path + 'True.csv')

print(true_data.printSchema())
print(true_data.count())
true_data.show(5)

# %%
#see if we have only unique titles and texts
print('nb unique title:', true_data.select('title').distinct().count())
print('nb unique text:', true_data.select('text').distinct().count())
print('nb rows:', true_data.count())

# %%
# Looking at the distinct values in 'subject'
print(true_data.select('subject').distinct().collect())

# %%
# Collecting contents of dataset to see an example of text column
print(true_data.collect()[0].__getitem__('title'))

# %% [markdown]
# ## **<font color="#FBBF44">1.3 FULL DATASET</font>**

# %%
# Adding a 'fake' column to specify the label of the record when combining both datasets
fake_data = fake_data.withColumn("fake", lit(1))
true_data = true_data.withColumn("fake", lit(0))

# Putting both datasets together and looking at an overview of them
full_data = fake_data.union(true_data)
print(full_data.printSchema())
print(full_data.count())
full_data.show(5)

# %%
# Looking at the distinct values in 'subject' for the full dataset
full_data.select('subject').distinct().collect()

# %%
# Brief overview of data
full_data.describe().show()

# %%
# Looking for Nulls and NaNs
full_data.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in full_data.columns]).show()

# %%
# Creating view to query from
full_data.createOrReplaceTempView("view")

# Query to get total number of elements for each category
total_subject = sqlContext.sql("""SELECT subject, COUNT(*) as total
                                    FROM view
                                    GROUP BY subject
                                    ORDER BY total desc""")

# Query to get total number of elements for each label
total_label = sqlContext.sql("""SELECT fake, COUNT(*) as total
                                    FROM view
                                    GROUP BY fake
                                    ORDER BY total desc""")

# Query to get total number of elements for each subject, divided by label
total_subject_label = sqlContext.sql("""SELECT subject, COUNT(*) as total, sum(fake) as fake, COUNT(*) - sum(fake) as true
                                    FROM view
                                    GROUP BY subject, fake
                                    ORDER BY total desc""")

print(total_subject.show())
print(total_label.show())
print(total_subject_label.show())

# Passing spark DF to pandas DF to plot more easily later
total_subject = total_subject.toPandas()
total_label = total_label.toPandas()

# %%
# News by subject
plt.bar(total_subject['subject'], total_subject['total'], color='darkslateblue')
plt.title("Amount of news by subject")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig('SubjectBefore.png', dpi=600)
plt.show()

# %%
# News by label
plt.bar(['Fake','Real'], total_label['total'])
plt.title("Total news by label")
plt.show()

# %% [markdown]
# ## **<font color="#FBBF44">1.4 PREPROCESSING</font>**

# %% [markdown]
# ### **<font color="#7dfa75">1.4.1 SUBJECT & DATE</font>**

# %% [markdown]
# **<font color="#ffff7">SUBJECT</font>**
#
# Since "worldnews" and "politicsNews" are subjects that appear only on the "True" dataset, in order to avoid bias we decided to transorm them into subjects already appearing in the "Fake" dataset, transforming "worldnews" into "news" and "politicsnews" into "politics", expecting them to match just fine

# %%
# Using regex_replace to transform the subjects
full_data_pre = full_data.withColumn("subject",regexp_replace(col('subject'), 'worldnews', 'News')) \
               .withColumn("subject",regexp_replace(col('subject'), 'politicsNews', 'politics'))
full_data_pre.show(5)

# %%
# Confirming the undesired subjects are no longer present in the dataset
full_data_pre.select('subject').distinct().collect()

# %%
# Querying again for the total amount of records with each subject, we can now see the difference in "News" and "Politics"
full_data_pre.createOrReplaceTempView("view")
total_subject = sqlContext.sql("""SELECT subject, COUNT(*) as total
                                    FROM view
                                    GROUP BY subject
                                    ORDER BY total desc""")
print(total_subject.show())
total_subject = total_subject.toPandas()

# %%
# News by subject
plt.bar(total_subject['subject'], total_subject['total'])
plt.title("Amount of news by subject")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig('SubjectAfter.png', dpi=600)
plt.show()

# %% [markdown]
# **<font color="#ffff7">DATE</font>**
#
# Regarding "Date", the idea is to have a consistent and accessible format to use for analysis. As of now, there are seem to be different formats present in the dataset, like full months and abbreviated ones, which can complicate their use.

# %%
# 1st step is to remove the comma from the date
data_date = full_data_pre.withColumn("date",regexp_replace(col('date'), ',', ''))

# Then splitting each element into its respective column
data_date = data_date.withColumn('year', split(data_date['date'], ' ').getItem(2)) \
                     .withColumn('month', split(data_date['date'], ' ').getItem(0)) \
                     .withColumn('day', split(data_date['date'], ' ').getItem(1))
print(data_date.show(1))

# Tranforming abbreviated months, e.g. "Dec" into their full form, "December", then taking this full form into a number, "12"
data_date = data_date.withColumn('month', when((length(data_date.month) > 3) | (data_date.month == "May"), data_date.month)
                     .otherwise(from_unixtime(unix_timestamp(data_date.month,'MMM'),'MMMM'))) \
                     .withColumn("month",from_unixtime(unix_timestamp(col("month"),'MMMM'),'MM'))
print(data_date.show(1))

# %%
# Looking for Nulls and NaNs before producing the final output for date, to see if anything went wrong
data_date.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in data_date.columns]).show()
data_date.filter(col("month").isNull()).show(5)

# %%
# Indeed, the previous cell shows us that 35 rows got NULL values, this due to a 3rd and unexpected format, a numerical one
# Given the small amount of them and for simplicity, dropping the rows is the chosen approach
data_date = data_date.dropna()
data_date.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in data_date.columns]).show()

# %%
# Finalizing the date preprocessing by putting together the values into their final column
data_date = data_date.select('title','text','subject',
                             concat_ws('-',data_date.year,data_date.month,data_date.day).alias("date"),
                             'fake')
print(data_date.show(5))

# %% [markdown]
# ### **<font color="#7dfa75">1.4.2 PREPARING FOR NLP</font>**

# %%
# Removing numbers and symbols to remain only with words
data_nlp = data_date

for col_name in ["title", "text"]:
    data_nlp = data_nlp.withColumn(col_name,regexp_replace(col(col_name), '\d+', ''))

data_nlp.select('text').show(1, truncate=False)

# %%
# Tokenizer
for col_name in ["title", "text"]:
  regex_tokenizer = RegexTokenizer(inputCol=col_name, outputCol=col_name+"Token", pattern="\\W")
  data_nlp = regex_tokenizer.transform(data_nlp)

# Adding id to keep track of records later on
data_nlp = data_nlp.withColumn("id", monotonically_increasing_id())

data_nlp = data_nlp.select('id', 'titleToken', 'textToken', 'subject', 'date', 'fake')
data_nlp.show(5)

# %%
# Removing stopwords
for col_name in ["titleToken", "textToken"]:
  remover = StopWordsRemover(inputCol=col_name, outputCol=col_name+"s")
  data_nlp = remover.transform(data_nlp)

data_nlp = data_nlp.select('id', 'titleTokens', 'textTokens', 'subject', 'date', 'fake')
(data_nlp.select('titleTokens').collect()[0][0])[0:10]

# %%
# Turning subject strings into numeric values, to later apply one hot encoding
indexer = StringIndexer(inputCol='subject', outputCol='numericSubject')
indexer_model = indexer.fit(data_nlp)
data_nlp_vec = indexer_model.transform(data_nlp)

# Applying one-hot encoding to turn categorical values in subject into a vector representing the diferent categories
OHE = OneHotEncoder(inputCol="numericSubject", outputCol="subject_vec", dropLast=False)
OHE_model = OHE.fit(data_nlp_vec)
data_nlp_vec = OHE_model.transform(data_nlp_vec).select('id',"titleTokens","textTokens","subject_vec","date","fake")
data_nlp_vec.show(5)

# %%
# In order to vectorize the date column, it is first split and put into separate columns
data_nlp_vec = data_nlp_vec.withColumn("date",regexp_replace(col('date'), '-', ' '))
data_nlp_vec = data_nlp_vec.withColumn('year', split(data_nlp_vec['date'], ' ').getItem(0)) \
                     .withColumn('month', split(data_nlp_vec['date'], ' ').getItem(1)) \
                     .withColumn('day', split(data_nlp_vec['date'], ' ').getItem(2))

# Casting as int to feed into VectorAssembler later on to vectorize
data_nlp_vec = data_nlp_vec.withColumn("year",col("year").cast('int')) \
                           .withColumn("month",col("month").cast('int')) \
                           .withColumn("day",col("day").cast('int'))
data_nlp_vec.show(5)

# %%
# Constructing the vector from the data obtained before
assembler = VectorAssembler(inputCols=['year','month','day'], outputCol="date_vec")
data_nlp_vec = assembler.transform(data_nlp_vec).select('id','titleTokens','textTokens','subject_vec','date_vec','fake')
data_nlp_vec.show(5)

# %%
# Using CountVectorizer, to transform the title and contents of the news articles into vectors that represent their features
for col_name in ["titleTokens", "textTokens"]:
  count_vectorizer = CountVectorizer(inputCol=col_name, outputCol=col_name+'_vecfull')
  cv_model = count_vectorizer.fit(data_nlp_vec)
  data_nlp_vec = cv_model.transform(data_nlp_vec)

data_nlp_vec_fullfeatures = data_nlp_vec.select('id',"titleTokens_vecfull","textTokens_vecfull","subject_vec","date_vec","fake")

for col_name in ["titleTokens", "textTokens"]:
  count_vectorizer = CountVectorizer(inputCol=col_name, outputCol=col_name+'_vec', minDF=.01, maxDF=.9)
  cv_model = count_vectorizer.fit(data_nlp_vec)
  data_nlp_vec = cv_model.transform(data_nlp_vec)

data_nlp_vec = data_nlp_vec.select('id',"titleTokens_vec","textTokens_vec","subject_vec","date_vec","fake")
data_nlp_vec.show(5)

# %%
# Putting together all previous vectors into a final one
assembler = VectorAssembler(inputCols=['titleTokens_vecfull',
                                       'textTokens_vecfull',
                                       'subject_vec',
                                       'date_vec'],
                            outputCol="full_features")
data_nlp_vec_fullfeatures = assembler.transform(data_nlp_vec_fullfeatures).select('id','titleTokens_vecfull','textTokens_vecfull','subject_vec','date_vec','full_features','fake')

assembler = VectorAssembler(inputCols=['titleTokens_vec',
                                       'textTokens_vec',
                                       'subject_vec',
                                       'date_vec'],
                            outputCol="full_features")
data_nlp_vec = assembler.transform(data_nlp_vec).select('id','titleTokens_vec','textTokens_vec','subject_vec','date_vec','full_features','fake')
data_nlp_vec.show(5)

# %%
data_nlp_vec_fullfeatures.show(5)

# %% [markdown]
# ## **<font color="#FBBF44">1.5 FURTHER EXPLORING</font>**

# %%
# Putting together subject and fake variables to calculate correlation matrix
assembler = VectorAssembler(inputCols=['subject_vec',
                                       'date_vec',
                                       'fake'],
                            outputCol="corr_vector")
data_corr = assembler.transform(data_nlp_vec).select('corr_vector')
data_corr.show(5)

# %%
# Columns related to the vectors used for the correlation
cols = ['News', 'Politics', 'Left-news', 'Gov-News', 'US-News', 'Middle-east', 'Year', 'Month', 'Day', 'Fake']

# Computing correlation matrix from the vector previously assembled
matrix = Correlation.corr(data_corr, 'corr_vector', method='spearman').collect()[0][0]
corr_matrix = matrix.toArray().tolist()
corr_matrix_df = pd.DataFrame(data=corr_matrix, columns=cols)

# Plotting correlation matrix
plt.figure(figsize=(16,4))
sns.heatmap(corr_matrix_df,
            xticklabels=corr_matrix_df.columns.values,
            yticklabels=corr_matrix_df.columns.values,
            cmap="winter",
            annot=True)
plt.title('Correlation betweeen subject, date and target variable')
plt.tight_layout()
plt.savefig('CorrMatrixSubject.png', dpi=600)
plt.show()

# %%
# Dividing the dataset to separate by true and fake
dates_fake = [data[0] for data in data_nlp.select('date').filter(col("fake") == 1).collect()]
dates_true = [data[0] for data in data_nlp.select('date').filter(col("fake") == 0).collect()]

dates_fake_plot = []
dates_true_plot = []

# Converting strings into date format to plot
for i in range(len(dates_fake)):
  temp = datetime.strptime(dates_fake[i], '%Y-%m-%d')
  dates_fake_plot.append(temp)

for i in range(len(dates_true)):
  temp = datetime.strptime(dates_true[i], '%Y-%m-%d')
  dates_true_plot.append(temp)

# %%
# Plotting distribution of dates
fig, ax = plt.subplots(figsize=(10,3))
n, bins, patches = ax.hist([dates_true_plot, dates_fake_plot], bins=150, stacked=True, color=["tab:blue", "tab:orange"])
plt.legend({'True': "tab:blue", 'Fake': "tab:orange"})
plt.title('Frequency distribution of dates')
fig.autofmt_xdate()
fig.tight_layout()
plt.savefig('PlotDates.png', dpi=600)
plt.show()

# %%
# Collecting all tokens from the previous dataset in order to put them into a single string
# and visualize a wordcloud with the most prominent words
words = [data[0] for data in data_nlp.select('textTokens').collect()]
full_words = []
for i in range(len(words)):
    full_words.append(" ".join(words[i]))

wordcloud_words = " ".join(full_words)

# %%
# Putting together all lists of fake words to form a single list containing occurrences of words
full_words_list = []
for i in range(len(words)):
    full_words_list += words[i]

# Creating an RDD with words
full_words_rdd = sc.parallelize(full_words_list)

# Counting occurences for each word
wordcount = full_words_rdd.flatMap(lambda x: x.split(' ')) \
                          .map(lambda x: (x,1)) \
                          .reduceByKey(lambda x,y:x+y) \
                          .map(lambda x: (x[1], x[0])) \
                          .sortByKey(False) \
                          .collect()

# Turning list of tuples previously obtained into two separate lists
word_frequency = list(map(list, zip(*wordcount)))

# %%
len(word_frequency[0])

# %%
# Word Frequency Full dataset
x = word_frequency[1]
y = word_frequency[0]

fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_title("Word Frequency distribution in Full Dataset")
ax.set_xticks(x[::8000])
ax.set_xticklabels(x[::8000], rotation=45, ha="right")
plt.tight_layout()
plt.savefig('WordFrequency.png', dpi=600)
plt.show()

# %%
plt.figure(figsize=(10,10))
wordcloud = WordCloud(stopwords=STOPWORDS,
                          background_color='white',
                          collocations=False,
                          width=1200,
                          height=1000
                         ).generate(wordcloud_words)
plt.axis('off')
wordcloud.to_file("WordcloudFull.png")
plt.imshow(wordcloud)

# %%
# Putting together all lists of fake words to form a single list containing occurrences of words
full_words_list = []
for i in range(len(words)):
    full_words_list += words[i]

# Creating an RDD with words
full_words_rdd = sc.parallelize(full_words_list)

# Counting occurences for each word
wordcount = full_words_rdd.flatMap(lambda x: x.split(' ')) \
                          .map(lambda x: (x,1)) \
                          .reduceByKey(lambda x,y:x+y) \
                          .map(lambda x: (x[1], x[0])) \
                          .sortByKey(False) \
                          .take(10)

# Turning list of tuples previously obtained into two separate lists
word_frequency = list(map(list, zip(*wordcount)))

# %%
# Word Frequency Full dataset
plt.bar(word_frequency[1], word_frequency[0])
plt.title("Top 10 frequent words in full dataset")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig('10FrequentFull.png', dpi=600)
plt.show()

# %%
fake_data = data_nlp.select("textTokens").filter(col("fake") == 1)
print(fake_data.count())
fake_data.show(5)

# %%
fake_words = [data[0] for data in fake_data.collect()]
fake_full_words = []
for i in range(len(fake_words)):
    fake_full_words.append(" ".join(fake_words[i]))

fake_wordcloud_words = " ".join(fake_full_words)
fake_wordcloud_words[0:100]

# %%
plt.figure(figsize=(10,10))
wordcloud = WordCloud(stopwords=STOPWORDS,
                          background_color='white',
                          collocations=False,
                          width=1200,
                          height=1000
                         ).generate(fake_wordcloud_words)
plt.axis('off')
wordcloud.to_file("WordcloudFake.png")
plt.imshow(wordcloud)

# %%
# Putting together all lists of fake words to form a single list containing occurrences of fake words
fake_words_list = []
for i in range(len(fake_words)):
    fake_words_list += fake_words[i]

# Creating an RDD with fake words
fake_words_rdd = sc.parallelize(fake_words_list)

# Counting occurences for each word
fake_wordcount = fake_words_rdd.flatMap(lambda x: x.split(' ')) \
                          .map(lambda x: (x,1)) \
                          .reduceByKey(lambda x,y:x+y) \
                          .map(lambda x: (x[1], x[0])) \
                          .sortByKey(False) \
                          .take(10)

# Turning list of tuples previously obtained into two separate lists
fake_word_frequency = list(map(list, zip(*fake_wordcount)))

# %%
# Word Frequency fake dataset
plt.bar(fake_word_frequency[1], fake_word_frequency[0])
plt.title("Top 10 frequent words in fake dataset")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig('10FrequentFake.png', dpi=600)
plt.show()

# %%
real_data = data_nlp.select("textTokens").filter(col("fake") == 0)
print(real_data.count())
real_data.show(5)

# %%
real_words = [data[0] for data in real_data.collect()]
real_full_words = []
for i in range(len(real_words)):
    real_full_words.append(" ".join(real_words[i]))

real_wordcloud_words = " ".join(real_full_words)
real_wordcloud_words[0:100]

# %%
plt.figure(figsize=(10,10))
wordcloud = WordCloud(stopwords=STOPWORDS,
                          background_color='white',
                          collocations=False,
                          width=1200,
                          height=1000
                         ).generate(real_wordcloud_words)
plt.axis('off')
wordcloud.to_file("WordcloudTrue.png")
plt.imshow(wordcloud)

# %%
# Putting together all lists of real words to form a single list containing occurrences of real words
real_words_list = []
for i in range(len(real_words)):
    real_words_list += real_words[i]

# Creating an RDD with real words
real_words_rdd = sc.parallelize(real_words_list)

# Counting occurences for each word
real_wordcount = real_words_rdd.flatMap(lambda x: x.split(' ')) \
                          .map(lambda x: (x,1)) \
                          .reduceByKey(lambda x,y:x+y) \
                          .map(lambda x: (x[1], x[0])) \
                          .sortByKey(False) \
                          .take(10)

# Turning list of tuples previously obtained into two separate lists
real_word_frequency = list(map(list, zip(*real_wordcount)))

# %%
# Word Frequency real dataset
plt.bar(real_word_frequency[1], real_word_frequency[0])
plt.title("Word Frequency in real dataset")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig('10FrequentTrue.png', dpi=600)
plt.show()

# %% [markdown]
# # **<font color="#34ebdb">2.0 TOPIC MODELLING</font>**

# %%
# Selecting the vectorized text variable with full features
data_cluster = data_nlp_vec_fullfeatures.select('id','textTokens_vecfull')
data_cluster.show(5)

# %% [markdown]
# ## **<font color="#FBBF44">2.1 K-MEANS</font>**

# %%
# Computing silhouette score the get optimal number of clusters with elbow method
silhouette_scores=[]
evaluator = ClusteringEvaluator(featuresCol='textTokens_vecfull', \
metricName='silhouette', distanceMeasure='squaredEuclidean')

for K in range(2,11):
  KMeans_=KMeans(featuresCol='textTokens_vecfull', k=K)
  KMeans_fit=KMeans_.fit(data_cluster)
  KMeans_transform=KMeans_fit.transform(data_cluster)
  evaluation_score=evaluator.evaluate(KMeans_transform)
  silhouette_scores.append(evaluation_score)

# %%
plt.figure(figsize=(10, 3))
plt.plot(range(2,11), silhouette_scores)
plt.title('Silhouette Scores for K-Means')
plt.xlabel('Number of Clusters')
plt.savefig('K-Means_silhouette.png', dpi=600)

# %%
kmeans = KMeans(featuresCol='textTokens_vecfull').setK(4).setSeed(1)
modelk = kmeans.fit(data_cluster)

# Make predictions
predictions = modelk.transform(data_cluster)

# Evaluate clustering by computing Silhouette score
evaluator = ClusteringEvaluator(featuresCol='textTokens_vecfull')

silhouette = evaluator.evaluate(predictions)
print("Silhouette with squared euclidean distance = " + str(silhouette))

# Shows the result.
centers = modelk.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)

# %%
# Viewing the label assigned to each record belonging to which cluster
data_cluster_label = modelk.transform(data_cluster)
data_cluster_label.show(5)

# %%
data_cluster_label = data_cluster_label.join(data_nlp, 'id')
data_cluster_label = data_cluster_label.select('id', 'textTokens', 'prediction')
data_cluster_label.show(5)

# %%
# Separating clusters and counting them
dk0 = data_cluster_label.filter(data_cluster_label.prediction == 0)
dk1 = data_cluster_label.filter(data_cluster_label.prediction == 1)
dk2 = data_cluster_label.filter(data_cluster_label.prediction == 2)
dk3 = data_cluster_label.filter(data_cluster_label.prediction == 3)

print('Amount of records in cluster 0:', dk0.count())
print('Amount of records in cluster 1:', dk1.count())
print('Amount of records in cluster 2:', dk2.count())
print('Amount of records in cluster 3:', dk3.count())

# %%
# Collecting all tokens from the previous dataset in order to put them into a single string
# and visualize a wordcloud with the most prominent words
words = [cl_data[0] for cl_data in dk0.select('textTokens').collect()]
full_words = []
for i in range(len(words)):
    full_words.append(" ".join(words[i]))

wordcloud_words = " ".join(full_words)


# Putting together all lists of fake words to form a single list containing occurrences of words
full_words_list = []
for i in range(len(words)):
    full_words_list += words[i]

# Creating an RDD with words
full_words_rdd = sc.parallelize(full_words_list)

# Counting occurences for each word
wordcount = full_words_rdd.flatMap(lambda x: x.split(' ')) \
                          .map(lambda x: (x,1)) \
                          .reduceByKey(lambda x,y:x+y) \
                          .map(lambda x: (x[1], x[0])) \
                          .sortByKey(False) \
                          .collect()

# Turning list of tuples previously obtained into two separate lists
word_frequency = list(map(list, zip(*wordcount)))


plt.figure(figsize=(10,10))
wordcloud = WordCloud(stopwords=STOPWORDS,
                          background_color='white',
                          collocations=False,
                          width=1200,
                          height=1000
                         ).generate(wordcloud_words)
plt.axis('off')
plt.imshow(wordcloud)

# %%
# Putting together all lists of fake words to form a single list containing occurrences of words
full_words_list = []
for i in range(len(words)):
    full_words_list += words[i]

# Creating an RDD with words
full_words_rdd = sc.parallelize(full_words_list)

# Counting occurences for each word
wordcount = full_words_rdd.flatMap(lambda x: x.split(' ')) \
                          .map(lambda x: (x,1)) \
                          .reduceByKey(lambda x,y:x+y) \
                          .map(lambda x: (x[1], x[0])) \
                          .sortByKey(False) \
                          .take(10)

# Turning list of tuples previously obtained into two separate lists
word_frequency = list(map(list, zip(*wordcount)))

# Word Frequency Full dataset
plt.bar(word_frequency[1], word_frequency[0])
plt.title("Word Frequency in full dataset")
plt.xticks(rotation=45, ha="right")
plt.show()

# %%
# Collecting all tokens from the previous dataset in order to put them into a single string
# and visualize a wordcloud with the most prominent words
words = [cl_data[0] for cl_data in dk1.select('textTokens').collect()]
full_words = []
for i in range(len(words)):
    full_words.append(" ".join(words[i]))

wordcloud_words = " ".join(full_words)


# Putting together all lists of fake words to form a single list containing occurrences of words
full_words_list = []
for i in range(len(words)):
    full_words_list += words[i]

# Creating an RDD with words
full_words_rdd = sc.parallelize(full_words_list)

# Counting occurences for each word
wordcount = full_words_rdd.flatMap(lambda x: x.split(' ')) \
                          .map(lambda x: (x,1)) \
                          .reduceByKey(lambda x,y:x+y) \
                          .map(lambda x: (x[1], x[0])) \
                          .sortByKey(False) \
                          .collect()

# Turning list of tuples previously obtained into two separate lists
word_frequency = list(map(list, zip(*wordcount)))


plt.figure(figsize=(10,10))
wordcloud = WordCloud(stopwords=STOPWORDS,
                          background_color='white',
                          collocations=False,
                          width=1200,
                          height=1000
                         ).generate(wordcloud_words)
plt.axis('off')
plt.imshow(wordcloud)

# %%
# Putting together all lists of fake words to form a single list containing occurrences of words
full_words_list = []
for i in range(len(words)):
    full_words_list += words[i]

# Creating an RDD with words
full_words_rdd = sc.parallelize(full_words_list)

# Counting occurences for each word
wordcount = full_words_rdd.flatMap(lambda x: x.split(' ')) \
                          .map(lambda x: (x,1)) \
                          .reduceByKey(lambda x,y:x+y) \
                          .map(lambda x: (x[1], x[0])) \
                          .sortByKey(False) \
                          .take(10)

# Turning list of tuples previously obtained into two separate lists
word_frequency = list(map(list, zip(*wordcount)))

# Word Frequency Full dataset
plt.bar(word_frequency[1], word_frequency[0])
plt.title("Word Frequency in full dataset")
plt.xticks(rotation=45, ha="right")
plt.show()

# %%
# Collecting all tokens from the previous dataset in order to put them into a single string
# and visualize a wordcloud with the most prominent words
words = [cl_data[0] for cl_data in dk2.select('textTokens').collect()]
full_words = []
for i in range(len(words)):
    full_words.append(" ".join(words[i]))

wordcloud_words = " ".join(full_words)


# Putting together all lists of fake words to form a single list containing occurrences of words
full_words_list = []
for i in range(len(words)):
    full_words_list += words[i]

# Creating an RDD with words
full_words_rdd = sc.parallelize(full_words_list)

# Counting occurences for each word
wordcount = full_words_rdd.flatMap(lambda x: x.split(' ')) \
                          .map(lambda x: (x,1)) \
                          .reduceByKey(lambda x,y:x+y) \
                          .map(lambda x: (x[1], x[0])) \
                          .sortByKey(False) \
                          .collect()

# Turning list of tuples previously obtained into two separate lists
word_frequency = list(map(list, zip(*wordcount)))


plt.figure(figsize=(10,10))
wordcloud = WordCloud(stopwords=STOPWORDS,
                          background_color='white',
                          collocations=False,
                          width=1200,
                          height=1000
                         ).generate(wordcloud_words)
plt.axis('off')
plt.imshow(wordcloud)

# %%
# Putting together all lists of fake words to form a single list containing occurrences of words
full_words_list = []
for i in range(len(words)):
    full_words_list += words[i]

# Creating an RDD with words
full_words_rdd = sc.parallelize(full_words_list)

# Counting occurences for each word
wordcount = full_words_rdd.flatMap(lambda x: x.split(' ')) \
                          .map(lambda x: (x,1)) \
                          .reduceByKey(lambda x,y:x+y) \
                          .map(lambda x: (x[1], x[0])) \
                          .sortByKey(False) \
                          .take(10)

# Turning list of tuples previously obtained into two separate lists
word_frequency = list(map(list, zip(*wordcount)))

# Word Frequency Full dataset
plt.bar(word_frequency[1], word_frequency[0])
plt.title("Word Frequency in full dataset")
plt.xticks(rotation=45, ha="right")
plt.show()

# %%
# Collecting all tokens from the previous dataset in order to put them into a single string
# and visualize a wordcloud with the most prominent words
words = [cl_data[0] for cl_data in dk3.select('textTokens').collect()]
full_words = []
for i in range(len(words)):
    full_words.append(" ".join(words[i]))

wordcloud_words = " ".join(full_words)


# Putting together all lists of fake words to form a single list containing occurrences of words
full_words_list = []
for i in range(len(words)):
    full_words_list += words[i]

# Creating an RDD with words
full_words_rdd = sc.parallelize(full_words_list)

# Counting occurences for each word
wordcount = full_words_rdd.flatMap(lambda x: x.split(' ')) \
                          .map(lambda x: (x,1)) \
                          .reduceByKey(lambda x,y:x+y) \
                          .map(lambda x: (x[1], x[0])) \
                          .sortByKey(False) \
                          .collect()

# Turning list of tuples previously obtained into two separate lists
word_frequency = list(map(list, zip(*wordcount)))


plt.figure(figsize=(10,10))
wordcloud = WordCloud(stopwords=STOPWORDS,
                          background_color='white',
                          collocations=False,
                          width=1200,
                          height=1000
                         ).generate(wordcloud_words)
plt.axis('off')
plt.imshow(wordcloud)

# %%
# Putting together all lists of fake words to form a single list containing occurrences of words
full_words_list = []
for i in range(len(words)):
    full_words_list += words[i]

# Creating an RDD with words
full_words_rdd = sc.parallelize(full_words_list)

# Counting occurences for each word
wordcount = full_words_rdd.flatMap(lambda x: x.split(' ')) \
                          .map(lambda x: (x,1)) \
                          .reduceByKey(lambda x,y:x+y) \
                          .map(lambda x: (x[1], x[0])) \
                          .sortByKey(False) \
                          .take(10)

# Turning list of tuples previously obtained into two separate lists
word_frequency = list(map(list, zip(*wordcount)))

# Word Frequency Full dataset
plt.bar(word_frequency[1], word_frequency[0])
plt.title("Word Frequency in full dataset")
plt.xticks(rotation=45, ha="right")
plt.show()

# %% [markdown]
# ## **<font color="#FBBF44">2.2 BISECTING K-MEANS</font>**

# %%
silhouette_scores=[]
evaluator = ClusteringEvaluator(featuresCol='textTokens_vecfull', \
metricName='silhouette', distanceMeasure='squaredEuclidean')

for K in range(2,11):
  BKMeans_= BisectingKMeans(featuresCol='textTokens_vecfull', k=K)
  BKMeans_fit=BKMeans_.fit(data_cluster)
  BKMeans_transform=BKMeans_fit.transform(data_cluster)
  evaluation_score=evaluator.evaluate(BKMeans_transform)
  silhouette_scores.append(evaluation_score)

# %%
plt.figure(figsize=(10, 3))
plt.plot(range(2,11), silhouette_scores)
plt.title('Silhouette Scores for Bisecting K-Means')
plt.xlabel('Number of Clusters')
plt.savefig('BK-Means_silhouette.png', dpi=600)

# %%
bkm = BisectingKMeans(featuresCol='textTokens_vecfull').setK(5).setSeed(1)
modelbk = bkm.fit(data_cluster)

# Make predictions
predictions = modelbk.transform(data_cluster)

# Evaluate clustering by computing Silhouette score
evaluator = ClusteringEvaluator(featuresCol='textTokens_vecfull')

silhouette = evaluator.evaluate(predictions)
print("Silhouette with squared euclidean distance = " + str(silhouette))

# Shows the result.
print("Cluster Centers: ")
centers = modelbk.clusterCenters()
for center in centers:
    print(center)

# %%
# Viewing the label assigned to each record belonging to which cluster
data_cluster_label = modelbk.transform(data_cluster)
data_cluster_label.show(5)

# %%
data_cluster_label = data_cluster_label.join(data_nlp, 'id')
data_cluster_label = data_cluster_label.select('id', 'textTokens', 'prediction')
data_cluster_label.show(5)

# %%
# Separating clusters and counting them
dk0 = data_cluster_label.filter(data_cluster_label.prediction == 0)
dk1 = data_cluster_label.filter(data_cluster_label.prediction == 1)
dk2 = data_cluster_label.filter(data_cluster_label.prediction == 2)
dk3 = data_cluster_label.filter(data_cluster_label.prediction == 3)
dk4 = data_cluster_label.filter(data_cluster_label.prediction == 4)

print('Amount of records in cluster 0:', dk0.count())
print('Amount of records in cluster 1:', dk1.count())
print('Amount of records in cluster 2:', dk2.count())
print('Amount of records in cluster 3:', dk3.count())
print('Amount of records in cluster 4:', dk4.count())

# %%
# Collecting all tokens from the previous dataset in order to put them into a single string
# and visualize a wordcloud with the most prominent words
words = [cl_data[0] for cl_data in dk0.select('textTokens').collect()]
full_words = []
for i in range(len(words)):
    full_words.append(" ".join(words[i]))

wordcloud_words = " ".join(full_words)


# Putting together all lists of fake words to form a single list containing occurrences of words
full_words_list = []
for i in range(len(words)):
    full_words_list += words[i]

# Creating an RDD with words
full_words_rdd = sc.parallelize(full_words_list)

# Counting occurences for each word
wordcount = full_words_rdd.flatMap(lambda x: x.split(' ')) \
                          .map(lambda x: (x,1)) \
                          .reduceByKey(lambda x,y:x+y) \
                          .map(lambda x: (x[1], x[0])) \
                          .sortByKey(False) \
                          .collect()

# Turning list of tuples previously obtained into two separate lists
word_frequency = list(map(list, zip(*wordcount)))


plt.figure(figsize=(10,10))
wordcloud = WordCloud(stopwords=STOPWORDS,
                          background_color='white',
                          collocations=False,
                          width=1200,
                          height=1000
                         ).generate(wordcloud_words)
plt.axis('off')
plt.imshow(wordcloud)

# %%
# Putting together all lists of fake words to form a single list containing occurrences of words
full_words_list = []
for i in range(len(words)):
    full_words_list += words[i]

# Creating an RDD with words
full_words_rdd = sc.parallelize(full_words_list)

# Counting occurences for each word
wordcount = full_words_rdd.flatMap(lambda x: x.split(' ')) \
                          .map(lambda x: (x,1)) \
                          .reduceByKey(lambda x,y:x+y) \
                          .map(lambda x: (x[1], x[0])) \
                          .sortByKey(False) \
                          .take(10)

# Turning list of tuples previously obtained into two separate lists
word_frequency = list(map(list, zip(*wordcount)))

# Word Frequency Full dataset
plt.bar(word_frequency[1], word_frequency[0])
plt.title("Word Frequency in full dataset")
plt.xticks(rotation=45, ha="right")
plt.show()

# %%
# Collecting all tokens from the previous dataset in order to put them into a single string
# and visualize a wordcloud with the most prominent words
words = [cl_data[0] for cl_data in dk1.select('textTokens').collect()]
full_words = []
for i in range(len(words)):
    full_words.append(" ".join(words[i]))

wordcloud_words = " ".join(full_words)


# Putting together all lists of fake words to form a single list containing occurrences of words
full_words_list = []
for i in range(len(words)):
    full_words_list += words[i]

# Creating an RDD with words
full_words_rdd = sc.parallelize(full_words_list)

# Counting occurences for each word
wordcount = full_words_rdd.flatMap(lambda x: x.split(' ')) \
                          .map(lambda x: (x,1)) \
                          .reduceByKey(lambda x,y:x+y) \
                          .map(lambda x: (x[1], x[0])) \
                          .sortByKey(False) \
                          .collect()

# Turning list of tuples previously obtained into two separate lists
word_frequency = list(map(list, zip(*wordcount)))


plt.figure(figsize=(10,10))
wordcloud = WordCloud(stopwords=STOPWORDS,
                          background_color='white',
                          collocations=False,
                          width=1200,
                          height=1000
                         ).generate(wordcloud_words)
plt.axis('off')
plt.imshow(wordcloud)

# %%
# Putting together all lists of fake words to form a single list containing occurrences of words
full_words_list = []
for i in range(len(words)):
    full_words_list += words[i]

# Creating an RDD with words
full_words_rdd = sc.parallelize(full_words_list)

# Counting occurences for each word
wordcount = full_words_rdd.flatMap(lambda x: x.split(' ')) \
                          .map(lambda x: (x,1)) \
                          .reduceByKey(lambda x,y:x+y) \
                          .map(lambda x: (x[1], x[0])) \
                          .sortByKey(False) \
                          .take(10)

# Turning list of tuples previously obtained into two separate lists
word_frequency = list(map(list, zip(*wordcount)))

# Word Frequency Full dataset
plt.bar(word_frequency[1], word_frequency[0])
plt.title("Word Frequency in full dataset")
plt.xticks(rotation=45, ha="right")
plt.show()

# %%
# Collecting all tokens from the previous dataset in order to put them into a single string
# and visualize a wordcloud with the most prominent words
words = [cl_data[0] for cl_data in dk2.select('textTokens').collect()]
full_words = []
for i in range(len(words)):
    full_words.append(" ".join(words[i]))

wordcloud_words = " ".join(full_words)


# Putting together all lists of fake words to form a single list containing occurrences of words
full_words_list = []
for i in range(len(words)):
    full_words_list += words[i]

# Creating an RDD with words
full_words_rdd = sc.parallelize(full_words_list)

# Counting occurences for each word
wordcount = full_words_rdd.flatMap(lambda x: x.split(' ')) \
                          .map(lambda x: (x,1)) \
                          .reduceByKey(lambda x,y:x+y) \
                          .map(lambda x: (x[1], x[0])) \
                          .sortByKey(False) \
                          .collect()

# Turning list of tuples previously obtained into two separate lists
word_frequency = list(map(list, zip(*wordcount)))


plt.figure(figsize=(10,10))
wordcloud = WordCloud(stopwords=STOPWORDS,
                          background_color='white',
                          collocations=False,
                          width=1200,
                          height=1000
                         ).generate(wordcloud_words)
plt.axis('off')
plt.imshow(wordcloud)

# %%
# Putting together all lists of fake words to form a single list containing occurrences of words
full_words_list = []
for i in range(len(words)):
    full_words_list += words[i]

# Creating an RDD with words
full_words_rdd = sc.parallelize(full_words_list)

# Counting occurences for each word
wordcount = full_words_rdd.flatMap(lambda x: x.split(' ')) \
                          .map(lambda x: (x,1)) \
                          .reduceByKey(lambda x,y:x+y) \
                          .map(lambda x: (x[1], x[0])) \
                          .sortByKey(False) \
                          .take(10)

# Turning list of tuples previously obtained into two separate lists
word_frequency = list(map(list, zip(*wordcount)))

# Word Frequency Full dataset
plt.bar(word_frequency[1], word_frequency[0])
plt.title("Word Frequency in full dataset")
plt.xticks(rotation=45, ha="right")
plt.show()

# %%
# Collecting all tokens from the previous dataset in order to put them into a single string
# and visualize a wordcloud with the most prominent words
words = [cl_data[0] for cl_data in dk3.select('textTokens').collect()]
full_words = []
for i in range(len(words)):
    full_words.append(" ".join(words[i]))

wordcloud_words = " ".join(full_words)


# Putting together all lists of fake words to form a single list containing occurrences of words
full_words_list = []
for i in range(len(words)):
    full_words_list += words[i]

# Creating an RDD with words
full_words_rdd = sc.parallelize(full_words_list)

# Counting occurences for each word
wordcount = full_words_rdd.flatMap(lambda x: x.split(' ')) \
                          .map(lambda x: (x,1)) \
                          .reduceByKey(lambda x,y:x+y) \
                          .map(lambda x: (x[1], x[0])) \
                          .sortByKey(False) \
                          .collect()

# Turning list of tuples previously obtained into two separate lists
word_frequency = list(map(list, zip(*wordcount)))


plt.figure(figsize=(10,10))
wordcloud = WordCloud(stopwords=STOPWORDS,
                          background_color='white',
                          collocations=False,
                          width=1200,
                          height=1000
                         ).generate(wordcloud_words)
plt.axis('off')
plt.imshow(wordcloud)

# %%
# Putting together all lists of fake words to form a single list containing occurrences of words
full_words_list = []
for i in range(len(words)):
    full_words_list += words[i]

# Creating an RDD with words
full_words_rdd = sc.parallelize(full_words_list)

# Counting occurences for each word
wordcount = full_words_rdd.flatMap(lambda x: x.split(' ')) \
                          .map(lambda x: (x,1)) \
                          .reduceByKey(lambda x,y:x+y) \
                          .map(lambda x: (x[1], x[0])) \
                          .sortByKey(False) \
                          .take(10)

# Turning list of tuples previously obtained into two separate lists
word_frequency = list(map(list, zip(*wordcount)))

# Word Frequency Full dataset
plt.bar(word_frequency[1], word_frequency[0])
plt.title("Word Frequency in full dataset")
plt.xticks(rotation=45, ha="right")
plt.show()

# %%
# Collecting all tokens from the previous dataset in order to put them into a single string
# and visualize a wordcloud with the most prominent words
words = [cl_data[0] for cl_data in dk4.select('textTokens').collect()]
full_words = []
for i in range(len(words)):
    full_words.append(" ".join(words[i]))

wordcloud_words = " ".join(full_words)


# Putting together all lists of fake words to form a single list containing occurrences of words
full_words_list = []
for i in range(len(words)):
    full_words_list += words[i]

# Creating an RDD with words
full_words_rdd = sc.parallelize(full_words_list)

# Counting occurences for each word
wordcount = full_words_rdd.flatMap(lambda x: x.split(' ')) \
                          .map(lambda x: (x,1)) \
                          .reduceByKey(lambda x,y:x+y) \
                          .map(lambda x: (x[1], x[0])) \
                          .sortByKey(False) \
                          .collect()

# Turning list of tuples previously obtained into two separate lists
word_frequency = list(map(list, zip(*wordcount)))


plt.figure(figsize=(10,10))
wordcloud = WordCloud(stopwords=STOPWORDS,
                          background_color='white',
                          collocations=False,
                          width=1200,
                          height=1000
                         ).generate(wordcloud_words)
plt.axis('off')
plt.imshow(wordcloud)

# %%
# Putting together all lists of fake words to form a single list containing occurrences of words
full_words_list = []
for i in range(len(words)):
    full_words_list += words[i]

# Creating an RDD with words
full_words_rdd = sc.parallelize(full_words_list)

# Counting occurences for each word
wordcount = full_words_rdd.flatMap(lambda x: x.split(' ')) \
                          .map(lambda x: (x,1)) \
                          .reduceByKey(lambda x,y:x+y) \
                          .map(lambda x: (x[1], x[0])) \
                          .sortByKey(False) \
                          .take(10)

# Turning list of tuples previously obtained into two separate lists
word_frequency = list(map(list, zip(*wordcount)))

# Word Frequency Full dataset
plt.bar(word_frequency[1], word_frequency[0])
plt.title("Word Frequency in full dataset")
plt.xticks(rotation=45, ha="right")
plt.show()

# %% [markdown]
# ## **<font color="#FBBF44">2.3 LATENT DIRICHLET ALLOCATION</font>**
#
#
#

# %%
# Selecting the vectorized text variable
data_cluster = data_nlp_vec.select('id','textTokens_vec')
data_cluster.show(5)

# %%
perplexity=[]

for K in tqdm(range(2,6)):
    lda_=LDA(k=K, maxIter=10, featuresCol = 'textTokens_vec')
    lda_fit=lda_.fit(data_cluster)
    evaluation_score=lda_fit.logPerplexity(data_cluster)
    perplexity.append(str(evaluation_score))

# %%
perplexity

# %%
fig, ax = plt.subplots(1,1, figsize =(10,8))
ax.plot(range(2,6),perplexity)
ax.set_xlabel('Number of Clusters')
ax.set_ylabel('Perplexity')

# %%
lda = LDA(k=30, maxIter=100, featuresCol = 'textTokens_vec')
modellda = lda.fit(data_cluster)

ll = modellda.logLikelihood(data_cluster)
lp = modellda.logPerplexity(data_cluster)

# %%
print("The lower bound on the log likelihood of the entire corpus: " + str(ll))
print("The upper bound on perplexity: " + str(lp))

# Describe topics.
vocab = cv_model.vocabulary
topicslda = modellda.describeTopics()
t = modellda.describeTopics().collect()
topic_inds = [ind.termIndices for ind in t]

topicsl = []
for topic in topic_inds:
  _topic = []
  for ind in topic:
    _topic.append(vocab[ind])
  topicsl.append(_topic)

# Shows the result
transformedlda = modellda.transform(data_cluster).select('id','topicDistribution')
transformedlda.show(5)

# %%
for i, topic in enumerate(topicsl, start=1):
  print(f"topic {i}: {topic}")

# %% [markdown]
# # **<font color="#34ebdb">3.0 CLASSIFICATION</font>**

# %%
def getAP(y_test, y_proba):
  AP = average_precision_score(y_test, y_proba, average='weighted')
  return np.round(AP, 4)

def getAUC(y_test, y_proba):
  AUC = roc_auc_score(y_test, y_proba, average='weighted')
  return np.round(AUC, 4)

# %%
# Putting together a vector for classification
assembler = VectorAssembler(inputCols=['titleTokens_vec',
                                       'subject_vec',
                                       'date_vec'],
                            outputCol="clf_vector")
data_clf = assembler.transform(data_nlp_vec)
data_clf.show(5)

# %%
# Splitting training and testing data
(data_train, data_test) = data_clf.select('clf_vector','fake').randomSplit([0.7, 0.3], seed = 42)
print('Number of training records: ', data_train.count())
print('Number of testing records: ', data_test.count())
data_test.show(5)

# %% [markdown]
# ## **<font color="#FBBF44">3.1 DECISION TREE</font>**

# %%
# Creating classifier with default parameters
dec_tree = DecisionTreeClassifier(featuresCol='clf_vector',labelCol='fake').fit(data_train)
predictions = dec_tree.transform(data_test)
predictions.show(5)

# %%
# Getting predictions and correct label to later pass as rdd to get classifier metrics
preds_and_labels = predictions.select('prediction', 'fake').withColumn("prediction",col("prediction").cast('float')) \
                                                           .withColumn("fake",col("fake").cast('float'))

bin_metrics = BinaryClassificationMetrics(preds_and_labels.rdd.map(tuple))
metrics = MulticlassMetrics(preds_and_labels.rdd.map(tuple))

# Again collecting predictions and true label, this time as lists, to use for example in confusion matrix
# Also collecting probabilities to be used in ROC curve
y_pred = predictions.select('prediction').rdd.keys().collect()
y_proba = predictions.select(vector_to_array("probability")[1]).rdd.keys().collect()
y_test = predictions.select("fake").rdd.keys().collect()

# %%
# Metrics report
print('Accuracy:', np.round(metrics.accuracy ,4))
print('AUC:', np.round(bin_metrics.areaUnderROC ,4))
print('AP:', np.round(bin_metrics.areaUnderPR ,4), '\n')

print('Precision for class 0:', np.round(metrics.precision(0.0) ,4))
print('Precision for class 1:', np.round(metrics.precision(1.0) ,4))
print('Weighted Precision:', np.round(metrics.weightedPrecision ,4), '\n')

print('Recall for class 0:', np.round(metrics.recall(0.0) ,4))
print('Recall for class 1:', np.round(metrics.recall(1.0) ,4))
print('Weighted Recall:', np.round(metrics.weightedRecall ,4), '\n')

print('F-score for class 0:', np.round(metrics.fMeasure(0.0) ,4))
print('F-score for class 1:', np.round(metrics.fMeasure(1.0) ,4))
print('Weighted F-score:', np.round(metrics.weightedFMeasure(), 4))

# %%
# Plotting confusion matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues")
plt.title('Confusion Matrix for Decision Tree')
plt.savefig('CM-DT.png', dpi=600)
plt.show()

# %%
# Getting false and true positive rate to plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)

plt.plot(fpr, tpr, label='DT AUC = '+str(np.round(bin_metrics.areaUnderROC,4)))
plt.ylabel('False Positive Rate')
plt.xlabel('True Positive Rate')
plt.title('ROC Curve for Decision Tree')
plt.legend(loc='lower right')
# plt.savefig('CM.png', dpi=600)
plt.show()

# %%
# Getting precision and recall to plot PR curve
prec, rec, thresholds = precision_recall_curve(y_test, y_proba)

plt.plot(prec, rec, label='DT AP = ' + str(np.round(bin_metrics.areaUnderPR,4)))
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.title('PR Curve for Decision Tree')
plt.legend(loc='lower left')
# plt.savefig('CM.png', dpi=600)
plt.show()

# %% [markdown]
# ## **<font color="#FBBF44">3.2 LOGISTIC REGRESSION</font>**

# %%
log_reg = LogisticRegression(featuresCol='clf_vector', labelCol='fake').fit(data_train)
predictions = log_reg.transform(data_test)
predictions.show(5)

# %%
beta = np.sort(log_reg.coefficients)
plt.plot(beta)
plt.title('Beta Coefficients')
plt.ylabel('β')
plt.show()

# %%
# Getting predictions and correct label to later pass as rdd to get classifier metrics
preds_and_labels = predictions.select('prediction', 'fake').withColumn("prediction",col("prediction").cast('float')) \
                                                           .withColumn("fake",col("fake").cast('float'))

bin_metrics = BinaryClassificationMetrics(preds_and_labels.rdd.map(tuple))
metrics = MulticlassMetrics(preds_and_labels.rdd.map(tuple))

# Again collecting predictions and true label, this time as lists, to use for example in confusion matrix
# Also collecting probabilities to be used in ROC curve
y_pred = predictions.select('prediction').rdd.keys().collect()
y_proba = predictions.select(vector_to_array("probability")[1]).rdd.keys().collect()
y_test = predictions.select("fake").rdd.keys().collect()

# %%
# Metrics report
print('Accuracy:', np.round(metrics.accuracy ,4))
print('AUC:', np.round(bin_metrics.areaUnderROC ,4))
print('AP:', np.round(bin_metrics.areaUnderPR ,4), '\n')

print('Precision for class 0:', np.round(metrics.precision(0.0) ,4))
print('Precision for class 1:', np.round(metrics.precision(1.0) ,4))
print('Weighted Precision:', np.round(metrics.weightedPrecision ,4), '\n')

print('Recall for class 0:', np.round(metrics.recall(0.0) ,4))
print('Recall for class 1:', np.round(metrics.recall(1.0) ,4))
print('Weighted Recall:', np.round(metrics.weightedRecall ,4), '\n')

print('F-score for class 0:', np.round(metrics.fMeasure(0.0) ,4))
print('F-score for class 1:', np.round(metrics.fMeasure(1.0) ,4))
print('Weighted F-score:', np.round(metrics.weightedFMeasure(), 4))

# %%
evaluator = BinaryClassificationEvaluator(labelCol="fake")
evaluator.evaluate(predictions)

# %%
# Plotting confusion matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues")
plt.title('Confusion Matrix for Logistic Regression')
plt.savefig('CM-LR.png', dpi=600)
plt.show()

# %%
# Getting false and true positive rate to plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)

plt.plot(fpr, tpr, label='LR AUC = '+str(np.round(bin_metrics.areaUnderROC,4)))
plt.ylabel('False Positive Rate')
plt.xlabel('True Positive Rate')
plt.title('ROC Curve for Logistic Regression')
plt.legend(loc='lower right')
# plt.savefig('CM.png', dpi=600)
plt.show()

# %%
# Getting precision and recall to plot PR curve
prec, rec, thresholds = precision_recall_curve(y_test, y_proba)

plt.plot(prec, rec, label='LR AP = ' + str(np.round(bin_metrics.areaUnderPR,4)))
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.title('PR Curve for Logistic Regression')
plt.legend(loc='lower left')
# plt.savefig('CM.png', dpi=600)
plt.show()

# %% [markdown]
# ## **<font color="#FBBF44">3.3 SUPPORT VECTOR MACHINE</font>**

# %%
svc = LinearSVC(featuresCol='clf_vector', labelCol='fake').fit(data_train)
predictions = svc.transform(data_test)
predictions.show(5)

# %%
beta = np.sort(svc.coefficients)
plt.plot(beta)
plt.title('Beta Coefficients')
plt.ylabel('β')
plt.show()

# %%
# Getting predictions and correct label to later pass as rdd to get classifier metrics
preds_and_labels = predictions.select('prediction', 'fake').withColumn("prediction",col("prediction").cast('float')) \
                                                           .withColumn("fake",col("fake").cast('float'))

bin_metrics = BinaryClassificationMetrics(preds_and_labels.rdd.map(tuple))
metrics = MulticlassMetrics(preds_and_labels.rdd.map(tuple))

# Again collecting predictions and true label, this time as lists, to use for example in confusion matrix
# Also collecting probabilities to be used in ROC curve
y_pred = predictions.select('prediction').rdd.keys().collect()
y_proba = predictions.select(vector_to_array("rawPrediction")[1]).rdd.keys().collect()
y_test = predictions.select("fake").rdd.keys().collect()

# %%
# Metrics report
print('Accuracy:', np.round(metrics.accuracy ,4))
print('AUC:', np.round(bin_metrics.areaUnderROC ,4))
print('AP:', np.round(bin_metrics.areaUnderPR ,4), '\n')

print('Precision for class 0:', np.round(metrics.precision(0.0) ,4))
print('Precision for class 1:', np.round(metrics.precision(1.0) ,4))
print('Weighted Precision:', np.round(metrics.weightedPrecision ,4), '\n')

print('Recall for class 0:', np.round(metrics.recall(0.0) ,4))
print('Recall for class 1:', np.round(metrics.recall(1.0) ,4))
print('Weighted Recall:', np.round(metrics.weightedRecall ,4), '\n')

print('F-score for class 0:', np.round(metrics.fMeasure(0.0) ,4))
print('F-score for class 1:', np.round(metrics.fMeasure(1.0) ,4))
print('Weighted F-score:', np.round(metrics.weightedFMeasure(), 4))

# %%
# Plotting confusion matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues")
plt.title('Confusion Matrix for SVC')
# plt.savefig('CM.png', dpi=600)
plt.show()

# %%
# Getting false and true positive rate to plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)

plt.plot(fpr, tpr, label='SVC AUC = '+str(np.round(bin_metrics.areaUnderROC,4)))
plt.ylabel('False Positive Rate')
plt.xlabel('True Positive Rate')
plt.title('ROC Curve for SVC')
plt.legend(loc='lower right')
# plt.savefig('CM.png', dpi=600)
plt.show()

# %%
# Getting precision and recall to plot PR curve
prec, rec, thresholds = precision_recall_curve(y_test, y_proba)

plt.plot(prec, rec, label='SVC AP = ' + str(np.round(bin_metrics.areaUnderPR,4)))
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.title('PR Curve for SVC')
plt.legend(loc='lower left')
# plt.savefig('CM.png', dpi=600)
plt.show()

# %% [markdown]
# ## **<font color="#FBBF44">3.4 RANDOM FOREST</font>**

# %%
# Creating classifier with default parameters
rand_forest = RandomForestClassifier(featuresCol='clf_vector',labelCol='fake').fit(data_train)
predictions = rand_forest.transform(data_test)
predictions.show(5)

# %%
# Getting predictions and correct label to later pass as rdd to get classifier metrics
preds_and_labels = predictions.select('prediction', 'fake').withColumn("prediction",col("prediction").cast('float')) \
                                                           .withColumn("fake",col("fake").cast('float'))

bin_metrics = BinaryClassificationMetrics(preds_and_labels.rdd.map(tuple))
metrics = MulticlassMetrics(preds_and_labels.rdd.map(tuple))

# Again collecting predictions and true label, this time as lists, to use for example in confusion matrix
# Also collecting probabilities to be used in ROC curve
y_pred = predictions.select('prediction').rdd.keys().collect()
y_proba = predictions.select(vector_to_array("probability")[1]).rdd.keys().collect()
y_test = predictions.select("fake").rdd.keys().collect()

# %%
# Metrics report
print('Accuracy:', np.round(metrics.accuracy ,4))
print('AUC:', np.round(bin_metrics.areaUnderROC ,4))
print('AP:', np.round(bin_metrics.areaUnderPR ,4), '\n')

print('Precision for class 0:', np.round(metrics.precision(0.0) ,4))
print('Precision for class 1:', np.round(metrics.precision(1.0) ,4))
print('Weighted Precision:', np.round(metrics.weightedPrecision ,4), '\n')

print('Recall for class 0:', np.round(metrics.recall(0.0) ,4))
print('Recall for class 1:', np.round(metrics.recall(1.0) ,4))
print('Weighted Recall:', np.round(metrics.weightedRecall ,4), '\n')

print('F-score for class 0:', np.round(metrics.fMeasure(0.0) ,4))
print('F-score for class 1:', np.round(metrics.fMeasure(1.0) ,4))
print('Weighted F-score:', np.round(metrics.weightedFMeasure(), 4))

# %%
# Plotting confusion matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues")
plt.title('Confusion Matrix for Random Forest')
# plt.savefig('CM.png', dpi=600)
plt.show()

# %%
# Getting false and true positive rate to plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)

plt.plot(fpr, tpr, label='RF AUC = '+str(np.round(bin_metrics.areaUnderROC,4)))
plt.ylabel('False Positive Rate')
plt.xlabel('True Positive Rate')
plt.title('ROC Curve for Random Forest')
plt.legend(loc='lower right')
# plt.savefig('CM.png', dpi=600)
plt.show()

# %%
# Getting precision and recall to plot PR curve
prec, rec, thresholds = precision_recall_curve(y_test, y_proba)

plt.plot(prec, rec, label='RF AP = ' + str(np.round(bin_metrics.areaUnderPR,4)))
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.title('PR Curve for Random Forest')
plt.legend(loc='lower left')
# plt.savefig('CM.png', dpi=600)
plt.show()

# %% [markdown]
# ## **<font color="#FBBF44">3.5 COMPARING CLASSIFIERS</font>**

# %%
model_dt = DecisionTreeClassifier(featuresCol='clf_vector', labelCol='fake').fit(data_train)
model_logreg = LogisticRegression(featuresCol='clf_vector', labelCol='fake').fit(data_train)
model_svc = LinearSVC(featuresCol='clf_vector', labelCol='fake').fit(data_train)
model_rfc = RandomForestClassifier(featuresCol='clf_vector', labelCol='fake').fit(data_train)

models = {
  'Decision Tree': model_dt,
  'Logistic Regression': model_logreg,
  'Support Vector Machine': model_svc,
  'Random Forest': model_rfc,
  }

# %%
y_pred = {}
y_proba = {}
y_test = {}

for name, model in models.items():
  print(name, 'metrics:')
  predictions = model.transform(data_test)

  preds_and_labels = predictions.select('prediction', 'fake').withColumn("prediction",col("prediction").cast('float')) \
                                                             .withColumn("fake",col("fake").cast('float'))
  metrics = MulticlassMetrics(preds_and_labels.rdd.map(tuple))

  y_pred[name] = predictions.select('prediction').rdd.keys().collect()
  if "proability" in predictions.columns:
    y_proba[name] = predictions.select(vector_to_array("probability")[1]).rdd.keys().collect()
  else:
    y_proba[name] = predictions.select(vector_to_array("rawPrediction")[1]).rdd.keys().collect()
  y_test[name] = predictions.select("fake").rdd.keys().collect()

  print('Accuracy:', np.round(metrics.accuracy ,4))
  print('Precision:', np.round(metrics.weightedPrecision ,4))
  print('Recall:', np.round(metrics.weightedRecall ,4))
  print('F-score:', np.round(metrics.weightedFMeasure(), 4), '\n')

# %%
fpr, tpr, thresholds = roc_curve(y_test['Decision Tree'], y_proba['Decision Tree'])
fpr2, tpr2, thresholds2 = roc_curve(y_test['Logistic Regression'], y_proba['Logistic Regression'])
fpr3, tpr3, thresholds3 = roc_curve(y_test['Support Vector Machine'], y_proba['Support Vector Machine'])
fpr4, tpr4, thresholds4 = roc_curve(y_test['Random Forest'], y_proba['Random Forest'])

plt.plot(fpr, tpr, color='olivedrab', label='DT AUC = ' +
         str(getAUC(y_test['Decision Tree'], y_pred['Decision Tree'])))
plt.plot(fpr2, tpr2, color='turquoise', label='LR AUC = ' +
         str(getAUC(y_test['Logistic Regression'], y_pred['Logistic Regression'])))
plt.plot(fpr3, tpr3, color='darkorange', label='SVC AUC = ' +
         str(getAUC(y_test['Support Vector Machine'], y_pred['Support Vector Machine'])))
plt.plot(fpr4, tpr4, color='navy', label='RF AUC = ' +
         str(getAUC(y_test['Random Forest'], y_pred['Random Forest'])))
plt.ylabel('False Positive Rate')
plt.xlabel('True Positive Rate')
plt.title('ROC Curve - Comparison across models')
plt.legend(loc='lower right')
plt.savefig('ComparisonModelsROC.png', dpi=600)
plt.show()

# %%
# Getting precision and recall to plot PR curve
prec, rec, thresholds = precision_recall_curve(y_test['Decision Tree'], y_proba['Decision Tree'])
prec2, rec2, thresholds2 = precision_recall_curve(y_test['Logistic Regression'], y_proba['Logistic Regression'])
prec3, rec3, thresholds3 = precision_recall_curve(y_test['Support Vector Machine'], y_proba['Support Vector Machine'])
prec4, rec4, thresholds4 = precision_recall_curve(y_test['Random Forest'], y_proba['Random Forest'])

plt.plot(prec, rec, color='olivedrab', label='DT AP = ' +
         str(getAUC(y_test['Decision Tree'], y_pred['Decision Tree'])))
plt.plot(prec2, rec2, color='turquoise', label='LR AP = ' +
         str(getAUC(y_test['Logistic Regression'], y_pred['Logistic Regression'])))
plt.plot(prec3, rec3, color='darkorange', label='SVC AP = ' +
         str(getAUC(y_test['Support Vector Machine'], y_pred['Support Vector Machine'])))
plt.plot(prec4, rec4, color='navy', label='RF AP = ' +
         str(getAUC(y_test['Random Forest'], y_pred['Random Forest'])))
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.title('PR Curve - Comparison across models')
plt.legend(loc='lower left')
plt.savefig('ComparisonModelsPR.png', dpi=600)
plt.show()

# %% [markdown]
# ## **<font color="#FBBF44">3.6 COMPARING TRAINING VECTORS</font>**

# %%
# Splitting training and testing data
(data_train, data_test) = data_clf.randomSplit([0.7, 0.3], seed = 42)
print('Number of training records: ', data_train.count())
print('Number of testing records: ', data_test.count())
data_test.show(5)

# %%
model_svc = LinearSVC(featuresCol='clf_vector', labelCol='fake').fit(data_train)
model_svc2 = LinearSVC(featuresCol='titleTokens_vec', labelCol='fake').fit(data_train)
model_svc3 = LinearSVC(featuresCol='textTokens_vec', labelCol='fake').fit(data_train)
model_svc4 = LinearSVC(featuresCol='subject_vec', labelCol='fake').fit(data_train)
model_svc5 = LinearSVC(featuresCol='date_vec', labelCol='fake').fit(data_train)
model_svc6 = LinearSVC(featuresCol='full_features', labelCol='fake').fit(data_train)

models = {
  'SVC': model_svc,
  'SVC2': model_svc2,
  'SVC3': model_svc3,
  'SVC4': model_svc4,
  'SVC5': model_svc5,
  'SVC6': model_svc6,
  }

# %%
y_pred2 = {}
y_proba2 = {}
y_test2 = {}
AUC2 = {}
AP2 = {}

for name, model in models.items():
  print(name, 'metrics')
  predictions = model.transform(data_test)

  preds_and_labels = predictions.select('prediction', 'fake').withColumn("prediction",col("prediction").cast('float')) \
                                                             .withColumn("fake",col("fake").cast('float'))

  metrics = MulticlassMetrics(preds_and_labels.rdd.map(tuple))

  y_pred2[name] = predictions.select('prediction').rdd.keys().collect()
  if "proability" in predictions.columns:
    y_proba2[name] = predictions.select(vector_to_array("probability")[1]).rdd.keys().collect()
  else:
    y_proba2[name] = predictions.select(vector_to_array("rawPrediction")[1]).rdd.keys().collect()
  y_test2[name] = predictions.select("fake").rdd.keys().collect()

  print('Accuracy:', np.round(metrics.accuracy ,4))
  print('Precision:', np.round(metrics.weightedPrecision ,4))
  print('Recall:', np.round(metrics.weightedRecall ,4))
  print('F-score:', np.round(metrics.weightedFMeasure(), 4), '\n')

# %%
fpr, tpr, thresholds = roc_curve(y_test2['SVC'], y_proba2['SVC'])
fpr2, tpr2, thresholds2 = roc_curve(y_test2['SVC2'], y_proba2['SVC2'])
fpr3, tpr3, thresholds3 = roc_curve(y_test2['SVC3'], y_proba2['SVC3'])
fpr4, tpr4, thresholds4 = roc_curve(y_test2['SVC4'], y_proba2['SVC4'])
fpr5, tpr5, thresholds5 = roc_curve(y_test2['SVC5'], y_proba2['SVC5'])
fpr6, tpr6, thresholds6 = roc_curve(y_test2['SVC6'], y_proba2['SVC6'])

plt.plot(fpr2, tpr2, color= 'turquoise', label='Title AUC = ' +
         str(getAUC(y_test2['SVC2'], y_pred2['SVC2'])))
plt.plot(fpr3, tpr3, color= 'darkorange', label='Text AUC = ' +
         str(getAUC(y_test2['SVC3'], y_pred2['SVC3'])))
plt.plot(fpr4, tpr4, color= 'navy', label='Subject AUC = ' +
         str(getAUC(y_test2['SVC4'], y_pred2['SVC4'])))
plt.plot(fpr5, tpr5, color= 'purple', label='Date AUC = ' +
         str(getAUC(y_test2['SVC5'], y_pred2['SVC5'])))
plt.plot(fpr6, tpr6, color= 'teal', label='Full features AUC = ' +
         str(getAUC(y_test2['SVC6'], y_pred2['SVC6'])))
plt.plot(fpr, tpr, color= 'olivedrab', label='Title+Subject+Date AUC = ' +
         str(getAUC(y_test2['SVC'], y_pred2['SVC'])))

plt.ylabel('False Positive Rate')
plt.xlabel('True Positive Rate')
plt.title('ROC Curve - Comparison across input data')
plt.legend(loc='lower right')
plt.savefig('ComparisonDataROC.png', dpi=600)
plt.show()

# %%
# Getting precision and recall to plot PR curve
prec, rec, thresholds = precision_recall_curve(y_test2['SVC'], y_proba2['SVC'])
prec2, rec2, thresholds2 = precision_recall_curve(y_test2['SVC2'], y_proba2['SVC2'])
prec3, rec3, thresholds3 = precision_recall_curve(y_test2['SVC3'], y_proba2['SVC3'])
prec4, rec4, thresholds4 = precision_recall_curve(y_test2['SVC4'], y_proba2['SVC4'])
prec5, rec5, thresholds5 = precision_recall_curve(y_test2['SVC5'], y_proba2['SVC5'])
prec6, rec6, thresholds6 = precision_recall_curve(y_test2['SVC6'], y_proba2['SVC6'])

plt.plot(prec2, rec2, color= 'turquoise', label='Ttitle AP = ' +
         str(getAP(y_test2['SVC2'], y_pred2['SVC2'])))
plt.plot(prec3, rec3, color= 'darkorange', label='Text AP = ' +
         str(getAP(y_test2['SVC3'], y_pred2['SVC3'])))
plt.plot(prec4, rec4, color= 'navy', label='Subject AP = ' +
         str(getAP(y_test2['SVC4'], y_pred2['SVC4'])))
plt.plot(prec5, rec5, color= 'purple', label='Date AP = ' +
         str(getAP(y_test2['SVC5'], y_pred2['SVC5'])))
plt.plot(prec6, rec6, color= 'teal', label='Full features AP = ' +
         str(getAP(y_test2['SVC6'], y_pred2['SVC6'])))
plt.plot(prec, rec, color= 'olivedrab', label='Title+Subject+Date AP = ' +
         str(getAP(y_test2['SVC'], y_pred2['SVC'])))
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.title('PR Curve - Comparison across input data')
plt.legend(loc='lower left')
plt.savefig('ComparisonDataPR.png', dpi=600)
plt.show()

# %% [markdown]
# ## **<font color="#FBBF44">3.7 PARAMETER HYPERTUNING</font>**

# %%
# Splitting training and testing data
(data_train, data_test) = data_clf.select('clf_vector','fake').randomSplit([0.7, 0.3], seed = 42)
print('Number of training records: ', data_train.count())
print('Number of testing records: ', data_test.count())
data_test.show(5)

# %%
dt = DecisionTreeClassifier(labelCol = "fake", featuresCol = "clf_vector")
lr = LogisticRegression(labelCol = "fake", featuresCol = "clf_vector")
svc = LinearSVC(labelCol = "fake", featuresCol = "clf_vector")
rf = RandomForestClassifier(labelCol = "fake", featuresCol = "clf_vector")

paramGrid_dt = ParamGridBuilder() \
              .addGrid(dt.maxDepth, [5,15,25]) \
              .addGrid(dt.maxBins, [10,30,50]) \
              .build()

paramGrid_lr = ParamGridBuilder() \
              .addGrid(lr.regParam, [.01,.1,1,10]) \
              .addGrid(lr.fitIntercept, [True, False]) \
              .addGrid(lr.elasticNetParam, [.01,.05,.1,.5]) \
              .addGrid(lr.maxIter, [100, 200, 300]) \
              .build()

paramGrid_svc = ParamGridBuilder() \
                .addGrid(svc.regParam, [.01,.1,1,10]) \
                .addGrid(svc.fitIntercept, [True, False]) \
                .addGrid(svc.standardization, [True, False]) \
                .addGrid(svc.maxIter, [100, 200, 300]) \
                .build()

paramGrid_rf = ParamGridBuilder() \
              .addGrid(rf.numTrees, [30,40,50,60]) \
              .addGrid(rf.maxDepth, [5,15,25,30]) \
              .build()

crossval_dt = CrossValidator(estimator = dt,
                             estimatorParamMaps = paramGrid_dt,
                             evaluator = BinaryClassificationEvaluator(labelCol = 'fake'),
                             seed = 42,
                             numFolds=5)

crossval_lr = CrossValidator(estimator = lr,
                             estimatorParamMaps = paramGrid_lr,
                             evaluator = BinaryClassificationEvaluator(labelCol = 'fake'),
                             seed = 42,
                             numFolds=5)

crossval_svc = CrossValidator(estimator = svc,
                              estimatorParamMaps = paramGrid_svc,
                              evaluator = BinaryClassificationEvaluator(labelCol = 'fake'),
                              seed = 42,
                              numFolds=5)

crossval_rf = CrossValidator(estimator = rf,
                             estimatorParamMaps = paramGrid_rf,
                             evaluator = BinaryClassificationEvaluator(labelCol = 'fake'),
                             seed = 42,
                             numFolds=5)

# %% [markdown]
# **<font color="#ffff7">DECISION TREE</font>**

# %%
cv_dt = crossval_dt.fit(data_train)
predictions_dt = cv_dt.transform(data_test)
predictions_dt.show(5)

# %%
preds_and_labels_dt = predictions_dt.select('prediction', 'fake').withColumn("prediction",col("prediction").cast('float')) \
                                                           .withColumn("fake",col("fake").cast('float'))

bin_metrics_dt = BinaryClassificationMetrics(preds_and_labels_dt.rdd.map(tuple))
metrics_dt = MulticlassMetrics(preds_and_labels_dt.rdd.map(tuple))

y_pred_dt = predictions_dt.select('prediction').rdd.keys().collect()
y_proba_dt = predictions_dt.select(vector_to_array("probability")[1]).rdd.keys().collect()
y_test_dt = predictions_dt.select("fake").rdd.keys().collect()
print('Accuracy:', np.round(metrics_dt.accuracy ,4))
print('Precision:', np.round(metrics_dt.weightedPrecision ,4))
print('Recall:', np.round(metrics_dt.weightedRecall ,4))
print('F-score:', np.round(metrics_dt.weightedFMeasure(), 4))
print('AUC:', np.round(bin_metrics_dt.areaUnderROC ,4))
print('AP:', np.round(bin_metrics_dt.areaUnderPR ,4))

cv_dt.bestModel

# %% [markdown]
# **<font color="#ffff7">LOGISTIC REGRESSION</font>**

# %%
cv_lr = crossval_lr.fit(data_train)
predictions_lr = cv_lr.transform(data_test)
predictions_lr.show(5)

# %%
preds_and_labels_lr = predictions_lr.select('prediction', 'fake').withColumn("prediction",col("prediction").cast('float')) \
                                                           .withColumn("fake",col("fake").cast('float'))

bin_metrics_lr = BinaryClassificationMetrics(preds_and_labels_lr.rdd.map(tuple))
metrics_lr = MulticlassMetrics(preds_and_labels_lr.rdd.map(tuple))

y_pred_lr = predictions_lr.select('prediction').rdd.keys().collect()
y_proba_lr = predictions_lr.select(vector_to_array("probability")[1]).rdd.keys().collect()
y_test_lr = predictions_lr.select("fake").rdd.keys().collect()
print('Accuracy:', np.round(metrics_lr.accuracy ,4))
print('Precision:', np.round(metrics_lr.weightedPrecision ,4))
print('Recall:', np.round(metrics_lr.weightedRecall ,4))
print('F-score:', np.round(metrics_lr.weightedFMeasure(), 4))
print('AUC:', np.round(bin_metrics_lr.areaUnderROC ,4))
print('AP:', np.round(bin_metrics_lr.areaUnderPR ,4))

cv_lr.bestModel

# %% [markdown]
# **<font color="#ffff7">SUPPORT VECTOR MACHINES</font>**

# %%
cv_svc = crossval_svc.fit(data_train)
predictions_svc = cv_svc.transform(data_test)
predictions_svc.show(5)

# %%
preds_and_labels_svc = predictions_svc.select('prediction', 'fake').withColumn("prediction",col("prediction").cast('float')) \
                                                           .withColumn("fake",col("fake").cast('float'))

bin_metrics_svc = BinaryClassificationMetrics(preds_and_labels_svc.rdd.map(tuple))
metrics_svc = MulticlassMetrics(preds_and_labels_svc.rdd.map(tuple))

y_pred_svc = predictions_svc.select('prediction').rdd.keys().collect()
y_proba_svc = predictions_svc.select(vector_to_array("rawPrediction")[1]).rdd.keys().collect()
y_test_svc = predictions_svc.select("fake").rdd.keys().collect()
print('Accuracy:', np.round(metrics_svc.accuracy ,4))
print('Precision:', np.round(metrics_svc.weightedPrecision ,4))
print('Recall:', np.round(metrics_svc.weightedRecall ,4))
print('F-score:', np.round(metrics_svc.weightedFMeasure(), 4))
print('AUC:', np.round(bin_metrics_svc.areaUnderROC ,4))
print('AP:', np.round(bin_metrics_svc.areaUnderPR ,4))

cv_svc.bestModel

# %% [markdown]
# **<font color="#ffff7">RANDOM FORESTS</font>**

# %%
cv_rf = crossval_rf.fit(data_train)
predictions_rf = cv_rf.transform(data_test)
predictions_rf.show(5)

# %%
preds_and_labels_rf = predictions_rf.select('prediction', 'fake').withColumn("prediction",col("prediction").cast('float')) \
                                                           .withColumn("fake",col("fake").cast('float'))

bin_metrics_rf = BinaryClassificationMetrics(preds_and_labels_rf.rdd.map(tuple))
metrics_rf = MulticlassMetrics(preds_and_labels_rf.rdd.map(tuple))

y_pred_rf = predictions_rf.select('prediction').rdd.keys().collect()
y_proba_rf = predictions_rf.select(vector_to_array("probability")[1]).rdd.keys().collect()
y_test_rf = predictions_rf.select("fake").rdd.keys().collect()
print('Accuracy:', np.round(metrics_rf.accuracy ,4))
print('Precision:', np.round(metrics_rf.weightedPrecision ,4))
print('Recall:', np.round(metrics_rf.weightedRecall ,4))
print('F-score:', np.round(metrics_rf.weightedFMeasure(), 4))
print('AUC:', np.round(bin_metrics_rf.areaUnderROC ,4))
print('AP:', np.round(bin_metrics_rf.areaUnderPR ,4))

cv_rf.bestModel

# %% [markdown]
# # **<font color="#34ebdb">4.0 FREQUENT PATTERN MINING</font>**

# %%
# Removing numbers and symbols to remain only with words
new_data = fake_data.withColumn("text",regexp_replace(col('text'), '\d+', ''))

# Tokenizer
regex_tokenizer = RegexTokenizer(inputCol="text", outputCol="regToken", pattern="\\W")
new_data = regex_tokenizer.transform(new_data)

# Removing stopwords
remover = StopWordsRemover(inputCol="regToken", outputCol="tokens")
new_data = remover.transform(new_data)
data_fpm = new_data.select("tokens")
data_fpm.show(5)

# %%
# Obtaining unique items per observation
un_pre = [data[0] for data in data_fpm.select('tokens').collect()]
unique_val = []
num = 0
for x in un_pre:
  unique_val.append((num,list(set(x))))
  num += 1

# %%
df = spark.createDataFrame(unique_val, ["id", "items"])

fpGrowth = FPGrowth(itemsCol="items", minSupport=0.25, minConfidence=0.6)
model = fpGrowth.fit(df)

# Display frequent itemsets.
model.freqItemsets.show(10)

# %%
# Display generated association rules.
model.associationRules.show(10)

# %%
# transform examines the input items against all the association rules and summarize the consequents as prediction
model.transform(df).show(10)

# %%
# checking for same consequent in the association rules
ass_rules_con = [data[0] for data in model.associationRules.select('consequent').collect()]

# creating a dictionary to count the occurence of the same world in the consequent of the association rules
dict_rules_con = {}
for x in ass_rules_con:
  k = tuple(x)
  if k not in dict_rules_con.keys():
    dict_rules_con[k] = 1
  else:
    dict_rules_con[k] += 1
#print(dict_rules_con)

# %%
# checking for same antecedent in the association rules
ass_rules_ant = [data[0] for data in model.associationRules.select('antecedent').collect()]

# number of association rules found with a minumum support of 0,25
print(len(ass_rules_ant))

#creating a dictionary to count the occurence of the same world in the antecedent of the association rules
dict_rules = {}
for x in ass_rules_ant:
  k = tuple(x)
  if k not in dict_rules.keys():
    dict_rules[k] = 1
  else:
    dict_rules[k] += 1
#print(dict_rules)

# %%
# checking for the top ten of frequent items
freq_item_data = [data[0] for data in model.freqItemsets.select('items').collect()]
freq_item = [data[0] for data in model.freqItemsets.select('freq').collect()]

sort_freq = sorted(freq_item)[len(freq_item)-10:]

most_used_item = []

for i in range(len(freq_item)):
  if freq_item[i] in sort_freq:
    most_used_item.append(freq_item_data[i])

#print(most_used_item)

# %%
# checking for model with different minSup [0.05,0.95]
c = 0.05
for x in range(18):
  print("Modello con minSup a {}\n".format(c))
  fpGrowth = FPGrowth(itemsCol="items", minSupport=c, minConfidence=0.6)
  c += 0.05
  model = fpGrowth.fit(df)

  # Display frequent itemsets.
  model.freqItemsets.show()

  # Display generated association rules.
  model.associationRules.show()

  # transform examines the input items against all the association rules and summarize the
  # consequents as prediction
  model.transform(df).show()


