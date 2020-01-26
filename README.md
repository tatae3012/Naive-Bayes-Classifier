# Naive-Bayes-Classifier
Naive Bayes model to identify spam and ham (that is non-spam) emails.

The public lingspam data set is used for training. This data set is available online and can be easily downloaded.
In each folder the spam email files start with ‘spm. . . ’ while the ham emails do not.

The first step is to create a vocabulary. The vocabulary is the set of unique words that are present in the
full data set. These are usually arranged in lexicographic order.

v1.csv contains the raw data set. After removing stop words, performing lemmatization and frequency pruning
the resulting data set is stored in v4.csv.

**Stop words are frequently occurring words like the, a, an, this, that, is etc. that are very unlikely to
help in deciding a category. Lemmatization is a process whereby a word is reduced to its lemma or root.
For example, the words run, running, ran, runs, etc. have the same root/lemma run. Frequency based
pruning is often done for text data. The intuition is that words that are very frequent or very infrequent in
the corpus are unlikely to be useful in categorization. So, low frequency and high frequency thresholds are
used to screen out vocabulary words that are below or above the threshold.

The data set is divided into training and testing repeatedly to increase the randomness.For a test document use 
the Naive Bayes decision rule to give a spam or ham label. Find the performance on each test set.
The average of all the obtained performances is the final result. This is stored in result.csv

In result.csv, vocabulary column denotes:
1 - Raw dataset
2 - + Stop word removed dataset
3 - + Lemmatized dataset
4 - + Frequency pruned dataset
(In the code I have saved 1 as v1 and 4 as v4, you can check out steps 2 and 3 too even though not saved.)
