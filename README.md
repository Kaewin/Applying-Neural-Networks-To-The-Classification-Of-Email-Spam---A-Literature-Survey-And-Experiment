# Applying Neural Networks To The Classification Of Email Spam - A Literature Survey And Experiment

![meme](https://imgs.xkcd.com/comics/phishing_license_2x.png)

## Links:

[Presentation Slides](https://docs.google.com/presentation/d/11_n-Ghqe0-_EscrFUHaHen3UuLhE-letcroQExchjUg/edit?usp=sharing)

# Abstract


# Introduction

## Background:

Unsolicited Bulk Email (UBE), commonly known as spam, has emerged as a pervasive problem in the digital age. Email as a primary method of communication has become ubiquitous and mandatory in the digital age. The rise of email has led to the rise of spam, owing to the nearly infitesimal cost of sending spam emails, and the effortless ability to reach hundreds of thousands of people with the click of a button.

Many early adopters of the internet can recall a time where spam was ubiqutous, with newly created email accounts seeming to be inundated with spam messages within days of being created. While the volume of spam has decreased in recent years, it remains a significant problem, with spam emails accounting for a significant portion of overall email traffic. According to [AAG's recent email spam statistics](https://aag-it.com/the-latest-phishing-statistics/), approximately 3.4 billion spam emails are sent daily, with 48% of all emails sent in 2022 being spam.

The impact of spam on both individuals and organizations is far-reaching and multifaceted. Spammers often advertise products or services that may be harmful, illegal, or offensive, exposing recipients to potential health risks or offensive materials. Additionally, the presence of spam emails in users' inboxes leads to decreased productivity as valuable time and attention are diverted to reading and filtering through spam messages, impacting work efficiency and overall output. Furthermore, the influx of spam emails strains mail servers, reducing efficiency and potentially leading to delays or system overloads.

The detection and filtering of spam is a classic problem in machine learning. Traditional rule-based approaches, as well as hand-coded solutions have failed to provide an effective and long-term ability to detect and filter spam. As a result machine learning approaches have been applied with great success. The ability to teach a machine to detect spam "bottom-up", using data and allowing the machine to find the patterns,  instead of "top down" has resulted in a more robust and effective solution to the problem of spam detection.

## Spam & Ham:

The labels "spam" and "ham" in the context of email categorization [originated](https://en.wikipedia.org/wiki/Email_spam) from a humorous reference to a sketch by the British comedy group Monty Python. In this sketch, a group of Vikings repeatedly chants "spam" while a waitress recites a menu item containing spam (a canned meat product). The connection between the repetitive chanting and unwanted emails led to the term "spam" being used to describe unsolicited bulk emails. On the other hand, "ham" is used to refer to legitimate or non-spam emails. The term "ham" is believed to have been derived as the opposite of "spam," creating a playful contrast between the two categories.

## Problem Statement: 

The author set out to invesitgate the impact of machine learning models ability to detect spam. In this paper, emphasis is placed on the detection of spam emails, as opposed to phishing emails. Specifically, this is a classification problem in machine learning - deciding between two classes, spam or ham.

In this research paper, several methods of spam detection are investigated. The performance of various machine learning algorithms is evaluated using a publicly available dataset of spam emails. The results of this study provide insight into the effectiveness of different machine learning algorithms for spam detection and highlight the strengths and weaknesses of each approach.

The primary models tested were traditional models, including Naive Bayes and Decision Trees, and a more advanced model using Neural Nets was created. Neural Nets have recently become popular in the field of spam detection due to their ability to learn complex patterns and features from data. The results of this study indicate that Neural Nets outperform traditional models in terms of accuracy and precision, making them a promising approach for spam detection.

# Background

what other people did

my approach

feature selection

## The Difficulty Of Email Datasets

Email by its very nature is a private form of communication. As a result, it is difficult to obtain large datasets of emails that can be used for research purposes. The majority of email datasets are proprietary and cannot be shared publicly due to privacy concerns. Additionally, the process of manually labeling emails as spam or ham is time-consuming and expensive, making it difficult to obtain large datasets of labeled emails.

As a result, several standard datasets have arisen, and show up frequently in the scientific literature. This includes the Sbpambase dataset from UCI, SpamAssassin 

# Data

The data sources used for this project were Spambase, Spamassassin, and the Enron corpus respectively. 

The Spambase data was used for establishing a baseline with Naive Bayes classification models. The Spamassassin and Enron datasets were combined to create a larger dataset for training and testing the machine learning models.

## Spambase:

The Spambase dataset is a collection of 4,601 email messages, 2,301 of which are spam and 2,300 of which are not spam. The dataset was created by researchers at the University of California, Irvine, and was first released in 1998. The emails in the Spambase dataset were collected from a variety of sources, including Usenet newsgroups, mailing lists, and web forums. For additional information on the spambase dataset, as well as background on spam itself, see the [following](https://archive.ics.uci.edu/dataset/94/spambase).

## SpamAssassin: 

The SpamAssassin dataset is a collection of email messages that were used to train the SpamAssassin spam filter. The dataset was created by the SpamAssassin team and was first released in 2002. The emails in the SpamAssassin dataset were collected from a variety of sources, including Usenet newsgroups, mailing lists, and web forums. It includes a total of 6047 messages, with an approximate 31% spam ratio. It can be found [here](https://spamassassin.apache.org/old/publiccorpus/).

## Enron:

The Enron dataset is a collection of email messages that were sent and received by employees of Enron Corporation. The dataset was released in 2003 by the Federal Energy Regulatory Commission (FERC) as part of its investigation into the Enron scandal. The datasset was later purchased by a computer scientist at the University of Massachusetts Amherst and released to the public in 2004. The Enron dataset contains approximately 500,000 emails exchanged between 158 employees over a period of several years.

For more information on the Enron corpus see [this paper](https://link.springer.com/chapter/10.1007/978-3-540-30115-8_22) as well as the respective [Wikipedia article](https://en.wikipedia.org/wiki/Enron_Corpus).

## The Combined Dataset:

It is important to note that the author did not use the Enron corpus itself, but the curated dataset created by the authors of the research paper [Spam Filtering With Naive Bayes - Which Naive Bayes?](https://www.researchgate.net/profile/Vangelis-Metsis/publication/221650814_Spam_Filtering_with_Naive_Bayes_-_Which_Naive_Bayes/links/00b4952977a32a9949000000/Spam-Filtering-with-Naive-Bayes-Which-Naive-Bayes.pdf). The Enron dataset itself can be found [here](https://www.cs.cmu.edu/~./enron/).

In the research paper the authors have curated a dataset, combining data from the Enron corpus as well as some of their own. This dataset was combined with the SpamAssassin dataset in order to provide a balanced number of spam and non-spam emails. It can be found [here](https://www2.aueb.gr/users/ion/data/enron-spam/).

# Methods & Results

When it comes to the metric used for the models, precision was chosen as it prioritizes the reduction of false positives. In the context of spam detection, false positives refer to legitimate emails being classified as spam. This is a significant concern as it can lead to important emails being missed or overlooked. On the other hand, false negatives refer to spam emails being classified as legitimate. While this is also undesirable, it is less of a concern as these emails can be filtered out manually.

## Dummy Model

From the two previously combined datasets, a dummy model was created. Using the DummyClassifier object from Sklearn, the model was created and predicted the average of the training data. Due to the nearly even split of ham and spam, this resulted in an accuracy of 50%. This dummy model provided a baseline from which to compare the other models.

## Naive Bayes

Many of the traditional predictive statistics models used for email ham/spam classifiction are based on the Naive Bayes algorithm. This algorithm is based on Bayes' theorem, which states that the probability of an event occurring given that another event has already occurred can be calculated using the following formula:

$P(A|B) = P(B|A) * P(A) / P(B)$

The primary advantages of using the Naive Bayes algorithm are that it is simple, fast, and efficient. Additionally, it is not affected by irrelevant features and can handle a large number of features. However, it is also known to be a poor estimator. In addition, it assumes that all features are independent, which is not always the case.

Two models were created, both Gaussian Naive Bayes models. The models were used for three runs total. With the first being on the preprocessed data from SpamBase. The second and third were run on data processed by the author.

For the run on the preprocessed data the model resulted in an accuracy of 82%. When it was run on the data processed by the author, the model resulted in a significant decrease, with the accuracy of the Enron database resulting in 62%, and the combined 11,000 emails resulting in an accuracy of 56%. The difference in accuracy of the two is likely a result of a class imbalance in the Enron dataset (and also why the spamassassin dataset was added in to begin with).

## Decision Tree

As a comparison, a Decision Tree model was created. This model was created using the DecisionTreeClassifier object from Sklearn. The model was run on the initial Enron dataset, resulting in an accuracy of 96%. When it was run on the data processed by the author, the model resulted in a decrease in accuracy, but not as severe as before. The accuracy of the combined 11,000 emails was 89%

## Neural Net

The primary model created was a neural net. This model was creating using Keras and Tensorflow. The model was designed with the following parameters:

* Input layer with 10 nodes
    * relu activation
* 4 hidden layers, each with 20 nodes
    * relu activation
* output layer 
    * 1 node 
    * sigmoid activation
* 10,000 epochs with early stopping
    * monitor was val_loss
    * patience was 500

This model was applied directly to the created dataset of 11,000 emails. This resuled in a runtime of around three minutes, and an accuracy that ranged between 93-96%. 

From here the weights were exported and then used to create a Streamlit application. This application was designed to allow users to input their own email and have the model predict whether it was spam or ham. The applet takes in both .eml files, as well as copied plain text. 

The applet can be found [here](https://github.com/Kaewin/capstone/blob/main/app2.py).

# Discussion

The primary limitation of the model produced is it's ability to process emails with alternative methods to deliver spam. For instance, having the body as an image, or having the body be a link to a website. The model would not be able to detect these emails as spam. Also, the model in it's current state primarlly runs on plain text. Most emails are heavily formatted with HTML, and the model will flag these emails as spam, even though they are not.

Compared to other approaches in the research surveyed, my model isn't nearly as powerful, but still effective. The author believes this comes down to the features used. The author's model only used the subject line and the body of the email. Other models used a variety of features, including the sender, the recipient, the time sent, and the length of the email, as well as derived features. 

# Conclusion

Improvements to the model could be made by using curated features with high information gain. The researchers in the paper [Feature selection for Spam and Phishing detection](https://ieeexplore.ieee.org/document/5706696) have identified over 40 of the most commonly used features for email spam detection, as well as 9 features with the highest information gain. Implementing these features would likely result in a more powerful model.

# Acknowledgements

I would like to express my sincere gratitude to Flatiron School for providing me with the opportunity to learn and grow as a data scientist. I am also grateful to my teachers, Jelly, Joseph, and David, for their invaluable guidance and support throughout the research process. This research would not have been possible without their help, guidance, and endless patience.


# References


## Bibliography: 

“A Plan for Spam.” Accessed June 30, 2023. http://www.paulgraham.com/spam.html.

Anthdm. “How I Used Machine Learning to Classify Emails and Turn Them into Insights (Part 1).” Medium, December 17, 2017. https://towardsdatascience.com/how-i-used-machine-learning-to-classify-emails-and-turn-them-into-insights-efed37c1e66.

“How I Used Machine Learning to Classify Emails and Turn Them into Insights (Part 2).” Medium, December 19, 2017. https://towardsdatascience.com/how-i-used-machine-learning-to-classify-emails-and-turn-them-into-insights-part-2-6a8f26477c86.

baeldung. “Publicly Available Spam Filter Training Sets | Baeldung on Computer Science,” June 23, 2020. https://www.baeldung.com/cs/spam-filter-training-sets.
“Better Bayesian Filtering.” Accessed July 3, 2023. https://www.paulgraham.com/better.html.

Cranor, Lorrie Faith, and Brian A. LaMacchia. “Spam!” Communications of the ACM 41, no. 8 (August 1, 1998): 74–83. https://doi.org/10.1145/280324.280336.
Douzi, Samira, Feda A. AlShahwan, Mouad Lemoudden, and Bouabid El Ouahidi. “Hybrid Email Spam Detection Model Using Artificial Intelligence.” International Journal of Machine Learning and Computing 10, no. 2 (February 29, 2020). https://doi.org/10.18178/ijmlc.2020.10.2.937.

Gaussian Naive Bayes, Clearly Explained!!!, 2020. https://www.youtube.com/watch?v=H3EjCKtlVog.

Guzella, Thiago S., and Walmir M. Caminhas. “A Review of Machine Learning Approaches to Spam Filtering.” Expert Systems with Applications 36, no. 7 (September 1, 2009): 10206–22. https://doi.org/10.1016/j.eswa.2009.02.037.

“Review: A Review of Machine Learning Approaches to Spam Filtering.” Expert Systems with Applications: An International Journal 36, no. 7 (September 1, 2009): 10206–22. https://doi.org/10.1016/j.eswa.2009.02.037.

Heckerman, David, Eric Horvitz, Mehran Sahami, and Susan Dumais. “A Bayesian Approach to Filtering Junk E-Mail,” 1998. https://www.microsoft.com/en-us/research/publication/a-bayesian-approach-to-filtering-junk-e-mail/.

Klimt, Bryan, and Yiming Yang. “The Enron Corpus: A New Dataset for Email Classification Research.” In Machine Learning: ECML 2004, edited by Jean-François Boulicaut, Floriana Esposito, Fosca Giannotti, and Dino Pedreschi, 3201:217–26. Lecture Notes in Computer Science. Berlin, Heidelberg: Springer Berlin Heidelberg, 2004. https://doi.org/10.1007/978-3-540-30115-8_22.

Magdy, Safaa, Yasmine Abouelseoud, and Mervat Mikhail. “Efficient Spam and Phishing Emails Filtering Based on Deep Learning.” Computer Networks 206 (April 7, 2022): 108826. https://doi.org/10.1016/j.comnet.2022.108826.

Metsis, Vangelis, Ion Androutsopoulos, and Georgios Paliouras. “Spam Filtering with Naive Bayes - Which Naive Bayes?,” 2006.
Naive Bayes, Clearly Explained!!!, 2020. https://www.youtube.com/watch?v=O2L2Uv9pdDA.

Yasin, Adwan, and Abdelmunem Abuhasan. “An Intelligent Classification Model for Phishing Email Detection.” International Journal of Network Security & Its Applications 8, no. 4 (July 30, 2016): 55–72. https://doi.org/10.5121/ijnsa.2016.8405.

“The Latest Phishing Statistics (Updated July 2023) | AAG IT Support.” Accessed July 11, 2023. https://aag-it.com/the-latest-phishing-statistics/.

Toolan, Fergus, and Joe Carthy. “Feature Selection for Spam and Phishing Detection.” In 2010 ECrime Researchers Summit, 1–12, 2010. https://doi.org/10.1109/ecrime.2010.5706696.

## Datasets Used:

Mark Hopkins, Erik Reeber. “Spambase.” UCI Machine Learning Repository, 1999. https://doi.org/10.24432/C53G6X.

“The Enron-Spam Datasets.” Accessed June 17, 2023. https://www2.aueb.gr/users/ion/data/enron-spam/.

“Index of /Old/Publiccorpus.” Accessed June 17, 2023. https://spamassassin.apache.org/old/publiccorpus/.
