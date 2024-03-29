# Applying Neural Networks To The Classification Of Email Spam - A Literature Survey And Experiment

![meme](https://imgs.xkcd.com/comics/phishing_license_2x.png)

## Links:

[Presentation PDF](https://github.com/Kaewin/Applying-Neural-Networks-To-The-Classification-Of-Email-Spam---A-Literature-Survey-And-Experiment/blob/main/capstone_slides.pdf)

[Presentation Slides (Google)](https://docs.google.com/presentation/d/11_n-Ghqe0-_EscrFUHaHen3UuLhE-letcroQExchjUg/edit?usp=sharing)

# Abstract

# Introduction

## Background:

Spam email, also known as unsolicited bulk email (UBE), has become a widespread issue in the digital era. As email has become a mandatory and widespread communication method, it has also increased spam. This is due to the low cost and ease of sending spam emails, which can reach hundreds of thousands of people with just one click. 

Those among the first to use the internet may remember how spam was everywhere, and new email accounts would quickly be flooded with spam messages. While the amount of spam has decreased recently, it remains a big issue, with spam emails making up a significant part of all email traffic. According to [recent email spam statistics from AAG](https://aag-it.com/the-latest-phishing-statistics/), around 3.4 billion spam emails are sent every day, and in 2022, almost half of all emails sent were spam.

Spam can have a significant and diverse impact on individuals and organizations. The nature of spam messaging often involves advertising products or services that may be harmful, illegal, or offensive. This can expose recipients to potential health risks or offensive materials. Furthermore, spam emails in users' inboxes can decrease productivity, as they have to spend valuable time filtering through spam messages. This can impact work efficiency and overall output. Additionally, the influx of spam emails can strain mail servers, reducing efficiency and potentially causing delays or system overloads.

Detecting and filtering spam has been a long-standing issue in machine learning. Conventional rule-based techniques and manually coded solutions have proved ineffective and short-lived in identifying and filtering spam. Consequently, machine learning techniques have been employed with remarkable outcomes. Enabling the machine to learn spam detection "bottom-up" through data analysis and pattern recognition, rather than "top-down" approaches, has resulted in a more reliable and efficient solution to the spam detection problem.

## Ham and Spam:

The terms "spam" and "ham" were originally coined in the context of email categorization. The reference point for this was a sketch by the British comedy group Monty Python where a group of Vikings repeatedly chants "spam." At the same time, a waitress recites a menu item containing spam (a canned meat product). The correlation between the repetitive chanting and unwanted emails led to the term "spam" used to describe unsolicited bulk emails. Conversely, "ham" refers to legitimate or non-spam emails. The term "ham" is believed to have been derived as the opposite of "spam," creating a playful contrast between the two categories.

## Problem Statement: 

The author investigated the impact of detecting spam with machine learning models, specifically neural nets. The paper emphasizes detecting spam emails rather than phishing ones and presents a binary classification problem -  spam or ham, rather than a ternary identification problem of all three categories - spam, ham, and phishing. 

Naive Bayes models have been traditionally used for the spam detection problem and were used in this case to establish a baseline. From there, a decision tree model was created, and then the author set out their primary goal of creating and training a neural net from the same data and feature set used on the Naive Bayes and Decision Tree Models. This model was then compared to the previous.

Due to their ability to learn complex patterns and features from data, neural nets have recently become popular in spam detection. The results of this study indicate that neural nets outperform traditional models in terms of accuracy and precision, making them a promising approach for spam detection.

# Literature Survey

In order to establish a foundation and understand where the current literature is with email spam and neural networks applied to spam, the author conducted a survey of several relevant research papers. 

## Heckerman et al. 1998 - A Bayesian Approach to Filtering Junk E-Mail

One of the earlier applications of techniques that would fall within machine learning, namely Naive Bayes, was found in Heckerman et al. (1998). In it, the authors argue for using a machine learning classifier over traditional rule-based techniques and that these filters are mature enough for real-world deployment. They also argue that using machine learning classifiers is a more robust solution to the problem of spam detection, as it is more adaptable to the ever-changing nature of spam.

The authors used a vector space model to turn emails into a workable form, where a vector represents each word in the email. The authors also used a binary representation of the data, where the presence of a word is represented by a 1, and the absence of a word is represented by a 0. 

Two experiments were conducted. Dimensionality reduction was used due to the previous step of vector space representation. The specific techniques used were Zipf's law and mutual information. Zipf's law was used only to include words used thrice. Then, the mutual information between each feature and the class was used to select the most informative text features. Some features were words, and others were hand-crafted features based on the text. Ultimately, they totaled 500 features, 35 phrasal, and 20 non-textual domain-specific features. This resulted in a regime split of words + phrases + domain-specific features.

![image1](https://github.com/Kaewin/Applying-Neural-Networks-To-The-Classification-Of-Email-Spam---A-Literature-Survey-And-Experiment/blob/main/images/heckerman1.png)

![image2](https://github.com/Kaewin/Applying-Neural-Networks-To-The-Classification-Of-Email-Spam---A-Literature-Survey-And-Experiment/blob/main/images/heckerman2.png)

From here, a naive Bayesian classifier was constructed, and then the results were shown on a ROC graph. The authors found that the naive Bayesian classifier performed well and that adding the domain-specific features improved the results. Notably, including more than just words gave superior results, and they further split their metrics into categories for each. So junk precision, junk recall, legitimate precision, legitimate recall. This is due to the impact of false positives and an effort to bias the results against false positives.

In addition to the above, they also used a cost-sensitive classification and only classified an email as junk if it was 99.9% sure it was junk.

In experiment 2, 1183 emails were used, 972 were spam, and 211 were ham. In this run, they instead opted for a 3-way split, legitimate, pornographic junk, and other junk. As before, feature selection was used, the same 3-way split of features, and the 99% threshold for classification. 

Interestingly, the authors found that the 3-way instead of 2-way split worsened the results. The primary reason was the increase in degrees of freedom of the model, requiring more parameters than the two-class model. This is known as a data-fragmentation problem. 

In the end, they conclude that the naive Bayesian classifier is an excellent solution to the spam detection problem, with an 80% reduction in spam from a live user's inbox, and that adding domain-specific features improves the results. They also conclude that the 2-way split is superior to the 3-way split and that such methods are suitable for real-world deployment.

## Androutsopoulos et al. 2000 - An Experimental Comparison

The authors aimed to build on the findings of the 1998 paper by Sahami et al. regarding spam filtering. They sought to create a model to learn from previously received spam and ham messages. To achieve this, they used a keyword-based filter and compared it to the Bayesian filter used in the previous study.

The authors constructed the Naive Bayesian classifier using manually labeled data from the received emails. They acknowledged Sahami et al. as the pioneers in applying machine learning methods to spam filtering and introduced new techniques such as lemmatization and stop-lists. They also investigated the effect of attribute size and training-corpus size on the model's accuracy.

The authors built their corpus using 1099 emails, including 418 spam and 618 ham messages. They encrypted the emails to maintain privacy while allowing machine learning models to use them. The dataset was made available to the public and named PU1.

![image1](https://github.com/Kaewin/Applying-Neural-Networks-To-The-Classification-Of-Email-Spam---A-Literature-Survey-And-Experiment/blob/main/images/paper_2_1.png)

![image2](https://github.com/Kaewin/Applying-Neural-Networks-To-The-Classification-Of-Email-Spam---A-Literature-Survey-And-Experiment/blob/main/images/paper_2_2.png)

To reduce the impact of random variation, the authors implemented 10-fold cross-validation. The dataset was randomly partitioned into ten parts, and each experiment was repeated ten times. The results showed that the NB filter outperformed the manual, label-based filter, even with a small training corpus. However, adding a lemmatizer did not significantly improve the model's accuracy.

In conclusion, the authors confirmed Sahami et al.'s findings and suggested additional safety measures and decision-theoretic cost analysis to make the filters viable.

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

<ol>
<li>“A Plan for Spam.” Accessed June 30, 2023. http://www.paulgraham.com/spam.html.

<li>Anthdm. “How I Used Machine Learning to Classify Emails and Turn Them into Insights (Part 1).” Medium, December 17, 2017. https://towardsdatascience.com/how-i-used-machine-learning-to-classify-emails-and-turn-them-into-insights-efed37c1e66.

<li>“How I Used Machine Learning to Classify Emails and Turn Them into Insights (Part 2).” Medium, December 19, 2017. https://towardsdatascience.com/how-i-used-machine-learning-to-classify-emails-and-turn-them-into-insights-part-2-6a8f26477c86.

<li>baeldung. “Publicly Available Spam Filter Training Sets | Baeldung on Computer Science,” June 23, 2020. https://www.baeldung.com/cs/spam-filter-training-sets.

<li>“Better Bayesian Filtering.” Accessed July 3, 2023. https://www.paulgraham.com/better.html.

<li>Cranor, Lorrie Faith, and Brian A. LaMacchia. “Spam!” Communications of the ACM 41, no. 8 (August 1, 1998): 74–83. https://doi.org/10.1145/280324.280336.

<li>Douzi, Samira, Feda A. AlShahwan, Mouad Lemoudden, and Bouabid El Ouahidi. “Hybrid Email Spam Detection Model Using Artificial Intelligence.” International Journal of Machine Learning and Computing 10, no. 2 (February 29, 2020). https://doi.org/10.18178/ijmlc.2020.10.2.937.

<li>Gaussian Naive Bayes, Clearly Explained!!!, 2020. https://www.youtube.com/watch?v=H3EjCKtlVog.

<li>Guzella, Thiago S., and Walmir M. Caminhas. “A Review of Machine Learning Approaches to Spam Filtering.” Expert Systems with Applications 36, no. 7 (September 1, 2009): 10206–22. https://doi.org/10.1016/j.eswa.2009.02.037.

<li>“Review: A Review of Machine Learning Approaches to Spam Filtering.” Expert Systems with Applications: An International Journal 36, no. 7 (September 1, 2009): 10206–22. https://doi.org/10.1016/j.eswa.2009.02.037.

<li>Heckerman, David, Eric Horvitz, Mehran Sahami, and Susan Dumais. “A Bayesian Approach to Filtering Junk E-Mail,” 1998. https://www.microsoft.com/en-us/research/publication/a-bayesian-approach-to-filtering-junk-e-mail/.

<li>Klimt, Bryan, and Yiming Yang. “The Enron Corpus: A New Dataset for Email Classification Research.” In Machine Learning: ECML 2004, edited by Jean-François Boulicaut, Floriana Esposito, Fosca Giannotti, and Dino Pedreschi, 3201:217–26. Lecture Notes in Computer Science. Berlin, Heidelberg: Springer Berlin Heidelberg, 2004. https://doi.org/10.1007/978-3-540-30115-8_22.

<li>Magdy, Safaa, Yasmine Abouelseoud, and Mervat Mikhail. “Efficient Spam and Phishing Emails Filtering Based on Deep Learning.” Computer Networks 206 (April 7, 2022): 108826. https://doi.org/10.1016/j.comnet.2022.108826.

<li>Metsis, Vangelis, Ion Androutsopoulos, and Georgios Paliouras. “Spam Filtering with Naive Bayes - Which Naive Bayes?,” 2006.

<li>Naive Bayes, Clearly Explained!!!, 2020. https://www.youtube.com/watch?v=O2L2Uv9pdDA.

<li>Yasin, Adwan, and Abdelmunem Abuhasan. “An Intelligent Classification Model for Phishing Email Detection.” International Journal of Network Security & Its Applications 8, no. 4 (July 30, 2016): 55–72. https://doi.org/10.5121/ijnsa.2016.8405.

<li>“The Latest Phishing Statistics (Updated July 2023) | AAG IT Support.” Accessed July 11, 2023. https://aag-it.com/the-latest-phishing-statistics/.

<li>Toolan, Fergus, and Joe Carthy. “Feature Selection for Spam and Phishing Detection.” In 2010 ECrime Researchers Summit, 1–12, 2010. https://doi.org/10.1109/ecrime.2010.5706696.
</ol>

## Datasets Used:

Mark Hopkins, Erik Reeber. “Spambase.” UCI Machine Learning Repository, 1999. https://doi.org/10.24432/C53G6X.

“The Enron-Spam Datasets.” Accessed June 17, 2023. https://www2.aueb.gr/users/ion/data/enron-spam/.

“Index of /Old/Publiccorpus.” Accessed June 17, 2023. https://spamassassin.apache.org/old/publiccorpus/.

## Images Used:

“Download Email Newsletter Email Marketing Royalty-Free Vector Graphic.” Accessed July 20, 2023. https://pixabay.com/vectors/email-newsletter-email-marketing-3249062/.

Editor, MailGuard. “What Does SPAM Stand For?” Accessed July 20, 2023. https://www.mailguard.com.au/blog/what-does-spam-stand-for/.

“Goku Thumbs up Meme Generator - Imgflip.” Accessed July 20, 2023. https://imgflip.com/memegenerator/394701793/Goku-thumbs-up.

Know Your Meme. “Boar Head Exploding Reaction Image | Boar Head Exploding.” Accessed July 20, 2023. https://knowyourmeme.com/photos/2492561-boar-head-exploding.

“Monty Python and the Spam Tram (3rd in Swiss at LaserRunner) · NetrunnerDB,” November 26, 2018. https://netrunnerdb.com/en/decklist/e98975ca-eb2d-4779-b149-396ea2d56230/monty-python-and-the-spam-tram-3rd-in-swiss-at-laserrunner-.

Nguyen, Tam. “What Is a Neural Network? A Computer Scientist Explains.” The Conversation, December 11, 2020. http://theconversation.com/what-is-a-neural-network-a-computer-scientist-explains-151897.

Rorvig, Mordechai. “Computer Scientists Prove Why Bigger Neural Networks Do Better.” Quanta Magazine, February 10, 2022. https://www.quantamagazine.org/computer-scientists-prove-why-bigger-neural-networks-do-better-20220210/.

saxena, shruti. “Precision vs Recall.” Medium (blog), May 13, 2018. https://medium.com/@shrutisaxena0617/precision-vs-recall-386cf9f89488.

Shutterstock. “Diversity Hands Raised Question Marks Stock Photo 178351865.” Accessed July 20, 2023. https://www.shutterstock.com/image-photo/diversity-hands-raised-question-marks-178351865.

xkcd. “Phishing License.” Accessed July 20, 2023. https://xkcd.com/1694/.
