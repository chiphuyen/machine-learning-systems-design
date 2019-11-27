# Case studies

To learn to design machine learning systems, it's helpful to read case studies to see how actual teams deal with different deployment requirements and constraints. Many companies, such as Airbnb, Lyft, Uber, Netflix, run excellent tech blogs where they share their experience using machine learning to improve their products and/or processes. If you're interested in a company, you should visit their tech blogs to see what they've been working on -- it might come up during your interviews! Below are some of these great case studies.

1. [Using Machine Learning to Predict Value of Homes On Airbnb](https://medium.com/airbnb-engineering/using-machine-learning-to-predict-value-of-homes-on-airbnb-9272d3d4739d) (Robert Chang, Airbnb Engineering & Data Science, 2017)
	
	In this detailed and well-written blog post, Chang described how Airbnb used machine learning to predict an important business metric: the value of homes on Airbnb. It walks you through the entire workflow: feature engineering, model selection, prototyping, moving prototypes to production. It's completed with lessons learned, tools used, and code snippets too.

2. [Using Machine Learning to Improve Streaming Quality at Netflix](https://medium.com/netflix-techblog/using-machine-learning-to-improve-streaming-quality-at-netflix-9651263ef09f) (Chaitanya Ekanadham, Netflix Technology Blog, 2018)
	
	As of 2018, Netflix streams to over 117M members worldwide, half of those living outside the US. This blog post describes some of their technical challenges and how they use machine learning to overcome these challenges, including to predict the network quality, detect device anomaly, and allocate resources for predictive caching.

3. [150 Successful Machine Learning Models: 6 Lessons Learned at Booking.com](https://blog.acolyer.org/2019/10/07/150-successful-machine-learning-models/) (Bernardi et al., KDD, 2019).
	
	As of 2019, Booking.com has around 150 machine learning models in production. These models solve a wide range of prediction (e.g. predicting users' travel preferences and how many people they travel with) and optimization (e.g.optimizing the background images and reviews to show for each user). Adrian Colyer gave a good summary of the six lessons learned here:
	* Machine learned models deliver strong business value.
	* Model performance is not the same as business performance.
	* Be clear about the problem you're trying to solve.
	* Prediction serving latency matters.
	* Get early feedback on model quality.
	* Test the business impact of your models using randomized controlled trials.

4. [How we grew from 0 to 4 million women on our fashion app, with a vertical machine learning approach](https://medium.com/hackernoon/how-we-grew-from-0-to-4-million-women-on-our-fashion-app-with-a-vertical-machine-learning-approach-f8b7fc0a89d7) (Gabriel Aldamiz, HackerNoon, 2018)
	
	To offer automated outfit advice, Chicisimo tried to qualify people's fashion taste using machine learning. Due to the ambiguous nature of the task, the biggest challenges are framing the problem and collecting the data for it, both challenges are addressed by the article. It also covers the problem that every consumer app struggles with: user retention.

5. [Machine Learning-Powered Search Ranking of Airbnb Experiences](https://medium.com/airbnb-engineering/machine-learning-powered-search-ranking-of-airbnb-experiences-110b4b1a0789) (Mihajlo Grbovic, Airbnb Engineering & Data Science, 2019)
	
	This article walks you step by step through a canonical example of the ranking and recommendation problem. Four main steps are system design, personalization, online scoring, and business aspect. The article explains which features to use, how to collect data and label it, why they chose Gradient Boosted Decision Tree, which testing metrics to use, what heuristics to take into account while ranking results, how to do A/B testing during deployment. Another wonderful thing about this post is that it also covers personalization to rank results differently for different users. 

6. [From shallow to deep learning in fraud](https://eng.lyft.com/from-shallow-to-deep-learning-in-fraud-9dafcbcef743) (Hao Yi Ong, Lyft Engineering, 2018)
	
	Fraud detection is one of the earliest use cases of machine learning in industry. This article explores the evolution of fraud detection algorithms used at Lyft. At first, an algorithm as simple as logistic regression with engineered features was enough to catch most fraud cases. Its simplicity allowed the team to understand the importance of different features. Later, when fraud techniques have become too sophisticated, more complex models are required. This article explores the tradeoff between complexity and interpretability, performance and ease of deployment.

7. [Space, Time and Groceries](https://tech.instacart.com/space-time-and-groceries-a315925acf3a) (Jeremy Stanley, Tech at Instacart, 2017)
	
	Instacart uses machine learning to solve the task of path optimization: how to most efficiently assign tasks for multiple shoppers and find the optimal paths for them.  The article explains the entire process of system design, from framing the problem, collecting data, algorithm and metric selection, topped with tutorial for beautiful visualization.

8. [Uber's Big Data Platform: 100+ Petabytes with Minute Latency](https://eng.uber.com/uber-big-data-platform/) (Reza Shiftehfar, Uber Engineering, 2018)
	
	With massive data comes massive engineering requirement. Relying heavily on data for decision making, "from forecasting rider demand during high traffic events to identifying and addressing bottlenecks in our driver-partner sign-up process", Uber has collected "over 100 petabytes of data that needs to be cleaned, stored, and served with minimum latency." This article focuses on the evolution of analytical data warehouse at Uber, from Vertica to Hadoop to their own Spark library Hudi, each with their limitations analyzed and addressed.

9. [Creating a Modern OCR Pipeline Using Computer Vision and Deep Learning](https://blogs.dropbox.com/tech/2017/04/creating-a-modern-ocr-pipeline-using-computer-vision-and-deep-learning/) (Brad Neuberg, Dropbox Engineering, 2017)
	
	An application as simple as a document scanner has two distinct components: optical character recognition and word detector. Each requires their own production pipeline, and the end-to-end system requires additional steps for training and tuning. This article also goes into detail the team's effort to collect data, which includes building their own data annotation platform.

10. [Scaling Machine Learning at Uber with Michelangelo](https://eng.uber.com/scaling-michelangelo/) (Jeremy Hermann and Mike Del Balso, Uber Engineering, 2019)
	
	Uber uses extensive machine learning in their production, and this article gives an impressive overview of their end-to-end workflow, where machine learning is being applied at Uber, and how their teams are organized.
	
11. [Deep Learning for Recommender Systems](https://bit.ly/2XXLEDV) (Justin Basilico, Research/Engineering at Netflix, 2018)
	
	Recommendations generate over $1B yearly revenue for Netflix. The joy of spending few seconds to find something great to watch, directly impacts customer satisfaction.	
	
12. [Making Netflix Machine Learning Algorithms Reliable](https://bit.ly/2ONtXmp) (Justin Basilico, Research/Engineering at Netflix, 2017) 

	Netflix develops a variety of machine learning algorithms (including regression, factorization, topic modeling, ensemble learning, neural networks, bandits, etc) to a variety of problems (personalized ranking, trending now, video similarities, search, top-n ranking, etc), to help members find content to watch and enjoy.


