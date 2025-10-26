# Predicting the Patterns and Causes of Research Paper Retractions (1940–2024)
## Acknowledgements
This project was completed in collaboration with Chi Doan, Trang Doan, and Zimu Su at Charles Darwin University (CDU).
# Context
In 2023, more than 10,000 research papers were retracted, the highest number ever recorded. This means more studies were found to have mistakes or dishonest work. The rise in retractions is worrying because governments often use scientific research to make laws and policies. If that research is wrong, it can lead to bad decisions and loss of public trust. Tracking and studying these retractions helps us understand why they happen and how to improve honesty and quality in science.

# Objective
  * Explore relevant patterns, interesting, and actionable trends of past retractions.
  * Predict important aspects of future retractions.

# Dataset
Dataset **“retractions35215.csv”** originates from the **Retraction Watch** organisation.
The dataset includes **35,215 rows** and **21 columns** and was **last updated on 15 January 2024**.

An additional dataset from the **Retraction Watch database** was used for model validation.

More information can be found at: [https://retractionwatch.com/retraction-watch-database-user-guide/retractionwatch-database-user-guide-appendix-a-fields/](https://retractionwatch.com/retraction-watch-database-user-guide/retractionwatch-database-user-guide-appendix-a-fields/)

# Methodology
The project used a structured methodology combining exploratory data analysis and predictive modelling. In the first phase, historical data on research paper retractions from the Retraction Watch database was analysed to uncover key patterns. Statistical and visual analysis techniques were applied to explore retraction trends by year, subject area, journal, publisher, institution, country, and article type. Common reasons for retractions, retraction duration, citation impact, and authorship patterns were examined to understand systemic issues in research integrity and publication processes.

In the second phase, a balanced dataset of retracted and non-retracted papers was created by merging Retraction Watch and Scopus data. Data cleaning, standardisation, and feature engineering were performed, including normalising author names, calculating retraction counts by publisher and journal, and adding metrics such as SJR scores and citation counts. Multiple machine-learning models—such as Naïve Bayes, Random Forest, and K-Nearest Neighbours—were trained and evaluated using pipelines that ensured consistent preprocessing. Model performance was compared using accuracy and classification metrics to identify the most effective predictive model for detecting potential future retractions.

# Key findings
## **Objective 1 – Trend Analysis of Research Paper Retractions**

The first objective aimed to identify major historical patterns and underlying causes of research paper retractions using the **Retraction Watch** dataset.

* **Sharp Increase Over Time:**
  The number of retractions has risen dramatically, reaching a peak of **6,034 in 2023**, compared to only one in 1940. This reflects both the rapid growth of scientific publications and increased scrutiny of research integrity.
  
<img src="https://github.com/user-attachments/assets/7b3d96de-2e6b-4db4-a663-15274809375e" 
     alt="image" 
     style="width:60%; height:auto;" />

* **Dominant Subject Areas:**
  **Biological and Health Sciences** recorded the highest retraction rates, followed by **Business, Technology, and Physical Sciences**. These fields face higher ethical and methodological pressures due to their societal and clinical relevance.

* **Journal, Publisher, and Regional Concentration:**
  Retracted papers are concentrated within a small number of publishers—**the top 10 accounted for 74.42%** of all retractions.
  Geographically, **China** contributed **49.33%** of total retractions, highlighting a strong regional pattern.
  
  <img src="https://github.com/user-attachments/assets/fb85bc61-5303-4a76-9e35-1967d96e5e88" alt="image" style="width:60%; height:auto;" />

* **Common Retraction Causes:**
  The most frequent issues were linked to **publication and peer review processes (40%)**, followed by **data integrity problems (36%)** and **authorship disputes (11%)**.

* **Additional Insights:**

  * Most retractions occurred **within one year of publication**.
  * **Single-author papers** were more likely to be retracted than multi-author works.
  * **Non-paywalled articles** experienced more retractions, likely due to higher public visibility.
  * Although many retracted papers had low citation counts, a few highly cited papers were also retracted, indicating that **influence does not equal integrity**.

---

## **Objective 2 – Predictive Modelling of Retractions**

The second objective focused on developing a **machine learning model** to predict potential future retractions based on the patterns identified in Objective 1.

* **Dataset Creation:**
  A combined dataset of **11,389 papers** was compiled—**5,044 retracted** (from Retraction Watch) and **6,345 non-retracted** (from Scopus). Data cleaning included normalising author names, removing anomalies, and adding variables such as **publisher/journal retraction history**, **SJR scores**, and **author retraction counts**.

* **Model Performance:**
  Multiple models were compared using a unified pipeline. The **K-Nearest Neighbours (KNN)** classifier achieved the **highest accuracy of 94.81%**, followed closely by **Random Forest at 94.35%**.

* **Feature Importance:**
  Random Forest analysis identified the **number of retractions by publisher**, **number of retractions by journal**, and **total retractions by authors** as the **most influential predictors** of future retractions.
  
  <img src="https://github.com/user-attachments/assets/ed174fbc-56c0-4dcf-80c1-15bfb2f83eff" alt="image" style="width:60%; height:auto;" />

* **Validation:**
  The KNN model was re-evaluated using these selected features, confirming their predictive strength with consistently high performance metrics.

These findings demonstrate that **author and publisher reputation** play a decisive role in determining retraction risk, and that predictive analytics can effectively flag at-risk publications, supporting improved research integrity.

# References
1.	Callaway, E. (2022). Retractions are increasing, but not enough. Nature. https://www.nature.com/articles/d41586-022-02071-6
2.	Callaway, E. (2023). Retractions: A record high highlights scrutiny. Nature. https://www.nature.com/articles/d41586-023-03974-8
3.	Fang, F. C., Steen, R. G., & Casadevall, A. (2012). Misconduct accounts for the majority of retracted scientific publications. Proceedings of the National Academy of Sciences, 109(42), 17028-17033. https://doi.org/10.1128/iai.05661-11
4.	Steen, R. G., Casadevall, A., & Fang, F. C. (2013). Why has the number of scientific retractions increased? PLOS Medicine. https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1001563
5.	Steen, R. G. (2013). Retractions in the scientific literature: is the incidence of research fraud increasing? Journal of Medical Ethics, 37(4), 249-253. https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0068397
6.	Modukuri, S. A., Rajtmajer, S., Squicciarini, A. C., Wu, J., & Giles, C. L. (2021). Understanding and predicting retractions of published work. ODU Digital Commons. Retrieved from https://digitalcommons.odu.edu/cgi/viewcontent.cgi?article=1279&context=computerscience_fac_pubs
7.	Clarinda Cerejo, 2013, What are the most common reasons for retraction, https://www.editage.com/insights/what-are-the-most-common-reasons-for-retraction?refer=scroll-to-1-article&refer-type=article 
8.	Gemma Conroy, 2019, The biggest reason for biomedical research retractions, https://www.nature.com/nature-index/news/the-biggest-reason-for-biomedical-retractions 
9.	Tang B.L, 2023, Some Insights into the Factors Influencing Continuous Citation of Retracted Scientific Papers, https://www.researchgate.net/publication/374555246_Some_Insights_into_the_Factors_Influencing_Continuous_Citation_of_Retracted_Scientific_Papers 


