# Complaint-Classification-Analysis
Comparing the top 5 ML strategies for imbalanced NLP; finds a Voting Ensemble is more robust than standalone linear models (SGD, PA), SVM+SMOTE, or LightGBM.
# Dataset Source
We worked with a single, focused dataset: complaints-2025-11-01_06_26.csv.

Data Count: 26,743 complaints.

Source Filtering: To create a high-quality dataset for this NLP task, specific filters were applied on the CFPB web portal before downloading. The data was filtered to only include complaints that contained a "Consumer complaint narrative" and were categorized under one of three products: 'Credit reporting', 'Debt collection', or 'Mortgage'.
# Preprocessing and Special Treatment
Yes, preprocessing and special treatment were critical to the success of this project.

Initial Filtering (at Source): The most important step was pre-filtering on the CFPB website. This ensured we had a 100% complete dataset with no rows missing the primary Consumer complaint narrative feature.

Code-Level Cleaning: A dropna() was used on the Product and Consumer complaint narrative columns to remove any potential null values.

Text Vectorization: No complex manual text cleaning was performed. Instead, we used scikit-learn's TfidfVectorizer (and CountVectorizer) to handle all text preprocessing, which included:

Converting all text to lowercase.

Removing common English "stop words" (e.g., 'the', 'is', 'a').

Filtering out very rare and very common terms (min_df=5, max_df=0.9).

Converting the cleaned text into numerical vectors for the models.

Imbalance Treatment: The most important "special treatment" was handling the severe 93.5% class imbalance. We proved that standard models were failing dueD to this. The core of our experiment was applying and comparing two key balancing strategies:
Algorithmic Balancing (e.g., class_weight='balanced')

Data-Level Balancing (e.g., SMOTE - Synthetic Minority Over-sampling TEchnique)
# Methods
Our approach was a systematic, comparative analysis designed to find the most robust model for a highly imbalanced classification problem. The core problem was that 93.5% of the data belonged to one class, making "lazy" models seem accurate while failing to identify rare complaints.

Our method was to first establish a baseline, then test and compare multiple, increasingly complex solutions to this core problem.

Baseline (The "Problem" Case): We first trained standard models like LinearSVC and RandomForest on the raw, imbalanced data. As expected, these models achieved high accuracy (96-97%) but were useless in practice. Their f1-scores for the rare 'Mortgage' class were as low as 0.32, proving this simple approach was not viable.

Comparative Analysis (The "Solutions"): We then tested and compared different strategies specifically designed to handle the imbalance. This project focused on finding the single best-performing model, with all other models serving as comparisons.

Alternative 1: Algorithmic Balancing (class_weight='balanced')

We tested fast, modern linear models: Stochastic Gradient Descent (SGD) Classifier and PassiveAggressiveClassifier.

This was a simple, one-line code change that forces the model to "pay more attention" to the rare classes.

Alternative 2: Data-Level Balancing (SMOTE)

We tested a different approach: LinearSVC with SMOTE (Synthetic Minority Over-sampling TEchnique).

This method changes the data itself by creating new, synthetic examples of the rare 'Mortgage' and 'Debt collection' complaints for the model to learn from.

Alternative 3: Different Model Families (LightGBM)

We tested a modern, high-performance boosted tree model, LightGBM, to see if a non-linear approach could perform better. This was a direct comparison to the RandomForest model, which failed completely.

The Champion Model (The "Best" Architecture):

After identifying the top-performing models from our experiments (SGD, PassiveAggressive, and SVM+SMOTE), we combined them into a final Voting Ensemble.

This "meta-model" architecture is our final and best solution. It is more robust than any single model because it uses a "democracy" approach: for a complaint to be classified, at least two of our three champion models must agree on the prediction. This makes the system far less likely to make an error on a single, ambiguous complaint.

This systematic approach was the right one because it didn't just pick one model. It isolated the problem (imbalance) and tested multiple valid solutions (algorithmic, data-level, and ensemble) to prove which one was truly the most effective and robust.
# Steps to Run the Code
# 1. Clone the Repository
git clone [YOUR_GITHUB_REPO_LINK]
cd [YOUR_REPO_NAME]
# 2. Create a Virtual Environment (Recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
# 3. Install Required Libraries
pip install pandas scikit-learn imbalanced-learn matplotlib seaborn lightgbm
# 4. Run the Final Model Scripts
# . To run the #1 Champion (Voting Ensemble):
python run_model_1_voting_ensemble.py
# . To run the #2 Model (PassiveAggressive):
python run_model_2_passive_aggressive.py
# . To run the #3 Model (SGDClassifier):
python run_model_3_sgd_classifier.py
# . To run the #4 Model (SVM + SMOTE):
python run_model_4_svm_smote.py
# . To run the #5 Model (LightGBM):
python run_model_5_lightgbm.py
# . View the Results
Each script will train the model, print the final classification_report (with accuracy, F1-score, etc.) to your console, and save a professional confusion matrix (e.g., confusion_matrix_voting_ensemble.png) to the root folder.
# Architectural Workflow of the The Vote Ensemble Model:
![Voting Ensemble Architecture](Mermaid.png)
# Experiments/Results Summary
The core of this project was a series of experiments to find the best architectural and hyperparameter choices for this specific imbalanced NLP task. We compared our engineered models against two baselines:

Standard Unbalanced Methods: (e.g., a default RandomForest or LinearSVC). As seen in our notebook, these "standard" methods failed, producing misleadingly high accuracy but failing on rare classes.

Advanced Baseline Method: (e.g., a default, 1-epoch DistilBERT).
![Project Summary Flowchart](images/_- visual selection (9).png)

Our experiments focused on finding the best combination of architectural choice (Linear vs. Tree vs. Ensemble) and balancing technique (None vs. class_weight='balanced' vs. SMOTE).
# Comparative Performance
The key performance metrics were the Macro Average F1-Score (for overall model balance) and the per-class F1-Score for 'Mortgage' (the rarest class). The following table illustrates the final experimental results for our Top 5 models:
![Final Model Performance Metrics](images/MetricsTable.png)
<img src="images/MetricsTable.png" alt="Final Model Performance Metrics" width="8000"/>
This table clearly shows that the simple, fast linear models using the class_weight hyperparameter were the most effective, and the Voting Ensemble (our final architectural choice) was the most robust.
Below are the confusion metrics for the top 2 best performing models:
![Confusion Matrix for Vote Ensemble](images/Im1.png)
![Confusion Matrix for PassiveAggressive (Balanced, CountVec)](images/im2.png)
# Conclusion
Our project was a step-by-step journey to find the best model for a highly imbalanced classification problem.

1. We started with a real-world dataset of consumer complaints, where 93.5% of the data was one class.

2. Our baseline models (like RandomForest) gave a high 97% accuracy, but this was a "vanity metric."

3. Visualizing the results showed they failed to identify the rare classes, with f1-scores as low as 0.32.

4. This proved the core problem was the class imbalance.

5. We then tested two different solutions:

Algorithmic Balancing (class_weight='balanced') on fast linear models (SGD, PassiveAggressive).

Data-Level Balancing (SMOTE) on a LinearSVC.

6. Both experiments were a success, dramatically improving the minority class f1-scores from ~0.32 to over 0.84.

7. To create the most robust solution, we combined our top 3 champions into a final Voting Ensemble.

8. This Voting Ensemble achieved the highest, most stable performance (0.86 Macro F1-Score), making it our champion model.

9. The key lesson learned is that a well-engineered "classic" ML solution (the balanced ensemble) was simpler, faster, and outperformed a baseline DistilBERT (Deep Learning) model for this specific, imbalanced task.
