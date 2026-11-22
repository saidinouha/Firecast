## Progress
- [x] Setup
- [x] Fire data download
<br>explorer firms_2013_2024(explore_fire.py)
<br>Clean the fire data(clean_fire.py) (we must install pip install geopandas in terminal first)


- [x] Weather data download
- [ ] Dataset building
- [ ] EDA and preprocessing
- [ ] Models and evaluation
- [ ] Presentation

## step 2 : Fire data download 

- explorer firms_2013_2024(explore_fire.py)
- Clean the fire data(clean_fire.py) (we must install pip install geopandas in terminal first)
- Check for persistent hotspots : 
    <br> A real vegetation fire burns for a day or two and moves on. A fixed source keeps showing up at the same location. We count, for each ~1 km spot, on how many different days it was detected.
    <br> (check_hotspots)
- 
### Define the grid cells
(build_grid.py)
The model predicts per zone, so we divide Tunisia into square cells of 0.5° (about 50 km). Each row of the final dataset will be one cell on one day. I chose 0.5° for two reasons: it gives enough fires per cell to learn from, and NASA POWER’s weather data has about 0.5° resolution, so a finer grid wouldn’t give more weather detail.

Each cell is identified by integer indices (no floating-point problems), and we’ll download the weather at each cell’s center.

Why cells with zero fires matter

Some southern cells may never have a fire. Keeping them is correct, because the model must also learn where fires do not happen. But if there are many all-zero cells, they inflate the “no fire” class without adding information. I’ll show you how to handle this after seeing your numbers.


## Step 3: Download the weather (NASA POWER)
What and why

For each of the 94 cells, we download daily weather for 2013 to 2024 at the cell’s center. These are the features the model learns from. Variables:

Column	Meaning
T2M, T2M_MAX, T2M_MIN	Mean, max, min temperature at 2 m (°C)
RH2M	Relative humidity (%)
WS2M	Wind speed at 2 m (m/s)
PRECTOTCORR	Rainfall (mm/day)

One request returns the entire 12 years for one cell, so we need only 94 requests. POWER’s native resolution is coarser than 0.5° (about 0.5° latitude × 0.625° longitude), so some neighboring cells may receive identical or nearly identical weather. That’s expected, not a bug.


## Step 4: Build the cell-day dataset

## Step 5: Exploration (EDA)
What and why

Exploration means looking at the data before modeling. The brief asks for three things here:
display and interpret the data, visualize with seaborn, and check for outliers.
This is also where you spot problems before a model hides them.


## Step 6: Split, transform, select features
What and why

- Time-based split. Train on 2013 to 2021, test on 2022 to 2024. The model is tested on the “future”, as in real use. A random split would put Monday in training and Tuesday in test, and the test would be too easy.
- log1p on rain and fire counts.
- Remove identifier columns. cell_id and date are only labels for rows, and year would let the model memorize a trend that can’t continue into the test years. month is redundant with doy_sin/doy_cos. Location is kept through lat_c and lon_c.
- Correlation filter, computed on the training years only. If two features have correlation above 0.9, keep one.
- Importance ranking to complete the feature selection the brief requires.
- 
scaling will happen in Step 7 inside a scikit-learn Pipeline, not in this step. That way, during cross-validation the scaler is fitted on each fold’s training part only. It’s the correct way to avoid leakage, and it’s easy.


Decision: which features to keep

A simple rule that you can defend in the presentation: keep a feature if it is in the top 12 of at least one method.
That keeps 13 features and drops four: humidity_7d, rain_log, rain_7d_log, and wind.

## Step 7: Baselines and models

What and why

We train two models and compare them against two trivial rules. The rules answer the question “is the model actually adding anything?”

Model	    Idea
Baseline A, monthly rate	Score = how often fires happened in that month in the training years. Ignores weather and place.
Baseline B, fire in last 7 days	Predict “fire tomorrow” if there was a fire in this cell in the last 7 days.
Logistic Regression;	A weighted sum of the features squeezed into a probability. Simple, readable.
Random Forest;	Many decision trees that vote. Can capture non-straight-line patterns like seasonality.


Key choices:

- class_weight="balanced" makes each rare fire count as much as many non-fires during training, which counters the 64:1 imbalance.
- GridSearchCV with 4 folds (your teacher’s CV=4) tries each parameter combination and keeps the best. I use TimeSeriesSplit(n_splits=4), so each fold trains on the past and validates on the following period, and the data is already sorted by date. Mention this to your teacher.
- Scoring is PR-AUC (average_precision), not accuracy. PR-AUC measures how well the model ranks fire days above non-fire days, ignoring any threshold. A random model scores about the positive rate, which is 0.0124 in your test set, so that’s the number to beat.
- Scaling is inside a Pipeline for Logistic Regression, so the scaler is fitted only on each fold’s training part. Random Forest doesn’t need scaling.
- The test set is used once, at the end, and never for choosing parameters.


Step 7 worked: both models are far above the random reference, and every result was measured on the held-out 2022–2024 test years.

What your results tell us

The models learn real signal. A random model scores a PR-AUC of 0.0124 (the test positive rate). Random Forest scores 0.179 (about 14× random) and Logistic Regression 0.160 (about 13×). ROC-AUC is 0.85 to 0.87. Random Forest wins, but only moderately.

I got one prediction wrong. I said Baseline B (fire in the last 7 days) would probably be competitive. On PR-AUC it isn’t: 0.045 against 0.160 and 0.179. There is a caveat, though. Baseline B produces only yes/no answers, so its PR-AUC isn’t directly comparable with the models’ ranking scores. At threshold 0.5 it even has slightly higher precision (8.6% vs 7.1%), but it finds far fewer fires (recall 44% vs 65%). Step 8 adds a fairer baseline that uses the fire count as a continuous score.

The monthly-rate baseline is weak (PR-AUC 0.019). Season alone barely beats random, so the models gain a lot from location and recent fire history.

Precision is low at threshold 0.5, as I predicted. Random Forest flags 11,537 cell-days and only 824 of them have a fire the next day, so about 13 of 14 alarms are “false”. It still finds 824 of the 1,274 fires (recall 65%). I can’t say from this output why so many alarms are wrong. One possibility is that the satellite misses fires (cloud cover, overpass timing), but I haven’t checked that.

Cross-validation scores are higher than test scores (0.246 vs 0.179 for RF). The folds cover different periods than the test, and the test has a lower fire rate (1.24% vs 1.53%). The fold standard deviation of 0.03 shows how much the score moves between periods, so this gap isn’t a sign of a bug.

Tuning changed little. For Logistic Regression, C=0.1, 1, and 10 give identical scores (0.2209), and C=0.01 is better by 0.0003, far inside the 0.03 noise. With 13 features and 300,000 rows, regularization barely matters, and that is a finding you can state. For Random Forest the differences are 0.2408 to 0.2456, also small.

One methodological weakness: the best values sit at the edge of the grids (C=0.01 is the smallest value, and max_depth=12 is the largest). A reviewer could ask whether better values exist just outside. A small extension answers that.

Optional (about 5 minutes): extend the grids

In train_models.py, change these two lines:

````
python

lr_search = run_search("logreg", lr_pipe, {"model__C": [0.001, 0.01, 0.1, 1, 10]}, n_jobs=-1, verbose=1)
````

````
python

rf_grid = {"max_depth": [8, 12, 20], "min_samples_leaf": [100, 300]}
````
Then rerun it. It will take longer, since there are 6 combinations and deeper trees. The script overwrites best_params.json and the predictions, and Step 8 reads those files. If it shows the same winners, you can say the optimum is not at a boundary. If you skip this, Step 8 still works.


## Step 8a: Evaluation
What and why

The brief asks for confusion matrices and summary tables. We also fix the threshold problem: 0.5 is arbitrary, and the models were trained with class weights, so it isn’t a sensible cut-off.

Choosing the threshold without cheating: the threshold must not be picked on the test set, or the test results would be optimistic. We pick it on out-of-fold predictions from the training years (each fold’s validation part, predicted by a model that didn’t see it), choosing the value that maximizes F1. F1 balances precision and recall, which is a neutral choice. A real fire service might prefer higher recall, and you can say so in the presentation. Then we apply that threshold to the test set once. Small approximation: the fold models are trained on less data than the final model, so their scores are slightly different.

The script also adds a continuous baseline (the 30-day fire count used as a score) and the test results per year.


there is a mistake in the first code , there is also the correction bellow , correct it befor running it 


Step 8b: Ablation study
What and why

An ablation removes one group of features at a time and measures how much the score drops. It answers the question your teacher will most likely ask: “what is your model actually using?” Our 13 features fall into four groups:

Group	Features
Fire history	fires_today_log, fires_last3_log, fires_last7_log, fires_last30_log
Location	lat_c, lon_c, frac_inside
Season	doy_sin, doy_cos
Weather	temp_max, temp_min, humidity, rain_30d_log

We keep the best parameters fixed (no re-tuning), so the comparison is fast and only the features change.