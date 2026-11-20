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
