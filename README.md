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
