# Mid Project - Social Survey Dashboard

Interactive financial-literacy dashboard built on the Israeli Central Bureau of
Statistics **Social Survey 2024** (Public Use File, 6,907 respondents).

**Live dashboard:** https://orarr2.github.io/Mid_project_Social_Survey/

## What this repository covers

This repo contains the **deployment layer** of the mid-project (my part of the
team's work). The scope here is:

- Design and generation of a single self-contained dashboard page (`index.html`).
- A 10-question Financial Literacy Quiz that gates entry to the dashboard: no
  visualization is revealed until every question is answered.
- Interactive visual analytics of the raw survey data (Overview, Demographics,
  Financial Behavior, Retirement Readiness, and a personal profile comparing
  the user's answers to the population).
- Model tab left as a placeholder for the trained model that will be plugged in
  by the modeling stage of the pipeline.

Cleaning, imputation, feature engineering and model training are handled in
other stages of the project pipeline and are not part of this repo.

## Live site

The dashboard is published as a static GitHub Pages site directly from `main`:

- URL: https://orarr2.github.io/Mid_project_Social_Survey/
- Nothing to install for the reader. Everything (Plotly.js, the map, the word
  cloud image) is embedded in `index.html`.

## Visualizations included

The dashboard uses the survey data **as-is**. No missing values are removed, no
rows are dropped, no reserved codes are replaced. CBS reserved codes (`888888` =
"don't know / refused" and `999999` = "not applicable / filter") appear in the
charts as their own categories where they occur.

| Chart | Location | Purpose |
| --- | --- | --- |
| KPI cards, raw preview, reserved-code table | Overview | Structural view of the data |
| Gender donut, age bar, income bar, education | Demographics | Population composition |
| **Folium map of Israel (Nafa subdistricts)** | Demographics | Geographic distribution of respondents with bubble size = respondent count |
| **Age x Income heatmap** | Demographics | Joint distribution as raw counts |
| Behavior bars (pension, study fund, crypto, ...) | Financial Behavior | Yes-share per behavior |
| **Financial word cloud** | Financial Behavior | Finance concepts weighted by data counts |
| Retirement satisfaction and month-end slack | Financial Behavior | Household finance snapshot |
| Satisfaction by age (line), coverage by income (bar), optimism (donut) | Retirement Readiness | Segment cross-cuts |
| **Age x Income scatter (bubble matrix)** | Retirement Readiness | Joint distribution as bubbles |
| **Gauge chart (financial knowledge score)** | Your Profile | Score display with color-coded bands |
| You vs the population (horizontal bar) | Your Profile | User's answers benchmarked against 6,907 respondents |
| Knowledge questions breakdown | Your Profile | Answer-by-answer table |
| Model placeholder cards | Model (Coming Soon) | Where the risk model + SHAP output will land |

## How to run locally

The survey data file is not committed to this repo. Place it locally at:

```
data/Social_Survey/data_24.xlsx
```

Then run:

```
pip install -r requirements.txt
jupyter notebook "Mid Project Dashboard.ipynb"
# Run > Run All Cells
```

The notebook writes `index.html` next to itself and opens it in your browser.
The first run reads the Excel file (about a minute) and writes a small parquet
cache for fast subsequent runs.

## Project structure

```
Mid Project Dashboard.ipynb   Build pipeline: reads the survey, builds the figures,
                              generates index.html
index.html                    Standalone dashboard page (published to GitHub Pages)
requirements.txt              Python dependencies
README.md                     This file
```

## Pipeline stages

1. Data collection and cleaning - upstream stage. Produces the cleaned dataset.
2. Feature engineering and model training - upstream stage. Produces the
   XGBoost artifact and its SHAP explanations.
3. **Deployment and dashboard (this repo).** Consumes the raw survey today, and
   will consume the trained model outputs in the next iteration to fill the
   Model tab.

## Data source

Israeli Central Bureau of Statistics, Social Survey 2024, Public Use File.


