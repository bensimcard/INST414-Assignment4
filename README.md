# Mapping the Marvel Social Network with K-Means Clustering

## Project Overview
If you were a Content Strategist at Disney+ tasked with greenlighting the next massive Marvel crossover event, how would you decide which characters to group together? 

This project uses network analysis and Machine Learning (K-Means Clustering) to mathematically identify "factions" within the Marvel Universe based purely on character interactions over 40 years of comic book history. Instead of relying on official team names (like the Avengers or X-Men), this script groups characters by analyzing their shared social circles and comic co-occurrences.

**Read the full analysis on Medium:** [Insert Link to Your Medium Post Here]

## Data Source
The dataset used in this project is a unimodal edge list of Marvel character co-occurrences. 
* **Source:** Fetched programmatically from [Dr. Melanie Walsh's Sample Social Network Datasets](https://github.com/melaniewalsh/sample-social-network-datasets).
* **Original Compilation:** The data was originally compiled by the Marvel Chronology Project and formalized for academic research by Alberich, Miro-Julia, and Rosselló (2002).

## Key Features
* **Data Cleaning:** Filters the massive Marvel network down to the Top 50 most connected characters to reduce sparsity and prevent the "hairball" effect.
* **Feature Engineering:** Converts the edge list into a mathematical Adjacency Matrix.
* **Elbow Method Optimization:** Programmatically generates an Elbow Plot to determine the optimal number of clusters ($k$).
* **Network Visualization:** Utilizes `NetworkX` to render a force-directed graph of the Marvel universe, color-coded by the resulting K-Means clusters.

## Repository Contents
* `marvel_clustering.py` (or `.ipynb`): The main Python script/notebook containing the data ingestion, K-Means modeling, and visualization code.
* `elbow_plot.png`: The generated graph used to determine our $k$ value.
* `marvel_network_clusters.png`: The force-directed network visualization of the final clusters.

## How to Run
To run this code locally, ensure you have the following Python libraries installed:
```bash
pip install pandas scikit-learn matplotlib networkx
---
*Note: This README was created with the assistance of Gemini.*
