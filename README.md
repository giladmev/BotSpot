# Bot Detection System in Social Networks

This codebase implements a bot detection system for social networks, focusing on Twitter data (specifically the TwiBot-22 dataset). The system extracts and analyzes network and user features to distinguish between bot and human accounts.
### Data Source Credit
This repository uses data from the dataset at [bunsenfeng.github.io](https://bunsenfeng.github.io/) by Shangbin Feng
## Project Overview

The system works through the following process:

1. **Data Loading**: Loads Twitter user data, network connections (edges), and labels (bot/human).
2. **Feature Extraction**: Extracts two types of features:
   - **Network features**: Centrality measures from the user interaction graph (pagerank, centrality metrics)
   - **User profile features**: Account characteristics (followers, tweets, account age)
3. **Statistical Analysis**: Compares feature distributions between bots and humans
4. **Visualization**: Creates visual comparisons to highlight differences

## Key Components

### Data Loading (`data_loader.py`)
- Reads user profiles, connections, and bot/human labels
- Constructs a directed graph representing user interactions
- Implements caching for performance

### Feature Extraction (`feature_extractor.py`)
- Extracts network centrality metrics (degree centrality, pagerank, etc.)
- Processes user profile features (followers count, tweet count, etc.)
- Calculates derived metrics (followers ratio, tweets per day)
- Implements caching for performance

### Statistical Analysis (`analysis_and_visualization.py`)
- Computes descriptive statistics for bot and human features
- Performs Mann-Whitney U tests to identify statistically significant differences
- Generates visualizations comparing bot vs human characteristics

## Feature Engineering

The system analyzes two main types of features:

### Network Features
- **Degree Centrality**: How connected a user is in the network
- **In/Out Degree Centrality**: Balance of followers vs following
- **Closeness Centrality**: How close a user is to all others in the network
- **Eigenvector Centrality**: Connection to influential users
- **PageRank**: Google's algorithm to determine importance
- **Coreness**: Position in the core-periphery structure

### User Profile Features
- **Activity metrics**: Tweet count, account age, tweets per day
- **Social metrics**: Followers/following count and ratios
- **Profile completeness**: Description length, location, URL
- **Account attributes**: Verification status, protected status

## Data Flow

1. The `main.py` orchestrates the entire process
2. Data is loaded through `load_full_twibot_data()`
3. Features are extracted using `extract_features()`
4. Statistical analysis is performed via `perform_statistical_analysis()`

## Visualizations

The system generates multiple visualizations:
- Box plots comparing feature distributions
- Violin plots showing density distributions
- Histograms of key metrics
- Correlation heatmaps between features
- Pair plots showing relationships between important features

The visualizations help identify distinctive patterns between bot and human accounts, providing insights for the bot detection process.