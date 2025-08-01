import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.data_loader import load_full_twibot_data
from src.feature_extractor import extract_features
from src.analysis_and_visualization import perform_statistical_analysis


def main():
    """
    Main function to orchestrate data loading, feature extraction, model training, and evaluation.
    This version loads and processes the FULL dataset, including user profile features,
    and efficiently handles user data by filtering relevant columns early.
    """
    print("Starting bot detection project with FULL dataset and enriched user features...")

    # --- Step 1: Load Full Data ---
    # load_full_twibot_data now returns graph, labels_df, AND a user_df with only relevant columns
    graph, labels_df, user_df = load_full_twibot_data(use_cached=True)

    if graph is None or labels_df is None or user_df is None:
        print("Failed to load graph, labels, or user info. Exiting.")
        return

    print(f"\nLoaded graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
    print(f"Labels DataFrame shape: {labels_df.shape}")
    print(f"User Info DataFrame shape (filtered columns): {user_df.shape}")
    print(f"User Info columns: {user_df.columns.tolist()}")  # Verify columns are indeed filtered

    # --- Step 2: Extract Features ---
    # Pass the user_df to extract_features
    feature_set_name = "full_dataset_graph_and_user_features_enriched"  # Updated name for cache
    node_features_df = extract_features(graph, user_df, feature_set_name=feature_set_name, use_cached=True)

    print("\n--- Extracted Features Preview ---")
    print(node_features_df.head())
    print(node_features_df.describe().to_string())

    merged_df = node_features_df.merge(labels_df[['label']], left_index=True, right_index=True, how='inner')
    merged_df['label_name'] = merged_df['label'].map({0: 'Human', 1: 'Bot'})
    perform_statistical_analysis(merged_df)

    print("\n--- Data Loading, Feature Extraction, and Analysis Complete ---")


if __name__ == "__main__":
    main()