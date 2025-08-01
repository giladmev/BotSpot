import networkx as nx
import pandas as pd
import hashlib
import joblib
import os
from datetime import datetime
from src.config import PROCESSED_DATA_DIR
from datetime import timezone


def calculate_degree_centrality(graph):
    print("Calculating Degree Centrality...")
    degree_centrality = nx.degree_centrality(graph)
    return pd.Series(degree_centrality, name='degree_centrality')


def calculate_in_degree_centrality(graph):
    print("Calculating In-Degree Centrality...")
    in_degree_centrality = nx.in_degree_centrality(graph)
    return pd.Series(in_degree_centrality, name='in_degree_centrality')


def calculate_out_degree_centrality(graph):
    print("Calculating Out-Degree Centrality...")
    out_degree_centrality = nx.out_degree_centrality(graph)
    return pd.Series(out_degree_centrality, name='out_degree_centrality')


def calculate_closeness_centrality(graph):
    print("Calculating Closeness Centrality...")
    closeness_centrality = nx.closeness_centrality(graph)
    return pd.Series(closeness_centrality, name='closeness_centrality')


def calculate_betweenness_centrality(graph):
    print("Calculating Betweenness Centrality...")
    betweenness_centrality = nx.betweenness_centrality(graph)
    return pd.Series(betweenness_centrality, name='betweenness_centrality')


def calculate_eigenvector_centrality(graph):
    print("Calculating Eigenvector Centrality...")
    try:
        eigenvector_centrality = nx.eigenvector_centrality(graph, max_iter=1000, tol=1e-06)
        return pd.Series(eigenvector_centrality, name='eigenvector_centrality')
    except nx.PowerIterationFailedConvergence:
        print("Warning: Eigenvector centrality did not converge. Returning NaNs.")
        return pd.Series({node: float('nan') for node in graph.nodes()}, name='eigenvector_centrality')
    except Exception as e:
        print(f"Error calculating Eigenvector Centrality: {e}")
        return pd.Series({node: float('nan') for node in graph.nodes()}, name='eigenvector_centrality')


def calculate_pagerank(graph):
    print("Calculating PageRank...")
    pagerank = nx.pagerank(graph)
    return pd.Series(pagerank, name='pagerank')


def calculate_coreness(graph):
    print("Calculating Coreness...")
    coreness = nx.core_number(graph)
    return pd.Series(coreness, name='coreness')


def extract_user_profile_features(user_df):
    print("\n--- Extracting User Profile Features ---")
    if 'user_id' in user_df.columns:
        user_df = user_df.set_index('user_id')

    features = pd.DataFrame(index=user_df.index)

    features['followers_count'] = user_df['public_metrics'].apply(
        lambda x: x.get('followers_count', 0) if isinstance(x, dict) else 0)
    features['following_count'] = user_df['public_metrics'].apply(
        lambda x: x.get('following_count', 0) if isinstance(x, dict) else 0)
    features['tweet_count'] = user_df['public_metrics'].apply(
        lambda x: x.get('tweet_count', 0) if isinstance(x, dict) else 0)
    features['listed_count'] = user_df['public_metrics'].apply(
        lambda x: x.get('listed_count', 0) if isinstance(x, dict) else 0)

    current_time = datetime.now(timezone.utc)  # Use UTC timezone for consistent comparison
    features['account_age_days'] = user_df['created_at'].apply(
        lambda x: (current_time - datetime.strptime(x, '%Y-%m-%d %H:%M:%S%z')).days if pd.notna(x) else 0
    )
    features['account_age_days'] = features['account_age_days'].clip(lower=0)  # Ensure no negative ages

    features['tweets_per_day'] = features['tweet_count'] / (features['account_age_days'] + 1)
    # Add 1 to denominators to avoid division by zero
    features['followers_ratio'] = features['followers_count'] / (features['following_count'] + 1)
    features['following_ratio'] = features['following_count'] / (features['followers_count'] + 1)
    features['tweets_per_day'] = features['tweet_count'] / (
                features['account_age_days'] + 1)

    features['verified'] = user_df['verified'].astype(int)
    features['protected'] = user_df['protected'].astype(int)

    features['description_length'] = user_df['description'].fillna('').apply(len)
    features['has_description'] = user_df['description'].notna().astype(int)
    features['has_url_in_profile'] = user_df['url'].notna().astype(int)
    features['has_location'] = user_df['location'].notna().astype(int)

    features.replace([float('inf'), -float('inf')], pd.NA, inplace=True)
    features = features.fillna(0)

    print("User Profile Feature Extraction Complete.")
    print(f"User Features DataFrame shape: {features.shape}")
    return features


def extract_features(graph, user_df, feature_set_name = "basic", use_cached = True):
    print(f"\n--- Extracting Graph and User Features ({feature_set_name} set) ---")

    graph_hash = str(nx.weisfeiler_lehman_graph_hash(graph, iterations=3, digest_size=16))

    user_df_for_hashing = user_df.copy()

    for col in user_df_for_hashing.columns:
        if user_df_for_hashing[col].apply(lambda x: isinstance(x, dict)).any():
            user_df_for_hashing[col] = user_df_for_hashing[col].apply(
                lambda x: str(x) if isinstance(x, dict) else x
            )

    user_data_hash = hashlib.md5(
        pd.util.hash_pandas_object(user_df_for_hashing, index=True).sum().astype(str).encode('utf-8')
    ).hexdigest()

    combined_hash = f"{graph_hash}_{user_data_hash}"

    features_cache_filename = f"features_{feature_set_name}_{combined_hash}.joblib"
    features_cache_path = os.path.join(PROCESSED_DATA_DIR, features_cache_filename)

    if use_cached and os.path.exists(features_cache_path):
        try:
            node_features_df = joblib.load(features_cache_path)
            if set(node_features_df.index) == set(graph.nodes()):
                print(f"Features loaded from cache: {features_cache_path}")
                print(f"Features DataFrame shape: {node_features_df.shape}")
                return node_features_df
            else:
                print("Cached features do not match current graph nodes. Recalculating.")
        except Exception as e:
            print(f"Error loading cached features: {e}. Recalculating.")
    else:
        print("Cached features not found or use_cached is False. Calculating features...")

    graph_features_df = pd.DataFrame(index=list(graph.nodes()))
    graph_features_df['degree_centrality'] = calculate_degree_centrality(graph)
    graph_features_df['in_degree_centrality'] = calculate_in_degree_centrality(graph)
    graph_features_df['out_degree_centrality'] = calculate_out_degree_centrality(graph)
    graph_features_df['closeness_centrality'] = calculate_closeness_centrality(graph)
    joblib.dump(graph_features_df, features_cache_path+'_closeness_centrality')
    # graph_features_df['betweenness_centrality'] = calculate_betweenness_centrality(graph)
    graph_features_df['eigenvector_centrality'] = calculate_eigenvector_centrality(graph)
    joblib.dump(graph_features_df, features_cache_path+'_eigenvector_centrality')
    graph_features_df['pagerank'] = calculate_pagerank(graph)
    joblib.dump(graph_features_df, features_cache_path+'_pagerank')
    graph_features_df['coreness'] = calculate_coreness(graph)
    joblib.dump(graph_features_df, features_cache_path+'_coreness')

    filtered_user_df_for_features = user_df[user_df.index.isin(graph.nodes())]
    user_profile_features_df = extract_user_profile_features(filtered_user_df_for_features)
    joblib.dump(user_profile_features_df, features_cache_path+'_user_profile')

    node_features_df = graph_features_df.merge(user_profile_features_df,
                                               left_index=True,
                                               right_index=True,
                                               how='left')

    node_features_df = node_features_df.fillna(0)

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    joblib.dump(node_features_df, features_cache_path)
    print(f"Features cached to {features_cache_path}")

    print("--- Feature Extraction Complete ---")
    print(f"Features DataFrame shape: {node_features_df.shape}")
    return node_features_df
