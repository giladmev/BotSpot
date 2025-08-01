import pandas as pd
import networkx as nx
import json
import os
import joblib
from src.config import TWIBOT_GLOBAL_DATA_PATHS, PROCESSED_DATA_DIR



def load_labels(filepath):
    print(f"Loading labels from {filepath}...")
    try:
        labels_df = pd.read_csv(filepath)
        labels_df.rename(columns={'id': 'user_id'}, inplace=True)
        labels_df['label'] = labels_df['label'].map({'human': 0, 'bot': 1})
        print(f"Loaded {len(labels_df)} labels.")
        return labels_df.set_index('user_id')
    except FileNotFoundError:
        print(f"Error: Label file not found at {filepath}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading labels: {e}")
        return pd.DataFrame()

def load_edges(filepath):
    print(f"Loading edges from {filepath}...")
    try:
        edges_df = pd.read_csv(filepath)
        print(f"Loaded {len(edges_df)} edges.")
        return edges_df
    except FileNotFoundError:
        print(f"Error: Edge file not found at {filepath}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading edges: {e}")
        return pd.DataFrame()


def load_user_info(filepath):
    print(f"Loading user info from {filepath}...")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            full_content = f.read()
        raw_user_data = json.loads(full_content)

        user_list = []
        if isinstance(raw_user_data, list):
            for item in raw_user_data:
                if isinstance(item, dict) and 'id' in item:
                    user_list.append(item)
                else:
                    print(f"Warning: Skipping malformed user entry in JSON (missing 'id' or not a dict): {item}")
        elif isinstance(raw_user_data, dict):
            found_list = False
            for key, value in raw_user_data.items():
                if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict) and 'id' in value[0]:
                    user_list = [item for item in value if isinstance(item, dict) and 'id' in item]
                    found_list = True
                    print(f"Detected user list under key: '{key}' with {len(user_list)} valid entries.")
                    break
            if not found_list:
                if 'id' in raw_user_data and isinstance(raw_user_data,
                                                        dict):  # If the root dict is a user object itself
                    user_list = [raw_user_data]
                else:
                    print(
                        f"Warning: Could not find a list of user objects or a single user object with 'id' in {filepath}.")
                    return pd.DataFrame()
        else:
            print(
                f"Error: Unexpected top-level JSON structure in {filepath}. Expected list or dict, got {type(raw_user_data)}.")
            return pd.DataFrame()

        if not user_list:
            print(f"No valid user data found after parsing JSON from {filepath}.")
            return pd.DataFrame()

        user_df = pd.DataFrame(user_list)
        print(f"DataFrame created. Initial columns: {user_df.columns.tolist()}")

        if 'id' not in user_df.columns:
            print(
                f"Critical Error: After DataFrame creation, 'id' column is still not found. Available columns: {user_df.columns.tolist()}")
            return pd.DataFrame()

        user_df.rename(columns={'id': 'user_id'}, inplace=True)
        user_df = user_df.set_index('user_id')

        relevant_user_columns_for_features = [
            'created_at', 'description', 'public_metrics',
            'url', 'location', 'verified', 'protected'
        ]
        for col in relevant_user_columns_for_features:
            if col not in user_df.columns:
                user_df[col] = None
        user_df = user_df[relevant_user_columns_for_features]
        user_df['public_metrics'] = user_df['public_metrics'].apply(lambda x: x if isinstance(x, dict) else {})

        print(
            f"Loaded {len(user_df)} user profiles and filtered to relevant columns. Final columns: {user_df.columns.tolist()}")
        return user_df
    except FileNotFoundError:
        print(f"Error: User info file not found at {filepath}")
        return pd.DataFrame()
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {filepath}: {e}. Ensure it's a valid single JSON object (array or dict).")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading user info: {e}")
        return pd.DataFrame()
def create_graph(user_ids, labels_df, edges_df):
    G = nx.DiGraph()

    filtered_edges_df = edges_df[
        (edges_df['source_id'].isin(user_ids)) &
        (edges_df['target_id'].isin(user_ids))
        ]
    print(f"Filtered edges for graph construction. Original: {len(edges_df)}, Used: {len(filtered_edges_df)}")

    labels_dict = labels_df['label'].to_dict()
    for user_id in user_ids:
        if user_id in labels_dict:
            G.add_node(user_id, is_bot=labels_dict[user_id])

    for _, row in filtered_edges_df.iterrows():
        G.add_edge(row['source_id'], row['target_id'], relation_type=row['relation'])

    print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    isolated_nodes = list(nx.isolates(G))
    if isolated_nodes:
        print(f"Warning: {len(isolated_nodes)} isolated nodes found in the graph after filtering edges.")
    return G


def load_full_twibot_data(use_cached = True):
    print("--- Loading Full TwiBot-22 Data ---")

    graph_cache_path = os.path.join(PROCESSED_DATA_DIR, 'full_twibot_graph.joblib')
    labels_cache_path = os.path.join(PROCESSED_DATA_DIR, 'full_twibot_labels.csv')
    user_df_cache_path = os.path.join(PROCESSED_DATA_DIR, 'full_twibot_user_info.joblib')  # Cache for filtered user_df

    if use_cached and os.path.exists(graph_cache_path) and \
            os.path.exists(labels_cache_path) and os.path.exists(user_df_cache_path):
        try:
            graph = joblib.load(graph_cache_path)
            labels_df = pd.read_csv(labels_cache_path, index_col='user_id')
            user_df = joblib.load(user_df_cache_path)  # Load cached, already filtered user_df
            print(
                f"Graph, labels, and user info loaded from cache: {graph_cache_path}, {labels_cache_path}, {user_df_cache_path}")
            return graph, labels_df, user_df
        except Exception as e:
            print(f"Error loading cached data: {e}. Rebuilding from raw data.")

    labels_df = load_labels(TWIBOT_GLOBAL_DATA_PATHS['label_csv'])
    if labels_df.empty:
        return None, None, None

    edges_df = load_edges(TWIBOT_GLOBAL_DATA_PATHS['edge_csv'])
    if edges_df.empty:
        return None, None, None

    user_df = load_user_info(TWIBOT_GLOBAL_DATA_PATHS['user_json'])
    if user_df.empty:
        return None, None, None

    all_user_ids = labels_df.index.tolist()

    graph = create_graph(all_user_ids, labels_df, edges_df)

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    joblib.dump(graph, graph_cache_path)
    labels_df.to_csv(labels_cache_path)
    joblib.dump(user_df, user_df_cache_path)  # Cache the filtered user_df
    print(f"Graph, labels, and user info cached to {graph_cache_path}, {labels_cache_path}, {user_df_cache_path}")

    print("--- Full Data Loading Complete ---")
    return graph, labels_df, user_df
