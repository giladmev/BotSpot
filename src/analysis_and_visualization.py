import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os
from config import PROCESSED_DATA_DIR
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_FEATURES_TO_COMPARE = [
    'degree_centrality', 'in_degree_centrality', 'out_degree_centrality',
    'closeness_centrality', 'eigenvector_centrality',
    'pagerank', 'coreness',  # Graph features
    'followers_count', 'following_count', 'tweet_count', 'listed_count',
    'verified', 'protected', 'description_length', 'has_description',
    'has_url_in_profile', 'has_location', 'account_age_days',
    'followers_ratio', 'following_ratio', 'tweets_per_day'  # User profile features
]

_KEY_FEATURES_FOR_HIST = ['followers_count', 'pagerank', 'tweet_count', 'account_age_days']
_PAIR_PLOT_FEATURES = ['pagerank', 'followers_count', 'account_age_days', 'tweet_count']
_MAX_PAIRPLOT_SAMPLES = 100000


def _print_descriptive_statistics(merged_df):
    des_merged_df = merged_df.drop(columns=['label'], errors='ignore').describe().transpose()
    print("\n--- Descriptive Statistics for All Features ---")
    pd.set_option('display.float_format', lambda x: f'{x:.7g}')  # Use general format with up to 6 significant digits
    print(des_merged_df.to_string())
    stats_path = os.path.join(PROCESSED_DATA_DIR, 'descriptive_statistics.csv')
    des_merged_df.to_csv(stats_path, float_format='%.7g')  # Save with same format for CSV


    bot_features = merged_df[merged_df['label_name'] == 'Bot']
    human_features = merged_df[merged_df['label_name'] == 'Human']

    print("\n--- Descriptive Statistics for Bots ---")
    des_bots_df = bot_features.drop(columns=['label', 'label_name'], errors='ignore').describe().transpose()
    pd.set_option('display.float_format', lambda x: f'{x:.7g}')  # Use general format with up to 6 significant digits
    print(des_bots_df.to_string())
    bots_stats_path = os.path.join(PROCESSED_DATA_DIR, 'bots_descriptive_statistics.csv')
    des_bots_df.to_csv(bots_stats_path, float_format='%.7g')  # Same format for CSV

    print("\n--- Descriptive Statistics for Humans ---")
    des_humans_df = human_features.drop(columns=['label', 'label_name'], errors='ignore').describe().transpose()
    print(des_humans_df.to_string())
    humans_stats_path = os.path.join(PROCESSED_DATA_DIR, 'humans_descriptive_statistics.csv')
    des_humans_df.to_csv(humans_stats_path, float_format='%.7g')
    pd.reset_option('display.float_format')  # Reset to default for other operations
    return des_bots_df, des_humans_df


def _perform_mann_whitney_test(merged_df):
    print("\n--- Statistical Comparison (Mann-Whitney U Test) between Bots and Humans ---")
    bot_features = merged_df[merged_df['label_name'] == 'Bot']
    human_features = merged_df[merged_df['label_name'] == 'Human']
    results = []

    for feature in _FEATURES_TO_COMPARE:
        if feature in merged_df.columns:
            bot_data = pd.to_numeric(bot_features[feature].dropna(), errors='coerce').dropna()
            human_data = pd.to_numeric(human_features[feature].dropna(), errors='coerce').dropna()

            if len(bot_data) > 1 and len(human_data) > 1:
                try:
                    stat, p_value = stats.mannwhitneyu(bot_data, human_data, alternative='two-sided')
                    results.append({
                        'Feature': feature,
                        'Bot_Median': bot_data.median(),
                        'Human_Median': human_data.median(),
                        'U_Statistic': stat,
                        'P_Value': f"{p_value:.3e}",
                        'Significance (p<0.05)': 'Yes' if p_value < 0.05 else 'No'
                    })
                except ValueError as e:
                    results.append({
                        'Feature': feature,
                        'Bot_Median': np.nan,
                        'Human_Median': np.nan,
                        'U_Statistic': np.nan,
                        'P_Value': 'Error in test',
                        'Significance (p<0.05)': 'N/A'
                    })
            else:
                results.append({
                    'Feature': feature,
                    'Bot_Median': np.nan,
                    'Human_Median': np.nan,
                    'U_Statistic': np.nan,
                    'P_Value': 'Not enough data',
                    'Significance (p<0.05)': 'N/A'
                })
        else:
            results.append({
                'Feature': feature,
                'Bot_Median': 'N/A',
                'Human_Median': 'N/A',
                'U_Statistic': 'N/A',
                'P_Value': 'N/A',
                'Significance (p<0.05)': 'Feature not found'
            })

    results_df = pd.DataFrame(results)
    print(results_df.to_string())
    return results_df


def _save_statistical_results(results_df):
    output_path = os.path.join(PROCESSED_DATA_DIR, 'statistical_comparison_results.csv')
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\nStatistical comparison results saved to: {output_path}")


def _generate_and_save_box_plots(merged_df, plots_dir):
    num_features = len([f for f in _FEATURES_TO_COMPARE if f in merged_df.columns])
    if num_features > 0:
        n_cols = 4
        n_rows = (num_features + n_cols - 1) // n_cols
        plt.figure(figsize=(n_cols * 4, n_rows * 3))

        plot_idx = 1
        for feature in _FEATURES_TO_COMPARE:
            if feature in merged_df.columns and pd.api.types.is_numeric_dtype(merged_df[feature]):
                plt.subplot(n_rows, n_cols, plot_idx)
                sns.boxplot(x='label_name', y=feature, data=merged_df, hue='label_name', palette={'Human': 'skyblue', 'Bot': 'salmon'}, legend=False)
                plt.title(f'{feature} by User Type (Box Plot)')
                plt.ylabel(feature)
                plt.xlabel('User Type')
                if merged_df[feature].max() > 1000 and merged_df[feature].min() >= 0:
                    plt.yscale('log')
                plot_idx += 1
        plt.tight_layout()
        box_plot_path = os.path.join(plots_dir, 'all_features_boxplot.png')
        plt.savefig(box_plot_path)
        plt.close()
        print(f"Saved box plots to: {box_plot_path}")
    else:
        print("No valid features to generate box plots.")


def _generate_and_save_violin_plots(merged_df, plots_dir):
    num_features = len([f for f in _FEATURES_TO_COMPARE if f in merged_df.columns])
    if num_features > 0:
        n_cols = 4
        n_rows = (num_features + n_cols - 1) // n_cols
        plt.figure(figsize=(n_cols * 4, n_rows * 3))
        plot_idx = 1
        for feature in _FEATURES_TO_COMPARE:
            if feature in merged_df.columns and pd.api.types.is_numeric_dtype(merged_df[feature]):
                plt.subplot(n_rows, n_cols, plot_idx)
                sns.violinplot(x='label_name', y=feature, data=merged_df, hue='label_name', palette={'Human': 'skyblue', 'Bot': 'salmon'}, legend=False)
                plt.title(f'{feature} by User Type (Violin Plot)')
                plt.ylabel(feature)
                plt.xlabel('User Type')
                if merged_df[feature].max() > 1000 and merged_df[feature].min() >= 0:
                    plt.yscale('log')
                plot_idx += 1
        plt.tight_layout()
        violin_plot_path = os.path.join(plots_dir, 'all_features_violinplot.png')
        plt.savefig(violin_plot_path)
        plt.close()
        print(f"Saved violin plots to: {violin_plot_path}")
    else:
        print("No valid features to generate violin plots.")


def _generate_and_save_histograms(merged_df, plots_dir):
    valid_key_hist_features = [f for f in _KEY_FEATURES_FOR_HIST if
                               f in merged_df.columns and pd.api.types.is_numeric_dtype(merged_df[f])]
    if len(valid_key_hist_features) > 0:
        n_cols_hist = 2
        n_rows_hist = (len(valid_key_hist_features) + n_cols_hist - 1) // n_cols_hist
        plt.figure(figsize=(n_cols_hist * 8, n_rows_hist * 6))
        hist_idx = 1
        for feature in valid_key_hist_features:
            plt.subplot(n_rows_hist, n_cols_hist, hist_idx)
            sns.histplot(data=merged_df, x=feature, hue='label_name', kde=True,
                         palette={'Human': 'skyblue', 'Bot': 'salmon'}, common_norm=False)
            plt.title(f'Distribution of {feature}')
            plt.xlabel(feature)
            plt.ylabel('Count')
            if merged_df[feature].max() > 0:
                plt.yscale('log')
            hist_idx += 1
        plt.tight_layout()
        hist_plot_path = os.path.join(plots_dir, 'key_features_histograms.png')
        plt.savefig(hist_plot_path)
        plt.close()
        print(f"Saved histograms to: {hist_plot_path}")
    else:
        print("No valid key features available for histogram visualization.")


def _generate_and_save_correlation_heatmap(merged_df, plots_dir):
    numeric_df_for_corr = merged_df[_FEATURES_TO_COMPARE + ['label']].select_dtypes(include=np.number).copy()

    if not numeric_df_for_corr.empty and len(numeric_df_for_corr.columns) > 1:
        plt.figure(figsize=(12, 10))
        corr_matrix = numeric_df_for_corr.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title('Correlation Heatmap of Features and Label')
        correlation_heatmap_path = os.path.join(plots_dir, 'correlation_heatmap.png')
        plt.savefig(correlation_heatmap_path)
        plt.close()
        print(f"Saved correlation heatmap to: {correlation_heatmap_path}")
    else:
        print("Not enough numeric features to generate correlation heatmap.")


def _generate_and_save_pair_plots(merged_df, plots_dir):
    valid_pair_plot_features = [f for f in _PAIR_PLOT_FEATURES if
                                f in merged_df.columns and pd.api.types.is_numeric_dtype(merged_df[f])]

    if len(valid_pair_plot_features) >= 2:
        # Subsampling for large datasets
        if len(merged_df) > _MAX_PAIRPLOT_SAMPLES:
            print(
                f"Dataset too large for full pair plot ({len(merged_df)} samples). Subsampling to {_MAX_PAIRPLOT_SAMPLES} samples.")
            sampled_df = merged_df.sample(n=_MAX_PAIRPLOT_SAMPLES, random_state=42)
        else:
            sampled_df = merged_df

        print(f"Generating pair plots for: {valid_pair_plot_features}")
        g = sns.pairplot(sampled_df, vars=valid_pair_plot_features, hue='label_name',
                         palette={'Human': 'skyblue', 'Bot': 'salmon'})

        g.fig.suptitle('Pair Plots of Key Features by User Type', y=1.02)
        pair_plot_path = os.path.join(plots_dir, 'key_features_pairplot.png')
        plt.savefig(pair_plot_path)
        plt.close()
        print(f"Saved pair plots to: {pair_plot_path}")
    else:
        print("Not enough valid features selected for pair plot visualization.")

def _generate_bok_plots_specific_features(des_bots_df, des_humans_df, plots_dir):
    features = ["followers_count", "following_count", "tweet_count", "closeness_centrality", "eigenvector_centrality", "description_length"]
    features_data = [{"feature": feat,
                      "bot_median": des_bots_df.loc[feat, '50%'],
                      "bot_q25": des_bots_df.loc[feat, '25%'],
                      "bot_q75": des_bots_df.loc[feat, '75%'],
                      "human_median": des_humans_df.loc[feat, '50%'],
                      "human_q25": des_humans_df.loc[feat, '25%'],
                      "human_q75": des_humans_df.loc[feat, '75%'],
                      "log_scale": feat in ["followers_count", "following_count", "tweet_count"]
                      } for feat in features]
    colors = ['#DB4545', '#1FB8CD']  # bots: red, humans: cyan
    labels = ['Bot', 'Human']
    y_labels = [
        "Foll. Cnt",
        "Follwg. Cnt",
        "Tweet Cnt",
        "Close. Centrlty",
        "Eigenvec. Centrlty",
        "Desc. Len"
    ]

    fig = make_subplots(rows=2, cols=3, subplot_titles=y_labels)

    for i, feat in enumerate(features_data):
        row = i // 3 + 1
        col = i % 3 + 1
        y_bot = [feat['bot_q25'], feat['bot_median'], feat['bot_q75']]
        y_human = [feat['human_q25'], feat['human_median'], feat['human_q75']]
        if feat['log_scale']:
            y_bot = list(np.log10([max(1, feat['bot_q25']), max(1, feat['bot_median']), max(1, feat['bot_q75'])]))
            y_human = list(
                np.log10([max(1, feat['human_q25']), max(1, feat['human_median']), max(1, feat['human_q75'])]))
        fig.add_trace(go.Box(
            y=y_bot,
            name='Bot',
            marker_color=colors[0],
            line_width=2,
            boxpoints=False,
            showlegend=(i == 0),
            whiskerwidth=0.7,
            width=0.5
        ), row=row, col=col)
        fig.add_trace(go.Box(
            y=y_human,
            name='Human',
            marker_color=colors[1],
            line_width=2,
            boxpoints=False,
            showlegend=(i == 0),
            whiskerwidth=0.7,
            width=0.5
        ), row=row, col=col)
        fig.update_yaxes(title_text="", row=row, col=col, tickfont=dict(size=11), title_font=dict(size=13))
        fig.update_xaxes(showticklabels=False, row=row, col=col)
        if feat['log_scale']:
            fig.update_yaxes(row=row, col=col, tickvals=[0, 1, 2, 3, 4, 5],
                             ticktext=['1', '10', '100', '1k', '10k', '100k'])

    fig.update_layout(
        title_text="Bot vs Human Feature Comparison",
        legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5)
    )

    fig.write_image(os.path.join(plots_dir, 'bot_human_feature_cmp.png'))


def perform_statistical_analysis(merged_df):
    print("\n--- Starting Statistical Analysis and Visualization ---")

    # Ensure plots directory exists
    plots_dir = os.path.join(PROCESSED_DATA_DIR, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    print(f"Saving plots to: {plots_dir}")

    des_bots_df, des_humans_df = _print_descriptive_statistics(merged_df)
    results_df = _perform_mann_whitney_test(merged_df)
    _save_statistical_results(results_df)

    print("\n--- Generating Visualizations ---")
    sns.set_style("whitegrid")  # Set style once

    _generate_and_save_box_plots(merged_df, plots_dir)
    _generate_and_save_violin_plots(merged_df, plots_dir)
    _generate_and_save_histograms(merged_df, plots_dir)
    _generate_and_save_correlation_heatmap(merged_df, plots_dir)
    _generate_and_save_pair_plots(merged_df, plots_dir)
    _generate_bok_plots_specific_features(des_bots_df, des_humans_df, plots_dir)

    print("\nStatistical analysis and visualization complete.")