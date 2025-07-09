import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import yaml

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'

COLORS = ["#0173B2", "#DE8F05", "#029E73", "#D55E00"]
sns.set_palette(COLORS)

def load_data(experiment_spec="ml-tmdb/alg_specific"):
    repo_offset = Path("../")
    config_file = Path(f"config/{experiment_spec}.yaml")
    results_dir = Path(f"results/{experiment_spec}/")
    
    os.makedirs(results_dir, exist_ok=True)
    
    with open(config_file, "r") as f:
        config_data = yaml.safe_load(f)
    
    provider_file = Path(f"{config_data['output']['directory']}/{config_data['output']['provider_file']}.csv")
    consumer_file = Path(f"{config_data['output']['directory']}/{config_data['output']['consumer_file']}.csv")
    provider_list_file = Path(f"{config_data['data']['directory']}/{config_data['data']['provider_file']}")
    
    provider_df = pd.read_csv(provider_file, 
                              dtype={'provider_id': int, 'provider_type': str, 
                                     'recommender': str, 'utility': float, 'time': int})
    
    provider_list_df = pd.read_csv(provider_list_file, 
                                   dtype={'provider_id': int, 'provider_type': str})
    
    consumer_df = pd.read_csv(consumer_file, 
                              dtype={'consumer_id': int, 'consumer_type': str, 'recommender': str, 
                                     'utility': float, 'agg_utility': float, 'time': int})
    
    all_days = provider_df['time'].unique()
    provider_list_df_temp = provider_list_df.copy()
    provider_list_df_temp['key'] = 1
    days_df = pd.DataFrame({'time': all_days, 'key': 1})
    complete_df = provider_list_df_temp.merge(days_df, on='key').drop('key', axis=1)
    
    groups = provider_df.groupby(['time', 'provider_id'])
    daily_total = groups['utility'].sum()
    provider_daily_totals = pd.DataFrame({'utility': daily_total}).reset_index()
    
    complete_provider_df = complete_df.merge(provider_daily_totals, 
                                            on=['provider_id', 'time'], 
                                            how='left')
    complete_provider_df['utility'] = complete_provider_df['utility'].fillna(0)
    
    return consumer_df, complete_provider_df, provider_list_df, config_data, results_dir

def plot_last_cycle_comparison(consumer_df, provider_df, config_data, results_dir):
    max_day = provider_df['time'].max()
    cycle_start = max_day - config_data.get('simulation', {}).get('num_days', 30)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    last_consumer_df = consumer_df[consumer_df['time'] >= cycle_start]
    consumer_sum = last_consumer_df.groupby('consumer_type')['utility'].sum().reset_index()
    sns.barplot(data=consumer_sum, x='consumer_type', y='utility', ax=ax1)
    ax1.set_xlabel('Consumer Type')
    ax1.set_ylabel('Total Utility')
    ax1.set_title('Consumer Utility (Last Cycle)')
    
    provider_id_df = provider_df.copy()
    provider_id_df = provider_id_df.rename(columns={'utility': 'per_provider_utility'})
    last_provider_df = provider_id_df[provider_id_df['time'] >= cycle_start]
    provider_sum = last_provider_df.groupby('provider_type')['per_provider_utility'].sum().reset_index()
    sns.barplot(data=provider_sum, x='provider_type', y='per_provider_utility', ax=ax2)
    ax2.set_xlabel('Provider Type')
    ax2.set_ylabel('Total Utility')
    ax2.set_title('Provider Utility (Last Cycle)')
    
    plt.tight_layout()
    plt.savefig(results_dir / 'last_cycle_comparison.pdf')
    plt.savefig(results_dir / 'last_cycle_comparison.png')
    
    return fig

def main():
    consumer_df, complete_provider_df, provider_list_df, config_data, results_dir = load_data()
    plot_last_cycle_comparison(consumer_df, complete_provider_df, config_data, results_dir)

if __name__ == "__main__":
    main()