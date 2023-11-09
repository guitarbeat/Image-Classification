# visualization.py

### Class Distributions
import matplotlib.pyplot as plt

def add_annotations(ax, bars, sub_df):
    """
    Adds annotations to the bars.
    """
    for bar, (_, row) in zip(bars, sub_df.iterrows()):
        x = bar.get_x() + bar.get_width() / 2.0
        y = bar.get_height() + 0.5  # Shift annotation slightly above the bar for clarity
        ax.annotate(f"C: {int(row['Count'])}\nW: {row['Weight']:.2f}", 
                    (x, y), 
                    ha='center', 
                    va='bottom', 
                    fontsize=8)

def plot_single_split(ax, df, split):
    """
    Plots the class distribution for a single split (train/test/valid).
    """
    filtered_df = df.loc[split]
    x_ticks = []
    x_tick_locs = []
    current_x = 0  # Keep track of the current x-location for ticks
    
    labels = filtered_df.index.get_level_values('label').unique()
    for label in labels:
        sub_df = filtered_df.loc[label]
        bars = ax.bar(sub_df.index, sub_df['Count'], label=f"{label}")
        add_annotations(ax, bars, sub_df)
        
        x_ticks.extend([f"{label}_{cls}" for cls in sub_df.index])
        x_tick_locs.extend([current_x + i for i in range(len(sub_df.index))])
        current_x += len(sub_df.index)  # Update the x-location for the next set of bars
    
    ax.legend()
    ax.set_xticks(x_tick_locs)  # Set tick locations
    ax.set_xticklabels(x_ticks, rotation=90, fontsize=8)  # Set tick labels
    ax.set_title(f"{split.capitalize()} Data")
    ax.set_ylabel("Count")  # Indicate that the bars represent counts


def plot_dataset_info(df):
    """
    Plots the class distribution for train, valid, and test splits.
    """
    splits = ['train', 'valid', 'test']
    fig, axs = plt.subplots(1, len(splits), figsize=(20, 8))
    
    for i, split in enumerate(splits):
        plot_single_split(axs[i], df, split)
        
    plt.tight_layout()
    plt.show()

# Example usage
plot_dataset_info(df_class_weights)
plot_dataset_info(rdf_class_weights)


