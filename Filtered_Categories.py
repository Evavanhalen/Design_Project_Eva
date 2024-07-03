import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module='seaborn._oldcore')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st
import os
from PIL import Image
from sklearn.preprocessing import LabelEncoder

# Define file paths using relative paths
file_path = 'Template_V03 (1).xlsx'
logo_path = 'ProRail logo.png'

# Load the data from the template
df = pd.read_excel(file_path, sheet_name='Template')

# Load and display the ProRail logo
logo = Image.open(logo_path)

# Display the logo in the sidebar
st.sidebar.image(logo, use_column_width=True)

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Load the data from the template
file_path = r"C:\Users\eefie\PycharmProjects\DesignProject\Data\Template_V03 (1).xlsx"
df = pd.read_excel(file_path, sheet_name='Template')

# Add a title and description
st.title('Urban/Suburban/Regional Train Track Types Analysis')
st.markdown("""
    This dashboard allows you to analyze and visualize various features of train tracks categorized into urban, suburban, and regional types.

    **Instructions:**
    1. Use the sidebar to include or exclude specific features in the analysis.
    2. Choose whether to exclude emplacement data.
    3. Select the types of plots you want to display.
    4. The dashboard provides options to display means, distributions, and summaries of numerical and non-numerical features.

    **Note:** The data is filtered based on the selections you make in the sidebar.
""")

# Add option to exclude emplacement data
exclude_emplacements = st.sidebar.checkbox('Exclude Emplacements', value=False)

# Filter out emplacement data if the checkbox is checked
if exclude_emplacements:
    df = df[df['Emplacement'] == 'No']

# Filter columns to relevant features, leave out track section numbers, geocodes, names
relevant_columns = df.loc[:, 'ERTMS in 2031':]

# Separate numerical and non-numerical columns
numerical_cols = relevant_columns.select_dtypes(include=[float, int]).columns
non_numerical_cols = relevant_columns.select_dtypes(exclude=[float, int]).columns

# Define specific columns to include/exclude
specific_columns = numerical_cols.tolist() + non_numerical_cols.tolist()

# Function to create include/exclude checkboxes
def create_include_checkbox(column_name):
    include_key = f"{column_name}_include"
    include = st.sidebar.checkbox(f"Include {column_name}", value=True, key=include_key)
    return include, column_name

# Apply include checkboxes to all relevant columns
include_filters = []
included_columns = []
for col in specific_columns:
    include, col_name = create_include_checkbox(col)
    if include:
        included_columns.append(col_name)

# Filter the dataframe based on selected columns
filtered_df = df[included_columns]

# Top selection for which plots to display
st.markdown("### Visualization Options")
graph_options = st.multiselect(
    'Select the graphs you want to see:',
    [
        'Display Numerical Means by Category',
        'Display Numerical Distributions',
        'Display Non-Numerical Distributions',
        'Display Numerical Summary',
        'Display Non-Numerical Summary Heatmap with Percentages'
    ]
)

# Group by 'Urban/Regional/Suburban' and calculate mean and standard deviation for numerical features and most frequent value for non-numerical features
numerical_cols = filtered_df.select_dtypes(include=[float, int]).columns
non_numerical_cols = filtered_df.select_dtypes(exclude=[float, int]).columns

mean_numerical = filtered_df.groupby('Urban/Regional/Suburban')[numerical_cols].mean()
mode_non_numerical = filtered_df.groupby('Urban/Regional/Suburban')[non_numerical_cols].agg(lambda x: x.mode()[0])
grouped_stds = filtered_df.groupby('Urban/Regional/Suburban')[numerical_cols].std()

# Combine numerical and non-numerical summaries
summary_numerical = mean_numerical
summary_non_numerical = mode_non_numerical
summary_std = grouped_stds

# Define the function to plot all numerical features
def plot_all_numerical_features(mean_values, std_values, categories, output_folder, group_size=6):
    numerical_features = mean_values.columns
    num_features = len(numerical_features)
    cols = 2  # Number of columns for subplots
    rows = (group_size // cols) + (group_size % cols > 0)  # Calculate number of rows needed per group

    for start_idx in range(0, num_features, group_size):
        end_idx = min(start_idx + group_size, num_features)
        fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))
        axes = axes.flatten()

        for i, col in enumerate(numerical_features[start_idx:end_idx]):
            ax = axes[i]
            means = mean_values[col]
            errors = std_values[col]
            ax.errorbar(categories, means, yerr=errors, fmt='o', color='blue', capsize=5, label='Standard Deviation')
            ax.scatter(categories, means, color='red', zorder=5, label=f'{col} (Mean)')
            ax.set_title(f'Means of {col} by Urban/Regional/Suburban Category')
            ax.set_ylabel(f'Mean {col}')
            ax.set_xlabel('Urban/Regional/Suburban Category')
            ax.legend(loc='upper right')
            ax.tick_params(axis='x', rotation=45)

        # Remove any empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout(pad=3.1)
        fig.subplots_adjust(top=0.9)
        fig.suptitle(f'Means of Numerical Features from {start_idx + 1} to {end_idx} by Urban/Regional/Suburban Category', fontsize=16)
        fig.canvas.draw()  # Ensure the figure is drawn before saving
        plt.savefig(f'{output_folder}/Numerical_Features_{start_idx + 1}_to_{end_idx}.png')
        st.pyplot(fig)
        plt.close(fig)


# Assuming mean_numerical and grouped_stds are DataFrames with the mean and std values of numerical features respectively
if 'Display Numerical Means by Category' in graph_options:
    plot_all_numerical_features(mean_numerical, grouped_stds, mean_numerical.index, output_folder)

## Plotting the distributions
# Define the function to plot distributions
def plot_distributions(columns, df, title, file_name_prefix, cols=2):
    num_plots = len(columns)
    rows = (num_plots // cols) + (num_plots % cols > 0)  # Calculate number of rows needed

    fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))
    axes = axes.flatten()

    for i, col in enumerate(columns):
        sns.boxplot(x='Urban/Regional/Suburban', y=col, data=df, ax=axes[i])
        axes[i].set_title(f'Distribution of {col}', fontsize=10, pad=10)
        axes[i].set_ylabel(col, fontsize=8)
        axes[i].set_xlabel('')
        axes[i].tick_params(axis='x', labelsize=6)

    # Remove any empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(pad=3.1)  # Adjust the padding between subplots
    fig.subplots_adjust(top=0.9)  # Adjust the top spacing to make room for the main title
    fig.suptitle(title, fontsize=16)  # Main title
    plt.savefig(os.path.join(output_folder, f'{file_name_prefix}_{title}.png'))  # Save the figure
    st.pyplot(fig)
    plt.close(fig)

# Split numerical columns into smaller groups for better readability
group_size = 6  # Number of subplots per figure

# Create subfigures for each group
if 'Display Numerical Distributions' in graph_options:
    for start_index in range(0, len(numerical_cols), group_size):
        end_index = min(start_index + group_size, len(numerical_cols))
        group = numerical_cols[start_index:end_index]
        plot_distributions(group, filtered_df, f'Distributions_of_Numerical_Features_{start_index + 1}_to_{end_index}', file_name_prefix=f'Features_{start_index + 1}_to_{end_index}')

    #
    # Handle the remaining columns if the division is not perfect
    if end_index < len(numerical_cols):
        remaining_cols = numerical_cols[end_index:]
        plot_distributions(remaining_cols, filtered_df, 'Distributions_of_Remaining_Numerical_Features', file_name_prefix='Remaining_Features')

# Define the function to plot non-numerical distributions
def plot_non_numerical_distributions(columns, df, title, file_name_prefix, cols=2):
    num_plots = len(columns)
    rows = (num_plots // cols) + (num_plots % cols > 0)  # Calculate number of rows needed

    fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))
    axes = axes.flatten()

    for i, col in enumerate(columns):
        sns.countplot(x='Urban/Regional/Suburban', hue=col, data=df, ax=axes[i])
        axes[i].set_title(f'Distribution of {col}', fontsize=10, pad=10)
        axes[i].set_xlabel('Urban/Regional/Suburban Category', fontsize=8)
        axes[i].tick_params(axis='x', labelsize=6, rotation=45)  # Adjust font size and rotation
        axes[i].legend(title=col, fontsize=6, title_fontsize=8)  # Adjust legend font size

    # Remove any empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(pad=3.1)  # Adjust the padding between subplots
    fig.subplots_adjust(top=0.9)  # Adjust the top spacing to make room for the main title
    fig.suptitle(title, fontsize=16)  # Main title
    plt.savefig(os.path.join(output_folder, f'{file_name_prefix}_{title}.png'))  # Save the figure
    st.pyplot(fig)
    plt.close(fig)

if 'Display Non-Numerical Distributions' in graph_options:
    plot_non_numerical_distributions(non_numerical_cols, filtered_df, 'Non-Numerical Feature Distributions', 'Non_Numerical_Distributions')

# Visualization function for numerical data
def plot_numerical_summary(summary, title):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    categories = ['Urban', 'Regional', 'Suburban']

    for i, category in enumerate(categories):
        summary.loc[category].plot(kind='bar', ax=axes[i])
        axes[i].set_title(f'Mean {category} Train Track Section')
        axes[i].set_ylabel('Mean Value')
        axes[i].set_xlabel('Features')
        axes[i].tick_params(axis='x', labelsize=8, rotation=90)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig(os.path.join(output_folder, 'Numerical_Summary.png'))  # Save the figure
    st.pyplot(fig)
    plt.close(fig)

if 'Display Numerical Summary' in graph_options:
    plot_numerical_summary(summary_numerical, 'Mean Urban/Regional/Suburban Train Track Sections')

# Visualization function for non-numerical data
def plot_non_numerical_summary_heatmap(summary, title):
    # Convert the summary to a dataframe suitable for heatmap
    summary_df = summary.T

    # Apply label encoding to convert categorical values to numerical values
    label_encoders = {}
    for column in summary_df.columns:
        le = LabelEncoder()
        summary_df[column] = le.fit_transform(summary_df[column])
        label_encoders[column] = le

    plt.figure(figsize=(12, 6))
    sns.heatmap(summary_df, annot=True, cmap='coolwarm', cbar=False, fmt="d", linewidths=.5)
    plt.title(title, fontsize=16)
    plt.xlabel('Categories')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'Non_Numerical_Summary_Heatmap.png'))  # Save the figure
    st.pyplot(plt)
    plt.close()

#if display_non_numerical_summary_heatmap:
   # plot_non_numerical_summary_heatmap(summary_non_numerical, 'Most Frequent Urban/Regional/Suburban Train Track Sections')

# Define the function to plot non-numerical summary heatmap with percentages
def plot_non_numerical_summary_heatmap_with_percentages(df, title):
    summary_percentages = pd.DataFrame()

    # Create a mapping of feature to category
    feature_category_mapping = {
        'ERTMS in 2031': ['Yes', 'No'],
        'Number of tracks': ['single', 'double', 'three'],
        'Type of track': ['primary', 'secondary', 'tertairy'],
        'Travelers per day': ['0-1000', '1000-25000', '5000-10000'],
        'Urban/Regional/Suburban': ['Urban', 'Regional', 'Suburban'],
        'Safety System': ['ATB NG', 'ATB VV'],
        'Detection system': ['Axle counters', 'Circuit'],
    }

    for feature in non_numerical_cols:
        feature_counts = df.groupby(['Urban/Regional/Suburban', feature]).size().unstack(fill_value=0)
        feature_percentages = feature_counts.div(feature_counts.sum(axis=1), axis=0) * 100
        summary_percentages = pd.concat([summary_percentages, feature_percentages])

    # Create new column names with category
    new_column_names = []
    for col in summary_percentages.columns:
        for feature, categories in feature_category_mapping.items():
            if col in categories:
                new_column_names.append(f'{feature}-{col}')
                break
        else:
            new_column_names.append(col)

    summary_percentages.columns = new_column_names

    # Create a heatmap with percentages
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.heatmap(summary_percentages, annot=True, cmap='coolwarm', cbar=False, fmt=".1f", linewidths=.5, ax=ax)
    ax.set_title(title, fontsize=16)
    ax.set_ylabel('Features')
    ax.set_xlabel('Categories')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'Non_Numerical_Summary_Heatmap_Percentages.png'))  # Save the figure
    st.pyplot(fig)
    plt.close()

if 'Display Display Non-Numerical Summary Heatmap with Percentages' in graph_options:
    plot_non_numerical_summary_heatmap_with_percentages(filtered_df, 'Most Frequent Urban/Regional/Suburban Train Track Sections with Percentages')

# Save the summary table to an Excel file
output_file = os.path.join(output_folder, 'Categories_Summary.xlsx')
with pd.ExcelWriter(output_file) as writer:
    summary_numerical.to_excel(writer, sheet_name='Numerical Features')
    summary_std.to_excel(writer, sheet_name='Standard Deviation')
    summary_non_numerical.to_excel(writer, sheet_name='Non-Numerical Features')

st.write("Summarized data saved to Excel file")

# Provide download link for the Excel file
with open(output_file, "rb") as file:
    btn = st.download_button(
        label="Download Summary Excel",
        data=file,
        file_name="Categories_Summary.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
