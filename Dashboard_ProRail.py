## This is the overall dashboard which combines the Filtered_Categories.py and Interface_Filtering.py
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from PIL import Image
from io import BytesIO
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning, module='seaborn._oldcore')

# Define file paths using relative paths
file_path = 'Template_V03 (6).xlsx'
logo_path = 'ProRail logo.png'

# Load the data from the template
df = pd.read_excel(file_path, sheet_name='Template')

# Load and display the ProRail logo
logo = Image.open(logo_path)

# Display the logo in the sidebar
st.sidebar.image(logo, use_column_width=True)

# Sidebar filters
st.sidebar.title('Filter Options')
ertms_filter = st.sidebar.multiselect('ERTMS in 2031', df['ERTMS in 2031'].unique(), default=df['ERTMS in 2031'].unique())
ertms_tranche1_filter = st.sidebar.multiselect('Tranche 1 ERTMS', df['Tranche 1 ERTMS'].unique(), default=df['Tranche 1 ERTMS'].unique())
track_type_filter = st.sidebar.multiselect('Type of track', df['Type of track'].unique(), default=df['Type of track'].unique())
travelers_filter = st.sidebar.multiselect('Travelers per day', df['Travelers per day'].unique(), default=df['Travelers per day'].unique())
urban_filter = st.sidebar.multiselect('Urban/Regional/Suburban', df['Urban/Regional/Suburban'].unique(), default=df['Urban/Regional/Suburban'].unique())
safety_filter = st.sidebar.multiselect('Safety System', df['Safety System'].unique(), default=df['Safety System'].unique())
detection_filter = st.sidebar.multiselect('Detection system', df['Detection system'].unique(), default=df['Detection system'].unique())
emplacement_filter = st.sidebar.multiselect('Emplacement', df['Emplacement'].unique(), default=df['Emplacement'].unique())
number_of_tracks_filter = st.sidebar.multiselect('Number of tracks', df['Number of tracks'].unique(), default=df['Number of tracks'].unique())
geocode_exact = st.sidebar.text_input('Geocode', '')

# Numeric filters
def create_numeric_filter(column_name, multiplier=1):
    min_val = df[column_name].min() * multiplier
    max_val = df[column_name].max() * multiplier
    if min_val == max_val:
        return st.sidebar.text_input(f'Exact {column_name}', value=str(min_val / multiplier)), None
    else:
        return st.sidebar.text_input(f'Exact {column_name}', ''), st.sidebar.slider(f'{column_name}', min_val, max_val, (min_val, max_val))

# Specific columns for sliders
slider_columns = [ 'Track length (km)', 'Peat', 'Sand', 'Loamy sand', 'Sandy clay loam', 'Light clay', 'Heavy clay',
    'Loam', 'Sand combination', 'Clay combination', 'Urban area',
    'km track', 'ATB beacon', 'Axle counters', 'Balise', 'Board signal', 'Crossing', 'Level Crossing',
    'Light signal', 'Matrix signal', 'Stations', 'Switches', 'Track current sections', 'Railway Viaduct',
    'Viaduct', 'Railway Bridge', 'Traffic Bridge', 'Railway Tunnel', 'Ecoduct',
]

filters = {}
for column in slider_columns:
    exact, slider = create_numeric_filter(column, 1 if column not in ['Peat', 'Sand', 'Loamy sand', 'Sandy clay loam', 'Light clay', 'Heavy clay', 'Sand combination', 'Clay combination', 'Urban area'] else 100)
    filters[column] = (exact, slider)

# Apply filters
filtered_df = df[(df['ERTMS in 2031'].isin(ertms_filter)) &
                 (df['Tranche 1 ERTMS'].isin(ertms_tranche1_filter)) &
                 (df['Type of track'].isin(track_type_filter)) &
                 (df['Travelers per day'].isin(travelers_filter)) &
                 (df['Urban/Regional/Suburban'].isin(urban_filter)) &
                 (df['Safety System'].isin(safety_filter)) &
                 (df['Detection system'].isin(detection_filter)) &
                 (df['Emplacement'].isin(emplacement_filter)) &
                 (df['Number of tracks'].isin(number_of_tracks_filter))]

def apply_numeric_filter(df, column_name, exact_value, range_value, multiplier=1):
    if exact_value:
        df = df[df[column_name] == float(exact_value) / multiplier]
    elif range_value:
        df = df[df[column_name].between(range_value[0] / multiplier, range_value[1] / multiplier)]
    return df

for column, (exact_value, range_value) in filters.items():
    filtered_df = apply_numeric_filter(filtered_df, column, exact_value, range_value, 1 if column not in ['Peat', 'Sand', 'Loamy sand', 'Sandy clay loam', 'Light clay', 'Heavy clay', 'Sand combination', 'Clay combination', 'Urban area'] else 100)

if geocode_exact:
    filtered_df = filtered_df[filtered_df['Geocode'] == geocode_exact]

# Display the results
st.title('Filtered Train Track Sections')
st.markdown("This dashboard allows the user to filter train track sections based on the filter options on the left side of the dashboard."
            "The table shows which track sections match the chosen criteria.")
st.write(f"Number of tracks matching criteria: {filtered_df.shape[0]}")
st.write(filtered_df)

total_tracks_count = df.shape[0]
filtered_tracks_count = filtered_df.shape[0]
percentage_matching_tracks = (filtered_tracks_count / total_tracks_count) * 100
st.write(f"Percentage of tracks matching criteria: {percentage_matching_tracks:.2f}%")

total_km_tracks = df['Track length (km)'].sum()
filtered_km_tracks = filtered_df['Track length (km)'].sum()
percentage_matching_km_tracks = (filtered_km_tracks / total_km_tracks) * 100

st.write(f"Total km of tracks: {total_km_tracks:.2f} km")
st.write(f"Km of tracks matching criteria: {filtered_km_tracks:.2f} km")
st.write(f"Percentage of km tracks matching criteria: {percentage_matching_km_tracks:.2f}%")

# Visualization Options
st.title('Visualization Options')
graph_options = st.multiselect(
    'Select the graphs you want to see:',
    ['Pie Chart (Count)', 'Pie Chart (KM)', 'Mean Train Track Section', 'Correlation Matrix', 'Histograms for Distribution', 'Display Numerical Means by Category', 'Display Numerical Distributions', 'Display Non-Numerical Distributions', 'Display Numerical Summary', 'Display Non-Numerical Summary Heatmap with Percentages' ]
)

# Pie chart for track count
if 'Pie Chart (Count)' in graph_options:
    st.title('Distribution of Matching Tracks (Count)')
    st.markdown("The pie chart shows the number of tracks that match the user-specified criteria")
    fig1, ax1 = plt.subplots()
    ax1.pie([filtered_tracks_count, total_tracks_count - filtered_tracks_count],
            labels=['Matching', 'Not Matching'], autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    st.pyplot(fig1)

# Pie chart for km track length
if 'Pie Chart (KM)' in graph_options:
    st.title('Distribution of Matching Tracks (KM)')
    st.markdown("The pie chart shows the kilometer of track that match the user-specified criteria")
    fig2, ax2 = plt.subplots()
    ax2.pie([filtered_km_tracks, total_km_tracks - filtered_km_tracks],
            labels=['Matching', 'Not Matching'], autopct='%1.1f%%', startangle=90)
    ax2.axis('equal')
    st.pyplot(fig2)

# Calculate the mean train track section
numerical_cols = df.select_dtypes(include=[float, int]).columns
non_numerical_cols = df.select_dtypes(exclude=[float, int]).columns.difference(['Geocode', 'To', 'From'])
mean_numerical_values = df[numerical_cols].mean()
mode_non_numerical_values = df[non_numerical_cols.drop(['Geocode', 'To', 'From'], errors='ignore')].mode().iloc[0]
mean_track_section = pd.concat([mean_numerical_values, mode_non_numerical_values])

# Mean Train Track Section
if 'Mean Train Track Section' in graph_options:
    st.title('Mean Train Track Section')
    st.markdown("""
## Mean Track Results

The mean track results provide a summary of the average values for both numerical and non-numerical features of the train tracks. This information helps in understanding the typical characteristics of the track sections under consideration.

**Numerical Columns:**
- These columns contain numerical data such as track length, number of signals, etc.
- The mean value is calculated for each numerical column.
- This provides an idea of the central tendency of the numerical features across the dataset.

**Non-Numerical Columns:**
- These columns contain categorical data such as the type of track, safety system, etc.
- The mode (most frequent value) is calculated for each non-numerical column.
- This gives an insight into the most common categories or attributes in the dataset.

### Key Points to Consider:

**Numerical Columns:**
- **Mean Value**: Represents the average value of the numerical features. It is calculated by summing all the values in a column and dividing by the number of values.
- **Interpretation**: The mean value provides a central value around which the data points are distributed.

**Non-Numerical Columns:**
- **Mode Value**: Represents the most frequent value or category in the non-numerical features.
- **Interpretation**: The mode value helps in identifying the most common category within the dataset.

### Visualization and Analysis:

- **Bar Charts for Numerical Columns**: Visual representations of the mean values for numerical columns help in easily comparing the average values across different features.
- **Tables for Non-Numerical Columns**: Displaying the mode values in a table format allows for a clear understanding of the most frequent categories.
""")
    st.subheader('Numerical Columns')
    fig, ax = plt.subplots()
    mean_numerical_values.plot(kind='bar', ax=ax)
    ax.set_ylabel('Mean Value')
    ax.set_title('Mean Values of Numerical Columns')
    st.pyplot(fig)

# Visualization: Table for Categorical Columns
    st.subheader('Categorical Columns')
    st.table(mode_non_numerical_values)

# Correlation Matrix of Numerical Features
if 'Correlation Matrix' in graph_options:
    st.title('Correlation Matrix of Numerical Features')
    st.markdown("""
A correlation matrix is a table showing correlation coefficients between sets of variables. Each cell in the table shows the correlation between two variables. The value is between -1 and 1.

**Key Points to Consider:**

**Correlation Coefficient:**
- **+1**: Perfect positive correlation (as one variable increases, the other increases).
- **0**: No correlation (no linear relationship between the variables).
- **-1**: Perfect negative correlation (as one variable increases, the other decreases).

**Strength of Correlation:**
- **0.0 to 0.1**: Negligible correlation.
- **0.1 to 0.3**: Weak correlation.
- **0.3 to 0.5**: Moderate correlation.
- **0.5 to 0.7**: Strong correlation.
- **0.7 to 1.0**: Very strong correlation.

**Sign of the Coefficient:**
- **Positive (+)**: Indicates that as one variable increases, the other variable also tends to increase.
- **Negative (-)**: Indicates that as one variable increases, the other variable tends to decrease.
""")
    fig4, ax4 = plt.subplots(figsize=(15, 15))  # Increase the figure size
    corr_matrix = df[numerical_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax4, annot_kws={"size": 8})  # Adjust font size
    ax4.set_title('Correlation Matrix')
    st.pyplot(fig4)

# Histograms for Distribution of Numerical Features
if 'Histograms for Distribution' in graph_options:
    st.title('Histograms for Distribution of Numerical Features')
    st.markdown("""
## Histograms for Distribution of Numerical Features

Histograms provide a visual representation of the distribution of numerical features in the dataset. They show how the data points are spread across different values, which helps in understanding the underlying patterns and distributions of the data.

### Key Points to Consider:

**Histogram Interpretation:**
- **Bars**: Each bar in a histogram represents the frequency of data points that fall within a specific range.
  - The height of the bar indicates how many data points are in that range.
- **Bins**: The range of values is divided into bins or intervals. The width of each bin affects the granularity of the histogram.
  - More bins provide a more detailed view, while fewer bins provide a more summarized view.

**Understanding Distribution Shapes:**
- **Normal Distribution**: A symmetric, bell-shaped curve where most data points cluster around the mean.
- **Skewed Distribution**: 
  - **Right-Skewed (Positive Skew)**: Most data points are concentrated on the left, with a long tail on the right.
  - **Left-Skewed (Negative Skew)**: Most data points are concentrated on the right, with a long tail on the left.
- **Uniform Distribution**: Data points are evenly distributed across the range.
- **Bimodal/Multimodal Distribution**: There are two or more peaks (modes) in the distribution.

### Analyzing Histograms:
- **Central Tendency**: Identifies the central value where data points tend to cluster.
- **Spread**: Measures how spread out the data points are (variance, standard deviation).
- **Outliers**: Identifies data points that fall far from the main distribution.
- **Skewness and Kurtosis**: 
  - **Skewness** indicates the asymmetry of the distribution.
  - **Kurtosis** indicates the "tailedness" of the distribution, or how heavy/light the tails are.
""")
    fig5, axes = plt.subplots(nrows=len(numerical_cols), ncols=1, figsize=(10, len(numerical_cols) * 4))
    plt.subplots_adjust(hspace=0.5)
    for col, ax in zip(numerical_cols, axes):
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f'Distribution of {col}')
    st.pyplot(fig5)

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

# Filter columns to relevant features, leave out track section numbers, geocodes, names
relevant_columns = df.loc[:, 'Emplacement':]

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
checked_df = df[included_columns]

# Group by 'Urban/Regional/Suburban' and calculate mean and standard deviation for numerical features and most frequent value for non-numerical features
numerical_cols = checked_df.select_dtypes(include=[float, int]).columns
non_numerical_cols = checked_df.select_dtypes(exclude=[float, int]).columns

mean_numerical = checked_df.groupby('Urban/Regional/Suburban')[numerical_cols].mean()
mode_non_numerical = checked_df.groupby('Urban/Regional/Suburban')[non_numerical_cols].agg(lambda x: x.mode()[0])
grouped_stds = checked_df.groupby('Urban/Regional/Suburban')[numerical_cols].std()

# Combine numerical and non-numerical summaries
summary_numerical = mean_numerical
summary_non_numerical = mode_non_numerical
summary_std = grouped_stds

# Define the function to plot all numerical features
def plot_all_numerical_features(mean_values, std_values, categories, group_size=6):
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
        st.pyplot(fig)
        plt.close(fig)
        

# Assuming mean_numerical and grouped_stds are DataFrames with the mean and std values of numerical features respectively
if 'Display Numerical Means by Category' in graph_options:
    plot_all_numerical_features(mean_numerical, grouped_stds, mean_numerical.index)

## Plotting the distributions
# Define the function to plot distributions
def plot_distributions(columns, df, title, cols=2):
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
    st.pyplot(fig)
    plt.close(fig)

# Split numerical columns into smaller groups for better readability
group_size = 6  # Number of subplots per figure

# Create subfigures for each group
if 'Display Numerical Distributions' in graph_options:
    for start_index in range(0, len(numerical_cols), group_size):
        end_index = min(start_index + group_size, len(numerical_cols))
        group = numerical_cols[start_index:end_index]
        plot_distributions(group, filtered_df, f'Distributions_of_Numerical_Features_{start_index + 1}_to_{end_index}')

    # Handle the remaining columns if the division is not perfect
    if end_index < len(numerical_cols):
        remaining_cols = numerical_cols[end_index:]
        plot_distributions(remaining_cols, filtered_df, 'Distributions_of_Remaining_Numerical_Features')

# Define the function to plot non-numerical distributions
def plot_non_numerical_distributions(columns, df, title, cols=2):
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
    st.pyplot(fig)
    plt.close(fig)

if 'Display Non-Numerical Distributions' in graph_options:
    plot_non_numerical_distributions(non_numerical_cols, filtered_df, 'Non-Numerical Feature Distributions')

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
    st.pyplot(fig)
    plt.close(fig)

if 'Display Numerical Summary' in graph_options:
    plot_numerical_summary(summary_numerical, 'Mean Urban/Regional/Suburban Train Track Sections')

# Define the function to plot non-numerical summary heatmap with percentages
def plot_non_numerical_summary_heatmap_with_percentages(df, title):
    summary_percentages = pd.DataFrame()

    # Create a mapping of feature to category
    feature_category_mapping = {
        'Tranche 1 ERTMS': ['Yes', 'No'],
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
    st.pyplot(fig)
    plt.close(fig)

if 'Display Non-Numerical Summary Heatmap with Percentages' in graph_options:
    plot_non_numerical_summary_heatmap_with_percentages(filtered_df, 'Most Frequent Urban/Regional/Suburban Train Track Sections with Percentages')


# Save the summary table to an in-memory Excel file
output = BytesIO()
with pd.ExcelWriter(output, engine='openpyxl') as writer:
    mean_track_section.to_excel(writer, sheet_name='Mean Track Section')
output.seek(0)

st.write("Summarized data is ready for download")

# Provide download link for the Excel file
st.download_button(
    label="Download Summary Excel",
    data=output,
    file_name="Mean_Track_Summary.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)


# Main content
st.write("Summarized data is ready for download")
st.title('Map of Train Track Sections')
# Load and display the ProRail logo
map_path = '67.png'
map = Image.open(map_path)
st.image(map, use_column_width=True)

    
