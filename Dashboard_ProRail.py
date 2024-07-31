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
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances

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

# Descriptive columns that should always be included
descriptive_columns = ['Track Section', 'Geocode', 'To', 'From']

# Sidebar for column selection using checkboxes
st.sidebar.title('Include/Exclude Columns and Filters')

# Dictionary to hold the inclusion state and filter values
column_inclusion = {}

# Create checkboxes for each column to include or exclude
for column in df.columns:
    if column in descriptive_columns:
        # Skip descriptive columns from having checkboxes
        continue
    
    # Check the data type to decide on multiselect or slider
    if pd.api.types.is_numeric_dtype(df[column]):
        include_column = st.sidebar.checkbox(f"Include {column}", value=True, key=f"{column}_include")
        if include_column:
            min_val = df[column].min()
            max_val = df[column].max()
            filter_values = st.sidebar.slider(f'{column}', min_val, max_val, (min_val, max_val), key=f"{column}_filter")
            column_inclusion[column] = (include_column, filter_values)
    else:
        include_column = st.sidebar.checkbox(f"Include {column}", value=True, key=f"{column}_include")
        if include_column:
            filter_values = st.sidebar.multiselect(f'{column}', df[column].unique(), default=df[column].unique(), key=f"{column}_filter")
            column_inclusion[column] = (include_column, filter_values)

# Start with descriptive columns always included
filtered_df = df[descriptive_columns].copy()

# Apply the filtering and column inclusion logic
for column, (include, filter_values) in column_inclusion.items():
    if include:
        if pd.api.types.is_numeric_dtype(df[column]):
            min_val, max_val = filter_values
            filtered_df = filtered_df.join(df[df[column].between(min_val, max_val)][[column]], how='inner')
        else:
            filtered_df = filtered_df.join(df[df[column].isin(filter_values)][[column]], how='inner')
st.title('Train Track Section Analysis')            
st.markdown("This is an interactive Dashboard presenting different train track sections in the Netherlands. The user is able to include and exclude certain characteristics"
            "from any statistical analysis and filter based on numerical or non-numerical values. The user can select the type of visualization they wish to see.")

# Display the filtered dataframe with an expander
st.header('Filtered Train Track Sections')
st.markdown("This dashboard allows the user to filter train track sections based on the filter options on the left side of the dashboard. The table shows which track sections match the chosen criteria.")

with st.expander("Click here to view the filtered track sections"):
    st.write(f"Number of tracks matching criteria: {filtered_df.shape[0]}")
    st.write(filtered_df)

# Display the filtered dataframe with a button
st.header('Filtered Train Track Sections')
st.markdown("This dashboard allows the user to filter train track sections based on the filter options on the left side of the dashboard. The table shows which track sections match the chosen criteria.")

if st.button('Show Filtered Track Sections'):
    st.write(f"Number of tracks matching criteria: {filtered_df.shape[0]}")
    st.write(filtered_df)

# Using an expander with an emoji
st.header('Filtered Train Track Sections')
st.markdown("This dashboard allows the user to filter train track sections based on the filter options on the left side of the dashboard. The table shows which track sections match the chosen criteria.")

with st.expander("📊 Click here to view the filtered track sections"):
    st.write(f"Number of tracks matching criteria: {filtered_df.shape[0]}")
    st.write(filtered_df)

# Using a button with an emoji
if st.button('📊 Show Filtered Track Sections'):
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
st.subheader('Visualization Options')
graph_options = st.multiselect(
    'Select the graphs you want to see:',
    ['Pie Chart (Count)', 'Pie Chart (KM)', 'Mean Train Track Section', 'Correlation Matrix', 'Histograms for Distribution']
)

# Calculate the mean train track section
numerical_cols = df.select_dtypes(include=[float, int]).columns.difference(descriptive_columns)
non_numerical_cols = df.select_dtypes(exclude=[float, int]).columns.difference(descriptive_columns)
mean_numerical_values = df[numerical_cols].mean()
mode_non_numerical_values = df[non_numerical_cols].mode().iloc[0]
mean_track_section = pd.concat([mean_numerical_values, mode_non_numerical_values])

# Filter columns to relevant features, leave out track section numbers, geocodes, names
relevant_columns = df.loc[:, 'Emplacement':]

# Create two columns
col1, col2 = st.columns(2)

# Pie chart for track count
with col1:
    st.subheader('Distribution of Matching Tracks (Count)')
    st.markdown("The pie chart shows the number of tracks that match the user-specified criteria")
    fig1, ax1 = plt.subplots(figsize=(4, 4))  # Adjust the size as needed
    ax1.pie([filtered_tracks_count, total_tracks_count - filtered_tracks_count],
            labels=['Matching', 'Not Matching'], autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    st.pyplot(fig1)

# Pie chart for km track length
with col2:
    st.subheader('Distribution of Matching Tracks (KM)')
    st.markdown("The pie chart shows the kilometer of track that match the user-specified criteria")
    fig2, ax2 = plt.subplots(figsize=(4, 4))  # Adjust the size as needed
    ax2.pie([filtered_km_tracks, total_km_tracks - filtered_km_tracks],
            labels=['Matching', 'Not Matching'], autopct='%1.1f%%', startangle=90)
    ax2.axis('equal')
    st.pyplot(fig2)

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
    # Filter the numerical columns based on the selected columns
    numerical_cols = filtered_df.select_dtypes(include=[float, int]).columns
    mean_numerical_values = filtered_df[numerical_cols].mean()
    
    # Plotting the mean values for numerical columns
    st.subheader('Numerical Columns')
    fig, ax = plt.subplots()
    mean_numerical_values.plot(kind='bar', ax=ax)
    ax.set_ylabel('Mean Value')
    ax.set_title('Mean Values of Numerical Columns')
    st.pyplot(fig)

    # Visualization: Table for Categorical Columns
    st.subheader('Categorical Columns')
    non_numerical_cols = filtered_df.select_dtypes(exclude=[float, int]).columns
    mode_non_numerical_values = filtered_df[non_numerical_cols].mode().iloc[0]

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
    numerical_cols = filtered_df.select_dtypes(include=[float, int]).columns
    corr_matrix = df[numerical_cols].corr()
    fig4, ax4 = plt.subplots(figsize=(15, 15))  # Increase the figure size
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
    
    numerical_cols = filtered_df.select_dtypes(include=[float, int]).columns  
    fig5, axes = plt.subplots(nrows=len(numerical_cols), ncols=1, figsize=(10, len(numerical_cols) * 4))
    plt.subplots_adjust(hspace=0.5)
    for col, ax in zip(numerical_cols, axes):
        sns.histplot(df[col].dropna(), kde=True, ax=ax)
        ax.set_title(f'Distribution of {col}')
    st.pyplot(fig5)

# Add a title and description
st.header('Urban/Suburban/Regional Train Track Types Analysis')
st.markdown("""
    This dashboard allows you to analyze and visualize various features of train tracks categorized into urban, suburban, and regional types.

    **Instructions:**
    1. Use the sidebar to include or exclude specific features in the analysis.
    2. Choose whether to exclude emplacement data.
    3. Select the types of plots you want to display.
    4. The dashboard provides options to display means, distributions, and summaries of numerical and non-numerical features.

    **Note:** The data is filtered based on the selections you make in the sidebar.
""")
# Visualization Options
st.subheader('Visualization Options')
graph_options = st.multiselect(
    'Select the graphs you want to see:',
    ['Display Numerical Means by Category', 'Display Numerical Distributions', 'Display Non-Numerical Distributions', 'Display Numerical Summary']
)
# Group by 'Urban/Regional/Suburban' and calculate mean and standard deviation for numerical features and most frequent value for non-numerical features
numerical_cols = filtered_df.select_dtypes(include=[float, int]).columns.difference(descriptive_columns)
non_numerical_cols = filtered_df.select_dtypes(exclude=[float, int]).columns.difference(descriptive_columns)

mean_numerical = filtered_df.groupby('Urban/Regional/Suburban')[numerical_cols].mean()
mode_non_numerical = filtered_df.groupby('Urban/Regional/Suburban')[non_numerical_cols].agg(lambda x: x.mode()[0])
grouped_stds = filtered_df.groupby('Urban/Regional/Suburban')[numerical_cols].std()

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

st.header('K-Clustering of Train Track Sections')
st.markdown("The k-means clustering algorithm is applied to the preprocessed data. K-means clustering"
"aims to partition n observations into k clusters in which each observation belongs to the"
"cluster with the nearest mean, serving as a prototype of the cluster. The k-means algorithm"
"minimizes the WCSS (Within-Cluster Sum of Square), also known as the inertia.")

#Select numerical columns for clustering from the included columns
numerical_cols = [col for col, (include, _) in column_inclusion.items() if include and pd.api.types.is_numeric_dtype(df[col])]

if numerical_cols:
    numerical_data = filtered_df[numerical_cols]

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    imputed_data = imputer.fit_transform(numerical_data)

    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(imputed_data)

    # Determine the optimal number of clusters using the elbow method
    wcss = []
    max_clusters = 15

    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
        kmeans.fit(scaled_data)
        wcss.append(kmeans.inertia_)

    # Visualization Options
    st.subheader('Visualization Options')
    graph_options = st.multiselect(
        'Select the graphs you want to see:',
        ['Elbow Curve', 'PCA Result', 'Pairplot']
    )

    if 'Elbow Curve' in graph_options:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, max_clusters + 1), wcss, marker='o')
        plt.title('Elbow Method for Optimal k')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
        plt.xticks(range(1, max_clusters + 1))
        plt.grid(True)
        st.pyplot()

    # Choose the number of clusters (Here fixed to 5, but you can make this dynamic)
    k = 5

    # Fit the k-means model
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_data)

    # Add cluster labels to the scaled data
    scaled_data_df = pd.DataFrame(scaled_data, columns=numerical_cols)
    scaled_data_df['Cluster'] = clusters

    # Reduce dimensions using PCA for visualization
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)
    pca_df = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = clusters

    if 'PCA Result' in graph_options:  # Plot the PCA result
        plt.figure(figsize=(10, 6))
        for cluster in range(k):
            plt.scatter(pca_df[pca_df['Cluster'] == cluster]['PC1'],
                        pca_df[pca_df['Cluster'] == cluster]['PC2'],
                        label=f'Cluster {cluster}')
        plt.title('PCA of Clusters')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        plt.grid(True)
        st.pyplot()

    # Pairplot for detailed visualization of clusters (subset of features)
    subset_features = numerical_cols[:5]  # Select first 5 numerical features for pairplot
    pairplot_data = pd.concat([pd.DataFrame(scaled_data, columns=numerical_cols), pd.Series(clusters, name='Cluster')], axis=1)
    pairplot_data = pairplot_data[['Cluster'] + list(subset_features)]
    pairplot_data['Cluster'] = pairplot_data['Cluster'].astype(str)  # Convert to string for better visualization

    if 'Pairplot' in graph_options:  # Plot pairplot
        sns.pairplot(pairplot_data, hue='Cluster', palette='Set1')
        plt.suptitle('Pairplot of Clusters (Subset of Features)', y=1.02)
        st.pyplot()

# Adding the cluster labels back to the original data to analyze cluster characteristics
df['Cluster'] = clusters

# Calculate the mean values of numeric features for each cluster
cluster_analysis = df.groupby('Cluster')[numerical_cols].mean()

# Analyze non-numerical values by cluster, excluding descriptive columns
non_numerical_cols_for_analysis = non_numerical_cols.difference(descriptive_columns)
non_numerical_analysis = df.groupby('Cluster')[non_numerical_cols_for_analysis].agg(lambda x: x.value_counts().index[0])


# Start with descriptive columns always included
filtered_df = df[descriptive_columns].copy()

# Apply the filtering and column inclusion logic
included_numerical_cols = []
included_non_numerical_cols = []

for column, (include, filter_values) in column_inclusion.items():
    if include:
        if pd.api.types.is_numeric_dtype(df[column]):
            included_numerical_cols.append(column)
            min_val, max_val = filter_values
            filtered_df = filtered_df.join(df[df[column].between(min_val, max_val)][[column]], how='inner')
        else:
            included_non_numerical_cols.append(column)
            filtered_df = filtered_df.join(df[df[column].isin(filter_values)][[column]], how='inner')

# Use the lists `included_numerical_cols` and `included_non_numerical_cols` for similarity calculations
def calculate_similarity(df, mean_vector, numerical_cols, non_numerical_cols):
    # Normalize numerical columns
    scaler = StandardScaler()
    df_numerical = scaler.fit_transform(df[numerical_cols])
    mean_numerical = scaler.transform([mean_vector[numerical_cols]])

    # Numerical similarity based on normalized Euclidean distance
    numerical_distances = euclidean_distances(df_numerical, mean_numerical)
    max_numerical_distance = numerical_distances.max()
    numerical_similarity = 1 - (numerical_distances / max_numerical_distance)

    # Non-numerical similarity based on mode matching
    non_numerical_similarity = df[non_numerical_cols].apply(lambda row: sum(row == mean_vector[non_numerical_cols]), axis=1)
    max_non_numerical_similarity = len(non_numerical_cols)
    non_numerical_similarity = non_numerical_similarity / max_non_numerical_similarity

    # Combine both similarities with equal weighting
    similarity_score = (numerical_similarity.flatten() + non_numerical_similarity) / 2
    return similarity_score

# Displaying similar tracks using the updated columns
def display_similar_tracks(df, mean_vector, numerical_cols, non_numerical_cols, section_type):
    similarities = calculate_similarity(df, mean_vector, numerical_cols, non_numerical_cols)
    df['Similarity'] = similarities
    similar_tracks = df.nlargest(10, 'Similarity')  # Show top 10 similar tracks
    st.write(f"Top 10 tracks similar to the {section_type} Mean Track Section")
    st.write(similar_tracks[['Track Section', 'Similarity'] + list(numerical_cols) + list(non_numerical_cols)])
    df.drop(columns=['Similarity'], inplace=True)  # Clean up

# Sidebar and Main Content
st.header('Track Section Similarity Analysis')

# Buttons for displaying similar tracks
if st.button('Mean Track Section in Real tracks'):
    display_similar_tracks(df, mean_track_section, included_numerical_cols, included_non_numerical_cols, 'Mean')

if st.button('Urban Track Section in Real tracks'):
    urban_mean = pd.concat([mean_numerical.loc['Urban'], mode_non_numerical.loc['Urban']])
    display_similar_tracks(df, urban_mean, included_numerical_cols, included_non_numerical_cols, 'Urban')

if st.button('Suburban Track Section in Real tracks'):
    suburban_mean = pd.concat([mean_numerical.loc['Suburban'], mode_non_numerical.loc['Suburban']])
    display_similar_tracks(df, suburban_mean, included_numerical_cols, included_non_numerical_cols, 'Suburban')

if st.button('Regional Track Section in Real tracks'):
    regional_mean = pd.concat([mean_numerical.loc['Regional'], mode_non_numerical.loc['Regional']])
    display_similar_tracks(df, regional_mean, included_numerical_cols, included_non_numerical_cols, 'Regional')

# For each cluster, similar implementation
for i in range(5):
    cluster_mean = pd.concat([cluster_analysis.loc[i], non_numerical_analysis.loc[i]])
    if st.button(f'Cluster {i} in Real tracks'):
        display_similar_tracks(df, cluster_mean, included_numerical_cols, included_non_numerical_cols, f'Cluster {i}')


st.subheader('Download Data Summaries to Excel')
# Save the summary table to an in-memory Excel file
output = BytesIO()
with pd.ExcelWriter(output, engine='openpyxl') as writer:
    mean_track_section.to_excel(writer, sheet_name='Mean Track Section')
output.seek(0)

# Provide download link for the Excel file
st.download_button(
    label="Download Summary of Mean Track to Excel",
    data=output,
    file_name="Mean_Track_Summary.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# Save the summary table to an in-memory Excel file
output = BytesIO()
with pd.ExcelWriter(output, engine='openpyxl') as writer:
    summary_numerical.to_excel(writer, sheet_name='Numerical Features')
    summary_std.to_excel(writer, sheet_name='Standard Deviation')
    summary_non_numerical.to_excel(writer, sheet_name='Non-Numerical Features')
output.seek(0)

# Provide download link for the Excel file
st.download_button(
    label="Download Summary of Urban/Suburban/Regional Tracks to Excel",
    data=output,
    file_name="Categories_Summary.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# Save the summary table to an in-memory Excel file
output = BytesIO()
with pd.ExcelWriter(output, engine='openpyxl') as writer:
    if not cluster_analysis.empty:
        cluster_analysis.to_excel(writer, sheet_name='Cluster_Summary')
    if not non_numerical_analysis.empty:
        non_numerical_analysis.to_excel(writer, sheet_name='Non_Numerical_Summary')
output.seek(0)


# Provide download link for the Excel file
st.download_button(
    label="Download Summary of K-Means Clusters to Excel",
    data=output,
    file_name="K_Means_Clusters.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# Main content
st.subheader('Map of Train Track Sections')
# Load and display the ProRail logo
map_path = '67.png'
map = Image.open(map_path)
st.image(map, use_column_width=True)

    
