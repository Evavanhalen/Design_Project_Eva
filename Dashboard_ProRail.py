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
pdf_file_path = 'User_Guide (1).pdf'

# Load the data from the template
df = pd.read_excel(file_path, sheet_name='Template')

# Load and display the ProRail logo
logo = Image.open(logo_path)

# Display the logo in the sidebar
st.sidebar.image(logo, use_column_width=True)

# Apply custom CSS
st.markdown("""
    <style>
    .stButton button {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

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
            "from any statistical analysis and filter based on numerical or non-numerical values. The user can select the type of visualization they wish to see. For more detailed information into the dashbaord, please download the user guide below.")

with open(pdf_file_path, "rb") as f:
    pdf_data = f.read()

# Create a download button
st.download_button(
    label="📕Download User Guide",
    data=pdf_data,
    file_name="User_Guide.pdf",
    mime="application/pdf"
)

# Main content
with st.expander("🗺️ Click here to view the Map of Train Track Sections"):
    st.subheader('Map of Train Track Sections')
    # Load and display the ProRail logo
    map_path = '67.png'
    map = Image.open(map_path)
    st.image(map, use_column_width=True)


st.markdown("""
    <h1 style='font-size:2.5em; color:navy;'>Filtering Section</h1>
    <hr style='border:2px solid navy;'>
    """, unsafe_allow_html=True)
st.markdown("This dashboard allows the user to filter train track sections based on the filter options on the left side of the dashboard. The table shows which track sections match the chosen criteria.")

with st.expander("🔎 Click here to view the filtered track sections"):
    st.write(f"Number of tracks matching criteria: {filtered_df.shape[0]}")
    st.write(filtered_df)


total_tracks_count = df.shape[0]
filtered_tracks_count = filtered_df.shape[0]
percentage_matching_tracks = (filtered_tracks_count / total_tracks_count) * 100

total_km_tracks = df['km track'].sum()
filtered_km_tracks = filtered_df['km track'].sum()
percentage_matching_km_tracks = (filtered_km_tracks / total_km_tracks) * 100

# Assuming track_length is a column in your DataFrame
total_track_length = df['Track length (km)'].sum()
filtered_track_length = filtered_df['Track length (km)'].sum()
percentage_matching_track_length = (filtered_track_length / total_track_length) * 100

# Visualization Options
st.subheader('Visualization Options')
graph_options = st.multiselect(
    'Select the graphs you want to see:',
    ['Pie Chart (Count)', 'Pie Chart (KM/Length)', 'Mean Train Track Section']
)

# Toggle between Track Length and Track KM for Pie Chart
track_measurement = st.radio(
    "Select the measurement for the second pie chart:",
    ('Track Kilometers', 'Track Length'),
    key="track_measurement_toggle"  # Add a unique key here
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

if 'Pie Chart (Count)' in graph_options:
    with col1:
        st.subheader('Distribution of Matching Tracks (Count)')
        st.markdown("The pie chart shows the number of tracks that match the user-specified criteria")
        fig1, ax1 = plt.subplots(figsize=(4, 4))  # Adjust the size as needed
        ax1.pie([filtered_tracks_count, total_tracks_count - filtered_tracks_count],
                labels=['Matching', 'Not Matching'], autopct='%1.1f%%', startangle=90)
        ax1.axis('equal')
        st.pyplot(fig1)
        st.write(f"Total tracks: {total_tracks_count}")
        st.write(f"Tracks matching criteria: {filtered_tracks_count}")
        st.write(f"Percentage matching criteria: {percentage_matching_tracks:.2f}%")

# Conditional display for Pie chart (KM/Length)
if 'Pie Chart (KM/Length)' in graph_options:
    with col2:
        if track_measurement == 'Track Kilometers':
            st.subheader('Distribution of Matching Tracks (KM)')
            st.markdown("The pie chart shows the kilometer of track that match the user-specified criteria")
            fig2, ax2 = plt.subplots(figsize=(4, 4))  # Adjust the size as needed
            ax2.pie([filtered_km_tracks, total_km_tracks - filtered_km_tracks],
                    labels=['Matching', 'Not Matching'], autopct='%1.1f%%', startangle=90)
            ax2.axis('equal')
            st.pyplot(fig2)
            st.write(f"Total km of tracks: {total_km_tracks:.2f} km")
            st.write(f"Km of tracks matching criteria: {filtered_km_tracks:.2f} km")
            st.write(f"Percentage of km tracks matching criteria: {percentage_matching_km_tracks:.2f}%")
        else:
            st.subheader('Distribution of Matching Tracks (Track Length)')
            st.markdown("The pie chart shows the total track length that matches the user-specified criteria")
            fig2, ax2 = plt.subplots(figsize=(4, 4))  # Adjust the size as needed
            ax2.pie([filtered_track_length, total_track_length - filtered_track_length],
                    labels=['Matching', 'Not Matching'], autopct='%1.1f%%', startangle=90)
            ax2.axis('equal')
            st.pyplot(fig2)
            st.write(f"Total track length: {total_track_length:.2f} units")
            st.write(f"Track length matching criteria: {filtered_track_length:.2f} units")
            st.write(f"Percentage matching track length: {percentage_matching_track_length:.2f}%")


# Mean Train Track Section
if 'Mean Train Track Section' in graph_options:
    st.subheader('Mean Train Track Section')
    with st.expander("📖 Click here for a detailed explanation of Mean Track Results"):
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
    
# Create two columns for side-by-side display
    col1, col2 = st.columns(2)

    # Plotting the mean values for numerical columns
    with col1:
        st.subheader('Numerical Columns')
        fig, ax = plt.subplots()
        mean_numerical_values.plot(kind='bar', ax=ax)
        ax.set_ylabel('Mean Value')
        ax.set_title('Mean Values of Numerical Columns')
        st.pyplot(fig)

    # Displaying the mode values in a table format for non-numerical columns
    with col2:
        st.subheader('Categorical Columns')
        st.table(mode_non_numerical_values)

        # Button to display histograms
    with st.expander('📊 Click here for statistical distribution insights'):
        st.subheader('Histograms for Distribution of Numerical Features')
        st.markdown("""
        Histograms provide a visual representation of the distribution of numerical features in the dataset. They show how the data points are spread across different values, which helps in understanding the underlying patterns and distributions of the data.
        """)

        # Display histograms in a smaller size with 3 columns
        numerical_cols = filtered_df.select_dtypes(include=[float, int]).columns  
        num_cols = 3  # Number of columns for histograms
        num_rows = (len(numerical_cols) + num_cols - 1) // num_cols  # Calculate number of rows needed

        for row in range(num_rows):
            cols = st.columns(num_cols)
            for col_idx, col_name in enumerate(numerical_cols[row*num_cols:(row+1)*num_cols]):
                with cols[col_idx]:
                    fig, ax = plt.subplots(figsize=(3, 3))  # Adjust the size as needed
                    sns.histplot(filtered_df[col_name].dropna(), kde=True, ax=ax)
                    ax.set_title(f'{col_name}')
                    st.pyplot(fig)
                    plt.close(fig)

# Function Definitions (placed outside the layout)
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

def display_similar_tracks(df, mean_vector, numerical_cols, non_numerical_cols, section_type):
    similarities = calculate_similarity(df, mean_vector, numerical_cols, non_numerical_cols)
    df['Similarity'] = similarities
    similar_tracks = df.nlargest(10, 'Similarity')  # Show top 10 similar tracks
    st.write(f"Top 10 tracks similar to the {section_type} Mean Track Section")
    st.write(similar_tracks[['Track Section', 'Similarity'] + list(numerical_cols) + list(non_numerical_cols)])
    df.drop(columns=['Similarity'], inplace=True)  # Clean up

# Filtering and inclusion logic (this should be placed before the column layout to ensure variables are available)
included_numerical_cols = []  # Initialize as an empty list
included_non_numerical_cols = []  # Initialize as an empty list

# Assume column_inclusion is a dictionary that has been populated earlier in the script
for column, (include, filter_values) in column_inclusion.items():
    if include:
        if pd.api.types.is_numeric_dtype(df[column]):
            included_numerical_cols.append(column)
        else:
            included_non_numerical_cols.append(column)

# Column Layout for the interactive elements
col1, col2 = st.columns([2, 3])

with col1:
    st.subheader('Download to Excel')
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        mean_track_section.to_excel(writer, sheet_name='Mean Track Section')
    output.seek(0)
    st.download_button(
        label="Download Summary of Mean Track to Excel",
        data=output,
        file_name="Mean_Track_Summary.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

with col2:
    st.subheader('Find a real-life match')
    if st.button('Mean Track Section in Real tracks'):
        display_similar_tracks(df, mean_track_section, included_numerical_cols, included_non_numerical_cols, 'Mean')


# Add a title and description
st.markdown("""
    <h1 style='font-size:2.4em; color:darkgreen;'>Urban/Suburban/Regional Train Tracks </h1>
    <hr style='border:2px solid darkgreen;'>
    """, unsafe_allow_html=True)
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
    ['Numerical Means by Category', 'Non-Numerical Modes by Category', 'Numerical Summary']
)
# Group by 'Urban/Regional/Suburban' and calculate mean and standard deviation for numerical features and most frequent value for non-numerical features
numerical_cols = filtered_df.select_dtypes(include=[float, int]).columns.difference(descriptive_columns)
non_numerical_cols = filtered_df.select_dtypes(exclude=[float, int]).columns.difference(descriptive_columns)

mean_numerical = filtered_df.groupby('Urban/Regional/Suburban')[numerical_cols].mean()
mode_non_numerical = filtered_df.groupby('Urban/Regional/Suburban')[non_numerical_cols].agg(lambda x: x.mode()[0])
grouped_stds = filtered_df.groupby('Urban/Regional/Suburban')[numerical_cols].std()

# Handle the case where there is never a standard deviation for the regional track type
for col in grouped_stds.columns:
    if grouped_stds[col].isna().all():
        grouped_stds[col] = 0

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
        

if 'Numerical Means by Category' in graph_options:
    plot_all_numerical_features(mean_numerical, grouped_stds, mean_numerical.index)

    # Add an expander for numerical distributions
    with st.expander("📊 Click here for detailed numerical distributions"):
        # Define the function to plot distributions
        def plot_distributions(columns, df, title, cols=3):
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
        for start_index in range(0, len(numerical_cols), group_size):
            end_index = min(start_index + group_size, len(numerical_cols))
            group = numerical_cols[start_index:end_index]
            plot_distributions(group, filtered_df, f'Distributions of Numerical Features {start_index + 1} to {end_index}')

        # Handle the remaining columns if the division is not perfect
        if end_index < len(numerical_cols):
            remaining_cols = numerical_cols[end_index:]
            plot_distributions(remaining_cols, filtered_df, 'Distributions of Remaining Numerical Features')

# Display mode of non-numerical columns by category
if 'Non-Numerical Modes by Category' in graph_options:
    st.subheader('Non-Numerical Feature Modes')
    st.write("Below are the most common values (mode) for the non-numerical features across different track categories.")
    st.table(mode_non_numerical)

    # Add an expander for detailed non-numerical distributions
    with st.expander("📊 Click here for detailed non-numerical feature distributions"):
        # Define the function to plot non-numerical distributions
        def plot_non_numerical_distributions(columns, df, title, cols=3):
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

        # Call the plotting function for non-numerical distributions
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

if 'Numerical Summary' in graph_options:
    st.subheader('Summary of Numerical Features by Category')
    plot_numerical_summary(summary_numerical, 'Mean Urban/Regional/Suburban Train Track Sections')

# Function Definitions (placed outside the layout)
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

def display_similar_tracks(df, mean_vector, numerical_cols, non_numerical_cols, section_type):
    similarities = calculate_similarity(df, mean_vector, numerical_cols, non_numerical_cols)
    df['Similarity'] = similarities
    similar_tracks = df.nlargest(10, 'Similarity')  # Show top 10 similar tracks
    st.write(f"Top 10 tracks similar to the {section_type} Mean Track Section")
    st.write(similar_tracks[['Track Section', 'Similarity'] + list(numerical_cols) + list(non_numerical_cols)])
    df.drop(columns=['Similarity'], inplace=True)  # Clean up

# Filtering and inclusion logic (this should be placed before the column layout to ensure variables are available)
included_numerical_cols = []  # Initialize as an empty list
included_non_numerical_cols = []  # Initialize as an empty list

# Assume column_inclusion is a dictionary that has been populated earlier in the script
for column, (include, filter_values) in column_inclusion.items():
    if include:
        if pd.api.types.is_numeric_dtype(df[column]):
            included_numerical_cols.append(column)
        else:
            included_non_numerical_cols.append(column)

# Column Layout for the interactive elements
col1, col2 = st.columns([2, 3])

with col1:
    st.subheader('Download to Excel')
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        summary_numerical.to_excel(writer, sheet_name='Numerical Features')
        summary_std.to_excel(writer, sheet_name='Standard Deviation')
        summary_non_numerical.to_excel(writer, sheet_name='Non-Numerical Features')
    output.seek(0)
    st.download_button(
        label="Download Summary of Urban/Suburban/Regional Tracks to Excel",
        data=output,
        file_name="Categories_Summary.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

with col2:
    st.subheader('Find a real-life match')
    if st.button('Urban Track Section in Real tracks'):
        urban_mean = pd.concat([mean_numerical.loc['Urban'], mode_non_numerical.loc['Urban']])
        display_similar_tracks(df, urban_mean, included_numerical_cols, included_non_numerical_cols, 'Urban')

    if st.button('Suburban Track Section in Real tracks'):
        suburban_mean = pd.concat([mean_numerical.loc['Suburban'], mode_non_numerical.loc['Suburban']])
        display_similar_tracks(df, suburban_mean, included_numerical_cols, included_non_numerical_cols, 'Suburban')

    if st.button('Regional Track Section in Real tracks'):
        regional_mean = pd.concat([mean_numerical.loc['Regional'], mode_non_numerical.loc['Regional']])
        display_similar_tracks(df, regional_mean, included_numerical_cols, included_non_numerical_cols, 'Regional')



# Function Definitions (placed at the beginning)
def calculate_similarity(df, mean_vector, numerical_cols, non_numerical_cols):
    scaler = StandardScaler()
    df_numerical = scaler.fit_transform(df[numerical_cols])
    mean_numerical = scaler.transform([mean_vector[numerical_cols]])
    numerical_distances = euclidean_distances(df_numerical, mean_numerical)
    max_numerical_distance = numerical_distances.max()
    numerical_similarity = 1 - (numerical_distances / max_numerical_distance)

    non_numerical_similarity = df[non_numerical_cols].apply(lambda row: sum(row == mean_vector[non_numerical_cols]), axis=1)
    max_non_numerical_similarity = len(non_numerical_cols)
    non_numerical_similarity = non_numerical_similarity / max_non_numerical_similarity

    similarity_score = (numerical_similarity.flatten() + non_numerical_similarity) / 2
    return similarity_score

def display_similar_tracks(df, mean_vector, numerical_cols, non_numerical_cols, section_type):
    similarities = calculate_similarity(df, mean_vector, numerical_cols, non_numerical_cols)
    df['Similarity'] = similarities
    similar_tracks = df.nlargest(10, 'Similarity')
    st.write(f"Top 10 tracks similar to the {section_type} Mean Track Section")
    st.write(similar_tracks[['Track Section', 'Similarity'] + list(numerical_cols) + list(non_numerical_cols)])
    df.drop(columns=['Similarity'], inplace=True)

st.markdown("""
    <h1 style='font-size:2.5em; color:darkred;'>K-Clustering of Train Track Sections</h1>
    <hr style='border:2px solid darkred;'>
    """, unsafe_allow_html=True)
st.markdown("The k-means clustering algorithm is applied to the preprocessed data. K-means clustering aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean, serving as a prototype of the cluster. The k-means algorithm minimizes the WCSS (Within-Cluster Sum of Square), also known as the inertia.")

st.subheader('Visualization Options')
graph_options = st.multiselect(
    'Select the graphs you want to see:',
    ['3D PCA', 'Pie Chart']
)

numerical_cols = [col for col, (include, _) in column_inclusion.items() if include and pd.api.types.is_numeric_dtype(df[col])]

# After applying filters, check the number of samples
if filtered_df.shape[0] == 0:
    st.warning("No data available after applying filters. Please adjust the filters.")
else:
    # Define the number of clusters, k
    default_k = 5  # Default number of clusters
    k = min(default_k, filtered_df.shape[0])  # Set k to the default or the number of samples, whichever is smaller
    
    if filtered_df.shape[0] < k:
        st.warning(f"Not enough data for {k} clusters. Only {filtered_df.shape[0]} samples available. Adjusting number of clusters to {filtered_df.shape[0]}.")
        k = filtered_df.shape[0]
    
    # Ensure numerical_data is defined and valid
    numerical_data = filtered_df[numerical_cols]

    # Convert numerical_data to a NumPy array for the checks
    numerical_data_array = numerical_data.to_numpy()

    # Ensure there's no NaN or infinite values before scaling
    if np.isnan(numerical_data_array).any() or np.isinf(numerical_data_array).any():
        st.error("Filtered data contains NaN or infinite values. Please check the data or adjust the filters.")
    else:
        # Impute missing values and scale the data
        imputer = SimpleImputer(strategy='mean')
        imputed_data = imputer.fit_transform(numerical_data_array)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(imputed_data)

        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(scaled_data)

        scaled_data_df = pd.DataFrame(scaled_data, columns=numerical_cols)
        scaled_data_df['Cluster'] = clusters
        filtered_df['Cluster'] = clusters
        
        # Ensure there are no NaN or infinite values in the scaled data
        if np.isnan(scaled_data).any() or np.isinf(scaled_data).any():
            raise ValueError("scaled_data contains NaN or infinite values.")

        # Ensure the number of clusters is less than or equal to the number of samples
        if scaled_data.shape[0] < k:
            raise ValueError(f"Number of clusters ({k}) cannot be greater than the number of samples ({scaled_data.shape[0]}).")

        # Fit the KMeans model
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_data)


    
# 3D PCA Plot and Pie Chart
if '3D PCA' in graph_options or 'Pie Chart' in graph_options:
    col1, col2 = st.columns(2)
    with st.expander("📖 Click here for an explanation of the visualizations"):
        st.markdown("""
        ## Explanation of Visualizations

        **3D PCA Plot**:
        - This plot shows the clusters found in the data using k-means clustering.
        - Each point represents a section of the train track.
        - The three axes (PC1, PC2, PC3) are the first three principal components, which are new features created to summarize the data.
        - Points that are close to each other are similar in terms of the selected features.
        - Different colors represent different clusters.

        **Pie Chart**:
        - This chart shows the distribution of the clusters.
        - Each slice of the pie represents one cluster.
        - The size of each slice shows how many sections of the train track belong to that cluster.
        """)

    with st.expander("📖 Click here for an explanation of the k-clustering algorithm"):
        st.markdown("""
        ## Explanation of the Clustering Algorithm

        **K-Means Clustering**:
        - The k-means clustering algorithm groups the data into clusters.
        - Each data point is assigned to the nearest cluster center, called a centroid.
        - The algorithm tries to minimize the distance between data points and their respective centroids.
        - This way, data points within the same cluster are similar to each other.
        - In this analysis, the algorithm has grouped the train track sections into 5 clusters.
        """)

    if '3D PCA' in graph_options:
        with col1:
            pca = PCA(n_components=3)
            pca_data = pca.fit_transform(scaled_data)
            pca_df = pd.DataFrame(pca_data, columns=['PC1', 'PC2', 'PC3'])
            pca_df['Cluster'] = clusters

            fig = plt.figure(figsize=(5, 3), dpi=200)  # Reduced size and high DPI for better quality
            ax = fig.add_subplot(111, projection='3d')

            colors = sns.color_palette("hsv", len(pca_df['Cluster'].unique()))

            for cluster in pca_df['Cluster'].unique():
                cluster_data = pca_df[pca_df['Cluster'] == cluster]
                ax.scatter(cluster_data['PC1'], cluster_data['PC2'], cluster_data['PC3'], 
                           label=f'Cluster {cluster}', s=50, alpha=0.6, color=colors[cluster])

            ax.set_title('3D PCA of Clusters', fontsize=10)
            ax.set_xlabel('PC1', fontsize=6)
            ax.set_ylabel('PC2', fontsize=6)
            ax.set_zlabel('PC3', fontsize=6)
            ax.legend(fontsize=6)
            st.pyplot(fig)

    if 'Pie Chart' in graph_options:
        with col2:
            cluster_counts = filtered_df['Cluster'].value_counts()

            fig, ax = plt.subplots(figsize=(4, 4), dpi=200)  # Reduced size and high DPI for better quality
            ax.pie(cluster_counts, labels=cluster_counts.index, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 6})
            ax.set_title('Cluster Distribution', fontsize=10)
            st.pyplot(fig)


cluster_analysis = filtered_df.groupby('Cluster')[numerical_cols].mean()
non_numerical_cols_for_analysis = non_numerical_cols.difference(descriptive_columns)
non_numerical_analysis = filtered_df.groupby('Cluster')[non_numerical_cols_for_analysis].agg(lambda x: x.value_counts().index[0])

# Column Layout for the interactive elements
col1, col2 = st.columns([2, 3])

# Column Layout for the interactive elements
col1, col2 = st.columns([2, 3])

with col1:
    st.subheader('Download to Excel')
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        if not cluster_analysis.empty:
            cluster_analysis.to_excel(writer, sheet_name='Cluster_Summary')
        if not non_numerical_analysis.empty:
            non_numerical_analysis.to_excel(writer, sheet_name='Non_Numerical_Summary')
    output.seek(0)
    st.download_button(
        label="Download Summary of K-Means Clusters to Excel",
        data=output,
        file_name="K_Means_Clusters.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

with col2:
    st.subheader('Find a real-life match')
    # Loop over the existing clusters instead of assuming indices
    for i in cluster_analysis.index:
        if i in non_numerical_analysis.index:
            cluster_mean = pd.concat([cluster_analysis.loc[i], non_numerical_analysis.loc[i]])
            if st.button(f'Cluster {i} in Real tracks'):
                display_similar_tracks(df, cluster_mean, included_numerical_cols, included_non_numerical_cols, f'Cluster {i}')
        else:
            st.warning(f"Cluster {i} not found due to filtering in left menu. Remove some filters to obtain up to 5 clusters.")

    
