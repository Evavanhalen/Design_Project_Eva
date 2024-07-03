import warnings
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from PIL import Image

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning, module='seaborn._oldcore')

# Define file paths using relative paths
file_path = 'Template_V03 (1).xlsx'
logo_path = 'ProRail logo.png'

# Load the data from the template
df = pd.read_excel(file_path, sheet_name='Template')

# Load and display the ProRail logo
logo = Image.open(logo_path)
st.sidebar.image(logo, use_column_width=True)

# Sidebar filters (unchanged)
st.sidebar.title('Filter Options')
ertms_filter = st.sidebar.multiselect('ERTMS in 2031', df['ERTMS in 2031'].unique(), default=df['ERTMS in 2031'].unique())
track_type_filter = st.sidebar.multiselect('Type of track', df['Type of track'].unique(), default=df['Type of track'].unique())
travelers_filter = st.sidebar.multiselect('Travelers per day', df['Travelers per day'].unique(), default=df['Travelers per day'].unique())
urban_filter = st.sidebar.multiselect('Urban/Regional/Suburban', df['Urban/Regional/Suburban'].unique(), default=df['Urban/Regional/Suburban'].unique())
safety_filter = st.sidebar.multiselect('Safety System', df['Safety System'].unique(), default=df['Safety System'].unique())
detection_filter = st.sidebar.multiselect('Detection system', df['Detection system'].unique(), default=df['Detection system'].unique())
emplacement_filter = st.sidebar.multiselect('Emplacement', df['Emplacement'].unique(), default=df['Emplacement'].unique())
number_of_tracks_filter = st.sidebar.multiselect('Number of tracks', df['Number of tracks'].unique(), default=df['Number of tracks'].unique())
geocode_exact = st.sidebar.text_input('Geocode', '')

# Helper function to create filters (unchanged)
def create_numeric_filter(column_name, multiplier=1):
    min_val = int(df[column_name].min() * multiplier)
    max_val = int(df[column_name].max() * multiplier)
    if min_val == max_val:
        return st.sidebar.text_input(f'Exact {column_name}', value=str(min_val / multiplier)), None
    else:
        return st.sidebar.text_input(f'Exact {column_name}', ''), st.sidebar.slider(f'{column_name}', min_val, max_val, (min_val, max_val))

# Specific columns for sliders (unchanged)
slider_columns = [
    'Track length (km)', 'km track', 'ATB beacon', 'Axle counters', 'Balise', 'Board signal', 'Crossing', 'Level Crossing',
    'Light signal', 'Matrix signal', 'Stations', 'Switches', 'Track current sections', 'Railway Viaduct',
    'Viaduct', 'Railway Bridge', 'Traffic Bridge', 'Railway Tunnel', 'Ecoduct',
    'Peat', 'Sand', 'Loamy sand', 'Sandy clay loam', 'Light clay', 'Heavy clay',
    'Loam', 'Sand combination', 'Clay combination', 'Urban area'
]

# Numeric filters with ranges and exact values (unchanged)
filters = {}
for column in slider_columns:
    exact, slider = create_numeric_filter(column, 1 if column not in ['Peat', 'Sand', 'Loamy sand', 'Sandy clay loam', 'Light clay', 'Heavy clay', 'Sand combination', 'Clay combination', 'Urban area'] else 100)
    filters[column] = (exact, slider)

# Apply filters (unchanged)
filtered_df = df[(df['ERTMS in 2031'].isin(ertms_filter)) &
                 (df['Type of track'].isin(track_type_filter)) &
                 (df['Travelers per day'].isin(travelers_filter)) &
                 (df['Urban/Regional/Suburban'].isin(urban_filter)) &
                 (df['Safety System'].isin(safety_filter)) &
                 (df['Detection system'].isin(detection_filter)) &
                 (df['Emplacement'].isin(emplacement_filter)) &
                 (df['Number of tracks'].isin(number_of_tracks_filter))]

# Apply numeric filters (unchanged)
def apply_numeric_filter(df, column_name, exact_value, range_value, multiplier=1):
    if exact_value:
        df = df[df[column_name] == float(exact_value) / multiplier]
    elif range_value:
        df = df[df[column_name].between(range_value[0] / multiplier, range_value[1] / multiplier)]
    return df

for column, (exact_value, range_value) in filters.items():
    filtered_df = apply_numeric_filter(filtered_df, column, exact_value, range_value, 1 if column not in ['Peat', 'Sand', 'Loamy sand', 'Sandy clay loam', 'Light clay', 'Heavy clay', 'Sand combination', 'Clay combination', 'Urban area'] else 100)

# Apply exact filter for Geocode (unchanged)
if geocode_exact:
    filtered_df = filtered_df[filtered_df['Geocode'] == geocode_exact]

# Display the results (unchanged)
st.title('Filtered Train Track Sections')
st.markdown("This dashboard allows the user to filter train track sections based on the filter options on the left side of the dashboard."
            "The table shows which track sections match the chosen criteria.")
st.write(f"Number of tracks matching criteria: {filtered_df.shape[0]}")
st.write(filtered_df)

# Calculate the percentage of tracks matching the criteria (unchanged)
total_tracks_count = df.shape[0]
filtered_tracks_count = filtered_df.shape[0]
percentage_matching_tracks = (filtered_tracks_count / total_tracks_count) * 100
st.write(f"Percentage of tracks matching criteria: {percentage_matching_tracks:.2f}%")

# Calculate the total km of tracks (unchanged)
total_km_tracks = df['Track length (km)'].sum()
filtered_km_tracks = filtered_df['Track length (km)'].sum()
percentage_matching_km_tracks = (filtered_km_tracks / total_km_tracks) * 100

st.write(f"Total km of tracks: {total_km_tracks:.2f} km")
st.write(f"Km of tracks matching criteria: {filtered_km_tracks:.2f} km")
st.write(f"Percentage of km tracks matching criteria: {percentage_matching_km_tracks:.2f}%")

# Graph selection (unchanged)
st.title('Visualization Options')
graph_options = st.multiselect(
    'Select the graphs you want to see:',
    ['Pie Chart (Count)', 'Pie Chart (KM)', 'Mean Train Track Section', 'Correlation Matrix', 'Histograms for Distribution']
)

# Pie chart for track count (unchanged)
if 'Pie Chart (Count)' in graph_options:
    st.title('Distribution of Matching Tracks (Count)')
    st.markdown("The pie chart shows the number of tracks that match the user-specified criteria")
    fig1, ax1 = plt.subplots()
    ax1.pie([filtered_tracks_count, total_tracks_count - filtered_tracks_count],
            labels=['Matching', 'Not Matching'], autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    st.pyplot(fig1)

# Pie chart for km track length (unchanged)
if 'Pie Chart (KM)' in graph_options:
    st.title('Distribution of Matching Tracks (KM)')
    st.markdown("The pie chart shows the kilometer of track that match the user-specified criteria")
    fig2, ax2 = plt.subplots()
    ax2.pie([filtered_km_tracks, total_km_tracks - filtered_km_tracks],
            labels=['Matching', 'Not Matching'], autopct='%1.1f%%', startangle=90)
    ax2.axis('equal')
    st.pyplot(fig2)

# Mean Train Track Section (unchanged)
if 'Mean Train Track Section' in graph_options:
    st.title('Mean Train Track Section')
    st.markdown("""
## Mean Track Results

The mean track results provide a summary of the average values for both numerical and non-numerical features of the train tracks. This information helps in understanding the typical characteristics of the track sections under consideration.
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

# Correlation Matrix of Numerical Features (unchanged)
if 'Correlation Matrix' in graph_options:
    st.title('Correlation Matrix of Numerical Features')
    st.markdown("""
A correlation matrix is a table showing correlation coefficients between sets of variables. Each cell in the table shows the correlation between two variables. The value is between -1 and 1.
""")
    fig4, ax4 = plt.subplots(figsize=(15, 15))  # Increase the figure size
    corr_matrix = df[numerical_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax4, annot_kws={"size": 8})  # Adjust font size
    ax4.set_title('Correlation Matrix')
    st.pyplot(fig4)

# Histograms for Distribution of Numerical Features (unchanged)
if 'Histograms for Distribution' in graph_options:
    st.title('Histograms for Distribution of Numerical Features')
    st.markdown("""
## Histograms for Distribution of Numerical Features

Histograms provide a visual representation of the distribution of numerical features in the dataset. They show how the data points are spread across different values, which helps in understanding the underlying patterns and distributions of the data.
""")
    fig5, axes = plt.subplots(nrows=len(numerical_cols), ncols=1, figsize=(10, len(numerical_cols) * 4))
    plt.subplots_adjust(hspace=0.5)
    for col, ax in zip(numerical_cols, axes):
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f'Distribution of {col}')
    st.pyplot(fig5)

# Download button for Mean Train Track Section Excel file
if 'Mean Train Track Section' in graph_options:
    st.title('Download Mean Track Section Data')
    st.markdown("You can download the mean track section data as an Excel file.")
    
    # Prepare the Excel file in memory
    output = BytesIO()
    mean_track_section.to_frame('Mean Value').to_excel(output, sheet_name='Mean Track Section', index=True)
    output.seek(0)  # Move cursor to the start of the BytesIO stream
    
    # Create a download button
    st.download_button(
        label="Download Mean Track Section Data",
        data=output,
        file_name='Mean_Track_Section.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
