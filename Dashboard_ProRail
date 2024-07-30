## This is the overall dashboard which combines the Filtered_Categories.py and Interface_Filtering.py
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from PIL import Image
from io import BytesIO

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
    ['Display Numerical Means by Category', 'Display Numerical Distributions', 'Display Non-Numerical Distributions', 'Display Numerical Summary', 'Display Non-Numerical Summary Heatmap with Percentages', 'Pie Chart (Count)', 'Pie Chart (KM)', 'Mean Train Track Section', 'Correlation Matrix', 'Histograms for Distribution']
)

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

# Display Visualization Options
if 'Display Numerical Means by Category' in graph_options:
    st.write("**Mean Values by Category**")
    mean_values = filtered_df.groupby('Urban/Regional/Suburban').mean()
    std_values = filtered_df.groupby('Urban/Regional/Suburban').std()
    categories = filtered_df['Urban/Regional/Suburban'].unique()
    plot_all_numerical_features(mean_values, std_values, categories, group_size=6)

# Function to display numerical summaries
def plot_numerical_summary(df):
    st.write("**Summary of Numerical Features**")
    st.write(df.describe())

if 'Display Numerical Summary' in graph_options:
    plot_numerical_summary(filtered_df)

# Function to display histograms for distributions
def plot_histograms(df):
    st.write("**Histograms for Distributions**")
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    for column in numerical_columns:
        plt.figure(figsize=(10, 4))
        sns.histplot(df[column], kde=True, bins=30)
        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        st.pyplot()
        plt.close()

if 'Histograms for Distribution' in graph_options:
    plot_histograms(filtered_df)

# Excel export function
def export_to_excel(df, filtered=True):
    with BytesIO() as buffer:
        writer = pd.ExcelWriter(buffer, engine='xlsxwriter')
        df.to_excel(writer, index=False, sheet_name='Filtered Summary' if filtered else 'Full Summary')
        writer.save()
        return buffer.getvalue()

# Button to download filtered data
if st.button('Download Filtered Summary as Excel'):
    filtered_file = export_to_excel(filtered_df, filtered=True)
    st.download_button(
        label='Download Filtered Summary',
        data=filtered_file,
        file_name='Filtered_Summary.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

# Button to download full data
if st.button('Download Full Summary as Excel'):
    full_file = export_to_excel(df, filtered=False)
    st.download_button(
        label='Download Full Summary',
        data=full_file,
        file_name='Full_Summary.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
