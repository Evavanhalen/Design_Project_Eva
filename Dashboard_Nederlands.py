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
file_path = 'Nederlands_Template (1).xlsx'
logo_path = 'ProRail logo.png'
pdf_file_path = 'Gebruikershandleiding.pdf'

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
descriptive_columns = ['Baanvak', 'Geocode', 'Van', 'Tot']

# Sidebar for column selection using checkboxes
st.sidebar.title('Opnemen/Uitsluiten bepaalde kolommen en filters')

# Dictionary to hold the inclusion state and filter values
column_inclusion = {}

# Create checkboxes for each column to include or exclude
for column in df.columns:
    if column in descriptive_columns:
        # Skip descriptive columns from having checkboxes
        continue
    
    # Check the data type to decide on multiselect or slider
    if pd.api.types.is_numeric_dtype(df[column]):
        include_column = st.sidebar.checkbox(f"Openemen {column}", value=True, key=f"{column}_include")
        if include_column:
            min_val = df[column].min()
            max_val = df[column].max()
            filter_values = st.sidebar.slider(f'{column}', min_val, max_val, (min_val, max_val), key=f"{column}_filter")
            column_inclusion[column] = (include_column, filter_values)
    else:
        include_column = st.sidebar.checkbox(f"Opnemen {column}", value=True, key=f"{column}_include")
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

st.title('Baanvakanalyse')            
st.markdown("Dit is een interactief Dashboard dat verschillende treinbaanvakken in Nederland laat zien. De gebruiker kan bepaalde kenmerken in- en uitsluiten"
            "van elke statistische analyse en filteren op basis van numerieke of niet-numerieke waarden. De gebruiker kan het type visualisatie selecteren dat hij wil zien. Voor meer gedetailleerde informatie over het dashbaord kunt u hieronder de gebruikershandleiding downloaden.")

with open(pdf_file_path, "rb") as f:
    pdf_data = f.read()

# Create a download button
st.download_button(
    label="📕Download Gebruikershandleiding",
    data=pdf_data,
    file_name="User_Guide.pdf",
    mime="application/pdf"
)

# Main content
with st.expander("🗺️ Klik hier om de kaart van de baanvakken te zien"):
    st.subheader('Kaart van de Baanvakken')
    # Load and display the ProRail logo
    map_path = '67.png'
    map = Image.open(map_path)
    st.image(map, use_column_width=True)


st.markdown("""
    <h1 style='font-size:2.5em; color:navy;'>Filteren op baanvak eigenschappen</h1>
    <hr style='border:2px solid navy;'>
    """, unsafe_allow_html=True)
st.markdown("Met dit dashboard kan de gebruiker treinbaanvakken filteren op basis van de filteropties aan de linkerkant van het dashboard. De tabel laat zien welke baanvakken voldoen aan de gekozen criteria.")
with st.expander("🔎 Klik hier om de gefilterde baanvakken te zien"):
    st.write(f"Aantal baanvakken die overeenkomen met de criteria: {filtered_df.shape[0]}")
    st.write(filtered_df)


total_tracks_count = df.shape[0]
filtered_tracks_count = filtered_df.shape[0]
percentage_matching_tracks = (filtered_tracks_count / total_tracks_count) * 100

total_km_tracks = df['km spoor'].sum()
filtered_km_tracks = filtered_df['km spoor'].sum()
percentage_matching_km_tracks = (filtered_km_tracks / total_km_tracks) * 100

# Assuming track_length is a column in your DataFrame
total_track_length = df['Baanvaklengt (km)'].sum()
filtered_track_length = filtered_df['Baanvaklengt (km)'].sum()
percentage_matching_track_length = (filtered_track_length / total_track_length) * 100

# Visualization Options
st.subheader('Visualisatie Opties')
graph_options = st.multiselect(
    'Selecteer de grafieken die je wilt zien:',
    ['Taartdiagram (Aantal)', 'Taartdiagram (KM/Lengte)', 'Gemiddeld Baanvak']
)

# Toggle between Track Length and Track KM for Pie Chart
track_measurement = st.radio(
    "Selecteer de eenheid voor het tweede taartdiagram:",
    ('Baanvak kilometers', 'Baanvak lengte'),
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

if 'Taartdiagram (Aantal)' in graph_options:
    with col1:
        st.subheader('Verdeling van overeenkomende baanvakken (aantal)')
        st.markdown("Het taartdiagram toont het aantal baanvakken dat voldoet aan de door de gebruiker opgegeven criteria")        
        fig1, ax1 = plt.subplots(figsize=(4, 4))  # Adjust the size as needed
        ax1.pie([filtered_tracks_count, total_tracks_count - filtered_tracks_count],
                labels=['Overeenkomend', 'Niet overeenkomend'], autopct='%1.1f%%', startangle=90)
        ax1.axis('equal')
        st.pyplot(fig1)
        st.write(f"Totaal aantal baanvakken: {total_tracks_count}")
        st.write(f"Aantal baanvakken wat overeenkomt met filteringscriteria: {filtered_tracks_count}")
        st.write(f"Percentage baanvakken dat overeenkomt met filteringscriteria: {percentage_matching_tracks:.2f}%")

# Conditional display for Pie chart (KM/Length)
if 'Taartdiagram (KM/Lengte)' in graph_options:
    with col2:
        if track_measurement == 'Track Kilometers':
            st.subheader('Verdeling van overeenkomende baanvakken (km)')
            st.markdown("Het taartdiagram toont het aantal baanvakken dat voldoet aan de door de gebruiker opgegeven criteria")        
            fig2, ax2 = plt.subplots(figsize=(4, 4))  # Adjust the size as needed
            ax2.pie([filtered_km_tracks, total_km_tracks - filtered_km_tracks],
                    labels=['Overeenkomend', 'Niet Overeenkomend'], autopct='%1.1f%%', startangle=90)
            ax2.axis('equal')
            st.pyplot(fig2)
            st.write(f"Totaal aantal spoorkilometers: {total_km_tracks:.2f} km")
            st.write(f"Kilometers baanvakken dat overeenkomt met de criteria: {filtered_km_tracks:.2f} km")
            st.write(f"Percentage kilometeres dat overeenkomt met de criteria: {percentage_matching_km_tracks:.2f}%")
        else:
            st.subheader('Verdeling van overeenkomende baanvakken (Baanvak lengte)')
            st.markdown("Het taartdiagram toont het aantal baanvakken dat voldoet aan de door de gebruiker opgegeven criteria")        
            fig2, ax2 = plt.subplots(figsize=(4, 4))  # Adjust the size as needed
            ax2.pie([filtered_track_length, total_track_length - filtered_track_length],
                    labels=['Overeenkomend', 'Niet overeenkomend'], autopct='%1.1f%%', startangle=90)
            ax2.axis('equal')
            st.pyplot(fig2)
            st.write(f"Totale baanvaklengte: {total_track_length:.2f} units")
            st.write(f"Baanvaklengte die overeenkomt met de criteria: {filtered_track_length:.2f} units")
            st.write(f"Percentage baanvaklengte die overeenkomt met de criteria: {percentage_matching_track_length:.2f}%")


# Mean Train Track Section
if 'Gemiddeld Baanvak' in graph_options:
    st.subheader('Gemiddeld Baanvak')
    with st.expander("📖 Klik hier voor een gedetailleerde uitleg van het gemiddelde baanvak"):
        st.markdown("""
    ## Gemiddeld Baanvak

    In deze resultaten is een samenvatting te vinden van de gemiddelde waardes voor de numeriek en niet-numerieke eigenschappen van een baanvak.

    **Numerieke kolommen:**
    - Deze kolommen bevatten numerieke gegevens zoals spoorlengte, aantal seinen, enz.
    - Voor elke numerieke kolom wordt de gemiddelde waarde berekend.
    - Dit geeft een idee van de centrale tendens van de numerieke kenmerken in de dataset.

    **Niet-numerieke kolommen:**
    - Deze kolommen bevatten categorische gegevens zoals het type spoor, veiligheidssysteem, enz.
    - De modus (meest frequente waarde) wordt berekend voor elke niet-numerieke kolom.
    - Dit geeft inzicht in de meest voorkomende categorieën of attributen in de dataset.

    ### Belangrijke aandachtspunten:

    **Numerieke kolommen:**
    - Gemiddelde waarde**: Geeft de gemiddelde waarde van de numerieke kenmerken weer. Deze wordt berekend door alle waarden in een kolom op te tellen en te delen door het aantal waarden.

- **Interpretatie**: De gemiddelde waarde biedt een centrale waarde waarrond de gegevenspunten zijn verdeeld.

**Niet-numerieke kolommen:**
- **Waarde**: Vertegenwoordigt de meest voorkomende waarde of categorie in de niet-numerieke kenmerken.- **Interpretatie**: De moduswaarde helpt bij het identificeren van de meest voorkomende categorie binnen de dataset.    ### Visualisatie en analyse:

    - **Balkdiagrammen voor numerieke kolommen**: Visuele weergaven van de gemiddelde waarden voor numerieke kolommen helpen bij het eenvoudig vergelijken van de gemiddelde waarden over verschillende kenmerken.
    - **Tabellen voor niet-numerieke kolommen**: Door de gemiddelden in tabelvorm weer te geven, krijgt u een duidelijk beeld van de meest voorkomende categorieën.
        """)
    # Filter the numerical columns based on the selected columns
    numerical_cols = filtered_df.select_dtypes(include=[float, int]).columns
    mean_numerical_values = filtered_df[numerical_cols].mean()
    
# Create two columns for side-by-side display
    col1, col2 = st.columns(2)

    # Plotting the mean values for numerical columns
    with col1:
        st.subheader('Numerieke Kolommen')
        fig, ax = plt.subplots()
        mean_numerical_values.plot(kind='bar', ax=ax)
        ax.set_ylabel('Gemiddelde Waarde')
        ax.set_title('Gemiddelde Waardes van Numerieke Kolommen')
        st.pyplot(fig)

    # Displaying the mode values in a table format for non-numerical columns
    with col2:
        st.subheader('Categorische Kolommen')
        st.table(mode_non_numerical_values)

        # Button to display histograms
    with st.expander('📊 Klik hier voor inzichten in de statistische verdelingen'):
        st.subheader('Histogrammen voor verdelingen van numerieke kenmerken')
        st.markdown("""
    Histogrammen geven een visuele voorstelling van de verdeling van numerieke kenmerken in de dataset. Ze laten zien hoe de gegevenspunten verdeeld zijn over verschillende waarden, wat helpt om de onderliggende patronen en verdelingen van de gegevens te begrijpen.        """)

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
    st.write(f"Top 10 baanvakken die het meest lijken op het {section_type} gemiddelde baanvak")
    st.write(similar_tracks[['Baanvak', 'Gelijkenis'] + list(numerical_cols) + list(non_numerical_cols)])
    df.drop(columns=['Gelijkenis'], inplace=True)  # Clean up

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
    st.subheader('Download naar Excel')
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        mean_track_section.to_excel(writer, sheet_name='Gemiddeld Baanvak')
    output.seek(0)
    st.download_button(
        label="Download gemiddeld baanvakdata naar Excel",
        data=output,
        file_name="Gemiddel_Baanvak.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

with col2:
    st.subheader('Vind een echt baanvak als match')
    if st.button('Gemiddeld baanvak in echte baanvakken'):
        display_similar_tracks(df, mean_track_section, included_numerical_cols, included_non_numerical_cols, 'Mean')


# Add a title and description
st.markdown("""
    <h1 style='font-size:2.4em; color:darkgreen;'> Stedelijk/Voorstedelijk/Regionaal Baanvakken </h1>
    <hr style='border:2px solid darkgreen;'>
    """, unsafe_allow_html=True)
st.markdown("""
    Met dit dashboard kun je verschillende kenmerken van baanvakken analyseren en visualiseren, ingedeeld in stedelijke, voorstedelijke en regionale typen.

    **Instructies:**
    1. Gebruik de zijbalk om specifieke kenmerken in de analyse op te nemen of uit te sluiten.
    2. Kies of je verplaatsingsgegevens wilt uitsluiten.
    3. Selecteer de typen plots die je wilt weergeven.
    4. Het dashboard biedt opties om gemiddelden, verdelingen en samenvattingen van numerieke en niet-numerieke kenmerken weer te geven.

    **Opmerking:** De gegevens worden gefilterd op basis van de selecties die je maakt in de zijbalk.
""")
# Visualization Options
st.subheader('Visualisatie Opties')
graph_options = st.multiselect(
    'Selecteer de grafieken die je wilt zien:',
    ['Numerieke Gemiddeldes per Categorie', 'Niet-Numerieke Modi per Categorie', 'Numerieke Samenvatting']
)
# Group by 'Urban/Regional/Suburban' and calculate mean and standard deviation for numerical features and most frequent value for non-numerical features
numerical_cols = filtered_df.select_dtypes(include=[float, int]).columns.difference(descriptive_columns)
non_numerical_cols = filtered_df.select_dtypes(exclude=[float, int]).columns.difference(descriptive_columns)

mean_numerical = filtered_df.groupby('Stedelijk/Voorstedelijk/Regionaal')[numerical_cols].mean()
mode_non_numerical = filtered_df.groupby('Stedelijk/Voorstedelijk/Regionaal')[non_numerical_cols].agg(lambda x: x.mode()[0])
grouped_stds = filtered_df.groupby('Stedelijk/Voorstedelijk/Regionaal')[numerical_cols].std()

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
            ax.errorbar(categories, means, yerr=errors, fmt='o', color='blue', capsize=5, label='Standaard Deviatie')
            ax.scatter(categories, means, color='red', zorder=5, label=f'{col} (Mean)')
            ax.set_title(f'Gemiddeldes van {col} per Stedelijk/Voorstedelijk/Regionaal Categorie')
            ax.set_ylabel(f'Gemiddelde {col}')
            ax.set_xlabel('Stedelijk/Voorstedelijk/Regionaal Categorie')
            ax.legend(loc='upper right')
            ax.tick_params(axis='x', rotation=45)

        # Remove any empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout(pad=3.1)
        fig.subplots_adjust(top=0.9)
        fig.suptitle(f'Gemiddeldes van Numerieke Kenmerken van {start_idx + 1} tot {end_idx} per Stedelijk/Voorstedelijk/Regionaal Categorie', fontsize=16)
        st.pyplot(fig)
        plt.close(fig)
        

if 'Numerieke Gemiddeldes per Categorie' in graph_options:
    plot_all_numerical_features(mean_numerical, grouped_stds, mean_numerical.index)

    # Add an expander for numerical distributions
    with st.expander("📊 Klik hier voor gedetailleerde numerieke verdelingen"):
        # Define the function to plot distributions
        def plot_distributions(columns, df, title, cols=3):
            num_plots = len(columns)
            rows = (num_plots // cols) + (num_plots % cols > 0)  # Calculate number of rows needed

            fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))
            axes = axes.flatten()

            for i, col in enumerate(columns):
                sns.boxplot(x='Stedelijk/Voorstedelijk/Regionaal', y=col, data=df, ax=axes[i])
                axes[i].set_title(f'Verdeling van {col}', fontsize=10, pad=10)
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
            plot_distributions(remaining_cols, filtered_df, 'Verdeling van overige numerieke kenmerken')

# Display mode of non-numerical columns by category
if 'Niet-Numerieke Modi per Categorie' in graph_options:
    st.subheader('Niet-Numerieke Kenmerken Modi')
    st.write("Hieronder staan de meest voorkomende waarden (modus) voor de niet-numerieke kenmerken in verschillende spoorcategorieën.")
    st.table(mode_non_numerical)

    # Add an expander for detailed non-numerical distributions
    with st.expander("📊 Klik hier voor gedetailleerde niet-numerieke functieverdelingen"):
        # Define the function to plot non-numerical distributions
        def plot_non_numerical_distributions(columns, df, title, cols=3):
            num_plots = len(columns)
            rows = (num_plots // cols) + (num_plots % cols > 0)  # Calculate number of rows needed

            fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))
            axes = axes.flatten()

            for i, col in enumerate(columns):
                sns.countplot(x='Stedelijk/Voorstedelijk/Regionaal', hue=col, data=df, ax=axes[i])
                axes[i].set_title(f'Verdeling van {col}', fontsize=10, pad=10)
                axes[i].set_xlabel('Stedelijk/Voorstedelijk/Regionaal Categorie', fontsize=8)
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
        plot_non_numerical_distributions(non_numerical_cols, filtered_df, 'Niet-numerieke kenmerkverdelingen')
# Visualization function for numerical data
def plot_numerical_summary(summary, title):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    categories = ['Stedelijk', 'Voorstedelijk', 'Regionaal']

    for i, category in enumerate(categories):
        summary.loc[category].plot(kind='bar', ax=axes[i])
        axes[i].set_title(f'Gemiddelde {category} Baanvak')
        axes[i].set_ylabel('Gemiddelde Waarde')
        axes[i].set_xlabel('Kenmerk')
        axes[i].tick_params(axis='x', labelsize=8, rotation=90)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    st.pyplot(fig)
    plt.close(fig)

if 'Numerieke Samenvatting' in graph_options:
    st.subheader('Samenvatting van Numerieke Kenmerken per Categorie')
    plot_numerical_summary(summary_numerical, 'Gemiddeld Stedelijk/Voorstedelijk/Regionaal Baanvak')

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
    st.write(f"Top 10 baanvakken die het meest lijken op het {section_type} gemiddelde baanvak")
    st.write(similar_tracks[['Baanvak', 'Gelijkenis'] + list(numerical_cols) + list(non_numerical_cols)])
    df.drop(columns=['Gelijkenis'], inplace=True)  # Clean up

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
    st.subheader('Download naar Excel')
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        summary_numerical.to_excel(writer, sheet_name='Numerical Features')
        summary_std.to_excel(writer, sheet_name='Standard Deviation')
        summary_non_numerical.to_excel(writer, sheet_name='Non-Numerical Features')
    output.seek(0)
    st.download_button(
        label="Download Samenvatting van de Stedelijke/Voorstedelijke/Regionale Baanvakken naar Excel",
        data=output,
        file_name="Categorieën_Samenvatting.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

with col2:
    st.subheader('Vind een levensechte match')
    if st.button('Stedelijk baanvak in echte baanvakken'):
        urban_mean = pd.concat([mean_numerical.loc['Stedelijk'], mode_non_numerical.loc['Stedelijk']])
        display_similar_tracks(df, urban_mean, included_numerical_cols, included_non_numerical_cols, 'Stedelijk')

    if st.button('Voorstedelijk baanvak in echte baanvakken'):
        suburban_mean = pd.concat([mean_numerical.loc['Voorstedelijk'], mode_non_numerical.loc['Voorstedelijk']])
        display_similar_tracks(df, suburban_mean, included_numerical_cols, included_non_numerical_cols, 'Voorstedelijk')

    if st.button('Regionaal baanvak in echte baanvakken'):
        regional_mean = pd.concat([mean_numerical.loc['Regionaal'], mode_non_numerical.loc['Regionaal']])
        display_similar_tracks(df, regional_mean, included_numerical_cols, included_non_numerical_cols, 'Regionaal')



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
    <h1 style='font-size:2.5em; color:darkred;'>K-Clustering van Baanvakken</h1>
    <hr style='border:2px solid darkred;'>
    """, unsafe_allow_html=True)
st.markdown("Het k-means clusteralgoritme wordt toegepast op de voorbewerkte gegevens. K-means clustering heeft als doel n observaties te verdelen in k clusters waarin elke observatie behoort tot het cluster met het dichtstbijzijnde gemiddelde, dat dient als prototype van het cluster. Het k-means algoritme minimaliseert de WCSS (Within-Cluster Sum of Square), ook bekend als de traagheid.")

st.subheader('Visualisatie Opties')
graph_options = st.multiselect(
    'Selecteer de grafieken die je wilt zien:',
    ['3D PCA', 'Taartdiagram']
)

numerical_cols = [col for col, (include, _) in column_inclusion.items() if include and pd.api.types.is_numeric_dtype(df[col])]

# After applying filters, check the number of samples
if filtered_df.shape[0] == 0:
    st.warning("Geen gegevens beschikbaar na het toepassen van filters. Pas de filters aan.")
else:
    # Define the number of clusters, k
    default_k = 5  # Default number of clusters
    k = min(default_k, filtered_df.shape[0])  # Set k to the default or the number of samples, whichever is smaller
    
    if filtered_df.shape[0] < k:
        st.warning(f"Niet genoeg gegevens voor {k} clusters. Alleen {filtered_df.shape[0]} monsters beschikbaar. Aantal clusters aanpassen naaro {filtered_df.shape[0]}.")
        k = filtered_df.shape[0]
    
    # Ensure numerical_data is defined and valid
    numerical_data = filtered_df[numerical_cols]

    # Convert numerical_data to a NumPy array for the checks
    numerical_data_array = numerical_data.to_numpy()

    # Ensure there's no NaN or infinite values before scaling
    if np.isnan(numerical_data_array).any() or np.isinf(numerical_data_array).any():
        st.error("Gefilterde gegevens bevatten NaN of oneindige waarden. Controleer de gegevens of pas de filters aan.")
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
if '3D PCA' in graph_options or 'Taartdiagram' in graph_options:
    col1, col2 = st.columns(2)
    with st.expander("📖 Klik hier voor uitleg over de visualisaties"):
        st.markdown("""
## Uitleg van visualisaties

        **3D PCA-plot**:
        - Deze plot toont de clusters die in de gegevens zijn gevonden met k-means clustering.
        - Elk punt stelt een deel van het treinspoor voor.
        - De drie assen (PC1, PC2, PC3) zijn de eerste drie principale componenten, die nieuwe kenmerken zijn om de gegevens samen te vatten.
        - Punten die dicht bij elkaar liggen, lijken op elkaar wat betreft de geselecteerde kenmerken.
        - Verschillende kleuren vertegenwoordigen verschillende clusters.

        **Taartdiagram**:
        - Deze grafiek toont de verdeling van de clusters.
        - Elk taartpunt staat voor een cluster.
        - De grootte van elk taartpunt geeft aan hoeveel delen van het treinspoor tot dat cluster behoren.
        """)

    with st.expander("📖 Klik hier voor een uitleg van het k-clusteralgoritme"):
        st.markdown("""
 ## Uitleg van het clusteralgoritme

        **K-Means clustering**:
        - Het k-means clusteralgoritme groepeert de gegevens in clusters.
        - Elk gegevenspunt wordt toegewezen aan het dichtstbijzijnde clustermiddelpunt, een centroïde genoemd.
        - Het algoritme probeert de afstand tussen gegevenspunten en hun respectieve centroïde te minimaliseren.
        - Op deze manier lijken gegevenspunten binnen hetzelfde cluster op elkaar.
        - In deze analyse heeft het algoritme de treinspoorsecties in 5 clusters gegroepeerd.
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

            ax.set_title('3D PCA van de Clusters', fontsize=10)
            ax.set_xlabel('PC1', fontsize=6)
            ax.set_ylabel('PC2', fontsize=6)
            ax.set_zlabel('PC3', fontsize=6)
            ax.legend(fontsize=6)
            st.pyplot(fig)

    if 'Taartdiagram' in graph_options:
        with col2:
            cluster_counts = filtered_df['Cluster'].value_counts()

            fig, ax = plt.subplots(figsize=(4, 4), dpi=200)  # Reduced size and high DPI for better quality
            ax.pie(cluster_counts, labels=cluster_counts.index, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 6})
            ax.set_title('Cluster Verdeling', fontsize=10)
            st.pyplot(fig)


cluster_analysis = filtered_df.groupby('Cluster')[numerical_cols].mean()
non_numerical_cols_for_analysis = non_numerical_cols.difference(descriptive_columns)
non_numerical_analysis = filtered_df.groupby('Cluster')[non_numerical_cols_for_analysis].agg(lambda x: x.value_counts().index[0])

# Column Layout for the interactive elements
col1, col2 = st.columns([2, 3])

# Column Layout for the interactive elements
col1, col2 = st.columns([2, 3])

with col1:
    st.subheader('Download naar Excel')
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        if not cluster_analysis.empty:
            cluster_analysis.to_excel(writer, sheet_name='Cluster_Samenvatting')
        if not non_numerical_analysis.empty:
            non_numerical_analysis.to_excel(writer, sheet_name='Niet_Numerieke_Samenvatting')
    output.seek(0)
    st.download_button(
        label="Download Samenvatting van K-Means Clusters naar Excel",
        data=output,
        file_name="K_Means_Clusters.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

with col2:
    st.subheader('Vind een levenechte match')
    # Loop over the existing clusters instead of assuming indices
    for i in cluster_analysis.index:
        if i in non_numerical_analysis.index:
            cluster_mean = pd.concat([cluster_analysis.loc[i], non_numerical_analysis.loc[i]])
            if st.button(f'Cluster {i} in echte baanvakken'):
                display_similar_tracks(df, cluster_mean, included_numerical_cols, included_non_numerical_cols, f'Cluster {i}')
        else:
            st.warning(f"Cluster {i} not found due to filtering in left menu. Remove some filters to obtain up to 5 clusters.")

    
