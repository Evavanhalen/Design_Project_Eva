## Dit is het algemene dashboard dat Filtered_Categories.py en Interface_Filtering.py combineert
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

# Onderdruk specifieke waarschuwingen
warnings.filterwarnings("ignore", category=FutureWarning, module='seaborn._oldcore')

# Definieer bestandspaden met relatieve paden
file_path = 'Nederlands_Template.xlsx'
logo_path = 'ProRail logo.png'

# Laad de gegevens uit de template
df = pd.read_excel(file_path, sheet_name='Template')

# Laad en toon het ProRail-logo
logo = Image.open(logo_path)

# Toon het logo in de zijbalk
st.sidebar.image(logo, use_column_width=True)

# Pas aangepaste CSS toe
st.markdown("""
    <style>
    .stButton button {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# Beschrijvende kolommen die altijd moeten worden opgenomen
descriptive_columns = ['Baanvak', 'Geocode', 'Van', 'Tot']

# Zijbalk voor kolomselectie met selectievakjes
st.sidebar.title('Kolommen en Filters In- of Uitschakelen Bla Bla')

# Woordenboek om de inclusiestatus en filterwaarden op te slaan
column_inclusion = {}

# Maak selectievakjes voor elke kolom om in of uit te schakelen
for column in df.columns:
    if column in descriptive_columns:
        # Sla beschrijvende kolommen over van het hebben van selectievakjes
        continue
    
    # Controleer het datatype om te beslissen over multiselect of schuifregelaar
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

# Begin met beschrijvende kolommen altijd inbegrepen
filtered_df = df[descriptive_columns].copy()

# Pas de filter- en kolominclusielogica toe
for column, (include, filter_values) in column_inclusion.items():
    if include:
        if pd.api.types.is_numeric_dtype(df[column]):
            min_val, max_val = filter_values
            filtered_df = filtered_df.join(df[df[column].between(min_val, max_val)][[column]], how='inner')
        else:
            filtered_df = filtered_df.join(df[df[column].isin(filter_values)][[column]], how='inner')

st.title('Analyse van Treinsecties in het Nederlands')            
st.markdown("Dit is een interactief dashboard dat verschillende treinsecties in Nederland presenteert. De gebruiker kan bepaalde kenmerken in- of uitschakelen"
            "voor elke statistische analyse en filteren op basis van numerieke of niet-numerieke waarden. De gebruiker kan het type visualisatie selecteren dat hij wil zien.")
# Hoofdinhoud
with st.expander("🗺️ Klik hier om de kaart van treinsecties te bekijken"):
    st.subheader('Kaart van Treinsecties')
    # Laad en toon het ProRail-logo
    map_path = '67.png'
    map = Image.open(map_path)
    st.image(map, use_column_width=True)


st.markdown("""
    <h1 style='font-size:2.5em; color:navy;'>Filtersectie</h1>
    <hr style='border:2px solid navy;'>
    """, unsafe_allow_html=True)
st.markdown("Dit dashboard stelt de gebruiker in staat om treinsecties te filteren op basis van de filteropties aan de linkerkant van het dashboard. De tabel toont welke treinsecties voldoen aan de gekozen criteria.")

with st.expander("🔎 Klik hier om de gefilterde treinsecties te bekijken"):
    st.write(f"Aantal sporen die voldoen aan de criteria: {filtered_df.shape[0]}")
    st.write(filtered_df)


total_tracks_count = df.shape[0]
filtered_tracks_count = filtered_df.shape[0]
percentage_matching_tracks = (filtered_tracks_count / total_tracks_count) * 100

total_km_tracks = df['km spoor'].sum()
filtered_km_tracks = filtered_df['km spoor'].sum()
percentage_matching_km_tracks = (filtered_km_tracks / total_km_tracks) * 100

# Visualisatieopties
st.subheader('Visualisatieopties')
graph_options = st.multiselect(
    'Selecteer de grafieken die je wilt zien:',
    ['Taartdiagram (Aantal)', 'Taartdiagram (KM)', 'Gemiddelde Treinsectie']
)

# Bereken de gemiddelde treinsectie
numerical_cols = df.select_dtypes(include=[float, int]).columns.difference(descriptive_columns)
non_numerical_cols = df.select_dtypes(exclude=[float, int]).columns.difference(descriptive_columns)
mean_numerical_values = df[numerical_cols].mean()
mode_non_numerical_values = df[non_numerical_cols].mode().iloc[0]
mean_track_section = pd.concat([mean_numerical_values, mode_non_numerical_values])

# Filter kolommen naar relevante kenmerken, laat spoornummers, geocodes en namen weg
relevant_columns = df.loc[:, 'Emplacement':]

# Maak twee kolommen
col1, col2 = st.columns(2)

if 'Taartdiagram (Aantal)' in graph_options:
    with col1:
        st.subheader('Verdeling van Overeenkomende Spoorsecties (Aantal)')
        st.markdown("Het taartdiagram toont het aantal sporen dat voldoet aan de door de gebruiker opgegeven criteria")
        fig1, ax1 = plt.subplots(figsize=(4, 4))  # Pas de grootte aan indien nodig
        ax1.pie([filtered_tracks_count, total_tracks_count - filtered_tracks_count],
                labels=['Overeenkomend', 'Niet Overeenkomend'], autopct='%1.1f%%', startangle=90)
        ax1.axis('equal')
        st.pyplot(fig1)
        st.write(f"Totaal aantal sporen: {total_tracks_count}")
        st.write(f"Sporen die voldoen aan de criteria: {filtered_tracks_count}")
        st.write(f"Percentage dat voldoet aan de criteria: {percentage_matching_tracks:.2f}%")

# Voorwaardelijke weergave voor Taartdiagram (KM)
if 'Taartdiagram (KM)' in graph_options:
    with col2:
        st.subheader('Verdeling van Overeenkomende Spoorsecties (KM)')
        st.markdown("Het taartdiagram toont het aantal kilometer spoor dat voldoet aan de door de gebruiker opgegeven criteria")
        fig2, ax2 = plt.subplots(figsize=(4, 4))  # Pas de grootte aan indien nodig
        ax2.pie([filtered_km_tracks, total_km_tracks - filtered_km_tracks],
                labels=['Overeenkomend', 'Niet Overeenkomend'], autopct='%1.1f%%', startangle=90)
        ax2.axis('equal')
        st.pyplot(fig2)
        st.write(f"Totaal aantal kilometer spoor: {total_km_tracks:.2f} km")
        st.write(f"Kilometer spoor dat voldoet aan de criteria: {filtered_km_tracks:.2f} km")
        st.write(f"Percentage van kilometers spoor dat voldoet aan de criteria: {percentage_matching_km_tracks:.2f}%")

# Gemiddelde Treinsectie
if 'Gemiddelde Treinsectie' in graph_options:
    st.subheader('Gemiddelde Treinsectie')
    with st.expander("📖 Klik hier voor een gedetailleerde uitleg van de gemiddelde treinsectie resultaten"):
        st.markdown("""
    ## Resultaten van de Gemiddelde Treinsectie

    De resultaten van de gemiddelde treinsectie geven een overzicht van de gemiddelde waarden voor zowel numerieke als niet-numerieke kenmerken van de treinsporen. Deze informatie helpt bij het begrijpen van de typische kenmerken van de beschouwde spoorsecties.

    **Numerieke Kolommen:**
    - Deze kolommen bevatten numerieke gegevens zoals spoorlengte, aantal signalen, enz.
    - De gemiddelde waarde wordt voor elke numerieke kolom berekend.
    - Dit geeft een idee van de centrale tendens van de numerieke kenmerken in de dataset.

    **Niet-Numerieke Kolommen:**
    - Deze kolommen bevatten categorische gegevens zoals het type spoor, veiligheidssysteem, enz.
    - De modus (meest voorkomende waarde) wordt voor elke niet-numerieke kolom berekend.
    - Dit geeft inzicht in de meest voorkomende categorieën of kenmerken in de dataset.

    ### Belangrijke Punten om te Overwegen:

    **Numerieke Kolommen:**
    - **Gemiddelde Waarde**: Vertegenwoordigt de gemiddelde waarde van de numerieke kenmerken. Het wordt berekend door alle waarden in een kolom op te tellen en te delen door het aantal waarden.
    - **Interpretatie**: De gemiddelde waarde biedt een centrale waarde waar de gegevenspunten omheen zijn verdeeld.

    **Niet-Numerieke Kolommen:**
    - **Modus Waarde**: Vertegenwoordigt de meest voorkomende waarde of categorie in de niet-numerieke kenmerken.
    - **Interpretatie**: De modus waarde helpt bij het identificeren van de meest voorkomende categorie binnen de dataset.

    ### Visualisatie en Analyse:

    - **Staafdiagrammen voor Numerieke Kolommen**: Visuele weergaven van de gemiddelde waarden voor numerieke kolommen helpen bij het gemakkelijk vergelijken van de gemiddelde waarden tussen verschillende kenmerken.
    - **Tabellen voor Niet-Numerieke Kolommen**: Het weergeven van de modus waarden in een tabelformaat zorgt voor een duidelijk begrip van de meest voorkomende categorieën.
        """)
    # Filter de numerieke kolommen op basis van de geselecteerde kolommen
    numerical_cols = filtered_df.select_dtypes(include=[float, int]).columns
    mean_numerical_values = filtered_df[numerical_cols].mean()
    
# Maak twee kolommen voor zij-aan-zij weergave
    col1, col2 = st.columns(2)

    # Plot de gemiddelde waarden voor numerieke kolommen
    with col1:
        st.subheader('Numerieke Kolommen')
        fig, ax = plt.subplots()
        mean_numerical_values.plot(kind='bar', ax=ax)
        ax.set_ylabel('Gemiddelde Waarde')
        ax.set_title('Gemiddelde Waarden van Numerieke Kolommen')
        st.pyplot(fig)

    # Toon de modus waarden in een tabelformaat voor niet-numerieke kolommen
    with col2:
        st.subheader('Categorische Kolommen')
        st.table(mode_non_numerical_values)

        # Knop om histogrammen weer te geven
    with st.expander('📊 Klik hier voor statistische verdelingsinzichten'):
        st.subheader('Histogrammen voor Verdeling van Numerieke Kenmerken')
        st.markdown("""
        Histogrammen bieden een visuele weergave van de verdeling van numerieke kenmerken in de dataset. Ze laten zien hoe de gegevenspunten zijn verspreid over verschillende waarden, wat helpt bij het begrijpen van de onderliggende patronen en verdelingen van de gegevens.
        """)

        # Toon histogrammen in een kleinere grootte met 3 kolommen
        numerical_cols = filtered_df.select_dtypes(include=[float, int]).columns  
        num_cols = 3  # Aantal kolommen voor histogrammen
        num_rows = (len(numerical_cols) + num_cols - 1) // num_cols  # Bereken het aantal benodigde rijen

        for row in range(num_rows):
            cols = st.columns(num_cols)
            for col_idx, col_name in enumerate(numerical_cols[row*num_cols:(row+1)*num_cols]):
                with cols[col_idx]:
                    fig, ax = plt.subplots(figsize=(3, 3))  # Pas de grootte aan indien nodig
                    sns.histplot(filtered_df[col_name].dropna(), kde=True, ax=ax)
                    ax.set_title(f'{col_name}')
                    st.pyplot(fig)
                    plt.close(fig)

# Functie Definities (geplaatst buiten de lay-out)
def calculate_similarity(df, mean_vector, numerical_cols, non_numerical_cols):
    # Normaliseer numerieke kolommen
    scaler = StandardScaler()
    df_numerical = scaler.fit_transform(df[numerical_cols])
    mean_numerical = scaler.transform([mean_vector[numerical_cols]])

    # Numerieke gelijkenis op basis van genormaliseerde Euclidische afstand
    numerical_distances = euclidean_distances(df_numerical, mean_numerical)
    max_numerical_distance = numerical_distances.max()
    numerical_similarity = 1 - (numerical_distances / max_numerical_distance)

    # Niet-numerieke gelijkenis op basis van modus matching
    non_numerical_similarity = df[non_numerical_cols].apply(lambda row: sum(row == mean_vector[non_numerical_cols]), axis=1)
    max_non_numerical_similarity = len(non_numerical_cols)
    non_numerical_similarity = non_numerical_similarity / max_non_numerical_similarity

    # Combineer beide gelijkenissen met gelijke weging
    similarity_score = (numerical_similarity.flatten() + non_numerical_similarity) / 2
    return similarity_score

def display_similar_tracks(df, mean_vector, numerical_cols, non_numerical_cols, section_type):
    similarities = calculate_similarity(df, mean_vector, numerical_cols, non_numerical_cols)
    df['Similarity'] = similarities
    similar_tracks = df.nlargest(10, 'Similarity')  # Toon de top 10 vergelijkbare sporen
    st.write(f"Top 10 sporen die lijken op de {section_type} gemiddelde treinsectie")
    st.write(similar_tracks[['Track Section', 'Similarity'] + list(numerical_cols) + list(non_numerical_cols)])
    df.drop(columns=['Similarity'], inplace=True)  # Opruimen

# Filter- en inclusielogica (dit moet worden geplaatst voordat de kolomlay-out wordt gemaakt om ervoor te zorgen dat variabelen beschikbaar zijn)
included_numerical_cols = []  # Initialiseer als een lege lijst
included_non_numerical_cols = []  # Initialiseer als een lege lijst

# Veronderstel dat column_inclusion een woordenboek is dat eerder in het script is gevuld
for column, (include, filter_values) in column_inclusion.items():
    if include:
        if pd.api.types.is_numeric_dtype(df[column]):
            included_numerical_cols.append(column)
        else:
            included_non_numerical_cols.append(column)

# Kolomlay-out voor de interactieve elementen
col1, col2 = st.columns([2, 3])

with col1:
    st.subheader('Download naar Excel')
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        mean_track_section.to_excel(writer, sheet_name='Gemiddelde Treinsectie')
    output.seek(0)
    st.download_button(
        label="Download Samenvatting van Gemiddelde Treinsectie naar Excel",
        data=output,
        file_name="Gemiddelde_Treinsectie_Samenvatting.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

with col2:
    st.subheader('Vind een real-life match')
    if st.button('Gemiddelde Treinsectie in Werkelijke Sporen'):
        display_similar_tracks(df, mean_track_section, included_numerical_cols, included_non_numerical_cols, 'Gemiddelde')


# Voeg een titel en beschrijving toe
st.markdown("""
    <h1 style='font-size:2.4em; color:darkgreen;'>Stedelijke/Suburbane/Regionale Treinsporen </h1>
    <hr style='border:2px solid darkgreen;'>
    """, unsafe_allow_html=True)
st.markdown("""
    Dit dashboard stelt je in staat om verschillende kenmerken van treinsporen te analyseren en visualiseren die zijn gecategoriseerd in stedelijke, suburbane en regionale types.

    **Instructies:**
    1. Gebruik de zijbalk om specifieke kenmerken in de analyse in of uit te schakelen.
    2. Kies of je emplacementgegevens wilt uitsluiten.
    3. Selecteer de soorten grafieken die je wilt weergeven.
    4. Het dashboard biedt opties om gemiddelden, verdelingen en samenvattingen van numerieke en niet-numerieke kenmerken weer te geven.

    **Opmerking:** De gegevens worden gefilterd op basis van de selecties die je in de zijbalk maakt.
""")
# Visualisatieopties
st.subheader('Visualisatieopties')
graph_options = st.multiselect(
    'Selecteer de grafieken die je wilt zien:',
    ['Numerieke Gemiddelden per Categorie', 'Niet-Numerieke Modus per Categorie', 'Numerieke Samenvatting']
)
# Groepeer op 'Urban/Regional/Suburban' en bereken het gemiddelde en de standaarddeviatie voor numerieke kenmerken en de meest voorkomende waarde voor niet-numerieke kenmerken
numerical_cols = filtered_df.select_dtypes(include=[float, int]).columns.difference(descriptive_columns)
non_numerical_cols = filtered_df.select_dtypes(exclude=[float, int]).columns.difference(descriptive_columns)

mean_numerical = filtered_df.groupby('Stedelijk/Voorstedelijk/Regionaal')[numerical_cols].mean()
mode_non_numerical = filtered_df.groupby('Stedelijk/Voorstedelijk/Regionaal')[non_numerical_cols].agg(lambda x: x.mode()[0])
grouped_stds = filtered_df.groupby('Stedelijk/Voorstedelijk/Regionaal')[numerical_cols].std()

# Combineer numerieke en niet-numerieke samenvattingen
summary_numerical = mean_numerical
summary_non_numerical = mode_non_numerical
summary_std = grouped_stds

# Definieer de functie om alle numerieke kenmerken te plotten
def plot_all_numerical_features(mean_values, std_values, categories, group_size=6):
    numerical_features = mean_values.columns
    num_features = len(numerical_features)
    cols = 2  # Aantal kolommen voor subplots
    rows = (group_size // cols) + (group_size % cols > 0)  # Bereken het aantal benodigde rijen per groep

    for start_idx in range(0, num_features, group_size):
        end_idx = min(start_idx + group_size, num_features)
        fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))
        axes = axes.flatten()

        for i, col in enumerate(numerical_features[start_idx:end_idx]):
            ax = axes[i]
            means = mean_values[col]
            errors = std_values[col]
            ax.errorbar(categories, means, yerr=errors, fmt='o', color='blue', capsize=5, label='Standaarddeviatie')
            ax.scatter(categories, means, color='red', zorder=5, label=f'{col} (Gemiddelde)')
            ax.set_title(f'Gemiddelden van {col} per Stedelijke/Regionale/Suburbane Categorie')
            ax.set_ylabel(f'Gemiddelde {col}')
            ax.set_xlabel('Stedelijke/Regionale/Suburbane Categorie')
            ax.legend(loc='upper right')
            ax.tick_params(axis='x', rotation=45)

        # Verwijder eventuele lege subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout(pad=3.1)
        fig.subplots_adjust(top=0.9)
        fig.suptitle(f'Gemiddelden van Numerieke Kenmerken van {start_idx + 1} tot {end_idx} per Stedelijke/Regionale/Suburbane Categorie', fontsize=16)
        st.pyplot(fig)
        plt.close(fig)
        

if 'Numerieke Gemiddelden per Categorie' in graph_options:
    plot_all_numerical_features(mean_numerical, grouped_stds, mean_numerical.index)

    # Voeg een uitklapper toe voor numerieke verdelingen
    with st.expander("📊 Klik hier voor gedetailleerde numerieke verdelingen"):
        # Definieer de functie om verdelingen te plotten
        def plot_distributions(columns, df, title, cols=3):
            num_plots = len(columns)
            rows = (num_plots // cols) + (num_plots % cols > 0)  # Bereken het aantal benodigde rijen

            fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))
            axes = axes.flatten()

            for i, col in enumerate(columns):
                sns.boxplot(x='Stedelijk/Voorstedelijk/Regionaal', y=col, data=df, ax=axes[i])
                axes[i].set_title(f'Verdeling van {col}', fontsize=10, pad=10)
                axes[i].set_ylabel(col, fontsize=8)
                axes[i].set_xlabel('')
                axes[i].tick_params(axis='x', labelsize=6)

            # Verwijder eventuele lege subplots
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            plt.tight_layout(pad=3.1)  # Pas de marge tussen subplots aan
            fig.subplots_adjust(top=0.9)  # Pas de bovenmarge aan om ruimte te maken voor de hoofdtekst
            fig.suptitle(title, fontsize=16)  # Hoofdtekst
            st.pyplot(fig)
            plt.close(fig)

        # Verdeel numerieke kolommen in kleinere groepen voor betere leesbaarheid
        group_size = 6  # Aantal subplots per figuur

        # Maak subfiguren voor elke groep
        for start_index in range(0, len(numerical_cols), group_size):
            end_index = min(start_index + group_size, len(numerical_cols))
            group = numerical_cols[start_index:end_index]
            plot_distributions(group, filtered_df, f'Verdelingen van Numerieke Kenmerken {start_index + 1} tot {end_index}')

        # Behandel de resterende kolommen als de verdeling niet perfect is
        if end_index < len(numerical_cols):
            remaining_cols = numerical_cols[end_index:]
            plot_distributions(remaining_cols, filtered_df, 'Verdelingen van Resterende Numerieke Kenmerken')

# Toon modus van niet-numerieke kolommen per categorie
if 'Niet-Numerieke Modus per Categorie' in graph_options:
    st.subheader('Modus van Niet-Numerieke Kenmerken')
    st.write("Hieronder staan de meest voorkomende waarden (modus) voor de niet-numerieke kenmerken in verschillende spoorcategorieën.")
    st.table(mode_non_numerical)

    # Voeg een uitklapper toe voor gedetailleerde niet-numerieke verdelingen
    with st.expander("📊 Klik hier voor gedetailleerde niet-numerieke kenmerken verdelingen"):
        # Definieer de functie om niet-numerieke verdelingen te plotten
        def plot_non_numerical_distributions(columns, df, title, cols=3):
            num_plots = len(columns)
            rows = (num_plots // cols) + (num_plots % cols > 0)  # Bereken het aantal benodigde rijen

            fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))
            axes = axes.flatten()

            for i, col in enumerate(columns):
                sns.countplot(x='Urban/Regional/Suburban', hue=col, data=df, ax=axes[i])
                axes[i].set_title(f'Verdeling van {col}', fontsize=10, pad=10)
                axes[i].set_xlabel('Stedelijke/Regionale/Suburbane Categorie', fontsize=8)
                axes[i].tick_params(axis='x', labelsize=6, rotation=45)  # Pas lettergrootte en rotatie aan
                axes[i].legend(title=col, fontsize=6, title_fontsize=8)  # Pas lettergrootte van de legenda aan

            # Verwijder eventuele lege subplots
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            plt.tight_layout(pad=3.1)  # Pas de marge tussen subplots aan
            fig.subplots_adjust(top=0.9)  # Pas de bovenmarge aan om ruimte te maken voor de hoofdtekst
            fig.suptitle(title, fontsize=16)  # Hoofdtekst
            st.pyplot(fig)
            plt.close(fig)

        # Roep de plotfunctie aan voor niet-numerieke verdelingen
        plot_non_numerical_distributions(non_numerical_cols, filtered_df, 'Niet-Numerieke Kenmerken Verdelingen')

# Visualisatiefunctie voor numerieke gegevens
def plot_numerical_summary(summary, title):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    categories = ['Stedelijk', 'Voorstedelijk', 'Regionaal']

    for i, category in enumerate(categories):
        summary.loc[category].plot(kind='bar', ax=axes[i])
        axes[i].set_title(f'Gemiddelde {category} Treinsectie')
        axes[i].set_ylabel('Gemiddelde Waarde')
        axes[i].set_xlabel('Kenmerken')
        axes[i].tick_params(axis='x', labelsize=8, rotation=90)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    st.pyplot(fig)
    plt.close(fig)

if 'Numerieke Samenvatting' in graph_options:
    st.subheader('Samenvatting van Numerieke Kenmerken per Categorie')
    plot_numerical_summary(summary_numerical, 'Gemiddelde Stedelijk/Voorstedelijk/Regionaal Treinsecties')

# Functie Definities (geplaatst buiten de lay-out)
def calculate_similarity(df, mean_vector, numerical_cols, non_numerical_cols):
    # Normaliseer numerieke kolommen
    scaler = StandardScaler()
    df_numerical = scaler.fit_transform(df[numerical_cols])
    mean_numerical = scaler.transform([mean_vector[numerical_cols]])

    # Numerieke gelijkenis op basis van genormaliseerde Euclidische afstand
    numerical_distances = euclidean_distances(df_numerical, mean_numerical)
    max_numerical_distance = numerical_distances.max()
    numerical_similarity = 1 - (numerical_distances / max_numerical_distance)

    # Niet-numerieke gelijkenis op basis van modus matching
    non_numerical_similarity = df[non_numerical_cols].apply(lambda row: sum(row == mean_vector[non_numerical_cols]), axis=1)
    max_non_numerical_similarity = len(non_numerical_cols)
    non_numerical_similarity = non_numerical_similarity / max_non_numerical_similarity

    # Combineer beide gelijkenissen met gelijke weging
    similarity_score = (numerical_similarity.flatten() + non_numerical_similarity) / 2
    return similarity_score

def display_similar_tracks(df, mean_vector, numerical_cols, non_numerical_cols, section_type):
    similarities = calculate_similarity(df, mean_vector, numerical_cols, non_numerical_cols)
    df['Similarity'] = similarities
    similar_tracks = df.nlargest(10, 'Similarity')  # Toon de top 10 vergelijkbare sporen
    st.write(f"Top 10 sporen die lijken op de {section_type} gemiddelde treinsectie")
    st.write(similar_tracks[['Track Section', 'Similarity'] + list(numerical_cols) + list(non_numerical_cols)])
    df.drop(columns=['Similarity'], inplace=True)

# Filter- en inclusielogica (dit moet worden geplaatst voordat de kolomlay-out wordt gemaakt om ervoor te zorgen dat variabelen beschikbaar zijn)
included_numerical_cols = []  # Initialiseer als een lege lijst
included_non_numerical_cols = []  # Initialiseer als een lege lijst

# Veronderstel dat column_inclusion een woordenboek is dat eerder in het script is gevuld
for column, (include, filter_values) in column_inclusion.items():
    if include:
        if pd.api.types.is_numeric_dtype(df[column]):
            included_numerical_cols.append(column)
        else:
            included_non_numerical_cols.append(column)

# Kolomlay-out voor de interactieve elementen
col1, col2 = st.columns([2, 3])

with col1:
    st.subheader('Download naar Excel')
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        summary_numerical.to_excel(writer, sheet_name='Numerieke Kenmerken')
        summary_std.to_excel(writer, sheet_name='Standaarddeviatie')
        summary_non_numerical.to_excel(writer, sheet_name='Niet-Numerieke Kenmerken')
    output.seek(0)
    st.download_button(
        label="Download Samenvatting van Stedelijke/Suburbane/Regionale Spoorsecties naar Excel",
        data=output,
        file_name="Categorieën_Samenvatting.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

with col2:
    st.subheader('Vind een real-life match')
    if st.button('Stedelijke Spoorsectie in Werkelijke Sporen'):
        urban_mean = pd.concat([mean_numerical.loc['Stedelijk'], mode_non_numerical.loc['Stedelijk']])
        display_similar_tracks(df, urban_mean, included_numerical_cols, included_non_numerical_cols, 'Stedelijk')

    if st.button('Suburbane Spoorsectie in Werkelijke Sporen'):
        suburban_mean = pd.concat([mean_numerical.loc['Voorstedelijk'], mode_non_numerical.loc['Voorstedelijk']])
        display_similar_tracks(df, suburban_mean, included_numerical_cols, included_non_numerical_cols, 'Suburbaan')

    if st.button('Regionale Spoorsectie in Werkelijke Sporen'):
        regional_mean = pd.concat([mean_numerical.loc['Regionaal'], mode_non_numerical.loc['Regionaal']])
        display_similar_tracks(df, regional_mean, included_numerical_cols, included_non_numerical_cols, 'Regionaal')



# Functie Definities (geplaatst aan het begin)
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
    st.write(f"Top 10 sporen die lijken op de {section_type} gemiddelde treinsectie")
    st.write(similar_tracks[['Track Section', 'Similarity'] + list(numerical_cols) + list(non_numerical_cols)])
    df.drop(columns=['Similarity'], inplace=True)

st.markdown("""
    <h1 style='font-size:2.5em; color:darkred;'>K-Clustering van Treinsecties</h1>
    <hr style='border:2px solid darkred;'>
    """, unsafe_allow_html=True)
st.markdown("Het k-means clustering algoritme wordt toegepast op de voorbewerkte gegevens. K-means clustering heeft tot doel n observaties te partitioneren in k clusters waarin elke observatie behoort tot het cluster met het dichtstbijzijnde gemiddelde, dat dient als prototype van het cluster. Het k-means algoritme minimaliseert de WCSS (Within-Cluster Sum of Square), ook bekend als de traagheid.")

# Visualisatieopties
st.subheader('Visualisatieopties')
graph_options = st.multiselect(
    'Selecteer de grafieken die je wilt zien:',
    ['PCA Resultaat', 'Pairplot']
)

numerical_cols = [col for col, (include, _) in column_inclusion.items() if include and pd.api.types.is_numeric_dtype(df[col])]

if numerical_cols:
    numerical_data = filtered_df[numerical_cols]
    imputer = SimpleImputer(strategy='mean')
    imputed_data = imputer.fit_transform(numerical_data)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(imputed_data)

    wcss = []
    max_clusters = 15

    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
        kmeans.fit(scaled_data)
        wcss.append(kmeans.inertia_)

    k = 5
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_data)

    scaled_data_df = pd.DataFrame(scaled_data, columns=numerical_cols)
    scaled_data_df['Cluster'] = clusters

    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)
    pca_df = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = clusters

    if 'PCA Resultaat' in graph_options:
        plt.figure(figsize=(10, 6))
        for cluster in range(k):
            plt.scatter(pca_df[pca_df['Cluster'] == cluster]['PC1'], pca_df[pca_df['Cluster'] == cluster]['PC2'], label=f'Cluster {cluster}')
        plt.title('PCA van Clusters')
        plt.xlabel('Hoofcomponent 1')
        plt.ylabel('Hoofcomponent 2')
        plt.legend()
        plt.grid(True)
        st.pyplot()

    subset_features = numerical_cols[:5]
    pairplot_data = pd.concat([pd.DataFrame(scaled_data, columns=numerical_cols), pd.Series(clusters, name='Cluster')], axis=1)
    pairplot_data = pairplot_data[['Cluster'] + list(subset_features)]
    pairplot_data['Cluster'] = pairplot_data['Cluster'].astype(str)

    if 'Pairplot' in graph_options:
        sns.pairplot(pairplot_data, hue='Cluster', palette='Set1')
        plt.suptitle('Pairplot van Clusters (Subset van Kenmerken)', y=1.02)
        st.pyplot()

filtered_df['Cluster'] = clusters

cluster_analysis = filtered_df.groupby('Cluster')[numerical_cols].mean()
non_numerical_cols_for_analysis = non_numerical_cols.difference(descriptive_columns)
non_numerical_analysis = filtered_df.groupby('Cluster')[non_numerical_cols_for_analysis].agg(lambda x: x.value_counts().index[0])

# Kolomlay-out voor de interactieve elementen
col1, col2 = st.columns([2, 3])

with col1:
    st.subheader('Download naar Excel')
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        if not cluster_analysis.empty:
            cluster_analysis.to_excel(writer, sheet_name='Cluster_Samenvatting')
        if not non_numerical_analysis.empty:
            non_numerical_analysis.to_excel(writer, sheet_name='Niet-Numerieke_Samenvatting')
    output.seek(0)
    st.download_button(
        label="Download Samenvatting van K-Means Clusters naar Excel",
        data=output,
        file_name="K_Means_Clusters.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

with col2:
    st.subheader('Vind een real-life match')
    for i in range(5):
        cluster_mean = pd.concat([cluster_analysis.loc[i], non_numerical_analysis.loc[i]])
        if st.button(f'Cluster {i} in Werkelijke Sporen'):
            display_similar_tracks(df, cluster_mean, included_numerical_cols, included_non_numerical_cols, f'Cluster {i}')
