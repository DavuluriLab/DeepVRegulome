import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test, logrank_test
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode
import scipy.stats as stats


st.set_page_config(
    layout="wide",
    page_title="Predicting Somatic Variant Effect using DNABERT"
)
# Title of the app
st.title("Atlas of Somatic Mutations in Gene Regulatory Elements: Application of DNABERT and DNABERT-2 on Whole Genome Sequencing Data.")

# Sidebar for selecting analysis parameters
st.sidebar.header("Select Analysis Parameters")

# Cancer type selection
#cancer_type = st.sidebar.selectbox("Select Cancer Type", ["Brain", "Lung", "Breast"])
cancer_type = "Brain"

# Analysis type selection
analysis_type = st.sidebar.selectbox("Genomic Regulatory Elements", ["Splice Sites", "Promoter Regions", "TFBS Models"])

# Data source selection
data_source = st.sidebar.selectbox("Genomic Analysis Tools", ["CaVEMan", "Sanger"])

# Define file paths
file_paths = {
    "Brain": {
        "Splice Sites": {
            "CaVEMan": "data/Brain/DNABERT_Output_files/CaVEMan_loss.xlsx",
            "Sanger": "data/Brain/DNABERT_Output_files/Sanger_loss.xlsx"
        },
        # Add paths for Promoter Regions and TFBS Models if available
    },
    # Add paths for Lung and Breast cancers if available
}

# Get the selected file path
file_path = file_paths.get(cancer_type, {}).get(analysis_type, {}).get(data_source, None)
clinical_file_path = f"data/{cancer_type}/Clinical_files/patient_clinical_updated.tsv"
clinvar_details_path = f"data/{cancer_type}/Clinical_files/{data_source}_merged_variant_clinvar_loss.tsv"




def calculate_p_values(df_kmf, df_transcript_info):
    kmf = KaplanMeierFitter()
    cph = CoxPHFitter(penalizer=0.1)  # Adding a penalizer

    # Add new columns to df_transcript_info to store results
    df_transcript_info['logrank_p_value'] = None
    df_transcript_info['coxph_p_value'] = None
    df_transcript_info['concordance_index'] = None
    df_transcript_info['hazard_ratio'] = None
    df_transcript_info['variant_information'] = None
    df_transcript_info['Disease_type Chi-square Statistics'] = None
    df_transcript_info['Disease_type Chi-square p-value'] = None

    for idx, row in df_transcript_info.iterrows():
        selected_patients_ids = [pid.split('_')[0] for pid in row['patient_ids'].split(',')]
        selected_patients = df_kmf[df_kmf['manifest_patient_id'].isin(selected_patients_ids)]
        df_transcript_info['variant_information'] = df_transcript_info.apply(
            lambda row: f"{row['chromosome']}:{row['variant_start_position']}:{row['ref_nucleotide']}>{row['alternative_nucleotide']}",
            axis=1
        )

        df_kmf["group"] = df_kmf.apply(lambda r: "B" if r['manifest_patient_id'] in selected_patients_ids else "A", axis=1)
        df_kmf["group_numeric"] = df_kmf["group"].apply(lambda x: 1 if x == "B" else 0)
        group_A = df_kmf[df_kmf['group'] == 'A']
        group_B = df_kmf[df_kmf['group'] == 'B']
        
        # Create a contingency table
        contingency_table = pd.crosstab(df_kmf['group'], df_kmf['disease_type'])
        # Perform Chi-square test
        chi2, p, dof, ex = stats.chi2_contingency(contingency_table)
        df_transcript_info.at[idx, 'Disease_type Chi-square Statistics'] = chi2
        df_transcript_info.at[idx, 'Disease_type Chi-square p-value'] = p

        if len(group_A) > 1 and len(group_B) > 1:  # Ensure there are enough data points for analysis
            
            
            # Perform log-rank test
            results_logrank = logrank_test(group_A['km_time'], group_B['km_time'], event_observed_A=group_A['km_status'], event_observed_B=group_B['km_status'])
            logrank_p_value = results_logrank.p_value

            # Fit Cox Proportional Hazards model
            df_kmf_clean = df_kmf[['manifest_patient_id', 'project_id', 'group', 'group_numeric', 'km_time', 'km_status']].dropna()

            # Diagnostic check for variance
            events = df_kmf_clean['km_status'].astype(bool)

            try:
                cph.fit(df_kmf_clean, duration_col='km_time', event_col='km_status', formula='group_numeric')
                
                coxph_p_value = cph.summary.loc['group_numeric', 'p']
                hazard_ratio = cph.summary.loc['group_numeric', 'exp(coef)']
                concordance_index = cph.concordance_index_

                # Store the results in the transcript info DataFrame
                df_transcript_info.at[idx, 'logrank_p_value'] = logrank_p_value
                df_transcript_info.at[idx, 'coxph_p_value'] = coxph_p_value
                df_transcript_info.at[idx, 'concordance_index'] = concordance_index
                df_transcript_info.at[idx, 'hazard_ratio'] = hazard_ratio
            except Exception as e:
                print(f"Error fitting CoxPH model for index {idx}: {e}")
                df_transcript_info.at[idx, 'logrank_p_value'] = None
                df_transcript_info.at[idx, 'coxph_p_value'] = None
                df_transcript_info.at[idx, 'concordance_index'] = None
                df_transcript_info.at[idx, 'hazard_ratio'] = None
        else:
            df_transcript_info.at[idx, 'logrank_p_value'] = None
            df_transcript_info.at[idx, 'coxph_p_value'] = None
            df_transcript_info.at[idx, 'concordance_index'] = None
            df_transcript_info.at[idx, 'hazard_ratio'] = None
    
    # Move 'variant_information' to the first column
    cols = ['variant_information'] + [col for col in df_transcript_info if col != 'variant_information']
    df_transcript_info = df_transcript_info[cols]
    return df_transcript_info


def plot_km_curve(group_A, group_B, title):
    results_logrank = logrank_test(group_A['km_time'], group_B['km_time'], event_observed_A=group_A['km_status'], event_observed_B=group_B['km_status'])
    logrank_p_value = results_logrank.p_value
    kmf = KaplanMeierFitter()
    kmf.fit(group_A['km_time'], event_observed=group_A['km_status'], label=f'Group A[{len(group_A)} patients]')
    kmf_A_sub = kmf.survival_function_
    ci_A = kmf.confidence_interval_
    kmf.fit(group_B['km_time'], event_observed=group_B['km_status'], label=f'Group B[{len(group_B)} patients]')
    kmf_B_sub = kmf.survival_function_
    ci_B = kmf.confidence_interval_
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=kmf_A_sub.index, y=kmf_A_sub.iloc[:, 0], mode='lines', name=f'Group A[{len(group_A)} patients]'))
    fig.add_trace(go.Scatter(x=kmf_B_sub.index, y=kmf_B_sub.iloc[:, 0], mode='lines', name=f'Group B[{len(group_B)} patients]'))
    fig.add_trace(go.Scatter(
        x=list(ci_A.index) + list(ci_A.index[::-1]),
        y=list(ci_A.iloc[:, 0]) + list(ci_A.iloc[:, 1][::-1]),
        fill='toself',
        fillcolor='rgba(31, 119, 180, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=list(ci_B.index) + list(ci_B.index[::-1]),
        y=list(ci_B.iloc[:, 0]) + list(ci_B.iloc[:, 1][::-1]),
        fill='toself',
        fillcolor='rgba(255, 127, 14, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False
    ))
    fig.update_layout(
        title={
            'text': f"{title}<br>Log Rank p-value: {logrank_p_value:.4f}",
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title='Time (Days)',
        yaxis_title='Survival Probability'
    )
    return(fig)



# Function to load the data from all sheets
@st.cache_data
def load_data(file_path):
    try:
        xls = pd.ExcelFile(file_path)
        df_variants_frequency = pd.read_excel(xls, 'Varaints_frequency')
        df_intersect_with_dbsnp = pd.read_excel(xls, 'Intersect_withDBSNP')
        df_transcript_info = pd.read_excel(xls, 'Transcript_Information')
        return df_variants_frequency, df_intersect_with_dbsnp, df_transcript_info
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

# Load the data
df_variants_frequency, df_intersect_with_dbsnp, df_transcript_info = load_data(file_path)
# Load clinical data
df_clinical = pd.read_csv(clinical_file_path, sep='\t')

#Loading Clinvar details
df_clinvar = pd.read_csv(clinvar_details_path, sep="\t")


if df_variants_frequency is not None and df_intersect_with_dbsnp is not None and df_transcript_info is not None:
    # Descriptions for columns
    descriptions = {
        "chromosome": "The chromosome where the variant is located.",
         analysis_type.split()[0].lower() + "_start_position": f"The start position of the {analysis_type}.",
         analysis_type.split()[0].lower() + "_end_position": f"The end position of the {analysis_type}.",
        "variant_start_position": "The start position of the variant.",
        "variant_end_position": "The end position of the variant.",
        "ref_nucleotide": "The reference nucleotide.",
        "alternative_nucleotide": "The alternative nucleotide.",
        "splice_sites_affected": f"The category of {analysis_type} affected.",
        "number_of_patients": "The number of patients with this variant.",
        "percentage_of_patients": "The percentage of patients with this variant.",
        "Loss of Function based on LogOddRatio": "The score calculated based on the DNABERT probability for both the reference and alternative sequences. A higher positive score indicates greater functionality disruption."
    }
    # Smalldescribtion
    # Dynamic description based on sidebar inputs
    description = f"""
    ### Analysis Overview
    <p>This dashboard provides insights into the genomic variants affecting {analysis_type.lower()} in GDC {cancer_type} cancer patients, analyzed using {data_source} Genomic Analysis Tool.
    The visualizations below display the distribution of these variants along with their predicted effects on regulatory elements based on DNABERT predictions.</p>
    <p>In addition to variant distribution, this analysis includes <strong>clinical significance</strong> assessments based on correlations with known data from DBSNP and GWAS datasets. Survival analysis plots are also provided, offering a deeper understanding of the potential impacts of these genomic variations on patient outcomes.</p> 
    """


    st.markdown(description, unsafe_allow_html=True)
    
    
    # Calculate the required height to center the content
    total_height = 600  # Example total height of the column
    content_height = 300  # Approximate height of the content
    top_padding = (total_height - content_height) // 2

    col1, col2 = st.columns([5, 4])



    with col1:
        try:
            # Add top padding
            st.write('<div style="height: {}px;"></div>'.format(top_padding), unsafe_allow_html=True)

            # Calculating unique counts for variant and splice site regions
            analysis_column =analysis_type.lower().replace(" ", "_")+"_affected"
            categories = df_variants_frequency[analysis_column].unique()
            
            # Initialize counts dictionary
            counts = {}
            # Calculating unique counts for each category
            for category in categories:
                #category_variants_count = df_variants_frequency[df_variants_frequency[analysis_column] == category][['chromosome', 'variant_start_position', 'variant_end_position']]
                category_variants_unique_count = df_variants_frequency[df_variants_frequency[analysis_column] == category][['chromosome', 'variant_start_position', 'variant_end_position', 'ref_nucleotide','alternative_nucleotide']].drop_duplicates()
                #category_sites_count = df_variants_frequency[df_variants_frequency[analysis_column] == category][['chromosome', 'splice_start_position', 'splice_end_position']]
                category_sites_unique_count = df_variants_frequency[df_variants_frequency[analysis_column] == category][['chromosome', analysis_type.split()[0].lower() + "_start_position", analysis_type.split()[0].lower() + "_end_position"]].drop_duplicates()
                unique_dbsnp_count = df_intersect_with_dbsnp[df_intersect_with_dbsnp[analysis_column] == category][['rsID','start','end']].drop_duplicates()
                unique_clinical_dbsnp_count = df_clinvar[df_clinvar[analysis_column] == category][['rsID','start','end']].drop_duplicates()
                
                
                counts[category] = {
                    'unique_variants': category_variants_unique_count.shape[0],
                    'unique_sites': category_sites_unique_count.shape[0],
                    'unique_dbsnp': unique_dbsnp_count.shape[0],
                    'unique_clinical_dbsnp': unique_clinical_dbsnp_count.shape[0]
                }

            # Create a DataFrame to display these counts
            data = {
                f"{analysis_type}": categories,
                "Unique Functional Sites": [counts[cat]['unique_sites'] for cat in categories],
                "Unique Variants": [counts[cat]['unique_variants'] for cat in categories],
                "Associated Unique DBSNPs": [counts[cat]['unique_dbsnp'] for cat in categories],
                "ClinVar DBSNPs": [counts[cat]['unique_clinical_dbsnp'] for cat in categories]
            }
             
            count_df = pd.DataFrame(data)
            count_df.index += 1
            st.dataframe(count_df)
            st.markdown("""
            <div align="center">
                <strong>Data Statistics for Candidate Variants in the Predicted {} Regions</strong><br>
                <span>Including the count of associated DBSNP IDs and ClinVar annotations.</span>
            </div>
            """.format(analysis_type), unsafe_allow_html=True)



        except Exception as e:
            st.error(f"Failed to load sunburst chart: {str(e)}")


    

    with col2:
        try:
            total_patients = df_clinical.shape[0]
            df_clinical['total_patients'] = f'Total Patients: {total_patients}'
            # Replace None values in relevant columns
            df_clinical['primary_diagnosis'] = df_clinical['primary_diagnosis'].fillna('Unknown')
            df_clinical['disease_type'] = df_clinical['disease_type'].fillna('Unknown')

            sunburst_fig = px.sunburst(
                df_clinical,
                path=['total_patients', 'project_id', 'disease_type', 'primary_diagnosis'],
                color='project_id',
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            sunburst_fig.update_layout(
                margin=dict(t=5, l=5, r=5, b=5),
                # sunburstcolorway=px.colors.qualitative.Set2,
                #sunburstcolorway=[ "#AB63FA",  "#00CC96",  "#FFA15A", "#EF553B", "#19D3F3","#636EFA"],
                extendsunburstcolors=True,
                font=dict(size=12)  # Increase font size here
            )
            
            # print(px.colors.qualitative.Set2)
            # # Print color assignments
            # color_assignments = {proj_id: px.colors.qualitative.Set2[i % len(px.colors.qualitative.Set2)]
            #                      for i, proj_id in enumerate(df_clinical['project_id'].unique())}
            # print(color_assignments)

            # Define hover templates
            hover_templates = {
                'total_patients': '<b>Total Patients</b><br>Count: %{value}<extra></extra>',
                'project_id': '<b>Project ID</b>: %{label}<br>Count: %{value}<extra></extra>',
                'disease_type': '<b>Disease Type</b>: %{label}<br>Count: %{value}<extra></extra>',
                'primary_diagnosis': '<b>Primary Diagnosis</b>: %{label}<br>Count: %{value}<extra></extra>'
            }

            # Apply hover templates to each trace based on level
            for trace in sunburst_fig.data:
                labels = trace['labels']
                trace_hovertemplates = []
                for label in labels:
                    if label.startswith('Total Patients'):
                        trace_hovertemplates.append(hover_templates['total_patients'])
                    elif label in df_clinical['project_id'].values:
                        trace_hovertemplates.append(hover_templates['project_id'])
                    elif label in df_clinical['disease_type'].values:
                        trace_hovertemplates.append(hover_templates['disease_type'])
                    elif label in df_clinical['primary_diagnosis'].values:
                        trace_hovertemplates.append(hover_templates['primary_diagnosis'])
                    else:
                        trace_hovertemplates.append('<b>%{label}</b><br>Count: %{value}<extra></extra>')
                trace.hovertemplate = trace_hovertemplates



            st.plotly_chart(sunburst_fig, use_container_width=True)
            st.markdown("<div style='text-align: center;'>Distribution Chart of Cancer Patients by Project ID.</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error processing clinical data: {str(e)}")

    df_clinical = df_clinical.drop(columns=['total_patients'])
    df_clinical = df_clinical.dropna(subset=['manifest_patient_id', 'project_id', 'km_time', 'km_status', 'disease_type'])
    
    
    description = f"""
    ### Details Description of the Candidate Splice Sites with associated Variants  
    <p> Here is the brief description of each columns of the dataframe.</p> 
    """
    st.markdown(description, unsafe_allow_html=True)
    
    # Display column descriptions
    html_content = ""
    for column, description in descriptions.items():
        html_content += f"<p style='margin-bottom:2px; text-indent:20px;'><b>- {column}:</b> {description}</p>"
    st.markdown(html_content, unsafe_allow_html=True)
    st.markdown("")
    
    
    if 'patient_ids' in df_variants_frequency.columns:
        df_variants_frequency_without_patient_ids = df_variants_frequency.drop(columns=['patient_ids'])
    else:
        df_variants_frequency_without_patient_ids = df_variants_frequency
    df_variants_frequency_without_patient_ids.index += 1
    st.dataframe(df_variants_frequency_without_patient_ids)



    # Pie chart for 'splice_sites_affected'
    st.write("""
    ### Splice Sites Affected Distribution

    This section provides an overview of the distribution of splice sites affected by genomic variants. The pie chart below categorizes the affected splice sites into 'acceptor' and 'donor' types. Each slice of the pie represents the proportion of affected splice sites within the selected cancer type and analysis parameters. This visualization helps in understanding which types of splice sites are more frequently affected by the variants in the dataset.
    """)
    pie_fig = px.pie(df_variants_frequency, names='splice_sites_affected')
    st.plotly_chart(pie_fig)

    
    # Filter data based on selected splice site
    # Add formatted text above the select box
    st.write("""#### Select Splice Site to Filter Data to visualize the selected splice data and the distribution of patients affected by variants:""")

    # Create the select box
    selected_site = st.selectbox(
        "Choose a splice site to see detailed information and patient distribution",
        df_variants_frequency['splice_sites_affected'].unique()
    )
    filtered_data = df_transcript_info[df_transcript_info['splice_sites_affected'] == selected_site].reset_index(drop=True)
    filtered_clinvar = df_clinvar[df_clinvar['splice_sites_affected']==f"{selected_site}"].reset_index(drop=True)
    #st.dataframe(filtered_clinvar)

    st.write(f"""
    ##### Filtered Data for '{selected_site}' Splice Site

    The table below displays the detailed information for the genomic variants that affect the selected splice site type ('{selected_site}'). This includes the chromosome location, nucleotide changes, and the predicted impact on splice site function.
    """)
    filtered_data.index +=1
    st.dataframe(filtered_data.drop(columns=['patient_ids']))
    # st.dataframe(filtered_clinvar)


    # Create bins for the histogram
    bins = [i for i in range(0, 101, 10)]
    filtered_data['percentage_bin'] = pd.cut(filtered_data['percentage_of_patients'], bins=bins, right=False)

    # Prepare data for Plotly
    bin_counts = filtered_data['percentage_bin'].value_counts().sort_index().reset_index()
    bin_counts.columns = ['percentage_bin', 'count']
    bin_counts['percentage_bin'] = bin_counts['percentage_bin'].astype(str)

    # Plotly bar chart
    st.write("""
    ##### Distribution of Patients Affected by Variant Regions

    This bar chart shows the distribution of patients affected by variant regions, grouped into percentage ranges (e.g., 0-10%, 10-20%, etc.). Each bar's height represents the number of patients within that range.

    To interact with the chart:
    <ul>
        <li><strong>Hover over the bars</strong> to see the count of patients in each percentage range.</li>
        <li><strong>Click on a bar</strong> to filter the data and display detailed variant information for that range.</li>
    </ul>

    This visualization helps identify how widely different variant regions affect the patient population, providing insights into the distribution and prevalence of specific variants within the selected splice site type.
    """, unsafe_allow_html=True)
    fig = px.bar(bin_counts, x='percentage_bin', y='count', color='percentage_bin',
                 title="Percentage of Patients Affected by Variant Regions.",
                 labels={'percentage_bin': 'Percentage of Patients', 'count': 'Count'},
                 color_discrete_sequence=px.colors.qualitative.Set2)

    # Add count labels on top of the bars
    fig.update_traces(texttemplate='%{y}', textposition='outside')

    # Capture click events using streamlit-plotly-events
    selected_points = plotly_events(fig, click_event=True, hover_event=False, select_event=False)

    # Display filtered data based on selected bin
    if selected_points:
        clicked_bin = selected_points[0]['x']
        st.write(f"""
        ### Detailed Variant Information for Patients in the {clicked_bin} Range.

        The table below shows the variant regions affecting patients within the {clicked_bin} percentage range.
        """)

        # Parse the bin range
        bin_start, bin_end = map(int, clicked_bin.replace('[', '').replace(')', '').split(','))
        filtered_rows = filtered_data[(filtered_data['percentage_of_patients'] >= bin_start) &
                                      (filtered_data['percentage_of_patients'] < bin_end)].reset_index(drop=True)
        
        df_transcript_info = calculate_p_values(df_clinical, filtered_rows)
        df_clinical = df_clinical.drop(columns=['group', 'group_numeric'])
        
        
        
         # Add a small header for the dataframe
        st.markdown("""
        ### Variants Information
        To observe the survival plot for that particular variant click on specific row.
        """)



        # Display the DataFrame without patient_ids column
        #st.dataframe(df_transcript_info.drop(columns=['patient_ids', 'percentage_bin', 'variant_start_position', 'variant_end_position', 'ref_nucleotide', 'alternative_nucleotide']))
        

        
        # Configure the grid options
        gb = GridOptionsBuilder.from_dataframe(df_transcript_info.drop(columns=['patient_ids', 'percentage_bin', 'variant_start_position', 'variant_end_position', 'ref_nucleotide', 'alternative_nucleotide']))
        # Increase font size for both cells and headers
        custom_css = {
            ".ag-header-cell-label": {
                "font-size": "16px !important",
            },
            ".ag-cell": {
                "font-size": "14px !important",
            },
            ".ag-row-selected": {
                "background-color": "#d3d3d3 !important",  # Light grey background for the selected row
            },
        }

        # Apply pagination, sidebar, single selection, and row numbers
        gb.configure_pagination(paginationAutoPageSize=False)
        gb.configure_side_bar()  # Add a sidebar
        gb.configure_selection('single', suppressRowDeselection=True)  # Enable single row selection
        gb.configure_grid_options(domLayout='normal')
        #gb.add_row_number_column()  # Add row numbers to the grid

        # Build grid options with custom CSS
        grid_options = gb.build()

        # Display the DataFrame with interactive elements
        grid_response = AgGrid(
            df_transcript_info,
            gridOptions=grid_options,
            data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            fit_columns_on_grid_load=False,
            enable_enterprise_modules=True,
            custom_css=custom_css  # Apply custom CSS for font size and visibility
        )
        

        # Extract selected rows safely
        selected_rows = grid_response.get('selected_rows', [])

        if len(selected_rows) > 0:
            #st.write("Selected Rows:", selected_rows)  # For debugging, showing the selected rows
            variant_info = selected_rows.iloc[0]['variant_information']
            selected_variant_data = df_transcript_info[df_transcript_info['variant_information'] == variant_info].iloc[0]
            selected_patients_ids = [pid.split('_')[0] for pid in selected_variant_data['patient_ids'].split(',')]
            selected_patients = df_clinical[df_clinical['manifest_patient_id'].isin(selected_patients_ids)]
            df_clinical["group"] = df_clinical.apply(lambda r: "B" if r['manifest_patient_id'] in selected_patients_ids else "A", axis=1)
            df_clinical["group_numeric"] = df_clinical["group"].apply(lambda x: 1 if x == "B" else 0)
            
            st.markdown("<h3 style='text-align: center;'>Detailed Variant Information from ClinVar</h3>", unsafe_allow_html=True)

            st.markdown("Below, you'll find a comprehensive breakdown of the variant's details, organized into categories for easy navigation. Each tab provides specific insights into different aspects of the variant, including its genomic details, population frequencies, clinical significance, molecular impact, and more.")

            # Ensure that selected_variant_start is converted to the correct type if needed
            selected_variant_start = int(variant_info.split(":")[1])

            # Filter the DataFrame
            filtered_result = filtered_clinvar[
                (filtered_clinvar['chromosome'] == selected_rows.iloc[0]['chromosome']) &
                (filtered_clinvar['splice_start_position'] == selected_rows.iloc[0]['splice_start_position']) &
                (filtered_clinvar['splice_end_position'] == selected_rows.iloc[0]['splice_end_position']) &
                (filtered_clinvar['variant_start_position'] == selected_variant_start)
            ].reset_index(drop=True)

            # Check if the filtered DataFrame is empty
            if filtered_result.empty:
                st.write(f"***There is no clinical information available from ClinVar for the variant {variant_info}.***")
            else:
                for index, row in filtered_result.iterrows():
                    #st.markdown(f"### Variant Information for ClinVar Entry {index+1}")
                    st.markdown("""
                        <style>
                        .tooltip {
                            position: relative;
                            display: inline-block;
                            border-bottom: 1px dotted black;
                        }

                        .tooltip .tooltiptext {
                            visibility: hidden;
                            width: 220px;
                            background-color: black;
                            color: #fff;
                            text-align: center;
                            border-radius: 6px;
                            padding: 5px;
                            position: absolute;
                            z-index: 1;
                            bottom: 125%; /* Position the tooltip above the text */
                            left: 50%;
                            margin-left: -110px;
                            opacity: 0;
                            transition: opacity 0.3s;
                        }

                        .tooltip:hover .tooltiptext {
                            visibility: visible;
                            opacity: 1;
                        }
                        
                        /* Tab container: Make tabs stretch across the full width */
                        div.stTabs div[data-baseweb="tab-list"] {
                            display: flex;
                            justify-content: space-between;
                        }

                        /* Individual tabs */
                        div.stTabs div[data-baseweb="tab"] {
                            flex-grow: 1; /* Make each tab grow to fill the available space */
                            background-color: #f0f0f5; /* Light grey background for tabs */
                            color: #333; /* Dark text color */
                            font-weight: bold; /* Bold font */
                            border-radius: 5px; /* Rounded corners */
                            padding: 10px; /* Padding inside tabs */
                            margin-right: 5px; /* Space between tabs */
                            text-align: center; /* Center the text in the tab */
                        }

                        /* Active tab */
                        div.stTabs div[data-baseweb="tab"][aria-selected="true"] {
                            background-color: #4CAF50; /* Green background for active tab */
                            color: white; /* White text for active tab */
                            border: 2px solid #4CAF50; /* Border for active tab */
                        }

                        /* Hover effect for tabs */
                        div.stTabs div[data-baseweb="tab"]:hover {
                            background-color: #ddd; /* Light grey background on hover */
                            color: #000; /* Black text on hover */
                            cursor: pointer; /* Pointer cursor on hover */
                        }
                                            </style>
                    """, unsafe_allow_html=True)
                    
                    
                    # Create tabs
                    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                        "Variant Description",
                        "Population Frequency",
                        "Clinical Significance and Disease Association",
                        "Molecular Consequence",
                        "Other Identifiers",
                        "Clinical Observations",
                        "Quality and Filtering"
                    ])

                    with tab1:
                        # Genomic Location and Variant Description
                        rsid = row['rsID']
                        rsid_link = f"https://www.ncbi.nlm.nih.gov/snp/{rsid}"
                        st.markdown(f"**rsID:** [**{rsid}**]({rsid_link})", unsafe_allow_html=True)
                        st.markdown(f"""
                            **HGVS Nomenclature** 
                            <span class="tooltip">[?]
                              <span class="tooltiptext">HGVS nomenclature that provides a standardized way to describe the variant</span>
                            </span>: {row['CLNHGVS']}
                        """, unsafe_allow_html=True)
                        st.markdown(f"""
                            **Variant Classification** 
                            <span class="tooltip">[?]
                              <span class="tooltiptext">Indicating the type of mutation</span>
                            </span>: {row['CLNVC']}
                        """, unsafe_allow_html=True)

                    with tab2:
                        # Population Frequency
                        st.write(f"**Minor Allele Frequency:** {row['minorAlleleFreq']}")
                        st.markdown(f"""
                            **AF ESP** 
                            <span class="tooltip">[?]
                              <span class="tooltiptext">Allele frequency in the Exome Sequencing Project (ESP), primarily from European and African American ancestry</span>
                            </span>: {row['AF_ESP']}
                        """, unsafe_allow_html=True)

                        # AF ExAC with Tooltip
                        st.markdown(f"""
                            **AF ExAC** 
                            <span class="tooltip">[?]
                              <span class="tooltiptext">Allele frequency in the Exome Aggregation Consortium (ExAC), which aggregates exome data from diverse populations</span>
                            </span>: {row['AF_EXAC']}
                        """, unsafe_allow_html=True)

                        # AF 1000 Genomes Project with Tooltip
                        st.markdown(f"""
                            **AF 1000 Genomes Project** 
                            <span class="tooltip">[?]
                              <span class="tooltiptext">Allele frequency in the 1000 Genomes Project, covering genetic variation across multiple populations worldwide</span>
                            </span>: {row['AF_TGP']}
                        """, unsafe_allow_html=True)

                    with tab3:
                        # Clinical Significance and Disease Association
                        st.write(f"**Clinical Significance:** {row['CLNSIG']}")
                        st.write(f"**Clinical Condition:** {row['CLNDN']}")
                        st.write(f"**Additional conditions related to the primary clinical condition:** {row['CLNDNINCL']}")
                        st.write(f"**Associated Disease Databases:** {row['CLNDISDB']}")
                        st.write(f"**Additional Associated Disease Databases:** {row['CLNDISDBINCL']}")
                        st.write(f"**Review Status:** {row['CLNREVSTAT']}")

                    with tab4:
                        # Molecular Consequence
                        st.write(f"**Molecular Consequence:** {row['MC']}")
                        gene_info = row['GENEINFO']
                        genes = gene_info.split('|')  # Split if there are multiple genes listed
                        # Initialize a list to hold formatted gene links
                        gene_links = []
                        # Iterate through each gene and create a hyperlink
                        for gene in genes:
                            gene_name, gene_id = gene.split(':')  # Split into gene name and gene ID
                            gene_link = f"https://www.ncbi.nlm.nih.gov/gene/{gene_id}"  # Create the link to NCBI Gene
                            gene_links.append(f"[**{gene_name}**]({gene_link})")  # Format as a Markdown link
                        # Combine original gene_info and the formatted links
                        st.markdown(f"**Gene Information:** {gene_info} ({' | '.join(gene_links)})", unsafe_allow_html=True)

                    with tab5:
                        # Other Identifiers
                        st.write(f"**Allele ID:**(Unique identifier within ClinVar) {row['ALLELEID']}")
                        st.write(f"**DBVARID:**(Identifier for structural variants in the dbVar database) {row['DBVARID']}")
                        st.write(f"**RS:** {row['RS']}")

                    with tab6:
                        # Clinical Observations
                        st.write(f"**SCIDN:** {row['SCIDN']}")
                        st.write(f"**SCIDNINCL:** {row['SCIDNINCL']}")
                        st.write(f"**SCIDISDB:** {row['SCIDISDB']}")
                        st.write(f"**SCIDISDBINCL:** {row['SCIDISDBINCL']}")
                        st.write(f"**SCIREVSTAT:** {row['SCIREVSTAT']}")
                        st.write(f"**SCI:** {row['SCI']}")
                        st.write(f"**SCIINCL:** {row['SCIINCL']}")

                    with tab7:
                        # Quality and Filtering
                        st.write(f"**Quality:** {row['QUAL']}")
                        st.write(f"**Filter:** {row['FILTER']}")
                    
                    
                    

                    st.markdown("---")  # Add a horizontal line to separate entries
            
            

            



            # Aggregating data to count occurrences
            df_counts = df_clinical.groupby(['group', 'project_id']).size().reset_index(name='count')
            fig1 = px.bar(df_counts, x='group', y='count', color='project_id', title='Project ID Distribution by Group', 
                          color_discrete_sequence=px.colors.qualitative.Set3)
            # Center the title and adjust font size
            fig1.update_layout(title={'x': 0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': {'size': 14}})


            

            df_counts = df_clinical.groupby(['group', 'gender']).size().reset_index(name='count')
            fig3 = px.bar(df_counts, x='group', y='count', color='gender', title='Gender Distribution by Group', 
                          color_discrete_sequence=px.colors.qualitative.Set2)
            fig3.update_layout(title={'x': 0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': {'size': 14}})


            # Aggregating data to count occurrences for race
            df_counts = df_clinical.groupby(['group', 'race','ethnicity']).size().reset_index(name='count')
            df_counts['group'] = df_counts['group'].replace({'A': 'Group A', 'B': 'Group B'})
            
            

            # Create the Sunburst chart
            fig4 = px.sunburst(
                df_counts,
                path=['group', 'race', 'ethnicity'],  # Define the hierarchy: group -> race
                values='count',  # The size of each segment
                color='group',  # Color by race to distinguish between categories
                title='Race and Ethnicity Distribution within Groups',
                color_discrete_sequence=px.colors.qualitative.Set2  # Use Plotly color palette
            )

            # Customize the layout to make it more visually clear
            fig4.update_layout(
                margin=dict(t=40, l=0, r=0, b=0),
                sunburstcolorway=px.colors.qualitative.Set2,  # Ensure consistent color usage
                uniformtext=dict(minsize=10, mode='hide'),  # Adjust text visibility
            )
            fig4.update_layout(title={'x': 0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': {'size': 14}})

            
            df_counts = df_clinical.groupby(['group', 'primary_diagnosis']).size().reset_index(name='count')
            fig5 = px.bar(df_counts, x='group', y='count', color='primary_diagnosis', title='Primary Diagnosis Distribution by Group', 
                          color_discrete_sequence=px.colors.qualitative.Set2)
            fig5.update_layout(title={'x': 0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': {'size': 14}})

            df_counts = df_clinical.groupby(['group', 'disease_type']).size().reset_index(name='count')
            fig6 = px.bar(df_counts, x='group', y='count', color='disease_type', title='Disease Type Distribution by Group', 
                          color_discrete_sequence=px.colors.qualitative.Set2)
            fig6.update_layout(title={'x': 0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': {'size': 14}})
            
            
            
            group_A = df_clinical[(df_clinical['group'] == 'A') & (df_clinical['disease_type'] == "GBM")]
            group_B = df_clinical[(df_clinical['group'] == 'B') & (df_clinical['disease_type'] == "GBM")]
            fig2 =  plot_km_curve(group_A, group_B, title=f"KM Plot for GBM Brain Cancer Patients with Variant {variant_info}")
            fig2.update_layout(title={'x': 0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': {'size': 14}})
            
            
            
            
            
            group_A = df_clinical[(df_clinical['group'] == 'A') & (df_clinical['disease_type'] == "GBM")]
            group_B = df_clinical[(df_clinical['group'] == 'B') & (df_clinical['disease_type'] == "LGG")]
            fig7 =  plot_km_curve(group_A, group_B, title=f"LGG Patients(Group-B) Harboring Variant {variant_info} vs. GBM Patients(Group-A) Lacking the Variant.")
            fig7.update_layout(title={'x': 0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': {'size': 14}})
            
            group_A = df_clinical[(df_clinical['group'] == 'A') & (df_clinical['disease_type'] == "LGG")]
            group_B = df_clinical[(df_clinical['group'] == 'B') & (df_clinical['disease_type'] == "GBM")]
            fig8 =  plot_km_curve(group_A, group_B, title=f"GBM Patients(Group-B) Harboring Variant {variant_info} vs. LGG Patients(Group-A) Lacking the Variant.")
            fig8.update_layout(title={'x': 0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': {'size': 14}})
            
            # Title centered
            st.markdown("<h3 style='text-align: center;'>Visualization of Clinical Data Stratification by Cohorts</h3>", unsafe_allow_html=True)

            # Arranging data distribution plots in a 2+2+1 format
            col1, col2, col3 = st.columns([1, 1, 1])  # Give more space to the column with the Sunburst chart

            with col1:
                st.plotly_chart(fig6)  # Disease Type Distribution by Group
                st.plotly_chart(fig5)  # Primary Diagnosis Distribution by Group

            with col2:
                st.plotly_chart(fig1)  # Project ID Distribution by Group
                st.plotly_chart(fig3)  # Gender Distribution by Group

            with col3:
                st.markdown("<div style='height: 250px;'></div>", unsafe_allow_html=True)  # Add vertical space before the Sunburst
                st.plotly_chart(fig4)  # Sunburst chart for Race and Ethnicity Distribution
                st.markdown("<div style='height: 100px;'></div>", unsafe_allow_html=True)  # Add vertical space after the Sunburst
            
            # Title centered
            st.markdown("<h3 style='text-align: center;'>Interpreting Cohort-Specific Survival Analysis for GBM Brain cancer Patients.</h3>", unsafe_allow_html=True)
            st.plotly_chart(fig2)  # First Kaplan-Meier plot


                
            
            st.markdown("<h3 style='text-align: center;'>Survival Plot of Mortality Risk in Different Brain Cancer Patients.</h3>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)

            with col1:
                st.plotly_chart(fig7)  # First Kaplan-Meier plot

            with col2:
                st.plotly_chart(fig8)  # Second Kaplan-Meier plot
            


            

            
            
            
    
            # group_A = df_clinical[df_clinical['group'] == 'A']
            # group_B = df_clinical[df_clinical['group'] == 'B']
            # plot_km_curve(group_A, group_B, title=variant_info, logrank_p_value=selected_rows.iloc[0]['logrank_p_value'])


        #return group_A, group_B, selected_variant_data['variant_information'], selected_variant_data['logrank_p_value']

#         # Handle the selection
#         if selected_rows:  # Ensure there are selected rows
#             st.write("Selected Variant Information:", selected_rows[0])  # Debugging selected variant
#             selected_variant = selected_rows[0]['variant_information']
#             st.session_state['variant_clicked'] = selected_variant

#         # Generate KM plot if a variant is clicked
#         if 'variant_clicked' in st.session_state and st.session_state['variant_clicked']:
#             st.write("Generating KM plot for variant:", st.session_state['variant_clicked'])  # Debugging KM plot generation
#             group_A, group_B, variant_information, logrank_p_value = get_km_plot_data(st.session_state['variant_clicked'])
#             plot_km_curve(group_A, group_B, idx=None, title=variant_information, logrank_p_value=logrank_p_value)



        # Function to generate KM plot on clicking a variant
#         @st.cache_data
#         def get_km_plot_data(variant_info):
#             selected_variant_data = df_transcript_info[df_transcript_info['variant_information'] == variant_info].iloc[0]
#             selected_patients_ids = [pid.split('_')[0] for pid in selected_variant_data['patient_ids'].split(',')]
#             selected_patients = df_clinical[df_clinical['manifest_patient_id'].isin(selected_patients_ids)]
#             df_clinical["group"] = df_clinical.apply(lambda r: "B" if r['manifest_patient_id'] in selected_patients_ids else "A", axis=1)
#             df_clinical["group_numeric"] = df_clinical["group"].apply(lambda x: 1 if x == "B" else 0)
#             group_A = df_clinical[df_clinical['group'] == 'A']
#             group_B = df_clinical[df_clinical['group'] == 'B']
#             return group_A, group_B, selected_variant_data['variant_information'], selected_variant_data['logrank_p_value']

#         # Handle button clicks
#         if 'variant_clicked' not in st.session_state:
#             st.session_state['variant_clicked'] = None

#         for idx, row in df_transcript_info.iterrows():
#             if st.button(f'Show Plot: {row["variant_information"]}', key=row["variant_information"]):
#                 st.session_state['variant_clicked'] = row['variant_information']
                

#         if st.session_state['variant_clicked']:
#             group_A, group_B, variant_information, logrank_p_value = get_km_plot_data(st.session_state['variant_clicked'])
#             plot_km_curve(group_A, group_B, idx=None, title=variant_information, logrank_p_value=logrank_p_value)
        

      

        # Display the table
        #st.plotly_chart(fig)


        
        
else:
    st.write("Failed to load data. Please check the file path and format.")
