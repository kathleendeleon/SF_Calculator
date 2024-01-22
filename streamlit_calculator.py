# from turtle import onclick
### https://www.snowflake.com/pricing/pricing-guide/
import streamlit as st
import pandas as pd
import csv
import datetime
from datetime import datetime
import time
import os
from PIL import Image
from streamlit.components.v1 import html

# Constants
#ROOT_FILE_PATH = os.path.abspath(os.path.dirname(__file__))
#SOURCE_FILE_PATH = ROOT_FILE_PATH + "/Source"
STORAGE_PRICE_PER_TB = 40
CAPACITY_PRICE_PER_TB = 23

# Variables
page_tile = "Snowflake Cost Calculator"
date_time = time.strftime("%Y%m%d-%H")  # format: YYYYMMDD-HH
output_filename = "credit_calculator_" + date_time + ".csv"

if 'warehouses' not in st.session_state:
    st.session_state['warehouses'] = []

if 'warehouse_entries' not in st.session_state:
    st.session_state['warehouse_entries'] = []

if 'clusters' not in st.session_state:
    st.session_state['clusters'] = []

if 'cluster_entries' not in st.session_state:
    st.session_state['cluster_entries'] = []

def reset():
    st.session_state['warehouses'] = []
    st.session_state['warehouse_entries'] = []
    st.session_state['clusters'] = []
    st.session_state['cluster_entries'] = []
    tb_per_month = 0


# Read in source data from credits, warehouse_details files in Source folder.
credit_data = pd.read_csv('credits.csv')
warehouse_df = pd.read_csv('warehouse_details.csv')

edition_credits = [
    ['Business Critical', 4],
    ['Enterprise', 3],
    ['Standard', 2]
    # ['Virtual Private Snowflake (VPS)', 1.25]
]
# item_credits = [2, 3, 4, 1.25]

providers_regions = [
    ['AWS', 'US East (Commercial Gov-N.VA)'],
    ['AWS', 'US Gov West 1'],
    ['AWS', 'US Gov West 1 (Fedramp High Plus)'],
    ['Azure', 'East US 2 (Virginia)'],
    ['Azure', 'South Central US (Texas)'],
    ['Azure', 'US Central (Iowa)'],
    ['GCP', 'US Central 1 (Iowa)'],
    ['AWS', 'US East (Northern Virginia)'],
    ['AWS', 'US East (Ohio)'],
    ['AWS', 'US East 1 Commercial Gov'],
    ['GCP', 'US East 4 (N. Virginia)'],
    ['AWS', 'US West (Oregon)'],
    ['Azure', 'US Gov Virginia'],
    ['Azure', 'West US 2 (Washington)'],
]

df_edition_credits = pd.DataFrame(edition_credits, columns=['Edition', 'Credit'])
df_providers_regions = pd.DataFrame(providers_regions, columns=['Provider', 'Region'])


snowflake_warehouse_sizes = {
    'XS': [1],
    'S': [2],
    'M': [4],
    'L': [8],
    'XL': [16],
    '2XL': [32],
    '3XL': [64],
    '4XL': [128]
}

features_standard = [
    '* Complete SQL data warehouse',
    '* Secure Data Sharing across regions / clouds',
    '* Premier Support 24 X 365',
    '* 1 day of time travel',
    '* Always-on enterprise grade encryption in transit and at rest',
    '* Customer dedicated virtual warehouses',
    '* Federated authentication',
    '* Database replication',
    '* External Functions',
    '* Snowsight',
    '* Create your own Data Exchange',
    '* Data Marketplace access'
]

features_enterprise = [
    '* Multi-cluster warehouse',
    '* Up to 90 days of time travel',
    '* Annual rekeying of all encrypted data',
    '* Materialized views',
    '* Search Optimization Service',
    '* Dynamic Data Masking',
    '* External Data Tokenization'
]

features_business_critical = [
    '* HIPAA support',
    '* PCI compliance',
    '* Data encryption everywhere',
    '* Tri-Secret Secure using customer-managed keys',
    '* AWS PrivateLink support',
    '* Azure Private Link support',
    '* Database failover and failback for business continuity'
]

st.set_page_config(page_title=page_tile, layout="wide")
col1, col2, col3 = st.columns([2,18,1])




image = Image.open("deloitte_sf.PNG")
st.image(image, width=600)


    

                    
if(True):
    st.header('Edition Features')
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        
        st.info("Standard")
        st.info("\n".join(sorted(features_standard)))
    with col2:
        st.info("Enterprise")
        st.info("\n".join(sorted(features_enterprise)))
    with col3:
        st.info("Business Critical")
        st.info("\n".join(sorted(features_business_critical)))
  




def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def remote_css(url):
    st.markdown(f'<link href="{url}" rel="stylesheet">',
                unsafe_allow_html=True)


def int_to_formatted_time(input_hour):
    format_time = datetime.strptime(str(input_hour), "%H")
    return format_time.strftime("%H:%M%p")


def add_warehouse_entry(
        warehouse,
        size,
        hours_per_day,
        days_per_week,
        hours_per_year,
        compute_credits,
        cloud_services,
        cloud_services_credits,
        credit_per_hour
):
    if warehouse not in st.session_state['warehouses']:
        st.session_state['warehouses'].append(warehouse)
        st.session_state['warehouse_entries'].append(
            [
                warehouse,
                size,
                hours_per_day,
                days_per_week,
                hours_per_year,
                compute_credits,
                cloud_services,
                cloud_services_credits,
                credit_per_hour
            ]
        )


def add_multi_cluster_entry(warehouse, size, start_time, end_time, hours_per_day, days_per_week, hours_per_year,
                            cloud_credits, cloud_service_credits):
    warehouse_cluster_name = (
            str(warehouse) +
            ': ' +
            int_to_formatted_time(start_time) +
            ' - ' +
            int_to_formatted_time(end_time)
    )

    if warehouse_cluster_name not in st.session_state['clusters']:
        st.session_state['clusters'].append(warehouse_cluster_name)
        st.session_state['cluster_entries'].append(
            [
                warehouse,
                warehouse_cluster_name,
                size,
                hours_per_day,
                days_per_week,
                hours_per_year,
                cloud_credits,
                cloud_service_credits
            ]
        )


def get_added_clusters():
    df_clusters = pd.DataFrame(
        st.session_state['cluster_entries'],
        columns=[
            'WAREHOUSE',
            'WAREHOUSE CLUSTER',
            'SIZE',
            'HOURS/DAY',
            'DAYS/WEEK',
            'HOURS/YEAR',
            'COMPUTE CREDITS',
            'CLOUD SERVICES CREDITS'
        ]
    )
    return df_clusters.astype({
        "WAREHOUSE": str,
        "WAREHOUSE CLUSTER": str,
        "SIZE": str,
        "HOURS/DAY": int,
        "DAYS/WEEK": int,
        "HOURS/YEAR": int,
        "COMPUTE CREDITS": int,
        'CLOUD SERVICES CREDITS': int
    })


def get_added_warehouses():
    df_warehouses = pd.DataFrame(
        st.session_state['warehouse_entries'],
        columns=[
            'WAREHOUSE',
            'SIZE',
            'HOURS/DAY',
            'DAYS/WEEK',
            'HOURS/YEAR',
            'COMPUTE CREDITS',
            'CLOUD SERVICES %',
            'CLOUD SERVICES CREDITS',
            'CREDIT/HOUR'
        ]
    )
    return df_warehouses.astype({
        "WAREHOUSE": str,
        "SIZE": str,
        "HOURS/DAY": int,
        "DAYS/WEEK": int,
        "HOURS/YEAR": int,
        "COMPUTE CREDITS": int,
        "CLOUD SERVICES %": int,
        "CLOUD SERVICES CREDITS": int,
        "CREDIT/HOUR": int
    })


size_df = warehouse_df['SIZE'].drop_duplicates()

m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #1F618D;
    color:#ffffff;
}
div.stButton > button:hover {
    background-color: #87CEFA;
    color:#F1C800;
    }
</style>""", unsafe_allow_html=True)

with (st.sidebar):
    # local variables
    unique_provider = df_providers_regions['Provider'].unique()

    st.header("Step 1: Select Edition")
    st.write('Snowflake offers three different data warehouse as-a-service editions, each providing progressively more features')
    input_edition = st.selectbox('Edition', df_edition_credits['Edition'], index=2)

    st.subheader("Deployment/Cloud")
    st.write('Select a Deployment and Cloud for the warehouses below')

    deployment_cloud_columns = st.columns([1, 2])
    with deployment_cloud_columns[0]:
        input_provider = st.selectbox('Provider', options=unique_provider)
    with deployment_cloud_columns[1]:
        input_region = st.selectbox('Region', options=sorted(df_providers_regions[df_providers_regions['Provider']
                                                                                  == input_provider]['Region']))
        
    st.write("                                                                                             ")


    st.header("Step 2: Specify Storage")
    st.write(" Snowflake provides the option to pre-purchase Capacity. The Capacity purchased is then consumed on a monthly basis")
    # TODO: change step to float, so can increment by half a Tb?
    tb_per_month = st.number_input("Tb/Month", min_value=0, value=0, step=1)
    warehouse_size_table = pd.DataFrame.from_dict(snowflake_warehouse_sizes)




    if 'data' not in st.session_state:
        data = pd.DataFrame({'Warehouse': [], 'Size': [], 'Hrs/Day': [], 'Days/Week': []})
        st.session_state.data = data
    data = st.session_state.data

    if 'data1' not in st.session_state:
        data1 = pd.DataFrame({'WAREHOUSE': [], 'SIZE': [], 'ESTIMATED HOURS': [], 'CREDIT/HOUR': [], 'CREDITS': []})
        st.session_state.data1 = data1
    data1 = st.session_state.data1


    st.write("                                                                                             ")
 

    st.header("Step 3: Specify Compute")
    

  
    warehouse_col, name_col= st.columns([0.5, 0.5])

    with warehouse_col:
        input_size = st.selectbox('VIRTUAL WAREHOUSE SIZE', size_df)

    with name_col:
        input_warehouse = st.text_input('WAREHOUSE NAME')


    df_warehouse_sizes = pd.DataFrame({
    'XS': ['1'],
    'S': ['2'],
    'M': ['4'],
    'L': ['8'],
    'XL': ['16'],
    '2XL': ['32'],
    '3XL': ['64'],
    '4XL': ['128']

})

    
    st.write('Snowflake Virtual Warehouse sizes and credits per hour')
    st.dataframe(df_warehouse_sizes, hide_index=True, use_container_width=True)


 

    column_hours_per_day, column_days_per_week, column_hours_per_year = st.columns([1, 1, 1])
    column_compute_credits, column_cloud_services, column_cloud_service_credits = st.columns([1, 1, 1])


    with column_hours_per_day:
        input_hours_per_day = st.number_input('HOURS/DAY', min_value=0, max_value=24, value=8)

    with column_days_per_week:
        input_days_per_week = st.number_input('DAYS/WEEK', min_value=1, max_value=7, value=5)

    with column_hours_per_year:
        input_hours_per_year = st.text_input('HOURS/YEAR', value=input_hours_per_day * input_days_per_week * 52,
                                             disabled=True)

    with column_compute_credits:
        input_compute_credits = st.text_input('COMPUTE CREDITS',
                                              value=int(input_hours_per_year) * snowflake_warehouse_sizes[input_size][
                                                  0],
                                              disabled=True)

    with column_cloud_services:
        input_cloud_services = st.number_input('CLOUD SERVICES %', min_value=1, max_value=100, value=10,
                                               disabled=True)
        # Remove disabled=True if we need to allow cloud services adjustments in a future iteration.

    with column_cloud_service_credits:
        input_cloud_service_credits = st.text_input('CLOUD SERVICES CREDITS',
                                                    int(input_cloud_services * (float(input_compute_credits) / 100.0)),
                                                    disabled=True)

    all_inputs = (
        input_warehouse,
        input_size,
        input_hours_per_day,
        input_days_per_week,
        input_hours_per_year,
        input_compute_credits,
        input_cloud_services,
        input_cloud_service_credits,
        snowflake_warehouse_sizes[input_size][0]
    )

    st.button('Add Warehouse', disabled=(not input_warehouse),
              on_click=add_warehouse_entry, args=all_inputs)
    
    st.write("                                                                                             ")

    if input_edition == 'Business Critical' or  input_edition ==  'Enterprise':

        with st.container():
            input_column_multi_clustered = st.checkbox('MULTI-CLUSTERED')

        mc_start_time, mc_end_time = st.columns([0.75, 0.75])
        mc_column_hours_per_day, mc_column_days_per_week, mc_column_hours_per_year = st.columns([0.75, 0.75, 0.75])

        with mc_start_time:
            input_mc_start_time = st.number_input(
                'START TIME',
                min_value=0,
                max_value=24,
                value=8,
                step=1,
                disabled=(not input_column_multi_clustered)
            )
        with mc_end_time:
            input_mc_end_time = st.number_input(
                'END TIME',
                min_value=input_mc_start_time,
                max_value=(input_mc_start_time + input_hours_per_day) or 24,
                step=1,
                value=input_mc_start_time + 1,
                disabled=(not input_column_multi_clustered)
            )
        with mc_column_hours_per_day:
            input_mc_hours_per_day = st.number_input(
                'HOURS/DAY',
                value=input_mc_end_time - input_mc_start_time,
                disabled=True
            )
        with mc_column_days_per_week:
            input_mc_days_per_week = st.number_input(
                'DAYS/WEEK',
                min_value=1,
                max_value=input_days_per_week,
                value=1,
                step=1,
                key=98,
                disabled=(not input_column_multi_clustered)
            )
        with mc_column_hours_per_year:
            input_mc_hours_per_year = st.text_input(
                'HOURS/YEAR',
                value=input_mc_hours_per_day * input_mc_days_per_week * 52,
                key=99,
                disabled=True
            )

        mc_cloud_compute_credits = int(input_mc_hours_per_year) * snowflake_warehouse_sizes[input_size][0]
        mc_cloud_service_credits = int(input_cloud_services * (float(mc_cloud_compute_credits) / 100.0))

        all_mc_inputs = (
            input_warehouse,
            input_size,
            input_mc_start_time,
            input_mc_end_time,
            input_mc_hours_per_day,
            input_mc_days_per_week,
            input_mc_hours_per_year,
            mc_cloud_compute_credits,
            mc_cloud_service_credits
        )

        st.button('Add Cluster', disabled=(not input_column_multi_clustered),
                on_click=add_multi_cluster_entry, args=all_mc_inputs)

    st.write("                                                                                             ")


    st.header("Step 4: Estimated Usage")

    with st.container():
        if st.session_state['warehouse_entries']:
            st.write("Warehouse Entries")
            st.dataframe(get_added_warehouses().T, use_container_width=True)

        if st.session_state['cluster_entries']:
            st.write("Warehouse Multi-Cluster Entries")
            st.dataframe(get_added_clusters().T, use_container_width=True)

    st.button('Reset Selections', on_click=reset)

st.header('Edition: ' + input_edition)


if input_edition == 'Standard':
    with st.expander("**Storage**", expanded=True):
        storage = tb_per_month * 12.0
        storage_value = storage * STORAGE_PRICE_PER_TB
        capacity_value = storage * CAPACITY_PRICE_PER_TB
        storage_str = '$'+ str(STORAGE_PRICE_PER_TB)
        capacity_str = '$'+ str(CAPACITY_PRICE_PER_TB)
        mdf = pd.DataFrame({'TB/YR': str(storage), 'ON-DEMAND STORAGE/TB': storage_str,
                            'ON-DEMAND TOTAL': str(storage_value), 'CAPACITY STORAGE/TB': capacity_str,'CAPACITY STORAGE TOTAL' : str(capacity_value)}, index=[0])

        st.dataframe(
            mdf,
            column_config={
                "STORAGE PRICE PER TB": st.column_config.NumberColumn(
                    "STORAGE PRICE PER TB",
                    help="Storage/Per TB in USD",
                    step=1,
                    format="$%d",
                ),
                "STORAGE VALUE": st.column_config.NumberColumn(
                    "STORAGE VALUE",
                    help="Storage Value in USD",
                    step=1,
                    format="$%d",
                ),
            },
            hide_index=True,
        )
    with (st.expander("**Warehouse Credits**", expanded=True)):
        if st.session_state['warehouse_entries']:
            df_added_warehouses = get_added_warehouses()

            df_added_clusters = pd.DataFrame({'A': []})
            if st.session_state['cluster_entries']:
                df_added_clusters = get_added_clusters()

            if not df_added_clusters.empty:
                select_columns_cluster_credits = df_added_clusters[
                    ["WAREHOUSE", "COMPUTE CREDITS", "CLOUD SERVICES CREDITS"]]
                compute_sum_cluster_credits = select_columns_cluster_credits.groupby("WAREHOUSE").sum().reset_index()
                # Added just for testing / debugging
                # st.write("Testing - Multi-Cluster Credits")
                # st.write("Cluster Sum By Warehouse Name")
                # st.write(compute_sum_cluster_credits)

                df_new_added_warehouse = pd.concat([df_added_warehouses, compute_sum_cluster_credits]).groupby(
                    "WAREHOUSE").sum().reset_index()
                # st.write("New Credit Sums - Adding Cluster Contributions")
                # st.write(df_new_added_warehouse)
                df_added_warehouses = df_new_added_warehouse

            # cloud services credit adjustment for easier sum.
            df_added_warehouses['CSA'] = df_added_warehouses['CLOUD SERVICES CREDITS'].where(
                df_added_warehouses['CLOUD SERVICES %'].gt(10.0),
                -df_added_warehouses['CLOUD SERVICES CREDITS']
            ).clip(upper=0)

            select_columns_warehouse_credits = df_added_warehouses[["WAREHOUSE", "SIZE", "HOURS/YEAR", "COMPUTE CREDITS", "CREDIT/HOUR"]]
            select_columns_warehouse_credits.rename({'HOURS/YEAR': 'ESTIMATED HOURS', 'COMPUTE CREDITS': 'CREDITS'}, axis=1,
                                                    inplace=True
                                                    )

            df_warehouse_credits = select_columns_warehouse_credits[["WAREHOUSE", "SIZE", "ESTIMATED HOURS", "CREDIT/HOUR", "CREDITS"]] \
                .astype({
                "WAREHOUSE" : str,
                "SIZE": str,
                "ESTIMATED HOURS": int,
                "CREDIT/HOUR": int,
                "CREDITS": int
            })

            total_cloud_services_credits = df_added_warehouses['CLOUD SERVICES CREDITS'].sum()
            total_cloud_services_credits_adjustments = df_added_warehouses['CSA'].sum()
            total_estimate_credits = df_added_warehouses['COMPUTE CREDITS'].sum()

            df_warehouse_credits_totals = pd.DataFrame(
                [
                    ["Cloud Services Credits:", "", "", "",total_cloud_services_credits],
                    ["Cloud Services Adjustments:", "", "","", total_cloud_services_credits_adjustments],
                    ["Total Estimated Credits:", "", "","", total_cloud_services_credits +
                    total_cloud_services_credits_adjustments +
                    total_estimate_credits]
                ],
                columns=[
                    'WAREHOUSE',
                    'SIZE',
                    "ESTIMATED HOURS",
                    "CREDIT/HOUR",
                    "CREDITS"
                ]
            )

            df_warehouse_credits.sort_values(by=['CREDIT/HOUR', 'CREDITS'], inplace=True)

            st.dataframe(pd.concat([df_warehouse_credits, df_warehouse_credits_totals]), hide_index=True)

    with st.expander("**Total**", expanded=True):
        if st.session_state['warehouse_entries']:
            item_credits = df_edition_credits[df_edition_credits['Edition'] == input_edition]['Credit']

            data_df = pd.DataFrame(
                dict(
                    ITEM=[
                        "Storage (" + str(storage) + " * $" + str("{:0,.2f}".format(float(STORAGE_PRICE_PER_TB))) + ")",
                        "Proposed Credits (" + str(total_estimate_credits) + " * $" + str(
                            "{:0,.2f}".format(float(item_credits))) + ")",
                        "Product Capacity Total",
                        "Capacity Total",
                        "Total"
                    ],
                    SUBTOTAL=[
                        int(storage_value),
                        float(total_estimate_credits * item_credits),
                        float(storage_value + (total_estimate_credits * item_credits)),
                        float(storage_value + (total_estimate_credits * item_credits)),
                        float(storage_value + (total_estimate_credits * item_credits))
                    ])
            )

            st.dataframe(
                data_df,
                column_config={
                    "ITEM": st.column_config.TextColumn(
                        "ITEM",
                        width="medium"
                    ),
                    "SUBTOTAL": st.column_config.NumberColumn(
                        "SUBTOTAL",
                        help="The price of the product in USD",
                        width="medium",
                        format="$%10.2f",
                    )
                },
                hide_index=True,
            )

if input_edition == 'Business Critical' or  input_edition ==  'Enterprise':
    with st.expander("**Storage**", expanded=True):
        storage = tb_per_month * 12.0
        storage_value = storage * STORAGE_PRICE_PER_TB
        capacity_value = storage * CAPACITY_PRICE_PER_TB
        storage_str = '$'+ str(STORAGE_PRICE_PER_TB)
        capacity_str = '$'+ str(CAPACITY_PRICE_PER_TB)
        mdf = pd.DataFrame({'TB/YR': str(storage), 'ON-DEMAND STORAGE/TB': storage_str,
                            'ON-DEMAND TOTAL': str(storage_value), 'CAPACITY STORAGE/TB': capacity_str,'CAPACITY STORAGE TOTAL' : str(capacity_value)}, index=[0])

        st.dataframe(
            mdf,
            column_config={
                "STORAGE PRICE PER TB": st.column_config.NumberColumn(
                    "STORAGE PRICE PER TB",
                    help="Storage/Per TB in USD",
                    step=1,
                    format="$%d",
                ),
                "STORAGE VALUE": st.column_config.NumberColumn(
                    "STORAGE VALUE",
                    help="Storage Value in USD",
                    step=1,
                    format="$%d",
                ),
            },
            hide_index=True,
        )
    with (st.expander("**Warehouse Credits**", expanded=True)):
        if st.session_state['warehouse_entries']:
            df_added_warehouses = get_added_warehouses()

            df_added_clusters = pd.DataFrame({'A': []})
            if st.session_state['cluster_entries']:
                df_added_clusters = get_added_clusters()

            if not df_added_clusters.empty:
                select_columns_cluster_credits = df_added_clusters[
                    ["WAREHOUSE", "COMPUTE CREDITS", "CLOUD SERVICES CREDITS"]]
                compute_sum_cluster_credits = select_columns_cluster_credits.groupby("WAREHOUSE").sum().reset_index()
                # Added just for testing / debugging
                # st.write("Testing - Multi-Cluster Credits")
                # st.write("Cluster Sum By Warehouse Name")
                # st.write(compute_sum_cluster_credits)

                df_new_added_warehouse = pd.concat([df_added_warehouses, compute_sum_cluster_credits]).groupby(
                    "WAREHOUSE").sum().reset_index()
                # st.write("New Credit Sums - Adding Cluster Contributions")
                # st.write(df_new_added_warehouse)
                df_added_warehouses = df_new_added_warehouse

            # cloud services credit adjustment for easier sum.
            df_added_warehouses['CSA'] = df_added_warehouses['CLOUD SERVICES CREDITS'].where(
                df_added_warehouses['CLOUD SERVICES %'].gt(10.0),
                -df_added_warehouses['CLOUD SERVICES CREDITS']
            ).clip(upper=0)

            select_columns_warehouse_credits = df_added_warehouses[["WAREHOUSE", "HOURS/YEAR", "COMPUTE CREDITS", "CREDIT/HOUR"]]
            select_columns_warehouse_credits.rename({'HOURS/YEAR': 'ESTIMATED HOURS', 'COMPUTE CREDITS': 'CREDITS'}, axis=1,
                                                    inplace=True
                                                    )

            df_warehouse_credits = select_columns_warehouse_credits[["WAREHOUSE", "ESTIMATED HOURS", "CREDIT/HOUR", "CREDITS"]] \
                .astype({
                "WAREHOUSE": str,
                "ESTIMATED HOURS": int,
                "CREDIT/HOUR": int,
                "CREDITS": int
            })

            total_cloud_services_credits = df_added_warehouses['CLOUD SERVICES CREDITS'].sum()
            total_cloud_services_credits_adjustments = df_added_warehouses['CSA'].sum()
            total_estimate_credits = df_added_warehouses['COMPUTE CREDITS'].sum()

            df_warehouse_credits_totals = pd.DataFrame(
                [
                    ["Cloud Services Credits :", "", "", total_cloud_services_credits],
                    ["Cloud Services Adjustments :", "", "", total_cloud_services_credits_adjustments],
                    ["Total Estimated Credits :", "", "", total_cloud_services_credits +
                    total_cloud_services_credits_adjustments +
                    total_estimate_credits]
                ],
                columns=[
                    "WAREHOUSE",
                    "ESTIMATED HOURS",
                    "CREDIT/HOUR",
                    "CREDITS"
                ]
            )

            df_warehouse_credits.sort_values(by=['CREDIT/HOUR', 'CREDITS'], inplace=True)

            st.dataframe(pd.concat([df_warehouse_credits, df_warehouse_credits_totals]), hide_index=True)

    with st.expander("**Total**", expanded=True):
        if st.session_state['warehouse_entries']:
            item_credits = df_edition_credits[df_edition_credits['Edition'] == input_edition]['Credit']

            data_df = pd.DataFrame(
                dict(
                    ITEM=[
                        "Storage (" + str(storage) + " * $" + str("{:0,.2f}".format(float(STORAGE_PRICE_PER_TB))) + ")",
                        "Proposed Credits (" + str(total_estimate_credits) + " * $" + str(
                            "{:0,.2f}".format(float(item_credits))) + ")",
                        "Product Capacity Total",
                        "Capacity Total",
                        "Total"
                    ],
                    SUBTOTAL=[
                        int(storage_value),
                        float(total_estimate_credits * item_credits),
                        float(storage_value + (total_estimate_credits * item_credits)),
                        float(storage_value + (total_estimate_credits * item_credits)),
                        float(storage_value + (total_estimate_credits * item_credits))
                    ])
            )

            st.dataframe(
                data_df,
                column_config={
                    "ITEM": st.column_config.TextColumn(
                        "ITEM",
                        width="medium"
                    ),
                    "SUBTOTAL": st.column_config.NumberColumn(
                        "SUBTOTAL",
                        help="The price of the product in USD",
                        width="medium",
                        format="$%10.2f",
                    )
                },
                hide_index=True,
            )

st.header('Pricing Estimate')
st.write('For Planning purposes only. Actual pricing may vary.')
url = "https://www.snowflake.com/pricing/pricing-guide/"
st.write("**Snowflake Pricing Guide** [link](%s)" % url)
