import streamlit as st
import pandas as pd
import boto3

st.set_page_config(page_title="PXC Optic Failure Engine", layout="wide")

st.title("PXC Communications: Predictive Transceiver Failure Engine")
st.markdown("Unsupervised AIOps model monitoring digital optical metrics across core Juniper interfaces.")

# Read S3 Gold layer using Streamlit Cloud Secrets
@st.cache_data(ttl=300)
def load_s3_data():
    s3 = boto3.client(
        's3',
        aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
        region_name=st.secrets["AWS_DEFAULT_REGION"]
    )
    bucket_name = st.secrets["S3_BUCKET_NAME"]
    key = 'gold/sfp_anomaly_output.csv'
    
    obj = s3.get_object(Bucket=bucket_name, Key=key)
    df = pd.read_csv(obj['Body'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

try:
    df = load_s3_data()

    # 1. Executive Summary Metrics (Latest state per interface)
    latest_df = df.sort_values('timestamp').groupby('interface').last().reset_index()
    
    total_interfaces = len(latest_df)
    warning_count = len(latest_df[latest_df['health_status'] == 'WARNING_DEGRADATION'])
    critical_count = len(latest_df[latest_df['health_status'] == 'CRITICAL_DEGRADATION'])
    healthy_count = total_interfaces - (warning_count + critical_count)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Monitored Interfaces", total_interfaces)
    col2.metric("Healthy Interfaces", healthy_count)
    col3.metric("Early Warnings (48h Lead)", warning_count, delta="Action Required", delta_color="normal")
    col4.metric("Critical Alerts", critical_count, delta="Immediate Swap Required", delta_color="inverse")

    st.markdown("---")

    # 2. Interface Health Summary Table
    st.subheader("📋 Current Optics Operational Summary")
    
    # Highlight status colors
    def style_status(val):
        if val == 'CRITICAL_DEGRADATION':
            return 'background-color: #ff4b4b; color: white; font-weight: bold;'
        elif val == 'WARNING_DEGRADATION':
            return 'background-color: #ffa500; color: black; font-weight: bold;'
        return 'background-color: #0e1117; color: #00ff7f;'

    display_cols = ['interface', 'health_status', 'laser_bias_ma', 'pre_fec_ber', 'temperature_c', 'anomaly_score', 'timestamp']
    styled_table = latest_df[display_cols].style.applymap(style_status, subset=['health_status'])
    st.dataframe(styled_table, use_container_width=True)

    st.markdown("---")

    # 3. Deep-Dive Historical Telemetry Analysis
    st.subheader("🔍 Historical Telemetry & Anomaly Score Inspection")
    
    selected_iface = st.selectbox("Select Interface to Analyze:", sorted(df['interface'].unique()))
    iface_df = df[df['interface'] == selected_iface].sort_values('timestamp')

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown(f"**Laser Bias Current (mA) — {selected_iface}**")
        st.line_chart(iface_df.set_index('timestamp')['laser_bias_ma'])

    with col_right:
        st.markdown(f"**Isolation Forest Anomaly Score — {selected_iface}**")
        st.line_chart(iface_df.set_index('timestamp')['anomaly_score'])

except Exception as e:
    st.error(f"Unable to load data from AWS S3: {e}")
