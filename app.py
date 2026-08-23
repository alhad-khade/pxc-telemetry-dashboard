import streamlit as st
import pandas as pd
import boto3

st.set_page_config(page_title="PXC Optic Failure Engine", layout="wide")

st.title("PXC Communications: Predictive Transceiver Failure Engine")
st.markdown("Real-time telemetry anomaly detection for core and CPE optical interfaces.")

# Read S3 using Streamlit Cloud Secrets
@st.cache_data(ttl=300) # Refreshes data every 5 minutes
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
    return pd.read_csv(obj['Body'])

try:
    df = load_s3_data()

    # Executive Summary Metrics
    col1, col2, col3 = st.columns(3)
    total_monitored = len(df['interface'].unique())
    failing_count = len(df[df['health_status'] == 'CRITICAL_DEGRADATION']['interface'].unique())
    
    col1.metric("Interfaces Monitored", total_monitored)
    col2.metric("Critical Degradation Alerts", failing_count, delta="-1 Action Required")
    col3.metric("Estimated SLA Penalties Avoided", f"£{failing_count * 15000:,.0f}")

    st.markdown("---")

    # Critical Alerts Table
    st.subheader("⚠️ Interfaces Requiring Proactive Replacement")
    critical_df = df[df['health_status'] == 'CRITICAL_DEGRADATION']
    st.dataframe(
        critical_df[['timestamp', 'device_id', 'interface', 'laser_bias_ma', 'pre_fec_ber', 'health_status']], 
        use_container_width=True
    )

    # Telemetry Visualizations
    st.subheader("📈 Laser Bias Current (mA) - Degradation Signal")
    st.line_chart(df.pivot(index='timestamp', columns='interface', values='laser_bias_ma'))

except Exception as e:
    st.error(f"Unable to load data from AWS S3: {e}")
