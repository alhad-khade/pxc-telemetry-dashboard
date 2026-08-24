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

    # 1. Historical & Peak Health Assessment per Interface
    status_severity = {
        'CRITICAL_DEGRADATION': 3,
        'WARNING_DEGRADATION': 2,
        'HEALTHY': 1
    }
    df['severity'] = df['health_status'].map(status_severity)

    # Calculate worst state, minimum anomaly score, and exact degradation time
    summary_records = []
    for (device_id, iface), group in df.groupby(['device_id', 'interface']):
        max_sev = group['severity'].max()
        min_score = group['anomaly_score'].min()
        max_bias = group['laser_bias_ma'].max()
        max_ber = group['pre_fec_ber'].max()
        
        # Capture the timestamp when the peak degraded health status was first observed
        if max_sev > 1:
            degraded_rows = group[group['severity'] == max_sev]
            degraded_at = degraded_rows['timestamp'].min()
        else:
            degraded_at = group['timestamp'].max()
            
        summary_records.append({
            'device_id': device_id,
            'interface': iface,
            'max_severity': max_sev,
            'max_laser_bias': max_bias,
            'max_ber': max_ber,
            'min_anomaly_score': min_score,
            'degraded_at': degraded_at
        })

    summary_df = pd.DataFrame(summary_records)
    severity_map = {3: 'CRITICAL_DEGRADATION', 2: 'WARNING_DEGRADATION', 1: 'HEALTHY'}
    summary_df['health_status'] = summary_df['max_severity'].map(severity_map)

    # Calculate KPI Card Totals
    total_interfaces = len(summary_df)
    warning_count = len(summary_df[summary_df['health_status'] == 'WARNING_DEGRADATION'])
    critical_count = len(summary_df[summary_df['health_status'] == 'CRITICAL_DEGRADATION'])
    healthy_count = total_interfaces - (warning_count + critical_count)
    
    # Financial KPI: £15,000 SLA penalty avoided per critical optic outage
    penalties_saved = critical_count * 15000

    # Render Executive KPI Metrics (5 Columns)
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Monitored Interfaces", total_interfaces)
    col2.metric("Healthy Interfaces", healthy_count)
    col3.metric("Early Warnings (48h Lead)", warning_count, delta=f"{warning_count} Caught", delta_color="normal")
    col4.metric("Critical Degradations", critical_count, delta=f"{critical_count} Interventions Needed", delta_color="inverse")
    col5.metric("Est. SLA Penalties Saved", f"£{penalties_saved:,.0f}", delta=f"{critical_count} Outages Avoided", delta_color="normal")

    st.markdown("---")

    # 2. Interface Health Summary Table
    st.subheader("📋 Optics Health & Backtest Summary (35-Day Window)")
    
    def style_status(val):
        if val == 'CRITICAL_DEGRADATION':
            return 'background-color: #ff4b4b; color: white; font-weight: bold;'
        elif val == 'WARNING_DEGRADATION':
            return 'background-color: #ffa500; color: black; font-weight: bold;'
        return 'background-color: #0e1117; color: #00ff7f;'

    # Reordered columns: device_id first, degraded_at replacing last_timestamp
    display_cols = ['device_id', 'interface', 'health_status', 'max_laser_bias', 'max_ber', 'min_anomaly_score', 'degraded_at']
    styled_table = summary_df[display_cols].style.map(style_status, subset=['health_status'])
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
