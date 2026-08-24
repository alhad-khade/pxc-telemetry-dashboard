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

    # 1. Severity mapping
    status_severity = {
        'CRITICAL_DEGRADATION': 3,
        'WARNING_DEGRADATION': 2,
        'HEALTHY': 1
    }
    df['severity'] = df['health_status'].map(status_severity)

    # Calculate worst state, anomaly score, degradation times, and lead time
    summary_records = []
    for (device_id, iface), group in df.groupby(['device_id', 'interface']):
        group = group.sort_values('timestamp')
        max_sev = group['severity'].max()
        min_score = group['anomaly_score'].min()
        max_bias = group['laser_bias_ma'].max()
        max_ber = group['pre_fec_ber'].max()
        
        # Calculate early detection timestamps & lead time
        if max_sev > 1:
            # Timestamp when early warning was first flagged
            first_warning_row = group[group['severity'] >= 2]
            first_detected_at = first_warning_row['timestamp'].min()
            
            # Timestamp when critical degradation occurred (if applicable)
            critical_row = group[group['severity'] == 3]
            if not critical_row.empty:
                degraded_at = critical_row['timestamp'].min()
                lead_time_td = degraded_at - first_detected_at
                lead_time_hrs = f"{round(lead_time_td.total_seconds() / 3600, 1)} hrs"
            else:
                degraded_at = first_detected_at
                latest_ts = group['timestamp'].max()
                lead_time_td = latest_ts - first_detected_at
                lead_time_hrs = f">{round(lead_time_td.total_seconds() / 3600, 1)} hrs"
        else:
            first_detected_at = pd.NaT
            degraded_at = group['timestamp'].max()
            lead_time_hrs = "N/A (Healthy)"

        summary_records.append({
            'device_id': device_id,
            'interface': iface,
            'max_severity': max_sev,
            'max_laser_bias': max_bias,
            'max_ber': max_ber,
            'min_anomaly_score': min_score,
            'first_detected_at': first_detected_at,
            'degraded_at': degraded_at,
            'advance_notice_hrs': lead_time_hrs
        })

    summary_df = pd.DataFrame(summary_records)
    severity_map = {3: 'CRITICAL_DEGRADATION', 2: 'WARNING_DEGRADATION', 1: 'HEALTHY'}
    summary_df['health_status'] = summary_df['max_severity'].map(severity_map)

    # Sort summary to bring degraded interfaces to the top for executive review
    summary_df = summary_df.sort_values(by=['max_severity', 'device_id'], ascending=[False, True])

    # Calculate KPI Card Totals
    total_interfaces = len(summary_df)
    warning_count = len(summary_df[summary_df['health_status'] == 'WARNING_DEGRADATION'])
    critical_count = len(summary_df[summary_df['health_status'] == 'CRITICAL_DEGRADATION'])
    healthy_count = len(summary_df[summary_df['health_status'] == 'HEALTHY'])
    
    # Financial KPI: £15,000 SLA penalty avoided per critical optic outage
    penalties_saved = critical_count * 15000

    # Render Executive KPI Metrics (5 Columns)
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Monitored Interfaces", total_interfaces)
    col2.metric("Healthy Interfaces", healthy_count)
    col3.metric("Early Warnings", warning_count, delta=f"{warning_count} Actionable", delta_color="normal")
    col4.metric("Critical Degradations", critical_count, delta=f"{critical_count} Interventions", delta_color="inverse")
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

    # Leading device_id + early detection lead time metrics
    display_cols = [
        'device_id', 
        'interface', 
        'health_status', 
        'advance_notice_hrs',
        'first_detected_at',
        'max_laser_bias', 
        'max_ber', 
        'min_anomaly_score'
    ]
    
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
