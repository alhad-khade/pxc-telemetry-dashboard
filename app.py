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

    # Sidebar Filter: Active Alerts vs Historical Window
    st.sidebar.header("Dashboard Configuration")
    view_mode = st.sidebar.radio(
        "Evaluation Perspective:",
        ["Active Alerts (Current State)", "Historical Lifetime Max (35-Day Window)"],
        index=0,
        help="Active Mode evaluates the latest telemetry state. Historical Mode maps the peak severity over 35 days."
    )

    # 1. Severity mapping
    status_severity = {
        'CRITICAL_DEGRADATION': 3,
        'WARNING_DEGRADATION': 2,
        'HEALTHY': 1
    }
    df['severity'] = df['health_status'].map(status_severity)

    # Aggregation per interface
    summary_records = []
    for (device_id, iface), group in df.groupby(['device_id', 'interface']):
        group = group.sort_values('timestamp')
        
        # Latest telemetry point for active evaluation
        latest_row = group.iloc[-1]
        current_sev = latest_row['severity']
        
        max_sev = group['severity'].max()
        min_score = group['anomaly_score'].min()
        max_bias = group['laser_bias_ma'].max()
        max_ber = group['pre_fec_ber'].max()
        
        # Select target severity depending on perspective selection
        evaluated_sev = current_sev if "Active Alerts" in view_mode else max_sev
        
        # Calculate early detection timestamps & lead time
        if max_sev > 1:
            first_warning_row = group[group['severity'] >= 2]
            first_detected_at = first_warning_row['timestamp'].min()
            
            critical_row = group[group['severity'] == 3]
            if not critical_row.empty:
                degraded_at = critical_row['timestamp'].min()
            else:
                degraded_at = group['timestamp'].max()

            lead_time_td = degraded_at - first_detected_at
            days_val = round(lead_time_td.total_seconds() / 86400, 1)
            
            # Format lead time clearly (Hours if < 1 day, Days if >= 1 day)
            if days_val < 1.0:
                hours_val = round(lead_time_td.total_seconds() / 3600, 1)
                advance_notice = f"{hours_val} hrs"
            else:
                advance_notice = f"{days_val} days"
        else:
            first_detected_at = pd.NaT
            degraded_at = group['timestamp'].max()
            advance_notice = "N/A (Healthy)"

        summary_records.append({
            'device_id': device_id,
            'interface': iface,
            'evaluated_severity': evaluated_sev,
            'max_severity': max_sev,
            'max_laser_bias': max_bias,
            'max_ber': max_ber,
            'min_anomaly_score': min_score,
            'first_detected_at': first_detected_at,
            'degraded_at': degraded_at,
            'advance_notice_days': advance_notice
        })

    summary_df = pd.DataFrame(summary_records)
    severity_map = {3: 'CRITICAL_DEGRADATION', 2: 'WARNING_DEGRADATION', 1: 'HEALTHY'}
    summary_df['health_status'] = summary_df['evaluated_severity'].map(severity_map)

    # Sort summary to surface degraded interfaces first
    summary_df = summary_df.sort_values(by=['evaluated_severity', 'device_id'], ascending=[False, True])

    # Executive Metric KPIs
    total_interfaces = len(summary_df)
    warning_count = len(summary_df[summary_df['health_status'] == 'WARNING_DEGRADATION'])
    critical_count = len(summary_df[summary_df['health_status'] == 'CRITICAL_DEGRADATION'])
    healthy_count = len(summary_df[summary_df['health_status'] == 'HEALTHY'])
    
    # Financial KPI (£15,000 per critical optic failure avoided)
    penalties_saved = critical_count * 15000

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Monitored Interfaces", total_interfaces)
    col2.metric("Healthy Interfaces", healthy_count)
    col3.metric("Early Warnings", warning_count, delta=f"{warning_count} Actionable", delta_color="normal")
    col4.metric("Critical Degradations", critical_count, delta=f"{critical_count} Interventions", delta_color="inverse")
    col5.metric("Est. SLA Penalties Saved", f"£{penalties_saved:,.0f}", delta=f"{critical_count} Outages Avoided", delta_color="normal")

    st.markdown("---")

    # 2. Interface Summary Table
    st.subheader(f"📋 Optics Health Summary ({view_mode})")
    
    def style_status(val):
        if val == 'CRITICAL_DEGRADATION':
            return 'background-color: #ff4b4b; color: white; font-weight: bold;'
        elif val == 'WARNING_DEGRADATION':
            return 'background-color: #ffa500; color: black; font-weight: bold;'
        return 'background-color: #0e1117; color: #00ff7f;'

    display_cols = [
        'device_id', 
        'interface', 
        'health_status', 
        'advance_notice_days',
        'first_detected_at',
        'max_laser_bias', 
        'max_ber', 
        'min_anomaly_score'
    ]
    
    styled_table = summary_df[display_cols].style.map(style_status, subset=['health_status'])
    st.dataframe(styled_table, use_container_width=True)

    st.markdown("---")

    # 3. Cascading Inspection Dropdowns (Device ID -> Interface)
    st.subheader("🔍 Historical Telemetry & Anomaly Score Inspection")
    
    col_dev, col_iface = st.columns(2)
    
    with col_dev:
        device_list = sorted(df['device_id'].unique())
        selected_device = st.selectbox("1. Select Device ID:", device_list)

    with col_iface:
        # Dynamically filter available interfaces based on selected Device ID
        available_ifaces = sorted(df[df['device_id'] == selected_device]['interface'].unique())
        selected_iface = st.selectbox("2. Select Interface:", available_ifaces)

    # Filter chart data for selected Device + Interface
    iface_df = df[(df['device_id'] == selected_device) & (df['interface'] == selected_iface)].sort_values('timestamp')

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown(f"**Laser Bias Current (mA) — {selected_device} [{selected_iface}]**")
        st.line_chart(iface_df.set_index('timestamp')['laser_bias_ma'])

    with col_right:
        st.markdown(f"**Isolation Forest Anomaly Score — {selected_device} [{selected_iface}]**")
        st.line_chart(iface_df.set_index('timestamp')['anomaly_score'])

except Exception as e:
    st.error(f"Unable to load data from AWS S3: {e}")
