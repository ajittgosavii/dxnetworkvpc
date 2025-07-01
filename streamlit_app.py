import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import math
from typing import Dict, List, Tuple, Optional

# Configure page
st.set_page_config(
    page_title="AWS Network Migration Analyzer",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 50%, #1d4ed8 100%);
        padding: 2rem;
        border-radius: 8px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 2px 10px rgba(30,58,138,0.1);
    }
    
    .network-card {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        padding: 1.5rem;
        border-radius: 6px;
        color: #374151;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border-left: 3px solid #22c55e;
    }
    
    .performance-card {
        background: linear-gradient(135deg, #fef7f0 0%, #fed7aa 100%);
        padding: 1.5rem;
        border-radius: 6px;
        color: #374151;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border-left: 3px solid #f97316;
    }
    
    .agent-card {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        padding: 1.5rem;
        border-radius: 6px;
        color: #374151;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border-left: 3px solid #64748b;
    }
    
    .warning-card {
        background: linear-gradient(135deg, #fefce8 0%, #fef3c7 100%);
        padding: 1.5rem;
        border-radius: 6px;
        color: #374151;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border-left: 3px solid #f59e0b;
    }
</style>
""", unsafe_allow_html=True)

class NetworkPathManager:
    """Manage network paths for production and non-production environments"""
    
    def __init__(self):
        self.network_paths = {
            'nonprod_sj_linux_nas_s3': {
                'name': 'Non-Prod: San Jose Linux NAS → AWS S3',
                'destination_storage': 'S3',
                'source': 'San Jose',
                'destination': 'AWS US-West-2 S3',
                'environment': 'non-production',
                'os_type': 'linux',
                'storage_type': 'nas',
                'segments': [
                    {
                        'name': 'Linux NAS to Linux Jump Server',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 2,
                        'reliability': 0.999,
                        'connection_type': 'internal_lan',
                        'cost_factor': 0.0,
                        'optimization_potential': 0.95
                    },
                    {
                        'name': 'Linux Jump Server to AWS S3 (DX)',
                        'bandwidth_mbps': 2000,  # Non-prod DX bottleneck
                        'latency_ms': 15,
                        'reliability': 0.998,
                        'connection_type': 'direct_connect',
                        'cost_factor': 2.0,
                        'optimization_potential': 0.92
                    }
                ]
            },
            'nonprod_sj_windows_share_s3': {
                'name': 'Non-Prod: San Jose Windows Share → AWS S3',
                'destination_storage': 'S3',
                'source': 'San Jose',
                'destination': 'AWS US-West-2 S3',
                'environment': 'non-production',
                'os_type': 'windows',
                'storage_type': 'share',
                'segments': [
                    {
                        'name': 'Windows Share to Windows Jump Server',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 3,
                        'reliability': 0.997,
                        'connection_type': 'internal_lan',
                        'cost_factor': 0.0,
                        'optimization_potential': 0.88
                    },
                    {
                        'name': 'Windows Jump Server to AWS S3 (DX)',
                        'bandwidth_mbps': 2000,  # Non-prod DX bottleneck
                        'latency_ms': 18,
                        'reliability': 0.998,
                        'connection_type': 'direct_connect',
                        'cost_factor': 2.0,
                        'optimization_potential': 0.90
                    }
                ]
            },
            'prod_sa_linux_nas_s3': {
                'name': 'Prod: San Antonio Linux NAS → San Jose → AWS Production VPC S3',
                'destination_storage': 'S3',
                'source': 'San Antonio',
                'destination': 'AWS US-West-2 Production VPC S3',
                'environment': 'production',
                'os_type': 'linux',
                'storage_type': 'nas',
                'segments': [
                    {
                        'name': 'San Antonio Linux NAS to Linux Jump Server',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 1,
                        'reliability': 0.999,
                        'connection_type': 'internal_lan',
                        'cost_factor': 0.0,
                        'optimization_potential': 0.97
                    },
                    {
                        'name': 'San Antonio to San Jose (Private Line)',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 12,
                        'reliability': 0.9995,
                        'connection_type': 'private_line',
                        'cost_factor': 3.0,
                        'optimization_potential': 0.94
                    },
                    {
                        'name': 'San Jose to AWS Production VPC S3 (DX)',
                        'bandwidth_mbps': 10000,  # Full 10Gbps in production
                        'latency_ms': 8,
                        'reliability': 0.9999,
                        'connection_type': 'direct_connect',
                        'cost_factor': 4.0,
                        'optimization_potential': 0.96
                    }
                ]
            },
            'prod_sa_windows_share_s3': {
                'name': 'Prod: San Antonio Windows Share → San Jose → AWS Production VPC S3',
                'destination_storage': 'S3',
                'source': 'San Antonio',
                'destination': 'AWS US-West-2 Production VPC S3',
                'environment': 'production',
                'os_type': 'windows',
                'storage_type': 'share',
                'segments': [
                    {
                        'name': 'San Antonio Windows Share to Windows Jump Server',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 2,
                        'reliability': 0.997,
                        'connection_type': 'internal_lan',
                        'cost_factor': 0.0,
                        'optimization_potential': 0.88
                    },
                    {
                        'name': 'San Antonio to San Jose (Private Line)',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 15,
                        'reliability': 0.9995,
                        'connection_type': 'private_line',
                        'cost_factor': 3.0,
                        'optimization_potential': 0.92
                    },
                    {
                        'name': 'San Jose to AWS Production VPC S3 (DX)',
                        'bandwidth_mbps': 10000,  # Full 10Gbps in production
                        'latency_ms': 10,
                        'reliability': 0.9999,
                        'connection_type': 'direct_connect',
                        'cost_factor': 4.0,
                        'optimization_potential': 0.94
                    }
                ]
            }
        }
    
    def get_network_path_key(self, config: Dict) -> str:
        """Get network path key based on configuration"""
        # Determine OS type
        os_lower = config['operating_system'].lower()
        if any(os_name in os_lower for os_name in ['linux', 'ubuntu', 'rhel', 'centos']):
            os_type = 'linux'
        elif 'windows' in os_lower:
            os_type = 'windows'
        else:
            os_type = 'linux'
        
        # Determine environment
        environment = config['environment']
        
        # Build path key
        if environment == 'non-production':
            if os_type == 'linux':
                return 'nonprod_sj_linux_nas_s3'
            else:
                return 'nonprod_sj_windows_share_s3'
        else:  # production
            if os_type == 'linux':
                return 'prod_sa_linux_nas_s3'
            else:
                return 'prod_sa_windows_share_s3'
    
    def calculate_network_performance(self, path_key: str, time_of_day: int = None) -> Dict:
        """Calculate network performance with time-of-day adjustments"""
        path = self.network_paths[path_key]
        
        if time_of_day is None:
            time_of_day = datetime.now().hour
        
        total_latency = 0
        min_bandwidth = float('inf')
        total_reliability = 1.0
        total_cost_factor = 0
        optimization_score = 1.0
        
        adjusted_segments = []
        
        for segment in path['segments']:
            # Base metrics
            segment_latency = segment['latency_ms']
            segment_bandwidth = segment['bandwidth_mbps']
            segment_reliability = segment['reliability']
            
            # Time-of-day congestion adjustments
            if segment['connection_type'] == 'internal_lan':
                congestion_factor = 1.1 if 9 <= time_of_day <= 17 else 0.95
            elif segment['connection_type'] == 'private_line':
                congestion_factor = 1.2 if 9 <= time_of_day <= 17 else 0.9
            elif segment['connection_type'] == 'direct_connect':
                congestion_factor = 1.05 if 9 <= time_of_day <= 17 else 0.98
            else:
                congestion_factor = 1.0
            
            # Apply congestion
            effective_bandwidth = segment_bandwidth / congestion_factor
            effective_latency = segment_latency * congestion_factor
            
            # OS-specific adjustments
            if path['os_type'] == 'windows' and segment['connection_type'] != 'internal_lan':
                effective_bandwidth *= 0.95
                effective_latency *= 1.1
            
            optimization_score *= segment['optimization_potential']
            
            # Accumulate metrics
            total_latency += effective_latency
            min_bandwidth = min(min_bandwidth, effective_bandwidth)
            total_reliability *= segment_reliability
            total_cost_factor += segment['cost_factor']
            
            adjusted_segments.append({
                **segment,
                'effective_bandwidth_mbps': effective_bandwidth,
                'effective_latency_ms': effective_latency,
                'congestion_factor': congestion_factor
            })
        
        # Calculate quality scores
        latency_score = max(0, 100 - (total_latency * 2))
        bandwidth_score = min(100, (min_bandwidth / 1000) * 20)
        reliability_score = total_reliability * 100
        
        network_quality = (latency_score * 0.25 + bandwidth_score * 0.45 + reliability_score * 0.30)
        
        return {
            'path_name': path['name'],
            'destination_storage': path['destination_storage'],
            'environment': path['environment'],
            'os_type': path['os_type'],
            'total_latency_ms': total_latency,
            'effective_bandwidth_mbps': min_bandwidth,
            'total_reliability': total_reliability,
            'network_quality_score': network_quality,
            'optimization_potential': (1 - optimization_score) * 100,
            'total_cost_factor': total_cost_factor,
            'segments': adjusted_segments
        }

class AgentManager:
    """Manage DataSync and DMS agent configurations"""
    
    def __init__(self):
        self.datasync_specs = {
            'small': {'throughput_mbps': 250, 'vcpu': 2, 'memory': 4, 'cost_hour': 0.0416},
            'medium': {'throughput_mbps': 500, 'vcpu': 2, 'memory': 4, 'cost_hour': 0.085},
            'large': {'throughput_mbps': 1000, 'vcpu': 4, 'memory': 8, 'cost_hour': 0.17},
            'xlarge': {'throughput_mbps': 2000, 'vcpu': 8, 'memory': 16, 'cost_hour': 0.34}
        }
        
        self.dms_specs = {
            'small': {'throughput_mbps': 200, 'vcpu': 2, 'memory': 4, 'cost_hour': 0.0416},
            'medium': {'throughput_mbps': 400, 'vcpu': 2, 'memory': 4, 'cost_hour': 0.085},
            'large': {'throughput_mbps': 800, 'vcpu': 4, 'memory': 8, 'cost_hour': 0.17},
            'xlarge': {'throughput_mbps': 1500, 'vcpu': 8, 'memory': 16, 'cost_hour': 0.34},
            'xxlarge': {'throughput_mbps': 2500, 'vcpu': 16, 'memory': 32, 'cost_hour': 0.68}
        }
    
    def calculate_agent_performance(self, agent_type: str, agent_size: str, num_agents: int, 
                                   platform_type: str = 'vmware') -> Dict:
        """Calculate agent performance considering VMware overhead"""
        
        if agent_type == 'datasync':
            base_spec = self.datasync_specs[agent_size]
        else:
            base_spec = self.dms_specs[agent_size]
        
        # VMware overhead
        vmware_efficiency = 0.92 if platform_type == 'vmware' else 1.0
        
        # Calculate per-agent performance
        per_agent_throughput = base_spec['throughput_mbps'] * vmware_efficiency
        
        # Calculate scaling efficiency (diminishing returns)
        if num_agents == 1:
            scaling_efficiency = 1.0
        elif num_agents <= 3:
            scaling_efficiency = 0.95
        elif num_agents <= 5:
            scaling_efficiency = 0.90
        else:
            scaling_efficiency = 0.85
        
        # Total agent capacity
        total_agent_throughput = per_agent_throughput * num_agents * scaling_efficiency
        
        # Costs
        per_agent_cost = base_spec['cost_hour'] * 24 * 30
        total_monthly_cost = per_agent_cost * num_agents
        
        return {
            'agent_type': agent_type,
            'agent_size': agent_size,
            'num_agents': num_agents,
            'platform_type': platform_type,
            'per_agent_throughput_mbps': per_agent_throughput,
            'total_agent_throughput_mbps': total_agent_throughput,
            'scaling_efficiency': scaling_efficiency,
            'vmware_efficiency': vmware_efficiency,
            'per_agent_monthly_cost': per_agent_cost,
            'total_monthly_cost': total_monthly_cost,
            'base_spec': base_spec
        }

def get_nic_efficiency(nic_type: str) -> float:
    """Get NIC efficiency based on type"""
    efficiencies = {
        'gigabit_copper': 0.85,
        'gigabit_fiber': 0.90,
        '10g_copper': 0.88,
        '10g_fiber': 0.92,
        '25g_fiber': 0.94,
        '40g_fiber': 0.95
    }
    return efficiencies.get(nic_type, 0.90)

def render_bandwidth_waterfall(config: Dict, network_perf: Dict, agent_perf: Dict):
    """Render bandwidth waterfall analysis"""
    st.markdown("**🌊 Bandwidth Waterfall Analysis: From Hardware to Migration Speed**")
    
    # Start with user's actual hardware
    user_nic_speed = config['nic_speed']
    nic_type = config['nic_type']
    os_type = config['operating_system']
    platform_type = config['server_type']
    
    # Step 1: Raw NIC Capacity
    stages = ['Your NIC\nCapacity']
    throughputs = [user_nic_speed]
    descriptions = [f"{user_nic_speed:,.0f} Mbps {nic_type.replace('_', ' ')} NIC"]
    
    # Step 2: NIC Hardware Efficiency
    nic_efficiency = get_nic_efficiency(nic_type)
    after_nic = user_nic_speed * nic_efficiency
    stages.append('After NIC\nProcessing')
    throughputs.append(after_nic)
    descriptions.append(f"{nic_type.replace('_', ' ').title()} hardware efficiency")
    
    # Step 3: OS Network Stack
    os_efficiency = 0.90 if 'linux' in os_type else 0.88
    after_os = after_nic * os_efficiency
    stages.append('After OS\nNetwork Stack')
    throughputs.append(after_os)
    descriptions.append(f"{os_type.replace('_', ' ').title()} network processing")
    
    # Step 4: VMware Virtualization
    if platform_type == 'vmware':
        vmware_efficiency = 0.92
        after_vmware = after_os * vmware_efficiency
        stages.append('After VMware\nVirtualization')
        throughputs.append(after_vmware)
        descriptions.append('VMware hypervisor overhead')
    else:
        after_vmware = after_os
    
    # Step 5: Protocol Overhead
    protocol_efficiency = 0.82 if config['environment'] == 'production' else 0.85
    after_protocol = after_vmware * protocol_efficiency
    stages.append('After Protocol\nOverhead')
    throughputs.append(after_protocol)
    descriptions.append(f"{config['environment'].title()} security protocols")
    
    # Step 6: Network Path Limitation
    network_bandwidth = network_perf['effective_bandwidth_mbps']
    after_network = min(after_protocol, network_bandwidth)
    network_is_bottleneck = after_protocol > network_bandwidth
    stages.append('After Network\nPath Limit')
    throughputs.append(after_network)
    descriptions.append(f"Network path: {network_bandwidth:,.0f} Mbps available")
    
    # Step 7: Agent Processing
    agent_capacity = agent_perf['total_agent_throughput_mbps']
    final_throughput = min(after_network, agent_capacity)
    stages.append('Final Migration\nThroughput')
    throughputs.append(final_throughput)
    descriptions.append(f"{agent_perf['num_agents']}x {agent_perf['agent_type'].upper()} agents")
    
    # Create visualization
    waterfall_data = pd.DataFrame({
        'Stage': stages,
        'Throughput (Mbps)': throughputs,
        'Description': descriptions
    })
    
    fig = px.bar(
        waterfall_data,
        x='Stage',
        y='Throughput (Mbps)',
        title=f"Bandwidth Analysis: {user_nic_speed:,.0f} Mbps Hardware → {final_throughput:.0f} Mbps Migration Speed",
        text='Throughput (Mbps)',
        color='Throughput (Mbps)',
        color_continuous_scale='RdYlGn'
    )
    
    fig.update_traces(texttemplate='%{text:.0f} Mbps', textposition='outside')
    fig.update_layout(height=500, showlegend=False)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Analysis summary
    total_loss = user_nic_speed - final_throughput
    total_loss_pct = (total_loss / user_nic_speed) * 100
    
    if network_is_bottleneck:
        st.warning(f"""
        ⚠️ **Network Infrastructure Bottleneck Detected:**
        • **Your Hardware:** {user_nic_speed:,.0f} Mbps {nic_type.replace('_', ' ')} NIC
        • **Network Limitation:** {network_bandwidth:,.0f} Mbps ({config['environment']} environment)
        • **Final Migration Speed:** {final_throughput:.0f} Mbps
        • **Total Efficiency Loss:** {total_loss_pct:.1f}%
        
        💡 **Recommendation:** Plan migration times using {final_throughput:.0f} Mbps actual speed
        """)
    else:
        if agent_capacity < after_network:
            st.error(f"""
            🔍 **Agent Bottleneck Detected:**
            • **Available Bandwidth:** {after_network:,.0f} Mbps
            • **Agent Capacity:** {agent_capacity:,.0f} Mbps
            • **Final Migration Speed:** {final_throughput:.0f} Mbps
            • **Recommendation:** Scale agents or optimize configuration
            """)
        else:
            st.success(f"""
            ✅ **Optimal Configuration:**
            • **Hardware Capacity:** {user_nic_speed:,.0f} Mbps
            • **Final Migration Speed:** {final_throughput:.0f} Mbps
            • **Total Efficiency:** {100-total_loss_pct:.1f}%
            """)

def create_network_diagram(network_perf: Dict):
    """Create network path diagram"""
    segments = network_perf.get('segments', [])
    if not segments:
        return None
    
    fig = go.Figure()
    
    num_segments = len(segments)
    x_positions = [i * 100 for i in range(num_segments + 1)]
    y_positions = [50] * (num_segments + 1)
    
    # Add network segments
    for i, segment in enumerate(segments):
        bandwidth = segment.get('effective_bandwidth_mbps', 0)
        latency = segment.get('effective_latency_ms', 0)
        reliability = segment.get('reliability', 0)
        
        line_width = max(2, min(10, bandwidth / 200))
        line_color = '#27ae60' if reliability > 0.999 else '#f39c12' if reliability > 0.995 else '#e74c3c'
        
        fig.add_trace(go.Scatter(
            x=[x_positions[i], x_positions[i+1]],
            y=[y_positions[i], y_positions[i+1]],
            mode='lines+markers',
            line=dict(width=line_width, color=line_color),
            marker=dict(size=15, color='#2c3e50', symbol='square'),
            name=segment['name'],
            hovertemplate=f"""
            <b>{segment['name']}</b><br>
            Bandwidth: {bandwidth:,.0f} Mbps<br>
            Latency: {latency:.1f} ms<br>
            Reliability: {reliability*100:.3f}%<br>
            <extra></extra>
            """
        ))
        
        # Add annotations
        mid_x = (x_positions[i] + x_positions[i+1]) / 2
        mid_y = y_positions[i] + 20
        
        fig.add_annotation(
            x=mid_x, y=mid_y,
            text=f"<b>{bandwidth:,.0f} Mbps</b><br>{latency:.1f} ms",
            showarrow=False,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#bdc3c7',
            borderwidth=1
        )
    
    # Add source and destination
    fig.add_trace(go.Scatter(
        x=[x_positions[0]], y=[y_positions[0]],
        mode='markers+text',
        marker=dict(size=25, color='#27ae60', symbol='circle'),
        text=['SOURCE'], textposition='bottom center',
        name='Source System'
    ))
    
    fig.add_trace(go.Scatter(
        x=[x_positions[-1]], y=[y_positions[-1]],
        mode='markers+text',
        marker=dict(size=25, color='#3498db', symbol='circle'),
        text=['AWS S3'], textposition='bottom center',
        name='AWS Destination'
    ))
    
    fig.update_layout(
        title=f"Network Path: {network_perf.get('path_name', 'Unknown')}",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        showlegend=False,
        height=350
    )
    
    return fig

def render_sidebar():
    """Render sidebar controls"""
    st.sidebar.header("🌐 Network Migration Configuration")
    
    # Operating System
    operating_system = st.sidebar.selectbox(
        "Operating System",
        ["windows_server_2019", "windows_server_2022", "rhel_8", "rhel_9", "ubuntu_20_04", "ubuntu_22_04"],
        index=3,
        format_func=lambda x: {
            'windows_server_2019': '🔵 Windows Server 2019',
            'windows_server_2022': '🔵 Windows Server 2022',
            'rhel_8': '🔴 Red Hat Enterprise Linux 8',
            'rhel_9': '🔴 Red Hat Enterprise Linux 9',
            'ubuntu_20_04': '🟠 Ubuntu Server 20.04 LTS',
            'ubuntu_22_04': '🟠 Ubuntu Server 22.04 LTS'
        }[x]
    )
    
    # Server Platform
    server_type = st.sidebar.selectbox(
        "Server Platform",
        ["physical", "vmware"],
        index=1,  # Default to VMware
        format_func=lambda x: "🏢 Physical Server" if x == "physical" else "☁️ VMware Virtual Machine"
    )
    
    # Hardware Configuration
    st.sidebar.subheader("⚙️ Hardware Configuration")
    ram_gb = st.sidebar.selectbox("RAM (GB)", [8, 16, 32, 64, 128, 256], index=2)
    cpu_cores = st.sidebar.selectbox("CPU Cores", [2, 4, 8, 16, 24, 32], index=2)
    
    # Network Interface
    nic_type = st.sidebar.selectbox(
        "NIC Type",
        ["gigabit_copper", "gigabit_fiber", "10g_copper", "10g_fiber", "25g_fiber", "40g_fiber"],
        index=3,
        format_func=lambda x: {
            'gigabit_copper': '🔶 1Gbps Copper',
            'gigabit_fiber': '🟡 1Gbps Fiber',
            '10g_copper': '🔵 10Gbps Copper',
            '10g_fiber': '🟢 10Gbps Fiber',
            '25g_fiber': '🟣 25Gbps Fiber',
            '40g_fiber': '🔴 40Gbps Fiber'
        }[x]
    )
    
    nic_speeds = {
        'gigabit_copper': 1000, 'gigabit_fiber': 1000,
        '10g_copper': 10000, '10g_fiber': 10000,
        '25g_fiber': 25000, '40g_fiber': 40000
    }
    nic_speed = nic_speeds[nic_type]
    
    # Migration Configuration
    st.sidebar.subheader("🔄 Migration Configuration")
    source_database_engine = st.sidebar.selectbox(
        "Source Database",
        ["mysql", "postgresql", "oracle", "sqlserver", "mongodb"],
        format_func=lambda x: x.upper()
    )
    
    database_engine = st.sidebar.selectbox(
        "Target Database",
        ["mysql", "postgresql", "oracle", "sqlserver", "mongodb"],
        format_func=lambda x: x.upper()
    )
    
    database_size_gb = st.sidebar.number_input("Database Size (GB)", min_value=100, max_value=100000, value=1000, step=100)
    
    # Environment
    environment = st.sidebar.selectbox("Environment", ["non-production", "production"])
    
    # Migration Type
    is_homogeneous = source_database_engine == database_engine
    migration_type = "DataSync" if is_homogeneous else "DMS"
    
    st.sidebar.info(f"**Migration Tool:** {migration_type}")
    
    # Agent Configuration
    st.sidebar.subheader("🤖 Agent Configuration")
    number_of_agents = st.sidebar.number_input("Number of Agents", min_value=1, max_value=10, value=2, step=1)
    
    if is_homogeneous:
        agent_size = st.sidebar.selectbox(
            "DataSync Agent Size",
            ["small", "medium", "large", "xlarge"],
            index=1,
            format_func=lambda x: {
                'small': '📦 Small (250 Mbps/agent)',
                'medium': '📦 Medium (500 Mbps/agent)',
                'large': '📦 Large (1000 Mbps/agent)',
                'xlarge': '📦 XLarge (2000 Mbps/agent)'
            }[x]
        )
    else:
        agent_size = st.sidebar.selectbox(
            "DMS Instance Size",
            ["small", "medium", "large", "xlarge", "xxlarge"],
            index=1,
            format_func=lambda x: {
                'small': '🔄 Small (200 Mbps/agent)',
                'medium': '🔄 Medium (400 Mbps/agent)',
                'large': '🔄 Large (800 Mbps/agent)',
                'xlarge': '🔄 XLarge (1500 Mbps/agent)',
                'xxlarge': '🔄 XXLarge (2500 Mbps/agent)'
            }[x]
        )
    
    return {
        'operating_system': operating_system,
        'server_type': server_type,
        'ram_gb': ram_gb,
        'cpu_cores': cpu_cores,
        'nic_type': nic_type,
        'nic_speed': nic_speed,
        'source_database_engine': source_database_engine,
        'database_engine': database_engine,
        'database_size_gb': database_size_gb,
        'environment': environment,
        'number_of_agents': number_of_agents,
        'agent_size': agent_size,
        'migration_type': migration_type.lower(),
        'is_homogeneous': is_homogeneous
    }

def main():
    """Main application"""
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌐 AWS Network Migration Analyzer</h1>
        <p>Comprehensive Network Path Analysis • Bandwidth Optimization • Agent Performance Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get configuration
    config = render_sidebar()
    
    # Initialize managers
    network_manager = NetworkPathManager()
    agent_manager = AgentManager()
    
    # Get network path
    path_key = network_manager.get_network_path_key(config)
    network_perf = network_manager.calculate_network_performance(path_key)
    
    # Get agent performance
    agent_type = 'datasync' if config['is_homogeneous'] else 'dms'
    agent_perf = agent_manager.calculate_agent_performance(
        agent_type, config['agent_size'], config['number_of_agents'], config['server_type']
    )
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌊 Bandwidth Analysis",
        "🌐 Network Paths",
        "🤖 Agent Performance",
        "📊 Performance Comparison"
    ])
    
    with tab1:
        st.subheader("🌊 Bandwidth Waterfall Analysis")
        render_bandwidth_waterfall(config, network_perf, agent_perf)
        
        # Performance impact table
        st.markdown("**📊 Detailed Performance Impact:**")
        
        impact_data = []
        running_throughput = config['nic_speed']
        
        # NIC Processing
        nic_efficiency = get_nic_efficiency(config['nic_type'])
        nic_loss = running_throughput * (1 - nic_efficiency)
        running_throughput *= nic_efficiency
        impact_data.append({
            'Layer': '🔌 NIC Hardware',
            'Component': f"{config['nic_type'].replace('_', ' ').title()}",
            'Throughput (Mbps)': f"{running_throughput:.0f}",
            'Efficiency (%)': f"{nic_efficiency * 100:.1f}%",
            'Loss (Mbps)': f"{nic_loss:.0f}"
        })
        
        # OS Network Stack
        os_efficiency = 0.90 if 'linux' in config['operating_system'] else 0.88
        os_loss = running_throughput * (1 - os_efficiency)
        running_throughput *= os_efficiency
        impact_data.append({
            'Layer': '💻 OS Network Stack',
            'Component': f"{config['operating_system'].replace('_', ' ').title()}",
            'Throughput (Mbps)': f"{running_throughput:.0f}",
            'Efficiency (%)': f"{os_efficiency * 100:.1f}%",
            'Loss (Mbps)': f"{os_loss:.0f}"
        })
        
        # VMware (if applicable)
        if config['server_type'] == 'vmware':
            vmware_efficiency = 0.92
            vmware_loss = running_throughput * (1 - vmware_efficiency)
            running_throughput *= vmware_efficiency
            impact_data.append({
                'Layer': '☁️ VMware Virtualization',
                'Component': 'VMware hypervisor overhead',
                'Throughput (Mbps)': f"{running_throughput:.0f}",
                'Efficiency (%)': f"{vmware_efficiency * 100:.1f}%",
                'Loss (Mbps)': f"{vmware_loss:.0f}"
            })
        
        # Protocol Overhead
        protocol_efficiency = 0.82 if config['environment'] == 'production' else 0.85
        protocol_loss = running_throughput * (1 - protocol_efficiency)
        running_throughput *= protocol_efficiency
        impact_data.append({
            'Layer': '🔗 Protocol Overhead',
            'Component': f"{config['environment'].title()} security protocols",
            'Throughput (Mbps)': f"{running_throughput:.0f}",
            'Efficiency (%)': f"{protocol_efficiency * 100:.1f}%",
            'Loss (Mbps)': f"{protocol_loss:.0f}"
        })
        
        df_impact = pd.DataFrame(impact_data)
        st.dataframe(df_impact, use_container_width=True)
    
    with tab2:
        st.subheader("🌐 Network Path Analysis")
        
        # Network overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🎯 Network Quality", f"{network_perf['network_quality_score']:.1f}/100")
        
        with col2:
            st.metric("⚡ Bandwidth", f"{network_perf['effective_bandwidth_mbps']:,.0f} Mbps")
        
        with col3:
            st.metric("🕐 Latency", f"{network_perf['total_latency_ms']:.1f} ms")
        
        with col4:
            st.metric("🛡️ Reliability", f"{network_perf['total_reliability']*100:.2f}%")
        
        # Network diagram
        st.markdown("**🗺️ Network Path Visualization:**")
        network_diagram = create_network_diagram(network_perf)
        if network_diagram:
            st.plotly_chart(network_diagram, use_container_width=True)
        
        # Path comparison
        st.markdown("**⚖️ Production vs Non-Production Comparison:**")
        
        # Get both paths for comparison
        if config['environment'] == 'production':
            alt_config = config.copy()
            alt_config['environment'] = 'non-production'
            alt_path_key = network_manager.get_network_path_key(alt_config)
            alt_network_perf = network_manager.calculate_network_performance(alt_path_key)
            
            comparison_data = {
                'Environment': ['Production (Current)', 'Non-Production'],
                'Bandwidth (Mbps)': [
                    network_perf['effective_bandwidth_mbps'],
                    alt_network_perf['effective_bandwidth_mbps']
                ],
                'Latency (ms)': [
                    network_perf['total_latency_ms'],
                    alt_network_perf['total_latency_ms']
                ],
                'Quality Score': [
                    network_perf['network_quality_score'],
                    alt_network_perf['network_quality_score']
                ],
                'Cost Factor': [
                    network_perf['total_cost_factor'],
                    alt_network_perf['total_cost_factor']
                ]
            }
        else:
            alt_config = config.copy()
            alt_config['environment'] = 'production'
            alt_path_key = network_manager.get_network_path_key(alt_config)
            alt_network_perf = network_manager.calculate_network_performance(alt_path_key)
            
            comparison_data = {
                'Environment': ['Non-Production (Current)', 'Production'],
                'Bandwidth (Mbps)': [
                    network_perf['effective_bandwidth_mbps'],
                    alt_network_perf['effective_bandwidth_mbps']
                ],
                'Latency (ms)': [
                    network_perf['total_latency_ms'],
                    alt_network_perf['total_latency_ms']
                ],
                'Quality Score': [
                    network_perf['network_quality_score'],
                    alt_network_perf['network_quality_score']
                ],
                'Cost Factor': [
                    network_perf['total_cost_factor'],
                    alt_network_perf['total_cost_factor']
                ]
            }
        
        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, use_container_width=True)
    
    with tab3:
        st.subheader("🤖 Agent Performance Analysis")
        
        # Agent overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "🔧 Agent Configuration",
                f"{agent_perf['num_agents']}x {agent_perf['agent_size'].title()}",
                delta=f"{agent_perf['agent_type'].upper()}"
            )
        
        with col2:
            st.metric(
                "⚡ Total Capacity",
                f"{agent_perf['total_agent_throughput_mbps']:,.0f} Mbps",
                delta=f"Per Agent: {agent_perf['per_agent_throughput_mbps']:.0f} Mbps"
            )
        
        with col3:
            st.metric(
                "🎯 Scaling Efficiency",
                f"{agent_perf['scaling_efficiency']*100:.1f}%",
                delta="Multi-agent coordination"
            )
        
        with col4:
            st.metric(
                "💰 Monthly Cost",
                f"${agent_perf['total_monthly_cost']:,.0f}",
                delta=f"${agent_perf['per_agent_monthly_cost']:.0f} per agent"
            )
        
        # VMware impact analysis
        st.markdown("**☁️ VMware Platform Impact:**")
        
        vmware_col1, vmware_col2 = st.columns(2)
        
        with vmware_col1:
            st.markdown(f"""
            <div class="agent-card">
                <h4>🖥️ Platform Configuration</h4>
                <p><strong>Platform:</strong> {config['server_type'].title()}</p>
                <p><strong>VMware Efficiency:</strong> {agent_perf['vmware_efficiency']*100:.1f}%</p>
                <p><strong>Performance Impact:</strong> {(1-agent_perf['vmware_efficiency'])*100:.1f}% overhead</p>
                <p><strong>Per-Agent Impact:</strong> {agent_perf['base_spec']['throughput_mbps'] - agent_perf['per_agent_throughput_mbps']:.0f} Mbps loss</p>
            </div>
            """, unsafe_allow_html=True)
        
        with vmware_col2:
            # Compare physical vs VMware
            physical_throughput = agent_perf['base_spec']['throughput_mbps'] * agent_perf['num_agents'] * agent_perf['scaling_efficiency']
            
            platform_comparison = {
                'Platform': ['Physical', 'VMware (Current)'],
                'Per Agent (Mbps)': [
                    agent_perf['base_spec']['throughput_mbps'],
                    agent_perf['per_agent_throughput_mbps']
                ],
                'Total Capacity (Mbps)': [
                    physical_throughput,
                    agent_perf['total_agent_throughput_mbps']
                ]
            }
            
            fig_platform = px.bar(
                platform_comparison,
                x='Platform',
                y=['Per Agent (Mbps)', 'Total Capacity (Mbps)'],
                title="Physical vs VMware Performance",
                barmode='group'
            )
            st.plotly_chart(fig_platform, use_container_width=True)
        
        # Agent scaling analysis
        st.markdown("**📈 Agent Scaling Analysis:**")
        
        scaling_data = []
        for num in range(1, 6):
            test_perf = agent_manager.calculate_agent_performance(
                agent_type, config['agent_size'], num, config['server_type']
            )
            scaling_data.append({
                'Agents': num,
                'Total Throughput (Mbps)': test_perf['total_agent_throughput_mbps'],
                'Scaling Efficiency (%)': test_perf['scaling_efficiency'] * 100,
                'Monthly Cost ($)': test_perf['total_monthly_cost']
            })
        
        df_scaling = pd.DataFrame(scaling_data)
        
        # Highlight current configuration
        df_scaling['Current'] = df_scaling['Agents'] == config['number_of_agents']
        
        fig_scaling = px.line(
            df_scaling,
            x='Agents',
            y='Total Throughput (Mbps)',
            title="Agent Scaling Performance",
            markers=True
        )
        
        # Add current point
        current_point = df_scaling[df_scaling['Current']]
        fig_scaling.add_scatter(
            x=current_point['Agents'],
            y=current_point['Total Throughput (Mbps)'],
            mode='markers',
            marker=dict(size=15, color='red'),
            name='Current Config'
        )
        
        st.plotly_chart(fig_scaling, use_container_width=True)
    
    with tab4:
        st.subheader("📊 Overall Performance Comparison")
        
        # Final throughput calculation
        final_throughput = min(
            network_perf['effective_bandwidth_mbps'],
            agent_perf['total_agent_throughput_mbps']
        )
        
        # Determine bottleneck
        if network_perf['effective_bandwidth_mbps'] < agent_perf['total_agent_throughput_mbps']:
            bottleneck = "Network"
            bottleneck_severity = "High" if network_perf['effective_bandwidth_mbps'] < agent_perf['total_agent_throughput_mbps'] * 0.8 else "Medium"
        else:
            bottleneck = "Agents"
            bottleneck_severity = "Medium"
        
        # Performance summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "🚀 Final Throughput",
                f"{final_throughput:,.0f} Mbps",
                delta=f"Bottleneck: {bottleneck}"
            )
        
        with col2:
            efficiency = (final_throughput / config['nic_speed']) * 100
            st.metric(
                "⚡ Overall Efficiency",
                f"{efficiency:.1f}%",
                delta=f"From {config['nic_speed']:,} Mbps NIC"
            )
        
        with col3:
            migration_time = (config['database_size_gb'] * 8 * 1000) / (final_throughput * 3600)
            st.metric(
                "⏱️ Migration Time",
                f"{migration_time:.1f} hours",
                delta=f"{config['database_size_gb']:,} GB database"
            )
        
        with col4:
            st.metric(
                "🛡️ Bottleneck Severity",
                bottleneck_severity,
                delta=f"{bottleneck} limited"
            )
        
        # Performance breakdown chart
        st.markdown("**📊 Performance Component Analysis:**")
        
        component_data = {
            'Component': ['NIC Capacity', 'Network Path', 'Agent Capacity', 'Final Throughput'],
            'Throughput (Mbps)': [
                config['nic_speed'],
                network_perf['effective_bandwidth_mbps'],
                agent_perf['total_agent_throughput_mbps'],
                final_throughput
            ]
        }
        
        fig_components = px.bar(
            component_data,
            x='Component',
            y='Throughput (Mbps)',
            title="Migration Performance Components",
            color='Throughput (Mbps)',
            color_continuous_scale='RdYlGn'
        )
        
        # Add bottleneck line
        bottleneck_value = network_perf['effective_bandwidth_mbps'] if bottleneck == "Network" else agent_perf['total_agent_throughput_mbps']
        fig_components.add_hline(
            y=bottleneck_value,
            line_dash="dash",
            line_color="red",
            annotation_text=f"{bottleneck} Bottleneck: {bottleneck_value:,.0f} Mbps"
        )
        
        st.plotly_chart(fig_components, use_container_width=True)
        
        # Recommendations
        st.markdown("**💡 Optimization Recommendations:**")
        
        if bottleneck == "Network":
            st.markdown(f"""
            <div class="warning-card">
                <h4>🌐 Network Optimization Required</h4>
                <p><strong>Issue:</strong> Network bandwidth ({network_perf['effective_bandwidth_mbps']:,.0f} Mbps) is limiting migration speed</p>
                <p><strong>Agent Capacity:</strong> {agent_perf['total_agent_throughput_mbps']:,.0f} Mbps available but unused</p>
                <p><strong>Recommendations:</strong></p>
                <ul>
                    <li>Upgrade to production environment for {alt_network_perf['effective_bandwidth_mbps']:,.0f} Mbps bandwidth</li>
                    <li>Schedule migration during off-peak hours</li>
                    <li>Consider network optimization techniques</li>
                    <li>Plan migration timeline using {final_throughput:,.0f} Mbps actual speed</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="agent-card">
                <h4>🤖 Agent Optimization Opportunities</h4>
                <p><strong>Status:</strong> Agents ({agent_perf['total_agent_throughput_mbps']:,.0f} Mbps) are the limiting factor</p>
                <p><strong>Network Capacity:</strong> {network_perf['effective_bandwidth_mbps']:,.0f} Mbps available</p>
                <p><strong>Recommendations:</strong></p>
                <ul>
                    <li>Scale up to larger agent sizes for better performance</li>
                    <li>Add more agents (currently {agent_perf['num_agents']})</li>
                    <li>Consider moving from VMware to physical for +8% performance</li>
                    <li>Optimize agent configuration for {config['migration_type'].upper()}</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Migration planning
        st.markdown("**📅 Migration Planning Summary:**")
        
        planning_col1, planning_col2 = st.columns(2)
        
        with planning_col1:
            st.markdown(f"""
            <div class="network-card">
                <h4>⏱️ Time Estimates</h4>
                <p><strong>Database Size:</strong> {config['database_size_gb']:,} GB</p>
                <p><strong>Effective Speed:</strong> {final_throughput:,.0f} Mbps</p>
                <p><strong>Migration Time:</strong> {migration_time:.1f} hours</p>
                <p><strong>Data Transfer:</strong> {config['database_size_gb'] * 8:,} Megabits</p>
                <p><strong>Buffer Time (20%):</strong> {migration_time * 1.2:.1f} hours</p>
            </div>
            """, unsafe_allow_html=True)
        
        with planning_col2:
            st.markdown(f"""
            <div class="performance-card">
                <h4>💰 Cost Summary</h4>
                <p><strong>Agent Monthly Cost:</strong> ${agent_perf['total_monthly_cost']:,.0f}</p>
                <p><strong>Per-Agent Cost:</strong> ${agent_perf['per_agent_monthly_cost']:,.0f}</p>
                <p><strong>Cost per Mbps:</strong> ${agent_perf['total_monthly_cost'] / agent_perf['total_agent_throughput_mbps']:.2f}</p>
                <p><strong>Migration Window:</strong> {migration_time:.1f} hours</p>
                <p><strong>Efficiency Rating:</strong> {efficiency:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()