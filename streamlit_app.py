import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import math
from typing import Dict, List, Tuple

# Page configuration
st.set_page_config(
    page_title="AWS Migration Network Pattern Analyzer",
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
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .pattern-card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 6px;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border-left: 3px solid #3b82f6;
        border: 1px solid #e5e7eb;
    }
    
    .vpc-warning-card {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        padding: 1.5rem;
        border-radius: 6px;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border-left: 3px solid #f59e0b;
        border: 1px solid #d97706;
    }
    
    .network-path-card {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        padding: 1.5rem;
        border-radius: 6px;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border-left: 3px solid #22c55e;
        border: 1px solid #e5e7eb;
    }
    
    .recommendation-card {
        background: linear-gradient(135deg, #fefdf8 0%, #fefce8 100%);
        padding: 1.5rem;
        border-radius: 6px;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border-left: 3px solid #f59e0b;
        border: 1px solid #e5e7eb;
    }
    
    .waterfall-card {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        padding: 1.5rem;
        border-radius: 6px;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border-left: 3px solid #0ea5e9;
        border: 1px solid #e5e7eb;
    }
</style>
""", unsafe_allow_html=True)

class NetworkPatternAnalyzer:
    """Core class for analyzing network migration patterns with VPC endpoint considerations"""
    
    def __init__(self):
        self.network_patterns = {
            # San Jose Patterns
            'sj_nonprod_vpc_endpoint': {
                'name': 'San Jose Non-Prod → AWS VPC Endpoint',
                'source': 'San Jose',
                'environment': 'non-production',
                'pattern_type': 'vpc_endpoint',
                'segments': [
                    {'name': 'San Jose Internal LAN', 'bandwidth_mbps': 10000, 'latency_ms': 1, 'reliability': 0.999},
                    {'name': 'VPC Endpoint Connection', 'bandwidth_mbps': 2000, 'latency_ms': 8, 'reliability': 0.998}
                ],
                'total_bandwidth_mbps': 2000,
                'total_latency_ms': 9,
                'cost_factor': 1.5,
                'security_level': 'high',
                'protocol_overhead': 0.05,  # 5% protocol overhead
                'network_congestion_factor': 0.9,  # 10% reduction due to congestion
                # VPC Endpoint specific limitations
                'vpc_endpoint_limitations': {
                    'ipv4_only': True,
                    'no_shared_vpc': True,
                    'no_dedicated_tenancy': True,
                    'requires_4_network_interfaces': True,
                    'additional_security_groups': True,
                    'privatelink_routing_overhead': 0.03  # 3% additional overhead for PrivateLink routing
                }
            },
            'sj_nonprod_direct_connect': {
                'name': 'San Jose Non-Prod → AWS Direct Connect',
                'source': 'San Jose',
                'environment': 'non-production',
                'pattern_type': 'direct_connect',
                'segments': [
                    {'name': 'San Jose Internal LAN', 'bandwidth_mbps': 10000, 'latency_ms': 1, 'reliability': 0.999},
                    {'name': 'Direct Connect (DX)', 'bandwidth_mbps': 2000, 'latency_ms': 12, 'reliability': 0.998}
                ],
                'total_bandwidth_mbps': 2000,
                'total_latency_ms': 13,
                'cost_factor': 2.0,
                'security_level': 'high',
                'protocol_overhead': 0.03,  # 3% protocol overhead
                'network_congestion_factor': 0.95  # 5% reduction due to congestion
            },
            'sj_prod_direct_connect': {
                'name': 'San Jose Production → AWS Direct Connect',
                'source': 'San Jose',
                'environment': 'production',
                'pattern_type': 'direct_connect',
                'segments': [
                    {'name': 'San Jose Internal LAN', 'bandwidth_mbps': 10000, 'latency_ms': 1, 'reliability': 0.999},
                    {'name': 'Production Direct Connect', 'bandwidth_mbps': 10000, 'latency_ms': 8, 'reliability': 0.9999}
                ],
                'total_bandwidth_mbps': 10000,
                'total_latency_ms': 9,
                'cost_factor': 3.5,
                'security_level': 'very_high',
                'protocol_overhead': 0.02,  # 2% protocol overhead
                'network_congestion_factor': 0.98  # 2% reduction due to congestion
            },
            # San Antonio Patterns
            'sa_prod_via_sj': {
                'name': 'San Antonio Production → San Jose → AWS',
                'source': 'San Antonio',
                'environment': 'production',
                'pattern_type': 'multi_hop',
                'segments': [
                    {'name': 'San Antonio Internal LAN', 'bandwidth_mbps': 10000, 'latency_ms': 1, 'reliability': 0.999},
                    {'name': 'SA to SJ Private Line', 'bandwidth_mbps': 10000, 'latency_ms': 12, 'reliability': 0.9995},
                    {'name': 'SJ to AWS Direct Connect', 'bandwidth_mbps': 10000, 'latency_ms': 8, 'reliability': 0.9999}
                ],
                'total_bandwidth_mbps': 10000,
                'total_latency_ms': 21,
                'cost_factor': 4.0,
                'security_level': 'very_high',
                'protocol_overhead': 0.04,  # 4% protocol overhead (multi-hop)
                'network_congestion_factor': 0.92  # 8% reduction due to multi-hop congestion
            }
        }
        
        # DataSync Agent Configurations with VPC Endpoint considerations
        self.datasync_agents = {
            'small': {
                'vcpu': 2, 'memory_gb': 4, 'throughput_mbps': 250, 'cost_per_hour': 0.042,
                'vpc_endpoint_compatible': True,
                'vpc_endpoint_throughput_reduction': 0.1  # 10% reduction through VPC endpoints
            },
            'medium': {
                'vcpu': 2, 'memory_gb': 8, 'throughput_mbps': 500, 'cost_per_hour': 0.085,
                'vpc_endpoint_compatible': True,
                'vpc_endpoint_throughput_reduction': 0.08  # 8% reduction through VPC endpoints
            },
            'large': {
                'vcpu': 4, 'memory_gb': 16, 'throughput_mbps': 1000, 'cost_per_hour': 0.17,
                'vpc_endpoint_compatible': True,
                'vpc_endpoint_throughput_reduction': 0.05  # 5% reduction through VPC endpoints
            },
            'xlarge': {
                'vcpu': 8, 'memory_gb': 32, 'throughput_mbps': 2000, 'cost_per_hour': 0.34,
                'vpc_endpoint_compatible': True,
                'vpc_endpoint_throughput_reduction': 0.03  # 3% reduction through VPC endpoints
            }
        }
        
        # DMS Instance Configurations
        self.dms_instances = {
            'small': {'vcpu': 2, 'memory_gb': 4, 'throughput_mbps': 200, 'cost_per_hour': 0.042},
            'medium': {'vcpu': 2, 'memory_gb': 8, 'throughput_mbps': 400, 'cost_per_hour': 0.085},
            'large': {'vcpu': 4, 'memory_gb': 16, 'throughput_mbps': 800, 'cost_per_hour': 0.17},
            'xlarge': {'vcpu': 8, 'memory_gb': 32, 'throughput_mbps': 1500, 'cost_per_hour': 0.34},
            'xxlarge': {'vcpu': 16, 'memory_gb': 64, 'throughput_mbps': 2500, 'cost_per_hour': 0.68}
        }
    
    def determine_optimal_pattern(self, source_location: str, environment: str, is_homogeneous: bool) -> str:
        """Determine the optimal network pattern based on requirements"""
        if source_location == 'San Jose':
            if environment == 'production':
                return 'sj_prod_direct_connect'
            else:
                return 'sj_nonprod_vpc_endpoint'
        elif source_location == 'San Antonio':
            return 'sa_prod_via_sj'
        return 'sj_nonprod_direct_connect'
    
    def assess_vpc_endpoint_compatibility(self, pattern_key: str, agent_type: str, agent_size: str) -> Dict:
        """Assess compatibility and limitations when using DataSync with VPC endpoints"""
        pattern = self.network_patterns[pattern_key]
        compatibility_assessment = {
            'is_vpc_endpoint': pattern['pattern_type'] == 'vpc_endpoint',
            'is_datasync': agent_type == 'datasync',
            'warnings': [],
            'performance_impacts': [],
            'requirements': []
        }
        
        if compatibility_assessment['is_vpc_endpoint'] and compatibility_assessment['is_datasync']:
            # VPC Endpoint specific considerations for DataSync
            vpc_limitations = pattern.get('vpc_endpoint_limitations', {})
            
            if vpc_limitations.get('ipv4_only'):
                compatibility_assessment['warnings'].append(
                    "VPC Endpoints only support IPv4 - IPv6 and dual-stack configurations not supported"
                )
            
            if vpc_limitations.get('no_shared_vpc'):
                compatibility_assessment['warnings'].append(
                    "Shared VPCs are not supported with DataSync VPC endpoints"
                )
            
            if vpc_limitations.get('no_dedicated_tenancy'):
                compatibility_assessment['warnings'].append(
                    "VPCs with dedicated tenancy are not supported"
                )
            
            if vpc_limitations.get('requires_4_network_interfaces'):
                compatibility_assessment['requirements'].append(
                    "DataSync creates 4 network interfaces in your VPC - ensure subnet capacity"
                )
                compatibility_assessment['requirements'].append(
                    "Agent must be able to reach all 4 network interface IP addresses"
                )
            
            if vpc_limitations.get('additional_security_groups'):
                compatibility_assessment['requirements'].append(
                    "Configure security groups for TCP 443 and TCP 1024-1062 port ranges"
                )
                compatibility_assessment['requirements'].append(
                    "Allow ephemeral outbound traffic and connection tracking"
                )
            
            # Performance impacts
            agent_spec = self.datasync_agents[agent_size]
            vpc_throughput_reduction = agent_spec.get('vpc_endpoint_throughput_reduction', 0.05)
            privatelink_overhead = vpc_limitations.get('privatelink_routing_overhead', 0.03)
            
            total_performance_impact = (vpc_throughput_reduction + privatelink_overhead) * 100
            compatibility_assessment['performance_impacts'].append(
                f"Expected {total_performance_impact:.1f}% throughput reduction due to VPC endpoint routing"
            )
            compatibility_assessment['performance_impacts'].append(
                "Additional latency from PrivateLink network interface routing"
            )
        
        return compatibility_assessment
    
    def calculate_migration_throughput(self, pattern_key: str, agent_type: str, agent_size: str, num_agents: int) -> Dict:
        """Calculate effective migration throughput with VPC endpoint considerations"""
        pattern = self.network_patterns[pattern_key]
        
        # Get agent specifications
        if agent_type == 'datasync':
            agent_spec = self.datasync_agents[agent_size]
        else:
            agent_spec = self.dms_instances[agent_size]
        
        # Calculate base agent capacity
        base_agent_throughput = agent_spec['throughput_mbps'] * num_agents
        
        # Apply VPC endpoint throughput reduction for DataSync
        if pattern['pattern_type'] == 'vpc_endpoint' and agent_type == 'datasync':
            vpc_reduction = agent_spec.get('vpc_endpoint_throughput_reduction', 0.05)
            vpc_adjusted_throughput = base_agent_throughput * (1 - vpc_reduction)
            
            # Apply PrivateLink routing overhead
            vpc_limitations = pattern.get('vpc_endpoint_limitations', {})
            privatelink_overhead = vpc_limitations.get('privatelink_routing_overhead', 0.03)
            vpc_adjusted_throughput *= (1 - privatelink_overhead)
        else:
            vpc_adjusted_throughput = base_agent_throughput
        
        # Apply scaling efficiency
        if num_agents == 1:
            scaling_efficiency = 1.0
        elif num_agents <= 3:
            scaling_efficiency = 0.95
        elif num_agents <= 5:
            scaling_efficiency = 0.90
        else:
            scaling_efficiency = 0.85
        
        effective_agent_throughput = vpc_adjusted_throughput * scaling_efficiency
        network_bandwidth = pattern['total_bandwidth_mbps']
        effective_throughput = min(effective_agent_throughput, network_bandwidth)
        
        # Calculate utilization
        network_utilization = (effective_throughput / network_bandwidth) * 100
        agent_utilization = (effective_throughput / effective_agent_throughput) * 100 if effective_agent_throughput > 0 else 0
        
        return {
            'effective_throughput_mbps': effective_throughput,
            'network_bandwidth_mbps': network_bandwidth,
            'agent_throughput_mbps': effective_agent_throughput,
            'vpc_adjusted_throughput_mbps': vpc_adjusted_throughput,
            'base_agent_throughput_mbps': base_agent_throughput,
            'network_utilization_percent': network_utilization,
            'agent_utilization_percent': agent_utilization,
            'bottleneck': 'network' if effective_throughput == network_bandwidth else 'agents',
            'scaling_efficiency': scaling_efficiency,
            'latency_ms': pattern['total_latency_ms'],
            'vpc_impact_percent': ((base_agent_throughput - vpc_adjusted_throughput) / base_agent_throughput * 100) if base_agent_throughput > 0 else 0
        }
    
    def calculate_bandwidth_waterfall(self, pattern_key: str, agent_type: str, agent_size: str, num_agents: int) -> Dict:
        """Calculate detailed bandwidth waterfall analysis with VPC endpoint considerations"""
        pattern = self.network_patterns[pattern_key]
        
        # Get agent specifications
        if agent_type == 'datasync':
            agent_spec = self.datasync_agents[agent_size]
        else:
            agent_spec = self.dms_instances[agent_size]
        
        # Step 1: Theoretical Maximum
        theoretical_max = max([segment['bandwidth_mbps'] for segment in pattern['segments']])
        
        # Step 2: Network Path Limitation
        network_limitation = min([segment['bandwidth_mbps'] for segment in pattern['segments']])
        network_reduction = theoretical_max - network_limitation
        
        # Step 3: Agent Capacity Limitation
        total_agent_capacity = agent_spec['throughput_mbps'] * num_agents
        agent_limited_bandwidth = min(network_limitation, total_agent_capacity)
        agent_reduction = network_limitation - agent_limited_bandwidth
        
        # Step 4: VPC Endpoint Specific Reductions (NEW)
        vpc_endpoint_adjusted_bandwidth = agent_limited_bandwidth
        vpc_endpoint_reduction = 0
        
        if pattern['pattern_type'] == 'vpc_endpoint' and agent_type == 'datasync':
            # DataSync VPC endpoint throughput reduction
            vpc_throughput_reduction = agent_spec.get('vpc_endpoint_throughput_reduction', 0.05)
            vpc_endpoint_adjusted_bandwidth *= (1 - vpc_throughput_reduction)
            
            # PrivateLink routing overhead
            vpc_limitations = pattern.get('vpc_endpoint_limitations', {})
            privatelink_overhead = vpc_limitations.get('privatelink_routing_overhead', 0.03)
            vpc_endpoint_adjusted_bandwidth *= (1 - privatelink_overhead)
            
            vpc_endpoint_reduction = agent_limited_bandwidth - vpc_endpoint_adjusted_bandwidth
        
        # Step 5: Scaling Efficiency Impact
        if num_agents == 1:
            scaling_efficiency = 1.0
        elif num_agents <= 3:
            scaling_efficiency = 0.95
        elif num_agents <= 5:
            scaling_efficiency = 0.90
        else:
            scaling_efficiency = 0.85
        
        scaling_adjusted_bandwidth = vpc_endpoint_adjusted_bandwidth * scaling_efficiency
        scaling_reduction = vpc_endpoint_adjusted_bandwidth - scaling_adjusted_bandwidth
        
        # Step 6: Protocol Overhead
        protocol_overhead = pattern.get('protocol_overhead', 0.03)
        protocol_adjusted_bandwidth = scaling_adjusted_bandwidth * (1 - protocol_overhead)
        protocol_reduction = scaling_adjusted_bandwidth - protocol_adjusted_bandwidth
        
        # Step 7: Network Congestion
        congestion_factor = pattern.get('network_congestion_factor', 0.95)
        final_effective_bandwidth = protocol_adjusted_bandwidth * congestion_factor
        congestion_reduction = protocol_adjusted_bandwidth - final_effective_bandwidth
        
        # Step 8: Quality of Service adjustments
        if pattern['environment'] == 'production':
            qos_factor = 0.98
        else:
            qos_factor = 0.95
        
        qos_adjusted_bandwidth = final_effective_bandwidth * qos_factor
        qos_reduction = final_effective_bandwidth - qos_adjusted_bandwidth
        
        # Build waterfall steps
        steps = [
            {'name': 'Theoretical Maximum', 'value': theoretical_max, 'cumulative': theoretical_max, 'type': 'positive'},
            {'name': 'Network Path Limit', 'value': -network_reduction, 'cumulative': network_limitation, 'type': 'negative'},
            {'name': 'Agent Capacity Limit', 'value': -agent_reduction, 'cumulative': agent_limited_bandwidth, 'type': 'negative'}
        ]
        
        # Add VPC endpoint step if applicable
        if vpc_endpoint_reduction > 0:
            steps.append({'name': 'VPC Endpoint Overhead', 'value': -vpc_endpoint_reduction, 'cumulative': vpc_endpoint_adjusted_bandwidth, 'type': 'negative'})
        
        steps.extend([
            {'name': 'Scaling Efficiency', 'value': -scaling_reduction, 'cumulative': scaling_adjusted_bandwidth, 'type': 'negative'},
            {'name': 'Protocol Overhead', 'value': -protocol_reduction, 'cumulative': protocol_adjusted_bandwidth, 'type': 'negative'},
            {'name': 'Network Congestion', 'value': -congestion_reduction, 'cumulative': final_effective_bandwidth, 'type': 'negative'},
            {'name': 'QoS Overhead', 'value': -qos_reduction, 'cumulative': qos_adjusted_bandwidth, 'type': 'negative'},
            {'name': 'Final Effective', 'value': qos_adjusted_bandwidth, 'cumulative': qos_adjusted_bandwidth, 'type': 'total'}
        ])
        
        return {
            'steps': steps,
            'summary': {
                'theoretical_max_mbps': theoretical_max,
                'network_limited_mbps': network_limitation,
                'agent_limited_mbps': agent_limited_bandwidth,
                'vpc_endpoint_adjusted_mbps': vpc_endpoint_adjusted_bandwidth,
                'final_effective_mbps': qos_adjusted_bandwidth,
                'total_reduction_mbps': theoretical_max - qos_adjusted_bandwidth,
                'vpc_endpoint_reduction_mbps': vpc_endpoint_reduction,
                'efficiency_percentage': (qos_adjusted_bandwidth / theoretical_max) * 100,
                'primary_bottleneck': self._identify_primary_bottleneck(network_reduction, agent_reduction, vpc_endpoint_reduction, scaling_reduction, protocol_reduction, congestion_reduction, qos_reduction),
                'scaling_efficiency': scaling_efficiency,
                'protocol_overhead_pct': protocol_overhead * 100,
                'congestion_impact_pct': (1 - congestion_factor) * 100,
                'qos_overhead_pct': (1 - qos_factor) * 100,
                'vpc_endpoint_impact_pct': (vpc_endpoint_reduction / theoretical_max) * 100 if vpc_endpoint_reduction > 0 else 0
            }
        }
    
    def _identify_primary_bottleneck(self, network_red, agent_red, vpc_red, scaling_red, protocol_red, congestion_red, qos_red) -> str:
        """Identify the primary bottleneck in the bandwidth waterfall"""
        reductions = {
            'Network Path': network_red,
            'Agent Capacity': agent_red,
            'VPC Endpoint': vpc_red,
            'Scaling Inefficiency': scaling_red,
            'Protocol Overhead': protocol_red,
            'Network Congestion': congestion_red,
            'QoS Overhead': qos_red
        }
        return max(reductions.items(), key=lambda x: x[1])[0]
    
    def estimate_migration_time(self, database_size_gb: int, effective_throughput_mbps: int) -> Dict:
        """Estimate migration time based on database size and throughput"""
        database_size_gbits = database_size_gb * 8
        
        if effective_throughput_mbps > 0:
            migration_time_hours = database_size_gbits / (effective_throughput_mbps / 1000) / 3600
        else:
            migration_time_hours = float('inf')
        
        setup_overhead_hours = 2
        validation_overhead_hours = database_size_gb / 1000
        total_migration_time = migration_time_hours + setup_overhead_hours + validation_overhead_hours
        
        return {
            'data_transfer_hours': migration_time_hours,
            'setup_hours': setup_overhead_hours,
            'validation_hours': validation_overhead_hours,
            'total_hours': total_migration_time,
            'total_days': total_migration_time / 24,
            'recommended_window_hours': math.ceil(total_migration_time * 1.2)
        }
    
    def generate_ai_recommendation(self, config: Dict, analysis_results: Dict) -> Dict:
        """Generate AI-powered recommendations with VPC endpoint considerations"""
        database_size = config['database_size_gb']
        migration_time = analysis_results['migration_time']
        throughput_analysis = analysis_results['throughput_analysis']
        vpc_compatibility = analysis_results.get('vpc_compatibility', {})
        
        recommendations = []
        priority_score = 0
        
        # VPC Endpoint specific recommendations
        if vpc_compatibility.get('is_vpc_endpoint') and vpc_compatibility.get('is_datasync'):
            if len(vpc_compatibility.get('warnings', [])) > 0:
                recommendations.append({
                    'type': 'vpc_endpoint_limitations',
                    'priority': 'high',
                    'description': f'VPC Endpoint has {len(vpc_compatibility["warnings"])} compatibility warnings that may impact DataSync performance.',
                    'impact': 'Potential configuration issues and performance degradation'
                })
                priority_score += 15
            
            vpc_impact = throughput_analysis.get('vpc_impact_percent', 0)
            if vpc_impact > 5:
                recommendations.append({
                    'type': 'vpc_endpoint_performance',
                    'priority': 'medium',
                    'description': f'VPC Endpoint reduces DataSync throughput by {vpc_impact:.1f}%. Consider Direct Connect for better performance.',
                    'impact': 'Moderate performance improvement with Direct Connect'
                })
                priority_score += 10
        
        if throughput_analysis['bottleneck'] == 'network':
            recommendations.append({
                'type': 'network_optimization',
                'priority': 'high',
                'description': 'Network bandwidth is the bottleneck. Consider upgrading network connection or scheduling during off-peak hours.',
                'impact': 'High throughput improvement possible'
            })
            priority_score += 20
        
        if throughput_analysis['bottleneck'] == 'agents':
            recommendations.append({
                'type': 'agent_optimization', 
                'priority': 'medium',
                'description': f'Agent capacity is limiting throughput. Consider scaling to more agents or larger agent sizes.',
                'impact': 'Moderate throughput improvement'
            })
            priority_score += 10
        
        if migration_time['total_hours'] > 72:
            recommendations.append({
                'type': 'timeline_optimization',
                'priority': 'high', 
                'description': 'Migration time exceeds 3 days. Consider parallel processing or incremental migration strategy.',
                'impact': 'Significant time reduction possible'
            })
            priority_score += 15
        
        if database_size > 10000:
            recommendations.append({
                'type': 'large_database_strategy',
                'priority': 'high',
                'description': 'Large database detected. Implement staged migration with initial bulk transfer and ongoing replication.',
                'impact': 'Reduces downtime window significantly'
            })
            priority_score += 25
        
        if throughput_analysis['scaling_efficiency'] < 0.9:
            recommendations.append({
                'type': 'scaling_optimization',
                'priority': 'medium',
                'description': 'Agent scaling efficiency is suboptimal. Consider consolidating to fewer, larger agents.',
                'impact': 'Better resource utilization'
            })
            priority_score += 5
        
        return {
            'recommendations': recommendations,
            'overall_priority_score': priority_score,
            'migration_complexity': 'high' if priority_score > 40 else 'medium' if priority_score > 20 else 'low',
            'confidence_level': 'high' if len(recommendations) <= 2 else 'medium'
        }

def render_header():
    """Render application header"""
    st.markdown("""
    <div class="main-header">
        <h1>🌐 AWS Database Migration Network Pattern Analyzer</h1>
        <p style="font-size: 1.2rem; margin-top: 0.5rem;">
            Network Path Analysis • VPC Endpoint Considerations • Latency Optimization • Throughput Calculation • Bandwidth Waterfall • AI-Powered Recommendations
        </p>
        <p style="font-size: 0.9rem; margin-top: 0.5rem; opacity: 0.9;">
            San Jose ↔ San Antonio ↔ AWS West-2 • VPC Endpoint • Direct Connect • DataSync • DMS
        </p>
    </div>
    """, unsafe_allow_html=True)

def render_sidebar_controls():
    """Render sidebar configuration controls"""
    st.sidebar.header("🔧 Migration Configuration")
    
    st.sidebar.subheader("📍 Source Location")
    source_location = st.sidebar.selectbox(
        "Data Center Location",
        ["San Jose", "San Antonio"],
        help="Select the source data center location"
    )
    
    environment = st.sidebar.selectbox(
        "Environment Type", 
        ["non-production", "production"],
        help="Production environments have dedicated high-bandwidth connections"
    )
    
    st.sidebar.subheader("🗄️ Database Configuration")
    source_database = st.sidebar.selectbox(
        "Source Database",
        ["MySQL", "PostgreSQL", "Oracle", "SQL Server", "MongoDB"],
        help="Source database engine"
    )
    
    target_database = st.sidebar.selectbox(
        "Target Database", 
        ["MySQL", "PostgreSQL", "Oracle", "SQL Server", "MongoDB"],
        help="Target database engine on AWS"
    )
    
    database_size_gb = st.sidebar.number_input(
        "Database Size (GB)",
        min_value=100,
        max_value=100000,
        value=1000,
        step=100,
        help="Total database size to migrate"
    )
    
    is_homogeneous = source_database == target_database
    migration_type = "Homogeneous" if is_homogeneous else "Heterogeneous"
    
    st.sidebar.info(f"**Migration Type:** {migration_type}")
    if is_homogeneous:
        st.sidebar.success("✅ Using AWS DataSync for homogeneous migration")
    else:
        st.sidebar.warning("⚠️ Using AWS DMS for heterogeneous migration")
    
    st.sidebar.subheader("🤖 Migration Agent Configuration")
    
    if is_homogeneous:
        agent_type = "datasync"
        agent_size = st.sidebar.selectbox(
            "DataSync Agent Size",
            ["small", "medium", "large", "xlarge"],
            index=1,
            format_func=lambda x: f"{x.title()} - {analyzer.datasync_agents[x]['throughput_mbps']} Mbps",
            help="DataSync agent configuration"
        )
    else:
        agent_type = "dms"
        agent_size = st.sidebar.selectbox(
            "DMS Instance Size", 
            ["small", "medium", "large", "xlarge", "xxlarge"],
            index=1,
            format_func=lambda x: f"{x.title()} - {analyzer.dms_instances[x]['throughput_mbps']} Mbps",
            help="DMS instance configuration"
        )
    
    num_agents = st.sidebar.number_input(
        "Number of Agents",
        min_value=1,
        max_value=8,
        value=2,
        help="Number of parallel migration agents"
    )
    
    st.sidebar.subheader("⏱️ Migration Requirements")
    max_downtime_hours = st.sidebar.number_input(
        "Maximum Downtime (hours)",
        min_value=1,
        max_value=168,
        value=24,
        help="Maximum acceptable downtime window"
    )
    
    return {
        'source_location': source_location,
        'environment': environment,
        'source_database': source_database,
        'target_database': target_database,
        'database_size_gb': database_size_gb,
        'is_homogeneous': is_homogeneous,
        'migration_type': migration_type,
        'agent_type': agent_type,
        'agent_size': agent_size,
        'num_agents': num_agents,
        'max_downtime_hours': max_downtime_hours
    }

def render_vpc_endpoint_analysis(config: Dict, analyzer: NetworkPatternAnalyzer):
    """Render VPC endpoint specific analysis"""
    pattern_key = analyzer.determine_optimal_pattern(
        config['source_location'], 
        config['environment'], 
        config['is_homogeneous']
    )
    
    vpc_compatibility = analyzer.assess_vpc_endpoint_compatibility(
        pattern_key, config['agent_type'], config['agent_size']
    )
    
    if vpc_compatibility['is_vpc_endpoint'] and vpc_compatibility['is_datasync']:
        st.markdown("**⚠️ VPC Endpoint + DataSync Compatibility Analysis:**")
        
        # Warnings
        if vpc_compatibility['warnings']:
            st.markdown("**🚨 Compatibility Warnings:**")
            for warning in vpc_compatibility['warnings']:
                st.warning(f"• {warning}")
        
        # Requirements
        if vpc_compatibility['requirements']:
            st.markdown("**📋 Network Requirements:**")
            for requirement in vpc_compatibility['requirements']:
                st.info(f"• {requirement}")
        
        # Performance Impacts
        if vpc_compatibility['performance_impacts']:
            st.markdown("**📉 Performance Impacts:**")
            for impact in vpc_compatibility['performance_impacts']:
                st.warning(f"• {impact}")
        
        # Summary card
        st.markdown(f"""
        <div class="vpc-warning-card">
            <h4>🔍 VPC Endpoint Impact Summary</h4>
            <p><strong>Configuration:</strong> DataSync agent with VPC Endpoint</p>
            <p><strong>Warnings:</strong> {len(vpc_compatibility['warnings'])} compatibility issues</p>
            <p><strong>Requirements:</strong> {len(vpc_compatibility['requirements'])} network configuration items</p>
            <p><strong>Performance Impact:</strong> Expected throughput reduction due to PrivateLink routing</p>
            <p><strong>Recommendation:</strong> Consider Direct Connect for production workloads requiring maximum performance</p>
        </div>
        """, unsafe_allow_html=True)
    
    return vpc_compatibility

def create_network_diagram(pattern_key: str, analyzer: NetworkPatternAnalyzer):
    """Create interactive network path diagram with VPC endpoint annotations"""
    pattern = analyzer.network_patterns[pattern_key]
    fig = go.Figure()
    
    num_segments = len(pattern['segments'])
    x_positions = [i * 200 for i in range(num_segments + 1)]
    y_position = 50
    
    for i, segment in enumerate(pattern['segments']):
        line_width = max(3, min(12, segment['bandwidth_mbps'] / 200))
        reliability = segment['reliability']
        
        # Special coloring for VPC endpoints
        if pattern['pattern_type'] == 'vpc_endpoint' and 'VPC' in segment['name']:
            line_color = '#f59e0b'  # Orange for VPC Endpoint
        elif reliability > 0.999:
            line_color = '#22c55e'
        elif reliability > 0.995:
            line_color = '#f59e0b'
        else:
            line_color = '#ef4444'
        
        fig.add_trace(go.Scatter(
            x=[x_positions[i], x_positions[i+1]],
            y=[y_position, y_position],
            mode='lines+markers',
            line=dict(width=line_width, color=line_color),
            marker=dict(size=12, color='#1e40af'),
            name=segment['name'],
            hovertemplate=f"""
            <b>{segment['name']}</b><br>
            Bandwidth: {segment['bandwidth_mbps']:,} Mbps<br>
            Latency: {segment['latency_ms']:.1f} ms<br>
            Reliability: {segment['reliability']*100:.3f}%<br>
            <extra></extra>
            """
        ))
        
        mid_x = (x_positions[i] + x_positions[i+1]) / 2
        
        # Add VPC Endpoint annotation
        annotation_text = f"{segment['bandwidth_mbps']:,} Mbps<br>{segment['latency_ms']:.1f} ms"
        if pattern['pattern_type'] == 'vpc_endpoint' and 'VPC' in segment['name']:
            annotation_text += "<br>⚠️ VPC Endpoint"
        
        fig.add_annotation(
            x=mid_x,
            y=y_position + 15,
            text=annotation_text,
            showarrow=False,
            font=dict(size=10),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='#e5e7eb',
            borderwidth=1
        )
    
    fig.add_trace(go.Scatter(
        x=[x_positions[0]],
        y=[y_position],
        mode='markers+text',
        marker=dict(size=25, color='#059669', symbol='square'),
        text=[pattern['source']],
        textposition='bottom center',
        name='Source',
        hovertemplate=f"<b>Source: {pattern['source']}</b><extra></extra>"
    ))
    
    fig.add_trace(go.Scatter(
        x=[x_positions[-1]],
        y=[y_position], 
        mode='markers+text',
        marker=dict(size=25, color='#dc2626', symbol='square'),
        text=['AWS West-2'],
        textposition='bottom center',
        name='Destination',
        hovertemplate="<b>Destination: AWS West-2</b><extra></extra>"
    ))
    
    title = f"Network Path: {pattern['name']}"
    if pattern['pattern_type'] == 'vpc_endpoint':
        title += " ⚠️ (VPC Endpoint Limitations Apply)"
    
    fig.update_layout(
        title=title,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        showlegend=False,
        height=300,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

def create_bandwidth_waterfall_chart(waterfall_data: Dict):
    """Create interactive bandwidth waterfall chart with VPC endpoint considerations"""
    steps = waterfall_data['steps']
    
    fig = go.Figure()
    
    # Add bars for waterfall effect
    x_labels = [step['name'] for step in steps]
    
    for i, step in enumerate(steps):
        if step['type'] == 'positive':
            # Starting point
            fig.add_trace(go.Bar(
                x=[step['name']],
                y=[step['value']],
                marker_color='#22c55e',
                name=step['name'],
                text=[f"{step['value']:.0f} Mbps"],
                textposition='outside',
                hovertemplate=f"<b>{step['name']}</b><br>Bandwidth: {step['value']:.0f} Mbps<extra></extra>"
            ))
        elif step['type'] == 'total':
            # Final result
            fig.add_trace(go.Bar(
                x=[step['name']],
                y=[step['value']],
                marker_color='#1e40af',
                name=step['name'],
                text=[f"{step['value']:.0f} Mbps"],
                textposition='outside',
                hovertemplate=f"<b>{step['name']}</b><br>Final Bandwidth: {step['value']:.0f} Mbps<extra></extra>"
            ))
        else:
            # Reduction steps - special color for VPC Endpoint
            color = '#f59e0b' if 'VPC Endpoint' in step['name'] else '#ef4444'
            fig.add_trace(go.Bar(
                x=[step['name']],
                y=[abs(step['value'])],
                marker_color=color,
                name=step['name'],
                text=[f"{step['value']:.0f} Mbps"],
                textposition='outside',
                hovertemplate=f"<b>{step['name']}</b><br>Reduction: {abs(step['value']):.0f} Mbps<br>Remaining: {step['cumulative']:.0f} Mbps<extra></extra>"
            ))
    
    fig.update_layout(
        title="Bandwidth Waterfall Analysis: Theoretical Max to Effective Throughput (Including VPC Endpoint Impact)",
        xaxis_title="Migration Pipeline Stages",
        yaxis_title="Bandwidth (Mbps)",
        showlegend=False,
        height=500,
        xaxis=dict(tickangle=45),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
        template="plotly_white"
    )
    
    return fig

def render_bandwidth_waterfall_tab(config: Dict, analyzer: NetworkPatternAnalyzer):
    """Render bandwidth waterfall analysis tab with VPC endpoint considerations"""
    st.subheader("💧 Bandwidth Waterfall Analysis")
    
    pattern_key = analyzer.determine_optimal_pattern(
        config['source_location'], 
        config['environment'], 
        config['is_homogeneous']
    )
    
    waterfall_data = analyzer.calculate_bandwidth_waterfall(
        pattern_key,
        config['agent_type'],
        config['agent_size'], 
        config['num_agents']
    )
    
    # Summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "🏁 Theoretical Max",
            f"{waterfall_data['summary']['theoretical_max_mbps']:,.0f} Mbps",
            delta="Starting point"
        )
    
    with col2:
        st.metric(
            "🚧 Network Limited",
            f"{waterfall_data['summary']['network_limited_mbps']:,.0f} Mbps",
            delta=f"-{waterfall_data['summary']['theoretical_max_mbps'] - waterfall_data['summary']['network_limited_mbps']:,.0f}"
        )
    
    with col3:
        st.metric(
            "🤖 Agent Limited",
            f"{waterfall_data['summary']['agent_limited_mbps']:,.0f} Mbps",
            delta=f"-{waterfall_data['summary']['network_limited_mbps'] - waterfall_data['summary']['agent_limited_mbps']:,.0f}"
        )
    
    with col4:
        # Show VPC endpoint impact if applicable
        vpc_impact = waterfall_data['summary'].get('vpc_endpoint_impact_pct', 0)
        if vpc_impact > 0:
            st.metric(
                "⚠️ VPC Endpoint Impact",
                f"{vpc_impact:.1f}%",
                delta=f"-{waterfall_data['summary']['vpc_endpoint_reduction_mbps']:,.0f} Mbps"
            )
        else:
            st.metric(
                "✅ Final Effective",
                f"{waterfall_data['summary']['final_effective_mbps']:,.0f} Mbps",
                delta=f"{waterfall_data['summary']['efficiency_percentage']:.1f}% efficient"
            )
    
    with col5:
        st.metric(
            "🎯 Primary Bottleneck",
            waterfall_data['summary']['primary_bottleneck'],
            delta=f"-{waterfall_data['summary']['total_reduction_mbps']:,.0f} total loss"
        )
    
    # Waterfall chart
    st.markdown("**📊 Bandwidth Degradation Waterfall:**")
    waterfall_chart = create_bandwidth_waterfall_chart(waterfall_data)
    st.plotly_chart(waterfall_chart, use_container_width=True)
    
    # Enhanced breakdown with VPC endpoint details
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔍 Detailed Impact Analysis:**")
        
        impact_data = []
        for step in waterfall_data['steps'][1:-1]:
            if step['value'] < 0:
                impact_data.append({
                    'Factor': step['name'],
                    'Reduction (Mbps)': f"{abs(step['value']):.0f}",
                    'Remaining (Mbps)': f"{step['cumulative']:.0f}",
                    'Impact (%)': f"{(abs(step['value']) / waterfall_data['summary']['theoretical_max_mbps']) * 100:.1f}%"
                })
        
        df_impact = pd.DataFrame(impact_data)
        st.dataframe(df_impact, use_container_width=True)
        
        insights_text = f"""
        <div class="waterfall-card">
            <h4>Key Insights</h4>
            <p><strong>Efficiency:</strong> {waterfall_data['summary']['efficiency_percentage']:.1f}% of theoretical maximum</p>
            <p><strong>Total Loss:</strong> {waterfall_data['summary']['total_reduction_mbps']:,.0f} Mbps reduction</p>
            <p><strong>Primary Bottleneck:</strong> {waterfall_data['summary']['primary_bottleneck']}</p>
            <p><strong>Protocol Overhead:</strong> {waterfall_data['summary']['protocol_overhead_pct']:.1f}%</p>
            <p><strong>Congestion Impact:</strong> {waterfall_data['summary']['congestion_impact_pct']:.1f}%</p>
        """
        
        # Add VPC endpoint specific insights
        vpc_impact = waterfall_data['summary'].get('vpc_endpoint_impact_pct', 0)
        if vpc_impact > 0:
            insights_text += f"""
            <p><strong>VPC Endpoint Impact:</strong> {vpc_impact:.1f}% throughput reduction</p>
            <p><strong>PrivateLink Overhead:</strong> Additional routing latency and processing</p>
            """
        
        insights_text += "</div>"
        
        st.markdown(insights_text, unsafe_allow_html=True)
    
    with col2:
        st.markdown("**📈 Efficiency Breakdown:**")
        
        # Create pie chart for bandwidth allocation
        labels = ['Effective Bandwidth']
        values = [waterfall_data['summary']['final_effective_mbps']]
        colors = ['#22c55e', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4', '#f97316', '#84cc16']
        
        for i, step in enumerate(waterfall_data['steps'][1:-1]):
            if step['value'] < 0:
                labels.append(step['name'] + ' Loss')
                values.append(abs(step['value']))
        
        fig_pie = px.pie(
            values=values,
            names=labels,
            title="Bandwidth Allocation Breakdown",
            color_discrete_sequence=colors
        )
        
        fig_pie.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Bandwidth: %{value:.0f} Mbps<br>Percentage: %{percent}<extra></extra>'
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
    
    return waterfall_data

def render_network_analysis_tab(config: Dict, analyzer: NetworkPatternAnalyzer):
    """Render network analysis tab with VPC endpoint considerations"""
    st.subheader("🌐 Network Path Analysis")
    
    pattern_key = analyzer.determine_optimal_pattern(
        config['source_location'], 
        config['environment'], 
        config['is_homogeneous']
    )
    
    pattern = analyzer.network_patterns[pattern_key]
    
    # VPC Endpoint compatibility assessment
    vpc_compatibility = render_vpc_endpoint_analysis(config, analyzer)
    
    throughput_analysis = analyzer.calculate_migration_throughput(
        pattern_key,
        config['agent_type'],
        config['agent_size'], 
        config['num_agents']
    )
    
    # Network overview metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "🌐 Network Pattern",
            pattern['pattern_type'].replace('_', ' ').title(),
            delta=f"From {config['source_location']}"
        )
    
    with col2:
        st.metric(
            "📡 Total Bandwidth", 
            f"{pattern['total_bandwidth_mbps']:,} Mbps",
            delta=f"Latency: {pattern['total_latency_ms']}ms"
        )
    
    with col3:
        vpc_impact = throughput_analysis.get('vpc_impact_percent', 0)
        delta_text = f"VPC Impact: -{vpc_impact:.1f}%" if vpc_impact > 0 else f"Utilization: {throughput_analysis['network_utilization_percent']:.1f}%"
        st.metric(
            "⚡ Effective Throughput",
            f"{throughput_analysis['effective_throughput_mbps']:,.0f} Mbps",
            delta=delta_text
        )
    
    with col4:
        st.metric(
            "🤖 Agent Configuration",
            f"{config['num_agents']}x {config['agent_size'].title()}",
            delta=f"{config['agent_type'].upper()}"
        )
    
    with col5:
        bottleneck = throughput_analysis['bottleneck']
        st.metric(
            "🎯 Bottleneck",
            bottleneck.title(),
            delta=f"Efficiency: {throughput_analysis['scaling_efficiency']*100:.1f}%"
        )
    
    # Network path visualization
    st.markdown("**🗺️ Network Path Visualization:**")
    network_diagram = create_network_diagram(pattern_key, analyzer)
    st.plotly_chart(network_diagram, use_container_width=True)
    
    return {
        'pattern_key': pattern_key,
        'pattern': pattern,
        'throughput_analysis': throughput_analysis,
        'vpc_compatibility': vpc_compatibility
    }

def render_migration_timing_tab(config: Dict, network_analysis: Dict, analyzer: NetworkPatternAnalyzer):
    """Render migration timing analysis tab"""
    st.subheader("⏱️ Migration Time Analysis")
    
    throughput_analysis = network_analysis['throughput_analysis']
    
    migration_time = analyzer.estimate_migration_time(
        config['database_size_gb'],
        throughput_analysis['effective_throughput_mbps']
    )
    
    # Time analysis metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "📊 Database Size",
            f"{config['database_size_gb']:,} GB",
            delta=f"{config['database_size_gb'] * 8:,} Gbits"
        )
    
    with col2:
        vpc_impact = throughput_analysis.get('vpc_impact_percent', 0)
        delta_text = f"VPC Impact: -{vpc_impact:.1f}%" if vpc_impact > 0 else f"{throughput_analysis['effective_throughput_mbps']/8:.0f} MB/s"
        st.metric(
            "⚡ Effective Speed",
            f"{throughput_analysis['effective_throughput_mbps']:,.0f} Mbps",
            delta=delta_text
        )
    
    with col3:
        st.metric(
            "🔄 Data Transfer Time",
            f"{migration_time['data_transfer_hours']:.1f} hours",
            delta=f"{migration_time['data_transfer_hours']/24:.1f} days"
        )
    
    with col4:
        st.metric(
            "⏰ Total Migration Time",
            f"{migration_time['total_hours']:.1f} hours",
            delta=f"{migration_time['total_days']:.1f} days"
        )
    
    with col5:
        meets_requirement = migration_time['total_hours'] <= config['max_downtime_hours']
        delta_text = "✅ Meets requirement" if meets_requirement else "❌ Exceeds limit"
        st.metric(
            "🎯 Downtime Check",
            f"{config['max_downtime_hours']} hrs limit",
            delta=delta_text
        )
    
    return migration_time

def render_ai_recommendations_tab(config: Dict, network_analysis: Dict, migration_time: Dict, analyzer: NetworkPatternAnalyzer):
    """Render AI recommendations tab with VPC endpoint considerations"""
    st.subheader("🤖 AI-Powered Migration Recommendations")
    
    analysis_results = {
        'throughput_analysis': network_analysis['throughput_analysis'],
        'migration_time': migration_time,
        'pattern': network_analysis['pattern'],
        'vpc_compatibility': network_analysis.get('vpc_compatibility', {})
    }
    
    ai_recommendations = analyzer.generate_ai_recommendation(config, analysis_results)
    
    # AI Analysis Overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "🎯 Migration Complexity",
            ai_recommendations['migration_complexity'].title(),
            delta=f"Priority Score: {ai_recommendations['overall_priority_score']}"
        )
    
    with col2:
        st.metric(
            "🤖 AI Confidence",
            ai_recommendations['confidence_level'].title(),
            delta=f"{len(ai_recommendations['recommendations'])} recommendations"
        )
    
    with col3:
        bottleneck = analysis_results['throughput_analysis']['bottleneck']
        st.metric(
            "🔍 Primary Bottleneck",
            bottleneck.title(),
            delta="Optimization target"
        )
    
    with col4:
        optimal_agents = min(8, max(1, int(config['max_downtime_hours'] / migration_time['total_hours'] * config['num_agents'])))
        st.metric(
            "⚙️ Optimal Agents",
            f"{optimal_agents}",
            delta=f"Current: {config['num_agents']}"
        )
    
    # AI Recommendations
    st.markdown("**💡 AI-Generated Recommendations:**")
    
    for i, rec in enumerate(ai_recommendations['recommendations'], 1):
        priority_color = {
            'high': '🔴',
            'medium': '🟡', 
            'low': '🟢'
        }.get(rec['priority'], '⚪')
        
        expanded = (rec['priority'] == 'high') or ('vpc_endpoint' in rec['type'])
        
        with st.expander(f"{priority_color} {rec['type'].replace('_', ' ').title()}", expanded=expanded):
            st.write(f"**Description:** {rec['description']}")
            st.write(f"**Expected Impact:** {rec['impact']}")
            st.write(f"**Priority Level:** {rec['priority'].title()}")

# Main application
def main():
    render_header()
    
    # Initialize analyzer
    analyzer = NetworkPatternAnalyzer()
    
    # Sidebar configuration
    config = render_sidebar_controls()
    
    # Main tabs - Updated to include VPC Endpoint considerations
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌐 Network Analysis", 
        "💧 Bandwidth Waterfall",
        "⏱️ Migration Timing", 
        "🤖 AI Recommendations"
    ])
    
    with tab1:
        network_analysis = render_network_analysis_tab(config, analyzer)
    
    with tab2:
        waterfall_data = render_bandwidth_waterfall_tab(config, analyzer)
    
    with tab3:
        migration_time = render_migration_timing_tab(config, network_analysis, analyzer)
    
    with tab4:
        render_ai_recommendations_tab(config, network_analysis, migration_time, analyzer)

# Initialize analyzer globally
analyzer = NetworkPatternAnalyzer()

if __name__ == "__main__":
    main()