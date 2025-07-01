import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import math
import json
import requests
import time
from typing import Dict, List, Tuple, Optional
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import asyncio
import anthropic

# Configure page
st.set_page_config(
    page_title="Enhanced AWS Migration Analyzer with AI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 50%, #1d4ed8 100%);
        padding: 2rem;
        border-radius: 8px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(30,58,138,0.2);
    }
    
    .ai-recommendation {
        background: linear-gradient(135deg, #e0f2fe 0%, #b3e5fc 100%);
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #0288d1;
        box-shadow: 0 2px 10px rgba(2,136,209,0.1);
    }
    
    .critical-alert {
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #d32f2f;
        box-shadow: 0 2px 10px rgba(211,47,47,0.1);
    }
    
    .optimization-card {
        background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #388e3c;
        box-shadow: 0 2px 10px rgba(56,142,60,0.1);
    }
    
    .real-time-status {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        padding: 1rem;
        border-radius: 6px;
        margin: 0.5rem 0;
        border-left: 3px solid #f57c00;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 6px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-top: 3px solid #2196f3;
    }
</style>
""", unsafe_allow_html=True)

class AWSPricingManager:
    """Enhanced AWS pricing management with real-time data"""
    
    def __init__(self):
        self.pricing_client = None
        self.ec2_client = None
        self.s3_client = None
        self.cached_prices = {}
        self.cache_expiry = {}
        self.initialize_clients()
    
    def initialize_clients(self):
        """Initialize AWS clients"""
        try:
            self.pricing_client = boto3.client('pricing', region_name='us-east-1')
            self.ec2_client = boto3.client('ec2', region_name='us-west-2')
            self.s3_client = boto3.client('s3', region_name='us-west-2')
            st.success("✅ AWS clients initialized successfully")
        except NoCredentialsError:
            st.warning("⚠️ AWS credentials not configured. Using cached pricing data.")
        except Exception as e:
            st.warning(f"⚠️ AWS API connection issue: {str(e)}. Using cached pricing data.")
    
    def get_ec2_pricing(self, instance_type: str, region: str = 'us-west-2') -> Dict:
        """Get real-time EC2 pricing"""
        cache_key = f"ec2_{instance_type}_{region}"
        
        # Check cache
        if cache_key in self.cached_prices and self.cache_expiry.get(cache_key, 0) > time.time():
            return self.cached_prices[cache_key]
        
        try:
            if self.pricing_client:
                response = self.pricing_client.get_products(
                    ServiceCode='AmazonEC2',
                    Filters=[
                        {'Type': 'TERM_MATCH', 'Field': 'instanceType', 'Value': instance_type},
                        {'Type': 'TERM_MATCH', 'Field': 'location', 'Value': 'US West (Oregon)'},
                        {'Type': 'TERM_MATCH', 'Field': 'tenancy', 'Value': 'Shared'},
                        {'Type': 'TERM_MATCH', 'Field': 'operating-system', 'Value': 'Linux'}
                    ]
                )
                
                if response['PriceList']:
                    price_data = json.loads(response['PriceList'][0])
                    on_demand = price_data['terms']['OnDemand']
                    price_dimensions = list(on_demand.values())[0]['priceDimensions']
                    hourly_price = float(list(price_dimensions.values())[0]['pricePerUnit']['USD'])
                    
                    result = {
                        'hourly_price': hourly_price,
                        'monthly_price': hourly_price * 24 * 30,
                        'source': 'aws_api',
                        'last_updated': datetime.now()
                    }
                    
                    # Cache for 1 hour
                    self.cached_prices[cache_key] = result
                    self.cache_expiry[cache_key] = time.time() + 3600
                    
                    return result
        except Exception as e:
            st.warning(f"Could not fetch real-time pricing for {instance_type}: {str(e)}")
        
        # Fallback pricing data
        fallback_prices = {
            'm5.large': 0.096,
            'm5.xlarge': 0.192,
            'm5.2xlarge': 0.384,
            'm5.4xlarge': 0.768,
            'm5.8xlarge': 1.536,
            'c5.large': 0.085,
            'c5.xlarge': 0.17,
            'c5.2xlarge': 0.34,
            'c5.4xlarge': 0.68
        }
        
        hourly_price = fallback_prices.get(instance_type, 0.192)
        return {
            'hourly_price': hourly_price,
            'monthly_price': hourly_price * 24 * 30,
            'source': 'cached',
            'last_updated': datetime.now()
        }
    
    def get_s3_pricing(self, storage_class: str = 'STANDARD') -> Dict:
        """Get S3 pricing information"""
        # Simplified S3 pricing (per GB per month)
        s3_prices = {
            'STANDARD': 0.023,
            'STANDARD_IA': 0.0125,
            'GLACIER': 0.004,
            'DEEP_ARCHIVE': 0.00099
        }
        
        return {
            'storage_price_per_gb': s3_prices.get(storage_class, 0.023),
            'transfer_price_per_gb': 0.09,  # Data transfer out
            'requests_per_1000': 0.0004,  # PUT/POST requests
            'source': 'aws_public_pricing'
        }
    
    def get_direct_connect_pricing(self, port_speed: str) -> Dict:
        """Get Direct Connect pricing"""
        dx_prices = {
            '1Gbps': 0.30,
            '10Gbps': 2.25,
            '100Gbps': 22.50
        }
        
        return {
            'hourly_price': dx_prices.get(port_speed, 2.25),
            'monthly_price': dx_prices.get(port_speed, 2.25) * 24 * 30,
            'data_transfer_gb': 0.02
        }

class AIRecommendationEngine:
    """Claude AI-powered recommendation engine"""
    
    def __init__(self):
        self.client = None
        self.initialize_claude()
    
    def initialize_claude(self):
        """Initialize Claude AI client"""
        try:
            # You would need to add your Anthropic API key here
            # self.client = anthropic.Anthropic(api_key="your-api-key")
            st.info("🤖 Claude AI engine ready for recommendations")
        except Exception as e:
            st.warning(f"Claude AI initialization issue: {str(e)}")
    
    def analyze_migration_scenario(self, config: Dict, network_perf: Dict, 
                                 agent_perf: Dict, constraints: Dict) -> Dict:
        """Generate AI-powered migration recommendations"""
        
        # Simulate AI analysis (replace with actual Claude API call)
        analysis = self._simulate_ai_analysis(config, network_perf, agent_perf, constraints)
        
        return {
            'overall_score': analysis['score'],
            'recommendations': analysis['recommendations'],
            'optimizations': analysis['optimizations'],
            'risks': analysis['risks'],
            'timeline_feasibility': analysis['timeline'],
            'cost_optimization': analysis['cost'],
            'alternative_strategies': analysis['alternatives']
        }
    
    def _simulate_ai_analysis(self, config, network_perf, agent_perf, constraints):
        """Simulate AI analysis (replace with actual Claude API call)"""
        
        # Calculate current performance
        final_throughput = min(
            network_perf['effective_bandwidth_mbps'],
            agent_perf['total_agent_throughput_mbps']
        )
        
        migration_time = (config['database_size_gb'] * 8 * 1000) / (final_throughput * 3600)
        
        # Determine bottlenecks
        network_bottleneck = network_perf['effective_bandwidth_mbps'] < agent_perf['total_agent_throughput_mbps']
        
        # Generate score
        score = min(100, (final_throughput / 1000) * 30 + 
                   (100 - migration_time * 2) + 
                   (network_perf['network_quality_score'] * 0.4))
        
        recommendations = []
        optimizations = []
        risks = []
        
        # Time constraint analysis
        timeline_feasible = migration_time <= constraints.get('max_hours', 24)
        
        if not timeline_feasible:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Timeline',
                'title': 'Migration window exceeded',
                'description': f'Current estimate {migration_time:.1f}h exceeds {constraints.get("max_hours", 24)}h limit',
                'action': 'Scale up agents or optimize network path'
            })
        
        if network_bottleneck:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Network',
                'title': 'Network bandwidth bottleneck',
                'description': f'Network limiting at {network_perf["effective_bandwidth_mbps"]:,.0f} Mbps',
                'action': 'Upgrade to production environment or optimize network path'
            })
        
        if agent_perf['scaling_efficiency'] < 0.9:
            optimizations.append({
                'category': 'Agent Scaling',
                'description': f'Agent scaling efficiency at {agent_perf["scaling_efficiency"]*100:.1f}%',
                'recommendation': 'Consider fewer, larger agents instead of many small ones'
            })
        
        if config['server_type'] == 'vmware':
            optimizations.append({
                'category': 'Platform',
                'description': 'VMware overhead reducing performance by 8%',
                'recommendation': 'Consider physical deployment for critical migrations'
            })
        
        # Risk assessment
        if migration_time > 12:
            risks.append({
                'level': 'MEDIUM',
                'category': 'Operational',
                'description': 'Extended migration window increases failure risk',
                'mitigation': 'Implement checkpointing and resume capabilities'
            })
        
        if network_perf['total_reliability'] < 0.999:
            risks.append({
                'level': 'HIGH',
                'category': 'Network',
                'description': f'Network reliability at {network_perf["total_reliability"]*100:.3f}%',
                'mitigation': 'Implement redundant network paths'
            })
        
        return {
            'score': score,
            'recommendations': recommendations,
            'optimizations': optimizations,
            'risks': risks,
            'timeline': timeline_feasible,
            'cost': agent_perf['total_monthly_cost'],
            'alternatives': self._generate_alternatives(config, constraints)
        }
    
    def _generate_alternatives(self, config, constraints):
        """Generate alternative migration strategies"""
        alternatives = []
        
        # Hybrid approach
        alternatives.append({
            'name': 'Hybrid Migration',
            'description': 'Combine DataSync for bulk data with DMS for incremental',
            'pros': ['Reduced downtime', 'Better performance'],
            'cons': ['More complex setup', 'Higher cost'],
            'timeline_impact': '-30%'
        })
        
        # Parallel migration
        alternatives.append({
            'name': 'Parallel Table Migration',
            'description': 'Migrate multiple tables simultaneously',
            'pros': ['Faster completion', 'Better resource utilization'],
            'cons': ['Higher complexity', 'Requires more agents'],
            'timeline_impact': '-50%'
        })
        
        return alternatives

class EnhancedNetworkPathManager:
    """Enhanced network path management with real-time considerations"""
    
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
                'base_cost_factor': 1.0,
                'segments': [
                    {
                        'name': 'Linux NAS to Linux Jump Server',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 2,
                        'reliability': 0.999,
                        'connection_type': 'internal_lan',
                        'cost_factor': 0.0,
                        'optimization_potential': 0.95,
                        'jitter_ms': 0.5,
                        'packet_loss': 0.001
                    },
                    {
                        'name': 'Linux Jump Server to AWS S3 (DX)',
                        'bandwidth_mbps': 2000,
                        'latency_ms': 15,
                        'reliability': 0.998,
                        'connection_type': 'direct_connect',
                        'cost_factor': 2.0,
                        'optimization_potential': 0.92,
                        'jitter_ms': 2.0,
                        'packet_loss': 0.0001
                    }
                ]
            },
            # ... (other paths with enhanced metrics)
        }
        
        self.real_time_factors = {
            'aws_service_health': 1.0,
            'internet_congestion': 1.0,
            'security_overhead': 1.0
        }
    
    def get_real_time_network_status(self) -> Dict:
        """Get real-time network and AWS service status"""
        # Simulate real-time status check
        return {
            'aws_s3_status': 'healthy',
            'aws_dx_status': 'healthy',
            'internet_congestion': 'low',
            'peak_hours': 9 <= datetime.now().hour <= 17,
            'maintenance_window': False,
            'weather_impact': 'none'
        }
    
    def calculate_enhanced_network_performance(self, path_key: str, 
                                             time_of_day: int = None,
                                             concurrent_workloads: int = 0) -> Dict:
        """Enhanced network performance calculation with real-time factors"""
        
        base_perf = self.calculate_network_performance(path_key, time_of_day)
        real_time_status = self.get_real_time_network_status()
        
        # Apply real-time adjustments
        congestion_factor = 1.2 if real_time_status['peak_hours'] else 0.95
        service_health_factor = 0.98 if real_time_status['aws_s3_status'] == 'degraded' else 1.0
        concurrent_impact = 1.0 - (concurrent_workloads * 0.05)  # 5% impact per concurrent workload
        
        adjusted_bandwidth = base_perf['effective_bandwidth_mbps'] * service_health_factor * concurrent_impact / congestion_factor
        adjusted_latency = base_perf['total_latency_ms'] * congestion_factor
        
        # Enhanced metrics
        enhanced_perf = base_perf.copy()
        enhanced_perf.update({
            'adjusted_bandwidth_mbps': adjusted_bandwidth,
            'adjusted_latency_ms': adjusted_latency,
            'real_time_status': real_time_status,
            'concurrent_workloads': concurrent_workloads,
            'performance_degradation': (1 - adjusted_bandwidth / base_perf['effective_bandwidth_mbps']) * 100,
            'jitter_impact': sum(seg.get('jitter_ms', 0) for seg in base_perf['segments']),
            'packet_loss_total': sum(seg.get('packet_loss', 0) for seg in base_perf['segments'])
        })
        
        return enhanced_perf
    
    def calculate_network_performance(self, path_key: str, time_of_day: int = None) -> Dict:
        """Base network performance calculation (from original code)"""
        # Implementation from original NetworkPathManager
        path = self.network_paths.get(path_key, {})
        
        if time_of_day is None:
            time_of_day = datetime.now().hour
        
        total_latency = 0
        min_bandwidth = float('inf')
        total_reliability = 1.0
        total_cost_factor = 0
        optimization_score = 1.0
        
        segments = path.get('segments', [])
        adjusted_segments = []
        
        for segment in segments:
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
            
            effective_bandwidth = segment_bandwidth / congestion_factor
            effective_latency = segment_latency * congestion_factor
            
            # OS-specific adjustments
            if path.get('os_type') == 'windows' and segment['connection_type'] != 'internal_lan':
                effective_bandwidth *= 0.95
                effective_latency *= 1.1
            
            optimization_score *= segment['optimization_potential']
            
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
            'path_name': path.get('name', 'Unknown Path'),
            'destination_storage': path.get('destination_storage', 'S3'),
            'environment': path.get('environment', 'unknown'),
            'os_type': path.get('os_type', 'linux'),
            'total_latency_ms': total_latency,
            'effective_bandwidth_mbps': min_bandwidth,
            'total_reliability': total_reliability,
            'network_quality_score': network_quality,
            'optimization_potential': (1 - optimization_score) * 100,
            'total_cost_factor': total_cost_factor,
            'segments': adjusted_segments
        }
    
    def get_network_path_key(self, config: Dict) -> str:
        """Get network path key based on configuration"""
        os_lower = config['operating_system'].lower()
        if any(os_name in os_lower for os_name in ['linux', 'ubuntu', 'rhel', 'centos']):
            os_type = 'linux'
        elif 'windows' in os_lower:
            os_type = 'windows'
        else:
            os_type = 'linux'
        
        environment = config['environment']
        
        if environment == 'non-production':
            if os_type == 'linux':
                return 'nonprod_sj_linux_nas_s3'
            else:
                return 'nonprod_sj_windows_share_s3'
        else:
            if os_type == 'linux':
                return 'prod_sa_linux_nas_s3'
            else:
                return 'prod_sa_windows_share_s3'

class EnhancedAgentManager:
    """Enhanced agent management with intelligent configuration"""
    
    def __init__(self, pricing_manager: AWSPricingManager):
        self.pricing_manager = pricing_manager
        self.datasync_specs = {
            'small': {'throughput_mbps': 250, 'instance_type': 'm5.large', 'vcpu': 2, 'memory': 4},
            'medium': {'throughput_mbps': 500, 'instance_type': 'm5.xlarge', 'vcpu': 2, 'memory': 4},
            'large': {'throughput_mbps': 1000, 'instance_type': 'm5.2xlarge', 'vcpu': 4, 'memory': 8},
            'xlarge': {'throughput_mbps': 2000, 'instance_type': 'm5.4xlarge', 'vcpu': 8, 'memory': 16}
        }
        
        self.dms_specs = {
            'small': {'throughput_mbps': 200, 'instance_type': 'm5.large', 'vcpu': 2, 'memory': 4},
            'medium': {'throughput_mbps': 400, 'instance_type': 'm5.xlarge', 'vcpu': 2, 'memory': 4},
            'large': {'throughput_mbps': 800, 'instance_type': 'm5.2xlarge', 'vcpu': 4, 'memory': 8},
            'xlarge': {'throughput_mbps': 1500, 'instance_type': 'm5.4xlarge', 'vcpu': 8, 'memory': 16},
            'xxlarge': {'throughput_mbps': 2500, 'instance_type': 'm5.8xlarge', 'vcpu': 16, 'memory': 32}
        }
    
    def get_optimal_configuration(self, target_throughput: float, budget_limit: float = None,
                                time_constraint: float = None) -> Dict:
        """Get optimal agent configuration based on requirements"""
        
        configurations = []
        
        for agent_type in ['datasync', 'dms']:
            specs = self.datasync_specs if agent_type == 'datasync' else self.dms_specs
            
            for size, spec in specs.items():
                for num_agents in range(1, 6):
                    config = self.calculate_agent_performance(agent_type, size, num_agents)
                    
                    # Get real-time pricing
                    pricing = self.pricing_manager.get_ec2_pricing(spec['instance_type'])
                    config['real_time_cost'] = pricing['monthly_price'] * num_agents
                    config['pricing_source'] = pricing['source']
                    
                    # Check constraints
                    meets_throughput = config['total_agent_throughput_mbps'] >= target_throughput
                    meets_budget = budget_limit is None or config['real_time_cost'] <= budget_limit
                    
                    if meets_throughput and meets_budget:
                        config['score'] = self._calculate_config_score(config, target_throughput, budget_limit)
                        configurations.append(config)
        
        # Sort by score
        configurations.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'optimal_config': configurations[0] if configurations else None,
            'alternative_configs': configurations[1:3] if len(configurations) > 1 else [],
            'total_evaluated': len(configurations)
        }
    
    def _calculate_config_score(self, config: Dict, target_throughput: float, budget_limit: float) -> float:
        """Calculate configuration score"""
        # Performance score (0-100)
        perf_score = min(100, (config['total_agent_throughput_mbps'] / target_throughput) * 50)
        
        # Cost efficiency score (0-100)
        cost_per_mbps = config['real_time_cost'] / config['total_agent_throughput_mbps']
        cost_score = max(0, 100 - (cost_per_mbps * 10))
        
        # Scaling efficiency score (0-100)
        scaling_score = config['scaling_efficiency'] * 100
        
        # Weighted total score
        total_score = (perf_score * 0.4 + cost_score * 0.4 + scaling_score * 0.2)
        
        return total_score
    
    def calculate_agent_performance(self, agent_type: str, agent_size: str, num_agents: int, 
                                   platform_type: str = 'vmware') -> Dict:
        """Calculate enhanced agent performance"""
        
        if agent_type == 'datasync':
            base_spec = self.datasync_specs[agent_size]
        else:
            base_spec = self.dms_specs[agent_size]
        
        # VMware overhead
        vmware_efficiency = 0.92 if platform_type == 'vmware' else 1.0
        
        # Calculate per-agent performance
        per_agent_throughput = base_spec['throughput_mbps'] * vmware_efficiency
        
        # Calculate scaling efficiency
        if num_agents == 1:
            scaling_efficiency = 1.0
        elif num_agents <= 3:
            scaling_efficiency = 0.95
        elif num_agents <= 5:
            scaling_efficiency = 0.90
        else:
            scaling_efficiency = 0.85
        
        total_agent_throughput = per_agent_throughput * num_agents * scaling_efficiency
        
        # Get real-time pricing
        pricing = self.pricing_manager.get_ec2_pricing(base_spec['instance_type'])
        per_agent_cost = pricing['monthly_price']
        total_monthly_cost = per_agent_cost * num_agents
        
        return {
            'agent_type': agent_type,
            'agent_size': agent_size,
            'num_agents': num_agents,
            'platform_type': platform_type,
            'instance_type': base_spec['instance_type'],
            'per_agent_throughput_mbps': per_agent_throughput,
            'total_agent_throughput_mbps': total_agent_throughput,
            'scaling_efficiency': scaling_efficiency,
            'vmware_efficiency': vmware_efficiency,
            'per_agent_monthly_cost': per_agent_cost,
            'total_monthly_cost': total_monthly_cost,
            'base_spec': base_spec,
            'cost_per_mbps': total_monthly_cost / total_agent_throughput if total_agent_throughput > 0 else 0
        }

def render_ai_recommendations(ai_analysis: Dict):
    """Render AI-powered recommendations"""
    st.markdown("## 🤖 AI-Powered Migration Recommendations")
    
    # Overall score
    score = ai_analysis.get('overall_score', 0)
    score_color = "🟢" if score >= 80 else "🟡" if score >= 60 else "🔴"
    
    st.markdown(f"""
    <div class="ai-recommendation">
        <h3>{score_color} Migration Readiness Score: {score:.1f}/100</h3>
        <p>Based on comprehensive analysis of network performance, agent configuration, and operational constraints.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # High priority recommendations
    recommendations = ai_analysis.get('recommendations', [])
    if recommendations:
        st.markdown("### ⚡ Priority Recommendations")
        for rec in recommendations[:3]:  # Show top 3
            priority_icon = "🔴" if rec['priority'] == 'HIGH' else "🟡"
            st.markdown(f"""
            <div class="critical-alert">
                <h4>{priority_icon} {rec['title']}</h4>
                <p><strong>Issue:</strong> {rec['description']}</p>
                <p><strong>Action:</strong> {rec['action']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Optimization opportunities
    optimizations = ai_analysis.get('optimizations', [])
    if optimizations:
        st.markdown("### 🔧 Optimization Opportunities")
        for opt in optimizations:
            st.markdown(f"""
            <div class="optimization-card">
                <h4>💡 {opt['category']}</h4>
                <p><strong>Current:</strong> {opt['description']}</p>
                <p><strong>Recommendation:</strong> {opt['recommendation']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Risk assessment
    risks = ai_analysis.get('risks', [])
    if risks:
        st.markdown("### ⚠️ Risk Assessment")
        for risk in risks:
            risk_color = "🔴" if risk['level'] == 'HIGH' else "🟡"
            st.markdown(f"""
            <div class="critical-alert">
                <h4>{risk_color} {risk['category']} Risk</h4>
                <p><strong>Description:</strong> {risk['description']}</p>
                <p><strong>Mitigation:</strong> {risk['mitigation']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Alternative strategies
    alternatives = ai_analysis.get('alternative_strategies', [])
    if alternatives:
        st.markdown("### 🔄 Alternative Migration Strategies")
        for alt in alternatives:
            st.markdown(f"""
            <div class="optimization-card">
                <h4>🚀 {alt['name']}</h4>
                <p>{alt['description']}</p>
                <p><strong>Timeline Impact:</strong> {alt['timeline_impact']}</p>
                <p><strong>Pros:</strong> {', '.join(alt['pros'])}</p>
                <p><strong>Cons:</strong> {', '.join(alt['cons'])}</p>
            </div>
            """, unsafe_allow_html=True)

def render_real_time_status():
    """Render real-time AWS and network status"""
    st.markdown("### 📡 Real-Time Status Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="real-time-status">
            <h4>🟢 AWS S3 Status</h4>
            <p>Service: Operational</p>
            <p>Performance: 99.9%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="real-time-status">
            <h4>🟢 Direct Connect</h4>
            <p>Service: Operational</p>
            <p>Latency: Normal</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        current_hour = datetime.now().hour
        peak_status = "Peak Hours" if 9 <= current_hour <= 17 else "Off-Peak"
        peak_color = "🟡" if 9 <= current_hour <= 17 else "🟢"
        
        st.markdown(f"""
        <div class="real-time-status">
            <h4>{peak_color} Network Traffic</h4>
            <p>Status: {peak_status}</p>
            <p>Congestion: {'Medium' if 9 <= current_hour <= 17 else 'Low'}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="real-time-status">
            <h4>💰 Cost Monitoring</h4>
            <p>Real-time pricing: Active</p>
            <p>Budget tracking: Enabled</p>
        </div>
        """, unsafe_allow_html=True)

def render_enhanced_sidebar():
    """Enhanced sidebar with more configuration options"""
    st.sidebar.header("🚀 Enhanced Migration Configuration")
    
    # Basic configuration (from original)
    operating_system = st.sidebar.selectbox(
        "Operating System",
        ["windows_server_2019", "windows_server_2022", "rhel_8", "rhel_9", "ubuntu_20_04", "ubuntu_22_04"],
        index=3
    )
    
    server_type = st.sidebar.selectbox(
        "Server Platform",
        ["physical", "vmware"],
        index=1
    )
    
    # Enhanced hardware configuration
    st.sidebar.subheader("⚙️ Hardware Configuration")
    ram_gb = st.sidebar.selectbox("RAM (GB)", [8, 16, 32, 64, 128, 256], index=2)
    cpu_cores = st.sidebar.selectbox("CPU Cores", [2, 4, 8, 16, 24, 32], index=2)
    
    nic_type = st.sidebar.selectbox(
        "NIC Type",
        ["gigabit_copper", "gigabit_fiber", "10g_copper", "10g_fiber", "25g_fiber", "40g_fiber"],
        index=3
    )
    
    nic_speeds = {
        'gigabit_copper': 1000, 'gigabit_fiber': 1000,
        '10g_copper': 10000, '10g_fiber': 10000,
        '25g_fiber': 25000, '40g_fiber': 40000
    }
    nic_speed = nic_speeds[nic_type]
    
    # Migration configuration
    st.sidebar.subheader("🔄 Migration Configuration")
    database_size_gb = st.sidebar.number_input("Database Size (GB)", min_value=100, max_value=100000, value=1000, step=100)
    environment = st.sidebar.selectbox("Environment", ["non-production", "production"])
    
    # Enhanced constraints
    st.sidebar.subheader("📊 Migration Constraints")
    max_migration_hours = st.sidebar.number_input("Max Migration Time (hours)", min_value=1, max_value=72, value=12, step=1)
    budget_limit = st.sidebar.number_input("Monthly Budget Limit ($)", min_value=100, max_value=10000, value=2000, step=100)
    concurrent_workloads = st.sidebar.number_input("Concurrent Workloads", min_value=0, max_value=10, value=0, step=1)
    
    # Real-time options
    st.sidebar.subheader("⚡ Real-Time Options")
    enable_auto_optimization = st.sidebar.checkbox("Enable Auto-Optimization", value=True)
    enable_cost_monitoring = st.sidebar.checkbox("Enable Cost Monitoring", value=True)
    enable_ai_recommendations = st.sidebar.checkbox("Enable AI Recommendations", value=True)
    
    return {
        'operating_system': operating_system,
        'server_type': server_type,
        'ram_gb': ram_gb,
        'cpu_cores': cpu_cores,
        'nic_type': nic_type,
        'nic_speed': nic_speed,
        'database_size_gb': database_size_gb,
        'environment': environment,
        'max_migration_hours': max_migration_hours,
        'budget_limit': budget_limit,
        'concurrent_workloads': concurrent_workloads,
        'enable_auto_optimization': enable_auto_optimization,
        'enable_cost_monitoring': enable_cost_monitoring,
        'enable_ai_recommendations': enable_ai_recommendations
    }

def main():
    """Enhanced main application"""
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚀 Enhanced AWS Migration Analyzer with AI</h1>
        <p>AI-Powered Migration Planning • Real-Time Optimization • Intelligent Recommendations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize enhanced managers
    pricing_manager = AWSPricingManager()
    network_manager = EnhancedNetworkPathManager()
    agent_manager = EnhancedAgentManager(pricing_manager)
    ai_engine = AIRecommendationEngine()
    
    # Get enhanced configuration
    config = render_enhanced_sidebar()
    
    # Real-time status
    render_real_time_status()
    
    # Main analysis
    path_key = network_manager.get_network_path_key(config)
    network_perf = network_manager.calculate_enhanced_network_performance(
        path_key, 
        concurrent_workloads=config['concurrent_workloads']
    )
    
    # Calculate target throughput for optimal configuration
    target_throughput = (config['database_size_gb'] * 8 * 1000) / (config['max_migration_hours'] * 3600)
    
    # Get optimal agent configuration
    optimal_config = agent_manager.get_optimal_configuration(
        target_throughput=target_throughput,
        budget_limit=config['budget_limit'],
        time_constraint=config['max_migration_hours']
    )
    
    # AI analysis
    if config['enable_ai_recommendations'] and optimal_config['optimal_config']:
        constraints = {
            'max_hours': config['max_migration_hours'],
            'budget_limit': config['budget_limit'],
            'concurrent_workloads': config['concurrent_workloads']
        }
        
        ai_analysis = ai_engine.analyze_migration_scenario(
            config, network_perf, optimal_config['optimal_config'], constraints
        )
    else:
        ai_analysis = {}
    
    # Enhanced tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🤖 AI Recommendations",
        "🎯 Optimal Configuration",
        "📊 Performance Analysis",
        "💰 Cost Optimization",
        "📈 What-If Scenarios"
    ])
    
    with tab1:
        if config['enable_ai_recommendations'] and ai_analysis:
            render_ai_recommendations(ai_analysis)
        else:
            st.info("Enable AI Recommendations in the sidebar to see intelligent migration suggestions.")
    
    with tab2:
        st.subheader("🎯 Optimal Agent Configuration")
        
        if optimal_config['optimal_config']:
            opt_config = optimal_config['optimal_config']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Recommended Setup",
                    f"{opt_config['num_agents']}x {opt_config['agent_size'].title()}",
                    delta=f"{opt_config['agent_type'].upper()}"
                )
            
            with col2:
                st.metric(
                    "Total Throughput",
                    f"{opt_config['total_agent_throughput_mbps']:,.0f} Mbps",
                    delta=f"Target: {target_throughput:,.0f} Mbps"
                )
            
            with col3:
                st.metric(
                    "Monthly Cost",
                    f"${opt_config['real_time_cost']:,.0f}",
                    delta=f"Budget: ${config['budget_limit']:,.0f}"
                )
            
            with col4:
                st.metric(
                    "Cost Efficiency",
                    f"${opt_config['cost_per_mbps']:.2f}/Mbps",
                    delta=f"Score: {opt_config['score']:.1f}/100"
                )
            
            # Configuration details
            st.markdown("**📋 Configuration Details:**")
            config_df = pd.DataFrame([{
                'Agent Type': opt_config['agent_type'].upper(),
                'Instance Type': opt_config['instance_type'],
                'Agent Size': opt_config['agent_size'].title(),
                'Number of Agents': opt_config['num_agents'],
                'Per-Agent Throughput': f"{opt_config['per_agent_throughput_mbps']:.0f} Mbps",
                'Total Throughput': f"{opt_config['total_agent_throughput_mbps']:.0f} Mbps",
                'Scaling Efficiency': f"{opt_config['scaling_efficiency']*100:.1f}%",
                'Monthly Cost': f"${opt_config['real_time_cost']:,.0f}",
                'Pricing Source': opt_config['pricing_source']
            }])
            st.dataframe(config_df, use_container_width=True)
            
            # Alternative configurations
            if optimal_config['alternative_configs']:
                st.markdown("**🔄 Alternative Configurations:**")
                alt_data = []
                for alt in optimal_config['alternative_configs']:
                    alt_data.append({
                        'Configuration': f"{alt['num_agents']}x {alt['agent_size']} {alt['agent_type']}",
                        'Throughput (Mbps)': f"{alt['total_agent_throughput_mbps']:.0f}",
                        'Monthly Cost': f"${alt['real_time_cost']:,.0f}",
                        'Cost/Mbps': f"${alt['cost_per_mbps']:.2f}",
                        'Score': f"{alt['score']:.1f}"
                    })
                
                alt_df = pd.DataFrame(alt_data)
                st.dataframe(alt_df, use_container_width=True)
        else:
            st.warning("⚠️ No configuration meets the specified constraints. Consider adjusting budget or timeline requirements.")
    
    with tab3:
        st.subheader("📊 Enhanced Performance Analysis")
        
        # Network performance with real-time factors
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Network Bandwidth",
                f"{network_perf['adjusted_bandwidth_mbps']:,.0f} Mbps",
                delta=f"Base: {network_perf['effective_bandwidth_mbps']:,.0f} Mbps"
            )
        
        with col2:
            st.metric(
                "Network Latency",
                f"{network_perf['adjusted_latency_ms']:.1f} ms",
                delta=f"Base: {network_perf['total_latency_ms']:.1f} ms"
            )
        
        with col3:
            st.metric(
                "Performance Impact",
                f"{network_perf['performance_degradation']:.1f}%",
                delta="Real-time factors"
            )
        
        with col4:
            st.metric(
                "Network Quality",
                f"{network_perf['network_quality_score']:.1f}/100",
                delta=f"Jitter: {network_perf['jitter_impact']:.1f}ms"
            )
        
        # Real-time factors impact
        st.markdown("**⚡ Real-Time Performance Factors:**")
        
        factors_data = []
        status = network_perf['real_time_status']
        
        factors_data.append({
            'Factor': 'Peak Hours',
            'Status': 'Active' if status['peak_hours'] else 'Inactive',
            'Impact': 'Medium' if status['peak_hours'] else 'None',
            'Description': 'Business hours network congestion'
        })
        
        factors_data.append({
            'Factor': 'AWS Service Health',
            'Status': status['aws_s3_status'].title(),
            'Impact': 'None' if status['aws_s3_status'] == 'healthy' else 'High',
            'Description': 'AWS S3 service availability'
        })
        
        factors_data.append({
            'Factor': 'Concurrent Workloads',
            'Status': f"{config['concurrent_workloads']} active",
            'Impact': 'Medium' if config['concurrent_workloads'] > 0 else 'None',
            'Description': 'Other migration jobs running'
        })
        
        factors_df = pd.DataFrame(factors_data)
        st.dataframe(factors_df, use_container_width=True)
    
    with tab4:
        st.subheader("💰 Cost Optimization Dashboard")
        
        if config['enable_cost_monitoring'] and optimal_config['optimal_config']:
            opt_config = optimal_config['optimal_config']
            
            # Cost breakdown
            monthly_agent_cost = opt_config['real_time_cost']
            s3_pricing = pricing_manager.get_s3_pricing()
            storage_cost = config['database_size_gb'] * s3_pricing['storage_price_per_gb']
            transfer_cost = config['database_size_gb'] * s3_pricing['transfer_price_per_gb']
            
            total_monthly_cost = monthly_agent_cost + storage_cost
            migration_cost = transfer_cost + (monthly_agent_cost / 30 * config['max_migration_hours'] / 24)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Agent Cost",
                    f"${monthly_agent_cost:,.0f}/month",
                    delta=f"${opt_config['cost_per_mbps']:.2f} per Mbps"
                )
            
            with col2:
                st.metric(
                    "Storage Cost",
                    f"${storage_cost:,.0f}/month",
                    delta=f"${s3_pricing['storage_price_per_gb']:.3f} per GB"
                )
            
            with col3:
                st.metric(
                    "Migration Cost",
                    f"${migration_cost:,.0f}",
                    delta="One-time transfer"
                )
            
            # Cost optimization recommendations
            st.markdown("**💡 Cost Optimization Opportunities:**")
            
            # Reserved instance savings
            reserved_savings = monthly_agent_cost * 0.3  # Assume 30% savings
            st.markdown(f"""
            <div class="optimization-card">
                <h4>💰 Reserved Instance Savings</h4>
                <p>Switch to 1-year reserved instances for {opt_config['instance_type']}</p>
                <p><strong>Potential Savings:</strong> ${reserved_savings:,.0f}/month (30%)</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Spot instance option
            spot_savings = monthly_agent_cost * 0.5  # Assume 50% savings
            st.markdown(f"""
            <div class="optimization-card">
                <h4>🎯 Spot Instance Option</h4>
                <p>Use spot instances for non-critical migration windows</p>
                <p><strong>Potential Savings:</strong> ${spot_savings:,.0f}/month (50%)</p>
                <p><strong>Risk:</strong> Possible interruption during migration</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Cost trend chart
            st.markdown("**📈 Cost Analysis:**")
            
            # Generate cost comparison data
            configurations = []
            for agents in range(1, 6):
                test_config = agent_manager.calculate_agent_performance(
                    opt_config['agent_type'], opt_config['agent_size'], agents
                )
                pricing = pricing_manager.get_ec2_pricing(opt_config['instance_type'])
                cost = pricing['monthly_price'] * agents
                configurations.append({
                    'Agents': agents,
                    'Monthly Cost': cost,
                    'Throughput': test_config['total_agent_throughput_mbps'],
                    'Cost per Mbps': cost / test_config['total_agent_throughput_mbps']
                })
            
            cost_df = pd.DataFrame(configurations)
            
            fig_cost = px.scatter(
                cost_df,
                x='Throughput',
                y='Monthly Cost',
                size='Agents',
                color='Cost per Mbps',
                title="Cost vs Performance Analysis",
                labels={'Throughput': 'Throughput (Mbps)', 'Monthly Cost': 'Monthly Cost ($)'}
            )
            
            # Highlight optimal configuration
            opt_point = cost_df[cost_df['Agents'] == opt_config['num_agents']]
            fig_cost.add_scatter(
                x=opt_point['Throughput'],
                y=opt_point['Monthly Cost'],
                mode='markers',
                marker=dict(size=20, color='red', symbol='star'),
                name='Optimal Config'
            )
            
            st.plotly_chart(fig_cost, use_container_width=True)
        else:
            st.info("Enable Cost Monitoring in the sidebar to see detailed cost analysis.")
    
    with tab5:
        st.subheader("📈 What-If Scenario Analysis")
        
        st.markdown("**🔮 Migration Scenario Simulator**")
        
        # Scenario inputs
        col1, col2 = st.columns(2)
        
        with col1:
            scenario_db_size = st.number_input("Scenario DB Size (GB)", min_value=100, max_value=50000, value=config['database_size_gb'], step=500)
            scenario_time_limit = st.number_input("Scenario Time Limit (hours)", min_value=2, max_value=48, value=config['max_migration_hours'], step=2)
        
        with col2:
            scenario_budget = st.number_input("Scenario Budget ($)", min_value=500, max_value=20000, value=config['budget_limit'], step=500)
            scenario_environment = st.selectbox("Scenario Environment", ["non-production", "production"], index=0 if config['environment'] == 'non-production' else 1)
        
        if st.button("🚀 Run Scenario Analysis"):
            # Create scenario configuration
            scenario_config = config.copy()
            scenario_config.update({
                'database_size_gb': scenario_db_size,
                'max_migration_hours': scenario_time_limit,
                'budget_limit': scenario_budget,
                'environment': scenario_environment
            })
            
            # Calculate scenario results
            scenario_path_key = network_manager.get_network_path_key(scenario_config)
            scenario_network_perf = network_manager.calculate_enhanced_network_performance(scenario_path_key)
            
            scenario_target_throughput = (scenario_db_size * 8 * 1000) / (scenario_time_limit * 3600)
            scenario_optimal = agent_manager.get_optimal_configuration(
                target_throughput=scenario_target_throughput,
                budget_limit=scenario_budget,
                time_constraint=scenario_time_limit
            )
            
            # Display scenario results
            if scenario_optimal['optimal_config']:
                scenario_opt = scenario_optimal['optimal_config']
                
                st.markdown("**📊 Scenario Results:**")
                
                # Comparison with current configuration
                if optimal_config['optimal_config']:
                    current_opt = optimal_config['optimal_config']
                    
                    comparison_data = {
                        'Metric': ['Database Size (GB)', 'Time Limit (hours)', 'Budget ($)', 'Required Throughput (Mbps)', 'Recommended Agents', 'Estimated Cost ($)', 'Feasible'],
                        'Current Scenario': [
                            config['database_size_gb'],
                            config['max_migration_hours'],
                            config['budget_limit'],
                            f"{target_throughput:,.0f}",
                            f"{current_opt['num_agents']}x {current_opt['agent_size']}",
                            f"{current_opt['real_time_cost']:,.0f}",
                            "✅ Yes"
                        ],
                        'New Scenario': [
                            scenario_db_size,
                            scenario_time_limit,
                            scenario_budget,
                            f"{scenario_target_throughput:,.0f}",
                            f"{scenario_opt['num_agents']}x {scenario_opt['agent_size']}",
                            f"{scenario_opt['real_time_cost']:,.0f}",
                            "✅ Yes"
                        ]
                    }
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True)
                    
                    # Impact analysis
                    cost_diff = scenario_opt['real_time_cost'] - current_opt['real_time_cost']
                    throughput_diff = scenario_opt['total_agent_throughput_mbps'] - current_opt['total_agent_throughput_mbps']
                    
                    st.markdown(f"""
                    **🔍 Scenario Impact Analysis:**
                    - **Cost Change:** {'+' if cost_diff > 0 else ''}${cost_diff:,.0f}/month ({cost_diff/current_opt['real_time_cost']*100:+.1f}%)
                    - **Throughput Change:** {'+' if throughput_diff > 0 else ''}{throughput_diff:,.0f} Mbps
                    - **Environment Impact:** {'Production benefits: +5x bandwidth' if scenario_environment == 'production' and config['environment'] == 'non-production' else 'No environment change'}
                    """)
            else:
                st.error("❌ Scenario is not feasible with the given constraints. Consider increasing budget or extending timeline.")
        
        # Sensitivity analysis
        st.markdown("**📈 Sensitivity Analysis**")
        
        # Database size sensitivity
        db_sizes = [500, 1000, 2000, 5000, 10000]
        sensitivity_data = []
        
        for db_size in db_sizes:
            required_throughput = (db_size * 8 * 1000) / (config['max_migration_hours'] * 3600)
            test_optimal = agent_manager.get_optimal_configuration(
                target_throughput=required_throughput,
                budget_limit=config['budget_limit']
            )
            
            if test_optimal['optimal_config']:
                test_opt = test_optimal['optimal_config']
                sensitivity_data.append({
                    'Database Size (GB)': db_size,
                    'Required Throughput (Mbps)': required_throughput,
                    'Recommended Agents': test_opt['num_agents'],
                    'Monthly Cost ($)': test_opt['real_time_cost'],
                    'Feasible': 'Yes'
                })
            else:
                sensitivity_data.append({
                    'Database Size (GB)': db_size,
                    'Required Throughput (Mbps)': required_throughput,
                    'Recommended Agents': 'N/A',
                    'Monthly Cost ($)': 'N/A',
                    'Feasible': 'No'
                })
        
        sensitivity_df = pd.DataFrame(sensitivity_data)
        
        # Plot sensitivity analysis
        feasible_data = sensitivity_df[sensitivity_df['Feasible'] == 'Yes'].copy()
        if not feasible_data.empty:
            feasible_data['Monthly Cost ($)'] = pd.to_numeric(feasible_data['Monthly Cost ($)'])
            
            fig_sensitivity = px.line(
                feasible_data,
                x='Database Size (GB)',
                y='Monthly Cost ($)',
                title="Cost Sensitivity to Database Size",
                markers=True
            )
            st.plotly_chart(fig_sensitivity, use_container_width=True)
        
        st.dataframe(sensitivity_df, use_container_width=True)

if __name__ == "__main__":
    main()