import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import math
import json
import requests
import boto3
from typing import Dict, List, Tuple, Optional
import asyncio
from dataclasses import dataclass
import numpy as np
import math  # Required for the new DataSync scaling calculations
# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="AWS Migration Network Analyzer | Enterprise Edition",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CORPORATE CSS STYLING
# =============================================================================

st.markdown("""
<style>
    /* Corporate Color Scheme */
    :root {
        --primary-blue: #1e3a8a;
        --secondary-blue: #3b82f6;
        --accent-blue: #60a5fa;
        --dark-gray: #1f2937;
        --medium-gray: #6b7280;
        --light-gray: #f3f4f6;
        --success-green: #059669;
        --warning-orange: #d97706;
        --error-red: #dc2626;
        --corporate-gradient: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #60a5fa 100%);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    
    /* Corporate Header */
    .corporate-header {
        background: var(--corporate-gradient);
        padding: 3rem 2rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(30, 58, 138, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .corporate-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000"><defs><pattern id="grid" width="50" height="50" patternUnits="userSpaceOnUse"><path d="M 50 0 L 0 0 0 50" fill="none" stroke="rgba(255,255,255,0.05)" stroke-width="1"/></pattern></defs><rect width="100%" height="100%" fill="url(%23grid)"/></svg>');
        opacity: 0.3;
    }
    
    .corporate-header h1 {
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        position: relative;
        z-index: 1;
    }
    
    .corporate-header .subtitle {
        font-size: 1.4rem;
        margin-bottom: 0.5rem;
        opacity: 0.95;
        position: relative;
        z-index: 1;
    }
    
    .corporate-header .tagline {
        font-size: 1rem;
        opacity: 0.85;
        position: relative;
        z-index: 1;
    }
    
    /* Corporate Cards */
    .corporate-card {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
        border: 1px solid #e5e7eb;
        border-left: 4px solid var(--secondary-blue);
        transition: all 0.3s ease;
    }
    
    .corporate-card:hover {
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
        transform: translateY(-2px);
    }
    
    .corporate-card h3, .corporate-card h4 {
        color: var(--primary-blue);
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    /* Status Cards */
    .status-card-success {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        border-left: 4px solid var(--success-green);
    }
    
    .status-card-warning {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        border-left: 4px solid var(--warning-orange);
    }
    
    .status-card-error {
        background: linear-gradient(135deg, #fef2f2 0%, #fecaca 100%);
        border-left: 4px solid var(--error-red);
    }
    
    .status-card-info {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border-left: 4px solid var(--secondary-blue);
    }
    
    /* Metric Cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        border: 1px solid #e5e7eb;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary-blue);
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: var(--medium-gray);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Best Pattern Highlight */
    .best-pattern-highlight {
        background: var(--corporate-gradient);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        font-weight: 600;
        font-size: 1.2rem;
        margin: 2rem 0;
        box-shadow: 0 8px 24px rgba(30, 58, 138, 0.3);
    }
    
    /* Waterfall Chart Container */
    .waterfall-container {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
        border: 1px solid #e5e7eb;
        margin: 1.5rem 0;
    }
    
    /* Data Tables */
    .dataframe {
        border: none !important;
    }
    
    .dataframe th {
        background-color: var(--light-gray) !important;
        color: var(--primary-blue) !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        font-size: 0.8rem !important;
        letter-spacing: 0.5px !important;
    }
    
    .dataframe td {
        border-bottom: 1px solid #e5e7eb !important;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background-color: #f8fafc;
    }
    
    .sidebar .sidebar-content {
        background-color: #f8fafc;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: var(--light-gray);
        border-radius: 8px;
        padding: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 6px;
        color: var(--medium-gray);
        font-weight: 500;
        padding: 12px 24px;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: white !important;
        color: var(--primary-blue) !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* Buttons */
    .stButton > button {
        background: var(--corporate-gradient);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        box-shadow: 0 4px 12px rgba(30, 58, 138, 0.3);
        transform: translateY(-1px);
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background-color: var(--light-gray);
        border-radius: 8px;
        color: var(--primary-blue);
        font-weight: 500;
    }
    
    /* Enhanced Progress Steps */
    .flow-step {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 0.75rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        border: 1px solid #e5e7eb;
        position: relative;
        transition: all 0.3s ease;
    }
    
    .flow-step:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
        transform: translateX(4px);
    }
    
    .flow-step-header {
        display: flex;
        align-items: center;
        margin-bottom: 0.5rem;
    }
    
    .flow-step-number {
        background: var(--secondary-blue);
        color: white;
        width: 32px;
        height: 32px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.9rem;
        margin-right: 1rem;
        flex-shrink: 0;
    }
    
    .flow-step-title {
        font-weight: 600;
        color: var(--primary-blue);
        font-size: 1.1rem;
        flex: 1;
    }
    
    .flow-step-details {
        margin-left: 3rem;
    }
    
    .flow-step-value {
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    
    .flow-step-description {
        color: var(--medium-gray);
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    
    .flow-step-remaining {
        color: var(--success-green);
        font-weight: 600;
        font-size: 1rem;
    }
    
    .flow-step-reduction {
        color: var(--error-red);
    }
    
    .flow-step-starting {
        border-left: 4px solid var(--success-green);
    }
    
    .flow-step-reduction-type {
        border-left: 4px solid var(--warning-orange);
    }
    
    .flow-step-final {
        border-left: 4px solid var(--primary-blue);
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
    }
    
    .flow-arrow {
        text-align: center;
        margin: 0.5rem 0;
        color: var(--medium-gray);
        font-size: 1.5rem;
        font-weight: 700;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .corporate-header h1 {
            font-size: 2rem;
        }
        
        .corporate-header .subtitle {
            font-size: 1.1rem;
        }
        
        .metric-grid {
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        }
        
        .flow-step-details {
            margin-left: 0;
            margin-top: 1rem;
        }
        
        .flow-step-header {
            flex-direction: column;
            align-items: flex-start;
        }
        
        .flow-step-number {
            margin-right: 0;
            margin-bottom: 0.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DATA CLASSES AND TYPE DEFINITIONS
# =============================================================================

@dataclass
class PatternAnalysis:
    """Data class for pattern analysis results"""
    pattern_name: str
    total_cost_usd: float
    migration_time_hours: float
    effective_bandwidth_mbps: float
    reliability_score: float
    complexity_score: float
    ai_recommendation_score: float
    use_cases: List[str]
    pros: List[str]
    cons: List[str]

# =============================================================================
# AWS PRICING CLIENT
# =============================================================================

class AWSPricingClient:
    """AWS Pricing API client for real-time cost data"""
    
    def __init__(self):
        self.pricing_client = None
        self.region = 'us-east-1'  # Pricing API is only available in us-east-1
        
    def initialize_client(self, aws_access_key: str = None, aws_secret_key: str = None):
        """Initialize AWS pricing client"""
        try:
            if aws_access_key and aws_secret_key:
                self.pricing_client = boto3.client(
                    'pricing',
                    region_name=self.region,
                    aws_access_key_id=aws_access_key,
                    aws_secret_access_key=aws_secret_key
                )
            else:
                # Try to use default credentials
                self.pricing_client = boto3.client('pricing', region_name=self.region)
        except Exception as e:
            st.warning(f"Could not initialize AWS pricing client: {e}")
            self.pricing_client = None
    
    def get_direct_connect_pricing(self, port_speed: str, location: str = 'US East (N. Virginia)') -> Dict:
        """Get Direct Connect pricing"""
        if not self.pricing_client:
            return self._get_mock_dx_pricing(port_speed)
        
        try:
            response = self.pricing_client.get_products(
                ServiceCode='AmazonConnect',
                Filters=[
                    {'Type': 'TERM_MATCH', 'Field': 'location', 'Value': location},
                    {'Type': 'TERM_MATCH', 'Field': 'portSpeed', 'Value': port_speed}
                ]
            )
            
            if response['PriceList']:
                price_data = json.loads(response['PriceList'][0])
                return self._parse_dx_pricing(price_data)
            else:
                return self._get_mock_dx_pricing(port_speed)
                
        except Exception as e:
            st.warning(f"Error fetching Direct Connect pricing: {e}")
            return self._get_mock_dx_pricing(port_speed)
    
    def get_ec2_pricing(self, instance_type: str, region: str = 'US East (N. Virginia)') -> Dict:
        """Get EC2 instance pricing"""
        if not self.pricing_client:
            return self._get_mock_ec2_pricing(instance_type)
        
        try:
            response = self.pricing_client.get_products(
                ServiceCode='AmazonEC2',
                Filters=[
                    {'Type': 'TERM_MATCH', 'Field': 'location', 'Value': region},
                    {'Type': 'TERM_MATCH', 'Field': 'instanceType', 'Value': instance_type},
                    {'Type': 'TERM_MATCH', 'Field': 'tenancy', 'Value': 'Shared'},
                    {'Type': 'TERM_MATCH', 'Field': 'operating-system', 'Value': 'Linux'}
                ]
            )
            
            if response['PriceList']:
                price_data = json.loads(response['PriceList'][0])
                return self._parse_ec2_pricing(price_data)
            else:
                return self._get_mock_ec2_pricing(instance_type)
                
        except Exception as e:
            st.warning(f"Error fetching EC2 pricing: {e}")
            return self._get_mock_ec2_pricing(instance_type)
    
    def _get_mock_dx_pricing(self, port_speed: str) -> Dict:
        """Mock Direct Connect pricing data"""
        pricing_map = {
            '1Gbps': {'port_hour': 0.30, 'data_transfer_gb': 0.02},
            '10Gbps': {'port_hour': 2.25, 'data_transfer_gb': 0.02},
            '100Gbps': {'port_hour': 22.50, 'data_transfer_gb': 0.015}
        }
        return pricing_map.get(port_speed, pricing_map['10Gbps'])
    
    def _get_mock_ec2_pricing(self, instance_type: str) -> Dict:
        """Mock EC2 pricing data"""
        pricing_map = {
            'm5.large': {'hourly': 0.096},
            'm5.xlarge': {'hourly': 0.192},
            'm5.2xlarge': {'hourly': 0.384},
            'm5.4xlarge': {'hourly': 0.768},
            'c5.xlarge': {'hourly': 0.17},
            'c5.2xlarge': {'hourly': 0.34},
            'r5.xlarge': {'hourly': 0.252},
            'dms.t3.medium': {'hourly': 0.0464},
            'dms.r5.large': {'hourly': 0.144},
            'dms.r5.xlarge': {'hourly': 0.288},
            'dms.r5.2xlarge': {'hourly': 0.576}
        }
        return pricing_map.get(instance_type, pricing_map['m5.xlarge'])

# =============================================================================
# CLAUDE AI CLIENT
# =============================================================================

class ClaudeAIClient:
    """Claude AI client for intelligent recommendations"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.base_url = "https://api.anthropic.com/v1/messages"
        
    def get_pattern_recommendation(self, analysis_data: Dict) -> Dict:
        """Get AI recommendation for best migration pattern"""
        if not self.api_key:
            return self._get_mock_ai_recommendation(analysis_data)
        
        try:
            prompt = self._build_analysis_prompt(analysis_data)
            
            headers = {
                "x-api-key": self.api_key,
                "content-type": "application/json",
                "anthropic-version": "2023-06-01"
            }
            
            payload = {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 1000,
                "messages": [{"role": "user", "content": prompt}]
            }
            
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                return self._parse_ai_response(result['content'][0]['text'])
            else:
                st.warning(f"Claude AI API error: {response.status_code}")
                return self._get_mock_ai_recommendation(analysis_data)
                
        except Exception as e:
            st.warning(f"Error calling Claude AI: {e}")
            return self._get_mock_ai_recommendation(analysis_data)
    
    def _build_analysis_prompt(self, analysis_data: Dict) -> str:
        """Build analysis prompt for Claude AI"""
        return f"""
        You are an AWS network architecture expert helping a database engineer choose the best migration pattern.
        
        Analysis Data:
        - Data Size: {analysis_data.get('data_size_gb', 0)} GB
        - Migration Service: {analysis_data.get('migration_service', 'unknown')}
        - Environment: {analysis_data.get('environment', 'unknown')}
        - Max Downtime: {analysis_data.get('max_downtime_hours', 0)} hours
        - Source Location: {analysis_data.get('source_location', 'unknown')}
        
        Available Patterns:
        1. VPC Endpoint: Lower cost, higher latency, works with compatible services
        2. Direct Connect: Higher cost, lower latency, dedicated bandwidth
        3. Multi-hop: Highest cost, variable latency, complex routing
        
        Please recommend the best pattern considering:
        - Total cost (infrastructure + time)
        - Migration time constraints
        - Database-specific requirements
        - Risk factors
        
        Respond in JSON format with:
        {{
            "recommended_pattern": "pattern_name",
            "confidence_score": 0.85,
            "reasoning": "explanation",
            "cost_justification": "why this cost is worth it",
            "risk_assessment": "potential risks",
            "database_considerations": "specific to database migration"
        }}
        """
    
    def _parse_ai_response(self, response_text: str) -> Dict:
        """Parse Claude AI response"""
        try:
            # Try to extract JSON from response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                return json.loads(json_str)
        except:
            pass
        
        # Fallback to mock response
        return {
            "recommended_pattern": "direct_connect",
            "confidence_score": 0.75,
            "reasoning": "Based on analysis, Direct Connect provides the best balance of performance and reliability",
            "cost_justification": "Higher upfront cost offset by reduced migration time and improved reliability",
            "risk_assessment": "Low risk with proper planning and testing",
            "database_considerations": "Direct Connect provides consistent latency crucial for database replication"
        }
    
    def _get_mock_ai_recommendation(self, analysis_data: Dict) -> Dict:
        """Mock AI recommendation based on simple rules"""
        data_size = analysis_data.get('data_size_gb', 0)
        environment = analysis_data.get('environment', 'non-production')
        
        if environment == 'production' and data_size > 500:
            pattern = "direct_connect"
            reasoning = "Production environment with large dataset requires dedicated, reliable connectivity"
        elif data_size < 100:
            pattern = "vpc_endpoint"
            reasoning = "Small dataset can efficiently use VPC endpoint with lower cost"
        else:
            pattern = "direct_connect"
            reasoning = "Medium to large dataset benefits from dedicated bandwidth"
        
        return {
            "recommended_pattern": pattern,
            "confidence_score": 0.80,
            "reasoning": reasoning,
            "cost_justification": "Optimized for data size and environment requirements",
            "risk_assessment": "Standard risk mitigation recommended",
            "database_considerations": "Ensure consistent connectivity for database integrity"
        }

# =============================================================================
# ENHANCED NETWORK ANALYZER CLASS
# =============================================================================

class EnhancedNetworkAnalyzer:
    """Comprehensive network analyzer with realistic infrastructure modeling and migration services + AI enhancements"""
    
    def __init__(self):
        # Operating System Network Stack Characteristics
        self.os_characteristics = {
            'windows_server_2019': {
                'name': 'Windows Server 2019',
                'tcp_stack_efficiency': 0.94,
                'memory_copy_overhead': 0.08,
                'interrupt_overhead': 0.05,
                'kernel_bypass_support': False,
                'max_tcp_window_size': '64KB',
                'rss_support': True,
                'network_virtualization_overhead': 0.12
            },
            'windows_server_2022': {
                'name': 'Windows Server 2022',
                'tcp_stack_efficiency': 0.96,
                'memory_copy_overhead': 0.06,
                'interrupt_overhead': 0.04,
                'kernel_bypass_support': True,
                'max_tcp_window_size': '1MB',
                'rss_support': True,
                'network_virtualization_overhead': 0.08
            },
            'linux_rhel8': {
                'name': 'Red Hat Enterprise Linux 8',
                'tcp_stack_efficiency': 0.97,
                'memory_copy_overhead': 0.04,
                'interrupt_overhead': 0.03,
                'kernel_bypass_support': True,
                'max_tcp_window_size': '16MB',
                'rss_support': True,
                'network_virtualization_overhead': 0.05
            },
            'linux_ubuntu': {
                'name': 'Ubuntu Linux (Latest)',
                'tcp_stack_efficiency': 0.98,
                'memory_copy_overhead': 0.03,
                'interrupt_overhead': 0.02,
                'kernel_bypass_support': True,
                'max_tcp_window_size': '16MB',
                'rss_support': True,
                'network_virtualization_overhead': 0.04
            }
        }
        
        # Network Interface Card Characteristics
        self.nic_characteristics = {
            '1gbps_standard': {
                'name': '1 Gbps Standard NIC',
                'theoretical_bandwidth_mbps': 1000,
                'real_world_efficiency': 0.94,
                'cpu_utilization_per_gbps': 0.15,
                'pcie_gen': '2.0',
                'pcie_lanes': 1,
                'hardware_offload_support': ['checksum'],
                'mtu_support': 1500,
                'buffer_size_mb': 1,
                'interrupt_coalescing': False
            },
            '10gbps_standard': {
                'name': '10 Gbps Standard NIC',
                'theoretical_bandwidth_mbps': 10000,
                'real_world_efficiency': 0.92,
                'cpu_utilization_per_gbps': 0.08,
                'pcie_gen': '3.0',
                'pcie_lanes': 4,
                'hardware_offload_support': ['checksum', 'segmentation', 'rss'],
                'mtu_support': 9000,
                'buffer_size_mb': 4,
                'interrupt_coalescing': True
            },
            '25gbps_high_performance': {
                'name': '25 Gbps High-Performance NIC',
                'theoretical_bandwidth_mbps': 25000,
                'real_world_efficiency': 0.96,
                'cpu_utilization_per_gbps': 0.04,
                'pcie_gen': '3.0',
                'pcie_lanes': 8,
                'hardware_offload_support': ['checksum', 'segmentation', 'rss', 'rdma'],
                'mtu_support': 9000,
                'buffer_size_mb': 16,
                'interrupt_coalescing': True
            },
            '100gbps_enterprise': {
                'name': '100 Gbps Enterprise NIC',
                'theoretical_bandwidth_mbps': 100000,
                'real_world_efficiency': 0.98,
                'cpu_utilization_per_gbps': 0.02,
                'pcie_gen': '4.0',
                'pcie_lanes': 16,
                'hardware_offload_support': ['checksum', 'segmentation', 'rss', 'rdma', 'encryption'],
                'mtu_support': 9000,
                'buffer_size_mb': 64,
                'interrupt_coalescing': True
            }
        }
        
        # LAN Infrastructure Characteristics
        self.lan_characteristics = {
            'gigabit_switch': {
                'name': 'Gigabit Ethernet Switch',
                'switching_capacity_gbps': 48,
                'port_buffer_mb': 12,
                'switching_latency_us': 5,
                'oversubscription_ratio': '3:1',
                'congestion_threshold': 0.8,
                'qos_support': True,
                'vlan_overhead': 0.01
            },
            '10gb_switch': {
                'name': '10 Gigabit Ethernet Switch',
                'switching_capacity_gbps': 480,
                'port_buffer_mb': 48,
                'switching_latency_us': 2,
                'oversubscription_ratio': '2:1',
                'congestion_threshold': 0.85,
                'qos_support': True,
                'vlan_overhead': 0.005
            },
            '25gb_switch': {
                'name': '25 Gigabit Ethernet Switch',
                'switching_capacity_gbps': 1200,
                'port_buffer_mb': 128,
                'switching_latency_us': 1,
                'oversubscription_ratio': '1.5:1',
                'congestion_threshold': 0.9,
                'qos_support': True,
                'vlan_overhead': 0.003
            },
            'spine_leaf_fabric': {
                'name': 'Spine-Leaf Fabric',
                'switching_capacity_gbps': 5000,
                'port_buffer_mb': 256,
                'switching_latency_us': 0.5,
                'oversubscription_ratio': '1:1',
                'congestion_threshold': 0.95,
                'qos_support': True,
                'vlan_overhead': 0.001
            }
        }
        
        # WAN Provider Characteristics
        self.wan_characteristics = {
            'mpls_tier1': {
                'name': 'MPLS Tier-1 Provider',
                'bandwidth_efficiency': 0.96,
                'latency_consistency': 0.98,
                'packet_loss_rate': 0.0001,
                'jitter_ms': 2,
                'burstable_overhead': 0.1,
                'qos_classes': 4,
                'sla_availability': 0.9999
            },
            'fiber_metro': {
                'name': 'Metro Fiber Ethernet',
                'bandwidth_efficiency': 0.98,
                'latency_consistency': 0.99,
                'packet_loss_rate': 0.00005,
                'jitter_ms': 1,
                'burstable_overhead': 0.05,
                'qos_classes': 8,
                'sla_availability': 0.99995
            },
            'internet_transit': {
                'name': 'Internet Transit',
                'bandwidth_efficiency': 0.85,
                'latency_consistency': 0.9,
                'packet_loss_rate': 0.001,
                'jitter_ms': 10,
                'burstable_overhead': 0.2,
                'qos_classes': 0,
                'sla_availability': 0.999
            },
            'aws_dx_dedicated': {
                'name': 'AWS Direct Connect Dedicated',
                'bandwidth_efficiency': 0.99,
                'latency_consistency': 0.999,
                'packet_loss_rate': 0.00001,
                'jitter_ms': 0.5,
                'burstable_overhead': 0.02,
                'qos_classes': 8,
                'sla_availability': 0.9999
            }
        }
        
        # AWS Direct Connect Specific Factors
        self.dx_characteristics = {
            '1gbps_dedicated': {
                'name': '1 Gbps Dedicated Connection',
                'committed_bandwidth_mbps': 1000,
                'burst_capability': 1.0,
                'aws_edge_processing_overhead': 0.02,
                'cross_connect_latency_ms': 1,
                'bgp_convergence_impact': 0.01,
                'virtual_interface_overhead': 0.005
            },
            '10gbps_dedicated': {
                'name': '10 Gbps Dedicated Connection',
                'committed_bandwidth_mbps': 10000,
                'burst_capability': 1.0,
                'aws_edge_processing_overhead': 0.01,
                'cross_connect_latency_ms': 0.8,
                'bgp_convergence_impact': 0.005,
                'virtual_interface_overhead': 0.003
            },
            '100gbps_dedicated': {
                'name': '100 Gbps Dedicated Connection',
                'committed_bandwidth_mbps': 100000,
                'burst_capability': 1.0,
                'aws_edge_processing_overhead': 0.005,
                'cross_connect_latency_ms': 0.5,
                'bgp_convergence_impact': 0.002,
                'virtual_interface_overhead': 0.001
            }
        }
        
        # Enhanced Network Patterns with realistic infrastructure + database scenarios
        self.network_patterns = {
            'sj_nonprod_vpc_endpoint': {
                'name': 'San Jose Non-Prod → AWS VPC Endpoint',
                'source': 'San Jose',
                'environment': 'non-production',
                'pattern_type': 'vpc_endpoint',
                'os_type': 'linux_rhel8',
                'nic_type': '10gbps_standard',
                'lan_type': '10gb_switch',
                'wan_type': 'fiber_metro',
                'dx_type': None,
                'committed_bandwidth_mbps': 2000,
                'baseline_latency_ms': 8,
                'cost_factor': 1.5,
                'security_level': 'high',
                'reliability_score': 0.85,
                'complexity_score': 0.3,
                'vpc_endpoint_limitations': {
                    'ipv4_only': True,
                    'no_shared_vpc': True,
                    'privatelink_routing_overhead': 0.03
                },
                'database_suitability': {
                    'oltp': 0.7,
                    'olap': 0.9,
                    'replication': 0.6,
                    'backup': 0.95
                }
            },
            'sj_nonprod_direct_connect': {
                'name': 'San Jose Non-Prod → AWS Direct Connect',
                'source': 'San Jose',
                'environment': 'non-production',
                'pattern_type': 'direct_connect',
                'os_type': 'linux_rhel8',
                'nic_type': '10gbps_standard',
                'lan_type': '10gb_switch',
                'wan_type': 'mpls_tier1',
                'dx_type': '10gbps_dedicated',
                'committed_bandwidth_mbps': 2000,
                'baseline_latency_ms': 12,
                'cost_factor': 2.0,
                'security_level': 'high',
                'reliability_score': 0.95,
                'complexity_score': 0.6,
                'database_suitability': {
                    'oltp': 0.9,
                    'olap': 0.95,
                    'replication': 0.95,
                    'backup': 0.9
                }
            },
            'sj_prod_direct_connect': {
                'name': 'San Jose Production → AWS Direct Connect',
                'source': 'San Jose',
                'environment': 'production',
                'pattern_type': 'direct_connect',
                'os_type': 'linux_ubuntu',
                'nic_type': '25gbps_high_performance',
                'lan_type': 'spine_leaf_fabric',
                'wan_type': 'aws_dx_dedicated',
                'dx_type': '100gbps_dedicated',
                'committed_bandwidth_mbps': 10000,
                'baseline_latency_ms': 6,
                'cost_factor': 3.5,
                'security_level': 'very_high',
                'reliability_score': 0.99,
                'complexity_score': 0.7,
                'database_suitability': {
                    'oltp': 0.98,
                    'olap': 0.98,
                    'replication': 0.99,
                    'backup': 0.95
                }
            },
            'sa_prod_via_sj': {
                'name': 'San Antonio Production → San Jose → AWS',
                'source': 'San Antonio',
                'environment': 'production',
                'pattern_type': 'multi_hop',
                'os_type': 'windows_server_2022',
                'nic_type': '25gbps_high_performance',
                'lan_type': 'spine_leaf_fabric',
                'wan_type': 'mpls_tier1',
                'dx_type': '100gbps_dedicated',
                'committed_bandwidth_mbps': 10000,
                'baseline_latency_ms': 18,
                'cost_factor': 4.0,
                'security_level': 'very_high',
                'reliability_score': 0.92,
                'complexity_score': 0.9,
                'database_suitability': {
                    'oltp': 0.85,
                    'olap': 0.95,
                    'replication': 0.88,
                    'backup': 0.9
                }
            }
        }
        
        # Enhanced Database Scenarios with AWS targets
        self.database_scenarios = {
            'mysql_oltp_rds': {
                'name': 'MySQL OLTP Database → RDS MySQL',
                'workload_type': 'oltp',
                'aws_target': 'rds',
                'target_service': 'Amazon RDS for MySQL',
                'latency_sensitivity': 'high',
                'bandwidth_requirement': 'medium',
                'consistency_requirement': 'strict',
                'recommended_services': ['dms'],
                'min_bandwidth_mbps': 500,
                'max_tolerable_latency_ms': 10,
                'migration_complexity': 'low',
                'downtime_sensitivity': 'high'
            },
            'postgresql_analytics_rds': {
                'name': 'PostgreSQL Analytics → RDS PostgreSQL',
                'workload_type': 'olap',
                'aws_target': 'rds',
                'target_service': 'Amazon RDS for PostgreSQL',
                'latency_sensitivity': 'medium',
                'bandwidth_requirement': 'high',
                'consistency_requirement': 'eventual',
                'recommended_services': ['dms', 'datasync'],
                'min_bandwidth_mbps': 1000,
                'max_tolerable_latency_ms': 50,
                'migration_complexity': 'medium',
                'downtime_sensitivity': 'medium'
            },
            'oracle_enterprise_rds': {
                'name': 'Oracle Enterprise → RDS Oracle',
                'workload_type': 'mixed',
                'aws_target': 'rds',
                'target_service': 'Amazon RDS for Oracle',
                'latency_sensitivity': 'very_high',
                'bandwidth_requirement': 'high',
                'consistency_requirement': 'strict',
                'recommended_services': ['dms'],
                'min_bandwidth_mbps': 2000,
                'max_tolerable_latency_ms': 5,
                'migration_complexity': 'high',
                'downtime_sensitivity': 'very_high'
            },
            'sqlserver_enterprise_ec2': {
                'name': 'SQL Server Enterprise → EC2',
                'workload_type': 'mixed',
                'aws_target': 'ec2',
                'target_service': 'SQL Server on Amazon EC2',
                'latency_sensitivity': 'high',
                'bandwidth_requirement': 'high',
                'consistency_requirement': 'strict',
                'recommended_services': ['dms', 'datasync'],
                'min_bandwidth_mbps': 1500,
                'max_tolerable_latency_ms': 8,
                'migration_complexity': 'high',
                'downtime_sensitivity': 'high'
            },
            'mongodb_cluster_documentdb': {
                'name': 'MongoDB Cluster → DocumentDB',
                'workload_type': 'mixed',
                'aws_target': 'documentdb',
                'target_service': 'Amazon DocumentDB',
                'latency_sensitivity': 'medium',
                'bandwidth_requirement': 'high',
                'consistency_requirement': 'eventual',
                'recommended_services': ['dms', 'datasync'],
                'min_bandwidth_mbps': 1500,
                'max_tolerable_latency_ms': 20,
                'migration_complexity': 'medium',
                'downtime_sensitivity': 'medium'
            },
            'mysql_analytics_aurora': {
                'name': 'MySQL Analytics → Aurora MySQL',
                'workload_type': 'olap',
                'aws_target': 'aurora',
                'target_service': 'Amazon Aurora MySQL',
                'latency_sensitivity': 'medium',
                'bandwidth_requirement': 'high',
                'consistency_requirement': 'eventual',
                'recommended_services': ['dms'],
                'min_bandwidth_mbps': 1200,
                'max_tolerable_latency_ms': 25,
                'migration_complexity': 'medium',
                'downtime_sensitivity': 'low'
            },
            'postgresql_oltp_aurora': {
                'name': 'PostgreSQL OLTP → Aurora PostgreSQL',
                'workload_type': 'oltp',
                'aws_target': 'aurora',
                'target_service': 'Amazon Aurora PostgreSQL',
                'latency_sensitivity': 'high',
                'bandwidth_requirement': 'medium',
                'consistency_requirement': 'strict',
                'recommended_services': ['dms'],
                'min_bandwidth_mbps': 800,
                'max_tolerable_latency_ms': 12,
                'migration_complexity': 'low',
                'downtime_sensitivity': 'high'
            },
            'mariadb_oltp_rds': {
                'name': 'MariaDB OLTP → RDS MariaDB',
                'workload_type': 'oltp',
                'aws_target': 'rds',
                'target_service': 'Amazon RDS for MariaDB',
                'latency_sensitivity': 'high',
                'bandwidth_requirement': 'medium',
                'consistency_requirement': 'strict',
                'recommended_services': ['dms'],
                'min_bandwidth_mbps': 600,
                'max_tolerable_latency_ms': 10,
                'migration_complexity': 'low',
                'downtime_sensitivity': 'high'
            }
        }
        
        # Comprehensive Migration Services (ALL ORIGINAL SERVICES PRESERVED)
        'datasync': {
                'name': 'AWS DataSync',
                'use_case': 'File and object data transfer',
                'protocols': ['NFS', 'SMB', 'HDFS', 'S3'],
                'vpc_endpoint_compatible': True,
                'encryption_in_transit': True,
                'encryption_at_rest': True,
                'application_efficiency': 0.92,
                'protocol_efficiency': 0.96,
                'latency_sensitivity': 'medium',
                'tcp_window_scaling_required': True,
                'vmware_deployment': True,  # NEW FLAG
                'database_compatibility': {
                    'file_based_backups': True,
                    'live_replication': False,
                    'transaction_logs': True
                },
            'sizes': {
                    'small': {
                        'vcpu': 4, 'memory_gb': 16, 'throughput_mbps': 400, 'cost_per_hour': 0.084,
                        'vpc_endpoint_throughput_reduction': 0.1,
                        'optimal_file_size_mb': '1-100',
                        'concurrent_transfers': 16,
                        'tcp_connections': 16,
                        'instance_type': 'm5.xlarge',
                        'vmware_overhead': 0.15,  # NEW: VMware virtualization overhead
                        'effective_throughput_mbps': 340  # NEW: After VMware overhead
                    },
                'medium': {
                        'vcpu': 8, 'memory_gb': 32, 'throughput_mbps': 1000, 'cost_per_hour': 0.168,
                        'vpc_endpoint_throughput_reduction': 0.08,
                        'optimal_file_size_mb': '100-1000',
                        'concurrent_transfers': 32,
                        'tcp_connections': 32,
                        'instance_type': 'm5.2xlarge',
                        'vmware_overhead': 0.12,
                        'effective_throughput_mbps': 880
                    },
                'large': {
                        'vcpu': 16, 'memory_gb': 64, 'throughput_mbps': 2000, 'cost_per_hour': 0.336,
                        'vpc_endpoint_throughput_reduction': 0.05,
                        'optimal_file_size_mb': '1000+',
                        'concurrent_transfers': 64,
                        'tcp_connections': 64,
                        'instance_type': 'm5.4xlarge',
                        'vmware_overhead': 0.10,
                        'effective_throughput_mbps': 1800
                    },
                'xlarge': {
                        'vcpu': 32, 'memory_gb': 128, 'throughput_mbps': 4000, 'cost_per_hour': 0.672,
                        'vpc_endpoint_throughput_reduction': 0.03,
                        'optimal_file_size_mb': '1000+',
                        'concurrent_transfers': 128,
                        'tcp_connections': 128,
                        'instance_type': 'm5.8xlarge',
                        'vmware_overhead': 0.08,
                        'effective_throughput_mbps': 3680
                    }
                }
            },
            'dms': {
                'name': 'AWS Database Migration Service',
                'use_case': 'Database migration and replication',
                'protocols': ['TCP/IP', 'SSL/TLS'],
                'vpc_endpoint_compatible': True,
                'encryption_in_transit': True,
                'encryption_at_rest': True,
                'application_efficiency': 0.88,
                'protocol_efficiency': 0.94,
                'latency_sensitivity': 'high',
                'requires_endpoints': True,
                'supports_cdc': True,
                'tcp_window_scaling_required': True,
                'database_compatibility': {
                    'file_based_backups': False,
                    'live_replication': True,
                    'transaction_logs': True,
                    'schema_conversion': True
                },
                'sizes': {
                    'small': {
                        'vcpu': 2, 'memory_gb': 4, 'throughput_mbps': 200, 'cost_per_hour': 0.042,
                        'max_connections': 50,
                        'optimal_table_size_gb': '1-10',
                        'tcp_connections': 4,
                        'instance_type': 'dms.t3.medium'
                    },
                    'medium': {
                        'vcpu': 2, 'memory_gb': 8, 'throughput_mbps': 400, 'cost_per_hour': 0.085,
                        'max_connections': 100,
                        'optimal_table_size_gb': '10-100',
                        'tcp_connections': 8,
                        'instance_type': 'dms.r5.large'
                    },
                    'large': {
                        'vcpu': 4, 'memory_gb': 16, 'throughput_mbps': 800, 'cost_per_hour': 0.17,
                        'max_connections': 200,
                        'optimal_table_size_gb': '100-500',
                        'tcp_connections': 16,
                        'instance_type': 'dms.r5.xlarge'
                    },
                    'xlarge': {
                        'vcpu': 8, 'memory_gb': 32, 'throughput_mbps': 1500, 'cost_per_hour': 0.34,
                        'max_connections': 400,
                        'optimal_table_size_gb': '500+',
                        'tcp_connections': 32,
                        'instance_type': 'dms.r5.2xlarge'
                    }
                }
            },
            'fsx_windows': {
                'name': 'Amazon FSx for Windows File Server',
                'use_case': 'Windows-based file shares and applications',
                'protocols': ['SMB', 'NFS', 'iSCSI'],
                'vpc_endpoint_compatible': False,
                'encryption_in_transit': True,
                'encryption_at_rest': True,
                'application_efficiency': 0.95,
                'protocol_efficiency': 0.93,
                'latency_sensitivity': 'low',
                'requires_active_directory': True,
                'supports_deduplication': True,
                'tcp_window_scaling_required': False,
                'database_compatibility': {
                    'file_based_backups': True,
                    'live_replication': False,
                    'transaction_logs': False
                },
                'sizes': {
                    'small': {
                        'storage_gb': 32, 'throughput_mbps': 16, 'cost_per_hour': 0.013,
                        'iops': 96, 'max_concurrent_users': 50
                    },
                    'medium': {
                        'storage_gb': 64, 'throughput_mbps': 32, 'cost_per_hour': 0.025,
                        'iops': 192, 'max_concurrent_users': 100
                    },
                    'large': {
                        'storage_gb': 2048, 'throughput_mbps': 512, 'cost_per_hour': 0.40,
                        'iops': 6144, 'max_concurrent_users': 500
                    }
                }
            },
            'fsx_lustre': {
                'name': 'Amazon FSx for Lustre',
                'use_case': 'High-performance computing and machine learning',
                'protocols': ['Lustre', 'POSIX'],
                'vpc_endpoint_compatible': False,
                'encryption_in_transit': True,
                'encryption_at_rest': True,
                'application_efficiency': 0.98,
                'protocol_efficiency': 0.97,
                'latency_sensitivity': 'very_low',
                'supports_s3_integration': True,
                'tcp_window_scaling_required': False,
                'database_compatibility': {
                    'file_based_backups': True,
                    'live_replication': False,
                    'transaction_logs': False
                },
                'sizes': {
                    'small': {
                        'storage_gb': 1200, 'throughput_mbps': 240, 'cost_per_hour': 0.15,
                        'iops': 'unlimited', 'max_concurrent_clients': 100
                    },
                    'large': {
                        'storage_gb': 7200, 'throughput_mbps': 1440, 'cost_per_hour': 0.90,
                        'iops': 'unlimited', 'max_concurrent_clients': 500
                    }
                }
            },
            'storage_gateway': {
                'name': 'AWS Storage Gateway',
                'use_case': 'Hybrid cloud storage integration',
                'protocols': ['NFS', 'SMB', 'iSCSI', 'VTL'],
                'vpc_endpoint_compatible': True,
                'encryption_in_transit': True,
                'encryption_at_rest': True,
                'application_efficiency': 0.85,
                'protocol_efficiency': 0.92,
                'latency_sensitivity': 'medium',
                'supports_caching': True,
                'tcp_window_scaling_required': True,
                'database_compatibility': {
                    'file_based_backups': True,
                    'live_replication': False,
                    'transaction_logs': True
                },
                'sizes': {
                    'small': {
                        'vcpu': 4, 'memory_gb': 16, 'throughput_mbps': 125, 'cost_per_hour': 0.05,
                        'cache_gb': 150, 'max_volumes': 32,
                        'instance_type': 'm5.xlarge'
                    },
                    'large': {
                        'vcpu': 16, 'memory_gb': 64, 'throughput_mbps': 500, 'cost_per_hour': 0.20,
                        'cache_gb': 600, 'max_volumes': 128,
                        'instance_type': 'm5.4xlarge'
                    }
                }
            }
        }
        
        # Initialize new clients
        self.pricing_client = AWSPricingClient()
        self.ai_client = ClaudeAIClient()
    
    def determine_optimal_pattern(self, source_location: str, environment: str, migration_service: str) -> str:
        """Determine optimal network pattern based on requirements"""
        if source_location == 'San Jose':
            if environment == 'production':
                return 'sj_prod_direct_connect'
            else:
                if migration_service in ['fsx_windows', 'fsx_lustre']:
                    return 'sj_nonprod_direct_connect'
                else:
                    return 'sj_nonprod_vpc_endpoint'
        elif source_location == 'San Antonio':
            return 'sa_prod_via_sj'
        return 'sj_nonprod_direct_connect'
    
    def calculate_realistic_bandwidth_waterfall(self, pattern_key: str, migration_service: str, service_size: str, num_instances: int) -> Dict:
        """Calculate realistic bandwidth waterfall with detailed infrastructure impact"""
        pattern = self.network_patterns[pattern_key]
        service = self.migration_services[migration_service]
        service_spec = service['sizes'][service_size]
        
        # Get infrastructure characteristics
        os_char = self.os_characteristics[pattern['os_type']]
        nic_char = self.nic_characteristics[pattern['nic_type']]
        lan_char = self.lan_characteristics[pattern['lan_type']]
        wan_char = self.wan_characteristics[pattern['wan_type']]
        dx_char = self.dx_characteristics.get(pattern['dx_type']) if pattern['dx_type'] else None
        
        steps = []
        
        # Step 1: Theoretical NIC Maximum
        theoretical_max = nic_char['theoretical_bandwidth_mbps']
        steps.append({
            'name': f'NIC Theoretical ({nic_char["name"]})',
            'value': theoretical_max,
            'cumulative': theoretical_max,
            'type': 'starting',
            'layer': 'nic',
            'step_number': 1
        })
        
        # Step 2: NIC Real-World Efficiency
        nic_efficient_bandwidth = theoretical_max * nic_char['real_world_efficiency']
        nic_reduction = theoretical_max - nic_efficient_bandwidth
        steps.append({
            'name': f'NIC Efficiency Loss ({(1-nic_char["real_world_efficiency"])*100:.1f}%)',
            'value': -nic_reduction,
            'cumulative': nic_efficient_bandwidth,
            'type': 'reduction',
            'layer': 'nic',
            'step_number': 2
        })
        
        # Step 3: Operating System Network Stack
        os_efficient_bandwidth = nic_efficient_bandwidth * os_char['tcp_stack_efficiency']
        os_reduction = nic_efficient_bandwidth - os_efficient_bandwidth
        steps.append({
            'name': f'OS Stack Overhead ({os_char["name"]})',
            'value': -os_reduction,
            'cumulative': os_efficient_bandwidth,
            'type': 'reduction',
            'layer': 'os',
            'step_number': 3
        })
        
        # Step 4: CPU Utilization Impact
        cpu_utilization = (os_efficient_bandwidth / 1000) * nic_char['cpu_utilization_per_gbps']
        cpu_impact_factor = max(0.8, 1 - max(0, (cpu_utilization - 0.6) * 1.5)) if cpu_utilization > 0.6 else 1.0
        cpu_adjusted_bandwidth = os_efficient_bandwidth * cpu_impact_factor
        cpu_reduction = os_efficient_bandwidth - cpu_adjusted_bandwidth
        steps.append({
            'name': f'CPU Utilization Impact ({cpu_utilization*100:.1f}%)',
            'value': -cpu_reduction,
            'cumulative': cpu_adjusted_bandwidth,
            'type': 'reduction',
            'layer': 'os',
            'step_number': 4
        })
        
        # Step 5: LAN Infrastructure
        lan_switching_capacity = lan_char['switching_capacity_gbps'] * 1000
        oversubscription_factor = float(lan_char['oversubscription_ratio'].split(':')[0])
        effective_lan_capacity = lan_switching_capacity / oversubscription_factor
        
        lan_limited_bandwidth = min(cpu_adjusted_bandwidth, effective_lan_capacity)
        lan_reduction = cpu_adjusted_bandwidth - lan_limited_bandwidth
        steps.append({
            'name': f'LAN Infrastructure ({lan_char["name"]})',
            'value': -lan_reduction,
            'cumulative': lan_limited_bandwidth,
            'type': 'reduction',
            'layer': 'lan',
            'step_number': 5
        })
        
        # Step 6: WAN Provider
        wan_bandwidth = min(lan_limited_bandwidth, pattern['committed_bandwidth_mbps'])
        wan_efficient_bandwidth = wan_bandwidth * wan_char['bandwidth_efficiency']
        wan_reduction = lan_limited_bandwidth - wan_efficient_bandwidth
        steps.append({
            'name': f'WAN Provider ({wan_char["name"]})',
            'value': -wan_reduction,
            'cumulative': wan_efficient_bandwidth,
            'type': 'reduction',
            'layer': 'wan',
            'step_number': 6
        })
        
        # Step 7: Direct Connect (if applicable)
        dx_adjusted_bandwidth = wan_efficient_bandwidth
        dx_reduction = 0
        step_number = 7
        if dx_char:
            dx_overhead = wan_efficient_bandwidth * (dx_char['aws_edge_processing_overhead'] + dx_char['virtual_interface_overhead'])
            dx_adjusted_bandwidth = wan_efficient_bandwidth - dx_overhead
            dx_reduction = dx_overhead
            steps.append({
                'name': f'Direct Connect Overhead ({dx_char["name"]})',
                'value': -dx_reduction,
                'cumulative': dx_adjusted_bandwidth,
                'type': 'reduction',
                'layer': 'dx',
                'step_number': step_number
            })
            step_number += 1
        
        # Step 8: VPC Endpoint (if applicable)
        vpc_adjusted_bandwidth = dx_adjusted_bandwidth
        vpc_reduction = 0
        if pattern['pattern_type'] == 'vpc_endpoint' and service['vpc_endpoint_compatible']:
            vpc_limitations = pattern.get('vpc_endpoint_limitations', {})
            vpc_overhead = vpc_limitations.get('privatelink_routing_overhead', 0.03)
            if migration_service == 'datasync':
                vpc_overhead += service_spec.get('vpc_endpoint_throughput_reduction', 0.05)
            vpc_adjusted_bandwidth = dx_adjusted_bandwidth * (1 - vpc_overhead)
            vpc_reduction = dx_adjusted_bandwidth - vpc_adjusted_bandwidth
            steps.append({
                'name': f'VPC Endpoint Overhead ({vpc_overhead*100:.1f}%)',
                'value': -vpc_reduction,
                'cumulative': vpc_adjusted_bandwidth,
                'type': 'reduction',
                'layer': 'vpc',
                'step_number': step_number
            })
            step_number += 1
        
        # Step 9: Protocol Overhead
        protocol_efficiency = service.get('protocol_efficiency', 0.95)
        protocol_adjusted_bandwidth = vpc_adjusted_bandwidth * protocol_efficiency
        protocol_reduction = vpc_adjusted_bandwidth - protocol_adjusted_bandwidth
        steps.append({
            'name': f'Protocol Overhead ({", ".join(service["protocols"])})',
            'value': -protocol_reduction,
            'cumulative': protocol_adjusted_bandwidth,
            'type': 'reduction',
            'layer': 'protocol',
            'step_number': step_number
        })
        step_number += 1
        
        # MODIFY the existing "Step 10: Service Capacity" section in calculate_realistic_bandwidth_waterfall()
# Replace around line 700-750:

        # OLD CODE:
        # Step 10: Service Capacity
        service_capacity = service_spec['throughput_mbps'] * num_instances
        service_limited_bandwidth = min(protocol_adjusted_bandwidth, service_capacity)
        service_reduction = protocol_adjusted_bandwidth - service_limited_bandwidth

        # NEW ENHANCED CODE:
        # Step 10: Service Capacity (Enhanced for DataSync VMware overhead)
        if migration_service == 'datasync' and service.get('vmware_deployment', False):
            # Use effective throughput that accounts for VMware overhead
            single_vm_capacity = service_spec.get('effective_throughput_mbps', 
                                                service_spec['throughput_mbps'] * (1 - service_spec.get('vmware_overhead', 0.1)))
            service_capacity = single_vm_capacity * num_instances
            
            # Add detailed DataSync analysis
            vmware_overhead_pct = service_spec.get('vmware_overhead', 0.1) * 100
            steps.append({
                'name': f'VMware Virtualization Overhead ({vmware_overhead_pct:.0f}%)',
                'value': -(service_spec['throughput_mbps'] * num_instances - service_capacity),
                'cumulative': service_capacity,
                'type': 'reduction',
                'layer': 'service',
                'step_number': step_number,
                'details': f"DataSync VM running on VMware ESXi with {vmware_overhead_pct:.0f}% virtualization overhead"
            })
            step_number += 1
        else:
            # Standard service capacity calculation
            service_capacity = service_spec['throughput_mbps'] * num_instances

        service_limited_bandwidth = min(protocol_adjusted_bandwidth, service_capacity)
        service_reduction = protocol_adjusted_bandwidth - service_limited_bandwidth

        # Enhanced step details for DataSync
        step_name = f'{service["name"]} Capacity'
        if migration_service == 'datasync':
            if num_instances > 1:
                step_name += f' ({num_instances} VMs @ {service_capacity/num_instances:.0f} Mbps each)'
            else:
                step_name += f' (Single VM @ {service_capacity:.0f} Mbps)'

        steps.append({
            'name': step_name,
            'value': -service_reduction,
            'cumulative': service_limited_bandwidth,
            'type': 'reduction',
            'layer': 'service',
            'step_number': step_number,
            'details': f"Service capacity: {service_capacity:.0f} Mbps, Network available: {protocol_adjusted_bandwidth:.0f} Mbps"
        })
        
        # Step 11: Application Efficiency
        app_efficiency = service.get('application_efficiency', 0.9)
        final_bandwidth = service_limited_bandwidth * app_efficiency
        app_reduction = service_limited_bandwidth - final_bandwidth
        steps.append({
            'name': f'Application Efficiency ({app_efficiency*100:.1f}%)',
            'value': -app_reduction,
            'cumulative': final_bandwidth,
            'type': 'reduction',
            'layer': 'service',
            'step_number': step_number
        })
        
        # Final effective bandwidth
        steps.append({
            'name': 'Final Effective Bandwidth',
            'value': final_bandwidth,
            'cumulative': final_bandwidth,
            'type': 'final',
            'layer': 'final',
            'step_number': step_number + 1
        })
        
        # Calculate summary
        total_reduction = theoretical_max - final_bandwidth
        efficiency_percentage = (final_bandwidth / theoretical_max) * 100
        
        # Identify primary bottleneck
        reductions = [(step['name'], abs(step['value']), step['layer']) 
                     for step in steps if step['type'] == 'reduction' and step['value'] < 0]
        primary_bottleneck = max(reductions, key=lambda x: x[1]) if reductions else ('None', 0, 'none')
        
        return {
            'steps': steps,
            'summary': {
                'theoretical_max_mbps': theoretical_max,
                'final_effective_mbps': final_bandwidth,
                'total_reduction_mbps': total_reduction,
                'efficiency_percentage': efficiency_percentage,
                'primary_bottleneck': primary_bottleneck[0],
                'primary_bottleneck_layer': primary_bottleneck[2],
                'primary_bottleneck_impact_mbps': primary_bottleneck[1],
                'service_name': service['name'],
                'baseline_latency_ms': pattern['baseline_latency_ms'],
                'service_utilization_percent': (final_bandwidth / service_capacity * 100) if service_capacity > 0 else 0,
                'network_utilization_percent': (final_bandwidth / pattern['committed_bandwidth_mbps'] * 100),
                'bottleneck': 'network' if final_bandwidth == pattern['committed_bandwidth_mbps'] else 'service'
            },
            'infrastructure_details': {
                'os': os_char,
                'nic': nic_char,
                'lan': lan_char,
                'wan': wan_char,
                'dx': dx_char
            }
        }
    
    def assess_service_compatibility(self, pattern_key: str, migration_service: str, service_size: str) -> Dict:
        """Assess service compatibility with network pattern"""
        pattern = self.network_patterns[pattern_key]
        service = self.migration_services[migration_service]
        
        compatibility_assessment = {
            'service_name': service['name'],
            'is_vpc_endpoint': pattern['pattern_type'] == 'vpc_endpoint',
            'vpc_endpoint_compatible': service.get('vpc_endpoint_compatible', False),
            'warnings': [],
            'requirements': [],
            'recommendations': []
        }
        
        # VPC Endpoint compatibility
        if compatibility_assessment['is_vpc_endpoint']:
            if not compatibility_assessment['vpc_endpoint_compatible']:
                compatibility_assessment['warnings'].append(
                    f"{service['name']} does not support VPC Endpoints. Direct Connect recommended."
                )
        
        # Service-specific requirements
        if service.get('requires_active_directory'):
            compatibility_assessment['requirements'].append(
                "Requires Active Directory integration"
            )
        
        if service.get('tcp_window_scaling_required') and pattern['baseline_latency_ms'] > 15:
            compatibility_assessment['recommendations'].append(
                f"TCP window scaling recommended for {pattern['baseline_latency_ms']}ms latency"
            )
        
        return compatibility_assessment
    
    def estimate_migration_time(self, data_size_gb: int, effective_throughput_mbps: int, migration_service: str) -> Dict:
        """Estimate migration timing with service-specific considerations"""
        data_size_gbits = data_size_gb * 8
        service = self.migration_services[migration_service]
        
        if effective_throughput_mbps > 0:
            migration_time_hours = data_size_gbits / (effective_throughput_mbps / 1000) / 3600
        else:
            migration_time_hours = float('inf')
        
        # Service-specific overhead
        setup_overhead_hours = 2 if migration_service == 'dms' else 1
        validation_overhead_hours = data_size_gb / 1000
        
        total_migration_time = migration_time_hours + setup_overhead_hours + validation_overhead_hours
        
        return {
            'data_transfer_hours': migration_time_hours,
            'setup_hours': setup_overhead_hours,
            'validation_hours': validation_overhead_hours,
            'total_hours': total_migration_time,
            'total_days': total_migration_time / 24,
            'recommended_window_hours': math.ceil(total_migration_time * 1.2),
            'service_name': service['name'],
            'supports_incremental': migration_service in ['dms', 'storage_gateway']
        }
    
    def generate_ai_recommendations(self, config: Dict, analysis_results: Dict) -> Dict:
        """Generate AI-powered recommendations"""
        migration_time = analysis_results['migration_time']
        waterfall_data = analysis_results['waterfall_data']
        service_compatibility = analysis_results.get('service_compatibility', {})
        
        recommendations = []
        priority_score = 0
        
        # Infrastructure bottleneck recommendations
        primary_bottleneck = waterfall_data['summary']['primary_bottleneck_layer']
        if primary_bottleneck == 'nic':
            recommendations.append({
                'type': 'infrastructure_upgrade',
                'priority': 'high',
                'description': 'Network Interface Card is the primary bottleneck. Consider upgrading to higher bandwidth NIC.',
                'impact': 'Significant throughput improvement possible'
            })
            priority_score += 20
        elif primary_bottleneck == 'lan':
            recommendations.append({
                'type': 'network_optimization',
                'priority': 'medium',
                'description': 'LAN infrastructure is limiting performance. Review switch capacity and oversubscription.',
                'impact': 'Moderate throughput improvement'
            })
            priority_score += 15
        
        # Service compatibility warnings
        if service_compatibility.get('warnings'):
            recommendations.append({
                'type': 'service_compatibility',
                'priority': 'critical',
                'description': f'Service compatibility issues detected: {len(service_compatibility["warnings"])} warnings',
                'impact': 'Migration may fail without addressing compatibility'
            })
            priority_score += 30
        
        # Migration timing recommendations
        if migration_time['total_hours'] > 48:
            recommendations.append({
                'type': 'timeline_optimization',
                'priority': 'medium',
                'description': 'Long migration time detected. Consider parallel processing or staged approach.',
                'impact': 'Reduced migration window'
            })
            priority_score += 10
        
        return {
            'recommendations': recommendations,
            'overall_priority_score': priority_score,
            'migration_complexity': 'high' if priority_score > 40 else 'medium' if priority_score > 20 else 'low',
            'confidence_level': 'high' if len(recommendations) <= 3 else 'medium'
        }
    
    # ADD these methods to the EnhancedNetworkAnalyzer class
# Insert around line 800-900, after existing methods but before "# NEW METHODS FOR AI AND COST ANALYSIS"

def calculate_optimal_datasync_instances(self, available_bandwidth_mbps: float, data_size_gb: int, target_transfer_time_hours: float) -> Dict:
    """Calculate optimal number of DataSync instances for production workloads"""
    
    # Get DataSync VM specs from migration services
    datasync_service = self.migration_services['datasync']
    
    # Calculate required throughput
    required_throughput_mbps = (data_size_gb * 8) / (target_transfer_time_hours * 3600) * 1000
    
    recommendations = {}
    
    # Analyze each size configuration
    for size_name, size_config in datasync_service['sizes'].items():
        vmware_overhead = size_config.get('vmware_overhead', 0.1)
        base_throughput = size_config['throughput_mbps']
        effective_throughput = size_config.get('effective_throughput_mbps', 
                                              base_throughput * (1 - vmware_overhead))
        
        # Calculate instances needed
        instances_needed = math.ceil(required_throughput_mbps / effective_throughput)
        
        # Check against available bandwidth
        max_possible_instances = min(instances_needed, 
                                   available_bandwidth_mbps // effective_throughput)
        
        total_throughput = max_possible_instances * effective_throughput
        
        recommendations[size_name] = {
            'instances_needed': instances_needed,
            'instances_possible': max_possible_instances,
            'effective_throughput_per_vm': effective_throughput,
            'total_throughput_mbps': total_throughput,
            'meets_requirement': total_throughput >= required_throughput_mbps,
            'vmware_requirements': {
                'total_cpu_cores': max_possible_instances * size_config['vcpu'],
                'total_memory_gb': max_possible_instances * size_config['memory_gb'],
                'network_bandwidth_mbps': max_possible_instances * base_throughput
            },
            'is_bottleneck': required_throughput_mbps > total_throughput
        }
    
    # Find best recommendation
    viable_options = {k: v for k, v in recommendations.items() if v['meets_requirement']}
    
    if viable_options:
        best_option_key = min(viable_options.keys(), 
                             key=lambda k: viable_options[k]['vmware_requirements']['total_cpu_cores'])
    else:
        best_option_key = max(recommendations.keys(),
                             key=lambda k: recommendations[k]['total_throughput_mbps'])
    
    return {
        'required_throughput_mbps': required_throughput_mbps,
        'available_bandwidth_mbps': available_bandwidth_mbps,
        'recommendations': recommendations,
        'best_option': best_option_key,
        'best_option_details': recommendations[best_option_key],
        'is_datasync_bottleneck': any(rec['is_bottleneck'] for rec in recommendations.values())
    }

def analyze_datasync_bottleneck_by_pattern(self, pattern_key: str, migration_service: str, 
                                          service_size: str, num_instances: int, 
                                          data_size_gb: int, target_hours: float) -> Dict:
    """Analyze if DataSync is the bottleneck for a specific pattern"""
    
    # Calculate network capacity
    waterfall_data = self.calculate_realistic_bandwidth_waterfall(
        pattern_key, migration_service, service_size, num_instances
    )
    
    available_bandwidth = waterfall_data['summary']['final_effective_mbps']
    
    # Only analyze if it's DataSync
    if migration_service == 'datasync':
        datasync_analysis = self.calculate_optimal_datasync_instances(
            available_bandwidth, data_size_gb, target_hours
        )
        
        return {
            'waterfall_data': waterfall_data,
            'datasync_analysis': datasync_analysis,
            'bottleneck_layer': 'service' if datasync_analysis['is_datasync_bottleneck'] else 'network',
            'bottleneck_explanation': self._explain_datasync_bottleneck(datasync_analysis, waterfall_data)
        }
    else:
        return {
            'waterfall_data': waterfall_data,
            'datasync_analysis': None,
            'bottleneck_layer': waterfall_data['summary']['bottleneck'],
            'bottleneck_explanation': f"Service: {migration_service} (not DataSync)"
        }

def _explain_datasync_bottleneck(self, datasync_analysis: Dict, waterfall_data: Dict) -> str:
    """Generate explanation for DataSync bottleneck analysis"""
    
    best_option = datasync_analysis['best_option_details']
    required = datasync_analysis['required_throughput_mbps']
    available_network = datasync_analysis['available_bandwidth_mbps']
    
    if datasync_analysis['is_datasync_bottleneck']:
        if best_option['instances_possible'] < best_option['instances_needed']:
            return (f"DataSync bottleneck: Need {best_option['instances_needed']} instances "
                   f"({required:,.0f} Mbps required) but network only supports "
                   f"{best_option['instances_possible']} instances ({available_network:,.0f} Mbps)")
        else:
            return (f"DataSync bottleneck: Single VM limited to {best_option['effective_throughput_per_vm']:,.0f} Mbps, "
                   f"need {best_option['instances_needed']} instances for {required:,.0f} Mbps requirement")
    else:
        return (f"Network bottleneck: DataSync can provide {best_option['total_throughput_mbps']:,.0f} Mbps "
                f"but network limits to {available_network:,.0f} Mbps")

def get_datasync_scaling_recommendations(self, pattern_key: str, config: Dict) -> Dict:
    """Get DataSync scaling recommendations for the current configuration"""
    
    if config['migration_service'] != 'datasync':
        return {'applicable': False, 'reason': 'Not using DataSync service'}
    
    # Calculate target transfer time
    target_hours = config.get('max_downtime_hours', 8)
    
    analysis = self.analyze_datasync_bottleneck_by_pattern(
        pattern_key, 
        config['migration_service'],
        config['service_size'],
        config['num_instances'],
        config['data_size_gb'],
        target_hours
    )
    
    if analysis['datasync_analysis']:
        datasync_data = analysis['datasync_analysis']
        best_option = datasync_data['best_option_details']
        
        recommendations = []
        
        if datasync_data['is_datasync_bottleneck']:
            recommendations.append({
                'type': 'scaling',
                'priority': 'high',
                'description': f"Scale to {best_option['instances_needed']} DataSync VMs to meet {target_hours}h transfer window",
                'vmware_impact': f"Requires {best_option['vmware_requirements']['total_cpu_cores']} CPU cores, "
                               f"{best_option['vmware_requirements']['total_memory_gb']} GB RAM",
                'cost_impact': f"Estimated {best_option['instances_needed']}x cost increase"
            })
        
        if best_option['instances_needed'] > 1:
            recommendations.append({
                'type': 'architecture',
                'priority': 'medium', 
                'description': "Deploy multiple DataSync VMs across different VMware hosts for better performance",
                'vmware_impact': "Distribute VMs to avoid resource contention",
                'cost_impact': "Minimal additional cost for VM distribution"
            })
        
        return {
            'applicable': True,
            'is_bottleneck': datasync_data['is_datasync_bottleneck'],
            'current_config': {
                'size': config['service_size'],
                'instances': config['num_instances'],
                'effective_throughput': best_option['total_throughput_mbps']
            },
            'optimal_config': {
                'size': datasync_data['best_option'],
                'instances': best_option['instances_needed'],
                'effective_throughput': best_option['total_throughput_mbps']
            },
            'recommendations': recommendations,
            'explanation': analysis['bottleneck_explanation']
        }
    
    return {'applicable': False, 'reason': 'Could not analyze DataSync configuration'}
    
    
    
    # NEW METHODS FOR AI AND COST ANALYSIS
    def analyze_all_patterns(self, config: Dict) -> List[PatternAnalysis]:
        """Analyze all available patterns for comparison"""
        results = []
        
        for pattern_key, pattern in self.network_patterns.items():
            # Skip patterns that don't match source location
            if config['source_location'] not in pattern['name']:
                continue
            
            try:
                analysis = self._analyze_single_pattern(pattern_key, config)
                results.append(analysis)
            except Exception as e:
                st.warning(f"Error analyzing pattern {pattern_key}: {e}")
        
        return sorted(results, key=lambda x: x.ai_recommendation_score, reverse=True)
    
    def _analyze_single_pattern(self, pattern_key: str, config: Dict) -> PatternAnalysis:
        """Analyze a single pattern"""
        pattern = self.network_patterns[pattern_key]
        
        # Calculate bandwidth and costs
        waterfall_data = self.calculate_realistic_bandwidth_waterfall(
            pattern_key, config['migration_service'], config['service_size'], config['num_instances']
        )
        
        total_cost = self._calculate_total_cost(pattern_key, config, waterfall_data)
        migration_time = self._estimate_migration_time(config, waterfall_data)
        
        # Calculate scores
        reliability_score = pattern['reliability_score']
        complexity_score = 1.0 - pattern['complexity_score']  # Invert for scoring
        
        # Database suitability
        db_scenario = config.get('database_scenario', 'mysql_oltp_rds')
        db_workload = self.database_scenarios.get(db_scenario, {}).get('workload_type', 'oltp')
        db_suitability = pattern['database_suitability'].get(db_workload, 0.8)
        
        # AI recommendation score (weighted combination)
        ai_score = (
            reliability_score * 0.3 +
            complexity_score * 0.2 +
            db_suitability * 0.3 +
            (1.0 - min(total_cost / 10000, 1.0)) * 0.2  # Cost factor (normalized)
        )
        
        # Generate use cases and pros/cons
        use_cases, pros, cons = self._generate_pattern_details(pattern, config)
        
        return PatternAnalysis(
            pattern_name=pattern['name'],
            total_cost_usd=total_cost,
            migration_time_hours=migration_time,
            effective_bandwidth_mbps=waterfall_data['summary']['final_effective_mbps'],
            reliability_score=reliability_score,
            complexity_score=pattern['complexity_score'],
            ai_recommendation_score=ai_score,
            use_cases=use_cases,
            pros=pros,
            cons=cons
        )
    
    def _calculate_total_cost(self, pattern_key: str, config: Dict, waterfall_data: Dict) -> float:
        """Calculate total migration cost"""
        pattern = self.network_patterns[pattern_key]
        service_config = self.migration_services[config['migration_service']]['sizes'][config['service_size']]
        
        # Infrastructure costs
        infrastructure_cost = 0
        
        # Direct Connect costs
        if pattern['pattern_type'] == 'direct_connect':
            dx_speed = '10Gbps'  # Default
            if pattern['committed_bandwidth_mbps'] >= 50000:
                dx_speed = '100Gbps'
            elif pattern['committed_bandwidth_mbps'] >= 5000:
                dx_speed = '10Gbps'
            else:
                dx_speed = '1Gbps'
            
            dx_pricing = self.pricing_client.get_direct_connect_pricing(dx_speed)
            migration_hours = self._estimate_migration_time(config, waterfall_data)
            infrastructure_cost += dx_pricing['port_hour'] * migration_hours
            
            # Data transfer costs
            data_transfer_gb = config['data_size_gb']
            infrastructure_cost += dx_pricing['data_transfer_gb'] * data_transfer_gb
        
        # Service instance costs
        instance_type = service_config.get('instance_type', 'm5.xlarge')
        instance_pricing = self.pricing_client.get_ec2_pricing(instance_type)
        migration_hours = self._estimate_migration_time(config, waterfall_data)
        service_cost = instance_pricing['hourly'] * migration_hours * config['num_instances']
        
        # Add operational overhead
        operational_overhead = (infrastructure_cost + service_cost) * 0.2
        
        return infrastructure_cost + service_cost + operational_overhead
    
    def _estimate_migration_time(self, config: Dict, waterfall_data: Dict) -> float:
        """Estimate migration time in hours"""
        effective_bandwidth_mbps = waterfall_data['summary']['final_effective_mbps']
        data_size_gb = config['data_size_gb']
        
        if effective_bandwidth_mbps > 0:
            data_transfer_hours = (data_size_gb * 8) / (effective_bandwidth_mbps / 1000) / 3600
        else:
            data_transfer_hours = float('inf')
        
        # Add setup and validation time
        setup_hours = 4 if config['migration_service'] == 'dms' else 2
        validation_hours = max(2, data_size_gb / 1000)
        
        return data_transfer_hours + setup_hours + validation_hours
    
    def _generate_pattern_details(self, pattern: Dict, config: Dict) -> Tuple[List[str], List[str], List[str]]:
        """Generate use cases, pros, and cons for a pattern"""
        pattern_type = pattern['pattern_type']
        
        if pattern_type == 'vpc_endpoint':
            use_cases = [
                "Small to medium database migrations",
                "Non-production environments",
                "Cost-sensitive migrations",
                "Services with VPC endpoint support"
            ]
            pros = [
                "Lower cost structure",
                "No Direct Connect setup required",
                "Good for compatible services",
                "AWS managed connectivity"
            ]
            cons = [
                "Higher latency than Direct Connect",
                "Limited service compatibility",
                "Shared bandwidth",
                "Internet routing dependencies"
            ]
        elif pattern_type == 'direct_connect':
            use_cases = [
                "Production database migrations",
                "Large data volumes (>500GB)",
                "Latency-sensitive applications",
                "Enterprise-grade connectivity needs"
            ]
            pros = [
                "Dedicated bandwidth",
                "Lower, consistent latency",
                "Higher reliability (99.9% SLA)",
                "Better security posture"
            ]
            cons = [
                "Higher setup costs",
                "Longer provisioning time",
                "Requires network expertise",
                "Monthly recurring costs"
            ]
        else:  # multi_hop
            use_cases = [
                "Remote location connectivity",
                "Complex network topologies",
                "Phased migration approaches",
                "Disaster recovery scenarios"
            ]
            pros = [
                "Can reach remote locations",
                "Flexible routing options",
                "Can leverage existing infrastructure",
                "Supports complex scenarios"
            ]
            cons = [
                "Highest complexity",
                "Multiple failure points",
                "Variable performance",
                "Difficult troubleshooting"
            ]
        
        return use_cases, pros, cons
    
    def get_ai_recommendation(self, pattern_analyses: List[PatternAnalysis], config: Dict) -> Dict:
        """Get AI recommendation for best pattern"""
        analysis_data = {
            'data_size_gb': config['data_size_gb'],
            'migration_service': config['migration_service'],
            'environment': config['environment'],
            'max_downtime_hours': config['max_downtime_hours'],
            'source_location': config['source_location'],
            'database_scenario': config.get('database_scenario', 'mysql_oltp_rds'),
            'patterns': [
                {
                    'name': p.pattern_name,
                    'cost': p.total_cost_usd,
                    'time': p.migration_time_hours,
                    'bandwidth': p.effective_bandwidth_mbps,
                    'reliability': p.reliability_score
                }
                for p in pattern_analyses
            ]
        }
        
        return self.ai_client.get_pattern_recommendation(analysis_data)

# =============================================================================
# HEADER AND UTILITY FUNCTIONS
# =============================================================================

def render_corporate_header():
    """Render enhanced corporate header"""
    st.markdown("""
    <div class="corporate-header">
        <h1>🏢 AWS Migration Network Analyzer</h1>
        <div class="subtitle">Enterprise Infrastructure Analysis Platform</div>
        <div class="tagline">Real-time AWS Pricing • AI-Powered Recommendations • Database-Optimized Migration Patterns</div>
    </div>
    """, unsafe_allow_html=True)

def get_api_credentials():
    """Get API credentials from Streamlit secrets"""
    try:
        aws_access_key = st.secrets.get("aws_access_key", "")
        aws_secret_key = st.secrets.get("aws_secret_key", "")
        claude_api_key = st.secrets.get("claude_api_key", "")
        
        # Show status in sidebar
        st.sidebar.markdown("### 🔐 API Status")
        if aws_access_key and aws_secret_key:
            st.sidebar.success("✅ AWS Credentials: Connected")
        else:
            st.sidebar.info("ℹ️ AWS Credentials: Using Mock Data")
        
        if claude_api_key:
            st.sidebar.success("✅ Claude AI: Connected")
        else:
            st.sidebar.info("ℹ️ Claude AI: Using Mock Responses")
        
        return {
            'aws_access_key': aws_access_key,
            'aws_secret_key': aws_secret_key,
            'claude_api_key': claude_api_key
        }
    
    except Exception as e:
        st.sidebar.warning(f"Error accessing secrets: {e}")
        return {
            'aws_access_key': "",
            'aws_secret_key': "",
            'claude_api_key': ""
        }

# =============================================================================
# SIDEBAR CONTROLS
# =============================================================================

def render_enhanced_sidebar_controls():
    """Render enhanced sidebar controls"""
    st.sidebar.header("🔧 Migration Configuration")
    
    # Get API credentials from secrets
    api_credentials = get_api_credentials()
    
    # Database Scenario Selection
    st.sidebar.subheader("🗄️ Database Scenario")
    
    # Get analyzer instance to access database scenarios
    analyzer = EnhancedNetworkAnalyzer()
    database_scenarios = list(analyzer.database_scenarios.keys())
    
    database_scenario = st.sidebar.selectbox(
        "Database Type & AWS Target",
        database_scenarios,
        format_func=lambda x: analyzer.database_scenarios[x]['name'],
        help="Select your database type and target AWS service"
    )
    
    # Display scenario details
    selected_scenario = analyzer.database_scenarios[database_scenario]
    
    st.sidebar.markdown(f"""
    **Target Service:** {selected_scenario['target_service']}  
    **Migration Complexity:** {selected_scenario['migration_complexity'].title()}  
    **Workload Type:** {selected_scenario['workload_type'].upper()}
    """)
    
    # Source Environment
    st.sidebar.subheader("📍 Source Environment")
    source_location = st.sidebar.selectbox(
        "Data Center Location",
        ["San Jose", "San Antonio"],
        help="Select source data center location"
    )
    
    environment = st.sidebar.selectbox(
        "Environment Type",
        ["non-production", "production"],
        help="Production environments require higher reliability"
    )
    
    # Infrastructure overrides
    with st.sidebar.expander("🏗️ Advanced Infrastructure Settings", expanded=False):
        os_type = st.sidebar.selectbox(
            "Operating System",
            list(analyzer.os_characteristics.keys()),
            format_func=lambda x: analyzer.os_characteristics[x]['name'],
            help="Override default OS selection"
        )
        
        nic_type = st.sidebar.selectbox(
            "Network Interface Card",
            list(analyzer.nic_characteristics.keys()),
            format_func=lambda x: analyzer.nic_characteristics[x]['name'],
            help="Override default NIC selection"
        )
    
    # Migration Service
    st.sidebar.subheader("🚀 Migration Service")
    
    # Filter migration services based on database scenario recommendations
    recommended_services = selected_scenario.get('recommended_services', ['datasync', 'dms'])
    all_services = ["datasync", "dms", "fsx_windows", "fsx_lustre", "storage_gateway"]
    
    migration_service = st.sidebar.selectbox(
        "AWS Migration Service",
        all_services,
        index=all_services.index(recommended_services[0]) if recommended_services[0] in all_services else 0,
        format_func=lambda x: {
            'datasync': '📁 AWS DataSync',
            'dms': '🗄️ AWS DMS (Recommended)',
            'fsx_windows': '🪟 FSx for Windows',
            'fsx_lustre': '⚡ FSx for Lustre',
            'storage_gateway': '🔗 Storage Gateway'
        }[x],
        help="Select AWS migration service"
    )
    
    if migration_service not in recommended_services:
        st.sidebar.warning(f"⚠️ {migration_service.upper()} not recommended for {selected_scenario['name']}")
    
    service_info = analyzer.migration_services[migration_service]
    service_sizes = list(service_info['sizes'].keys())
    
    service_size = st.sidebar.selectbox(
        f"{service_info['name']} Size",
        service_sizes,
        index=1 if len(service_sizes) > 1 else 0,
        format_func=lambda x: f"{x.title()} - {service_info['sizes'][x].get('throughput_mbps', 'Variable')} {'Mbps' if 'throughput_mbps' in service_info['sizes'][x] else ''}",
        help="Service instance configuration"
    )
    
    # Number of instances
    if migration_service in ['fsx_windows', 'fsx_lustre']:
        num_instances = 1
        st.sidebar.info("FSx services are managed - single instance")
    else:
        num_instances = st.sidebar.number_input(
            "Number of Instances",
            min_value=1,
            max_value=8,
            value=2,
            help="Number of parallel instances"
        )
    
    # Data Configuration
    st.sidebar.subheader("💾 Data Configuration")
    data_size_gb = st.sidebar.number_input(
        "Database Size (GB)",
        min_value=10,
        max_value=50000,
        value=1000,
        step=100,
        help="Total database size to migrate"
    )
    
    max_downtime_hours = st.sidebar.number_input(
        "Maximum Downtime (hours)",
        min_value=1,
        max_value=168,
        value=8,
        help="Maximum acceptable downtime for migration"
    )
    
    # Show scenario requirements
    st.sidebar.markdown("### 📋 Scenario Requirements")
    st.sidebar.markdown(f"""
    **Min Bandwidth:** {selected_scenario['min_bandwidth_mbps']} Mbps  
    **Max Latency:** {selected_scenario['max_tolerable_latency_ms']}ms  
    **Downtime Sensitivity:** {selected_scenario['downtime_sensitivity'].title()}
    """)
    
    # Combine with API credentials
    config = {
        'database_scenario': database_scenario,
        'source_location': source_location,
        'environment': environment,
        'os_type': os_type if 'os_type' in locals() else None,
        'nic_type': nic_type if 'nic_type' in locals() else None,
        'migration_service': migration_service,
        'service_size': service_size,
        'num_instances': num_instances,
        'data_size_gb': data_size_gb,
        'max_downtime_hours': max_downtime_hours
    }
    
    # Add API credentials
    config.update(api_credentials)
    
    return config

# =============================================================================
# CHART CREATION FUNCTIONS
# =============================================================================

def create_enhanced_waterfall_chart(waterfall_data: Dict):
    """Create enhanced sequential waterfall chart showing proper infrastructure flow"""
    steps = waterfall_data['steps']
    
    # Layer colors for corporate theme
    layer_colors = {
        'nic': '#1e3a8a',      # Primary blue
        'os': '#3b82f6',       # Secondary blue  
        'lan': '#059669',      # Success green
        'wan': '#d97706',      # Warning orange
        'dx': '#dc2626',       # Error red
        'vpc': '#7c3aed',      # Purple
        'protocol': '#0891b2', # Cyan
        'service': '#16a34a',  # Dark green
        'final': '#1e40af'     # Dark blue
    }
    
    # Prepare data for waterfall chart
    x_labels = []
    y_values = []
    colors = []
    text_labels = []
    
    # Starting point
    starting_step = next(step for step in steps if step['type'] == 'starting')
    x_labels.append(f"1. {starting_step['name']}")
    y_values.append(starting_step['value'])
    colors.append(layer_colors.get(starting_step['layer'], '#6b7280'))
    text_labels.append(f"{starting_step['value']:,.0f} Mbps")
    
    # Running total for waterfall effect
    running_total = starting_step['value']
    
    # Reduction steps
    reduction_steps = [step for step in steps if step['type'] == 'reduction']
    for step in reduction_steps:
        step_num = step.get('step_number', 0)
        x_labels.append(f"{step_num}. {step['name']}")
        y_values.append(abs(step['value']))  # Use absolute value for bar height
        colors.append(layer_colors.get(step['layer'], '#6b7280'))
        text_labels.append(f"-{abs(step['value']):,.0f} Mbps\n({step['cumulative']:,.0f} remaining)")
        running_total += step['value']  # step['value'] is negative
    
    # Final result
    final_step = next(step for step in steps if step['type'] == 'final')
    step_num = final_step.get('step_number', len(steps))
    x_labels.append(f"{step_num}. {final_step['name']}")
    y_values.append(final_step['value'])
    colors.append(layer_colors.get(final_step['layer'], '#1e40af'))
    text_labels.append(f"{final_step['value']:,.0f} Mbps")
    
    # Create waterfall chart
    fig = go.Figure()
    
    # Add bars
    for i, (x, y, color, text) in enumerate(zip(x_labels, y_values, colors, text_labels)):
        # Determine if this is a reduction step
        is_reduction = i > 0 and i < len(x_labels) - 1
        
        fig.add_trace(go.Bar(
            x=[x],
            y=[y],
            marker_color=color,
            text=[text],
            textposition='outside' if not is_reduction else 'inside',
            textfont=dict(color='white' if is_reduction else 'black', size=10),
            name=x,
            hovertemplate=f"<b>{x}</b><br>Impact: {y:,.0f} Mbps<extra></extra>",
            opacity=0.8 if is_reduction else 1.0
        ))
    
    # Add connecting lines to show flow
    for i in range(len(x_labels) - 1):
        if i == 0:
            continue  # Skip first connection
        
        # Calculate positions for connecting lines
        start_y = sum(y_values[:i+1]) if i < len(x_labels) - 1 else y_values[i]
        end_y = sum(y_values[:i+2]) if i+1 < len(x_labels) - 1 else y_values[i+1]
        
        fig.add_shape(
            type="line",
            x0=i, y0=start_y,
            x1=i+1, y1=end_y,
            line=dict(color="rgba(128,128,128,0.5)", width=1, dash="dot")
        )
    
    fig.update_layout(
        title={
            'text': "Infrastructure Impact Waterfall Analysis",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#1e3a8a'}
        },
        xaxis_title="Infrastructure Components",
        yaxis_title="Bandwidth (Mbps)",
        showlegend=False,
        height=600,
        xaxis=dict(
            tickangle=45,
            tickfont=dict(size=10)
        ),
        yaxis=dict(
            tickformat=",.0f"
        ),
        template="plotly_white",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=80, b=120, l=80, r=80)
    )
    
    return fig

def create_sequential_flow_diagram(waterfall_data: Dict):
    """Create a sequential bar chart showing the bandwidth reduction flow - NO TEXT, ONLY CHARTS"""
    steps = waterfall_data['steps']
    
    st.markdown("### 📊 Sequential Infrastructure Flow")
    
    # Filter and prepare steps for the chart
    flow_steps = [step for step in steps if step['type'] in ['starting', 'reduction', 'final']]
    
    # Prepare data for the sequential flow chart
    x_labels = []
    y_values = []
    colors = []
    text_labels = []
    cumulative_values = []
    
    # Layer colors for corporate theme
    layer_colors = {
        'nic': '#1e3a8a',      # Primary blue
        'os': '#3b82f6',       # Secondary blue  
        'lan': '#059669',      # Success green
        'wan': '#d97706',      # Warning orange
        'dx': '#dc2626',       # Error red
        'vpc': '#7c3aed',      # Purple
        'protocol': '#0891b2', # Cyan
        'service': '#16a34a',  # Dark green
        'final': '#1e40af'     # Dark blue
    }
    
    # Process each step for the chart
    for i, step in enumerate(flow_steps):
        step_num = step.get('step_number', i + 1)
        # Simplify the name to avoid formatting issues
        short_name = step['name'][:30] + ('...' if len(step['name']) > 30 else '')
        x_labels.append(f"Step {step_num}: {short_name}")
        cumulative_values.append(step['cumulative'])
        
        if step['type'] == 'starting':
            y_values.append(step['value'])
            colors.append('#059669')  # Success green
            text_labels.append(f"{step['value']:,.0f} Mbps (Start)")
            
        elif step['type'] == 'reduction':
            y_values.append(step['value'])  # Negative value
            colors.append(layer_colors.get(step['layer'], '#dc2626'))
            reduction_pct = (abs(step['value']) / flow_steps[0]['value']) * 100 if flow_steps[0]['value'] > 0 else 0
            text_labels.append(f"{step['value']:,.0f} Mbps ({reduction_pct:.1f}% loss)")
            
        elif step['type'] == 'final':
            y_values.append(step['value'])
            colors.append('#1e40af')  # Final blue
            efficiency = (step['value'] / flow_steps[0]['value']) * 100 if flow_steps[0]['value'] > 0 else 0
            text_labels.append(f"{step['value']:,.0f} Mbps ({efficiency:.1f}% eff.)")
    
    # Create the figure
    fig = go.Figure()
    
    # Add the main bars
    fig.add_trace(go.Bar(
        x=x_labels,
        y=y_values,
        marker_color=colors,
        text=text_labels,
        textposition='outside',
        textfont=dict(size=10),
        name='Bandwidth Impact',
        hovertemplate="<b>%{x}</b><br>Impact: %{y:,.0f} Mbps<extra></extra>",
        opacity=0.85
    ))
    
    # Add cumulative bandwidth line
    fig.add_trace(go.Scatter(
        x=x_labels,
        y=cumulative_values,
        mode='lines+markers',
        line=dict(color='#1e3a8a', width=4),
        marker=dict(size=10, color='#1e3a8a'),
        name='Cumulative Bandwidth',
        yaxis='y2',
        hovertemplate="<b>Cumulative</b><br>%{y:,.0f} Mbps<extra></extra>"
    ))
    
    # Simplified layout configuration
    fig.update_layout(
        title="Infrastructure Impact Sequential Analysis",
        xaxis_title="Infrastructure Processing Steps",
        yaxis_title="Bandwidth Impact (Mbps)",
        yaxis2=dict(
            title="Cumulative Bandwidth (Mbps)",
            overlaying='y',
            side='right'
        ),
        showlegend=True,
        height=600,
        xaxis=dict(
            tickangle=45,
            tickfont=dict(size=9)
        ),
        yaxis=dict(
            tickformat=",.0f",
            zeroline=True,
            zerolinecolor='rgba(128,128,128,0.5)',
            zerolinewidth=2
        ),
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_network_topology_diagram():
    """Create a comprehensive network topology diagram for production and non-production environments"""
    
    # Create subplots for both environments
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Production Environment: San Antonio → San Jose → AWS West 2", 
                        "Non-Production Environment: San Jose → AWS West 2"),
        vertical_spacing=0.15,
        specs=[[{"type": "scatter"}], [{"type": "scatter"}]]
    )
    
    # Production Environment Nodes
    prod_nodes = {
        'San Antonio DC': {'x': 1, 'y': 2, 'color': '#dc2626', 'size': 25},
        'San Jose DC': {'x': 3, 'y': 2, 'color': '#d97706', 'size': 25},
        'AWS West 2': {'x': 5, 'y': 2, 'color': '#059669', 'size': 30},
        'Production VPC': {'x': 5, 'y': 1.5, 'color': '#1e40af', 'size': 20},
        'DataSync (SA)': {'x': 1, 'y': 1.5, 'color': '#7c3aed', 'size': 15},
        'DataSync (SJ)': {'x': 3, 'y': 1.5, 'color': '#7c3aed', 'size': 15},
    }
    
    # Non-Production Environment Nodes
    nonprod_nodes = {
        'San Jose DC': {'x': 2, 'y': 1, 'color': '#d97706', 'size': 25},
        'AWS West 2': {'x': 4, 'y': 1, 'color': '#059669', 'size': 30},
        'Non-Prod VPC': {'x': 4, 'y': 0.5, 'color': '#3b82f6', 'size': 20},
        'DataSync (SJ)': {'x': 2, 'y': 0.5, 'color': '#7c3aed', 'size': 15},
    }
    
    # Add Production Environment nodes
    for name, props in prod_nodes.items():
        fig.add_trace(go.Scatter(
            x=[props['x']], y=[props['y']],
            mode='markers+text',
            marker=dict(size=props['size'], color=props['color']),
            text=[name],
            textposition="top center",
            name=name,
            showlegend=False,
            hovertemplate=f"<b>{name}</b><br>Environment: Production<extra></extra>"
        ), row=1, col=1)
    
    # Add Non-Production Environment nodes
    for name, props in nonprod_nodes.items():
        fig.add_trace(go.Scatter(
            x=[props['x']], y=[props['y']],
            mode='markers+text',
            marker=dict(size=props['size'], color=props['color']),
            text=[name],
            textposition="top center",
            name=name,
            showlegend=False,
            hovertemplate=f"<b>{name}</b><br>Environment: Non-Production<extra></extra>"
        ), row=2, col=1)
    
    # Production Environment Connections
    prod_connections = [
        {'from': 'San Antonio DC', 'to': 'San Jose DC', 'label': '10Gbps Shared Link', 'color': '#dc2626'},
        {'from': 'San Jose DC', 'to': 'AWS West 2', 'label': '10Gbps DX Link', 'color': '#059669'},
        {'from': 'AWS West 2', 'to': 'Production VPC', 'label': 'VPC Connection', 'color': '#1e40af'},
    ]
    
    # Non-Production Environment Connections
    nonprod_connections = [
        {'from': 'San Jose DC', 'to': 'AWS West 2', 'label': '2Gbps DX Link', 'color': '#d97706'},
        {'from': 'AWS West 2', 'to': 'Non-Prod VPC', 'label': 'VPC Connection', 'color': '#3b82f6'},
    ]
    
    # Add production connections
    for conn in prod_connections:
        from_node = prod_nodes[conn['from']]
        to_node = prod_nodes[conn['to']]
        
        fig.add_trace(go.Scatter(
            x=[from_node['x'], to_node['x']],
            y=[from_node['y'], to_node['y']],
            mode='lines',
            line=dict(color=conn['color'], width=4),
            name=conn['label'],
            showlegend=False,
            hovertemplate=f"<b>{conn['label']}</b><extra></extra>"
        ), row=1, col=1)
        
        # Add connection label
        mid_x = (from_node['x'] + to_node['x']) / 2
        mid_y = (from_node['y'] + to_node['y']) / 2 + 0.1
        
        fig.add_trace(go.Scatter(
            x=[mid_x], y=[mid_y],
            mode='text',
            text=[conn['label']],
            textfont=dict(size=10, color=conn['color']),
            showlegend=False,
            hoverinfo='skip'
        ), row=1, col=1)
    
    # Add non-production connections
    for conn in nonprod_connections:
        from_node = nonprod_nodes[conn['from']]
        to_node = nonprod_nodes[conn['to']]
        
        fig.add_trace(go.Scatter(
            x=[from_node['x'], to_node['x']],
            y=[from_node['y'], to_node['y']],
            mode='lines',
            line=dict(color=conn['color'], width=4),
            name=conn['label'],
            showlegend=False,
            hovertemplate=f"<b>{conn['label']}</b><extra></extra>"
        ), row=2, col=1)
        
        # Add connection label
        mid_x = (from_node['x'] + to_node['x']) / 2
        mid_y = (from_node['y'] + to_node['y']) / 2 + 0.1
        
        fig.add_trace(go.Scatter(
            x=[mid_x], y=[mid_y],
            mode='text',
            text=[conn['label']],
            textfont=dict(size=10, color=conn['color']),
            showlegend=False,
            hoverinfo='skip'
        ), row=2, col=1)
    
    # Update layout
    fig.update_layout(
        title={
            'text': "Enterprise Network Architecture: Production & Non-Production Environments",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#1e3a8a'}
        },
        height=800,
        template="plotly_white",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=100, b=50, l=50, r=50)
    )
    
    # Update axes
    for i in range(1, 3):
        fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False, row=i, col=1)
        fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False, row=i, col=1)
    
    return fig

def create_service_flows_diagram():
    """Create detailed service flow diagrams"""
    
    # Service flow data
    services = ['VPC Endpoint', 'FSx', 'DataSync', 'DMS', 'Storage Gateway']
    
    flow_data = {
        'Production': {
            'VPC Endpoint': {'source': 'San Antonio/San Jose', 'target': 'S3', 'path': 'Direct to S3', 'bandwidth': '10Gbps'},
            'FSx': {'source': 'San Antonio/San Jose', 'target': 'S3', 'path': 'Via Production VPC', 'bandwidth': '10Gbps'},
            'DataSync': {'source': 'VMware DC', 'target': 'S3', 'path': 'Direct to S3', 'bandwidth': '10Gbps'},
            'DMS': {'source': 'Database', 'target': 'S3', 'path': 'Via Production VPC', 'bandwidth': '10Gbps'},
            'Storage Gateway': {'source': 'Production VPC', 'target': 'S3', 'path': 'VPC to S3', 'bandwidth': '10Gbps'}
        },
        'Non-Production': {
            'VPC Endpoint': {'source': 'San Jose', 'target': 'S3', 'path': 'Direct to S3', 'bandwidth': '2Gbps'},
            'FSx': {'source': 'San Jose', 'target': 'S3', 'path': 'Via Non-Prod VPC', 'bandwidth': '2Gbps'},
            'DataSync': {'source': 'VMware DC', 'target': 'S3', 'path': 'Direct to S3', 'bandwidth': '2Gbps'},
            'DMS': {'source': 'Database', 'target': 'S3', 'path': 'Via Non-Prod VPC', 'bandwidth': '2Gbps'},
            'Storage Gateway': {'source': 'Non-Prod VPC', 'target': 'S3', 'path': 'VPC to S3', 'bandwidth': '2Gbps'}
        }
    }
    
    # Create flow diagram
    fig = go.Figure()
    
    # Service positioning
    service_positions = {
        'VPC Endpoint': {'x': 1, 'y': 5, 'color': '#1e40af'},
        'FSx': {'x': 2, 'y': 4, 'color': '#059669'},
        'DataSync': {'x': 1.5, 'y': 3, 'color': '#7c3aed'},
        'DMS': {'x': 2, 'y': 2, 'color': '#dc2626'},
        'Storage Gateway': {'x': 1, 'y': 1, 'color': '#d97706'}
    }
    
    # S3 target
    s3_pos = {'x': 4, 'y': 3, 'color': '#16a34a'}
    
    # Add service nodes
    for service, pos in service_positions.items():
        fig.add_trace(go.Scatter(
            x=[pos['x']], y=[pos['y']],
            mode='markers+text',
            marker=dict(size=30, color=pos['color']),
            text=[service],
            textposition="middle center",
            name=service,
            showlegend=False,
            hovertemplate=f"<b>{service}</b><br>AWS Migration Service<extra></extra>"
        ))
    
    # Add S3 target
    fig.add_trace(go.Scatter(
        x=[s3_pos['x']], y=[s3_pos['y']],
        mode='markers+text',
        marker=dict(size=40, color=s3_pos['color']),
        text=['Amazon S3'],
        textposition="middle center",
        name='S3',
        showlegend=False,
        hovertemplate="<b>Amazon S3</b><br>Target Storage Service<extra></extra>"
    ))
    
    # Add flow arrows
    for service, pos in service_positions.items():
        fig.add_annotation(
            x=s3_pos['x'], y=s3_pos['y'],
            ax=pos['x'], ay=pos['y'],
            xref="x", yref="y",
            axref="x", ayref="y",
            arrowhead=2,
            arrowsize=1,
            arrowwidth=3,
            arrowcolor=pos['color'],
            text=""
        )
    
    fig.update_layout(
        title={
            'text': "AWS Migration Services → S3 Data Flow Patterns",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#1e3a8a'}
        },
        height=600,
        template="plotly_white",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False)
    )
    
    return fig

def create_service_comparison_table():
    """Create service comparison table"""
    
    service_data = [
        {
            'Service': 'VPC Endpoint',
            'Production Path': 'San Antonio → San Jose → AWS (10Gbps)',
            'Non-Prod Path': 'San Jose → AWS (2Gbps)',
            'Target': 'S3',
            'Location': 'AWS Managed',
            'Use Case': 'Private connectivity to S3'
        },
        {
            'Service': 'FSx',
            'Production Path': 'San Antonio → San Jose → Production VPC',
            'Non-Prod Path': 'San Jose → Non-Prod VPC',
            'Target': 'S3',
            'Location': 'Production/Non-Prod VPC',
            'Use Case': 'High-performance file systems'
        },
        {
            'Service': 'DataSync',
            'Production Path': 'VMware DC (SA/SJ) → S3',
            'Non-Prod Path': 'VMware DC (SJ) → S3',
            'Target': 'S3',
            'Location': 'On-premises VMware',
            'Use Case': 'One-time and scheduled data transfer'
        },
        {
            'Service': 'DMS',
            'Production Path': 'Database → Production VPC → S3',
            'Non-Prod Path': 'Database → Non-Prod VPC → S3',
            'Target': 'S3',
            'Location': 'Production/Non-Prod VPC',
            'Use Case': 'Database migration and replication'
        },
        {
            'Service': 'Storage Gateway',
            'Production Path': 'Production VPC → S3',
            'Non-Prod Path': 'Non-Prod VPC → S3',
            'Target': 'S3',
            'Location': 'Production/Non-Prod VPC',
            'Use Case': 'Hybrid cloud storage integration'
        }
    ]
    
    return pd.DataFrame(service_data)

# =============================================================================
# TAB RENDERING FUNCTIONS
# =============================================================================

def render_realistic_analysis_tab(config: Dict, analyzer: EnhancedNetworkAnalyzer):
    """Render realistic bandwidth analysis tab with enhanced corporate styling"""
    st.subheader("💧 Infrastructure Impact Analysis")
    
    # Show database scenario info
    db_scenario = analyzer.database_scenarios[config['database_scenario']]
    
    st.markdown(f"""
    <div class="corporate-card status-card-info">
        <h3>🗄️ Database Migration Scenario</h3>
        <p><strong>Source Database:</strong> {db_scenario['name']}</p>
        <p><strong>Target Service:</strong> {db_scenario['target_service']}</p>
        <p><strong>Workload Type:</strong> {db_scenario['workload_type'].upper()}</p>
        <p><strong>Migration Complexity:</strong> {db_scenario['migration_complexity'].title()}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Determine pattern
    pattern_key = analyzer.determine_optimal_pattern(
        config['source_location'], 
        config['environment'], 
        config['migration_service']
    )
    
    # Override infrastructure if specified
    if config.get('os_type') or config.get('nic_type'):
        pattern = analyzer.network_patterns[pattern_key].copy()
        if config.get('os_type'):
            pattern['os_type'] = config['os_type']
        if config.get('nic_type'):
            pattern['nic_type'] = config['nic_type']
        analyzer.network_patterns[pattern_key] = pattern
    
    # Calculate realistic waterfall
    waterfall_data = analyzer.calculate_realistic_bandwidth_waterfall(
        pattern_key,
        config['migration_service'],
        config['service_size'],
        config['num_instances']
    )
    
    summary = waterfall_data['summary']
    
    # Summary metrics in corporate style
    st.markdown("""
    <div class="metric-grid">
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">%s</div>
            <div class="metric-label">Theoretical Max (Mbps)</div>
        </div>
        """ % f"{summary['theoretical_max_mbps']:,.0f}", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">%s</div>
            <div class="metric-label">Final Effective (Mbps)</div>
        </div>
        """ % f"{summary['final_effective_mbps']:,.0f}", unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">%s%%</div>
            <div class="metric-label">Efficiency</div>
        </div>
        """ % f"{summary['efficiency_percentage']:.1f}", unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">%s</div>
            <div class="metric-label">Primary Bottleneck</div>
        </div>
        """ % summary['primary_bottleneck_layer'].title(), unsafe_allow_html=True)
    
    with col5:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">%s%%</div>
            <div class="metric-label">Network Utilization</div>
        </div>
        """ % f"{summary['network_utilization_percent']:.1f}", unsafe_allow_html=True)
    
    with col6:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">%s%%</div>
            <div class="metric-label">Service Utilization</div>
        </div>
        """ % f"{summary['service_utilization_percent']:.1f}", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Check if requirements are met
    meets_bandwidth = summary['final_effective_mbps'] >= db_scenario['min_bandwidth_mbps']
    meets_latency = summary['baseline_latency_ms'] <= db_scenario['max_tolerable_latency_ms']
    
    if not meets_bandwidth or not meets_latency:
        st.markdown(f"""
        <div class="corporate-card status-card-warning">
            <h3>⚠️ Database Requirements Check</h3>
            <p><strong>Bandwidth Requirement:</strong> {'✅ Met' if meets_bandwidth else '❌ Not Met'} 
               ({summary['final_effective_mbps']:,.0f} Mbps vs {db_scenario['min_bandwidth_mbps']} Mbps required)</p>
            <p><strong>Latency Requirement:</strong> {'✅ Met' if meets_latency else '❌ Not Met'} 
               ({summary['baseline_latency_ms']}ms vs {db_scenario['max_tolerable_latency_ms']}ms required)</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="corporate-card status-card-success">
            <h3>✅ Database Requirements Satisfied</h3>
            <p><strong>Bandwidth:</strong> {summary['final_effective_mbps']:,.0f} Mbps (Required: {db_scenario['min_bandwidth_mbps']} Mbps)</p>
            <p><strong>Latency:</strong> {summary['baseline_latency_ms']}ms (Max: {db_scenario['max_tolerable_latency_ms']}ms)</p>
        </div>
        """, unsafe_allow_html=True)
    
        # ADD this section in render_realistic_analysis_tab() 
    # Insert after the database requirements check (around line 1400):

    # Enhanced DataSync Analysis (ADD THIS SECTION)
    if config['migration_service'] == 'datasync':
        datasync_scaling = analyzer.get_datasync_scaling_recommendations(pattern_key, config)
        
        if datasync_scaling['applicable']:
            # DataSync-specific analysis card
            card_type = "status-card-warning" if datasync_scaling['is_bottleneck'] else "status-card-success"
            
            current_config = datasync_scaling['current_config']
            optimal_config = datasync_scaling['optimal_config']
            
            scaling_content = f"""
            <h3>🚀 DataSync Scaling Analysis</h3>
            <p><strong>Current Configuration:</strong> {current_config['instances']} × {config['service_size']} VM(s) = {current_config['effective_throughput']:,.0f} Mbps</p>
            <p><strong>Optimal Configuration:</strong> {optimal_config['instances']} × {optimal_config['size']} VM(s) = {optimal_config['effective_throughput']:,.0f} Mbps</p>
            <p><strong>Analysis:</strong> {datasync_scaling['explanation']}</p>
            """
            
            if datasync_scaling['recommendations']:
                scaling_content += "<h4>💡 Scaling Recommendations:</h4>"
                for rec in datasync_scaling['recommendations']:
                    priority_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(rec['priority'], "⚪")
                    scaling_content += f"""
                    <p>{priority_icon} <strong>{rec['type'].title()}:</strong> {rec['description']}</p>
                    <p style="margin-left: 1rem; font-size: 0.9em; color: var(--medium-gray);">
                    VMware Impact: {rec['vmware_impact']}<br>
                    Cost Impact: {rec['cost_impact']}
                    </p>
                    """
            
            st.markdown(f"""
            <div class="corporate-card {card_type}">
                {scaling_content}
            </div>
            """, unsafe_allow_html=True)

    # ADD this DataSync VMware deployment information section as well:
    if config['migration_service'] == 'datasync':
        with st.expander("🖥️ VMware Deployment Details", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **📋 VMware Requirements per DataSync VM:**
                - **vCPU:** 4-32 cores (depending on size)
                - **Memory:** 16-128 GB RAM 
                - **Storage:** 80 GB minimum
                - **Network:** Dedicated vSwitch recommended
                - **Hypervisor:** VMware ESXi 6.5+ supported
                """)
                
                st.markdown("""
                **⚙️ Performance Optimization:**
                - Enable hardware acceleration
                - Reserve 50% CPU, 100% memory
                - Use SSD storage for VM files
                - Configure jumbo frames (9000 MTU)
                - Separate VMs across different hosts
                """)
            
            with col2:
                # Calculate total VMware requirements
                if datasync_scaling['applicable']:
                    optimal = datasync_scaling['optimal_config']
                    service_spec = analyzer.migration_services['datasync']['sizes'][optimal['size']]
                    
                    total_vcpu = service_spec['vcpu'] * optimal['instances']
                    total_memory = service_spec['memory_gb'] * optimal['instances']
                    total_storage = 80 * optimal['instances']  # 80GB per VM
                    
                    st.markdown(f"""
                    **🏗️ Total VMware Resource Requirements:**
                    - **Total vCPU:** {total_vcpu} cores
                    - **Total Memory:** {total_memory} GB
                    - **Total Storage:** {total_storage} GB
                    - **Network Bandwidth:** {optimal['effective_throughput']:,.0f} Mbps
                    - **Recommended Hosts:** {math.ceil(optimal['instances'] / 2)} (max 2 VMs per host)
                    """)
                    
                    st.markdown(f"""
                    **💾 Resource Planning:**
                    - **Physical CPU ratio:** 2:1 overcommit max
                    - **Memory overcommit:** 1.2:1 max  
                    - **Network ports:** {optimal['instances']} × 10GbE
                    - **Storage IOPS:** 500 IOPS per VM minimum
                    """)
        
    
    
    # Sequential flow diagram
    create_sequential_flow_diagram(waterfall_data)
    
    # Enhanced waterfall chart
    st.markdown("""
    <div class="waterfall-container">
    """, unsafe_allow_html=True)
    
    waterfall_chart = create_enhanced_waterfall_chart(waterfall_data)
    st.plotly_chart(waterfall_chart, use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Detailed breakdown in corporate cards
    col1, col2 = st.columns(2)
    
    with col1:
        infrastructure = waterfall_data['infrastructure_details']
        
        st.markdown(f"""
        <div class="corporate-card status-card-info">
            <h3>🏗️ Infrastructure Component Details</h3>
            <p><strong>Operating System:</strong> {infrastructure['os']['name']}</p>
            <p><strong>TCP Efficiency:</strong> {infrastructure['os']['tcp_stack_efficiency']*100:.1f}%</p>
            <p><strong>Network Interface:</strong> {infrastructure['nic']['name']}</p>
            <p><strong>NIC Efficiency:</strong> {infrastructure['nic']['real_world_efficiency']*100:.1f}%</p>
            <p><strong>LAN Infrastructure:</strong> {infrastructure['lan']['name']}</p>
            <p><strong>Oversubscription Ratio:</strong> {infrastructure['lan']['oversubscription_ratio']}</p>
            <p><strong>WAN Provider:</strong> {infrastructure['wan']['name']}</p>
            <p><strong>WAN Efficiency:</strong> {infrastructure['wan']['bandwidth_efficiency']*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        primary_bottleneck = summary['primary_bottleneck_layer']
        
        if primary_bottleneck == 'nic':
            recommendation = "Upgrade to higher bandwidth NIC (25/100 Gbps)"
            card_type = "status-card-warning"
        elif primary_bottleneck == 'os':
            recommendation = "Optimize OS network stack, enable kernel bypass"
            card_type = "status-card-warning"
        elif primary_bottleneck == 'lan':
            recommendation = "Reduce oversubscription, increase switch capacity"
            card_type = "status-card-warning"
        elif primary_bottleneck == 'wan':
            recommendation = "Upgrade WAN bandwidth or provider tier"
            card_type = "status-card-error"
        elif primary_bottleneck == 'service':
            recommendation = "Scale service instances or upgrade instance size"
            card_type = "status-card-success"
        else:
            recommendation = "Review overall infrastructure architecture"
            card_type = "status-card-info"
        
        st.markdown(f"""
        <div class="corporate-card {card_type}">
            <h3>🎯 Optimization Recommendations</h3>
            <p><strong>Primary Bottleneck:</strong> {summary['primary_bottleneck']}</p>
            <p><strong>Impact:</strong> {summary['primary_bottleneck_impact_mbps']:,.0f} Mbps reduction</p>
            <p><strong>Recommendation:</strong> {recommendation}</p>
            <p><strong>Expected Improvement:</strong> Up to {summary['primary_bottleneck_impact_mbps']:,.0f} Mbps</p>
        </div>
        """, unsafe_allow_html=True)
    
    return waterfall_data

def render_migration_analysis_tab(config: Dict, waterfall_data: Dict, analyzer: EnhancedNetworkAnalyzer):
    """Render migration timing and compatibility analysis with corporate styling"""
    st.subheader("⏱️ Migration Analysis & Compatibility")
    
    # Service compatibility
    pattern_key = analyzer.determine_optimal_pattern(
        config['source_location'], 
        config['environment'], 
        config['migration_service']
    )
    
    service_compatibility = analyzer.assess_service_compatibility(
        pattern_key, config['migration_service'], config['service_size']
    )
    
    # Migration timing
    migration_time = analyzer.estimate_migration_time(
        config['data_size_gb'],
        waterfall_data['summary']['final_effective_mbps'],
        config['migration_service']
    )
    
    # Compatibility and timing metrics in corporate style
    st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{config['migration_service'].upper()}</div>
            <div class="metric-label">Migration Service</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        vpc_status = "✅ Yes" if service_compatibility['vpc_endpoint_compatible'] else "❌ No"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{vpc_status}</div>
            <div class="metric-label">VPC Endpoint Support</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{config['data_size_gb']:,}</div>
            <div class="metric-label">Data Size (GB)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{waterfall_data['summary']['final_effective_mbps']:,.0f}</div>
            <div class="metric-label">Effective Speed (Mbps)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{migration_time['data_transfer_hours']:.1f}h</div>
            <div class="metric-label">Transfer Time</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col6:
        meets_requirement = migration_time['total_hours'] <= config['max_downtime_hours']
        status_text = "✅ Meets SLA" if meets_requirement else "❌ Exceeds SLA"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{status_text}</div>
            <div class="metric-label">SLA Compliance</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Detailed analysis in corporate cards
    col1, col2 = st.columns(2)
    
    with col1:
        # Determine card type based on compatibility issues
        if service_compatibility['warnings']:
            card_type = "status-card-error"
        elif service_compatibility['requirements']:
            card_type = "status-card-warning"
        else:
            card_type = "status-card-success"
        
        compatibility_content = ""
        
        if service_compatibility['warnings']:
            compatibility_content += "<h4>🚨 Warnings:</h4>"
            for warning in service_compatibility['warnings']:
                compatibility_content += f"<p>• {warning}</p>"
        
        if service_compatibility['requirements']:
            compatibility_content += "<h4>📋 Requirements:</h4>"
            for requirement in service_compatibility['requirements']:
                compatibility_content += f"<p>• {requirement}</p>"
        
        if service_compatibility['recommendations']:
            compatibility_content += "<h4>💡 Recommendations:</h4>"
            for recommendation in service_compatibility['recommendations']:
                compatibility_content += f"<p>• {recommendation}</p>"
        
        if not any([service_compatibility['warnings'], service_compatibility['requirements'], service_compatibility['recommendations']]):
            compatibility_content = "<p>✅ No compatibility issues detected</p>"
        
        st.markdown(f"""
        <div class="corporate-card {card_type}">
            <h3>📋 Service Compatibility Analysis</h3>
            {compatibility_content}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Timeline analysis
        timeline_data = [
            {"Phase": "Setup & Config", "Hours": migration_time['setup_hours']},
            {"Phase": "Data Transfer", "Hours": migration_time['data_transfer_hours']},
            {"Phase": "Validation", "Hours": migration_time['validation_hours']}
        ]
        
        df_timeline = pd.DataFrame(timeline_data)
        
        # Migration timeline card
        meets_sla = migration_time['total_hours'] <= config['max_downtime_hours']
        timeline_card_type = "status-card-success" if meets_sla else "status-card-error"
        
        incremental_support = "✅ Supports incremental migration" if migration_time.get('supports_incremental') else "❌ Full migration required"
        
        st.markdown(f"""
        <div class="corporate-card {timeline_card_type}">
            <h3>⏱️ Migration Timeline Analysis</h3>
            <p><strong>Service:</strong> {migration_time['service_name']}</p>
            <p><strong>Total Time:</strong> {migration_time['total_hours']:.1f} hours</p>
            <p><strong>Recommended Window:</strong> {migration_time['recommended_window_hours']:.1f} hours</p>
            <p><strong>SLA Compliance:</strong> {'✅ Pass' if meets_sla else '❌ Fail'}</p>
            <p><strong>Incremental Support:</strong> {incremental_support}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**Timeline Breakdown:**")
        st.dataframe(df_timeline, use_container_width=True, hide_index=True)
    
    return {
        'migration_time': migration_time,
        'service_compatibility': service_compatibility,
        'waterfall_data': waterfall_data
    }

def render_ai_recommendations_tab(config: Dict, analysis_results: Dict, analyzer: EnhancedNetworkAnalyzer):
    """Render AI-powered recommendations with corporate styling"""
    st.subheader("🤖 AI-Powered Migration Recommendations")
    
    ai_recommendations = analyzer.generate_ai_recommendations(config, analysis_results)
    
    # Overview metrics in corporate style
    st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        complexity_color = {
            'low': 'var(--success-green)',
            'medium': 'var(--warning-orange)',
            'high': 'var(--error-red)'
        }.get(ai_recommendations['migration_complexity'], 'var(--medium-gray)')
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: {complexity_color};">{ai_recommendations['migration_complexity'].title()}</div>
            <div class="metric-label">Migration Complexity</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        confidence_color = {
            'high': 'var(--success-green)',
            'medium': 'var(--warning-orange)',
            'low': 'var(--error-red)'
        }.get(ai_recommendations['confidence_level'], 'var(--medium-gray)')
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: {confidence_color};">{ai_recommendations['confidence_level'].title()}</div>
            <div class="metric-label">AI Confidence</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        waterfall_summary = analysis_results['waterfall_data']['summary']
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{waterfall_summary['primary_bottleneck_layer'].title()}</div>
            <div class="metric-label">Primary Bottleneck</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        migration_time = analysis_results['migration_time']
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{migration_time['recommended_window_hours']:.1f}h</div>
            <div class="metric-label">Migration Window</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Recommendations by priority in corporate cards
    st.markdown("### 💡 Prioritized Recommendations")
    
    # Group by priority
    critical_recs = [rec for rec in ai_recommendations['recommendations'] if rec['priority'] == 'critical']
    high_recs = [rec for rec in ai_recommendations['recommendations'] if rec['priority'] == 'high']
    medium_recs = [rec for rec in ai_recommendations['recommendations'] if rec['priority'] == 'medium']
    low_recs = [rec for rec in ai_recommendations['recommendations'] if rec['priority'] == 'low']
    
    if critical_recs:
        st.markdown("#### 🚨 Critical Issues")
        for i, rec in enumerate(critical_recs, 1):
            st.markdown(f"""
            <div class="corporate-card status-card-error">
                <h4>🔴 {rec['type'].replace('_', ' ').title()}</h4>
                <p><strong>Description:</strong> {rec['description']}</p>
                <p><strong>Expected Impact:</strong> {rec['impact']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    if high_recs:
        st.markdown("#### ⚠️ High Priority")
        for i, rec in enumerate(high_recs, 1):
            st.markdown(f"""
            <div class="corporate-card status-card-warning">
                <h4>🟠 {rec['type'].replace('_', ' ').title()}</h4>
                <p><strong>Description:</strong> {rec['description']}</p>
                <p><strong>Expected Impact:</strong> {rec['impact']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    if medium_recs:
        st.markdown("#### 📋 Medium Priority")
        for i, rec in enumerate(medium_recs, 1):
            st.markdown(f"""
            <div class="corporate-card status-card-info">
                <h4>🟡 {rec['type'].replace('_', ' ').title()}</h4>
                <p><strong>Description:</strong> {rec['description']}</p>
                <p><strong>Expected Impact:</strong> {rec['impact']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    if low_recs:
        st.markdown("#### ✅ Low Priority")
        for i, rec in enumerate(low_recs, 1):
            st.markdown(f"""
            <div class="corporate-card status-card-success">
                <h4>🟢 {rec['type'].replace('_', ' ').title()}</h4>
                <p><strong>Description:</strong> {rec['description']}</p>
                <p><strong>Expected Impact:</strong> {rec['impact']}</p>
            </div>
            """, unsafe_allow_html=True)
    
        # ADD this section to render_ai_recommendations_tab() 
    # Insert after the existing recommendations sections (around line 1650):

    # Enhanced DataSync AI Recommendations (ADD THIS SECTION)
    if config['migration_service'] == 'datasync':
        pattern_key = analyzer.determine_optimal_pattern(
            config['source_location'], 
            config['environment'], 
            config['migration_service']
        )
        
        datasync_scaling = analyzer.get_datasync_scaling_recommendations(pattern_key, config)
        
        if datasync_scaling['applicable']:
            st.markdown("### 🚀 DataSync-Specific AI Recommendations")
            
            # Performance recommendations
            if datasync_scaling['is_bottleneck']:
                st.markdown(f"""
                <div class="corporate-card status-card-error">
                    <h4>🔴 DataSync Performance Bottleneck Detected</h4>
                    <p><strong>Issue:</strong> DataSync VM capacity is limiting your migration performance</p>
                    <p><strong>Current:</strong> {datasync_scaling['current_config']['instances']} × {config['service_size']} VM(s)</p>
                    <p><strong>Recommended:</strong> {datasync_scaling['optimal_config']['instances']} × {datasync_scaling['optimal_config']['size']} VM(s)</p>
                    <p><strong>Expected Improvement:</strong> {(datasync_scaling['optimal_config']['effective_throughput'] / datasync_scaling['current_config']['effective_throughput'] - 1) * 100:.0f}% faster transfer</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="corporate-card status-card-success">
                    <h4>✅ DataSync Configuration Optimal</h4>
                    <p>Your current DataSync configuration is not the bottleneck in your migration pipeline.</p>
                    <p><strong>Effective Throughput:</strong> {datasync_scaling['current_config']['effective_throughput']:,.0f} Mbps</p>
                </div>
                """, unsafe_allow_html=True)
            
            # VMware-specific recommendations
            optimal_instances = datasync_scaling['optimal_config']['instances']
            if optimal_instances > 1:
                st.markdown(f"""
                <div class="corporate-card status-card-info">
                    <h4>🖥️ VMware Deployment Strategy</h4>
                    <p><strong>Multi-VM Deployment:</strong> Deploy {optimal_instances} DataSync VMs for optimal performance</p>
                    <p><strong>Host Distribution:</strong> Spread VMs across {math.ceil(optimal_instances / 2)} VMware hosts</p>
                    <p><strong>Resource Isolation:</strong> Use CPU and memory reservations to guarantee performance</p>
                    <p><strong>Network Optimization:</strong> Dedicated vSwitches and port groups for DataSync traffic</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Cost vs Performance Analysis
            current_throughput = datasync_scaling['current_config']['effective_throughput']
            optimal_throughput = datasync_scaling['optimal_config']['effective_throughput']
            performance_gain = (optimal_throughput / current_throughput - 1) * 100 if current_throughput > 0 else 0
            
            # Estimate cost difference
            current_instances = datasync_scaling['current_config']['instances']
            optimal_instances = datasync_scaling['optimal_config']['instances']
            cost_multiplier = optimal_instances / current_instances if current_instances > 0 else optimal_instances
            
            st.markdown(f"""
            <div class="corporate-card">
                <h4>💰 Cost vs Performance Trade-off Analysis</h4>
                <p><strong>Performance Gain:</strong> {performance_gain:.0f}% faster migration</p>
                <p><strong>Cost Impact:</strong> {cost_multiplier:.1f}x DataSync VM costs</p>
                <p><strong>Migration Time Reduction:</strong> {(1 - 1/cost_multiplier) * 100:.0f}% shorter window</p>
                <p><strong>ROI Consideration:</strong> Shorter migration window may offset higher VM costs</p>
            </div>
            """, unsafe_allow_html=True)

    # Also ADD this to the generate_ai_recommendations method in the EnhancedNetworkAnalyzer class
    # Add around line 900 in the generate_ai_recommendations method:

    def generate_ai_recommendations(self, config: Dict, analysis_results: Dict) -> Dict:
        """Generate AI-powered recommendations (ENHANCED VERSION)"""
        migration_time = analysis_results['migration_time']
        waterfall_data = analysis_results['waterfall_data']
        service_compatibility = analysis_results.get('service_compatibility', {})
        
        recommendations = []
        priority_score = 0
        
        # NEW: DataSync-specific recommendations
        if config['migration_service'] == 'datasync':
            pattern_key = self.determine_optimal_pattern(
                config['source_location'], 
                config['environment'], 
                config['migration_service']
            )
            
            datasync_scaling = self.get_datasync_scaling_recommendations(pattern_key, config)
            
            if datasync_scaling['applicable'] and datasync_scaling['is_bottleneck']:
                recommendations.append({
                    'type': 'datasync_scaling',
                    'priority': 'high',
                    'description': f'DataSync bottleneck detected. Scale to {datasync_scaling["optimal_config"]["instances"]} VMs for optimal performance.',
                    'impact': f'{(datasync_scaling["optimal_config"]["effective_throughput"] / datasync_scaling["current_config"]["effective_throughput"] - 1) * 100:.0f}% performance improvement'
                })
                priority_score += 25
        
        # ... rest of existing recommendation logic ...
    
    
    
    # Summary card
    waterfall_summary = analysis_results['waterfall_data']['summary']
    migration_time = analysis_results['migration_time']
    
    st.markdown(f"""
    <div class="corporate-card">
        <h3>🎯 Migration Strategy Summary</h3>
        <p><strong>Service:</strong> {waterfall_summary['service_name']}</p>
        <p><strong>Complexity:</strong> {ai_recommendations['migration_complexity'].title()}</p>
        <p><strong>Confidence:</strong> {ai_recommendations['confidence_level'].title()}</p>
        <p><strong>Timeline:</strong> {migration_time['total_hours']:.1f} hours ({migration_time['total_days']:.1f} days)</p>
        <p><strong>Primary Bottleneck:</strong> {waterfall_summary['primary_bottleneck_layer'].title()}</p>
        <p><strong>Infrastructure Efficiency:</strong> {waterfall_summary['efficiency_percentage']:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)

def render_pattern_comparison_tab(analyzer: EnhancedNetworkAnalyzer, config: Dict):
    """Render pattern comparison analysis with corporate styling"""
    st.subheader("🔄 Network Pattern Comparison")
    
    # Initialize clients with API keys if provided
    if config.get('aws_access_key') and config.get('aws_secret_key'):
        analyzer.pricing_client.initialize_client(config['aws_access_key'], config['aws_secret_key'])
    
    if config.get('claude_api_key'):
        analyzer.ai_client.api_key = config['claude_api_key']
    
    # Analyze all patterns
    with st.spinner("Analyzing network patterns..."):
        pattern_analyses = analyzer.analyze_all_patterns(config)
    
    if not pattern_analyses:
        st.error("No suitable patterns found for the selected configuration.")
        return None, None
    
    # Get AI recommendation
    with st.spinner("Getting AI recommendations..."):
        ai_recommendation = analyzer.get_ai_recommendation(pattern_analyses, config)
    
    # Display best pattern recommendation
    best_pattern = pattern_analyses[0]
    st.markdown(f"""
    <div class="best-pattern-highlight">
        🏆 AI RECOMMENDED: {best_pattern.pattern_name}
        <br>
        Confidence: {ai_recommendation['confidence_score']*100:.0f}% | 
        Cost: ${best_pattern.total_cost_usd:,.0f} | 
        Time: {best_pattern.migration_time_hours:.1f}h
    </div>
    """, unsafe_allow_html=True)
    
    # Comparison metrics in corporate style
    st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">${min(p.total_cost_usd for p in pattern_analyses):,.0f}</div>
            <div class="metric-label">Best Cost</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{max(p.effective_bandwidth_mbps for p in pattern_analyses):,.0f}</div>
            <div class="metric-label">Best Speed (Mbps)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{min(p.migration_time_hours for p in pattern_analyses):.1f}h</div>
            <div class="metric-label">Fastest Migration</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{max(p.reliability_score for p in pattern_analyses)*100:.1f}%</div>
            <div class="metric-label">Best Reliability</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{best_pattern.ai_recommendation_score:.2f}</div>
            <div class="metric-label">AI Score</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Detailed comparison table
    st.markdown("### 📊 Detailed Pattern Comparison")
    
    comparison_data = []
    for pattern in pattern_analyses:
        comparison_data.append({
            'Pattern': pattern.pattern_name.split('→')[0].strip(),
            'Cost ($)': f"{pattern.total_cost_usd:,.0f}",
            'Time (h)': f"{pattern.migration_time_hours:.1f}",
            'Bandwidth (Mbps)': f"{pattern.effective_bandwidth_mbps:,.0f}",
            'Reliability': f"{pattern.reliability_score*100:.1f}%",
            'Complexity': f"{pattern.complexity_score*100:.0f}%",
            'AI Score': f"{pattern.ai_recommendation_score:.2f}"
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    st.dataframe(df_comparison, use_container_width=True, hide_index=True)
    
    # Cost vs Performance Chart with corporate styling
    fig_scatter = go.Figure()
    
    for i, pattern in enumerate(pattern_analyses):
        color = '#059669' if i == 0 else '#3b82f6'  # Green for best, blue for others
        size = 20 if i == 0 else 15
        
        fig_scatter.add_trace(go.Scatter(
            x=[pattern.total_cost_usd],
            y=[pattern.migration_time_hours],
            mode='markers+text',
            marker=dict(size=size, color=color),
            text=[pattern.pattern_name.split('→')[0].strip()],
            textposition="top center",
            name=pattern.pattern_name.split('→')[0].strip(),
            hovertemplate=f"<b>{pattern.pattern_name}</b><br>" +
                         f"Cost: ${pattern.total_cost_usd:,.0f}<br>" +
                         f"Time: {pattern.migration_time_hours:.1f}h<br>" +
                         f"Bandwidth: {pattern.effective_bandwidth_mbps:,.0f} Mbps<br>" +
                         f"AI Score: {pattern.ai_recommendation_score:.2f}<extra></extra>"
        ))
    
    fig_scatter.update_layout(
        title={
            'text': "Migration Cost vs Time Analysis",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#1e3a8a'}
        },
        xaxis_title="Total Cost ($)",
        yaxis_title="Migration Time (hours)",
        showlegend=False,
        template="plotly_white",
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    return pattern_analyses, ai_recommendation

def render_network_path_diagram_tab():
    """Render the complete network path diagram tab"""
    
    st.header("🌐 Network Path Architecture")
    st.markdown("Comprehensive view of enterprise network architecture for AWS migration services")
    
    # Environment overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🏢 Production Environment**
        - **Source:** San Antonio Data Center
        - **Transit:** San Jose Data Center  
        - **Link:** 10Gbps Shared + 10Gbps DX
        - **Target:** AWS West 2 Production VPC
        - **DataSync Location:** VMware (SA/SJ)
        """)
    
    with col2:
        st.markdown("""
        **🔧 Non-Production Environment**
        - **Source:** San Jose Data Center
        - **Link:** 2Gbps Direct Connect
        - **Target:** AWS West 2 Non-Prod VPC
        - **DataSync Location:** VMware (SJ)
        """)
    
    # Network topology diagram
    st.subheader("🏗️ Network Topology")
    topology_fig = create_network_topology_diagram()
    st.plotly_chart(topology_fig, use_container_width=True)
    
    # Service flows
    st.subheader("🔄 Service Flow Patterns")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        service_flows_fig = create_service_flows_diagram()
        st.plotly_chart(service_flows_fig, use_container_width=True)
    
    with col2:
        st.markdown("""
        **Service Locations:**
        
        **🏢 On-Premises (VMware)**
        - DataSync Agent
        
        **☁️ AWS VPC**
        - FSx File Systems
        - DMS Replication Instances  
        - Storage Gateway
        
        **🌐 AWS Managed**
        - VPC Endpoints
        - S3 Target Storage
        """)
    
    # Service comparison table
    st.subheader("📊 Service Path Comparison")
    service_df = create_service_comparison_table()
    st.dataframe(service_df, use_container_width=True, hide_index=True)
    
    # Key insights
    st.subheader("💡 Architecture Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🚀 Performance Characteristics**
        - Production: 10Gbps bandwidth
        - Non-Production: 2Gbps bandwidth
        - DataSync: On-premises agent
        - Low latency via Direct Connect
        """)
    
    with col2:
        st.markdown("""
        **🔒 Security & Compliance**
        - Private connectivity via DX
        - VPC isolation for environments
        - Encrypted data in transit
        - AWS managed endpoints
        """)
    
    with col3:
        st.markdown("""
        **💰 Cost Optimization**
        - Shared San Jose infrastructure
        - Right-sized DX connections
        - Service-specific routing
        - Efficient data transfer patterns
        """)
    
    # Network path details
    with st.expander("📋 Detailed Network Path Analysis", expanded=False):
        
        st.markdown("### Production Environment Data Flow")
        st.markdown("""
        1. **San Antonio DC** → 10Gbps Shared Link → **San Jose DC**
        2. **San Jose DC** → 10Gbps Direct Connect → **AWS West 2**
        3. **AWS West 2** → **Production VPC** → **Target Services (S3)**
        
        **DataSync Agent:** Deployed on VMware infrastructure in San Antonio or San Jose
        """)
        
        st.markdown("### Non-Production Environment Data Flow")
        st.markdown("""
        1. **San Jose DC** → 2Gbps Direct Connect → **AWS West 2**
        2. **AWS West 2** → **Non-Prod VPC** → **Target Services (S3)**
        
        **DataSync Agent:** Deployed on VMware infrastructure in San Jose
        """)
        
        st.markdown("### Service-Specific Routing")
        st.markdown("""
        - **VPC Endpoint:** Direct private connection to S3, bypassing internet
        - **FSx:** File system service within VPC, syncs to S3
        - **DataSync:** On-premises agent transfers directly to S3
        - **DMS:** Database replication through VPC to S3
        - **Storage Gateway:** Hybrid storage bridge from VPC to S3
        """)

def render_database_guidance_tab(config: Dict):
    """Render database engineer guidance with corporate styling"""
    st.subheader("📚 Database Engineer's Migration Guide")
    
    database_scenario = config.get('database_scenario', 'mysql_oltp_rds')
    
    # Get analyzer instance to access database scenarios
    analyzer = EnhancedNetworkAnalyzer()
    selected_scenario = analyzer.database_scenarios[database_scenario]
    
    # Enhanced Database-specific guidance
    guidance_content = {
        'mysql_oltp_rds': {
            'title': '🔄 MySQL OLTP Database → RDS MySQL',
            'key_considerations': [
                "Binary log settings for replication consistency",
                "InnoDB buffer pool warming strategies",
                "Connection pool configuration",
                "Read replica lag monitoring",
                "Parameter group optimization for RDS"
            ],
            'network_requirements': {
                'min_bandwidth': '500 Mbps',
                'max_latency': '10ms',
                'consistency': 'Strict (ACID compliance required)'
            },
            'recommended_approach': 'Use AWS DMS with full load + CDC for minimal downtime',
            'testing_strategy': [
                "Test replication lag under peak load",
                "Validate foreign key constraints",
                "Performance test critical queries",
                "Failover/failback procedures",
                "RDS monitoring and alerting setup"
            ],
            'migration_steps': [
                "Setup DMS replication instance",
                "Create source and target endpoints",
                "Configure CDC for ongoing replication",
                "Perform initial data load",
                "Monitor lag and validate data",
                "Cut over during maintenance window"
            ]
        },
        'postgresql_analytics_rds': {
            'title': '📊 PostgreSQL Analytics → RDS PostgreSQL',
            'key_considerations': [
                "Vacuum and analyze statistics",
                "Extension compatibility (PostGIS, etc.)",
                "Large table partitioning strategy",
                "Query performance optimization",
                "RDS parameter tuning for analytics workloads"
            ],
            'network_requirements': {
                'min_bandwidth': '1000 Mbps',
                'max_latency': '50ms',
                'consistency': 'Eventual (some lag acceptable for analytics)'
            },
            'recommended_approach': 'Combination of DMS for initial load + DataSync for large data files',
            'testing_strategy': [
                "Validate complex analytical queries",
                "Test ETL pipeline compatibility",
                "Check data type conversions",
                "Performance baseline comparison",
                "Extension functionality validation"
            ],
            'migration_steps': [
                "Assess extension compatibility",
                "Export/import custom functions",
                "Migrate schema using DMS SCT",
                "Bulk load historical data",
                "Setup incremental analytics refresh",
                "Validate query performance"
            ]
        },
        'oracle_enterprise_rds': {
            'title': '🏢 Oracle Enterprise → RDS Oracle',
            'key_considerations': [
                "Oracle-specific features compatibility",
                "PL/SQL code conversion needs",
                "Tablespace and datafile strategy",
                "RAC to RDS conversion complexity",
                "License considerations for RDS Oracle"
            ],
            'network_requirements': {
                'min_bandwidth': '2000 Mbps',
                'max_latency': '5ms',
                'consistency': 'Strict (Enterprise SLA requirements)'
            },
            'recommended_approach': 'AWS DMS with SCT for schema conversion + careful testing',
            'testing_strategy': [
                "Schema conversion validation",
                "Application compatibility testing",
                "Performance regression testing",
                "Disaster recovery validation",
                "Oracle-specific feature testing"
            ],
            'migration_steps': [
                "Run AWS SCT assessment",
                "Convert schema and procedures",
                "Setup DMS replication",
                "Migrate data with full load + CDC",
                "Application connection string updates",
                "Performance tuning and optimization"
            ]
        },
        'sqlserver_enterprise_ec2': {
            'title': '🪟 SQL Server Enterprise → EC2',
            'key_considerations': [
                "Windows licensing on EC2",
                "AlwaysOn Availability Groups setup",
                "Storage configuration (EBS optimization)",
                "Security group and network configuration",
                "Backup strategy for EC2 environment"
            ],
            'network_requirements': {
                'min_bandwidth': '1500 Mbps',
                'max_latency': '8ms',
                'consistency': 'Strict (Enterprise requirements)'
            },
            'recommended_approach': 'Native SQL Server tools + DMS for validation',
            'testing_strategy': [
                "AlwaysOn configuration testing",
                "Application driver compatibility",
                "Performance under load testing",
                "Backup and restore procedures",
                "Disaster recovery validation"
            ],
            'migration_steps': [
                "Setup EC2 instances with SQL Server",
                "Configure storage and networking",
                "Setup native replication or log shipping",
                "Migrate databases using native backup/restore",
                "Configure AlwaysOn if required",
                "Application cutover and validation"
            ]
        },
        'mongodb_cluster_documentdb': {
            'title': '🍃 MongoDB Cluster → DocumentDB',
            'key_considerations': [
                "DocumentDB API compatibility assessment",
                "Index strategy optimization for DocumentDB",
                "Connection string and driver updates",
                "Aggregation pipeline compatibility",
                "Change streams configuration"
            ],
            'network_requirements': {
                'min_bandwidth': '1500 Mbps',
                'max_latency': '20ms',
                'consistency': 'Configurable (adjust read/write concerns)'
            },
            'recommended_approach': 'Native MongoDB tools + DMS for validation',
            'testing_strategy': [
                "API compatibility verification",
                "Application driver testing",
                "Performance comparison testing",
                "Aggregation pipeline validation",
                "Change streams functionality"
            ],
            'migration_steps': [
                "Assess DocumentDB compatibility",
                "Setup DocumentDB cluster",
                "Use mongodump/mongorestore for migration",
                "Validate data integrity",
                "Update application connection strings",
                "Performance testing and optimization"
            ]
        },
        'mysql_analytics_aurora': {
            'title': '📊 MySQL Analytics → Aurora MySQL',
            'key_considerations': [
                "Aurora MySQL engine version compatibility",
                "Parallel query optimization for analytics",
                "Aurora Serverless for variable workloads",
                "Global database for multi-region analytics",
                "Aurora ML integration opportunities"
            ],
            'network_requirements': {
                'min_bandwidth': '1200 Mbps',
                'max_latency': '25ms',
                'consistency': 'Eventual (analytics workloads)'
            },
            'recommended_approach': 'AWS DMS for continuous replication',
            'testing_strategy': [
                "Parallel query performance testing",
                "Analytics workload validation",
                "Aurora Serverless scaling testing",
                "Cross-region replication testing",
                "Performance baseline comparison"
            ],
            'migration_steps': [
                "Setup Aurora MySQL cluster",
                "Configure DMS for initial load + CDC",
                "Enable parallel query for analytics",
                "Optimize for analytical workloads",
                "Setup read replicas if needed",
                "Application cutover and validation"
            ]
        },
        'postgresql_oltp_aurora': {
            'title': '🔄 PostgreSQL OLTP → Aurora PostgreSQL',
            'key_considerations': [
                "Aurora PostgreSQL compatibility",
                "Connection pooling optimization",
                "Aurora Serverless for variable OLTP loads",
                "Point-in-time recovery configuration",
                "Performance Insights setup"
            ],
            'network_requirements': {
                'min_bandwidth': '800 Mbps',
                'max_latency': '12ms',
                'consistency': 'Strict (OLTP requirements)'
            },
            'recommended_approach': 'AWS DMS with minimal downtime',
            'testing_strategy': [
                "OLTP transaction testing",
                "Connection pooling optimization",
                "Failover testing",
                "Performance monitoring validation",
                "Application compatibility testing"
            ],
            'migration_steps': [
                "Setup Aurora PostgreSQL cluster",
                "Configure DMS replication",
                "Test application connectivity",
                "Optimize for OLTP workloads",
                "Setup monitoring and alerting",
                "Planned cutover execution"
            ]
        },
        'mariadb_oltp_rds': {
            'title': '🗄️ MariaDB OLTP → RDS MariaDB',
            'key_considerations': [
                "MariaDB version compatibility",
                "Storage engine considerations",
                "Connection handling optimization",
                "Replication configuration",
                "Parameter group optimization"
            ],
            'network_requirements': {
                'min_bandwidth': '600 Mbps',
                'max_latency': '10ms',
                'consistency': 'Strict (OLTP requirements)'
            },
            'recommended_approach': 'AWS DMS for live migration',
            'testing_strategy': [
                "MariaDB-specific feature testing",
                "Storage engine validation",
                "Replication lag monitoring",
                "Performance comparison",
                "Application compatibility verification"
            ],
            'migration_steps': [
                "Setup RDS MariaDB instance",
                "Configure DMS endpoints",
                "Initialize replication",
                "Monitor and validate data",
                "Application connection updates",
                "Cutover and post-migration validation"
            ]
        }
    }
    
    selected_guidance = guidance_content.get(database_scenario, guidance_content['mysql_oltp_rds'])
    
    # Display guidance
    st.markdown(f"""
    <div class="corporate-card">
        <h3>{selected_guidance['title']}</h3>
        <p><strong>Recommended Approach:</strong> {selected_guidance['recommended_approach']}</p>
        <p><strong>Migration Complexity:</strong> {selected_scenario['migration_complexity'].title()}</p>
        <p><strong>Downtime Sensitivity:</strong> {selected_scenario['downtime_sensitivity'].title()}</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        considerations_content = ""
        for consideration in selected_guidance['key_considerations']:
            considerations_content += f"<p>• {consideration}</p>"
        
        testing_content = ""
        for test in selected_guidance['testing_strategy']:
            testing_content += f"<p>• {test}</p>"
        
        st.markdown(f"""
        <div class="corporate-card status-card-info">
            <h3>🔧 Key Technical Considerations</h3>
            {considerations_content}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="corporate-card status-card-success">
            <h3>📋 Testing Strategy</h3>
            {testing_content}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        requirements = selected_guidance['network_requirements']
        
        st.markdown(f"""
        <div class="corporate-card">
            <h3>🌐 Network Requirements</h3>
            
            <div class="metric-card" style="margin: 1rem 0;">
                <div class="metric-value" style="color: var(--secondary-blue);">{requirements['min_bandwidth']}</div>
                <div class="metric-label">Minimum Bandwidth</div>
            </div>
            
            <div class="metric-card" style="margin: 1rem 0;">
                <div class="metric-value" style="color: var(--warning-orange);">{requirements['max_latency']}</div>
                <div class="metric-label">Maximum Latency</div>
            </div>
            
            <div class="metric-card" style="margin: 1rem 0;">
                <div class="metric-value" style="color: var(--success-green); font-size: 1.2em;">{requirements['consistency']}</div>
                <div class="metric-label">Consistency Model</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Migration steps
        steps_content = ""
        for i, step in enumerate(selected_guidance['migration_steps'], 1):
            steps_content += f"<p><strong>{i}.</strong> {step}</p>"
        
        st.markdown(f"""
        <div class="corporate-card status-card-warning">
            <h3>📝 Migration Steps</h3>
            {steps_content}
        </div>
        """, unsafe_allow_html=True)

def render_ai_insights_tab(pattern_analyses: List[PatternAnalysis], ai_recommendation: Dict, config: Dict):
    """Render AI insights and recommendations with corporate styling"""
    st.subheader("🤖 AI-Powered Migration Insights")
    
    if not pattern_analyses or not ai_recommendation:
        st.warning("Please run pattern comparison first to get AI insights.")
        return
    
    # AI Recommendation Summary
    st.markdown(f"""
    <div class="corporate-card">
        <h3>🎯 AI Recommendation: {ai_recommendation['recommended_pattern'].replace('_', ' ').title()}</h3>
        <p><strong>Confidence Level:</strong> {ai_recommendation['confidence_score']*100:.0f}%</p>
        <p><strong>Reasoning:</strong> {ai_recommendation['reasoning']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Database-Specific Insights
    database_scenario = config.get('database_scenario', 'mysql_oltp_rds')
    
    # Get analyzer instance to access database scenarios
    analyzer = EnhancedNetworkAnalyzer()
    db_info = analyzer.database_scenarios[database_scenario]
    
    st.markdown(f"""
    <div class="corporate-card status-card-info">
        <h3>🗄️ Database-Specific Considerations for {db_info['name']}</h3>
        <p><strong>Target Service:</strong> {db_info['target_service']}</p>
        <p><strong>Database Insights:</strong> {ai_recommendation['database_considerations']}</p>
        <p><strong>Risk Assessment:</strong> {ai_recommendation['risk_assessment']}</p>
        <p><strong>Cost Justification:</strong> {ai_recommendation['cost_justification']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Pattern Deep Dive
    best_pattern = pattern_analyses[0]
    
    col1, col2 = st.columns(2)
    
    with col1:
        pros_content = ""
        for pro in best_pattern.pros:
            pros_content += f"<p>• {pro}</p>"
        
        use_cases_content = ""
        for use_case in best_pattern.use_cases:
            use_cases_content += f"<p>• {use_case}</p>"
        
        st.markdown(f"""
        <div class="corporate-card status-card-success">
            <h3>✅ Recommended Pattern Advantages</h3>
            {pros_content}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="corporate-card status-card-info">
            <h3>🎯 Best Use Cases</h3>
            {use_cases_content}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        cons_content = ""
        for con in best_pattern.cons:
            cons_content += f"<p>• {con}</p>"
        
        optimization_content = ""
        if best_pattern.complexity_score > 0.7:
            optimization_content += "<p>• High complexity - consider professional services engagement</p>"
        if best_pattern.total_cost_usd > 5000:
            optimization_content += "<p>• High cost - evaluate phased migration approach</p>"
        if best_pattern.migration_time_hours > config['max_downtime_hours']:
            optimization_content += "<p>• Exceeds downtime SLA - consider incremental migration</p>"
        else:
            optimization_content += "<p>• Meets downtime requirements</p>"
        
        st.markdown(f"""
        <div class="corporate-card status-card-warning">
            <h3>⚠️ Considerations & Limitations</h3>
            {cons_content}
        </div>
        """, unsafe_allow_html=True)
        
        optimization_card_type = "status-card-error" if best_pattern.migration_time_hours > config['max_downtime_hours'] else "status-card-success"
        
        st.markdown(f"""
        <div class="corporate-card {optimization_card_type}">
            <h3>💡 Optimization Recommendations</h3>
            {optimization_content}
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point"""
    render_corporate_header()
    
    # Sidebar configuration
    config = render_enhanced_sidebar_controls()
    
    # Initialize analyzer
    analyzer = EnhancedNetworkAnalyzer()
    
    # Initialize API clients with credentials from secrets
    if config.get('aws_access_key') and config.get('aws_secret_key'):
        analyzer.pricing_client.initialize_client(config['aws_access_key'], config['aws_secret_key'])
    
    if config.get('claude_api_key'):
        analyzer.ai_client.api_key = config['claude_api_key']
    
    # UPDATED: 6 TABS including the new Network Path Diagram
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "💧 Infrastructure Analysis",    # Enhanced version of original Realistic Analysis
        "⏱️ Migration Analysis",         # Enhanced version of original Migration Analysis 
        "🤖 AI Recommendations",         # Enhanced version of original AI Recommendations
        "🔄 Pattern Comparison",         # Enhanced comparison with corporate styling
        "🌐 Network Architecture",       # NEW - Network Path Diagram
        "📚 Database Guide"              # Database engineer guidance
    ])
    
    with tab1:
        waterfall_data = render_realistic_analysis_tab(config, analyzer)
    
    with tab2:
        if 'waterfall_data' in locals():
            analysis_results = render_migration_analysis_tab(config, waterfall_data, analyzer)
        else:
            st.markdown("""
            <div class="corporate-card status-card-warning">
                <h3>⚠️ Analysis Required</h3>
                <p>Please run Infrastructure Analysis first to proceed with Migration Analysis.</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab3:
        if 'analysis_results' in locals():
            render_ai_recommendations_tab(config, analysis_results, analyzer)
        else:
            st.markdown("""
            <div class="corporate-card status-card-warning">
                <h3>⚠️ Previous Analysis Required</h3>
                <p>Please complete Infrastructure and Migration Analysis first to get AI recommendations.</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab4:
        pattern_analyses, ai_recommendation = render_pattern_comparison_tab(analyzer, config)
        
        # Store results in session state for use in other tabs
        if pattern_analyses and ai_recommendation:
            st.session_state['pattern_analyses'] = pattern_analyses
            st.session_state['ai_recommendation'] = ai_recommendation
    
    with tab5:
        # NEW TAB: Network Architecture Diagram
        render_network_path_diagram_tab()
    
    with tab6:
        render_database_guidance_tab(config)
        
        # Optional: Show AI insights if pattern comparison has been run
        if 'pattern_analyses' in st.session_state and 'ai_recommendation' in st.session_state:
            st.markdown("---")
            render_ai_insights_tab(
                st.session_state['pattern_analyses'], 
                st.session_state['ai_recommendation'], 
                config
            )

if __name__ == "__main__":
    main()