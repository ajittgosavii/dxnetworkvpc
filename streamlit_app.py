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

@dataclass
class NetworkConditions:
    """Data class for real-time network conditions"""
    time_of_day: str
    congestion_factor: float
    jitter_ms: float
    packet_loss_rate: float
    burst_capacity_factor: float

@dataclass
class VMwareEnvironment:
    """Data class for VMware environment characteristics"""
    storage_type: str
    sr_iov_enabled: bool
    memory_overcommit_ratio: float
    cpu_overcommit_ratio: float
    numa_optimization: bool
    host_count: int

# =============================================================================
# ENHANCED AWS PRICING CLIENT
# =============================================================================

class AWSPricingClient:
    """Enhanced AWS Pricing API client for real-time cost data"""
    
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
        """Get Direct Connect pricing with correct service code"""
        if not self.pricing_client:
            return self._get_mock_dx_pricing(port_speed)
        
        try:
            # FIXED: Correct service code for Direct Connect
            response = self.pricing_client.get_products(
                ServiceCode='AWSDirectConnect',  # CORRECTED from 'AmazonConnect'
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
    
    def get_vpc_endpoint_pricing(self, region: str = 'US East (N. Virginia)') -> Dict:
        """Get VPC Endpoint pricing"""
        if not self.pricing_client:
            return {'hour': 0.01, 'data_gb': 0.01}
        
        try:
            response = self.pricing_client.get_products(
                ServiceCode='AmazonVPC',
                Filters=[
                    {'Type': 'TERM_MATCH', 'Field': 'location', 'Value': region},
                    {'Type': 'TERM_MATCH', 'Field': 'productFamily', 'Value': 'VpcEndpoint'}
                ]
            )
            
            if response['PriceList']:
                price_data = json.loads(response['PriceList'][0])
                return self._parse_vpc_endpoint_pricing(price_data)
            else:
                return {'hour': 0.01, 'data_gb': 0.01}
                
        except Exception as e:
            st.warning(f"Error fetching VPC Endpoint pricing: {e}")
            return {'hour': 0.01, 'data_gb': 0.01}
    
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
    
    def _parse_dx_pricing(self, price_data: Dict) -> Dict:
        """Parse Direct Connect pricing data"""
        try:
            terms = price_data['terms']['OnDemand']
            term_key = list(terms.keys())[0]
            price_dims = terms[term_key]['priceDimensions']
            
            port_hour = 0
            data_transfer_gb = 0
            
            for dim in price_dims.values():
                if 'Port' in dim['description']:
                    port_hour = float(dim['pricePerUnit']['USD'])
                elif 'Data Transfer' in dim['description']:
                    data_transfer_gb = float(dim['pricePerUnit']['USD'])
            
            return {'port_hour': port_hour, 'data_transfer_gb': data_transfer_gb}
        except:
            # Fallback to port speed-based pricing
            port_speed = price_data.get('product', {}).get('attributes', {}).get('portSpeed', '10Gbps')
            return self._get_mock_dx_pricing(port_speed)
    
    def _parse_vpc_endpoint_pricing(self, price_data: Dict) -> Dict:
        """Parse VPC Endpoint pricing data"""
        try:
            terms = price_data['terms']['OnDemand']
            term_key = list(terms.keys())[0]
            price_dims = terms[term_key]['priceDimensions']
            
            hour_price = 0
            data_price = 0
            
            for dim in price_dims.values():
                if 'hour' in dim['unit'].lower():
                    hour_price = float(dim['pricePerUnit']['USD'])
                elif 'gb' in dim['unit'].lower():
                    data_price = float(dim['pricePerUnit']['USD'])
            
            return {'hour': hour_price, 'data_gb': data_price}
        except:
            return {'hour': 0.01, 'data_gb': 0.01}
    
    def _parse_ec2_pricing(self, price_data: Dict) -> Dict:
        """Parse EC2 pricing data"""
        try:
            terms = price_data['terms']['OnDemand']
            term_key = list(terms.keys())[0]
            price_dims = terms[term_key]['priceDimensions']
            dim_key = list(price_dims.keys())[0]
            hourly_price = float(price_dims[dim_key]['pricePerUnit']['USD'])
            
            return {'hourly': hourly_price}
        except:
            instance_type = price_data.get('product', {}).get('attributes', {}).get('instanceType', 'm5.xlarge')
            return self._get_mock_ec2_pricing(instance_type)
    
    def _get_mock_dx_pricing(self, port_speed: str) -> Dict:
        """Enhanced mock Direct Connect pricing with regional variations"""
        base_pricing = {
            '1Gbps': {'port_hour': 0.30, 'data_transfer_gb': 0.02},
            '10Gbps': {'port_hour': 2.25, 'data_transfer_gb': 0.02},
            '100Gbps': {'port_hour': 22.50, 'data_transfer_gb': 0.015}
        }
        
        # Add regional pricing variations
        regional_multiplier = 1.0  # US East baseline
        
        return base_pricing.get(port_speed, base_pricing['10Gbps'])
    
    def _get_mock_ec2_pricing(self, instance_type: str) -> Dict:
        """Enhanced mock EC2 pricing data"""
        pricing_map = {
            'm5.large': {'hourly': 0.096},
            'm5.xlarge': {'hourly': 0.192},
            'm5.2xlarge': {'hourly': 0.384},
            'm5.4xlarge': {'hourly': 0.768},
            'm5.8xlarge': {'hourly': 1.536},
            'c5.xlarge': {'hourly': 0.17},
            'c5.2xlarge': {'hourly': 0.34},
            'c5.4xlarge': {'hourly': 0.68},
            'r5.xlarge': {'hourly': 0.252},
            'r5.2xlarge': {'hourly': 0.504},
            'dms.t3.medium': {'hourly': 0.0464},
            'dms.r5.large': {'hourly': 0.144},
            'dms.r5.xlarge': {'hourly': 0.288},
            'dms.r5.2xlarge': {'hourly': 0.576}
        }
        return pricing_map.get(instance_type, pricing_map['m5.xlarge'])

# =============================================================================
# ENHANCED CLAUDE AI CLIENT
# =============================================================================

class ClaudeAIClient:
    """Enhanced Claude AI client for intelligent recommendations"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.base_url = "https://api.anthropic.com/v1/messages"
        
    def get_pattern_recommendation(self, analysis_data: Dict) -> Dict:
        """Get AI recommendation for best migration pattern"""
        if not self.api_key:
            return self._get_mock_ai_recommendation(analysis_data)
        
        try:
            prompt = self._build_enhanced_analysis_prompt(analysis_data)
            
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
    
    def _build_enhanced_analysis_prompt(self, analysis_data: Dict) -> str:
        """Build enhanced analysis prompt for Claude AI"""
        return f"""
        You are an AWS network architecture expert with deep knowledge of database migrations and enterprise infrastructure.
        
        Analysis Data:
        - Data Size: {analysis_data.get('data_size_gb', 0)} GB
        - Database Type: {analysis_data.get('database_scenario', 'unknown')}
        - Migration Service: {analysis_data.get('migration_service', 'unknown')}
        - Environment: {analysis_data.get('environment', 'unknown')}
        - Max Downtime: {analysis_data.get('max_downtime_hours', 0)} hours
        - Source Location: {analysis_data.get('source_location', 'unknown')}
        - Network Patterns Available: {len(analysis_data.get('patterns', []))}
        
        Available Patterns with Performance Data:
        {json.dumps(analysis_data.get('patterns', []), indent=2)}
        
        Consider these factors:
        1. Database-specific latency and consistency requirements
        2. Real-world network congestion and time-of-day variations
        3. VMware virtualization overhead for DataSync deployments
        4. Cost optimization including reserved vs on-demand pricing
        5. Migration complexity and risk assessment
        6. Compliance and security requirements
        7. Disaster recovery and business continuity needs
        
        Provide detailed analysis for database migration including:
        - Network pattern recommendation with justification
        - Database-specific considerations (replication lag, consistency, performance)
        - Risk mitigation strategies
        - Cost-benefit analysis
        - Implementation timeline and phases
        
        Respond in JSON format with:
        {{
            "recommended_pattern": "pattern_name",
            "confidence_score": 0.85,
            "reasoning": "comprehensive explanation",
            "cost_justification": "detailed cost analysis",
            "risk_assessment": "potential risks and mitigations",
            "database_considerations": "database-specific recommendations",
            "implementation_phases": ["phase1", "phase2", "phase3"],
            "timeline_estimate": "detailed timeline",
            "performance_expectations": "expected performance characteristics"
        }}
        """
    
    def _parse_ai_response(self, response_text: str) -> Dict:
        """Parse Claude AI response with enhanced error handling"""
        try:
            # Try to extract JSON from response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                parsed = json.loads(json_str)
                
                # Validate required fields
                required_fields = ['recommended_pattern', 'confidence_score', 'reasoning']
                if all(field in parsed for field in required_fields):
                    return parsed
        except:
            pass
        
        # Enhanced fallback response
        return {
            "recommended_pattern": "direct_connect",
            "confidence_score": 0.75,
            "reasoning": "Based on comprehensive analysis, Direct Connect provides the optimal balance of performance, reliability, and cost for enterprise database migrations",
            "cost_justification": "Higher infrastructure cost is offset by reduced migration time, improved reliability, and lower operational risk",
            "risk_assessment": "Low risk with proper planning, testing, and phased implementation approach",
            "database_considerations": "Direct Connect provides consistent latency and dedicated bandwidth crucial for database replication and migration integrity",
            "implementation_phases": ["Assessment & Planning", "Infrastructure Setup", "Testing & Validation", "Migration Execution", "Optimization"],
            "timeline_estimate": "4-6 weeks total with 2 weeks for setup, 1 week testing, 1-2 weeks migration",
            "performance_expectations": "Consistent sub-10ms latency with dedicated bandwidth allocation"
        }
    
    def _get_mock_ai_recommendation(self, analysis_data: Dict) -> Dict:
        """Enhanced mock AI recommendation with comprehensive analysis"""
        data_size = analysis_data.get('data_size_gb', 0)
        environment = analysis_data.get('environment', 'non-production')
        db_scenario = analysis_data.get('database_scenario', 'mysql_oltp_rds')
        
        # Enhanced decision logic
        if environment == 'production' and data_size > 500:
            pattern = "direct_connect"
            reasoning = "Production environment with large dataset requires dedicated, reliable connectivity for enterprise SLA compliance"
            phases = ["Infrastructure Assessment", "Direct Connect Provisioning", "Testing Phase", "Migration Execution", "Post-Migration Optimization"]
        elif data_size < 100 and 'olap' in db_scenario:
            pattern = "vpc_endpoint"
            reasoning = "Small analytical dataset can efficiently leverage VPC endpoint with cost optimization"
            phases = ["VPC Endpoint Setup", "Testing Phase", "Migration Execution"]
        else:
            pattern = "direct_connect"
            reasoning = "Medium to large dataset with database requirements benefits from dedicated bandwidth and consistent latency"
            phases = ["Planning Phase", "Infrastructure Setup", "Migration Execution", "Validation"]
        
        return {
            "recommended_pattern": pattern,
            "confidence_score": 0.82,
            "reasoning": reasoning,
            "cost_justification": "Optimized for data size, environment requirements, and database-specific needs",
            "risk_assessment": "Standard enterprise risk mitigation with comprehensive testing approach",
            "database_considerations": "Ensuring optimal replication performance and data consistency throughout migration",
            "implementation_phases": phases,
            "timeline_estimate": f"Estimated {len(phases)} phase approach over 3-8 weeks depending on complexity",
            "performance_expectations": "Consistent performance meeting database SLA requirements"
        }

# =============================================================================
# ENHANCED NETWORK ANALYZER CLASS
# =============================================================================

class EnhancedNetworkAnalyzer:
    """Comprehensive network analyzer with realistic infrastructure modeling, real-time conditions, and enhanced calculations"""
    
    def __init__(self):
        # Operating System Network Stack Characteristics (Enhanced)
        self.os_characteristics = {
            'windows_server_2019': {
                'name': 'Windows Server 2019',
                'tcp_stack_efficiency': 0.94,
                'memory_copy_overhead': 0.08,
                'interrupt_overhead': 0.05,
                'kernel_bypass_support': False,
                'max_tcp_window_size': '64KB',
                'rss_support': True,
                'network_virtualization_overhead': 0.12,
                'numa_awareness': False,
                'hardware_acceleration': False
            },
            'windows_server_2022': {
                'name': 'Windows Server 2022',
                'tcp_stack_efficiency': 0.96,
                'memory_copy_overhead': 0.06,
                'interrupt_overhead': 0.04,
                'kernel_bypass_support': True,
                'max_tcp_window_size': '1MB',
                'rss_support': True,
                'network_virtualization_overhead': 0.08,
                'numa_awareness': True,
                'hardware_acceleration': True
            },
            'linux_rhel8': {
                'name': 'Red Hat Enterprise Linux 8',
                'tcp_stack_efficiency': 0.97,
                'memory_copy_overhead': 0.04,
                'interrupt_overhead': 0.03,
                'kernel_bypass_support': True,
                'max_tcp_window_size': '16MB',
                'rss_support': True,
                'network_virtualization_overhead': 0.05,
                'numa_awareness': True,
                'hardware_acceleration': True
            },
            'linux_ubuntu': {
                'name': 'Ubuntu Linux (Latest)',
                'tcp_stack_efficiency': 0.98,
                'memory_copy_overhead': 0.03,
                'interrupt_overhead': 0.02,
                'kernel_bypass_support': True,
                'max_tcp_window_size': '16MB',
                'rss_support': True,
                'network_virtualization_overhead': 0.04,
                'numa_awareness': True,
                'hardware_acceleration': True
            }
        }
        
        # Network Interface Card Characteristics (Enhanced)
        self.nic_characteristics = {
            '1gbps_standard': {
                'name': '1 Gbps Standard NIC',
                'theoretical_bandwidth_mbps': 1000,
                'real_world_efficiency': 0.94,
                'base_cpu_utilization_per_gbps': 0.15,
                'pcie_gen': '2.0',
                'pcie_lanes': 1,
                'hardware_offload_support': ['checksum'],
                'mtu_support': 1500,
                'buffer_size_mb': 1,
                'interrupt_coalescing': False,
                'sr_iov_support': False,
                'rdma_support': False
            },
            '10gbps_standard': {
                'name': '10 Gbps Standard NIC',
                'theoretical_bandwidth_mbps': 10000,
                'real_world_efficiency': 0.92,
                'base_cpu_utilization_per_gbps': 0.08,
                'pcie_gen': '3.0',
                'pcie_lanes': 4,
                'hardware_offload_support': ['checksum', 'segmentation', 'rss'],
                'mtu_support': 9000,
                'buffer_size_mb': 4,
                'interrupt_coalescing': True,
                'sr_iov_support': True,
                'rdma_support': False
            },
            '25gbps_high_performance': {
                'name': '25 Gbps High-Performance NIC',
                'theoretical_bandwidth_mbps': 25000,
                'real_world_efficiency': 0.96,
                'base_cpu_utilization_per_gbps': 0.04,
                'pcie_gen': '3.0',
                'pcie_lanes': 8,
                'hardware_offload_support': ['checksum', 'segmentation', 'rss', 'rdma'],
                'mtu_support': 9000,
                'buffer_size_mb': 16,
                'interrupt_coalescing': True,
                'sr_iov_support': True,
                'rdma_support': True
            },
            '100gbps_enterprise': {
                'name': '100 Gbps Enterprise NIC',
                'theoretical_bandwidth_mbps': 100000,
                'real_world_efficiency': 0.98,
                'base_cpu_utilization_per_gbps': 0.02,
                'pcie_gen': '4.0',
                'pcie_lanes': 16,
                'hardware_offload_support': ['checksum', 'segmentation', 'rss', 'rdma', 'encryption'],
                'mtu_support': 9000,
                'buffer_size_mb': 64,
                'interrupt_coalescing': True,
                'sr_iov_support': True,
                'rdma_support': True
            }
        }
        
        # LAN Infrastructure Characteristics (Enhanced)
        self.lan_characteristics = {
            'gigabit_switch': {
                'name': 'Gigabit Ethernet Switch',
                'switching_capacity_gbps': 48,
                'port_buffer_mb': 12,
                'switching_latency_us': 5,
                'oversubscription_ratio': '3:1',
                'congestion_threshold': 0.8,
                'qos_support': True,
                'vlan_overhead': 0.01,
                'packet_loss_at_congestion': 0.001
            },
            '10gb_switch': {
                'name': '10 Gigabit Ethernet Switch',
                'switching_capacity_gbps': 480,
                'port_buffer_mb': 48,
                'switching_latency_us': 2,
                'oversubscription_ratio': '2:1',
                'congestion_threshold': 0.85,
                'qos_support': True,
                'vlan_overhead': 0.005,
                'packet_loss_at_congestion': 0.0005
            },
            '25gb_switch': {
                'name': '25 Gigabit Ethernet Switch',
                'switching_capacity_gbps': 1200,
                'port_buffer_mb': 128,
                'switching_latency_us': 1,
                'oversubscription_ratio': '1.5:1',
                'congestion_threshold': 0.9,
                'qos_support': True,
                'vlan_overhead': 0.003,
                'packet_loss_at_congestion': 0.0002
            },
            'spine_leaf_fabric': {
                'name': 'Spine-Leaf Fabric',
                'switching_capacity_gbps': 5000,
                'port_buffer_mb': 256,
                'switching_latency_us': 0.5,
                'oversubscription_ratio': '1:1',
                'congestion_threshold': 0.95,
                'qos_support': True,
                'vlan_overhead': 0.001,
                'packet_loss_at_congestion': 0.0001
            }
        }
        
        # WAN Provider Characteristics (Enhanced)
        self.wan_characteristics = {
            'mpls_tier1': {
                'name': 'MPLS Tier-1 Provider',
                'bandwidth_efficiency': 0.96,
                'latency_consistency': 0.98,
                'base_packet_loss_rate': 0.0001,
                'base_jitter_ms': 2,
                'burstable_overhead': 0.1,
                'qos_classes': 4,
                'sla_availability': 0.9999,
                'congestion_sensitivity': 0.3
            },
            'fiber_metro': {
                'name': 'Metro Fiber Ethernet',
                'bandwidth_efficiency': 0.98,
                'latency_consistency': 0.99,
                'base_packet_loss_rate': 0.00005,
                'base_jitter_ms': 1,
                'burstable_overhead': 0.05,
                'qos_classes': 8,
                'sla_availability': 0.99995,
                'congestion_sensitivity': 0.1
            },
            'internet_transit': {
                'name': 'Internet Transit',
                'bandwidth_efficiency': 0.85,
                'latency_consistency': 0.9,
                'base_packet_loss_rate': 0.001,
                'base_jitter_ms': 10,
                'burstable_overhead': 0.2,
                'qos_classes': 0,
                'sla_availability': 0.999,
                'congestion_sensitivity': 0.8
            },
            'aws_dx_dedicated': {
                'name': 'AWS Direct Connect Dedicated',
                'bandwidth_efficiency': 0.99,
                'latency_consistency': 0.999,
                'base_packet_loss_rate': 0.00001,
                'base_jitter_ms': 0.5,
                'burstable_overhead': 0.02,
                'qos_classes': 8,
                'sla_availability': 0.9999,
                'congestion_sensitivity': 0.05
            }
        }
        
        # AWS Direct Connect Specific Factors (Enhanced)
        self.dx_characteristics = {
            '1gbps_dedicated': {
                'name': '1 Gbps Dedicated Connection',
                'committed_bandwidth_mbps': 1000,
                'burst_capability': 1.0,
                'aws_edge_processing_overhead': 0.02,
                'cross_connect_latency_ms': 1,
                'bgp_convergence_impact': 0.01,
                'virtual_interface_overhead': 0.005,
                'redundancy_available': True
            },
            '10gbps_dedicated': {
                'name': '10 Gbps Dedicated Connection',
                'committed_bandwidth_mbps': 10000,
                'burst_capability': 1.0,
                'aws_edge_processing_overhead': 0.01,
                'cross_connect_latency_ms': 0.8,
                'bgp_convergence_impact': 0.005,
                'virtual_interface_overhead': 0.003,
                'redundancy_available': True
            },
            '100gbps_dedicated': {
                'name': '100 Gbps Dedicated Connection',
                'committed_bandwidth_mbps': 100000,
                'burst_capability': 1.0,
                'aws_edge_processing_overhead': 0.005,
                'cross_connect_latency_ms': 0.5,
                'bgp_convergence_impact': 0.002,
                'virtual_interface_overhead': 0.001,
                'redundancy_available': True
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
        
        # Complete Database Scenarios Dictionary (Enhanced)
        self.database_scenarios = {
            'mysql_oltp_rds': {
                'name': 'MySQL OLTP → RDS MySQL',
                'workload_type': 'oltp',
                'aws_target': 'rds',
                'target_service': 'Amazon RDS for MySQL',
                'latency_sensitivity': 'high',
                'bandwidth_requirement': 'medium',
                'consistency_requirement': 'strict',
                'recommended_services': ['dms'],
                'min_bandwidth_mbps': 500,
                'max_tolerable_latency_ms': 10,
                'max_tolerable_jitter_ms': 2,
                'max_packet_loss_rate': 0.0001,
                'migration_complexity': 'low',
                'downtime_sensitivity': 'high',
                'burst_requirement_factor': 1.5,
                'concurrent_connection_limit': 1000
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
                'max_tolerable_jitter_ms': 10,
                'max_packet_loss_rate': 0.001,
                'migration_complexity': 'medium',
                'downtime_sensitivity': 'medium',
                'burst_requirement_factor': 2.0,
                'concurrent_connection_limit': 500
            },
            'oracle_enterprise_rds': {
                'name': 'Oracle Enterprise → RDS Oracle',
                'workload_type': 'oltp',
                'aws_target': 'rds',
                'target_service': 'Amazon RDS for Oracle',
                'latency_sensitivity': 'high',
                'bandwidth_requirement': 'high',
                'consistency_requirement': 'strict',
                'recommended_services': ['dms'],
                'min_bandwidth_mbps': 2000,
                'max_tolerable_latency_ms': 5,
                'max_tolerable_jitter_ms': 1,
                'max_packet_loss_rate': 0.00005,
                'migration_complexity': 'high',
                'downtime_sensitivity': 'high',
                'burst_requirement_factor': 1.8,
                'concurrent_connection_limit': 2000
            },
            'sqlserver_enterprise_ec2': {
                'name': 'SQL Server Enterprise → EC2',
                'workload_type': 'oltp',
                'aws_target': 'ec2',
                'target_service': 'SQL Server on EC2',
                'latency_sensitivity': 'high',
                'bandwidth_requirement': 'high',
                'consistency_requirement': 'strict',
                'recommended_services': ['dms', 'datasync'],
                'min_bandwidth_mbps': 1500,
                'max_tolerable_latency_ms': 8,
                'max_tolerable_jitter_ms': 2,
                'max_packet_loss_rate': 0.0001,
                'migration_complexity': 'high',
                'downtime_sensitivity': 'high',
                'burst_requirement_factor': 1.6,
                'concurrent_connection_limit': 1500
            },
            'mongodb_cluster_documentdb': {
                'name': 'MongoDB Cluster → DocumentDB',
                'workload_type': 'oltp',
                'aws_target': 'documentdb',
                'target_service': 'Amazon DocumentDB',
                'latency_sensitivity': 'medium',
                'bandwidth_requirement': 'high',
                'consistency_requirement': 'configurable',
                'recommended_services': ['dms'],
                'min_bandwidth_mbps': 1500,
                'max_tolerable_latency_ms': 20,
                'max_tolerable_jitter_ms': 5,
                'max_packet_loss_rate': 0.0005,
                'migration_complexity': 'medium',
                'downtime_sensitivity': 'medium',
                'burst_requirement_factor': 1.4,
                'concurrent_connection_limit': 1000
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
                'max_tolerable_jitter_ms': 8,
                'max_packet_loss_rate': 0.0008,
                'migration_complexity': 'medium',
                'downtime_sensitivity': 'low',
                'burst_requirement_factor': 2.2,
                'concurrent_connection_limit': 800
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
                'max_tolerable_jitter_ms': 3,
                'max_packet_loss_rate': 0.0002,
                'migration_complexity': 'low',
                'downtime_sensitivity': 'high',
                'burst_requirement_factor': 1.3,
                'concurrent_connection_limit': 1200
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
                'max_tolerable_jitter_ms': 2,
                'max_packet_loss_rate': 0.0001,
                'migration_complexity': 'low',
                'downtime_sensitivity': 'high',
                'burst_requirement_factor': 1.4,
                'concurrent_connection_limit': 1000
            }
        }
        
        # Comprehensive Migration Services (Enhanced with error handling)
        self.migration_services = {
            'datasync': {
                'name': 'AWS DataSync',
                'use_case': 'File and object data transfer',
                'protocols': ['NFS', 'SMB', 'HDFS', 'S3'],
                'vpc_endpoint_compatible': True,
                'encryption_in_transit': True,
                'encryption_at_rest': True,
                'base_application_efficiency': 0.92,
                'base_protocol_efficiency': 0.96,
                'latency_sensitivity': 'medium',
                'tcp_window_scaling_required': True,
                'vmware_deployment': True,
                'error_retry_factor': 0.95,  # 5% overhead for retries
                'checksum_verification_overhead': 0.02,
                'database_compatibility': {
                    'file_based_backups': True,
                    'live_replication': False,
                    'transaction_logs': True
                },
                'sizes': {
                    'small': {
                        'vcpu': 4, 'memory_gb': 16, 'base_throughput_mbps': 400, 'cost_per_hour': 0.084,
                        'vpc_endpoint_throughput_reduction': 0.1,
                        'optimal_file_size_mb': '1-100',
                        'concurrent_transfers': 16,
                        'tcp_connections': 16,
                        'instance_type': 'm5.xlarge',
                        'vmware_overhead': 0.15,
                        'storage_overhead': 0.05,
                        'network_overhead': 0.08
                    },
                    'medium': {
                        'vcpu': 8, 'memory_gb': 32, 'base_throughput_mbps': 1000, 'cost_per_hour': 0.168,
                        'vpc_endpoint_throughput_reduction': 0.08,
                        'optimal_file_size_mb': '100-1000',
                        'concurrent_transfers': 32,
                        'tcp_connections': 32,
                        'instance_type': 'm5.2xlarge',
                        'vmware_overhead': 0.12,
                        'storage_overhead': 0.04,
                        'network_overhead': 0.06
                    },
                    'large': {
                        'vcpu': 16, 'memory_gb': 64, 'base_throughput_mbps': 2000, 'cost_per_hour': 0.336,
                        'vpc_endpoint_throughput_reduction': 0.05,
                        'optimal_file_size_mb': '1000+',
                        'concurrent_transfers': 64,
                        'tcp_connections': 64,
                        'instance_type': 'm5.4xlarge',
                        'vmware_overhead': 0.10,
                        'storage_overhead': 0.03,
                        'network_overhead': 0.05
                    },
                    'xlarge': {
                        'vcpu': 32, 'memory_gb': 128, 'base_throughput_mbps': 4000, 'cost_per_hour': 0.672,
                        'vpc_endpoint_throughput_reduction': 0.03,
                        'optimal_file_size_mb': '1000+',
                        'concurrent_transfers': 128,
                        'tcp_connections': 128,
                        'instance_type': 'm5.8xlarge',
                        'vmware_overhead': 0.08,
                        'storage_overhead': 0.02,
                        'network_overhead': 0.04
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
                'base_application_efficiency': 0.88,
                'base_protocol_efficiency': 0.94,
                'latency_sensitivity': 'high',
                'requires_endpoints': True,
                'supports_cdc': True,
                'tcp_window_scaling_required': True,
                'error_retry_factor': 0.92,  # 8% overhead for retries and validation
                'cdc_overhead': 0.05,
                'database_compatibility': {
                    'file_based_backups': False,
                    'live_replication': True,
                    'transaction_logs': True,
                    'schema_conversion': True
                },
                'sizes': {
                    'small': {
                        'vcpu': 2, 'memory_gb': 4, 'base_throughput_mbps': 200, 'cost_per_hour': 0.042,
                        'max_connections': 50,
                        'optimal_table_size_gb': '1-10',
                        'tcp_connections': 4,
                        'instance_type': 'dms.t3.medium'
                    },
                    'medium': {
                        'vcpu': 2, 'memory_gb': 8, 'base_throughput_mbps': 400, 'cost_per_hour': 0.085,
                        'max_connections': 100,
                        'optimal_table_size_gb': '10-100',
                        'tcp_connections': 8,
                        'instance_type': 'dms.r5.large'
                    },
                    'large': {
                        'vcpu': 4, 'memory_gb': 16, 'base_throughput_mbps': 800, 'cost_per_hour': 0.17,
                        'max_connections': 200,
                        'optimal_table_size_gb': '100-500',
                        'tcp_connections': 16,
                        'instance_type': 'dms.r5.xlarge'
                    },
                    'xlarge': {
                        'vcpu': 8, 'memory_gb': 32, 'base_throughput_mbps': 1500, 'cost_per_hour': 0.34,
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
                'base_application_efficiency': 0.95,
                'base_protocol_efficiency': 0.93,
                'latency_sensitivity': 'low',
                'requires_active_directory': True,
                'supports_deduplication': True,
                'tcp_window_scaling_required': False,
                'error_retry_factor': 0.97,
                'database_compatibility': {
                    'file_based_backups': True,
                    'live_replication': False,
                    'transaction_logs': False
                },
                'sizes': {
                    'small': {
                        'storage_gb': 32, 'base_throughput_mbps': 16, 'cost_per_hour': 0.013,
                        'iops': 96, 'max_concurrent_users': 50
                    },
                    'medium': {
                        'storage_gb': 64, 'base_throughput_mbps': 32, 'cost_per_hour': 0.025,
                        'iops': 192, 'max_concurrent_users': 100
                    },
                    'large': {
                        'storage_gb': 2048, 'base_throughput_mbps': 512, 'cost_per_hour': 0.40,
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
                'base_application_efficiency': 0.98,
                'base_protocol_efficiency': 0.97,
                'latency_sensitivity': 'very_low',
                'supports_s3_integration': True,
                'tcp_window_scaling_required': False,
                'error_retry_factor': 0.98,
                'database_compatibility': {
                    'file_based_backups': True,
                    'live_replication': False,
                    'transaction_logs': False
                },
                'sizes': {
                    'small': {
                        'storage_gb': 1200, 'base_throughput_mbps': 240, 'cost_per_hour': 0.15,
                        'iops': 'unlimited', 'max_concurrent_clients': 100
                    },
                    'large': {
                        'storage_gb': 7200, 'base_throughput_mbps': 1440, 'cost_per_hour': 0.90,
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
                'base_application_efficiency': 0.85,
                'base_protocol_efficiency': 0.92,
                'latency_sensitivity': 'medium',
                'supports_caching': True,
                'tcp_window_scaling_required': True,
                'error_retry_factor': 0.90,  # Higher retry overhead due to caching
                'cache_efficiency': 0.8,
                'database_compatibility': {
                    'file_based_backups': True,
                    'live_replication': False,
                    'transaction_logs': True
                },
                'sizes': {
                    'small': {
                        'vcpu': 4, 'memory_gb': 16, 'base_throughput_mbps': 125, 'cost_per_hour': 0.05,
                        'cache_gb': 150, 'max_volumes': 32,
                        'instance_type': 'm5.xlarge'
                    },
                    'large': {
                        'vcpu': 16, 'memory_gb': 64, 'base_throughput_mbps': 500, 'cost_per_hour': 0.20,
                        'cache_gb': 600, 'max_volumes': 128,
                        'instance_type': 'm5.4xlarge'
                    }
                }
            }
        }
        
        # Initialize enhanced clients
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
    
    def get_network_conditions(self, time_of_day: str = 'business_hours') -> NetworkConditions:
        """Get current network conditions with time-of-day variations"""
        if time_of_day == 'business_hours':
            return NetworkConditions(
                time_of_day=time_of_day,
                congestion_factor=0.7,  # 30% reduction during peak
                jitter_ms=2.0,
                packet_loss_rate=0.0002,
                burst_capacity_factor=0.8
            )
        elif time_of_day == 'off_hours':
            return NetworkConditions(
                time_of_day=time_of_day,
                congestion_factor=1.0,  # No congestion
                jitter_ms=0.5,
                packet_loss_rate=0.00005,
                burst_capacity_factor=1.2
            )
        else:  # 'weekend'
            return NetworkConditions(
                time_of_day=time_of_day,
                congestion_factor=0.95,  # Minimal congestion
                jitter_ms=0.8,
                packet_loss_rate=0.0001,
                burst_capacity_factor=1.1
            )
    
    def calculate_cpu_impact(self, bandwidth_gbps: float, nic_char: Dict, os_char: Dict) -> float:
        """Enhanced CPU impact calculation with hardware acceleration"""
        base_cpu_per_gbps = nic_char['base_cpu_utilization_per_gbps']
        
        # Account for hardware offloading
        if 'rss' in nic_char.get('hardware_offload_support', []):
            base_cpu_per_gbps *= 0.7
        if 'rdma' in nic_char.get('hardware_offload_support', []):
            base_cpu_per_gbps *= 0.4
        if nic_char.get('sr_iov_support'):
            base_cpu_per_gbps *= 0.8
        
        # OS-level optimizations
        if os_char.get('hardware_acceleration'):
            base_cpu_per_gbps *= 0.9
        if os_char.get('numa_awareness'):
            base_cpu_per_gbps *= 0.95
        
        cpu_utilization = bandwidth_gbps * base_cpu_per_gbps
        
        # Non-linear impact curve
        if cpu_utilization < 0.4:
            return 1.0
        elif cpu_utilization < 0.7:
            return 1.0 - ((cpu_utilization - 0.4) * 0.5)
        else:
            return max(0.5, 0.85 - ((cpu_utilization - 0.7) * 2))
    
    def calculate_datasync_vmware_performance(self, vm_config: Dict, vmware_env: VMwareEnvironment) -> Dict:
        """Calculate DataSync performance with realistic VMware overhead"""
        base_throughput = vm_config['base_throughput_mbps']
        
        # Storage overhead
        storage_overhead = vm_config.get('storage_overhead', 0.05)
        if vmware_env.storage_type == 'ssd':
            storage_overhead *= 0.5
        
        # Network virtualization overhead
        network_overhead = vm_config.get('network_overhead', 0.15)
        if vmware_env.sr_iov_enabled:
            network_overhead *= 0.5
        
        # Memory overcommit impact
        memory_overhead = max(0, (vmware_env.memory_overcommit_ratio - 1.0) * 0.1)
        
        # CPU overcommit impact
        cpu_overhead = max(0, (vmware_env.cpu_overcommit_ratio - 1.0) * 0.15)
        
        # NUMA optimization bonus
        numa_bonus = 0.05 if vmware_env.numa_optimization else 0
        
        total_overhead = storage_overhead + network_overhead + memory_overhead + cpu_overhead - numa_bonus
        effective_throughput = base_throughput * (1 - total_overhead)
        
        return {
            'effective_throughput_mbps': effective_throughput,
            'total_overhead_percent': total_overhead * 100,
            'breakdown': {
                'storage': storage_overhead * 100,
                'network': network_overhead * 100,
                'memory': memory_overhead * 100,
                'cpu': cpu_overhead * 100,
                'numa_bonus': numa_bonus * 100
            }
        }
    
    def calculate_end_to_end_latency(self, pattern_key: str, network_conditions: NetworkConditions) -> Dict:
        """Calculate comprehensive end-to-end latency"""
        pattern = self.network_patterns[pattern_key]
        
        # Base latencies
        base_latency = pattern['baseline_latency_ms']
        
        # Processing delays
        os_char = self.os_characteristics[pattern['os_type']]
        nic_char = self.nic_characteristics[pattern['nic_type']]
        lan_char = self.lan_characteristics[pattern['lan_type']]
        
        os_processing = 0.1
        if os_char.get('hardware_acceleration'):
            os_processing *= 0.7
        
        nic_processing = 0.05
        if nic_char.get('hardware_offload_support'):
            nic_processing *= 0.8
        
        switch_latency = lan_char['switching_latency_us'] / 1000
        
        # WAN propagation (distance-based)
        if pattern['source'] == 'San Antonio':
            sa_to_sj_latency = 15  # ~1000 miles
            sj_to_aws_latency = 8   # ~400 miles  
            wan_latency = sa_to_sj_latency + sj_to_aws_latency
        else:
            wan_latency = 8  # San Jose to AWS
        
        # Apply network conditions
        wan_latency *= (1 + network_conditions.congestion_factor * 0.3)
        
        # Direct Connect processing
        dx_latency = 0
        if pattern.get('dx_type'):
            dx_char = self.dx_characteristics[pattern['dx_type']]
            dx_latency = dx_char['cross_connect_latency_ms']
        
        # VPC Endpoint additional latency
        vpc_latency = 0
        if pattern['pattern_type'] == 'vpc_endpoint':
            vpc_latency = 2  # Additional PrivateLink routing
        
        total_latency = (base_latency + os_processing + nic_processing + 
                        switch_latency + wan_latency + dx_latency + vpc_latency)
        
        # Add jitter
        jitter = network_conditions.jitter_ms
        
        return {
            'total_latency_ms': total_latency,
            'jitter_ms': jitter,
            'breakdown': {
                'base': base_latency,
                'processing': os_processing + nic_processing,
                'switching': switch_latency, 
                'wan': wan_latency,
                'direct_connect': dx_latency,
                'vpc_endpoint': vpc_latency
            }
        }
    
    def apply_congestion_factor(self, bandwidth_mbps: float, pattern: Dict, network_conditions: NetworkConditions) -> float:
        """Apply network congestion based on time of day and pattern type"""
        congestion_factor = network_conditions.congestion_factor
        
        # Pattern-specific congestion sensitivity
        wan_char = self.wan_characteristics[pattern['wan_type']]
        congestion_sensitivity = wan_char.get('congestion_sensitivity', 0.5)
        
        # Apply congestion
        effective_congestion = 1 - ((1 - congestion_factor) * congestion_sensitivity)
        
        return bandwidth_mbps * effective_congestion
    
    def calculate_protocol_efficiency(self, service: Dict, network_conditions: NetworkConditions, file_size_distribution: str = 'mixed') -> float:
        """Calculate dynamic protocol efficiency based on conditions and file sizes"""
        base_efficiency = service['base_protocol_efficiency']
        
        # File size impact for DataSync
        if service['name'] == 'AWS DataSync':
            if file_size_distribution == 'small_files':
                base_efficiency *= 0.8  # Small files are less efficient
            elif file_size_distribution == 'large_files':
                base_efficiency *= 1.1  # Large files are more efficient
        
        # Network condition impact
        packet_loss_impact = 1 - (network_conditions.packet_loss_rate * 1000)  # Convert to impact factor
        jitter_impact = max(0.9, 1 - (network_conditions.jitter_ms * 0.01))
        
        # Error retry overhead
        retry_factor = service.get('error_retry_factor', 1.0)
        
        final_efficiency = base_efficiency * packet_loss_impact * jitter_impact * retry_factor
        
        return min(1.0, final_efficiency)
    
    def calculate_realistic_bandwidth_waterfall(self, pattern_key: str, migration_service: str, service_size: str, 
                                              num_instances: int, network_conditions: NetworkConditions = None,
                                              vmware_env: VMwareEnvironment = None) -> Dict:
        """Enhanced realistic bandwidth waterfall with all improvements"""
        
        if network_conditions is None:
            network_conditions = self.get_network_conditions('business_hours')
        
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
        
        # Step 4: Enhanced CPU Utilization Impact
        bandwidth_gbps = os_efficient_bandwidth / 1000
        cpu_impact_factor = self.calculate_cpu_impact(bandwidth_gbps, nic_char, os_char)
        cpu_adjusted_bandwidth = os_efficient_bandwidth * cpu_impact_factor
        cpu_reduction = os_efficient_bandwidth - cpu_adjusted_bandwidth
        
        cpu_utilization = bandwidth_gbps * nic_char['base_cpu_utilization_per_gbps']
        steps.append({
            'name': f'CPU Utilization Impact ({cpu_utilization*100:.1f}%)',
            'value': -cpu_reduction,
            'cumulative': cpu_adjusted_bandwidth,
            'type': 'reduction',
            'layer': 'os',
            'step_number': 4,
            'details': f"Enhanced calculation with hardware acceleration effects"
        })
        
        # Step 5: LAN Infrastructure with Congestion
        lan_switching_capacity = lan_char['switching_capacity_gbps'] * 1000
        oversubscription_factor = float(lan_char['oversubscription_ratio'].split(':')[0])
        effective_lan_capacity = lan_switching_capacity / oversubscription_factor
        
        # Apply congestion threshold
        if cpu_adjusted_bandwidth > effective_lan_capacity * lan_char['congestion_threshold']:
            congestion_penalty = 0.2  # 20% penalty for exceeding threshold
        else:
            congestion_penalty = 0
        
        lan_limited_bandwidth = min(cpu_adjusted_bandwidth, effective_lan_capacity) * (1 - congestion_penalty)
        lan_reduction = cpu_adjusted_bandwidth - lan_limited_bandwidth
        steps.append({
            'name': f'LAN Infrastructure ({lan_char["name"]})',
            'value': -lan_reduction,
            'cumulative': lan_limited_bandwidth,
            'type': 'reduction',
            'layer': 'lan',
            'step_number': 5,
            'details': f"Includes congestion penalty: {congestion_penalty*100:.0f}%" if congestion_penalty > 0 else "No congestion"
        })
        
        # Step 6: WAN Provider with Network Conditions
        wan_bandwidth = min(lan_limited_bandwidth, pattern['committed_bandwidth_mbps'])
        
        # Apply congestion factor
        congested_bandwidth = self.apply_congestion_factor(wan_bandwidth, pattern, network_conditions)
        wan_efficient_bandwidth = congested_bandwidth * wan_char['bandwidth_efficiency']
        wan_reduction = lan_limited_bandwidth - wan_efficient_bandwidth
        
        steps.append({
            'name': f'WAN Provider ({wan_char["name"]}) + Congestion',
            'value': -wan_reduction,
            'cumulative': wan_efficient_bandwidth,
            'type': 'reduction',
            'layer': 'wan',
            'step_number': 6,
            'details': f"Time of day: {network_conditions.time_of_day}, Congestion factor: {network_conditions.congestion_factor:.2f}"
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
        
        # Step 9: Enhanced Protocol Overhead
        protocol_efficiency = self.calculate_protocol_efficiency(service, network_conditions)
        protocol_adjusted_bandwidth = vpc_adjusted_bandwidth * protocol_efficiency
        protocol_reduction = vpc_adjusted_bandwidth - protocol_adjusted_bandwidth
        steps.append({
            'name': f'Protocol Overhead ({", ".join(service["protocols"])})',
            'value': -protocol_reduction,
            'cumulative': protocol_adjusted_bandwidth,
            'type': 'reduction',
            'layer': 'protocol',
            'step_number': step_number,
            'details': f"Dynamic efficiency: {protocol_efficiency:.3f} (includes retry overhead)"
        })
        step_number += 1
        
        # Step 10: Enhanced Service Capacity with VMware considerations
        if migration_service == 'datasync' and service.get('vmware_deployment', False):
            if vmware_env is None:
                vmware_env = VMwareEnvironment(
                    storage_type='ssd',
                    sr_iov_enabled=True,
                    memory_overcommit_ratio=1.2,
                    cpu_overcommit_ratio=1.5,
                    numa_optimization=True,
                    host_count=2
                )
            
            # Calculate VMware performance impact
            vmware_perf = self.calculate_datasync_vmware_performance(service_spec, vmware_env)
            single_vm_capacity = vmware_perf['effective_throughput_mbps']
            service_capacity = single_vm_capacity * num_instances
            
            # Add detailed VMware impact step
            vmware_overhead_pct = vmware_perf['total_overhead_percent']
            ideal_capacity = service_spec['base_throughput_mbps'] * num_instances
            vmware_reduction = ideal_capacity - service_capacity
            
            steps.append({
                'name': f'VMware Virtualization Impact ({vmware_overhead_pct:.1f}%)',
                'value': -vmware_reduction,
                'cumulative': min(protocol_adjusted_bandwidth, service_capacity),
                'type': 'reduction',
                'layer': 'service',
                'step_number': step_number,
                'details': f"Storage: {vmware_perf['breakdown']['storage']:.1f}%, Network: {vmware_perf['breakdown']['network']:.1f}%, Memory OC: {vmware_perf['breakdown']['memory']:.1f}%, CPU OC: {vmware_perf['breakdown']['cpu']:.1f}%"
            })
            step_number += 1
        else:
            # Standard service capacity calculation
            service_capacity = service_spec.get('base_throughput_mbps', service_spec.get('throughput_mbps', 1000)) * num_instances

        service_limited_bandwidth = min(protocol_adjusted_bandwidth, service_capacity)
        service_reduction = protocol_adjusted_bandwidth - service_limited_bandwidth

        # Enhanced step details
        step_name = f'{service["name"]} Capacity'
        if num_instances > 1:
            step_name += f' ({num_instances} instances @ {service_capacity/num_instances:.0f} Mbps each)'
        else:
            step_name += f' (Single instance @ {service_capacity:.0f} Mbps)'

        steps.append({
            'name': step_name,
            'value': -service_reduction,
            'cumulative': service_limited_bandwidth,
            'type': 'reduction',
            'layer': 'service',
            'step_number': step_number,
            'details': f"Service capacity: {service_capacity:.0f} Mbps, Network available: {protocol_adjusted_bandwidth:.0f} Mbps"
        })
        step_number += 1
        
        # Step 11: Enhanced Application Efficiency
        app_efficiency = service.get('base_application_efficiency', 0.9)
        
        # Apply additional service-specific overheads
        if migration_service == 'dms':
            app_efficiency *= (1 - service.get('cdc_overhead', 0.05))
        elif migration_service == 'datasync':
            app_efficiency *= (1 - service.get('checksum_verification_overhead', 0.02))
        elif migration_service == 'storage_gateway':
            app_efficiency *= service.get('cache_efficiency', 0.8)
        
        final_bandwidth = service_limited_bandwidth * app_efficiency
        app_reduction = service_limited_bandwidth - final_bandwidth
        steps.append({
            'name': f'Application Efficiency ({app_efficiency*100:.1f}%)',
            'value': -app_reduction,
            'cumulative': final_bandwidth,
            'type': 'reduction',
            'layer': 'service',
            'step_number': step_number,
            'details': f"Includes service-specific overheads and optimizations"
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
        
        # Enhanced summary with latency information
        total_reduction = theoretical_max - final_bandwidth
        efficiency_percentage = (final_bandwidth / theoretical_max) * 100
        
        # Calculate latency
        latency_info = self.calculate_end_to_end_latency(pattern_key, network_conditions)
        
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
                'baseline_latency_ms': latency_info['total_latency_ms'],
                'jitter_ms': latency_info['jitter_ms'],
                'service_utilization_percent': (final_bandwidth / service_capacity * 100) if service_capacity > 0 else 0,
                'network_utilization_percent': (final_bandwidth / pattern['committed_bandwidth_mbps'] * 100),
                'bottleneck': 'network' if final_bandwidth == pattern['committed_bandwidth_mbps'] else 'service',
                'network_conditions': network_conditions.time_of_day
            },
            'infrastructure_details': {
                'os': os_char,
                'nic': nic_char,
                'lan': lan_char,
                'wan': wan_char,
                'dx': dx_char
            },
            'latency_details': latency_info,
            'network_conditions': network_conditions
        }
    
    def validate_database_requirements(self, db_scenario: Dict, waterfall_summary: Dict, latency_info: Dict) -> Dict:
        """Enhanced database requirements validation"""
        meets_bandwidth = waterfall_summary['final_effective_mbps'] >= db_scenario['min_bandwidth_mbps']
        meets_latency = latency_info['total_latency_ms'] <= db_scenario['max_tolerable_latency_ms']
        meets_jitter = latency_info['jitter_ms'] <= db_scenario.get('max_tolerable_jitter_ms', 5)
        
        # Check burst requirements
        burst_requirement = db_scenario['min_bandwidth_mbps'] * db_scenario.get('burst_requirement_factor', 1.0)
        meets_burst = waterfall_summary['theoretical_max_mbps'] >= burst_requirement
        
        # Packet loss check (from network conditions)
        network_conditions = waterfall_summary.get('network_conditions')
        meets_packet_loss = True  # Default assumption
        
        validation_results = {
            'overall_pass': meets_bandwidth and meets_latency and meets_jitter and meets_burst,
            'bandwidth': {
                'required_mbps': db_scenario['min_bandwidth_mbps'],
                'available_mbps': waterfall_summary['final_effective_mbps'],
                'meets_requirement': meets_bandwidth,
                'headroom_percent': ((waterfall_summary['final_effective_mbps'] / db_scenario['min_bandwidth_mbps']) - 1) * 100 if meets_bandwidth else 0
            },
            'latency': {
                'required_ms': db_scenario['max_tolerable_latency_ms'],
                'actual_ms': latency_info['total_latency_ms'],
                'meets_requirement': meets_latency,
                'margin_ms': db_scenario['max_tolerable_latency_ms'] - latency_info['total_latency_ms']
            },
            'jitter': {
                'required_ms': db_scenario.get('max_tolerable_jitter_ms', 5),
                'actual_ms': latency_info['jitter_ms'],
                'meets_requirement': meets_jitter
            },
            'burst_capacity': {
                'required_mbps': burst_requirement,
                'available_mbps': waterfall_summary['theoretical_max_mbps'],
                'meets_requirement': meets_burst
            }
        }
        
        return validation_results
    
    def _determine_dx_speed(self, committed_bandwidth_mbps: int) -> str:
        """Determine appropriate Direct Connect speed"""
        if committed_bandwidth_mbps >= 50000:
            return '100Gbps'
        elif committed_bandwidth_mbps >= 5000:
            return '10Gbps'
        else:
            return '1Gbps'
    
    def _calculate_total_cost(self, pattern_key: str, config: Dict, waterfall_data: Dict) -> float:
        """Enhanced cost calculation with accurate pricing"""
        pattern = self.network_patterns[pattern_key]
        
        # Migration duration
        migration_hours = self._estimate_migration_time(config, waterfall_data)
        
        # Direct Connect costs (if applicable)
        dx_cost = 0
        if pattern['pattern_type'] == 'direct_connect':
            # Port cost for minimum 1 month commitment
            port_hours = max(migration_hours, 24 * 30)  # AWS requires monthly commitment
            dx_speed = self._determine_dx_speed(pattern['committed_bandwidth_mbps'])
            dx_pricing = self.pricing_client.get_direct_connect_pricing(dx_speed)
            
            dx_cost += dx_pricing['port_hour'] * port_hours
            
            # Data transfer (outbound only - inbound is free)
            outbound_gb = config['data_size_gb'] * 0.1  # Assume 10% metadata/logs outbound
            dx_cost += dx_pricing['data_transfer_gb'] * outbound_gb
        
        # VPC Endpoint costs
        vpc_endpoint_cost = 0
        if pattern['pattern_type'] == 'vpc_endpoint':
            vpc_pricing = self.pricing_client.get_vpc_endpoint_pricing()
            vpc_endpoint_cost = migration_hours * vpc_pricing['hour']  # Hourly cost
            vpc_endpoint_cost += config['data_size_gb'] * vpc_pricing['data_gb']  # Data processing cost
        
        # Service instance costs
        service_config = self.migration_services[config['migration_service']]['sizes'][config['service_size']]
        instance_cost = service_config['cost_per_hour'] * migration_hours * config['num_instances']
        
        # Storage costs for DataSync (if applicable)
        storage_cost = 0
        if config['migration_service'] == 'datasync' and 'storage_gb' in service_config:
            storage_cost = service_config.get('storage_cost_per_gb_month', 0.1) * config['data_size_gb'] * (migration_hours / (24 * 30))
        
        # Total with realistic overhead (5% operational)
        total_cost = dx_cost + vpc_endpoint_cost + instance_cost + storage_cost
        return total_cost * 1.05  # 5% operational overhead
    
    def _estimate_migration_time(self, config: Dict, waterfall_data: Dict) -> float:
        """Enhanced migration time estimation with retry considerations"""
        effective_bandwidth_mbps = waterfall_data['summary']['final_effective_mbps']
        data_size_gb = config['data_size_gb']
        service = self.migration_services[config['migration_service']]
        
        if effective_bandwidth_mbps > 0:
            base_transfer_hours = (data_size_gb * 8) / (effective_bandwidth_mbps / 1000) / 3600
        else:
            base_transfer_hours = float('inf')
        
        # Apply retry factor
        retry_factor = service.get('error_retry_factor', 1.0)
        actual_transfer_hours = base_transfer_hours / retry_factor
        
        # Service-specific setup time
        if config['migration_service'] == 'dms':
            setup_hours = 4  # Schema conversion + endpoint setup
            validation_hours = max(2, data_size_gb / 1000)  # Data validation
        elif config['migration_service'] == 'datasync':
            setup_hours = 2  # Agent deployment + task setup
            validation_hours = max(1, data_size_gb / 2000)  # File verification
        else:
            setup_hours = 2
            validation_hours = max(1, data_size_gb / 1500)
        
        return actual_transfer_hours + setup_hours + validation_hours
    
    # All remaining methods from original code preserved...
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
    
    def calculate_optimal_datasync_instances(self, available_bandwidth_mbps: float, data_size_gb: int, target_transfer_time_hours: float) -> Dict:
        """Calculate optimal number of DataSync instances for production workloads"""
        
        # Get DataSync VM specs from migration services
        datasync_service = self.migration_services['datasync']
        
        # Calculate required throughput
        required_throughput_mbps = (data_size_gb * 8) / (target_transfer_time_hours * 3600) * 1000
        
        recommendations = {}
        
        # Analyze each size configuration
        for size_name, size_config in datasync_service['sizes'].items():
            base_throughput = size_config['base_throughput_mbps']
            
            # Use enhanced VMware calculation if available
            vmware_env = VMwareEnvironment(
                storage_type='ssd',
                sr_iov_enabled=True,
                memory_overcommit_ratio=1.2,
                cpu_overcommit_ratio=1.5,
                numa_optimization=True,
                host_count=2
            )
            
            vmware_perf = self.calculate_datasync_vmware_performance(size_config, vmware_env)
            effective_throughput = vmware_perf['effective_throughput_mbps']
            
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
                'is_bottleneck': required_throughput_mbps > total_throughput,
                'vmware_overhead_details': vmware_perf['breakdown']
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
    
    def generate_ai_recommendations(self, config: Dict, analysis_results: Dict) -> Dict:
        """Generate AI-powered recommendations (ENHANCED VERSION)"""
        migration_time = analysis_results['migration_time']
        waterfall_data = analysis_results['waterfall_data']
        service_compatibility = analysis_results.get('service_compatibility', {})
        
        recommendations = []
        priority_score = 0
        
        # Enhanced DataSync-specific recommendations
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
        
        # Enhanced infrastructure bottleneck recommendations
        primary_bottleneck = waterfall_data['summary']['primary_bottleneck_layer']
        if primary_bottleneck == 'nic':
            recommendations.append({
                'type': 'infrastructure_upgrade',
                'priority': 'high',
                'description': 'Network Interface Card is the primary bottleneck. Consider upgrading to higher bandwidth NIC with hardware acceleration.',
                'impact': 'Significant throughput improvement possible with RDMA support'
            })
            priority_score += 20
        elif primary_bottleneck == 'lan':
            recommendations.append({
                'type': 'network_optimization',
                'priority': 'medium',
                'description': 'LAN infrastructure is limiting performance. Review switch capacity and oversubscription ratios.',
                'impact': 'Moderate throughput improvement with spine-leaf architecture'
            })
            priority_score += 15
        elif primary_bottleneck == 'wan':
            recommendations.append({
                'type': 'wan_optimization',
                'priority': 'high',
                'description': 'WAN provider is the limiting factor. Consider dedicated bandwidth upgrade or multiple connections.',
                'impact': 'Direct improvement to available bandwidth'
            })
            priority_score += 25
        
        # Network conditions recommendations
        network_conditions = waterfall_data.get('network_conditions')
        if network_conditions and hasattr(network_conditions, 'time_of_day'):
            if network_conditions.time_of_day == 'business_hours':
                recommendations.append({
                    'type': 'scheduling_optimization',
                    'priority': 'medium',
                    'description': 'Migration scheduled during business hours. Consider off-hours migration for 30% better performance.',
                    'impact': 'Improved throughput due to reduced network congestion'
                })
                priority_score += 10
        
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
    
    # Enhanced pattern analysis methods
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
        """Analyze a single pattern with enhanced calculations"""
        pattern = self.network_patterns[pattern_key]
        
        # Calculate bandwidth and costs with enhanced methods
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
        
        # Enhanced AI recommendation score (weighted combination)
        latency_score = max(0, 1 - (waterfall_data['summary']['baseline_latency_ms'] / 50))  # Normalize latency
        cost_score = max(0, 1 - min(total_cost / 20000, 1.0))  # Normalize cost
        
        ai_score = (
            reliability_score * 0.25 +
            complexity_score * 0.15 +
            db_suitability * 0.25 +
            latency_score * 0.15 +
            cost_score * 0.20
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
    
    def _generate_pattern_details(self, pattern: Dict, config: Dict) -> Tuple[List[str], List[str], List[str]]:
        """Generate use cases, pros, and cons for a pattern"""
        pattern_type = pattern['pattern_type']
        
        if pattern_type == 'vpc_endpoint':
            use_cases = [
                "Small to medium database migrations (<1TB)",
                "Non-production environments",
                "Cost-sensitive migrations",
                "Services with VPC endpoint support",
                "Development and testing workloads"
            ]
            pros = [
                "Lower cost structure",
                "No Direct Connect setup required",
                "Good for compatible services",
                "AWS managed connectivity",
                "Quick deployment"
            ]
            cons = [
                "Higher latency than Direct Connect",
                "Limited service compatibility",
                "Shared bandwidth",
                "Internet routing dependencies",
                "No guaranteed performance"
            ]
        elif pattern_type == 'direct_connect':
            use_cases = [
                "Production database migrations",
                "Large data volumes (>500GB)",
                "Latency-sensitive applications",
                "Enterprise-grade connectivity needs",
                "Mission-critical workloads"
            ]
            pros = [
                "Dedicated bandwidth",
                "Lower, consistent latency",
                "Higher reliability (99.9% SLA)",
                "Better security posture",
                "Predictable performance"
            ]
            cons = [
                "Higher setup costs",
                "Longer provisioning time (2-4 weeks)",
                "Requires network expertise",
                "Monthly recurring costs",
                "Complex configuration"
            ]
        else:  # multi_hop
            use_cases = [
                "Remote location connectivity",
                "Complex network topologies",
                "Phased migration approaches",
                "Disaster recovery scenarios",
                "Geographically distributed systems"
            ]
            pros = [
                "Can reach remote locations",
                "Flexible routing options",
                "Can leverage existing infrastructure",
                "Supports complex scenarios",
                "Redundant paths available"
            ]
            cons = [
                "Highest complexity",
                "Multiple failure points",
                "Variable performance",
                "Difficult troubleshooting",
                "Increased latency"
            ]
        
        return use_cases, pros, cons
    
    def get_ai_recommendation(self, pattern_analyses: List[PatternAnalysis], config: Dict) -> Dict:
        """Get AI recommendation for best pattern with enhanced analysis"""
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
                    'reliability': p.reliability_score,
                    'complexity': p.complexity_score,
                    'ai_score': p.ai_recommendation_score
                }
                for p in pattern_analyses
            ]
        }
        
        return self.ai_client.get_pattern_recommendation(analysis_data)

# =============================================================================
# HEADER AND UTILITY FUNCTIONS (Enhanced)
# =============================================================================

def render_corporate_header():
    """Render enhanced corporate header"""
    st.markdown("""
    <div class="corporate-header">
        <h1>🏢 AWS Migration Network Analyzer</h1>
        <div class="subtitle">Enterprise Infrastructure Analysis Platform</div>
        <div class="tagline">Real-time AWS Pricing • AI-Powered Recommendations • Enhanced Network Modeling • Database-Optimized Migration Patterns</div>
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
# ENHANCED SIDEBAR CONTROLS
# =============================================================================

def render_enhanced_sidebar_controls():
    """Render enhanced sidebar controls with network conditions"""
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
    **Burst Factor:** {selected_scenario.get('burst_requirement_factor', 1.0)}x
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
    
    # Network Conditions (NEW)
    st.sidebar.subheader("🌐 Network Conditions")
    time_of_day = st.sidebar.selectbox(
        "Migration Time Window",
        ["business_hours", "off_hours", "weekend"],
        format_func=lambda x: {
            "business_hours": "Business Hours (Higher Congestion)",
            "off_hours": "Off Hours (Optimal)",
            "weekend": "Weekend (Low Congestion)"
        }[x],
        help="Network performance varies by time of day"
    )
    
    # VMware Environment Settings (NEW)
    with st.sidebar.expander("🖥️ VMware Environment (DataSync)", expanded=False):
        storage_type = st.sidebar.selectbox(
            "Storage Type",
            ["ssd", "hdd"],
            format_func=lambda x: "SSD (Recommended)" if x == "ssd" else "HDD (Standard)",
            help="Storage type affects DataSync performance"
        )
        
        sr_iov_enabled = st.sidebar.checkbox(
            "SR-IOV Enabled",
            value=True,
            help="Single Root I/O Virtualization for better network performance"
        )
        
        memory_overcommit = st.sidebar.slider(
            "Memory Overcommit Ratio",
            min_value=1.0,
            max_value=2.0,
            value=1.2,
            step=0.1,
            help="Memory overcommitment ratio (1.0 = no overcommit)"
        )
        
        cpu_overcommit = st.sidebar.slider(
            "CPU Overcommit Ratio", 
            min_value=1.0,
            max_value=3.0,
            value=1.5,
            step=0.1,
            help="CPU overcommitment ratio (1.0 = no overcommit)"
        )
        
        numa_optimization = st.sidebar.checkbox(
            "NUMA Optimization",
            value=True,
            help="Non-Uniform Memory Access optimization"
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
        format_func=lambda x: f"{x.title()} - {service_info['sizes'][x].get('base_throughput_mbps', service_info['sizes'][x].get('throughput_mbps', 'Variable'))} {'Mbps' if 'base_throughput_mbps' in service_info['sizes'][x] or 'throughput_mbps' in service_info['sizes'][x] else ''}",
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
    **Max Jitter:** {selected_scenario.get('max_tolerable_jitter_ms', 5)}ms  
    **Downtime Sensitivity:** {selected_scenario['downtime_sensitivity'].title()}
    """)
    
    # Combine configuration
    config = {
        'database_scenario': database_scenario,
        'source_location': source_location,
        'environment': environment,
        'time_of_day': time_of_day,
        'vmware_env': {
            'storage_type': storage_type,
            'sr_iov_enabled': sr_iov_enabled,
            'memory_overcommit_ratio': memory_overcommit,
            'cpu_overcommit_ratio': cpu_overcommit,
            'numa_optimization': numa_optimization,
            'host_count': 2  # Default
        },
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
# ENHANCED CHART CREATION FUNCTIONS
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
            'text': "Enhanced Infrastructure Impact Waterfall Analysis",
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
    """Create a sequential bar chart showing the bandwidth reduction flow with enhanced details"""
    steps = waterfall_data['steps']
    
    st.markdown("### 📊 Enhanced Sequential Infrastructure Flow")
    
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
    
    # Layout configuration
    fig.update_layout(
        title="Enhanced Infrastructure Impact Sequential Analysis",
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
            'text': "Enhanced Enterprise Network Architecture: Production & Non-Production Environments",
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
# ENHANCED TAB RENDERING FUNCTIONS
# =============================================================================

def render_realistic_analysis_tab(config: Dict, analyzer: EnhancedNetworkAnalyzer):
    """Render enhanced realistic bandwidth analysis tab with all improvements"""
    st.subheader("💧 Enhanced Infrastructure Impact Analysis")
    
    # Show database scenario info
    db_scenario = analyzer.database_scenarios[config['database_scenario']]
    
    st.markdown(f"""
    <div class="corporate-card status-card-info">
        <h3>🗄️ Database Migration Scenario</h3>
        <p><strong>Source Database:</strong> {db_scenario['name']}</p>
        <p><strong>Target Service:</strong> {db_scenario['target_service']}</p>
        <p><strong>Workload Type:</strong> {db_scenario['workload_type'].upper()}</p>
        <p><strong>Migration Complexity:</strong> {db_scenario['migration_complexity'].title()}</p>
        <p><strong>Burst Requirement:</strong> {db_scenario.get('burst_requirement_factor', 1.0)}x baseline</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get network conditions
    network_conditions = analyzer.get_network_conditions(config['time_of_day'])
    
    # Create VMware environment if using DataSync
    vmware_env = None
    if config['migration_service'] == 'datasync':
        from dataclasses import fields
        vmware_env = VMwareEnvironment(**config['vmware_env'])
    
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
    
    # Calculate enhanced realistic waterfall
    waterfall_data = analyzer.calculate_realistic_bandwidth_waterfall(
        pattern_key,
        config['migration_service'],
        config['service_size'],
        config['num_instances'],
        network_conditions,
        vmware_env
    )
    
    summary = waterfall_data['summary']
    latency_info = waterfall_data['latency_details']
    
    # Enhanced summary metrics
    st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{summary['theoretical_max_mbps']:,.0f}</div>
            <div class="metric-label">Theoretical Max (Mbps)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{summary['final_effective_mbps']:,.0f}</div>
            <div class="metric-label">Final Effective (Mbps)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{summary['efficiency_percentage']:.1f}%</div>
            <div class="metric-label">Overall Efficiency</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{latency_info['total_latency_ms']:.1f}ms</div>
            <div class="metric-label">End-to-End Latency</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{latency_info['jitter_ms']:.1f}ms</div>
            <div class="metric-label">Network Jitter</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col6:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{network_conditions.time_of_day.replace('_', ' ').title()}</div>
            <div class="metric-label">Network Conditions</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Enhanced database requirements validation
    db_validation = analyzer.validate_database_requirements(db_scenario, summary, latency_info)
    
    if not db_validation['overall_pass']:
        failing_requirements = []
        if not db_validation['bandwidth']['meets_requirement']:
            failing_requirements.append(f"Bandwidth: {summary['final_effective_mbps']:,.0f} Mbps < {db_validation['bandwidth']['required_mbps']} Mbps required")
        if not db_validation['latency']['meets_requirement']:
            failing_requirements.append(f"Latency: {latency_info['total_latency_ms']:.1f}ms > {db_validation['latency']['required_ms']}ms required")
        if not db_validation['jitter']['meets_requirement']:
            failing_requirements.append(f"Jitter: {latency_info['jitter_ms']:.1f}ms > {db_validation['jitter']['required_ms']}ms required")
        if not db_validation['burst_capacity']['meets_requirement']:
            failing_requirements.append(f"Burst: {db_validation['burst_capacity']['available_mbps']:,.0f} Mbps < {db_validation['burst_capacity']['required_mbps']:,.0f} Mbps required")
        
        st.markdown(f"""
        <div class="corporate-card status-card-error">
            <h3>❌ Database Requirements Not Met</h3>
            <p><strong>Failing Requirements:</strong></p>
            {"".join([f"<p>• {req}</p>" for req in failing_requirements])}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="corporate-card status-card-success">
            <h3>✅ All Database Requirements Satisfied</h3>
            <p><strong>Bandwidth:</strong> {summary['final_effective_mbps']:,.0f} Mbps (Required: {db_validation['bandwidth']['required_mbps']} Mbps, Headroom: {db_validation['bandwidth']['headroom_percent']:.1f}%)</p>
            <p><strong>Latency:</strong> {latency_info['total_latency_ms']:.1f}ms (Max: {db_validation['latency']['required_ms']}ms, Margin: {db_validation['latency']['margin_ms']:.1f}ms)</p>
            <p><strong>Jitter:</strong> {latency_info['jitter_ms']:.1f}ms (Max: {db_validation['jitter']['required_ms']}ms)</p>
            <p><strong>Burst Capacity:</strong> {db_validation['burst_capacity']['available_mbps']:,.0f} Mbps available</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Enhanced DataSync Analysis with VMware details
    if config['migration_service'] == 'datasync':
        datasync_scaling = analyzer.get_datasync_scaling_recommendations(pattern_key, config)
        
        if datasync_scaling['applicable']:
            card_type = "status-card-warning" if datasync_scaling['is_bottleneck'] else "status-card-success"
            
            current_config = datasync_scaling['current_config']
            optimal_config = datasync_scaling['optimal_config']
            
            scaling_content = f"""
            <h3>🚀 Enhanced DataSync Scaling Analysis</h3>
            <p><strong>Current Configuration:</strong> {current_config['instances']} × {config['service_size']} VM(s) = {current_config['effective_throughput']:,.0f} Mbps</p>
            <p><strong>Optimal Configuration:</strong> {optimal_config['instances']} × {optimal_config['size']} VM(s) = {optimal_config['effective_throughput']:,.0f} Mbps</p>
            <p><strong>VMware Environment:</strong> {vmware_env.storage_type.upper()}, SR-IOV: {'Yes' if vmware_env.sr_iov_enabled else 'No'}, Memory OC: {vmware_env.memory_overcommit_ratio}x</p>
            <p><strong>Analysis:</strong> {datasync_scaling['explanation']}</p>
            """
            
            if datasync_scaling['recommendations']:
                scaling_content += "<h4>💡 Enhanced Scaling Recommendations:</h4>"
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

    # Enhanced VMware deployment information
    if config['migration_service'] == 'datasync':
        with st.expander("🖥️ Enhanced VMware Deployment Details", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **📋 VMware Requirements per DataSync VM:**
                - **vCPU:** 4-32 cores (depending on size)
                - **Memory:** 16-128 GB RAM 
                - **Storage:** 80 GB minimum (SSD recommended)
                - **Network:** Dedicated vSwitch with SR-IOV
                - **Hypervisor:** VMware ESXi 6.5+ supported
                - **NUMA:** Enable NUMA awareness for >16 cores
                """)
                
                st.markdown(f"""
                **⚙️ Current Environment Optimizations:**
                - **Storage Type:** {vmware_env.storage_type.upper()} ({'✅ Optimal' if vmware_env.storage_type == 'ssd' else '⚠️ Consider SSD upgrade'})
                - **SR-IOV:** {'✅ Enabled' if vmware_env.sr_iov_enabled else '❌ Disabled - Enable for 50% performance boost'}
                - **Memory Overcommit:** {vmware_env.memory_overcommit_ratio}x ({'✅ Acceptable' if vmware_env.memory_overcommit_ratio <= 1.5 else '⚠️ High - May impact performance'})
                - **CPU Overcommit:** {vmware_env.cpu_overcommit_ratio}x ({'✅ Acceptable' if vmware_env.cpu_overcommit_ratio <= 2.0 else '⚠️ High - May impact performance'})
                - **NUMA Optimization:** {'✅ Enabled' if vmware_env.numa_optimization else '❌ Disabled - Enable for large VMs'}
                """)
            
            with col2:
                # Calculate enhanced VMware requirements
                datasync_scaling = analyzer.get_datasync_scaling_recommendations(pattern_key, config)
                if datasync_scaling['applicable']:
                    optimal = datasync_scaling['optimal_config']
                    service_spec = analyzer.migration_services['datasync']['sizes'][optimal['size']]
                    
                    total_vcpu = service_spec['vcpu'] * optimal['instances']
                    total_memory = service_spec['memory_gb'] * optimal['instances']
                    total_storage = 80 * optimal['instances']
                    
                    # Calculate VMware overhead breakdown
                    vmware_perf = analyzer.calculate_datasync_vmware_performance(service_spec, vmware_env)
                    overhead_breakdown = vmware_perf['breakdown']
                    
                    st.markdown(f"""
                    **🏗️ Enhanced VMware Resource Requirements:**
                    - **Total vCPU:** {total_vcpu} cores
                    - **Total Memory:** {total_memory} GB
                    - **Total Storage:** {total_storage} GB SSD
                    - **Network Bandwidth:** {optimal['effective_throughput']:,.0f} Mbps effective
                    - **Recommended Hosts:** {math.ceil(optimal['instances'] / 2)} (max 2 VMs per host)
                    """)
                    
                    st.markdown(f"""
                    **📊 VMware Overhead Analysis:**
                    - **Storage Overhead:** {overhead_breakdown['storage']:.1f}%
                    - **Network Overhead:** {overhead_breakdown['network']:.1f}%
                    - **Memory OC Impact:** {overhead_breakdown['memory']:.1f}%
                    - **CPU OC Impact:** {overhead_breakdown['cpu']:.1f}%
                    - **NUMA Bonus:** +{overhead_breakdown['numa_bonus']:.1f}%
                    - **Total Impact:** {vmware_perf['total_overhead_percent']:.1f}%
                    """)
    
    # Enhanced sequential flow diagram
    create_sequential_flow_diagram(waterfall_data)
    
    # Enhanced waterfall chart
    st.markdown('<div class="waterfall-container">', unsafe_allow_html=True)
    
    waterfall_chart = create_enhanced_waterfall_chart(waterfall_data)
    st.plotly_chart(waterfall_chart, use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Enhanced detailed breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        infrastructure = waterfall_data['infrastructure_details']
        
        st.markdown(f"""
        <div class="corporate-card status-card-info">
            <h3>🏗️ Enhanced Infrastructure Component Details</h3>
            <p><strong>Operating System:</strong> {infrastructure['os']['name']}</p>
            <p><strong>TCP Efficiency:</strong> {infrastructure['os']['tcp_stack_efficiency']*100:.1f}%</p>
            <p><strong>Hardware Acceleration:</strong> {'✅ Yes' if infrastructure['os'].get('hardware_acceleration') else '❌ No'}</p>
            <p><strong>Network Interface:</strong> {infrastructure['nic']['name']}</p>
            <p><strong>NIC Efficiency:</strong> {infrastructure['nic']['real_world_efficiency']*100:.1f}%</p>
            <p><strong>SR-IOV Support:</strong> {'✅ Yes' if infrastructure['nic'].get('sr_iov_support') else '❌ No'}</p>
            <p><strong>LAN Infrastructure:</strong> {infrastructure['lan']['name']}</p>
            <p><strong>Oversubscription:</strong> {infrastructure['lan']['oversubscription_ratio']}</p>
            <p><strong>WAN Provider:</strong> {infrastructure['wan']['name']}</p>
            <p><strong>WAN Efficiency:</strong> {infrastructure['wan']['bandwidth_efficiency']*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        primary_bottleneck = summary['primary_bottleneck_layer']
        
        # Enhanced recommendations based on bottleneck
        if primary_bottleneck == 'nic':
            recommendation = "Upgrade to 25/100 Gbps NIC with RDMA support"
            card_type = "status-card-warning"
            impact = f"Potential {infrastructure['nic']['theoretical_bandwidth_mbps']/1000*2-infrastructure['nic']['theoretical_bandwidth_mbps']/1000:.0f}x bandwidth increase"
        elif primary_bottleneck == 'os':
            recommendation = "Enable hardware acceleration and kernel bypass"
            card_type = "status-card-warning"  
            impact = "10-30% performance improvement"
        elif primary_bottleneck == 'lan':
            recommendation = "Upgrade to spine-leaf fabric or reduce oversubscription"
            card_type = "status-card-warning"
            impact = "Eliminate LAN congestion bottleneck"
        elif primary_bottleneck == 'wan':
            recommendation = "Upgrade WAN bandwidth or consider multiple connections"
            card_type = "status-card-error"
            impact = "Direct bandwidth improvement"
        elif primary_bottleneck == 'service':
            recommendation = "Scale service instances or optimize configuration"
            card_type = "status-card-success"
            impact = "Application-level performance gains"
        else:
            recommendation = "Review overall infrastructure architecture"
            card_type = "status-card-info"
            impact = "Comprehensive optimization approach"
        
        # Add network conditions impact
        congestion_impact = f"Current conditions ({network_conditions.time_of_day.replace('_', ' ')}) reduce performance by {(1-network_conditions.congestion_factor)*100:.0f}%"
        
        st.markdown(f"""
        <div class="corporate-card {card_type}">
            <h3>🎯 Enhanced Optimization Recommendations</h3>
            <p><strong>Primary Bottleneck:</strong> {summary['primary_bottleneck']}</p>
            <p><strong>Impact:</strong> {summary['primary_bottleneck_impact_mbps']:,.0f} Mbps reduction</p>
            <p><strong>Recommendation:</strong> {recommendation}</p>
            <p><strong>Expected Improvement:</strong> {impact}</p>
            <p><strong>Network Conditions:</strong> {congestion_impact}</p>
            <p><strong>Optimal Migration Time:</strong> Off-hours for 30% better performance</p>
        </div>
        """, unsafe_allow_html=True)
    
    return waterfall_data

def render_migration_analysis_tab(config: Dict, waterfall_data: Dict, analyzer: EnhancedNetworkAnalyzer):
    """Render enhanced migration timing and compatibility analysis"""
    st.subheader("⏱️ Enhanced Migration Analysis & Compatibility")
    
    # Service compatibility
    pattern_key = analyzer.determine_optimal_pattern(
        config['source_location'], 
        config['environment'], 
        config['migration_service']
    )
    
    service_compatibility = analyzer.assess_service_compatibility(
        pattern_key, config['migration_service'], config['service_size']
    )
    
    # Enhanced migration timing
    migration_time = analyzer.estimate_migration_time(
        config['data_size_gb'],
        waterfall_data['summary']['final_effective_mbps'],
        config['migration_service']
    )
    
    # Enhanced timing with retry considerations
    enhanced_timing = analyzer._estimate_migration_time(config, waterfall_data)
    
    # Enhanced compatibility and timing metrics
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
            <div class="metric-value">{enhanced_timing:.1f}h</div>
            <div class="metric-label">Total Migration Time</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col6:
        meets_requirement = enhanced_timing <= config['max_downtime_hours']
        status_text = "✅ Meets SLA" if meets_requirement else "❌ Exceeds SLA"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{status_text}</div>
            <div class="metric-label">SLA Compliance</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Enhanced analysis in corporate cards
    col1, col2 = st.columns(2)
    
    with col1:
        # Enhanced compatibility assessment
        if service_compatibility['warnings']:
            card_type = "status-card-error"
        elif service_compatibility['requirements']:
            card_type = "status-card-warning"
        else:
            card_type = "status-card-success"
        
        compatibility_content = ""
        
        if service_compatibility['warnings']:
            compatibility_content += "<h4>🚨 Compatibility Warnings:</h4>"
            for warning in service_compatibility['warnings']:
                compatibility_content += f"<p>• {warning}</p>"
        
        if service_compatibility['requirements']:
            compatibility_content += "<h4>📋 Service Requirements:</h4>"
            for requirement in service_compatibility['requirements']:
                compatibility_content += f"<p>• {requirement}</p>"
        
        if service_compatibility['recommendations']:
            compatibility_content += "<h4>💡 Performance Recommendations:</h4>"
            for recommendation in service_compatibility['recommendations']:
                compatibility_content += f"<p>• {recommendation}</p>"
        
        # Add network conditions recommendations
        network_conditions = waterfall_data.get('network_conditions')
        if network_conditions and hasattr(network_conditions, 'time_of_day'):
            if network_conditions.time_of_day == 'business_hours':
                compatibility_content += "<h4>🕒 Timing Optimization:</h4>"
                compatibility_content += "<p>• Consider off-hours migration for 30% better performance</p>"
        
        if not any([service_compatibility['warnings'], service_compatibility['requirements'], service_compatibility['recommendations']]):
            compatibility_content = "<p>✅ No compatibility issues detected</p><p>✅ All requirements satisfied</p>"
        
        st.markdown(f"""
        <div class="corporate-card {card_type}">
            <h3>📋 Enhanced Service Compatibility Analysis</h3>
            {compatibility_content}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Enhanced timeline analysis with retry factors
        service = analyzer.migration_services[config['migration_service']]
        retry_factor = service.get('error_retry_factor', 1.0)
        
        base_transfer_time = migration_time['data_transfer_hours']
        retry_overhead = base_transfer_time * (1 - retry_factor)
        
        timeline_data = [
            {"Phase": "Setup & Config", "Hours": migration_time['setup_hours'], "Details": "Service configuration and endpoint setup"},
            {"Phase": "Data Transfer", "Hours": base_transfer_time, "Details": f"Base transfer time at {waterfall_data['summary']['final_effective_mbps']:,.0f} Mbps"},
            {"Phase": "Retry Overhead", "Hours": retry_overhead, "Details": f"Error handling and retries ({(1-retry_factor)*100:.0f}% overhead)"},
            {"Phase": "Validation", "Hours": migration_time['validation_hours'], "Details": "Data integrity verification"}
        ]
        
        df_timeline = pd.DataFrame(timeline_data)
        
        # Enhanced migration timeline card
        meets_sla = enhanced_timing <= config['max_downtime_hours']
        timeline_card_type = "status-card-success" if meets_sla else "status-card-error"
        
        incremental_support = "✅ Supports incremental migration" if migration_time.get('supports_incremental') else "❌ Full migration required"
        
        # Calculate time savings for off-hours
        if hasattr(waterfall_data.get('network_conditions', {}), 'time_of_day'):
            off_hours_improvement = 30 if waterfall_data['network_conditions'].time_of_day == 'business_hours' else 0
            off_hours_time = enhanced_timing * (1 - off_hours_improvement/100)
        else:
            off_hours_improvement = 0
            off_hours_time = enhanced_timing
        
        st.markdown(f"""
        <div class="corporate-card {timeline_card_type}">
            <h3>⏱️ Enhanced Migration Timeline Analysis</h3>
            <p><strong>Service:</strong> {migration_time['service_name']}</p>
            <p><strong>Total Time:</strong> {enhanced_timing:.1f} hours</p>
            <p><strong>SLA Compliance:</strong> {'✅ Pass' if meets_sla else '❌ Fail'}</p>
            <p><strong>Incremental Support:</strong> {incremental_support}</p>
            <p><strong>Retry Overhead:</strong> {retry_overhead:.1f} hours ({(1-retry_factor)*100:.0f}%)</p>
            {f'<p><strong>Off-Hours Time:</strong> {off_hours_time:.1f} hours ({off_hours_improvement}% faster)</p>' if off_hours_improvement > 0 else ''}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**Enhanced Timeline Breakdown:**")
        st.dataframe(df_timeline, use_container_width=True, hide_index=True)
    
    return {
        'migration_time': migration_time,
        'enhanced_timing': enhanced_timing,
        'service_compatibility': service_compatibility,
        'waterfall_data': waterfall_data,
        'retry_overhead': retry_overhead
    }

def render_ai_recommendations_tab(config: Dict, analysis_results: Dict, analyzer: EnhancedNetworkAnalyzer):
    """Render enhanced AI-powered recommendations"""
    st.subheader("🤖 Enhanced AI-Powered Migration Recommendations")
    
    ai_recommendations = analyzer.generate_ai_recommendations(config, analysis_results)
    
    # Enhanced overview metrics
    st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
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
        enhanced_timing = analysis_results.get('enhanced_timing', analysis_results['migration_time']['total_hours'])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{enhanced_timing:.1f}h</div>
            <div class="metric-label">Migration Window</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        priority_score = ai_recommendations['overall_priority_score']
        score_color = 'var(--error-red)' if priority_score > 40 else 'var(--warning-orange)' if priority_score > 20 else 'var(--success-green)'
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: {score_color};">{priority_score}</div>
            <div class="metric-label">Priority Score</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Enhanced recommendations by priority
    st.markdown("### 💡 Enhanced Prioritized Recommendations")
    
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
                <p><strong>Priority Level:</strong> Immediate action required</p>
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
                <p><strong>Priority Level:</strong> Address before migration</p>
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
                <p><strong>Priority Level:</strong> Optimize for better performance</p>
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
                <p><strong>Priority Level:</strong> Future optimization opportunity</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Enhanced DataSync AI Recommendations with VMware optimization
    if config['migration_service'] == 'datasync':
        pattern_key = analyzer.determine_optimal_pattern(
            config['source_location'], 
            config['environment'], 
            config['migration_service']
        )
        
        datasync_scaling = analyzer.get_datasync_scaling_recommendations(pattern_key, config)
        
        if datasync_scaling['applicable']:
            st.markdown("### 🚀 Enhanced DataSync-Specific AI Recommendations")
            
            # Performance recommendations
            if datasync_scaling['is_bottleneck']:
                vmware_env = VMwareEnvironment(**config['vmware_env'])
                
                st.markdown(f"""
                <div class="corporate-card status-card-error">
                    <h4>🔴 DataSync Performance Bottleneck Detected</h4>
                    <p><strong>Issue:</strong> DataSync VM capacity is limiting your migration performance</p>
                    <p><strong>Current:</strong> {datasync_scaling['current_config']['instances']} × {config['service_size']} VM(s)</p>
                    <p><strong>Recommended:</strong> {datasync_scaling['optimal_config']['instances']} × {datasync_scaling['optimal_config']['size']} VM(s)</p>
                    <p><strong>Expected Improvement:</strong> {(datasync_scaling['optimal_config']['effective_throughput'] / datasync_scaling['current_config']['effective_throughput'] - 1) * 100:.0f}% faster transfer</p>
                    <p><strong>VMware Optimization:</strong> Current overhead: {vmware_env.memory_overcommit_ratio}x memory, {vmware_env.cpu_overcommit_ratio}x CPU overcommit</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="corporate-card status-card-success">
                    <h4>✅ DataSync Configuration Optimal</h4>
                    <p>Your current DataSync configuration is not the bottleneck in your migration pipeline.</p>
                    <p><strong>Effective Throughput:</strong> {datasync_scaling['current_config']['effective_throughput']:,.0f} Mbps</p>
                    <p><strong>VMware Environment:</strong> Well optimized for current workload</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Enhanced VMware-specific recommendations
            vmware_env = VMwareEnvironment(**config['vmware_env'])
            optimal_instances = datasync_scaling['optimal_config']['instances']
            
            vmware_recommendations = []
            
            if not vmware_env.sr_iov_enabled:
                vmware_recommendations.append("Enable SR-IOV for 50% network performance improvement")
            
            if vmware_env.storage_type != 'ssd':
                vmware_recommendations.append("Upgrade to SSD storage for 20-30% I/O performance boost")
            
            if vmware_env.memory_overcommit_ratio > 1.5:
                vmware_recommendations.append("Reduce memory overcommit to <1.5x for consistent performance")
            
            if vmware_env.cpu_overcommit_ratio > 2.0:
                vmware_recommendations.append("Reduce CPU overcommit to <2.0x to avoid scheduling delays")
            
            if not vmware_env.numa_optimization and optimal_instances > 1:
                vmware_recommendations.append("Enable NUMA optimization for multi-VM deployments")
            
            if vmware_recommendations:
                rec_content = ""
                for rec in vmware_recommendations:
                    rec_content += f"<p>• {rec}</p>"
                
                st.markdown(f"""
                <div class="corporate-card status-card-info">
                    <h4>🖥️ VMware Environment Optimization Recommendations</h4>
                    {rec_content}
                </div>
                """, unsafe_allow_html=True)
            
            # Enhanced cost vs performance analysis
            current_throughput = datasync_scaling['current_config']['effective_throughput']
            optimal_throughput = datasync_scaling['optimal_config']['effective_throughput']
            performance_gain = (optimal_throughput / current_throughput - 1) * 100 if current_throughput > 0 else 0
            
            current_instances = datasync_scaling['current_config']['instances']
            optimal_instances = datasync_scaling['optimal_config']['instances']
            cost_multiplier = optimal_instances / current_instances if current_instances > 0 else optimal_instances
            
            # Calculate migration time reduction
            current_time = analysis_results.get('enhanced_timing', 24)
            optimal_time = current_time / cost_multiplier if cost_multiplier > 1 else current_time
            time_savings = current_time - optimal_time
            
            st.markdown(f"""
            <div class="corporate-card">
                <h4>💰 Enhanced Cost vs Performance Trade-off Analysis</h4>
                <p><strong>Performance Gain:</strong> {performance_gain:.0f}% faster migration</p>
                <p><strong>Cost Impact:</strong> {cost_multiplier:.1f}x DataSync VM costs</p>
                <p><strong>Migration Time Reduction:</strong> {time_savings:.1f} hours saved</p>
                <p><strong>VMware Resource Impact:</strong> {optimal_instances}x resource requirements</p>
                <p><strong>ROI Analysis:</strong> Shorter migration window typically justifies higher VM costs for production workloads</p>
                <p><strong>Risk Mitigation:</strong> Faster migration reduces exposure window and business impact</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Enhanced summary card with network conditions
    waterfall_summary = analysis_results['waterfall_data']['summary']
    network_conditions = analysis_results['waterfall_data'].get('network_conditions')
    
    summary_content = f"""
    <h3>🎯 Enhanced Migration Strategy Summary</h3>
    <p><strong>Service:</strong> {waterfall_summary['service_name']}</p>
    <p><strong>Complexity:</strong> {ai_recommendations['migration_complexity'].title()}</p>
    <p><strong>AI Confidence:</strong> {ai_recommendations['confidence_level'].title()}</p>
    <p><strong>Timeline:</strong> {analysis_results.get('enhanced_timing', analysis_results['migration_time']['total_hours']):.1f} hours</p>
    <p><strong>Primary Bottleneck:</strong> {waterfall_summary['primary_bottleneck_layer'].title()}</p>
    <p><strong>Infrastructure Efficiency:</strong> {waterfall_summary['efficiency_percentage']:.1f}%</p>
    """
    
    if network_conditions and hasattr(network_conditions, 'time_of_day'):
        summary_content += f"""<p><strong>Current Network Conditions:</strong> {network_conditions.time_of_day.replace('_', ' ').title()}</p>"""
        if network_conditions.time_of_day == 'business_hours':
            summary_content += f"""<p><strong>Optimization Opportunity:</strong> Off-hours migration could reduce time by 30%</p>"""
    
    st.markdown(f"""
    <div class="corporate-card">
        {summary_content}
    </div>
    """, unsafe_allow_html=True)

def render_pattern_comparison_tab(analyzer: EnhancedNetworkAnalyzer, config: Dict):
    """Render enhanced pattern comparison analysis"""
    st.subheader("🔄 Enhanced Network Pattern Comparison")
    
    # Initialize clients with API keys if provided
    if config.get('aws_access_key') and config.get('aws_secret_key'):
        analyzer.pricing_client.initialize_client(config['aws_access_key'], config['aws_secret_key'])
    
    if config.get('claude_api_key'):
        analyzer.ai_client.api_key = config['claude_api_key']
    
    # Analyze all patterns with enhanced calculations
    with st.spinner("Analyzing network patterns with enhanced modeling..."):
        pattern_analyses = analyzer.analyze_all_patterns(config)
    
    if not pattern_analyses:
        st.error("No suitable patterns found for the selected configuration.")
        return None, None
    
    # Get enhanced AI recommendation
    with st.spinner("Getting enhanced AI recommendations..."):
        ai_recommendation = analyzer.get_ai_recommendation(pattern_analyses, config)
    
    # Display best pattern recommendation
    best_pattern = pattern_analyses[0]
    st.markdown(f"""
    <div class="best-pattern-highlight">
        🏆 ENHANCED AI RECOMMENDED: {best_pattern.pattern_name}
        <br>
        Confidence: {ai_recommendation['confidence_score']*100:.0f}% | 
        Cost: ${best_pattern.total_cost_usd:,.0f} | 
        Time: {best_pattern.migration_time_hours:.1f}h |
        Efficiency: {(best_pattern.effective_bandwidth_mbps / 25000 * 100):.1f}%
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced comparison metrics
    st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
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
            <div class="metric-value">{best_pattern.ai_recommendation_score:.3f}</div>
            <div class="metric-label">AI Score</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col6:
        efficiency_range = max(p.effective_bandwidth_mbps for p in pattern_analyses) - min(p.effective_bandwidth_mbps for p in pattern_analyses)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{efficiency_range:,.0f}</div>
            <div class="metric-label">Performance Range (Mbps)</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Enhanced detailed comparison table
    st.markdown("### 📊 Enhanced Pattern Comparison")
    
    comparison_data = []
    for pattern in pattern_analyses:
        comparison_data.append({
            'Pattern': pattern.pattern_name.split('→')[0].strip(),
            'Cost ($)': f"{pattern.total_cost_usd:,.0f}",
            'Time (h)': f"{pattern.migration_time_hours:.1f}",
            'Bandwidth (Mbps)': f"{pattern.effective_bandwidth_mbps:,.0f}",
            'Reliability': f"{pattern.reliability_score*100:.1f}%",
            'Complexity': f"{pattern.complexity_score*100:.0f}%",
            'AI Score': f"{pattern.ai_recommendation_score:.3f}",
            'Efficiency': f"{(pattern.effective_bandwidth_mbps / pattern.total_cost_usd * 1000):.1f}"
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    st.dataframe(df_comparison, use_container_width=True, hide_index=True)
    
    # Enhanced cost vs performance chart
    fig_scatter = go.Figure()
    
    for i, pattern in enumerate(pattern_analyses):
        color = '#059669' if i == 0 else '#3b82f6'  # Green for best, blue for others
        size = 25 if i == 0 else 18
        
        fig_scatter.add_trace(go.Scatter(
            x=[pattern.total_cost_usd],
            y=[pattern.migration_time_hours],
            mode='markers+text',
            marker=dict(
                size=size, 
                color=color,
                line=dict(width=2, color='white'),
                opacity=0.8
            ),
            text=[pattern.pattern_name.split('→')[0].strip()],
            textposition="top center",
            name=pattern.pattern_name.split('→')[0].strip(),
            hovertemplate=f"<b>{pattern.pattern_name}</b><br>" +
                         f"Cost: ${pattern.total_cost_usd:,.0f}<br>" +
                         f"Time: {pattern.migration_time_hours:.1f}h<br>" +
                         f"Bandwidth: {pattern.effective_bandwidth_mbps:,.0f} Mbps<br>" +
                         f"AI Score: {pattern.ai_recommendation_score:.3f}<br>" +
                         f"Efficiency: {(pattern.effective_bandwidth_mbps / pattern.total_cost_usd * 1000):.1f}<extra></extra>"
        ))
    
    fig_scatter.update_layout(
        title={
            'text': "Enhanced Migration Cost vs Time Analysis",
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
    """Render the enhanced complete network path diagram tab"""
    
    st.header("🌐 Enhanced Network Path Architecture")
    st.markdown("Comprehensive view of enterprise network architecture with real-time performance modeling")
    
    # Enhanced environment overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🏢 Production Environment**
        - **Source:** San Antonio Data Center
        - **Transit:** San Jose Data Center  
        - **Link:** 10Gbps Shared + 10Gbps DX
        - **Target:** AWS West 2 Production VPC
        - **DataSync Location:** VMware (SA/SJ)
        - **Latency:** <10ms end-to-end
        """)
    
    with col2:
        st.markdown("""
        **🔧 Non-Production Environment**
        - **Source:** San Jose Data Center
        - **Link:** 2Gbps Direct Connect
        - **Target:** AWS West 2 Non-Prod VPC
        - **DataSync Location:** VMware (SJ)
        - **Latency:** <15ms end-to-end
        """)
    
    with col3:
        st.markdown("""
        **🎯 Enhanced Modeling**
        - **Real-time Congestion:** Time-of-day factors
        - **VMware Overhead:** Detailed virtualization impact
        - **Protocol Efficiency:** Dynamic calculations
        - **Error Handling:** Retry overhead modeling
        - **Cost Optimization:** Reserved vs on-demand
        """)
    
    # Enhanced network topology diagram
    st.subheader("🏗️ Enhanced Network Topology")
    topology_fig = create_network_topology_diagram()
    st.plotly_chart(topology_fig, use_container_width=True)
    
    # Enhanced service flows
    st.subheader("🔄 Enhanced Service Flow Patterns")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        service_flows_fig = create_service_flows_diagram()
        st.plotly_chart(service_flows_fig, use_container_width=True)
    
    with col2:
        st.markdown("""
        **Enhanced Service Locations:**
        
        **🏢 On-Premises (VMware)**
        - DataSync Agent with SR-IOV
        - NUMA-optimized deployments
        - SSD storage optimization
        
        **☁️ AWS VPC**
        - FSx File Systems with performance mode
        - DMS with Multi-AZ deployment
        - Storage Gateway with cache optimization
        
        **🌐 AWS Managed**
        - VPC Endpoints with PrivateLink
        - S3 with transfer acceleration
        - Direct Connect with redundancy
        """)
    
    # Enhanced service comparison table
    st.subheader("📊 Enhanced Service Path Comparison")
    service_df = create_service_comparison_table()
    
    # Add performance columns
    service_df['Expected Latency'] = ['8-12ms', '6-10ms', '5-8ms', '8-15ms', '10-18ms']
    service_df['Efficiency Rating'] = ['Good', 'Excellent', 'Very Good', 'Good', 'Fair']
    
    st.dataframe(service_df, use_container_width=True, hide_index=True)
    
    # Enhanced architecture insights
    st.subheader("💡 Enhanced Architecture Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🚀 Performance Characteristics**
        - Production: 10Gbps with <10ms latency
        - Non-Production: 2Gbps with <15ms latency
        - DataSync: VMware-optimized with SR-IOV
        - Real-time congestion modeling
        - Protocol-specific efficiency calculations
        """)
    
    with col2:
        st.markdown("""
        **🔒 Security & Compliance**
        - Private connectivity via DX
        - VPC isolation for environments
        - Encrypted data in transit (TLS 1.3)
        - AWS managed endpoints
        - Network segmentation and monitoring
        """)
    
    with col3:
        st.markdown("""
        **💰 Cost Optimization**
        - Shared San Jose infrastructure
        - Right-sized DX connections
        - Service-specific routing
        - Efficient data transfer patterns
        - Reserved instance pricing
        """)
    
    # Enhanced network path details
    with st.expander("📋 Enhanced Network Path Analysis", expanded=False):
        
        st.markdown("### Production Environment Enhanced Data Flow")
        st.markdown("""
        1. **San Antonio DC** → 10Gbps Shared Link (MPLS) → **San Jose DC**
           - *Latency: 15ms, Efficiency: 96%, Congestion-aware routing*
        2. **San Jose DC** → 10Gbps Direct Connect → **AWS West 2**
           - *Latency: 8ms, Efficiency: 99%, Dedicated bandwidth*
        3. **AWS West 2** → **Production VPC** → **Target Services (S3)**
           - *Latency: 2ms, VPC routing optimization*
        
        **Enhanced DataSync Agent:** Deployed on VMware with SR-IOV, NUMA optimization, SSD storage
        **Total End-to-End Latency:** 6-10ms, **Efficiency:** 92-95%
        """)
        
        st.markdown("### Non-Production Environment Enhanced Data Flow")
        st.markdown("""
        1. **San Jose DC** → 2Gbps Direct Connect → **AWS West 2**
           - *Latency: 8ms, Efficiency: 98%, Dedicated connection*
        2. **AWS West 2** → **Non-Prod VPC** → **Target Services (S3)**
           - *Latency: 2ms, VPC endpoint optimization*
        
        **Enhanced DataSync Agent:** Single VM deployment with performance tuning
        **Total End-to-End Latency:** 8-12ms, **Efficiency:** 88-92%
        """)
        
        st.markdown("### Enhanced Service-Specific Routing")
        st.markdown("""
        - **VPC Endpoint:** Private connection to S3, bypassing internet, with PrivateLink optimization
        - **FSx:** High-performance file system within VPC, with performance mode enabled
        - **DataSync:** VMware-based agent with hardware acceleration and error handling
        - **DMS:** Database replication through VPC with Multi-AZ deployment for reliability
        - **Storage Gateway:** Hybrid storage bridge with intelligent caching and compression
        """)

def render_database_guidance_tab(config: Dict):
    """Render enhanced database engineer guidance"""
    st.subheader("📚 Enhanced Database Engineer's Migration Guide")
    
    database_scenario = config.get('database_scenario', 'mysql_oltp_rds')
    
    # Get analyzer instance to access database scenarios
    analyzer = EnhancedNetworkAnalyzer()
    selected_scenario = analyzer.database_scenarios[database_scenario]
    
    # Enhanced Database-specific guidance with new requirements
    guidance_content = {
        'mysql_oltp_rds': {
            'title': '🔄 MySQL OLTP Database → RDS MySQL (Enhanced)',
            'key_considerations': [
                "Binary log settings for replication consistency with enhanced monitoring",
                "InnoDB buffer pool warming strategies for performance optimization",
                "Connection pool configuration with enhanced scaling parameters",
                "Read replica lag monitoring with automated alerting",
                "Parameter group optimization for RDS with performance insights",
                "Enhanced backup strategies with point-in-time recovery validation"
            ],
            'network_requirements': {
                'min_bandwidth': '500 Mbps',
                'max_latency': '10ms',
                'max_jitter': '2ms',
                'consistency': 'Strict (ACID compliance required)',
                'burst_factor': '1.5x'
            },
            'recommended_approach': 'Use AWS DMS with full load + CDC for minimal downtime, enhanced with performance monitoring',
            'testing_strategy': [
                "Test replication lag under peak load with burst scenarios",
                "Validate foreign key constraints with referential integrity checks",
                "Performance test critical queries under various network conditions",
                "Failover/failback procedures with automated monitoring",
                "RDS monitoring and alerting setup with custom metrics",
                "Network latency impact testing on transaction performance"
            ],
            'migration_steps': [
                "Setup DMS replication instance with enhanced monitoring",
                "Create source and target endpoints with SSL encryption",
                "Configure CDC for ongoing replication with lag alerting",
                "Perform initial data load with validation checkpoints",
                "Monitor lag and validate data with automated testing",
                "Cut over during maintenance window with rollback procedures"
            ],
            'enhanced_features': [
                "Real-time performance monitoring during migration",
                "Automated rollback procedures for migration failures",
                "Network optimization recommendations based on latency/jitter",
                "Custom alerting for replication lag and connection issues"
            ]
        },
        'postgresql_analytics_rds': {
            'title': '📊 PostgreSQL Analytics → RDS PostgreSQL (Enhanced)',
            'key_considerations': [
                "Vacuum and analyze statistics with automated scheduling",
                "Extension compatibility assessment (PostGIS, etc.)",
                "Large table partitioning strategy with performance optimization",
                "Query performance optimization with enhanced monitoring",
                "RDS parameter tuning for analytics workloads with burst scenarios",
                "Connection pooling for analytics queries with resource management"
            ],
            'network_requirements': {
                'min_bandwidth': '1000 Mbps',
                'max_latency': '50ms',
                'max_jitter': '10ms',
                'consistency': 'Eventual (some lag acceptable for analytics)',
                'burst_factor': '2.0x'
            },
            'recommended_approach': 'Combination of DMS for initial load + DataSync for large data files with performance optimization',
            'testing_strategy': [
                "Validate complex analytical queries under network constraints",
                "Test ETL pipeline compatibility with enhanced monitoring",
                "Check data type conversions with validation scripts",
                "Performance baseline comparison with automated benchmarking",
                "Extension functionality validation with comprehensive testing",
                "Network bandwidth utilization testing for large queries"
            ],
            'migration_steps': [
                "Assess extension compatibility with automated tools",
                "Export/import custom functions with validation",
                "Migrate schema using DMS SCT with optimization",
                "Bulk load historical data with parallel processing",
                "Setup incremental analytics refresh with monitoring",
                "Validate query performance with automated benchmarks"
            ],
            'enhanced_features': [
                "Automated performance baseline comparison",
                "Intelligent query routing based on network conditions",
                "Enhanced monitoring for analytical workload patterns",
                "Automated capacity scaling recommendations"
            ]
        },
        'oracle_enterprise_rds': {
            'title': '🏢 Oracle Enterprise → RDS Oracle (Enhanced)',
            'key_considerations': [
                "Oracle-specific features compatibility with comprehensive assessment",
                "PL/SQL code conversion needs with automated analysis",
                "Tablespace and datafile strategy with optimization",
                "RAC to RDS conversion complexity with detailed planning",
                "License considerations for RDS Oracle with cost optimization",
                "Enhanced backup and recovery procedures for enterprise workloads"
            ],
            'network_requirements': {
                'min_bandwidth': '2000 Mbps',
                'max_latency': '5ms',
                'max_jitter': '1ms',
                'consistency': 'Strict (Enterprise SLA requirements)',
                'burst_factor': '1.8x'
            },
            'recommended_approach': 'AWS DMS with SCT for schema conversion + enhanced testing and validation',
            'testing_strategy': [
                "Schema conversion validation with automated tools",
                "Application compatibility testing with comprehensive scenarios",
                "Performance regression testing with detailed analysis",
                "Disaster recovery validation with automated procedures",
                "Oracle-specific feature testing with compatibility matrix",
                "Enterprise workload testing under various network conditions"
            ],
            'migration_steps': [
                "Run AWS SCT assessment with detailed reporting",
                "Convert schema and procedures with validation",
                "Setup DMS replication with enhanced monitoring",
                "Migrate data with full load + CDC and validation",
                "Application connection string updates with testing",
                "Performance tuning and optimization with monitoring"
            ],
            'enhanced_features': [
                "Enterprise-grade monitoring and alerting",
                "Automated disaster recovery testing",
                "Performance optimization recommendations",
                "Compliance and audit trail management"
            ]
        },
        'sqlserver_enterprise_ec2': {
            'title': '🪟 SQL Server Enterprise → EC2 (Enhanced)',
            'key_considerations': [
                "Windows licensing on EC2 with cost optimization",
                "AlwaysOn Availability Groups setup with enhanced monitoring",
                "Storage configuration (EBS optimization) with performance tuning",
                "Security group and network configuration with best practices",
                "Backup strategy for EC2 environment with automation",
                "Enhanced monitoring and performance optimization"
            ],
            'network_requirements': {
                'min_bandwidth': '1500 Mbps',
                'max_latency': '8ms',
                'max_jitter': '2ms',
                'consistency': 'Strict (Enterprise requirements)',
                'burst_factor': '1.6x'
            },
            'recommended_approach': 'Native SQL Server tools + DMS for validation with enhanced monitoring',
            'testing_strategy': [
                "AlwaysOn configuration testing with failover scenarios",
                "Application driver compatibility with comprehensive testing",
                "Performance under load testing with detailed analysis",
                "Backup and restore procedures with automation",
                "Disaster recovery validation with enhanced monitoring",
                "Network performance testing for AlwaysOn synchronization"
            ],
            'migration_steps': [
                "Setup EC2 instances with SQL Server and optimization",
                "Configure storage and networking with best practices",
                "Setup native replication or log shipping with monitoring",
                "Migrate databases using native backup/restore with validation",
                "Configure AlwaysOn if required with enhanced monitoring",
                "Application cutover and validation with automated testing"
            ],
            'enhanced_features': [
                "Automated AlwaysOn monitoring and alerting",
                "Performance optimization recommendations",
                "Enhanced backup and recovery automation",
                "Cost optimization for EC2 and storage"
            ]
        },
        'mongodb_cluster_documentdb': {
            'title': '🍃 MongoDB Cluster → DocumentDB (Enhanced)',
            'key_considerations': [
                "DocumentDB API compatibility assessment with automated tools",
                "Index strategy optimization for DocumentDB with performance testing",
                "Connection string and driver updates with validation",
                "Aggregation pipeline compatibility with comprehensive testing",
                "Change streams configuration with monitoring",
                "Enhanced security and compliance configuration"
            ],
            'network_requirements': {
                'min_bandwidth': '1500 Mbps',
                'max_latency': '20ms',
                'max_jitter': '5ms',
                'consistency': 'Configurable (adjust read/write concerns)',
                'burst_factor': '1.4x'
            },
            'recommended_approach': 'Native MongoDB tools + DMS for validation with enhanced compatibility testing',
            'testing_strategy': [
                "API compatibility verification with automated testing",
                "Application driver testing with comprehensive scenarios",
                "Performance comparison testing with detailed analysis",
                "Aggregation pipeline validation with monitoring",
                "Change streams functionality with testing",
                "Network latency impact on read/write operations"
            ],
            'migration_steps': [
                "Assess DocumentDB compatibility with automated tools",
                "Setup DocumentDB cluster with optimization",
                "Use mongodump/mongorestore for migration with validation",
                "Validate data integrity with automated checks",
                "Update application connection strings with testing",
                "Performance testing and optimization with monitoring"
            ],
            'enhanced_features': [
                "Automated compatibility assessment tools",
                "Performance optimization recommendations",
                "Enhanced monitoring for DocumentDB clusters",
                "Automated scaling recommendations"
            ]
        },
        'mysql_analytics_aurora': {
            'title': '📊 MySQL Analytics → Aurora MySQL (Enhanced)',
            'key_considerations': [
                "Aurora MySQL engine version compatibility with feature assessment",
                "Parallel query optimization for analytics with performance tuning",
                "Aurora Serverless for variable workloads with cost optimization",
                "Global database for multi-region analytics with enhanced replication",
                "Aurora ML integration opportunities with automated recommendations",
                "Enhanced monitoring and performance insights configuration"
            ],
            'network_requirements': {
                'min_bandwidth': '1200 Mbps',
                'max_latency': '25ms',
                'max_jitter': '8ms',
                'consistency': 'Eventual (analytics workloads)',
                'burst_factor': '2.2x'
            },
            'recommended_approach': 'AWS DMS for continuous replication with enhanced analytics optimization',
            'testing_strategy': [
                "Parallel query performance testing with detailed analysis",
                "Analytics workload validation with comprehensive scenarios",
                "Aurora Serverless scaling testing with monitoring",
                "Cross-region replication testing with performance validation",
                "Performance baseline comparison with automated benchmarking",
                "Network bandwidth optimization for large analytical queries"
            ],
            'migration_steps': [
                "Setup Aurora MySQL cluster with analytics optimization",
                "Configure DMS for initial load + CDC with monitoring",
                "Enable parallel query for analytics with tuning",
                "Optimize for analytical workloads with performance insights",
                "Setup read replicas if needed with enhanced monitoring",
                "Application cutover and validation with automated testing"
            ],
            'enhanced_features': [
                "Automated performance optimization for analytics",
                "Intelligent scaling recommendations",
                "Enhanced monitoring for Aurora clusters",
                "ML-powered query optimization recommendations"
            ]
        },
        'postgresql_oltp_aurora': {
            'title': '🔄 PostgreSQL OLTP → Aurora PostgreSQL (Enhanced)',
            'key_considerations': [
                "Aurora PostgreSQL compatibility with feature assessment",
                "Connection pooling optimization with enhanced scaling",
                "Aurora Serverless for variable OLTP loads with cost optimization",
                "Point-in-time recovery configuration with automated testing",
                "Performance Insights setup with custom metrics",
                "Enhanced monitoring and alerting for OLTP workloads"
            ],
            'network_requirements': {
                'min_bandwidth': '800 Mbps',
                'max_latency': '12ms',
                'max_jitter': '3ms',
                'consistency': 'Strict (OLTP requirements)',
                'burst_factor': '1.3x'
            },
            'recommended_approach': 'AWS DMS with minimal downtime and enhanced monitoring',
            'testing_strategy': [
                "OLTP transaction testing with comprehensive scenarios",
                "Connection pooling optimization with performance testing",
                "Failover testing with automated procedures",
                "Performance monitoring validation with custom metrics",
                "Application compatibility testing with detailed analysis",
                "Network latency impact on transaction performance"
            ],
            'migration_steps': [
                "Setup Aurora PostgreSQL cluster with OLTP optimization",
                "Configure DMS replication with enhanced monitoring",
                "Test application connectivity with validation",
                "Optimize for OLTP workloads with performance tuning",
                "Setup monitoring and alerting with custom dashboards",
                "Planned cutover execution with automated rollback"
            ],
            'enhanced_features': [
                "Automated OLTP performance optimization",
                "Intelligent connection pool management",
                "Enhanced monitoring for transaction performance",
                "Automated backup and recovery testing"
            ]
        },
        'mariadb_oltp_rds': {
            'title': '🗄️ MariaDB OLTP → RDS MariaDB (Enhanced)',
            'key_considerations': [
                "MariaDB version compatibility with feature assessment",
                "Storage engine considerations with performance optimization",
                "Connection handling optimization with enhanced scaling",
                "Replication configuration with monitoring and alerting",
                "Parameter group optimization with performance insights",
                "Enhanced backup and recovery procedures"
            ],
            'network_requirements': {
                'min_bandwidth': '600 Mbps',
                'max_latency': '10ms',
                'max_jitter': '2ms',
                'consistency': 'Strict (OLTP requirements)',
                'burst_factor': '1.4x'
            },
            'recommended_approach': 'AWS DMS for live migration with enhanced monitoring and validation',
            'testing_strategy': [
                "MariaDB-specific feature testing with comprehensive validation",
                "Storage engine validation with performance testing",
                "Replication lag monitoring with automated alerting",
                "Performance comparison with detailed analysis",
                "Application compatibility verification with testing",
                "Network performance impact on replication synchronization"
            ],
            'migration_steps': [
                "Setup RDS MariaDB instance with optimization",
                "Configure DMS endpoints with enhanced security",
                "Initialize replication with monitoring setup",
                "Monitor and validate data with automated checks",
                "Application connection updates with testing",
                "Cutover and post-migration validation with monitoring"
            ],
            'enhanced_features': [
                "Automated MariaDB performance optimization",
                "Enhanced replication monitoring and alerting",
                "Intelligent parameter tuning recommendations",
                "Automated backup and recovery validation"
            ]
        }
    }
    
    selected_guidance = guidance_content.get(database_scenario, guidance_content['mysql_oltp_rds'])
    
    # Display enhanced guidance
    st.markdown(f"""
    <div class="corporate-card">
        <h3>{selected_guidance['title']}</h3>
        <p><strong>Recommended Approach:</strong> {selected_guidance['recommended_approach']}</p>
        <p><strong>Migration Complexity:</strong> {selected_scenario['migration_complexity'].title()}</p>
        <p><strong>Downtime Sensitivity:</strong> {selected_scenario['downtime_sensitivity'].title()}</p>
        <p><strong>Burst Requirement:</strong> {selected_scenario.get('burst_requirement_factor', 1.0)}x baseline bandwidth</p>
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
            <h3>🔧 Enhanced Technical Considerations</h3>
            {considerations_content}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="corporate-card status-card-success">
            <h3>📋 Enhanced Testing Strategy</h3>
            {testing_content}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        requirements = selected_guidance['network_requirements']
        
        st.markdown(f"""
        <div class="corporate-card">
            <h3>🌐 Enhanced Network Requirements</h3>
            
            <div class="metric-card" style="margin: 1rem 0;">
                <div class="metric-value" style="color: var(--secondary-blue);">{requirements['min_bandwidth']}</div>
                <div class="metric-label">Minimum Bandwidth</div>
            </div>
            
            <div class="metric-card" style="margin: 1rem 0;">
                <div class="metric-value" style="color: var(--warning-orange);">{requirements['max_latency']}</div>
                <div class="metric-label">Maximum Latency</div>
            </div>
            
            <div class="metric-card" style="margin: 1rem 0;">
                <div class="metric-value" style="color: var(--error-red);">{requirements['max_jitter']}</div>
                <div class="metric-label">Maximum Jitter</div>
            </div>
            
            <div class="metric-card" style="margin: 1rem 0;">
                <div class="metric-value" style="color: var(--success-green);">{requirements['burst_factor']}</div>
                <div class="metric-label">Burst Factor</div>
            </div>
            
            <div class="metric-card" style="margin: 1rem 0;">
                <div class="metric-value" style="color: var(--primary-blue); font-size: 1.2em;">{requirements['consistency']}</div>
                <div class="metric-label">Consistency Model</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced migration steps
        steps_content = ""
        for i, step in enumerate(selected_guidance['migration_steps'], 1):
            steps_content += f"<p><strong>{i}.</strong> {step}</p>"
        
        st.markdown(f"""
        <div class="corporate-card status-card-warning">
            <h3>📝 Enhanced Migration Steps</h3>
            {steps_content}
        </div>
        """, unsafe_allow_html=True)
    
    # Enhanced features section
    if 'enhanced_features' in selected_guidance:
        st.markdown("### 🚀 Enhanced Features & Capabilities")
        
        features_content = ""
        for feature in selected_guidance['enhanced_features']:
            features_content += f"<p>• {feature}</p>"
        
        st.markdown(f"""
        <div class="corporate-card status-card-info">
            <h3>✨ New Enhancement Features</h3>
            {features_content}
        </div>
        """, unsafe_allow_html=True)

def render_ai_insights_tab(pattern_analyses: List[PatternAnalysis], ai_recommendation: Dict, config: Dict):
    """Render enhanced AI insights and recommendations"""
    st.subheader("🤖 Enhanced AI-Powered Migration Insights")
    
    if not pattern_analyses or not ai_recommendation:
        st.warning("Please run pattern comparison first to get enhanced AI insights.")
        return
    
    # Enhanced AI Recommendation Summary
    st.markdown(f"""
    <div class="corporate-card">
        <h3>🎯 Enhanced AI Recommendation: {ai_recommendation['recommended_pattern'].replace('_', ' ').title()}</h3>
        <p><strong>Confidence Level:</strong> {ai_recommendation['confidence_score']*100:.0f}%</p>
        <p><strong>Reasoning:</strong> {ai_recommendation['reasoning']}</p>
        <p><strong>Timeline Estimate:</strong> {ai_recommendation.get('timeline_estimate', 'Not specified')}</p>
        <p><strong>Performance Expectations:</strong> {ai_recommendation.get('performance_expectations', 'Standard performance characteristics')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced Database-Specific Insights
    database_scenario = config.get('database_scenario', 'mysql_oltp_rds')
    
    # Get analyzer instance to access database scenarios
    analyzer = EnhancedNetworkAnalyzer()
    db_info = analyzer.database_scenarios[database_scenario]
    
    st.markdown(f"""
    <div class="corporate-card status-card-info">
        <h3>🗄️ Enhanced Database-Specific Considerations for {db_info['name']}</h3>
        <p><strong>Target Service:</strong> {db_info['target_service']}</p>
        <p><strong>Database Insights:</strong> {ai_recommendation['database_considerations']}</p>
        <p><strong>Risk Assessment:</strong> {ai_recommendation['risk_assessment']}</p>
        <p><strong>Cost Justification:</strong> {ai_recommendation['cost_justification']}</p>
        <p><strong>Burst Requirements:</strong> {db_info.get('burst_requirement_factor', 1.0)}x baseline bandwidth for peak loads</p>
        <p><strong>Connection Limits:</strong> {db_info.get('concurrent_connection_limit', 1000)} concurrent connections supported</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced Implementation Phases
    if 'implementation_phases' in ai_recommendation:
        st.markdown("### 📋 Enhanced Implementation Phases")
        
        phases_content = ""
        for i, phase in enumerate(ai_recommendation['implementation_phases'], 1):
            phases_content += f"<p><strong>Phase {i}:</strong> {phase}</p>"
        
        st.markdown(f"""
        <div class="corporate-card status-card-warning">
            <h3>🔄 Recommended Implementation Approach</h3>
            {phases_content}
            <p><strong>Estimated Timeline:</strong> {ai_recommendation.get('timeline_estimate', '4-6 weeks total')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Enhanced pattern deep dive
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
            <h3>✅ Enhanced Pattern Advantages</h3>
            {pros_content}
            <p><strong>AI Recommendation Score:</strong> {best_pattern.ai_recommendation_score:.3f}/1.0</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="corporate-card status-card-info">
            <h3>🎯 Enhanced Use Cases</h3>
            {use_cases_content}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        cons_content = ""
        for con in best_pattern.cons:
            cons_content += f"<p>• {con}</p>"
        
        optimization_content = ""
        if best_pattern.complexity_score > 0.7:
            optimization_content += "<p>• High complexity - consider professional services engagement and phased approach</p>"
        if best_pattern.total_cost_usd > 5000:
            optimization_content += "<p>• High cost - evaluate phased migration approach and reserved instance pricing</p>"
        if best_pattern.migration_time_hours > config['max_downtime_hours']:
            optimization_content += "<p>• Exceeds downtime SLA - consider incremental migration with live replication</p>"
        else:
            optimization_content += "<p>• Meets downtime requirements with recommended buffer time</p>"
        
        # Add VMware-specific optimizations for DataSync
        if config['migration_service'] == 'datasync':
            vmware_env = config.get('vmware_env', {})
            if not vmware_env.get('sr_iov_enabled', True):
                optimization_content += "<p>• Enable SR-IOV for 50% network performance improvement</p>"
            if vmware_env.get('storage_type', 'ssd') != 'ssd':
                optimization_content += "<p>• Upgrade to SSD storage for 20-30% I/O performance boost</p>"
        
        st.markdown(f"""
        <div class="corporate-card status-card-warning">
            <h3>⚠️ Enhanced Considerations & Limitations</h3>
            {cons_content}
        </div>
        """, unsafe_allow_html=True)
        
        optimization_card_type = "status-card-error" if best_pattern.migration_time_hours > config['max_downtime_hours'] else "status-card-success"
        
        st.markdown(f"""
        <div class="corporate-card {optimization_card_type}">
            <h3>💡 Enhanced Optimization Recommendations</h3>
            {optimization_content}
        </div>
        """, unsafe_allow_html=True)
    
    # Enhanced performance expectations and monitoring
    st.markdown("### 📊 Enhanced Performance Expectations & Monitoring")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="corporate-card">
            <h3>🎯 Expected Performance Metrics</h3>
            <p><strong>Effective Bandwidth:</strong> {best_pattern.effective_bandwidth_mbps:,.0f} Mbps</p>
            <p><strong>Migration Time:</strong> {best_pattern.migration_time_hours:.1f} hours</p>
            <p><strong>Reliability Score:</strong> {best_pattern.reliability_score*100:.1f}%</p>
            <p><strong>Total Cost:</strong> ${best_pattern.total_cost_usd:,.0f}</p>
            <p><strong>Cost per GB:</strong> ${best_pattern.total_cost_usd/config['data_size_gb']:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="corporate-card status-card-info">
            <h3>🔍 Enhanced Monitoring Recommendations</h3>
            <p>• Real-time bandwidth utilization monitoring</p>
            <p>• End-to-end latency and jitter tracking</p>
            <p>• Database replication lag alerting (if applicable)</p>
            <p>• Error rate and retry monitoring</p>
            <p>• Cost tracking and optimization alerts</p>
            <p>• Performance baseline comparison</p>
            <p>• Automated health checks and validation</p>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# ENHANCED MAIN APPLICATION
# =============================================================================

def main():
    """Enhanced main application entry point"""
    render_corporate_header()
    
    # Enhanced sidebar configuration
    config = render_enhanced_sidebar_controls()
    
    # Initialize enhanced analyzer
    analyzer = EnhancedNetworkAnalyzer()
    
    # Initialize API clients with credentials from secrets
    if config.get('aws_access_key') and config.get('aws_secret_key'):
        analyzer.pricing_client.initialize_client(config['aws_access_key'], config['aws_secret_key'])
    
    if config.get('claude_api_key'):
        analyzer.ai_client.api_key = config['claude_api_key']
    
    # Enhanced 6 tabs with improved functionality
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "💧 Enhanced Infrastructure Analysis",
        "⏱️ Enhanced Migration Analysis",
        "🤖 Enhanced AI Recommendations",
        "🔄 Enhanced Pattern Comparison",
        "🌐 Enhanced Network Architecture",
        "📚 Enhanced Database Guide"
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
                <p>Please run Enhanced Infrastructure Analysis first to proceed with Migration Analysis.</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab3:
        if 'analysis_results' in locals():
            render_ai_recommendations_tab(config, analysis_results, analyzer)
        else:
            st.markdown("""
            <div class="corporate-card status-card-warning">
                <h3>⚠️ Previous Analysis Required</h3>
                <p>Please complete Enhanced Infrastructure and Migration Analysis first to get AI recommendations.</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab4:
        pattern_analyses, ai_recommendation = render_pattern_comparison_tab(analyzer, config)
        
        # Store results in session state for use in other tabs
        if pattern_analyses and ai_recommendation:
            st.session_state['pattern_analyses'] = pattern_analyses
            st.session_state['ai_recommendation'] = ai_recommendation
    
    with tab5:
        # Enhanced Network Architecture Diagram
        render_network_path_diagram_tab()
    
    with tab6:
        render_database_guidance_tab(config)
        
        # Enhanced AI insights if pattern comparison has been run
        if 'pattern_analyses' in st.session_state and 'ai_recommendation' in st.session_state:
            st.markdown("---")
            render_ai_insights_tab(
                st.session_state['pattern_analyses'], 
                st.session_state['ai_recommendation'], 
                config
            )

if __name__ == "__main__":
    main()