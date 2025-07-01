import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import math
import json
import requests
import boto3
from typing import Dict, List, Tuple, Optional
import asyncio
import aiohttp
from dataclasses import dataclass
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Enhanced AWS Migration Network Analyzer",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS styling (preserving original + new styles)
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(15,23,42,0.3);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .pattern-card, .service-warning-card, .network-path-card, .recommendation-card, .waterfall-card, .service-card, .infrastructure-card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 6px;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border-left: 3px solid #3b82f6;
        border: 1px solid #e5e7eb;
    }
    
    .service-warning-card {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left: 3px solid #f59e0b;
    }
    
    .infrastructure-card {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border-left: 3px solid #16a34a;
    }
    
    .pattern-comparison-card {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #3b82f6;
        border: 1px solid #e2e8f0;
    }
    
    .ai-recommendation-card {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #f59e0b;
        border: 1px solid #f3f4f6;
    }
    
    .cost-analysis-card {
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #16a34a;
        border: 1px solid #f3f4f6;
    }
    
    .database-scenario-card {
        background: linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%);
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #6366f1;
        border: 1px solid #f3f4f6;
    }
    
    .network-metric-card {
        background: white;
        padding: 1rem;
        border-radius: 6px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border: 1px solid #e5e7eb;
    }
    
    .best-pattern-highlight {
        background: linear-gradient(135deg, #065f46 0%, #047857 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

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
                "model": "claude-3-sonnet-20240229",
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

class EnhancedNetworkAnalyzer:
    """Comprehensive network analyzer with realistic infrastructure modeling and migration services + AI enhancements"""
    
    def __init__(self):
        # PRESERVING ALL ORIGINAL CHARACTERISTICS
        
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
        
        # Comprehensive Migration Services (ALL ORIGINAL SERVICES PRESERVED)
        self.migration_services = {
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
                'database_compatibility': {
                    'file_based_backups': True,
                    'live_replication': False,
                    'transaction_logs': True
                },
                'sizes': {
                    'small': {
                        'vcpu': 2, 'memory_gb': 4, 'throughput_mbps': 250, 'cost_per_hour': 0.042,
                        'vpc_endpoint_throughput_reduction': 0.1,
                        'optimal_file_size_mb': '1-100',
                        'concurrent_transfers': 8,
                        'tcp_connections': 8,
                        'instance_type': 'm5.large'
                    },
                    'medium': {
                        'vcpu': 2, 'memory_gb': 8, 'throughput_mbps': 500, 'cost_per_hour': 0.085,
                        'vpc_endpoint_throughput_reduction': 0.08,
                        'optimal_file_size_mb': '100-1000',
                        'concurrent_transfers': 16,
                        'tcp_connections': 16,
                        'instance_type': 'm5.xlarge'
                    },
                    'large': {
                        'vcpu': 4, 'memory_gb': 16, 'throughput_mbps': 1000, 'cost_per_hour': 0.17,
                        'vpc_endpoint_throughput_reduction': 0.05,
                        'optimal_file_size_mb': '1000+',
                        'concurrent_transfers': 32,
                        'tcp_connections': 32,
                        'instance_type': 'm5.2xlarge'
                    },
                    'xlarge': {
                        'vcpu': 8, 'memory_gb': 32, 'throughput_mbps': 2000, 'cost_per_hour': 0.34,
                        'vpc_endpoint_throughput_reduction': 0.03,
                        'optimal_file_size_mb': '1000+',
                        'concurrent_transfers': 64,
                        'tcp_connections': 64,
                        'instance_type': 'm5.4xlarge'
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
        
        # Database-specific migration scenarios (NEW)
        self.database_scenarios = {
            'mysql_oltp': {
                'name': 'MySQL OLTP Database',
                'workload_type': 'oltp',
                'latency_sensitivity': 'high',
                'bandwidth_requirement': 'medium',
                'consistency_requirement': 'strict',
                'recommended_services': ['dms'],
                'min_bandwidth_mbps': 500,
                'max_tolerable_latency_ms': 10
            },
            'postgresql_analytics': {
                'name': 'PostgreSQL Analytics',
                'workload_type': 'olap',
                'latency_sensitivity': 'medium',
                'bandwidth_requirement': 'high',
                'consistency_requirement': 'eventual',
                'recommended_services': ['dms', 'datasync'],
                'min_bandwidth_mbps': 1000,
                'max_tolerable_latency_ms': 50
            },
            'oracle_enterprise': {
                'name': 'Oracle Enterprise Database',
                'workload_type': 'mixed',
                'latency_sensitivity': 'very_high',
                'bandwidth_requirement': 'high',
                'consistency_requirement': 'strict',
                'recommended_services': ['dms'],
                'min_bandwidth_mbps': 2000,
                'max_tolerable_latency_ms': 5
            },
            'mongodb_cluster': {
                'name': 'MongoDB Cluster',
                'workload_type': 'mixed',
                'latency_sensitivity': 'medium',
                'bandwidth_requirement': 'high',
                'consistency_requirement': 'eventual',
                'recommended_services': ['dms', 'datasync'],
                'min_bandwidth_mbps': 1500,
                'max_tolerable_latency_ms': 20
            }
        }
        
        # Initialize new clients
        self.pricing_client = AWSPricingClient()
        self.ai_client = ClaudeAIClient()
    
    # PRESERVING ALL ORIGINAL METHODS
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
        """Calculate realistic bandwidth waterfall with detailed infrastructure impact (ORIGINAL METHOD PRESERVED)"""
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
            'type': 'positive',
            'layer': 'nic'
        })
        
        # Step 2: NIC Real-World Efficiency
        nic_efficient_bandwidth = theoretical_max * nic_char['real_world_efficiency']
        nic_reduction = theoretical_max - nic_efficient_bandwidth
        steps.append({
            'name': f'NIC Efficiency ({nic_char["real_world_efficiency"]*100:.1f}%)',
            'value': -nic_reduction,
            'cumulative': nic_efficient_bandwidth,
            'type': 'negative',
            'layer': 'nic'
        })
        
        # Step 3: Operating System Network Stack
        os_efficient_bandwidth = nic_efficient_bandwidth * os_char['tcp_stack_efficiency']
        os_reduction = nic_efficient_bandwidth - os_efficient_bandwidth
        steps.append({
            'name': f'OS Stack ({os_char["name"]})',
            'value': -os_reduction,
            'cumulative': os_efficient_bandwidth,
            'type': 'negative',
            'layer': 'os'
        })
        
        # Step 4: CPU Utilization Impact
        cpu_utilization = (os_efficient_bandwidth / 1000) * nic_char['cpu_utilization_per_gbps']
        cpu_impact_factor = max(0.8, 1 - max(0, (cpu_utilization - 0.6) * 1.5)) if cpu_utilization > 0.6 else 1.0
        cpu_adjusted_bandwidth = os_efficient_bandwidth * cpu_impact_factor
        cpu_reduction = os_efficient_bandwidth - cpu_adjusted_bandwidth
        steps.append({
            'name': f'CPU Impact ({cpu_utilization*100:.1f}% util)',
            'value': -cpu_reduction,
            'cumulative': cpu_adjusted_bandwidth,
            'type': 'negative',
            'layer': 'os'
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
            'type': 'negative',
            'layer': 'lan'
        })
        
        # Step 6: WAN Provider
        wan_bandwidth = min(lan_limited_bandwidth, pattern['committed_bandwidth_mbps'])
        wan_efficient_bandwidth = wan_bandwidth * wan_char['bandwidth_efficiency']
        wan_reduction = lan_limited_bandwidth - wan_efficient_bandwidth
        steps.append({
            'name': f'WAN Provider ({wan_char["name"]})',
            'value': -wan_reduction,
            'cumulative': wan_efficient_bandwidth,
            'type': 'negative',
            'layer': 'wan'
        })
        
        # Step 7: Direct Connect (if applicable)
        dx_adjusted_bandwidth = wan_efficient_bandwidth
        dx_reduction = 0
        if dx_char:
            dx_overhead = wan_efficient_bandwidth * (dx_char['aws_edge_processing_overhead'] + dx_char['virtual_interface_overhead'])
            dx_adjusted_bandwidth = wan_efficient_bandwidth - dx_overhead
            dx_reduction = dx_overhead
            steps.append({
                'name': f'Direct Connect ({dx_char["name"]})',
                'value': -dx_reduction,
                'cumulative': dx_adjusted_bandwidth,
                'type': 'negative',
                'layer': 'dx'
            })
        
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
                'name': f'VPC Endpoint Overhead',
                'value': -vpc_reduction,
                'cumulative': vpc_adjusted_bandwidth,
                'type': 'negative',
                'layer': 'vpc'
            })
        
        # Step 9: Protocol Overhead
        protocol_efficiency = service.get('protocol_efficiency', 0.95)
        protocol_adjusted_bandwidth = vpc_adjusted_bandwidth * protocol_efficiency
        protocol_reduction = vpc_adjusted_bandwidth - protocol_adjusted_bandwidth
        steps.append({
            'name': f'Protocol Overhead ({", ".join(service["protocols"])})',
            'value': -protocol_reduction,
            'cumulative': protocol_adjusted_bandwidth,
            'type': 'negative',
            'layer': 'protocol'
        })
        
        # Step 10: Service Capacity
        service_capacity = service_spec['throughput_mbps'] * num_instances
        service_limited_bandwidth = min(protocol_adjusted_bandwidth, service_capacity)
        service_reduction = protocol_adjusted_bandwidth - service_limited_bandwidth
        steps.append({
            'name': f'{service["name"]} Capacity',
            'value': -service_reduction,
            'cumulative': service_limited_bandwidth,
            'type': 'negative',
            'layer': 'service'
        })
        
        # Step 11: Application Efficiency
        app_efficiency = service.get('application_efficiency', 0.9)
        final_bandwidth = service_limited_bandwidth * app_efficiency
        app_reduction = service_limited_bandwidth - final_bandwidth
        steps.append({
            'name': f'Application Efficiency ({app_efficiency*100:.1f}%)',
            'value': -app_reduction,
            'cumulative': final_bandwidth,
            'type': 'negative',
            'layer': 'service'
        })
        
        # Final effective bandwidth
        steps.append({
            'name': 'Final Effective',
            'value': final_bandwidth,
            'cumulative': final_bandwidth,
            'type': 'total',
            'layer': 'final'
        })
        
        # Calculate summary
        total_reduction = theoretical_max - final_bandwidth
        efficiency_percentage = (final_bandwidth / theoretical_max) * 100
        
        # Identify primary bottleneck
        reductions = [(step['name'], abs(step['value']), step['layer']) 
                     for step in steps if step['type'] == 'negative' and step['value'] < 0]
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
        """Assess service compatibility with network pattern (ORIGINAL METHOD PRESERVED)"""
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
        """Estimate migration timing with service-specific considerations (ORIGINAL METHOD PRESERVED)"""
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
        """Generate AI-powered recommendations (ORIGINAL METHOD PRESERVED)"""
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
        db_scenario = config.get('database_scenario', 'mysql_oltp')
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
            'database_scenario': config.get('database_scenario', 'mysql_oltp'),
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

def render_header():
    """Render enhanced application header"""
    st.markdown("""
    <div class="main-header">
        <h1>🚀 AI-Enhanced AWS Migration Network Analyzer</h1>
        <p style="font-size: 1.3rem; margin-top: 0.5rem;">
            Complete Infrastructure Analysis • Real-Time AWS Pricing • Claude AI Recommendations
        </p>
        <p style="font-size: 1rem; margin-top: 0.5rem; opacity: 0.9;">
            Realistic Bandwidth Waterfall • Database Engineer Focused • Pattern Comparison
        </p>
    </div>
    """, unsafe_allow_html=True)

def render_enhanced_sidebar_controls():
    """Render enhanced sidebar controls"""
    st.sidebar.header("🔧 Migration Configuration")
    
    # API Configuration
    with st.sidebar.expander("🔑 API Configuration", expanded=False):
        aws_access_key = st.text_input("AWS Access Key (Optional)", type="password", help="For real-time pricing")
        aws_secret_key = st.text_input("AWS Secret Key (Optional)", type="password", help="For real-time pricing")
        claude_api_key = st.text_input("Claude AI API Key (Optional)", type="password", help="For AI recommendations")
    
    # Database Scenario Selection
    st.sidebar.subheader("🗄️ Database Scenario")
    database_scenario = st.sidebar.selectbox(
        "Database Type & Workload",
        ["mysql_oltp", "postgresql_analytics", "oracle_enterprise", "mongodb_cluster"],
        format_func=lambda x: {
            'mysql_oltp': '🔄 MySQL OLTP Database',
            'postgresql_analytics': '📊 PostgreSQL Analytics',
            'oracle_enterprise': '🏢 Oracle Enterprise DB',
            'mongodb_cluster': '🍃 MongoDB Cluster'
        }[x],
        help="Select your database type and primary workload"
    )
    
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
    analyzer = EnhancedNetworkAnalyzer()
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
    migration_service = st.sidebar.selectbox(
        "AWS Migration Service",
        ["datasync", "dms", "fsx_windows", "fsx_lustre", "storage_gateway"],
        format_func=lambda x: {
            'datasync': '📁 AWS DataSync',
            'dms': '🗄️ AWS DMS',
            'fsx_windows': '🪟 FSx for Windows',
            'fsx_lustre': '⚡ FSx for Lustre',
            'storage_gateway': '🔗 Storage Gateway'
        }[x],
        help="Select AWS migration service"
    )
    
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
    
    return {
        'aws_access_key': aws_access_key,
        'aws_secret_key': aws_secret_key,
        'claude_api_key': claude_api_key,
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

def create_realistic_waterfall_chart(waterfall_data: Dict):
    """Create waterfall chart with layer-specific coloring (ORIGINAL FUNCTION PRESERVED)"""
    steps = waterfall_data['steps']
    
    layer_colors = {
        'nic': '#3b82f6',      # Blue
        'os': '#8b5cf6',       # Purple  
        'lan': '#10b981',      # Green
        'wan': '#f59e0b',      # Orange
        'dx': '#ef4444',       # Red
        'vpc': '#f97316',      # Orange-red
        'protocol': '#06b6d4', # Cyan
        'service': '#84cc16',  # Lime
        'final': '#1e40af'     # Dark blue
    }
    
    fig = go.Figure()
    
    for step in steps:
        color = layer_colors.get(step.get('layer', 'default'), '#6b7280')
        
        if step['type'] == 'positive':
            fig.add_trace(go.Bar(
                x=[step['name']],
                y=[step['value']],
                marker_color=color,
                name=step['name'],
                text=[f"{step['value']:.0f} Mbps"],
                textposition='outside',
                hovertemplate=f"<b>{step['name']}</b><br>Bandwidth: {step['value']:.0f} Mbps<extra></extra>"
            ))
        elif step['type'] == 'total':
            fig.add_trace(go.Bar(
                x=[step['name']],
                y=[step['value']],
                marker_color=color,
                name=step['name'],
                text=[f"{step['value']:.0f} Mbps"],
                textposition='outside',
                hovertemplate=f"<b>{step['name']}</b><br>Final: {step['value']:.0f} Mbps<extra></extra>"
            ))
        else:
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
        title="Realistic Infrastructure Impact Analysis",
        xaxis_title="Infrastructure Components",
        yaxis_title="Bandwidth (Mbps)",
        showlegend=False,
        height=600,
        xaxis=dict(tickangle=45),
        template="plotly_white"
    )
    
    return fig

def render_realistic_analysis_tab(config: Dict, analyzer: EnhancedNetworkAnalyzer):
    """Render realistic bandwidth analysis tab (ORIGINAL FUNCTION PRESERVED)"""
    st.subheader("💧 Realistic Infrastructure Impact Analysis")
    
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
    
    # Summary metrics
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric(
            "🏁 Theoretical Max",
            f"{summary['theoretical_max_mbps']:,.0f} Mbps",
            delta="NIC capacity"
        )
    
    with col2:
        st.metric(
            "🎯 Final Effective",
            f"{summary['final_effective_mbps']:,.0f} Mbps", 
            delta=f"{summary['efficiency_percentage']:.1f}% efficient"
        )
    
    with col3:
        st.metric(
            "📉 Total Reduction",
            f"{summary['total_reduction_mbps']:,.0f} Mbps",
            delta="Infrastructure overhead"
        )
    
    with col4:
        st.metric(
            "🔍 Primary Bottleneck",
            summary['primary_bottleneck_layer'].title(),
            delta=f"{summary['primary_bottleneck_impact_mbps']:,.0f} Mbps impact"
        )
    
    with col5:
        st.metric(
            "🌐 Network Utilization",
            f"{summary['network_utilization_percent']:.1f}%",
            delta="Bandwidth usage"
        )
    
    with col6:
        st.metric(
            "⚙️ Service Utilization", 
            f"{summary['service_utilization_percent']:.1f}%",
            delta=summary['bottleneck'].title() + " bound"
        )
    
    # Waterfall chart
    st.markdown("**📊 Infrastructure Impact Waterfall:**")
    waterfall_chart = create_realistic_waterfall_chart(waterfall_data)
    st.plotly_chart(waterfall_chart, use_container_width=True)
    
    # Detailed breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🏗️ Infrastructure Component Details:**")
        
        infrastructure = waterfall_data['infrastructure_details']
        
        st.markdown(f"""
        <div class="infrastructure-card">
            <h4>Server Infrastructure</h4>
            <p><strong>OS:</strong> {infrastructure['os']['name']}</p>
            <p><strong>TCP Efficiency:</strong> {infrastructure['os']['tcp_stack_efficiency']*100:.1f}%</p>
            <p><strong>NIC:</strong> {infrastructure['nic']['name']}</p>
            <p><strong>NIC Efficiency:</strong> {infrastructure['nic']['real_world_efficiency']*100:.1f}%</p>
            <p><strong>LAN:</strong> {infrastructure['lan']['name']}</p>
            <p><strong>Oversubscription:</strong> {infrastructure['lan']['oversubscription_ratio']}</p>
            <p><strong>WAN:</strong> {infrastructure['wan']['name']}</p>
            <p><strong>WAN Efficiency:</strong> {infrastructure['wan']['bandwidth_efficiency']*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("**🎯 Optimization Recommendations:**")
        
        primary_bottleneck = summary['primary_bottleneck_layer']
        
        if primary_bottleneck == 'nic':
            recommendation = "Upgrade to higher bandwidth NIC (25/100 Gbps)"
        elif primary_bottleneck == 'os':
            recommendation = "Optimize OS network stack, enable kernel bypass"
        elif primary_bottleneck == 'lan':
            recommendation = "Reduce oversubscription, increase switch capacity"
        elif primary_bottleneck == 'wan':
            recommendation = "Upgrade WAN bandwidth or provider tier"
        elif primary_bottleneck == 'service':
            recommendation = "Scale service instances or upgrade instance size"
        else:
            recommendation = "Review overall infrastructure architecture"
        
        st.markdown(f"""
        <div class="recommendation-card">
            <h4>Primary Optimization Target</h4>
            <p><strong>Bottleneck:</strong> {summary['primary_bottleneck']}</p>
            <p><strong>Impact:</strong> {summary['primary_bottleneck_impact_mbps']:,.0f} Mbps reduction</p>
            <p><strong>Recommendation:</strong> {recommendation}</p>
            <p><strong>Expected Improvement:</strong> Up to {summary['primary_bottleneck_impact_mbps']:,.0f} Mbps</p>
        </div>
        """, unsafe_allow_html=True)
    
    return waterfall_data

def render_migration_analysis_tab(config: Dict, waterfall_data: Dict, analyzer: EnhancedNetworkAnalyzer):
    """Render migration timing and compatibility analysis (ORIGINAL FUNCTION PRESERVED)"""
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
    
    # Compatibility and timing metrics
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric(
            "🚀 Migration Service",
            config['migration_service'].upper(),
            delta=service_compatibility['service_name']
        )
    
    with col2:
        vpc_compatible = "✅ Yes" if service_compatibility['vpc_endpoint_compatible'] else "❌ No"
        st.metric(
            "🔗 VPC Endpoint Support",
            vpc_compatible,
            delta="Compatibility check"
        )
    
    with col3:
        st.metric(
            "📊 Data Size",
            f"{config['data_size_gb']:,} GB",
            delta=f"{config['data_size_gb'] * 8:,} Gbits"
        )
    
    with col4:
        st.metric(
            "⚡ Effective Speed",
            f"{waterfall_data['summary']['final_effective_mbps']:,.0f} Mbps",
            delta=f"{waterfall_data['summary']['final_effective_mbps']/8:.0f} MB/s"
        )
    
    with col5:
        st.metric(
            "🕒 Transfer Time",
            f"{migration_time['data_transfer_hours']:.1f} hours",
            delta=f"{migration_time['data_transfer_hours']/24:.1f} days"
        )
    
    with col6:
        meets_requirement = migration_time['total_hours'] <= config['max_downtime_hours']
        delta_text = "✅ Meets SLA" if meets_requirement else "❌ Exceeds SLA"
        st.metric(
            "⏰ Total vs SLA",
            f"{migration_time['total_hours']:.1f}h / {config['max_downtime_hours']}h",
            delta=delta_text
        )
    
    # Detailed analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📋 Service Compatibility Analysis:**")
        
        if service_compatibility['warnings']:
            st.markdown("**🚨 Warnings:**")
            for warning in service_compatibility['warnings']:
                st.warning(f"• {warning}")
        
        if service_compatibility['requirements']:
            st.markdown("**📋 Requirements:**")
            for requirement in service_compatibility['requirements']:
                st.info(f"• {requirement}")
        
        if service_compatibility['recommendations']:
            st.markdown("**💡 Recommendations:**")
            for recommendation in service_compatibility['recommendations']:
                st.success(f"• {recommendation}")
        
        if not any([service_compatibility['warnings'], service_compatibility['requirements'], service_compatibility['recommendations']]):
            st.success("✅ No compatibility issues detected")
    
    with col2:
        st.markdown("**⏱️ Migration Timeline Breakdown:**")
        
        timeline_data = [
            {"Phase": "Setup & Config", "Hours": migration_time['setup_hours']},
            {"Phase": "Data Transfer", "Hours": migration_time['data_transfer_hours']},
            {"Phase": "Validation", "Hours": migration_time['validation_hours']}
        ]
        
        df_timeline = pd.DataFrame(timeline_data)
        st.dataframe(df_timeline, use_container_width=True)
        
        if migration_time.get('supports_incremental'):
            st.success(f"✅ {migration_time['service_name']} supports incremental migration")
        
        st.markdown(f"""
        <div class="service-card">
            <h4>Migration Summary</h4>
            <p><strong>Service:</strong> {migration_time['service_name']}</p>
            <p><strong>Total Time:</strong> {migration_time['total_hours']:.1f} hours</p>
            <p><strong>Recommended Window:</strong> {migration_time['recommended_window_hours']:.1f} hours</p>
            <p><strong>SLA Compliance:</strong> {'✅ Pass' if meets_requirement else '❌ Fail'}</p>
        </div>
        """, unsafe_allow_html=True)
    
    return {
        'migration_time': migration_time,
        'service_compatibility': service_compatibility,
        'waterfall_data': waterfall_data
    }

def render_ai_recommendations_tab(config: Dict, analysis_results: Dict, analyzer: EnhancedNetworkAnalyzer):
    """Render AI-powered recommendations (ORIGINAL FUNCTION PRESERVED)"""
    st.subheader("🤖 AI-Powered Migration Recommendations")
    
    ai_recommendations = analyzer.generate_ai_recommendations(config, analysis_results)
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "🎯 Migration Complexity",
            ai_recommendations['migration_complexity'].title(),
            delta=f"Score: {ai_recommendations['overall_priority_score']}"
        )
    
    with col2:
        st.metric(
            "🤖 AI Confidence",
            ai_recommendations['confidence_level'].title(),
            delta=f"{len(ai_recommendations['recommendations'])} recommendations"
        )
    
    with col3:
        waterfall_summary = analysis_results['waterfall_data']['summary']
        st.metric(
            "🔍 Primary Bottleneck",
            waterfall_summary['primary_bottleneck_layer'].title(),
            delta=f"{waterfall_summary['primary_bottleneck_impact_mbps']:,.0f} Mbps"
        )
    
    with col4:
        migration_time = analysis_results['migration_time']
        st.metric(
            "⏰ Migration Window",
            f"{migration_time['recommended_window_hours']:.1f}h",
            delta=f"{migration_time['total_days']:.1f} days"
        )
    
    # Recommendations by priority
    st.markdown("**💡 Prioritized Recommendations:**")
    
    # Group by priority
    critical_recs = [rec for rec in ai_recommendations['recommendations'] if rec['priority'] == 'critical']
    high_recs = [rec for rec in ai_recommendations['recommendations'] if rec['priority'] == 'high']
    medium_recs = [rec for rec in ai_recommendations['recommendations'] if rec['priority'] == 'medium']
    low_recs = [rec for rec in ai_recommendations['recommendations'] if rec['priority'] == 'low']
    
    if critical_recs:
        st.markdown("### 🚨 Critical Issues")
        for i, rec in enumerate(critical_recs, 1):
            with st.expander(f"🔴 {rec['type'].replace('_', ' ').title()}", expanded=True):
                st.error(f"**Description:** {rec['description']}")
                st.write(f"**Expected Impact:** {rec['impact']}")
    
    if high_recs:
        st.markdown("### ⚠️ High Priority")
        for i, rec in enumerate(high_recs, 1):
            with st.expander(f"🟠 {rec['type'].replace('_', ' ').title()}", expanded=True):
                st.warning(f"**Description:** {rec['description']}")
                st.write(f"**Expected Impact:** {rec['impact']}")
    
    if medium_recs:
        st.markdown("### 📋 Medium Priority")
        for i, rec in enumerate(medium_recs, 1):
            with st.expander(f"🟡 {rec['type'].replace('_', ' ').title()}", expanded=False):
                st.info(f"**Description:** {rec['description']}")
                st.write(f"**Expected Impact:** {rec['impact']}")
    
    if low_recs:
        st.markdown("### ✅ Low Priority")
        for i, rec in enumerate(low_recs, 1):
            with st.expander(f"🟢 {rec['type'].replace('_', ' ').title()}", expanded=False):
                st.success(f"**Description:** {rec['description']}")
                st.write(f"**Expected Impact:** {rec['impact']}")
    
    # Summary
    waterfall_summary = analysis_results['waterfall_data']['summary']
    migration_time = analysis_results['migration_time']
    
    st.markdown(f"""
    <div class="recommendation-card">
        <h4>🎯 Migration Strategy Summary</h4>
        <p><strong>Service:</strong> {waterfall_summary['service_name']}</p>
        <p><strong>Complexity:</strong> {ai_recommendations['migration_complexity'].title()}</p>
        <p><strong>Confidence:</strong> {ai_recommendations['confidence_level'].title()}</p>
        <p><strong>Timeline:</strong> {migration_time['total_hours']:.1f} hours ({migration_time['total_days']:.1f} days)</p>
        <p><strong>Primary Bottleneck:</strong> {waterfall_summary['primary_bottleneck_layer'].title()}</p>
        <p><strong>Infrastructure Efficiency:</strong> {waterfall_summary['efficiency_percentage']:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)

def render_pattern_comparison_tab(analyzer: EnhancedNetworkAnalyzer, config: Dict):
    """Render pattern comparison analysis (NEW)"""
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
    
    # Comparison metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "💰 Best Cost",
            f"${min(p.total_cost_usd for p in pattern_analyses):,.0f}",
            delta=f"vs ${max(p.total_cost_usd for p in pattern_analyses):,.0f}"
        )
    
    with col2:
        st.metric(
            "⚡ Best Speed",
            f"{max(p.effective_bandwidth_mbps for p in pattern_analyses):,.0f} Mbps",
            delta="Highest throughput"
        )
    
    with col3:
        st.metric(
            "🕒 Fastest Migration",
            f"{min(p.migration_time_hours for p in pattern_analyses):.1f}h",
            delta="Minimum time"
        )
    
    with col4:
        st.metric(
            "🛡️ Best Reliability",
            f"{max(p.reliability_score for p in pattern_analyses)*100:.1f}%",
            delta="Highest uptime"
        )
    
    with col5:
        st.metric(
            "🎯 AI Score",
            f"{best_pattern.ai_recommendation_score:.2f}",
            delta="Recommended"
        )
    
    # Detailed comparison table
    st.markdown("**📊 Detailed Pattern Comparison:**")
    
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
    st.dataframe(df_comparison, use_container_width=True)
    
    # Cost vs Performance Chart
    fig_scatter = go.Figure()
    
    for i, pattern in enumerate(pattern_analyses):
        color = '#16a34a' if i == 0 else '#3b82f6'  # Green for best, blue for others
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
        title="Migration Cost vs Time Analysis",
        xaxis_title="Total Cost ($)",
        yaxis_title="Migration Time (hours)",
        showlegend=False,
        template="plotly_white",
        height=500
    )
    
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    return pattern_analyses, ai_recommendation

def render_ai_insights_tab(pattern_analyses: List[PatternAnalysis], ai_recommendation: Dict, config: Dict):
    """Render AI insights and recommendations (NEW)"""
    st.subheader("🤖 AI-Powered Migration Insights")
    
    if not pattern_analyses or not ai_recommendation:
        st.warning("Please run pattern comparison first to get AI insights.")
        return
    
    # AI Recommendation Summary
    st.markdown(f"""
    <div class="ai-recommendation-card">
        <h3>🎯 AI Recommendation: {ai_recommendation['recommended_pattern'].replace('_', ' ').title()}</h3>
        <p><strong>Confidence Level:</strong> {ai_recommendation['confidence_score']*100:.0f}%</p>
        <p><strong>Reasoning:</strong> {ai_recommendation['reasoning']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Database-Specific Insights
    database_scenario = config.get('database_scenario', 'mysql_oltp')
    db_name = {
        'mysql_oltp': 'MySQL OLTP',
        'postgresql_analytics': 'PostgreSQL Analytics',
        'oracle_enterprise': 'Oracle Enterprise',
        'mongodb_cluster': 'MongoDB Cluster'
    }[database_scenario]
    
    st.markdown(f"""
    <div class="database-scenario-card">
        <h4>🗄️ Database-Specific Considerations for {db_name}</h4>
        <p><strong>Database Insights:</strong> {ai_recommendation['database_considerations']}</p>
        <p><strong>Risk Assessment:</strong> {ai_recommendation['risk_assessment']}</p>
        <p><strong>Cost Justification:</strong> {ai_recommendation['cost_justification']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Pattern Deep Dive
    best_pattern = pattern_analyses[0]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**✅ Recommended Pattern Advantages:**")
        for pro in best_pattern.pros:
            st.success(f"• {pro}")
        
        st.markdown("**🎯 Best Use Cases:**")
        for use_case in best_pattern.use_cases:
            st.info(f"• {use_case}")
    
    with col2:
        st.markdown("**⚠️ Considerations & Limitations:**")
        for con in best_pattern.cons:
            st.warning(f"• {con}")
        
        st.markdown("**💡 Optimization Recommendations:**")
        if best_pattern.complexity_score > 0.7:
            st.warning("• High complexity - consider professional services engagement")
        if best_pattern.total_cost_usd > 5000:
            st.info("• High cost - evaluate phased migration approach")
        if best_pattern.migration_time_hours > config['max_downtime_hours']:
            st.error("• Exceeds downtime SLA - consider incremental migration")
        else:
            st.success("• Meets downtime requirements")

def render_database_guidance_tab(config: Dict):
    """Render database engineer guidance (NEW)"""
    st.subheader("📚 Database Engineer's Migration Guide")
    
    database_scenario = config.get('database_scenario', 'mysql_oltp')
    
    # Database-specific guidance
    guidance_content = {
        'mysql_oltp': {
            'title': '🔄 MySQL OLTP Database Migration',
            'key_considerations': [
                "Binary log settings for replication consistency",
                "InnoDB buffer pool warming strategies",
                "Connection pool configuration",
                "Read replica lag monitoring"
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
                "Failover/failback procedures"
            ]
        },
        'postgresql_analytics': {
            'title': '📊 PostgreSQL Analytics Migration',
            'key_considerations': [
                "Vacuum and analyze statistics",
                "Extension compatibility (PostGIS, etc.)",
                "Large table partitioning strategy",
                "Query performance optimization"
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
                "Performance baseline comparison"
            ]
        },
        'oracle_enterprise': {
            'title': '🏢 Oracle Enterprise Database Migration',
            'key_considerations': [
                "Oracle-specific features compatibility",
                "PL/SQL code conversion needs",
                "Tablespace and datafile strategy",
                "RAC to RDS conversion complexity"
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
                "Disaster recovery validation"
            ]
        },
        'mongodb_cluster': {
            'title': '🍃 MongoDB Cluster Migration',
            'key_considerations': [
                "Sharding strategy preservation",
                "Index optimization for AWS",
                "Connection string updates",
                "Replica set configuration"
            ],
            'network_requirements': {
                'min_bandwidth': '1500 Mbps',
                'max_latency': '20ms',
                'consistency': 'Configurable (adjust read/write concerns)'
            },
            'recommended_approach': 'Native MongoDB tools + DMS for validation',
            'testing_strategy': [
                "Shard balancing verification",
                "Application driver compatibility",
                "Performance under load",
                "Backup and restore procedures"
            ]
        }
    }
    
    selected_guidance = guidance_content[database_scenario]
    
    # Display guidance
    st.markdown(f"""
    <div class="database-scenario-card">
        <h3>{selected_guidance['title']}</h3>
        <p><strong>Recommended Approach:</strong> {selected_guidance['recommended_approach']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔧 Key Technical Considerations:**")
        for consideration in selected_guidance['key_considerations']:
            st.info(f"• {consideration}")
        
        st.markdown("**📋 Testing Strategy:**")
        for test in selected_guidance['testing_strategy']:
            st.success(f"• {test}")
    
    with col2:
        st.markdown("**🌐 Network Requirements:**")
        requirements = selected_guidance['network_requirements']
        
        st.markdown(f"""
        <div class="network-metric-card">
            <h4>Minimum Bandwidth</h4>
            <p style="font-size: 1.5em; color: #3b82f6;">{requirements['min_bandwidth']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="network-metric-card">
            <h4>Maximum Latency</h4>
            <p style="font-size: 1.5em; color: #f59e0b;">{requirements['max_latency']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="network-metric-card">
            <h4>Consistency Model</h4>
            <p style="font-size: 1.2em; color: #16a34a;">{requirements['consistency']}</p>
        </div>
        """, unsafe_allow_html=True)

# Main application
def main():
    render_header()
    
    # Sidebar configuration
    config = render_enhanced_sidebar_controls()
    
    # Initialize analyzer
    analyzer = EnhancedNetworkAnalyzer()
    
    # Main tabs - NOW WITH 5 TABS PRESERVING ALL ORIGINAL FUNCTIONALITY + NEW FEATURES
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💧 Realistic Analysis",  # ORIGINAL
        "⏱️ Migration Analysis",   # ORIGINAL 
        "🤖 Original AI Recommendations",  # ORIGINAL
        "🔄 Pattern Comparison",    # NEW
        "📚 Database Guide"        # NEW
    ])
    
    with tab1:
        waterfall_data = render_realistic_analysis_tab(config, analyzer)
    
    with tab2:
        if 'waterfall_data' in locals():
            analysis_results = render_migration_analysis_tab(config, waterfall_data, analyzer)
        else:
            st.info("Please run Realistic Analysis first.")
    
    with tab3:
        if 'analysis_results' in locals():
            render_ai_recommendations_tab(config, analysis_results, analyzer)
        else:
            st.info("Please run Migration Analysis first.")
    
    with tab4:
        pattern_analyses, ai_recommendation = render_pattern_comparison_tab(analyzer, config)
    
    with tab5:
        render_database_guidance_tab(config)

if __name__ == "__main__":
    main()