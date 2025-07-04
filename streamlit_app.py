import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import math
import random
from typing import Dict, List, Tuple, Optional
import json
import requests
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import anthropic
import asyncio
import logging
from dataclasses import dataclass
import yaml
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import concurrent.futures
import threading


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AWS Enterprise Database Migration Analyzer AI v3.0",
    page_icon="🤖",
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
    
    .main-header h1 {
        margin: 0 0 0.5rem 0;
        font-size: 2.2rem;
        font-weight: 600;
    }
    
    .professional-card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 6px;
        color: #374151;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border-left: 3px solid #3b82f6;
        border: 1px solid #e5e7eb;
    }
    
    .insight-card {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        padding: 1.5rem;
        border-radius: 6px;
        color: #374151;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border-left: 3px solid #3b82f6;
        border: 1px solid #e5e7eb;
    }
    
    .metric-card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 6px;
        border-left: 3px solid #3b82f6;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border: 1px solid #e5e7eb;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .agent-scaling-card {
        background: #ffffff;
        padding: 1rem;
        border-radius: 6px;
        margin: 0.5rem 0;
        color: #374151;
        font-size: 0.9rem;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .api-status-card {
        background: #ffffff;
        padding: 1rem;
        border-radius: 6px;
        margin: 0.5rem 0;
        color: #374151;
        font-size: 0.9rem;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .status-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 8px;
        vertical-align: middle;
    }
    
    .status-online { background-color: #22c55e; }
    .status-offline { background-color: #ef4444; }
    .status-warning { background-color: #f59e0b; }
    
    .enterprise-footer {
        background: linear-gradient(135deg, #1f2937 0%, #374151 100%);
        color: white;
        padding: 2rem;
        border-radius: 6px;
        margin-top: 2rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .cost-table {
        font-size: 0.9rem;
        border-collapse: collapse;
        width: 100%;
    }
    
    .cost-table th {
        background-color: #f8fafc;
        font-weight: 600;
        padding: 0.75rem;
        border: 1px solid #e5e7eb;
    }
    
    .cost-table td {
        padding: 0.75rem;
        border: 1px solid #e5e7eb;
    }
    
    .service-detail-card {
        background: #ffffff;
        padding: 1rem;
        border-radius: 6px;
        margin: 0.5rem 0;
        border-left: 3px solid #3b82f6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

@dataclass
class APIStatus:
    anthropic_connected: bool = False
    aws_pricing_connected: bool = False
    aws_compute_optimizer_connected: bool = False
    last_update: datetime = None
    error_message: str = None

class EnhancedAWSAPIManager:
    """Enhanced AWS API manager with comprehensive service pricing"""
    
    def __init__(self):
        self.session = None
        self.pricing_client = None
        self.ec2_client = None
        self.rds_client = None
        self.connected = False
        
        try:
            self.session = boto3.Session()
            self.pricing_client = self.session.client('pricing', region_name='us-east-1')
            self.ec2_client = self.session.client('ec2', region_name='us-west-2')
            self.rds_client = self.session.client('rds', region_name='us-west-2')
            
            # Test connection
            self.pricing_client.describe_services(MaxResults=1)
            self.connected = True
            logger.info("Enhanced AWS API clients initialized successfully")
        except (NoCredentialsError, ClientError) as e:
            logger.warning(f"AWS API initialization failed: {e}")
            self.connected = False
    
    async def get_comprehensive_pricing(self, region: str = 'us-west-2') -> Dict:
        """Fetch comprehensive real-time AWS pricing for all migration services"""
        if not self.connected:
            return self._fallback_comprehensive_pricing(region)
        
        try:
            pricing_data = {
                'region': region,
                'last_updated': datetime.now(),
                'data_source': 'aws_api',
                'services': {}
            }
            
            # Get pricing for all services in parallel
            tasks = [
                self._get_ec2_pricing(region),
                self._get_rds_pricing(region),
                self._get_ebs_pricing(region),
                self._get_s3_pricing(region),
                self._get_datasync_pricing(region),
                self._get_dms_pricing(region),
                self._get_direct_connect_pricing(region),
                self._get_fsx_pricing(region),
                self._get_cloudwatch_pricing(region),
                self._get_vpc_pricing(region)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            service_names = [
                'ec2', 'rds', 'ebs', 's3', 'datasync', 
                'dms', 'direct_connect', 'fsx', 'cloudwatch', 'vpc'
            ]
            
            for i, result in enumerate(results):
                if not isinstance(result, Exception):
                    pricing_data['services'][service_names[i]] = result
                else:
                    logger.warning(f"Failed to get {service_names[i]} pricing: {result}")
                    pricing_data['services'][service_names[i]] = self._get_fallback_service_pricing(service_names[i])
            
            return pricing_data
            
        except Exception as e:
            logger.error(f"Failed to fetch comprehensive AWS pricing: {e}")
            return self._fallback_comprehensive_pricing(region)
    
    async def _get_ec2_pricing(self, region: str) -> Dict:
        """Get comprehensive EC2 pricing including all instance types"""
        instance_families = {
            't3': ['t3.micro', 't3.small', 't3.medium', 't3.large', 't3.xlarge', 't3.2xlarge'],
            'c5': ['c5.large', 'c5.xlarge', 'c5.2xlarge', 'c5.4xlarge', 'c5.9xlarge'],
            'r6i': ['r6i.large', 'r6i.xlarge', 'r6i.2xlarge', 'r6i.4xlarge', 'r6i.8xlarge'],
            'm5': ['m5.large', 'm5.xlarge', 'm5.2xlarge', 'm5.4xlarge'],
            'r5': ['r5.large', 'r5.xlarge', 'r5.2xlarge', 'r5.4xlarge']
        }
        
        pricing_data = {}
        
        for family, instances in instance_families.items():
            pricing_data[family] = {}
            for instance_type in instances:
                try:
                    response = await asyncio.to_thread(
                        self.pricing_client.get_products,
                        ServiceCode='AmazonEC2',
                        Filters=[
                            {'Type': 'TERM_MATCH', 'Field': 'instanceType', 'Value': instance_type},
                            {'Type': 'TERM_MATCH', 'Field': 'location', 'Value': self._region_to_location(region)},
                            {'Type': 'TERM_MATCH', 'Field': 'tenancy', 'Value': 'Shared'},
                            {'Type': 'TERM_MATCH', 'Field': 'operatingSystem', 'Value': 'Linux'}
                        ],
                        MaxResults=1
                    )
                    
                    if response['PriceList']:
                        price_data = json.loads(response['PriceList'][0])
                        terms = price_data.get('terms', {}).get('OnDemand', {})
                        if terms:
                            term_data = list(terms.values())[0]
                            price_dimensions = term_data.get('priceDimensions', {})
                            if price_dimensions:
                                price_info = list(price_dimensions.values())[0]
                                attributes = price_data.get('product', {}).get('attributes', {})
                                
                                pricing_data[family][instance_type] = {
                                    'vcpu': int(attributes.get('vcpu', 2)),
                                    'memory': self._extract_memory_gb(attributes.get('memory', '4 GiB')),
                                    'network_performance': attributes.get('networkPerformance', 'Low'),
                                    'storage': attributes.get('storage', 'EBS only'),
                                    'cost_per_hour': float(price_info['pricePerUnit']['USD']),
                                    'cost_per_month': float(price_info['pricePerUnit']['USD']) * 24 * 30,
                                    'pricing_unit': price_info.get('unit', 'Hour')
                                }
                                
                except Exception as e:
                    logger.warning(f"Failed to get pricing for {instance_type}: {e}")
                    pricing_data[family][instance_type] = self._get_fallback_instance_pricing(instance_type)
        
        return pricing_data
    
    async def _get_rds_pricing(self, region: str) -> Dict:
        """Get comprehensive RDS pricing for all database engines"""
        db_engines = ['mysql', 'postgres', 'oracle-ee', 'sqlserver-ex', 'sqlserver-se']
        instance_classes = ['db.t3.medium', 'db.t3.large', 'db.r6g.large', 'db.r6g.xlarge', 'db.r6g.2xlarge']
        
        pricing_data = {}
        
        for engine in db_engines:
            pricing_data[engine] = {}
            for instance_class in instance_classes:
                try:
                    response = await asyncio.to_thread(
                        self.pricing_client.get_products,
                        ServiceCode='AmazonRDS',
                        Filters=[
                            {'Type': 'TERM_MATCH', 'Field': 'instanceType', 'Value': instance_class},
                            {'Type': 'TERM_MATCH', 'Field': 'location', 'Value': self._region_to_location(region)},
                            {'Type': 'TERM_MATCH', 'Field': 'databaseEngine', 'Value': engine}
                        ],
                        MaxResults=1
                    )
                    
                    if response['PriceList']:
                        price_data = json.loads(response['PriceList'][0])
                        terms = price_data.get('terms', {}).get('OnDemand', {})
                        if terms:
                            term_data = list(terms.values())[0]
                            price_dimensions = term_data.get('priceDimensions', {})
                            if price_dimensions:
                                price_info = list(price_dimensions.values())[0]
                                attributes = price_data.get('product', {}).get('attributes', {})
                                
                                pricing_data[engine][instance_class] = {
                                    'vcpu': int(attributes.get('vcpu', 2)),
                                    'memory': self._extract_memory_gb(attributes.get('memory', '4 GiB')),
                                    'network_performance': attributes.get('networkPerformance', 'Low'),
                                    'cost_per_hour': float(price_info['pricePerUnit']['USD']),
                                    'cost_per_month': float(price_info['pricePerUnit']['USD']) * 24 * 30,
                                    'deployment_option': attributes.get('deploymentOption', 'Single-AZ'),
                                    'license_model': attributes.get('licenseModel', 'No license required')
                                }
                                
                except Exception as e:
                    logger.warning(f"Failed to get RDS pricing for {engine} {instance_class}: {e}")
                    pricing_data[engine][instance_class] = self._get_fallback_rds_pricing(instance_class)
        
        return pricing_data
    
    async def _get_ebs_pricing(self, region: str) -> Dict:
        """Get EBS storage pricing for all volume types"""
        volume_types = ['gp2', 'gp3', 'io1', 'io2', 'st1', 'sc1']
        
        pricing_data = {}
        
        for volume_type in volume_types:
            try:
                response = await asyncio.to_thread(
                    self.pricing_client.get_products,
                    ServiceCode='AmazonEC2',
                    Filters=[
                        {'Type': 'TERM_MATCH', 'Field': 'productFamily', 'Value': 'Storage'},
                        {'Type': 'TERM_MATCH', 'Field': 'volumeType', 'Value': volume_type},
                        {'Type': 'TERM_MATCH', 'Field': 'location', 'Value': self._region_to_location(region)}
                    ],
                    MaxResults=1
                )
                
                if response['PriceList']:
                    price_data = json.loads(response['PriceList'][0])
                    terms = price_data.get('terms', {}).get('OnDemand', {})
                    if terms:
                        term_data = list(terms.values())[0]
                        price_dimensions = term_data.get('priceDimensions', {})
                        if price_dimensions:
                            price_info = list(price_dimensions.values())[0]
                            
                            pricing_data[volume_type] = {
                                'cost_per_gb_month': float(price_info['pricePerUnit']['USD']),
                                'unit': price_info.get('unit', 'GB-Month'),
                                'description': price_info.get('description', f'{volume_type} storage')
                            }
                            
            except Exception as e:
                logger.warning(f"Failed to get EBS pricing for {volume_type}: {e}")
                pricing_data[volume_type] = self._get_fallback_ebs_pricing(volume_type)
        
        return pricing_data
    
    async def _get_s3_pricing(self, region: str) -> Dict:
        """Get S3 pricing for different storage classes"""
        storage_classes = ['Standard', 'Standard-IA', 'Glacier', 'Glacier Deep Archive']
        
        pricing_data = {}
        
        for storage_class in storage_classes:
            try:
                response = await asyncio.to_thread(
                    self.pricing_client.get_products,
                    ServiceCode='AmazonS3',
                    Filters=[
                        {'Type': 'TERM_MATCH', 'Field': 'storageClass', 'Value': storage_class},
                        {'Type': 'TERM_MATCH', 'Field': 'location', 'Value': self._region_to_location(region)}
                    ],
                    MaxResults=1
                )
                
                if response['PriceList']:
                    price_data = json.loads(response['PriceList'][0])
                    terms = price_data.get('terms', {}).get('OnDemand', {})
                    if terms:
                        term_data = list(terms.values())[0]
                        price_dimensions = term_data.get('priceDimensions', {})
                        if price_dimensions:
                            price_info = list(price_dimensions.values())[0]
                            
                            pricing_data[storage_class] = {
                                'cost_per_gb_month': float(price_info['pricePerUnit']['USD']),
                                'unit': price_info.get('unit', 'GB-Month'),
                                'description': price_info.get('description', f'S3 {storage_class}')
                            }
                            
            except Exception as e:
                logger.warning(f"Failed to get S3 pricing for {storage_class}: {e}")
                pricing_data[storage_class] = self._get_fallback_s3_pricing(storage_class)
        
        return pricing_data
    
    async def _get_datasync_pricing(self, region: str) -> Dict:
        """Get DataSync pricing"""
        try:
            response = await asyncio.to_thread(
                self.pricing_client.get_products,
                ServiceCode='AWSDataSync',
                Filters=[
                    {'Type': 'TERM_MATCH', 'Field': 'location', 'Value': self._region_to_location(region)}
                ],
                MaxResults=5
            )
            
            pricing_data = {}
            for product_str in response.get('PriceList', []):
                product = json.loads(product_str)
                terms = product.get('terms', {}).get('OnDemand', {})
                if terms:
                    term_data = list(terms.values())[0]
                    price_dimensions = term_data.get('priceDimensions', {})
                    if price_dimensions:
                        price_info = list(price_dimensions.values())[0]
                        attributes = product.get('product', {}).get('attributes', {})
                        
                        pricing_data['data_transfer'] = {
                            'cost_per_gb': float(price_info['pricePerUnit']['USD']),
                            'unit': price_info.get('unit', 'GB'),
                            'description': price_info.get('description', 'DataSync data transfer')
                        }
                        break
            
            return pricing_data if pricing_data else self._get_fallback_datasync_pricing()
            
        except Exception as e:
            logger.warning(f"Failed to get DataSync pricing: {e}")
            return self._get_fallback_datasync_pricing()
    
    async def _get_dms_pricing(self, region: str) -> Dict:
        """Get DMS pricing for replication instances"""
        instance_classes = ['dms.t3.micro', 'dms.t3.small', 'dms.t3.medium', 'dms.c5.large', 'dms.c5.xlarge']
        
        pricing_data = {}
        
        for instance_class in instance_classes:
            try:
                response = await asyncio.to_thread(
                    self.pricing_client.get_products,
                    ServiceCode='AWSDatabaseMigrationSvc',
                    Filters=[
                        {'Type': 'TERM_MATCH', 'Field': 'instanceType', 'Value': instance_class},
                        {'Type': 'TERM_MATCH', 'Field': 'location', 'Value': self._region_to_location(region)}
                    ],
                    MaxResults=1
                )
                
                if response['PriceList']:
                    price_data = json.loads(response['PriceList'][0])
                    terms = price_data.get('terms', {}).get('OnDemand', {})
                    if terms:
                        term_data = list(terms.values())[0]
                        price_dimensions = term_data.get('priceDimensions', {})
                        if price_dimensions:
                            price_info = list(price_dimensions.values())[0]
                            
                            pricing_data[instance_class] = {
                                'cost_per_hour': float(price_info['pricePerUnit']['USD']),
                                'cost_per_month': float(price_info['pricePerUnit']['USD']) * 24 * 30,
                                'unit': price_info.get('unit', 'Hour'),
                                'description': price_info.get('description', f'DMS {instance_class}')
                            }
                            
            except Exception as e:
                logger.warning(f"Failed to get DMS pricing for {instance_class}: {e}")
                pricing_data[instance_class] = self._get_fallback_dms_pricing(instance_class)
        
        return pricing_data
    
    async def _get_direct_connect_pricing(self, region: str) -> Dict:
        """Get Direct Connect pricing"""
        return {
            '1Gbps': {
                'port_hours': 0.30,
                'data_transfer_out': 0.02,
                'description': '1 Gbps Dedicated Connection'
            },
            '10Gbps': {
                'port_hours': 2.25,
                'data_transfer_out': 0.02,
                'description': '10 Gbps Dedicated Connection'
            },
            '100Gbps': {
                'port_hours': 22.50,
                'data_transfer_out': 0.02,
                'description': '100 Gbps Dedicated Connection'
            }
        }
    
    async def _get_fsx_pricing(self, region: str) -> Dict:
        """Get FSx pricing for Windows and Lustre"""
        return {
            'fsx_windows': {
                'cost_per_gb_month': 0.13,
                'throughput_cost_per_mbps_month': 2.20,
                'description': 'FSx for Windows File Server'
            },
            'fsx_lustre': {
                'cost_per_gb_month': 0.14,
                'throughput_cost_per_mbps_month': 0.50,
                'description': 'FSx for Lustre'
            }
        }
    
    async def _get_cloudwatch_pricing(self, region: str) -> Dict:
        """Get CloudWatch pricing"""
        return {
            'metrics': {
                'cost_per_metric_per_month': 0.30,
                'description': 'Custom metrics'
            },
            'logs': {
                'cost_per_gb_ingested': 0.50,
                'cost_per_gb_storage_per_month': 0.03,
                'description': 'CloudWatch Logs'
            },
            'dashboards': {
                'cost_per_dashboard_per_month': 3.00,
                'description': 'CloudWatch dashboards'
            }
        }
    
    async def _get_vpc_pricing(self, region: str) -> Dict:
        """Get VPC and networking pricing"""
        return {
            'nat_gateway': {
                'cost_per_hour': 0.045,
                'cost_per_gb_processed': 0.045,
                'description': 'NAT Gateway'
            },
            'vpc_endpoints': {
                'cost_per_hour': 0.01,
                'cost_per_gb_processed': 0.01,
                'description': 'VPC Endpoints'
            },
            'elastic_ip': {
                'cost_per_hour_idle': 0.005,
                'description': 'Elastic IP addresses'
            }
        }
    
    def _region_to_location(self, region: str) -> str:
        """Convert AWS region to location name for pricing API"""
        region_map = {
            'us-east-1': 'US East (N. Virginia)',
            'us-west-2': 'US West (Oregon)',
            'eu-west-1': 'Europe (Ireland)',
            'ap-southeast-1': 'Asia Pacific (Singapore)'
        }
        return region_map.get(region, 'US West (Oregon)')
    
    def _extract_memory_gb(self, memory_str: str) -> int:
        """Extract memory in GB from AWS memory string"""
        try:
            import re
            match = re.search(r'([\d.]+)', memory_str)
            if match:
                return int(float(match.group(1)))
            return 4
        except:
            return 4
    
    def _fallback_comprehensive_pricing(self, region: str) -> Dict:
        """Comprehensive fallback pricing data"""
        return {
            'region': region,
            'last_updated': datetime.now(),
            'data_source': 'fallback',
            'services': {
                'ec2': self._get_fallback_service_pricing('ec2'),
                'rds': self._get_fallback_service_pricing('rds'),
                'ebs': self._get_fallback_service_pricing('ebs'),
                's3': self._get_fallback_service_pricing('s3'),
                'datasync': self._get_fallback_service_pricing('datasync'),
                'dms': self._get_fallback_service_pricing('dms'),
                'direct_connect': self._get_fallback_service_pricing('direct_connect'),
                'fsx': self._get_fallback_service_pricing('fsx'),
                'cloudwatch': self._get_fallback_service_pricing('cloudwatch'),
                'vpc': self._get_fallback_service_pricing('vpc')
            }
        }
    
    def _get_fallback_service_pricing(self, service: str) -> Dict:
        """Get fallback pricing for specific service"""
        fallback_pricing = {
            'ec2': {
                't3': {
                    't3.medium': {'vcpu': 2, 'memory': 4, 'cost_per_hour': 0.0416, 'cost_per_month': 29.95},
                    't3.large': {'vcpu': 2, 'memory': 8, 'cost_per_hour': 0.0832, 'cost_per_month': 59.90},
                    't3.xlarge': {'vcpu': 4, 'memory': 16, 'cost_per_hour': 0.1664, 'cost_per_month': 119.81}
                },
                'c5': {
                    'c5.large': {'vcpu': 2, 'memory': 4, 'cost_per_hour': 0.085, 'cost_per_month': 61.20},
                    'c5.xlarge': {'vcpu': 4, 'memory': 8, 'cost_per_hour': 0.17, 'cost_per_month': 122.40}
                },
                'r6i': {
                    'r6i.large': {'vcpu': 2, 'memory': 16, 'cost_per_hour': 0.252, 'cost_per_month': 181.44},
                    'r6i.xlarge': {'vcpu': 4, 'memory': 32, 'cost_per_hour': 0.504, 'cost_per_month': 362.88}
                }
            },
            'rds': {
                'mysql': {
                    'db.t3.medium': {'vcpu': 2, 'memory': 4, 'cost_per_hour': 0.068, 'cost_per_month': 48.96},
                    'db.r6g.large': {'vcpu': 2, 'memory': 16, 'cost_per_hour': 0.48, 'cost_per_month': 345.60}
                }
            },
            'ebs': {
                'gp3': {'cost_per_gb_month': 0.08},
                'io1': {'cost_per_gb_month': 0.125},
                'io2': {'cost_per_gb_month': 0.125}
            },
            's3': {
                'Standard': {'cost_per_gb_month': 0.023},
                'Standard-IA': {'cost_per_gb_month': 0.0125}
            },
            'datasync': {
                'data_transfer': {'cost_per_gb': 0.0125}
            },
            'dms': {
                'dms.t3.medium': {'cost_per_hour': 0.0416, 'cost_per_month': 29.95}
            },
            'direct_connect': {
                '1Gbps': {'port_hours': 0.30, 'data_transfer_out': 0.02},
                '10Gbps': {'port_hours': 2.25, 'data_transfer_out': 0.02}
            },
            'fsx': {
                'fsx_windows': {'cost_per_gb_month': 0.13},
                'fsx_lustre': {'cost_per_gb_month': 0.14}
            },
            'cloudwatch': {
                'metrics': {'cost_per_metric_per_month': 0.30},
                'logs': {'cost_per_gb_ingested': 0.50}
            },
            'vpc': {
                'nat_gateway': {'cost_per_hour': 0.045}
            }
        }
        
        return fallback_pricing.get(service, {})
    
    def _get_fallback_instance_pricing(self, instance_type: str) -> Dict:
        """Get fallback pricing for specific instance"""
        return {'vcpu': 2, 'memory': 4, 'cost_per_hour': 0.05, 'cost_per_month': 36.00}
    
    def _get_fallback_rds_pricing(self, instance_class: str) -> Dict:
        """Get fallback RDS pricing"""
        return {'vcpu': 2, 'memory': 4, 'cost_per_hour': 0.068, 'cost_per_month': 48.96}
    
    def _get_fallback_ebs_pricing(self, volume_type: str) -> Dict:
        """Get fallback EBS pricing"""
        return {'cost_per_gb_month': 0.08}
    
    def _get_fallback_s3_pricing(self, storage_class: str) -> Dict:
        """Get fallback S3 pricing"""
        return {'cost_per_gb_month': 0.023}
    
    def _get_fallback_datasync_pricing(self) -> Dict:
        """Get fallback DataSync pricing"""
        return {'data_transfer': {'cost_per_gb': 0.0125}}
    
    def _get_fallback_dms_pricing(self, instance_class: str) -> Dict:
        """Get fallback DMS pricing"""
        return {'cost_per_hour': 0.0416, 'cost_per_month': 29.95}

class AnthropicAIManager:
    """Enhanced Anthropic AI manager with improved error handling and connection"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or st.secrets.get("ANTHROPIC_API_KEY")
        self.client = None
        self.connected = False
        self.error_message = None
        
        if self.api_key:
            try:
                self.client = anthropic.Anthropic(api_key=self.api_key)
                test_message = self.client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=10,
                    messages=[{"role": "user", "content": "test"}]
                )
                self.connected = True
                logger.info("Anthropic AI client initialized and tested successfully")
            except Exception as e:
                self.connected = False
                self.error_message = str(e)
                logger.error(f"Failed to initialize Anthropic client: {e}")
        else:
            self.connected = False
            self.error_message = "No API key provided"

class ComprehensiveAWSServiceSizer:
    """Comprehensive AWS service sizing for all migration components"""
    
    def __init__(self, aws_api_manager: EnhancedAWSAPIManager):
        self.aws_api = aws_api_manager
    
    async def generate_comprehensive_aws_sizing(self, config: Dict, analysis: Dict) -> Dict:
        """Generate comprehensive AWS service sizing and specifications"""
        
        # Get real-time pricing data
        pricing_data = await self.aws_api.get_comprehensive_pricing()
        
        # Calculate sizing for all services
        compute_sizing = await self._calculate_compute_sizing(config, pricing_data)
        storage_sizing = await self._calculate_storage_sizing(config, pricing_data)
        migration_services_sizing = await self._calculate_migration_services_sizing(config, analysis, pricing_data)
        network_sizing = await self._calculate_network_sizing(config, pricing_data)
        monitoring_sizing = await self._calculate_monitoring_sizing(config, pricing_data)
        backup_services_sizing = await self._calculate_backup_services_sizing(config, pricing_data)
        
        return {
            'pricing_data_source': pricing_data.get('data_source', 'fallback'),
            'last_updated': pricing_data.get('last_updated', datetime.now()),
            'region': pricing_data.get('region', 'us-west-2'),
            'compute_services': compute_sizing,
            'storage_services': storage_sizing,
            'migration_services': migration_services_sizing,
            'network_services': network_sizing,
            'monitoring_services': monitoring_sizing,
            'backup_services': backup_services_sizing,
            'total_monthly_cost': self._calculate_total_monthly_cost([
                compute_sizing, storage_sizing, migration_services_sizing,
                network_sizing, monitoring_sizing, backup_services_sizing
            ]),
            'service_summary': self._generate_service_summary([
                compute_sizing, storage_sizing, migration_services_sizing,
                network_sizing, monitoring_sizing, backup_services_sizing
            ])
        }
    
    async def _calculate_compute_sizing(self, config: Dict, pricing_data: Dict) -> Dict:
        """Calculate comprehensive compute sizing (EC2/RDS)"""
        target_platform = config.get('target_platform', 'rds')
        database_size_gb = config.get('database_size_gb', 1000)
        environment = config.get('environment', 'non-production')
        
        compute_services = {}
        
        if target_platform == 'rds':
            # RDS Primary Instance
            rds_instance_type = self._determine_rds_instance_type(config)
            rds_pricing = pricing_data.get('services', {}).get('rds', {})
            db_engine = config.get('database_engine', 'mysql')
            
            rds_instance_pricing = rds_pricing.get(db_engine, {}).get(rds_instance_type, {
                'cost_per_hour': 0.48, 'cost_per_month': 345.60, 'vcpu': 2, 'memory': 16
            })
            
            compute_services['rds_primary'] = {
                'service_name': 'Amazon RDS Primary Instance',
                'instance_type': rds_instance_type,
                'database_engine': db_engine,
                'specifications': {
                    'vcpu': rds_instance_pricing.get('vcpu', 2),
                    'memory_gb': rds_instance_pricing.get('memory', 16),
                    'network_performance': rds_instance_pricing.get('network_performance', 'Up to 10 Gbps'),
                    'deployment_option': 'Multi-AZ' if environment == 'production' else 'Single-AZ'
                },
                'pricing': {
                    'cost_per_hour': rds_instance_pricing.get('cost_per_hour', 0.48),
                    'monthly_cost': rds_instance_pricing.get('cost_per_month', 345.60),
                    'annual_cost': rds_instance_pricing.get('cost_per_month', 345.60) * 12
                },
                'quantity': 1
            }
            
            # Read Replicas
            if database_size_gb > 1000 or environment == 'production':
                read_replica_count = 2 if environment == 'production' else 1
                compute_services['rds_read_replicas'] = {
                    'service_name': 'Amazon RDS Read Replicas',
                    'instance_type': rds_instance_type,
                    'database_engine': db_engine,
                    'specifications': {
                        'vcpu': rds_instance_pricing.get('vcpu', 2),
                        'memory_gb': rds_instance_pricing.get('memory', 16),
                        'network_performance': rds_instance_pricing.get('network_performance', 'Up to 10 Gbps'),
                        'deployment_option': 'Single-AZ'
                    },
                    'pricing': {
                        'cost_per_hour': rds_instance_pricing.get('cost_per_hour', 0.48),
                        'monthly_cost': rds_instance_pricing.get('cost_per_month', 345.60) * read_replica_count,
                        'annual_cost': rds_instance_pricing.get('cost_per_month', 345.60) * read_replica_count * 12
                    },
                    'quantity': read_replica_count
                }
        
        else:  # EC2
            # EC2 Primary Instance
            ec2_instance_type = self._determine_ec2_instance_type(config)
            ec2_pricing = pricing_data.get('services', {}).get('ec2', {})
            
            # Determine instance family
            instance_family = ec2_instance_type.split('.')[0]
            ec2_instance_pricing = ec2_pricing.get(instance_family, {}).get(ec2_instance_type, {
                'cost_per_hour': 0.252, 'cost_per_month': 181.44, 'vcpu': 2, 'memory': 16
            })
            
            # SQL Server Always On consideration
            sql_deployment = config.get('sql_server_deployment_type', 'standalone')
            instance_count = 3 if sql_deployment == 'always_on' else 1
            
            compute_services['ec2_database'] = {
                'service_name': 'Amazon EC2 Database Instances',
                'instance_type': ec2_instance_type,
                'database_engine': config.get('ec2_database_engine', config.get('database_engine', 'mysql')),
                'deployment_type': sql_deployment,
                'specifications': {
                    'vcpu': ec2_instance_pricing.get('vcpu', 2),
                    'memory_gb': ec2_instance_pricing.get('memory', 16),
                    'network_performance': ec2_instance_pricing.get('network_performance', 'Up to 10 Gbps'),
                    'ebs_optimized': True,
                    'enhanced_networking': True
                },
                'pricing': {
                    'cost_per_hour': ec2_instance_pricing.get('cost_per_hour', 0.252),
                    'monthly_cost': ec2_instance_pricing.get('cost_per_month', 181.44) * instance_count,
                    'annual_cost': ec2_instance_pricing.get('cost_per_month', 181.44) * instance_count * 12
                },
                'quantity': instance_count
            }
            
            # Additional clustering requirements for Always On
            if sql_deployment == 'always_on':
                compute_services['ec2_database']['cluster_requirements'] = {
                    'wsfc_quorum': 'File Share Witness or Cloud Witness',
                    'cluster_network': 'Dedicated cluster communication network',
                    'shared_storage': 'EBS volumes with cross-AZ replication',
                    'load_balancer': 'Application Load Balancer for listener'
                }
        
        return compute_services
    
    async def _calculate_storage_sizing(self, config: Dict, pricing_data: Dict) -> Dict:
        """Calculate comprehensive storage sizing"""
        database_size_gb = config.get('database_size_gb', 1000)
        target_platform = config.get('target_platform', 'rds')
        destination_storage = config.get('destination_storage_type', 'S3')
        migration_method = config.get('migration_method', 'direct_replication')
        
        storage_services = {}
        
        # Primary Database Storage
        if target_platform == 'rds':
            # RDS Storage
            storage_size = max(database_size_gb * 1.5, 100)
            storage_type = 'gp3'
            
            ebs_pricing = pricing_data.get('services', {}).get('ebs', {})
            storage_cost_per_gb = ebs_pricing.get(storage_type, {}).get('cost_per_gb_month', 0.08)
            
            storage_services['rds_storage'] = {
                'service_name': 'Amazon RDS Storage (GP3)',
                'storage_type': storage_type.upper(),
                'specifications': {
                    'size_gb': storage_size,
                    'baseline_iops': 3000,
                    'baseline_throughput_mbps': 125,
                    'encryption': 'Enabled',
                    'backup_retention': '30 days' if config.get('environment') == 'production' else '7 days'
                },
                'pricing': {
                    'cost_per_gb_month': storage_cost_per_gb,
                    'monthly_cost': storage_size * storage_cost_per_gb,
                    'annual_cost': storage_size * storage_cost_per_gb * 12
                }
            }
        else:
            # EC2 EBS Storage
            storage_size = max(database_size_gb * 2.0, 100)
            storage_type = 'gp3'
            
            ebs_pricing = pricing_data.get('services', {}).get('ebs', {})
            storage_cost_per_gb = ebs_pricing.get(storage_type, {}).get('cost_per_gb_month', 0.08)
            
            # Calculate number of instances for storage
            sql_deployment = config.get('sql_server_deployment_type', 'standalone')
            instance_count = 3 if sql_deployment == 'always_on' else 1
            
            storage_services['ec2_ebs_storage'] = {
                'service_name': 'Amazon EBS Storage (GP3)',
                'storage_type': storage_type.upper(),
                'specifications': {
                    'size_gb_per_instance': storage_size,
                    'total_size_gb': storage_size * instance_count,
                    'baseline_iops': 3000,
                    'baseline_throughput_mbps': 125,
                    'encryption': 'Enabled',
                    'instance_count': instance_count
                },
                'pricing': {
                    'cost_per_gb_month': storage_cost_per_gb,
                    'monthly_cost': storage_size * storage_cost_per_gb * instance_count,
                    'annual_cost': storage_size * storage_cost_per_gb * instance_count * 12
                }
            }
        
        # Destination Storage
        if destination_storage == 'S3':
            s3_pricing = pricing_data.get('services', {}).get('s3', {})
            s3_cost_per_gb = s3_pricing.get('Standard', {}).get('cost_per_gb_month', 0.023)
            
            storage_services['s3_destination'] = {
                'service_name': 'Amazon S3 Standard Storage',
                'storage_type': 'S3 Standard',
                'specifications': {
                    'size_gb': database_size_gb * 1.2,
                    'durability': '99.999999999% (11 9s)',
                    'availability': '99.99%',
                    'storage_class': 'Standard',
                    'lifecycle_policies': 'Enabled'
                },
                'pricing': {
                    'cost_per_gb_month': s3_cost_per_gb,
                    'monthly_cost': database_size_gb * 1.2 * s3_cost_per_gb,
                    'annual_cost': database_size_gb * 1.2 * s3_cost_per_gb * 12
                }
            }
        elif destination_storage == 'FSx_Windows':
            fsx_pricing = pricing_data.get('services', {}).get('fsx', {})
            fsx_cost_per_gb = fsx_pricing.get('fsx_windows', {}).get('cost_per_gb_month', 0.13)
            
            storage_services['fsx_windows'] = {
                'service_name': 'Amazon FSx for Windows File Server',
                'storage_type': 'FSx Windows',
                'specifications': {
                    'size_gb': database_size_gb * 1.2,
                    'throughput_capacity_mbps': 64,
                    'deployment_type': 'Multi-AZ',
                    'backup_retention': '7 days',
                    'deduplication': 'Enabled'
                },
                'pricing': {
                    'cost_per_gb_month': fsx_cost_per_gb,
                    'monthly_cost': database_size_gb * 1.2 * fsx_cost_per_gb,
                    'annual_cost': database_size_gb * 1.2 * fsx_cost_per_gb * 12
                }
            }
        elif destination_storage == 'FSx_Lustre':
            fsx_pricing = pricing_data.get('services', {}).get('fsx', {})
            fsx_cost_per_gb = fsx_pricing.get('fsx_lustre', {}).get('cost_per_gb_month', 0.14)
            
            storage_services['fsx_lustre'] = {
                'service_name': 'Amazon FSx for Lustre',
                'storage_type': 'FSx Lustre',
                'specifications': {
                    'size_gb': database_size_gb * 1.2,
                    'throughput_capacity_mbps': 500,
                    'deployment_type': 'Persistent',
                    'data_compression': 'LZ4',
                    's3_integration': 'Enabled'
                },
                'pricing': {
                    'cost_per_gb_month': fsx_cost_per_gb,
                    'monthly_cost': database_size_gb * 1.2 * fsx_cost_per_gb,
                    'annual_cost': database_size_gb * 1.2 * fsx_cost_per_gb * 12
                }
            }
        
        # Backup Storage (if backup/restore method)
        if migration_method == 'backup_restore':
            backup_size_multiplier = config.get('backup_size_multiplier', 0.7)
            backup_size_gb = database_size_gb * backup_size_multiplier
            
            s3_pricing = pricing_data.get('services', {}).get('s3', {})
            s3_cost_per_gb = s3_pricing.get('Standard', {}).get('cost_per_gb_month', 0.023)
            
            storage_services['backup_storage'] = {
                'service_name': 'Amazon S3 Backup Storage',
                'storage_type': 'S3 Standard',
                'specifications': {
                    'size_gb': backup_size_gb,
                    'purpose': 'Migration backup files',
                    'lifecycle_policy': '30 days retention',
                    'backup_storage_type': config.get('backup_storage_type', 'nas_drive')
                },
                'pricing': {
                    'cost_per_gb_month': s3_cost_per_gb,
                    'monthly_cost': backup_size_gb * s3_cost_per_gb,
                    'annual_cost': backup_size_gb * s3_cost_per_gb * 12
                }
            }
        
        return storage_services
    
    async def _calculate_migration_services_sizing(self, config: Dict, analysis: Dict, pricing_data: Dict) -> Dict:
        """Calculate migration services sizing (DataSync/DMS)"""
        migration_method = config.get('migration_method', 'direct_replication')
        num_agents = config.get('number_of_agents', 1)
        database_size_gb = config.get('database_size_gb', 1000)
        
        migration_services = {}
        
        if migration_method == 'backup_restore':
            # DataSync Agents
            agent_size = config.get('datasync_agent_size', 'medium')
            ec2_pricing = pricing_data.get('services', {}).get('ec2', {})
            
            # Map agent size to EC2 instance type
            agent_instance_map = {
                'small': ('t3', 't3.medium'),
                'medium': ('c5', 'c5.large'),
                'large': ('c5', 'c5.xlarge'),
                'xlarge': ('c5', 'c5.2xlarge')
            }
            
            instance_family, instance_type = agent_instance_map.get(agent_size, ('c5', 'c5.large'))
            agent_pricing = ec2_pricing.get(instance_family, {}).get(instance_type, {
                'cost_per_hour': 0.085, 'cost_per_month': 61.20, 'vcpu': 2, 'memory': 4
            })
            
            migration_services['datasync_agents'] = {
                'service_name': 'AWS DataSync Agents (EC2)',
                'agent_type': 'DataSync',
                'instance_type': instance_type,
                'specifications': {
                    'vcpu': agent_pricing.get('vcpu', 2),
                    'memory_gb': agent_pricing.get('memory', 4),
                    'max_throughput_mbps': self._get_agent_throughput(agent_size),
                    'concurrent_tasks': self._get_agent_concurrent_tasks(agent_size),
                    'network_optimization': 'Enabled'
                },
                'pricing': {
                    'cost_per_hour': agent_pricing.get('cost_per_hour', 0.085),
                    'monthly_cost': agent_pricing.get('cost_per_month', 61.20) * num_agents,
                    'annual_cost': agent_pricing.get('cost_per_month', 61.20) * num_agents * 12
                },
                'quantity': num_agents
            }
            
            # DataSync Service Costs
            datasync_pricing = pricing_data.get('services', {}).get('datasync', {})
            cost_per_gb = datasync_pricing.get('data_transfer', {}).get('cost_per_gb', 0.0125)
            
            backup_size_multiplier = config.get('backup_size_multiplier', 0.7)
            backup_size_gb = database_size_gb * backup_size_multiplier
            
            migration_services['datasync_transfer'] = {
                'service_name': 'AWS DataSync Data Transfer',
                'transfer_type': 'One-time migration',
                'specifications': {
                    'data_size_gb': backup_size_gb,
                    'source_type': config.get('backup_storage_type', 'nas_drive'),
                    'destination_type': 'S3',
                    'compression': 'Enabled',
                    'encryption_in_transit': 'TLS 1.2'
                },
                'pricing': {
                    'cost_per_gb': cost_per_gb,
                    'one_time_cost': backup_size_gb * cost_per_gb,
                    'monthly_cost': 0,  # One-time transfer
                    'annual_cost': 0
                }
            }
        
        else:
            # DMS Replication Instance
            is_homogeneous = config['source_database_engine'] == config['database_engine']
            
            if not is_homogeneous:
                agent_size = config.get('dms_agent_size', 'medium')
                dms_pricing = pricing_data.get('services', {}).get('dms', {})
                
                # Map DMS agent size to instance type
                dms_instance_map = {
                    'small': 'dms.t3.medium',
                    'medium': 'dms.c5.large',
                    'large': 'dms.c5.xlarge',
                    'xlarge': 'dms.c5.2xlarge',
                    'xxlarge': 'dms.c5.4xlarge'
                }
                
                instance_type = dms_instance_map.get(agent_size, 'dms.c5.large')
                dms_instance_pricing = dms_pricing.get(instance_type, {
                    'cost_per_hour': 0.085, 'cost_per_month': 61.20
                })
                
                migration_services['dms_replication'] = {
                    'service_name': 'AWS DMS Replication Instance',
                    'instance_type': instance_type,
                    'migration_type': 'Heterogeneous (Schema Conversion)',
                    'specifications': {
                        'allocated_storage_gb': 100,
                        'multi_az': config.get('environment') == 'production',
                        'replication_subnet_group': 'Required',
                        'source_engine': config.get('source_database_engine'),
                        'target_engine': config.get('database_engine')
                    },
                    'pricing': {
                        'cost_per_hour': dms_instance_pricing.get('cost_per_hour', 0.085),
                        'monthly_cost': dms_instance_pricing.get('cost_per_month', 61.20) * num_agents,
                        'annual_cost': dms_instance_pricing.get('cost_per_month', 61.20) * num_agents * 12
                    },
                    'quantity': num_agents
                }
                
                # Schema Conversion Tool
                migration_services['sct_conversion'] = {
                    'service_name': 'AWS Schema Conversion Tool (SCT)',
                    'conversion_type': f"{config.get('source_database_engine')} to {config.get('database_engine')}",
                    'specifications': {
                        'license_cost': 'Free',
                        'assessment_report': 'Included',
                        'conversion_support': 'Automated + Manual',
                        'code_analysis': 'Included'
                    },
                    'pricing': {
                        'cost_per_hour': 0,
                        'monthly_cost': 0,
                        'annual_cost': 0,
                        'note': 'No additional cost for SCT software'
                    }
                }
        
        return migration_services
    
    async def _calculate_network_sizing(self, config: Dict, pricing_data: Dict) -> Dict:
        """Calculate network services sizing"""
        environment = config.get('environment', 'non-production')
        database_size_gb = config.get('database_size_gb', 1000)
        
        network_services = {}
        
        # Direct Connect
        dx_capacity = '10Gbps' if environment == 'production' else '1Gbps'
        dx_pricing = pricing_data.get('services', {}).get('direct_connect', {})
        dx_cost = dx_pricing.get(dx_capacity, {})
        
        network_services['direct_connect'] = {
            'service_name': f'AWS Direct Connect ({dx_capacity})',
            'connection_type': 'Dedicated Connection',
            'specifications': {
                'bandwidth': dx_capacity,
                'port_speed': dx_capacity,
                'location': 'Customer Data Center',
                'redundancy': 'Single connection',
                'bgp_sessions': '1 per VIF',
                'vlan_support': 'Up to 50 VLANs'
            },
            'pricing': {
                'port_hours': dx_cost.get('port_hours', 2.25),
                'data_transfer_out_per_gb': dx_cost.get('data_transfer_out', 0.02),
                'monthly_port_cost': dx_cost.get('port_hours', 2.25) * 24 * 30,
                'estimated_monthly_transfer_cost': (database_size_gb * 0.1) * dx_cost.get('data_transfer_out', 0.02),
                'annual_cost': (dx_cost.get('port_hours', 2.25) * 24 * 30 * 12) + ((database_size_gb * 0.1 * 12) * dx_cost.get('data_transfer_out', 0.02))
            }
        }
        
        # VPC Endpoints
        vpc_pricing = pricing_data.get('services', {}).get('vpc', {})
        vpc_endpoint_cost = vpc_pricing.get('vpc_endpoints', {})
        
        network_services['vpc_endpoints'] = {
            'service_name': 'VPC Endpoints for AWS Services',
            'endpoint_type': 'Interface Endpoints',
            'specifications': {
                'endpoints_required': ['S3', 'DMS', 'DataSync', 'CloudWatch'],
                'availability_zones': 2,
                'dns_resolution': 'Private DNS names',
                'security_groups': 'Custom security groups'
            },
            'pricing': {
                'cost_per_hour': vpc_endpoint_cost.get('cost_per_hour', 0.01) * 4,  # 4 endpoints
                'monthly_cost': vpc_endpoint_cost.get('cost_per_hour', 0.01) * 4 * 24 * 30,
                'annual_cost': vpc_endpoint_cost.get('cost_per_hour', 0.01) * 4 * 24 * 30 * 12
            }
        }
        
        # NAT Gateway (if needed)
        nat_gateway_cost = vpc_pricing.get('nat_gateway', {})
        
        network_services['nat_gateway'] = {
            'service_name': 'NAT Gateway',
            'gateway_type': 'NAT Gateway',
            'specifications': {
                'bandwidth': 'Up to 45 Gbps',
                'availability_zones': 2,
                'data_processing': 'Up to 55 Gbps',
                'use_case': 'Outbound internet access for private subnets'
            },
            'pricing': {
                'cost_per_hour': nat_gateway_cost.get('cost_per_hour', 0.045) * 2,  # 2 AZs
                'cost_per_gb_processed': nat_gateway_cost.get('cost_per_gb_processed', 0.045),
                'monthly_cost': nat_gateway_cost.get('cost_per_hour', 0.045) * 2 * 24 * 30,
                'annual_cost': nat_gateway_cost.get('cost_per_hour', 0.045) * 2 * 24 * 30 * 12
            }
        }
        
        return network_services
    
    async def _calculate_monitoring_sizing(self, config: Dict, pricing_data: Dict) -> Dict:
        """Calculate monitoring services sizing"""
        environment = config.get('environment', 'non-production')
        
        monitoring_services = {}
        cloudwatch_pricing = pricing_data.get('services', {}).get('cloudwatch', {})
        
        # CloudWatch Metrics
        estimated_metrics = 50 if environment == 'production' else 25
        metrics_cost = cloudwatch_pricing.get('metrics', {})
        
        monitoring_services['cloudwatch_metrics'] = {
            'service_name': 'Amazon CloudWatch Custom Metrics',
            'metric_type': 'Custom Metrics',
            'specifications': {
                'estimated_metrics_count': estimated_metrics,
                'resolution': '1 minute',
                'retention_period': '15 months',
                'alarms_included': 10,
                'dashboards_included': 3
            },
            'pricing': {
                'cost_per_metric_month': metrics_cost.get('cost_per_metric_per_month', 0.30),
                'monthly_cost': estimated_metrics * metrics_cost.get('cost_per_metric_per_month', 0.30),
                'annual_cost': estimated_metrics * metrics_cost.get('cost_per_metric_per_month', 0.30) * 12
            }
        }
        
        # CloudWatch Logs
        estimated_log_ingestion_gb = 10 if environment == 'production' else 5
        logs_cost = cloudwatch_pricing.get('logs', {})
        
        monitoring_services['cloudwatch_logs'] = {
            'service_name': 'Amazon CloudWatch Logs',
            'log_type': 'Application and System Logs',
            'specifications': {
                'estimated_ingestion_gb_month': estimated_log_ingestion_gb,
                'retention_period': '30 days',
                'log_groups': ['Application', 'Database', 'Migration', 'System'],
                'insights_queries': 'Included'
            },
            'pricing': {
                'cost_per_gb_ingested': logs_cost.get('cost_per_gb_ingested', 0.50),
                'cost_per_gb_storage_month': logs_cost.get('cost_per_gb_storage_per_month', 0.03),
                'monthly_ingestion_cost': estimated_log_ingestion_gb * logs_cost.get('cost_per_gb_ingested', 0.50),
                'monthly_storage_cost': estimated_log_ingestion_gb * logs_cost.get('cost_per_gb_storage_per_month', 0.03),
                'monthly_cost': (estimated_log_ingestion_gb * logs_cost.get('cost_per_gb_ingested', 0.50)) + (estimated_log_ingestion_gb * logs_cost.get('cost_per_gb_storage_per_month', 0.03)),
                'annual_cost': ((estimated_log_ingestion_gb * logs_cost.get('cost_per_gb_ingested', 0.50)) + (estimated_log_ingestion_gb * logs_cost.get('cost_per_gb_storage_per_month', 0.03))) * 12
            }
        }
        
        # CloudWatch Dashboards
        dashboard_count = 3
        dashboard_cost = cloudwatch_pricing.get('dashboards', {})
        
        monitoring_services['cloudwatch_dashboards'] = {
            'service_name': 'Amazon CloudWatch Dashboards',
            'dashboard_type': 'Custom Dashboards',
            'specifications': {
                'dashboard_count': dashboard_count,
                'widgets_per_dashboard': 20,
                'update_frequency': 'Real-time',
                'dashboard_types': ['Migration Progress', 'System Health', 'Cost Monitoring']
            },
            'pricing': {
                'cost_per_dashboard_month': dashboard_cost.get('cost_per_dashboard_per_month', 3.00),
                'monthly_cost': dashboard_count * dashboard_cost.get('cost_per_dashboard_per_month', 3.00),
                'annual_cost': dashboard_count * dashboard_cost.get('cost_per_dashboard_per_month', 3.00) * 12
            }
        }
        
        return monitoring_services
    
    async def _calculate_backup_services_sizing(self, config: Dict, pricing_data: Dict) -> Dict:
        """Calculate backup and disaster recovery services sizing"""
        database_size_gb = config.get('database_size_gb', 1000)
        environment = config.get('environment', 'non-production')
        target_platform = config.get('target_platform', 'rds')
        
        backup_services = {}
        
        # AWS Backup
        backup_frequency = 'Daily' if environment == 'production' else 'Weekly'
        retention_days = 30 if environment == 'production' else 7
        
        # Estimate backup storage size (typically 50-70% of database size with compression)
        backup_size_gb = database_size_gb * 0.6
        
        s3_pricing = pricing_data.get('services', {}).get('s3', {})
        backup_storage_cost = s3_pricing.get('Standard-IA', {}).get('cost_per_gb_month', 0.0125)
        
        backup_services['aws_backup'] = {
            'service_name': 'AWS Backup Service',
            'backup_type': 'Automated Database Backups',
            'specifications': {
                'backup_frequency': backup_frequency,
                'retention_period_days': retention_days,
                'estimated_backup_size_gb': backup_size_gb,
                'compression': 'Enabled',
                'encryption': 'AES-256',
                'cross_region_copy': environment == 'production'
            },
            'pricing': {
                'storage_cost_per_gb_month': backup_storage_cost,
                'restore_cost_per_gb': 0.02,
                'monthly_storage_cost': backup_size_gb * backup_storage_cost,
                'annual_cost': backup_size_gb * backup_storage_cost * 12
            }
        }
        
        # Point-in-Time Recovery (for RDS)
        if target_platform == 'rds':
            backup_services['rds_pitr'] = {
                'service_name': 'RDS Point-in-Time Recovery',
                'recovery_type': 'Transaction Log Backups',
                'specifications': {
                    'rpo_minutes': 5,
                    'retention_period_days': retention_days,
                    'automated_backups': 'Enabled',
                    'backup_window': '03:00-04:00 UTC',
                    'maintenance_window': 'Sun:04:00-Sun:05:00 UTC'
                },
                'pricing': {
                    'included_in_rds': True,
                    'additional_cost': 0,
                    'note': 'Included in RDS instance cost'
                }
            }
        
        # Disaster Recovery
        if environment == 'production':
            backup_services['disaster_recovery'] = {
                'service_name': 'Disaster Recovery Setup',
                'dr_type': 'Cross-Region Standby',
                'specifications': {
                    'rto_hours': 4,
                    'rpo_minutes': 15,
                    'standby_region': 'us-east-1',
                    'data_replication': 'Asynchronous',
                    'automated_failover': 'Manual trigger'
                },
                'pricing': {
                    'standby_instance_cost': 200,  # Estimated monthly cost for standby
                    'cross_region_data_transfer': database_size_gb * 0.1 * 0.02,  # 10% monthly change
                    'monthly_cost': 200 + (database_size_gb * 0.1 * 0.02),
                    'annual_cost': (200 + (database_size_gb * 0.1 * 0.02)) * 12
                }
            }
        
        return backup_services
    
    def _determine_rds_instance_type(self, config: Dict) -> str:
        """Determine optimal RDS instance type based on requirements"""
        memory_gb = config.get('current_db_max_memory_gb', 0)
        cpu_cores = config.get('current_db_max_cpu_cores', 0)
        database_size_gb = config.get('database_size_gb', 1000)
        
        # Apply 25% buffer for cloud environment
        required_memory = memory_gb * 1.25 if memory_gb > 0 else database_size_gb / 100
        required_cpu = cpu_cores * 1.25 if cpu_cores > 0 else database_size_gb / 500
        
        if required_memory > 64 or required_cpu > 16:
            return 'db.r6g.4xlarge'
        elif required_memory > 32 or required_cpu > 8:
            return 'db.r6g.2xlarge'
        elif required_memory > 16 or required_cpu > 4:
            return 'db.r6g.xlarge'
        elif required_memory > 8 or required_cpu > 2:
            return 'db.r6g.large'
        else:
            return 'db.t3.medium'
    
    def _determine_ec2_instance_type(self, config: Dict) -> str:
        """Determine optimal EC2 instance type based on requirements"""
        memory_gb = config.get('current_db_max_memory_gb', 0)
        cpu_cores = config.get('current_db_max_cpu_cores', 0)
        database_size_gb = config.get('database_size_gb', 1000)
        
        # Apply 30% buffer for self-managed environment
        required_memory = memory_gb * 1.30 if memory_gb > 0 else database_size_gb / 80
        required_cpu = cpu_cores * 1.30 if cpu_cores > 0 else database_size_gb / 400
        
        if required_memory > 128 or required_cpu > 16:
            return 'r6i.4xlarge'
        elif required_memory > 64 or required_cpu > 8:
            return 'r6i.2xlarge'
        elif required_memory > 32 or required_cpu > 4:
            return 'r6i.xlarge'
        elif required_memory > 16 or required_cpu > 2:
            return 'r6i.large'
        else:
            return 't3.large'
    
    def _get_agent_throughput(self, agent_size: str) -> int:
        """Get maximum throughput for agent size"""
        throughput_map = {
            'small': 250,
            'medium': 500,
            'large': 1000,
            'xlarge': 2000,
            'xxlarge': 2500
        }
        return throughput_map.get(agent_size, 500)
    
    def _get_agent_concurrent_tasks(self, agent_size: str) -> int:
        """Get maximum concurrent tasks for agent size"""
        tasks_map = {
            'small': 10,
            'medium': 25,
            'large': 50,
            'xlarge': 100,
            'xxlarge': 200
        }
        return tasks_map.get(agent_size, 25)
    
    def _calculate_total_monthly_cost(self, service_categories: List[Dict]) -> float:
        """Calculate total monthly cost across all service categories"""
        total_cost = 0
        
        for category in service_categories:
            for service_key, service_data in category.items():
                if isinstance(service_data, dict) and 'pricing' in service_data:
                    pricing = service_data['pricing']
                    if 'monthly_cost' in pricing:
                        total_cost += pricing['monthly_cost']
        
        return total_cost
    
    def _generate_service_summary(self, service_categories: List[Dict]) -> Dict:
        """Generate comprehensive service summary"""
        summary = {
            'total_services': 0,
            'service_breakdown': {},
            'cost_breakdown': {},
            'specifications_summary': {}
        }
        
        for category in service_categories:
            for service_key, service_data in category.items():
                if isinstance(service_data, dict):
                    summary['total_services'] += 1
                    service_name = service_data.get('service_name', service_key)
                    
                    # Cost breakdown
                    if 'pricing' in service_data:
                        monthly_cost = service_data['pricing'].get('monthly_cost', 0)
                        summary['cost_breakdown'][service_name] = monthly_cost
                    
                    # Service type breakdown
                    service_type = service_name.split()[0]  # Get first word (AWS, Amazon, etc.)
                    if service_type not in summary['service_breakdown']:
                        summary['service_breakdown'][service_type] = 0
                    summary['service_breakdown'][service_type] += 1
        
        return summary

class ComprehensiveCostAnalyzer:
    """Comprehensive cost analyzer with detailed tabular analysis"""
    
    def __init__(self, aws_service_sizer: ComprehensiveAWSServiceSizer):
        self.service_sizer = aws_service_sizer
    
    async def generate_detailed_cost_analysis(self, config: Dict, analysis: Dict) -> Dict:
        """Generate comprehensive cost analysis in tabular format"""
        
        # Get comprehensive AWS sizing
        aws_sizing = await self.service_sizer.generate_comprehensive_aws_sizing(config, analysis)
        
        # Extract cost data from all services
        cost_tables = self._extract_all_cost_data(aws_sizing, config, analysis)
        
        # Generate cost projections
        cost_projections = self._generate_cost_projections(cost_tables)
        
        # Generate optimization recommendations
        optimization_recommendations = self._generate_optimization_recommendations(cost_tables, config)
        
        return {
            'cost_tables': cost_tables,
            'cost_projections': cost_projections,
            'optimization_recommendations': optimization_recommendations,
            'pricing_source': aws_sizing.get('pricing_data_source', 'fallback'),
            'last_updated': aws_sizing.get('last_updated', datetime.now()),
            'total_analysis': self._generate_total_cost_analysis(cost_tables)
        }
    
    def _extract_all_cost_data(self, aws_sizing: Dict, config: Dict, analysis: Dict) -> Dict:
        """Extract all cost data into structured tables"""
        
        cost_tables = {
            'monthly_recurring_costs': [],
            'one_time_costs': [],
            'variable_costs': [],
            'service_specifications': [],
            'cost_breakdown_by_category': []
        }
        
        # Process all service categories
        service_categories = [
            ('Compute Services', aws_sizing.get('compute_services', {})),
            ('Storage Services', aws_sizing.get('storage_services', {})),
            ('Migration Services', aws_sizing.get('migration_services', {})),
            ('Network Services', aws_sizing.get('network_services', {})),
            ('Monitoring Services', aws_sizing.get('monitoring_services', {})),
            ('Backup Services', aws_sizing.get('backup_services', {}))
        ]
        
        category_totals = {}
        
        for category_name, services in service_categories:
            category_monthly_total = 0
            category_one_time_total = 0
            
            for service_key, service_data in services.items():
                if not isinstance(service_data, dict):
                    continue
                
                service_name = service_data.get('service_name', service_key)
                pricing = service_data.get('pricing', {})
                specs = service_data.get('specifications', {})
                quantity = service_data.get('quantity', 1)
                
                # Monthly recurring costs
                monthly_cost = pricing.get('monthly_cost', 0)
                if monthly_cost > 0:
                    cost_tables['monthly_recurring_costs'].append({
                        'Category': category_name,
                        'Service Name': service_name,
                        'Service Type': service_data.get('instance_type', service_data.get('storage_type', 'N/A')),
                        'Quantity': quantity,
                        'Unit Cost/Month': f"${monthly_cost / max(quantity, 1):,.2f}",
                        'Total Monthly Cost': f"${monthly_cost:,.2f}",
                        'Annual Cost': f"${monthly_cost * 12:,.2f}",
                        'Notes': self._get_service_notes(service_data)
                    })
                    category_monthly_total += monthly_cost
                
                # One-time costs
                one_time_cost = pricing.get('one_time_cost', 0)
                if one_time_cost > 0:
                    cost_tables['one_time_costs'].append({
                        'Category': category_name,
                        'Service Name': service_name,
                        'Cost Type': 'Setup/Migration',
                        'One-Time Cost': f"${one_time_cost:,.2f}",
                        'Description': pricing.get('note', 'One-time setup cost')
                    })
                    category_one_time_total += one_time_cost
                
                # Variable costs
                if 'cost_per_gb' in pricing or 'cost_per_hour' in pricing:
                    cost_tables['variable_costs'].append({
                        'Category': category_name,
                        'Service Name': service_name,
                        'Pricing Model': 'Pay-per-use',
                        'Unit Rate': self._extract_unit_rate(pricing),
                        'Estimated Monthly Usage': self._estimate_monthly_usage(service_data, config),
                        'Estimated Monthly Cost': f"${monthly_cost:,.2f}"
                    })
                
                # Service specifications
                cost_tables['service_specifications'].append({
                    'Category': category_name,
                    'Service Name': service_name,
                    'Specifications': self._format_specifications(specs),
                    'Instance/Storage Type': service_data.get('instance_type', service_data.get('storage_type', 'N/A')),
                    'Quantity': quantity,
                    'Monthly Cost': f"${monthly_cost:,.2f}",
                    'Purpose': self._get_service_purpose(service_data, config)
                })
            
            # Category breakdown
            category_totals[category_name] = {
                'monthly_total': category_monthly_total,
                'one_time_total': category_one_time_total
            }
            
            cost_tables['cost_breakdown_by_category'].append({
                'Category': category_name,
                'Service Count': len([s for s in services.keys()]),
                'Monthly Total': f"${category_monthly_total:,.2f}",
                'Annual Total': f"${category_monthly_total * 12:,.2f}",
                'One-Time Total': f"${category_one_time_total:,.2f}",
                'Percentage of Total': f"{(category_monthly_total / max(sum(ct['monthly_total'] for ct in category_totals.values()), 1)) * 100:.1f}%"
            })
        
        return cost_tables
    
    def _get_service_notes(self, service_data: Dict) -> str:
        """Generate service-specific notes"""
        notes = []
        
        if service_data.get('deployment_type') == 'always_on':
            notes.append("SQL Server Always On (3-node cluster)")
        
        if service_data.get('specifications', {}).get('multi_az'):
            notes.append("Multi-AZ deployment")
        
        if 'backup' in service_data.get('service_name', '').lower():
            notes.append("Backup and DR service")
        
        if 'migration' in service_data.get('service_name', '').lower():
            notes.append("Migration-specific service")
        
        return '; '.join(notes) if notes else 'Standard configuration'
    
    def _extract_unit_rate(self, pricing: Dict) -> str:
        """Extract unit rate from pricing data"""
        if 'cost_per_gb' in pricing:
            return f"${pricing['cost_per_gb']:.4f}/GB"
        elif 'cost_per_hour' in pricing:
            return f"${pricing['cost_per_hour']:.4f}/hour"
        elif 'cost_per_gb_month' in pricing:
            return f"${pricing['cost_per_gb_month']:.4f}/GB/month"
        else:
            return "Fixed monthly"
    
    def _estimate_monthly_usage(self, service_data: Dict, config: Dict) -> str:
        """Estimate monthly usage for variable cost services"""
        service_name = service_data.get('service_name', '')
        
        if 'datasync' in service_name.lower():
            backup_size = config.get('database_size_gb', 1000) * config.get('backup_size_multiplier', 0.7)
            return f"{backup_size:,.0f} GB (one-time)"
        elif 'direct connect' in service_name.lower():
            return f"{config.get('database_size_gb', 1000) * 0.1:,.0f} GB/month"
        elif 'logs' in service_name.lower():
            return f"{10 if config.get('environment') == 'production' else 5} GB/month"
        else:
            return "Fixed allocation"
    
    def _format_specifications(self, specs: Dict) -> str:
        """Format specifications into readable string"""
        if not specs:
            return "Standard configuration"
        
        spec_parts = []
        
        if 'vcpu' in specs:
            spec_parts.append(f"{specs['vcpu']} vCPU")
        
        if 'memory_gb' in specs:
            spec_parts.append(f"{specs['memory_gb']} GB RAM")
        
        if 'size_gb' in specs:
            spec_parts.append(f"{specs['size_gb']:,.0f} GB storage")
        
        if 'throughput_capacity_mbps' in specs:
            spec_parts.append(f"{specs['throughput_capacity_mbps']} Mbps")
        
        if 'max_throughput_mbps' in specs:
            spec_parts.append(f"{specs['max_throughput_mbps']} Mbps max")
        
        return '; '.join(spec_parts) if spec_parts else "See service details"
    
    def _get_service_purpose(self, service_data: Dict, config: Dict) -> str:
        """Determine service purpose based on context"""
        service_name = service_data.get('service_name', '')
        
        if 'primary' in service_name.lower():
            return f"Primary {config.get('database_engine', 'database')} database"
        elif 'replica' in service_name.lower():
            return "Read scaling and high availability"
        elif 'backup' in service_name.lower():
            return "Data protection and compliance"
        elif 'migration' in service_name.lower():
            return "Data migration and synchronization"
        elif 'monitor' in service_name.lower():
            return "Performance monitoring and alerting"
        elif 'network' in service_name.lower():
            return "Secure connectivity and data transfer"
        else:
            return "Supporting infrastructure"
    
    def _generate_cost_projections(self, cost_tables: Dict) -> Dict:
        """Generate cost projections and trends"""
        
        monthly_costs = cost_tables.get('monthly_recurring_costs', [])
        one_time_costs = cost_tables.get('one_time_costs', [])
        
        # Calculate totals
        total_monthly = sum(float(item['Total Monthly Cost'].replace('$', '').replace(',', '')) for item in monthly_costs)
        total_one_time = sum(float(item['One-Time Cost'].replace('$', '').replace(',', '')) for item in one_time_costs)
        
        # Generate projections
        projections = {
            'cost_summary': {
                'total_monthly': total_monthly,
                'total_one_time': total_one_time,
                'annual_recurring': total_monthly * 12,
                'three_year_total': (total_monthly * 36) + total_one_time
            },
            'monthly_projections': [],
            'cost_trends': {
                'year_1': total_one_time + (total_monthly * 12),
                'year_2': total_monthly * 12,
                'year_3': total_monthly * 12,
                'average_monthly_y1': (total_one_time / 12) + total_monthly,
                'average_monthly_y2_y3': total_monthly
            }
        }
        
        # Generate monthly breakdown for first year
        for month in range(1, 13):
            monthly_cost = total_monthly
            if month == 1:
                monthly_cost += total_one_time  # Add one-time costs to first month
            
            projections['monthly_projections'].append({
                'Month': f"Month {month}",
                'Recurring Costs': f"${total_monthly:,.2f}",
                'One-Time Costs': f"${total_one_time if month == 1 else 0:,.2f}",
                'Total Monthly': f"${monthly_cost:,.2f}",
                'Cumulative Total': f"${(total_monthly * month) + total_one_time:,.2f}"
            })
        
        return projections
    
    def _generate_optimization_recommendations(self, cost_tables: Dict, config: Dict) -> List[Dict]:
        """Generate cost optimization recommendations"""
        
        recommendations = []
        monthly_costs = cost_tables.get('monthly_recurring_costs', [])
        
        # Analyze cost distribution
        total_monthly = sum(float(item['Total Monthly Cost'].replace('$', '').replace(',', '')) for item in monthly_costs)
        
        # Compute optimization
        compute_costs = [item for item in monthly_costs if 'compute' in item['Category'].lower()]
        compute_total = sum(float(item['Total Monthly Cost'].replace('$', '').replace(',', '')) for item in compute_costs)
        
        if compute_total > total_monthly * 0.6:
            recommendations.append({
                'category': 'Compute Optimization',
                'priority': 'High',
                'recommendation': 'Consider Reserved Instances for compute services',
                'potential_savings': '20-30%',
                'estimated_monthly_savings': f"${compute_total * 0.25:,.2f}",
                'implementation_effort': 'Low',
                'details': 'Reserved Instances can provide significant savings for predictable workloads'
            })
        
        # Storage optimization
        storage_costs = [item for item in monthly_costs if 'storage' in item['Category'].lower()]
        storage_total = sum(float(item['Total Monthly Cost'].replace('$', '').replace(',', '')) for item in storage_costs)
        
        if storage_total > total_monthly * 0.25:
            recommendations.append({
                'category': 'Storage Optimization',
                'priority': 'Medium',
                'recommendation': 'Implement storage lifecycle policies',
                'potential_savings': '15-25%',
                'estimated_monthly_savings': f"${storage_total * 0.20:,.2f}",
                'implementation_effort': 'Medium',
                'details': 'Move infrequently accessed data to cheaper storage tiers'
            })
        
        # Environment-specific recommendations
        if config.get('environment') == 'non-production':
            recommendations.append({
                'category': 'Development Environment',
                'priority': 'High',
                'recommendation': 'Use Spot Instances for non-production workloads',
                'potential_savings': '60-70%',
                'estimated_monthly_savings': f"${compute_total * 0.65:,.2f}",
                'implementation_effort': 'Medium',
                'details': 'Spot Instances can dramatically reduce costs for fault-tolerant workloads'
            })
        
        # Migration-specific recommendations
        migration_costs = [item for item in monthly_costs if 'migration' in item['Category'].lower()]
        if migration_costs:
            recommendations.append({
                'category': 'Migration Services',
                'priority': 'Medium',
                'recommendation': 'Right-size migration agents based on actual throughput',
                'potential_savings': '10-20%',
                'estimated_monthly_savings': f"${sum(float(item['Total Monthly Cost'].replace('$', '').replace(',', '')) for item in migration_costs) * 0.15:,.2f}",
                'implementation_effort': 'Low',
                'details': 'Monitor agent utilization and adjust size accordingly'
            })
        
        # SQL Server Always On specific
        if config.get('sql_server_deployment_type') == 'always_on':
            recommendations.append({
                'category': 'SQL Server Optimization',
                'priority': 'Medium',
                'recommendation': 'Evaluate Always On necessity vs cost',
                'potential_savings': '60-70%',
                'estimated_monthly_savings': f"${compute_total * 0.65:,.2f}",
                'implementation_effort': 'High',
                'details': 'Consider if high availability requirements justify 3x cost increase'
            })
        
        return recommendations
    
    def _generate_total_cost_analysis(self, cost_tables: Dict) -> Dict:
        """Generate total cost analysis summary"""
        
        monthly_costs = cost_tables.get('monthly_recurring_costs', [])
        one_time_costs = cost_tables.get('one_time_costs', [])
        categories = cost_tables.get('cost_breakdown_by_category', [])
        
        total_monthly = sum(float(item['Total Monthly Cost'].replace('$', '').replace(',', '')) for item in monthly_costs)
        total_one_time = sum(float(item['One-Time Cost'].replace('$', '').replace(',', '')) for item in one_time_costs)
        
        return {
            'summary_metrics': {
                'total_services': len(monthly_costs),
                'total_monthly_cost': total_monthly,
                'total_one_time_cost': total_one_time,
                'annual_cost': total_monthly * 12,
                'three_year_tco': (total_monthly * 36) + total_one_time
            },
            'cost_distribution': {
                'largest_category': max(categories, key=lambda x: float(x['Monthly Total'].replace('$', '').replace(',', '')))['Category'] if categories else 'Unknown',
                'average_service_cost': total_monthly / max(len(monthly_costs), 1),
                'monthly_range': {
                    'minimum': min((float(item['Total Monthly Cost'].replace('$', '').replace(',', '')) for item in monthly_costs), default=0),
                    'maximum': max((float(item['Total Monthly Cost'].replace('$', '').replace(',', '')) for item in monthly_costs), default=0)
                }
            },
            'financial_metrics': {
                'average_monthly_y1': (total_one_time / 12) + total_monthly,
                'average_monthly_y2_plus': total_monthly,
                'total_investment': total_one_time,
                'operational_expense': total_monthly
            }
        }

# Helper functions for rendering
def render_enhanced_header():
    """Enhanced header with professional styling"""
    ai_manager = AnthropicAIManager()
    aws_api = EnhancedAWSAPIManager()
    
    ai_status = "🟢" if ai_manager.connected else "🔴"
    aws_status = "🟢" if aws_api.connected else "🔴"
    
    st.markdown(f"""
    <div class="main-header">
        <h1>🤖 AWS Enterprise Database Migration Analyzer AI v3.0</h1>
        <p style="font-size: 1.2rem; margin-top: 0.5rem;">
            Professional-Grade Migration Analysis • AI-Powered Insights • Real-time AWS Integration • Comprehensive Service Sizing • Enterprise Cost Analysis
        </p>
        <div style="margin-top: 1rem; font-size: 0.8rem;">
            <span style="margin-right: 20px;">{ai_status} Anthropic Claude AI</span>
            <span style="margin-right: 20px;">{aws_status} AWS Pricing APIs</span>
            <span style="margin-right: 20px;">🟢 Comprehensive Service Sizing</span>
            <span style="margin-right: 20px;">🟢 Enterprise Cost Analysis</span>
            <span style="margin-right: 20px;">🟢 Real-time Pricing</span>
            <span>🟢 Tabular Cost Breakdown</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_api_status_sidebar():
    """Enhanced API status sidebar"""
    st.sidebar.markdown("### 🔌 System Status")
    
    ai_manager = AnthropicAIManager()
    aws_api = EnhancedAWSAPIManager()
    
    # Anthropic AI Status
    ai_status_class = "status-online" if ai_manager.connected else "status-offline"
    ai_status_text = "Connected" if ai_manager.connected else "Disconnected"
    
    st.sidebar.markdown(f"""
    <div class="api-status-card">
        <span class="status-indicator {ai_status_class}"></span>
        <strong>Anthropic Claude AI:</strong> {ai_status_text}
        {f"<br><small>Error: {ai_manager.error_message[:50]}...</small>" if ai_manager.error_message else ""}
    </div>
    """, unsafe_allow_html=True)
    
    # AWS API Status
    aws_status_class = "status-online" if aws_api.connected else "status-warning"
    aws_status_text = "Connected" if aws_api.connected else "Using Fallback Data"
    
    st.sidebar.markdown(f"""
    <div class="api-status-card">
        <span class="status-indicator {aws_status_class}"></span>
        <strong>AWS Pricing API:</strong> {aws_status_text}
    </div>
    """, unsafe_allow_html=True)

def render_enhanced_sidebar_controls():
    """Enhanced sidebar with comprehensive configuration options"""
    st.sidebar.header("🤖 AI-Powered Migration Configuration v3.0")
    
    render_api_status_sidebar()
    st.sidebar.markdown("---")
    
    # Operating System Selection
    st.sidebar.subheader("💻 Operating System")
    operating_system = st.sidebar.selectbox(
        "OS Selection",
        ["windows_server_2019", "windows_server_2022", "rhel_8", "rhel_9", "ubuntu_20_04", "ubuntu_22_04"],
        index=3,
        format_func=lambda x: {
            'windows_server_2019': '🔵 Windows Server 2019',
            'windows_server_2022': '🔵 Windows Server 2022 (Latest)',
            'rhel_8': '🔴 Red Hat Enterprise Linux 8',
            'rhel_9': '🔴 Red Hat Enterprise Linux 9 (Latest)',
            'ubuntu_20_04': '🟠 Ubuntu Server 20.04 LTS',
            'ubuntu_22_04': '🟠 Ubuntu Server 22.04 LTS (Latest)'
        }[x]
    )
    
    # Platform Configuration
    st.sidebar.subheader("🖥️ Server Platform")
    server_type = st.sidebar.selectbox(
        "Platform Type",
        ["physical", "vmware"],
        format_func=lambda x: "🏢 Physical Server" if x == "physical" else "☁️ VMware Virtual Machine"
    )
    
    # Hardware Configuration
    st.sidebar.subheader("⚙️ Hardware Configuration")
    ram_gb = st.sidebar.selectbox("RAM (GB)", [8, 16, 32, 64, 128, 256, 512], index=2)
    cpu_cores = st.sidebar.selectbox("CPU Cores", [2, 4, 8, 16, 24, 32, 48, 64], index=2)
    cpu_ghz = st.sidebar.selectbox("CPU GHz", [2.0, 2.4, 2.8, 3.2, 3.6, 4.0], index=3)
    
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
    st.sidebar.subheader("🔄 Migration Setup")
    
    source_database_engine = st.sidebar.selectbox(
        "Source Database",
        ["mysql", "postgresql", "oracle", "sqlserver", "mongodb"],
        index=3,
        format_func=lambda x: {
            'mysql': '🐬 MySQL', 'postgresql': '🐘 PostgreSQL', 'oracle': '🏛️ Oracle',
            'sqlserver': '🪟 SQL Server', 'mongodb': '🍃 MongoDB'
        }[x]
    )
    
    # Current Database Performance Metrics
    st.sidebar.subheader("📊 Current Database Performance")
    st.sidebar.markdown("*Enter your actual on-premise database metrics for accurate AWS sizing*")
    
    current_db_max_memory_gb = st.sidebar.number_input(
        "Current DB Max Memory (GB)", 
        min_value=1, max_value=1024, value=int(min(ram_gb * 0.8, 64)), step=1,
        help="Maximum memory currently allocated to your database"
    )
    
    current_db_max_cpu_cores = st.sidebar.number_input(
        "Current DB Max CPU Cores", 
        min_value=1, max_value=128, value=int(min(cpu_cores, 16)), step=1,
        help="Number of CPU cores currently allocated to your database"
    )
    
    current_db_max_iops = st.sidebar.number_input(
        "Current DB Max IOPS", 
        min_value=100, max_value=500000, value=10000, step=1000,
        help="Maximum IOPS your database currently achieves"
    )
    
    current_db_max_throughput_mbps = st.sidebar.number_input(
        "Current DB Max Throughput (MB/s)", 
        min_value=10, max_value=10000, value=500, step=50,
        help="Maximum throughput your database currently achieves"
    )
    
    st.sidebar.markdown("---")
    
    # Target Platform Selection
    target_platform = st.sidebar.selectbox(
        "Target Platform",
        ["rds", "ec2"],
        format_func=lambda x: {
            'rds': '☁️ Amazon RDS (Managed Service)',
            'ec2': '🖥️ Amazon EC2 (Self-Managed)'
        }[x]
    )
    
    # Initialize variables
    database_engine = None
    ec2_database_engine = None
    sql_server_deployment_type = None
    
    # Target Database Selection based on platform
    if target_platform == "rds":
        database_engine = st.sidebar.selectbox(
            "Target Database (AWS RDS)",
            ["mysql", "postgresql", "oracle", "sqlserver", "mongodb"],
            index=3 if source_database_engine == "sqlserver" else 0,
            format_func=lambda x: {
                'mysql': '☁️ RDS MySQL', 'postgresql': '☁️ RDS PostgreSQL', 'oracle': '☁️ RDS Oracle',
                'sqlserver': '☁️ RDS SQL Server', 'mongodb': '☁️ DocumentDB'
            }[x]
        )
        ec2_database_engine = None
    else:  # EC2
        if source_database_engine == "sqlserver":
            database_engine = st.sidebar.selectbox(
                "Target Database (EC2)",
                ["sqlserver", "mysql", "postgresql", "oracle", "mongodb"],
                index=0,
                format_func=lambda x: {
                    'sqlserver': '🪟 EC2 with SQL Server (Recommended for SQL Server sources)',
                    'mysql': '🖥️ EC2 with MySQL', 'postgresql': '🖥️ EC2 with PostgreSQL', 
                    'oracle': '🖥️ EC2 with Oracle', 'mongodb': '🖥️ EC2 with MongoDB'
                }[x]
            )
        else:
            database_engine = st.sidebar.selectbox(
                "Target Database (EC2)",
                ["mysql", "postgresql", "oracle", "sqlserver", "mongodb"],
                index=0 if source_database_engine != "sqlserver" else 3,
                format_func=lambda x: {
                    'mysql': '🖥️ EC2 with MySQL', 'postgresql': '🖥️ EC2 with PostgreSQL', 'oracle': '🖥️ EC2 with Oracle',
                    'sqlserver': '🖥️ EC2 with SQL Server', 'mongodb': '🖥️ EC2 with MongoDB'
                }[x]
            )
        ec2_database_engine = database_engine
        
        # SQL Server Deployment Type
        if database_engine == "sqlserver":
            st.sidebar.markdown("**🔧 SQL Server Deployment Configuration:**")
            sql_server_deployment_type = st.sidebar.selectbox(
                "SQL Server Deployment Type",
                ["standalone", "always_on"],
                format_func=lambda x: {
                    'standalone': '🖥️ Standalone SQL Server (Single Instance)',
                    'always_on': '🔄 SQL Server Always On (3-Node Cluster)'
                }[x]
            )
            
            if sql_server_deployment_type == "always_on":
                st.sidebar.info("""
                **🔄 SQL Server Always On Cluster:**
                • 3 EC2 instances (Primary + 2 Replicas)
                • High Availability & Disaster Recovery
                • Automatic failover capability
                • ~3x cost of standalone deployment
                """)
            else:
                st.sidebar.info("""
                **🖥️ Standalone SQL Server:**
                • Single EC2 instance
                • Cost-effective deployment
                • Manual backup and recovery
                """)
    
    # Database Configuration
    database_size_gb = st.sidebar.number_input("Database Size (GB)", 
                                              min_value=100, max_value=100000, value=1000, step=100)
    
    downtime_tolerance_minutes = st.sidebar.number_input("Max Downtime (minutes)", 
                                                        min_value=1, max_value=480, value=60)
    
    performance_requirements = st.sidebar.selectbox("Performance Requirement", ["standard", "high"])
    environment = st.sidebar.selectbox("Environment", ["non-production", "production"])
    
    # Migration Method
    st.sidebar.subheader("🔄 Migration Method")
    migration_method = st.sidebar.selectbox(
        "Migration Method",
        ["backup_restore", "direct_replication"],
        format_func=lambda x: {
            'backup_restore': '📦 Backup/Restore via DataSync (File Transfer)',
            'direct_replication': '🔄 Direct Replication via DMS (Live Sync)'
        }[x]
    )
    
    # Backup Storage Configuration
    backup_storage_type = None
    backup_size_multiplier = None
    
    if migration_method == 'backup_restore':
        st.sidebar.subheader("💾 Backup Storage Configuration")
        
        if source_database_engine in ['sqlserver']:
            backup_storage_type = st.sidebar.selectbox(
                "Backup Storage Type",
                ["windows_share", "nas_drive"],
                index=0,
                format_func=lambda x: {
                    'windows_share': '🪟 Windows Share Drive (Default for SQL Server)',
                    'nas_drive': '🗄️ NAS Drive (Alternative)'
                }[x]
            )
        else:
            backup_storage_type = st.sidebar.selectbox(
                "Backup Storage Type",
                ["nas_drive", "windows_share"],
                index=0,
                format_func=lambda x: {
                    'nas_drive': '🗄️ NAS Drive',
                    'windows_share': '🪟 Windows Share Drive'
                }[x]
            )
        
        backup_size_multiplier = st.sidebar.selectbox(
            "Backup Size vs Database",
            [0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
            index=2,
            format_func=lambda x: f"{int(x*100)}% of DB size ({x:.1f}x multiplier)"
        )
    
    # Destination Storage
    st.sidebar.subheader("🗄️ Destination Storage")
    destination_storage_type = st.sidebar.selectbox(
        "AWS Destination Storage",
        ["S3", "FSx_Windows", "FSx_Lustre"],
        format_func=lambda x: {
            'S3': '☁️ Amazon S3 (Standard)',
            'FSx_Windows': '🪟 Amazon FSx for Windows File Server',
            'FSx_Lustre': '⚡ Amazon FSx for Lustre (High Performance)'
        }[x]
    )
    
    # Agent Configuration
    st.sidebar.subheader("🤖 Migration Agent Configuration")
    
    # Determine primary tool
    if migration_method == 'backup_restore':
        primary_tool = "DataSync"
        is_homogeneous = True
    else:
        is_homogeneous = source_database_engine == database_engine
        primary_tool = "DataSync" if is_homogeneous else "DMS"
    
    st.sidebar.success(f"**Primary Tool:** AWS {primary_tool}")
    
    if migration_method == 'backup_restore':
        st.sidebar.info(f"**Method:** Backup/Restore via DataSync from {backup_storage_type.replace('_', ' ').title() if backup_storage_type else 'N/A'}")
    else:
        st.sidebar.info(f"**Method:** Direct replication ({'homogeneous' if is_homogeneous else 'heterogeneous'})")
    
    number_of_agents = st.sidebar.number_input(
        "Number of Migration Agents",
        min_value=1, max_value=10, value=2, step=1,
        help=f"Number of {primary_tool} agents for parallel processing"
    )
    
    if migration_method == 'backup_restore' or is_homogeneous:
        datasync_agent_size = st.sidebar.selectbox(
            "DataSync Agent Size",
            ["small", "medium", "large", "xlarge"],
            index=1,
            format_func=lambda x: {
                'small': '📦 Small (t3.medium) - 250 Mbps/agent',
                'medium': '📦 Medium (c5.large) - 500 Mbps/agent',
                'large': '📦 Large (c5.xlarge) - 1000 Mbps/agent',
                'xlarge': '📦 XLarge (c5.2xlarge) - 2000 Mbps/agent'
            }[x]
        )
        dms_agent_size = None
    else:
        dms_agent_size = st.sidebar.selectbox(
            "DMS Instance Size",
            ["small", "medium", "large", "xlarge", "xxlarge"],
            index=1,
            format_func=lambda x: {
                'small': '🔄 Small (t3.medium) - 200 Mbps/agent',
                'medium': '🔄 Medium (c5.large) - 400 Mbps/agent',
                'large': '🔄 Large (c5.xlarge) - 800 Mbps/agent',
                'xlarge': '🔄 XLarge (c5.2xlarge) - 1500 Mbps/agent',
                'xxlarge': '🔄 XXLarge (c5.4xlarge) - 2500 Mbps/agent'
            }[x]
        )
        datasync_agent_size = None
    
    # Return configuration
    return {
        'operating_system': operating_system,
        'server_type': server_type,
        'ram_gb': ram_gb,
        'cpu_cores': cpu_cores,
        'cpu_ghz': cpu_ghz,
        'nic_type': nic_type,
        'nic_speed': nic_speed,
        'source_database_engine': source_database_engine,
        'target_platform': target_platform,
        'database_engine': database_engine,
        'ec2_database_engine': ec2_database_engine,
        'sql_server_deployment_type': sql_server_deployment_type,
        'database_size_gb': database_size_gb,
        'current_db_max_memory_gb': current_db_max_memory_gb,
        'current_db_max_cpu_cores': current_db_max_cpu_cores,
        'current_db_max_iops': current_db_max_iops,
        'current_db_max_throughput_mbps': current_db_max_throughput_mbps,
        'downtime_tolerance_minutes': downtime_tolerance_minutes,
        'performance_requirements': performance_requirements,
        'backup_storage_type': backup_storage_type,
        'backup_size_multiplier': backup_size_multiplier,
        'migration_method': migration_method,
        'destination_storage_type': destination_storage_type,
        'environment': environment,
        'number_of_agents': number_of_agents,
        'datasync_agent_size': datasync_agent_size,
        'dms_agent_size': dms_agent_size,
        'enable_ai_analysis': True
    }

def render_comprehensive_cost_analysis_tab(analysis: Dict, config: Dict):
    """Render comprehensive cost analysis with detailed tabular format"""
    st.subheader("💰 Comprehensive AWS Cost Analysis - Enterprise Tabular Format")
    
    comprehensive_costs = analysis.get('comprehensive_cost_analysis', {})
    
    if not comprehensive_costs:
        st.warning("Comprehensive cost data not available. Please run the analysis first.")
        return
    
    cost_tables = comprehensive_costs.get('cost_tables', {})
    cost_projections = comprehensive_costs.get('cost_projections', {})
    optimization_recommendations = comprehensive_costs.get('optimization_recommendations', [])
    total_analysis = comprehensive_costs.get('total_analysis', {})
    
    # Executive Summary Metrics
    st.markdown("**💸 Executive Cost Summary:**")
    summary_metrics = total_analysis.get('summary_metrics', {})
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "💰 Total Monthly",
            f"${summary_metrics.get('total_monthly_cost', 0):,.2f}",
            delta=f"Services: {summary_metrics.get('total_services', 0)}"
        )
    
    with col2:
        st.metric(
            "🔄 One-Time Costs",
            f"${summary_metrics.get('total_one_time_cost', 0):,.2f}",
            delta="Setup & Migration"
        )
    
    with col3:
        st.metric(
            "📅 Annual Cost",
            f"${summary_metrics.get('annual_cost', 0):,.2f}",
            delta="Recurring only"
        )
    
    with col4:
        st.metric(
            "📊 3-Year TCO",
            f"${summary_metrics.get('three_year_tco', 0):,.2f}",
            delta="Total investment"
        )
    
    with col5:
        pricing_source = comprehensive_costs.get('pricing_source', 'fallback')
        st.metric(
            "📡 Pricing Source",
            "Real-time" if pricing_source == 'aws_api' else "Fallback",
            delta="AWS API" if pricing_source == 'aws_api' else "Static data"
        )
    
    # Monthly Recurring Costs Table
    st.markdown("---")
    st.markdown("**📊 Monthly Recurring Costs - Detailed Breakdown**")
    
    monthly_costs = cost_tables.get('monthly_recurring_costs', [])
    if monthly_costs:
        df_monthly = pd.DataFrame(monthly_costs)
        
        # Add styling and formatting
        st.dataframe(
            df_monthly,
            use_container_width=True,
            column_config={
                "Total Monthly Cost": st.column_config.TextColumn("Total Monthly Cost", help="Monthly recurring cost for this service"),
                "Annual Cost": st.column_config.TextColumn("Annual Cost", help="Projected annual cost"),
                "Quantity": st.column_config.NumberColumn("Quantity", help="Number of instances/units"),
                "Notes": st.column_config.TextColumn("Notes", help="Additional service information")
            }
        )
        
        # Calculate and show totals
        total_monthly_from_table = sum(float(item['Total Monthly Cost'].replace('$', '').replace(',', '')) for item in monthly_costs)
        st.success(f"**Total Monthly Recurring Cost: ${total_monthly_from_table:,.2f}**")
    else:
        st.info("No monthly recurring costs identified.")
    
    # One-Time Costs Table
    st.markdown("---")
    st.markdown("**🔄 One-Time Costs - Setup and Migration**")
    
    one_time_costs = cost_tables.get('one_time_costs', [])
    if one_time_costs:
        df_one_time = pd.DataFrame(one_time_costs)
        st.dataframe(
            df_one_time,
            use_container_width=True,
            column_config={
                "One-Time Cost": st.column_config.TextColumn("One-Time Cost", help="One-time setup or migration cost"),
                "Description": st.column_config.TextColumn("Description", help="What this cost covers")
            }
        )
        
        total_one_time_from_table = sum(float(item['One-Time Cost'].replace('$', '').replace(',', '')) for item in one_time_costs)
        st.warning(f"**Total One-Time Cost: ${total_one_time_from_table:,.2f}**")
    else:
        st.info("No one-time costs identified.")
    
    # Service Specifications Table
    st.markdown("---")
    st.markdown("**🔧 Service Specifications and Configurations**")
    
    service_specs = cost_tables.get('service_specifications', [])
    if service_specs:
        df_specs = pd.DataFrame(service_specs)
        st.dataframe(
            df_specs,
            use_container_width=True,
            column_config={
                "Specifications": st.column_config.TextColumn("Specifications", help="Technical specifications for this service"),
                "Purpose": st.column_config.TextColumn("Purpose", help="Business purpose of this service"),
                "Monthly Cost": st.column_config.TextColumn("Monthly Cost", help="Monthly cost for this service")
            }
        )
    else:
        st.info("Service specifications not available.")
    
    # Cost Breakdown by Category
    st.markdown("---")
    st.markdown("**📈 Cost Breakdown by Service Category**")
    
    category_breakdown = cost_tables.get('cost_breakdown_by_category', [])
    if category_breakdown:
        df_categories = pd.DataFrame(category_breakdown)
        st.dataframe(
            df_categories,
            use_container_width=True,
            column_config={
                "Percentage of Total": st.column_config.TextColumn("% of Total", help="Percentage of total monthly cost"),
                "Service Count": st.column_config.NumberColumn("Services", help="Number of services in this category")
            }
        )
        
        # Create pie chart for cost distribution
        col1, col2 = st.columns(2)
        
        with col1:
            category_costs = []
            category_names = []
            for item in category_breakdown:
                cost_value = float(item['Monthly Total'].replace('$', '').replace(',', ''))
                if cost_value > 0:
                    category_costs.append(cost_value)
                    category_names.append(item['Category'])
            
            if category_costs:
                fig_pie = px.pie(
                    values=category_costs,
                    names=category_names,
                    title="Monthly Cost Distribution by Category"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Create bar chart for service count
            service_counts = [item['Service Count'] for item in category_breakdown]
            fig_bar = px.bar(
                x=category_names,
                y=service_counts,
                title="Service Count by Category",
                labels={'x': 'Category', 'y': 'Number of Services'}
            )
            st.plotly_chart(fig_bar, use_container_width=True)
    
    # Variable Costs Table
    if cost_tables.get('variable_costs'):
        st.markdown("---")
        st.markdown("**⚡ Variable and Usage-Based Costs**")
        
        variable_costs = cost_tables.get('variable_costs', [])
        df_variable = pd.DataFrame(variable_costs)
        st.dataframe(
            df_variable,
            use_container_width=True,
            column_config={
                "Unit Rate": st.column_config.TextColumn("Unit Rate", help="Cost per unit of usage"),
                "Estimated Monthly Usage": st.column_config.TextColumn("Est. Monthly Usage", help="Estimated monthly usage"),
                "Estimated Monthly Cost": st.column_config.TextColumn("Est. Monthly Cost", help="Estimated monthly cost based on usage")
            }
        )
    
    # Cost Projections
    st.markdown("---")
    st.markdown("**📊 Cost Projections and Financial Analysis**")
    
    if cost_projections:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**💰 Multi-Year Cost Summary**")
            cost_summary = cost_projections.get('cost_summary', {})
            
            projection_data = {
                'Metric': [
                    'Total Monthly Recurring',
                    'Total One-Time Setup',
                    'Year 1 Total',
                    'Year 2 Total',
                    'Year 3 Total',
                    '3-Year Total'
                ],
                'Amount': [
                    f"${cost_summary.get('total_monthly', 0):,.2f}",
                    f"${cost_summary.get('total_one_time', 0):,.2f}",
                    f"${cost_projections.get('cost_trends', {}).get('year_1', 0):,.2f}",
                    f"${cost_projections.get('cost_trends', {}).get('year_2', 0):,.2f}",
                    f"${cost_projections.get('cost_trends', {}).get('year_3', 0):,.2f}",
                    f"${cost_summary.get('three_year_total', 0):,.2f}"
                ]
            }
            
            df_projections = pd.DataFrame(projection_data)
            st.dataframe(df_projections, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("**📈 Monthly Cost Progression**")
            monthly_projections = cost_projections.get('monthly_projections', [])[:6]  # First 6 months
            
            if monthly_projections:
                df_monthly_proj = pd.DataFrame(monthly_projections)
                st.dataframe(
                    df_monthly_proj,
                    use_container_width=True,
                    column_config={
                        "Cumulative Total": st.column_config.TextColumn("Cumulative Total", help="Running total of all costs")
                    }
                )
    
    # Cost Optimization Recommendations
    st.markdown("---")
    st.markdown("**💡 Cost Optimization Recommendations**")
    
    if optimization_recommendations:
        for i, rec in enumerate(optimization_recommendations, 1):
            priority = rec.get('priority', 'Medium')
            if priority == 'High':
                priority_color = "🔴"
            elif priority == 'Medium':
                priority_color = "🟡"
            else:
                priority_color = "🟢"
            
            with st.expander(f"{priority_color} **{priority} Priority**: {rec.get('recommendation', 'Optimization')}", expanded=(i <= 2)):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**💰 Financial Impact**")
                    st.write(f"**Potential Savings:** {rec.get('potential_savings', 'TBD')}")
                    st.write(f"**Est. Monthly Savings:** {rec.get('estimated_monthly_savings', 'TBD')}")
                    st.write(f"**Category:** {rec.get('category', 'General')}")
                
                with col2:
                    st.markdown("**⚙️ Implementation**")
                    st.write(f"**Effort Required:** {rec.get('implementation_effort', 'Medium')}")
                    st.write(f"**Priority Level:** {priority}")
                    st.write(f"**Timeline:** {'Immediate' if priority == 'High' else '1-3 months'}")
                
                with col3:
                    st.markdown("**📋 Details**")
                    st.write(rec.get('details', 'No additional details available'))
    else:
        st.info("💡 Current configuration appears cost-optimized based on requirements.")
    
    # Export Options
    st.markdown("---")
    st.markdown("**📥 Export Cost Analysis**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export Detailed Cost Tables", use_container_width=True):
            # Create comprehensive export data
            export_data = {
                'cost_analysis_summary': total_analysis,
                'monthly_recurring_costs': monthly_costs,
                'one_time_costs': one_time_costs,
                'service_specifications': service_specs,
                'cost_projections': cost_projections,
                'optimization_recommendations': optimization_recommendations,
                'analysis_timestamp': datetime.now().isoformat(),
                'configuration': config
            }
            
            st.download_button(
                label="📥 Download Cost Analysis (JSON)",
                data=json.dumps(export_data, indent=2),
                file_name=f"aws_cost_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col2:
        if st.button("📈 Export Cost Summary", use_container_width=True):
            # Create CSV export for cost summary
            summary_data = []
            for item in monthly_costs:
                summary_data.append({
                    'Service': item['Service Name'],
                    'Category': item['Category'],
                    'Monthly_Cost': item['Total Monthly Cost'].replace(', '').replace(',', ''),
                    'Annual_Cost': item['Annual Cost'].replace(', '').replace(',', ''),
                    'Quantity': item['Quantity']
                })
            
            if summary_data:
                df_export = pd.DataFrame(summary_data)
                csv_data = df_export.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download CSV Summary",
                    data=csv_data,
                    file_name=f"aws_cost_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    with col3:
        pricing_source = comprehensive_costs.get('pricing_source', 'fallback')
        if pricing_source == 'aws_api':
            st.success("✅ Using Real-time AWS Pricing")
        else:
            st.warning("⚠️ Using Fallback Pricing Data")

def render_comprehensive_aws_sizing_tab(analysis: Dict, config: Dict):
    """Render comprehensive AWS sizing with all services details"""
    st.subheader("🎯 Comprehensive AWS Service Sizing & Architecture")
    
    aws_sizing = analysis.get('comprehensive_aws_sizing', {})
    
    if not aws_sizing:
        st.warning("AWS sizing data not available. Please run the analysis first.")
        return
    
    # Service Overview Dashboard
    st.markdown("**☁️ AWS Services Overview:**")
    
    service_summary = aws_sizing.get('service_summary', {})
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_services = service_summary.get('total_services', 0)
        st.metric(
            "🔧 Total Services",
            total_services,
            delta=f"Categories: {len(aws_sizing.get('compute_services', {})) + len(aws_sizing.get('storage_services', {})) + len(aws_sizing.get('migration_services', {}))}"
        )
    
    with col2:
        total_monthly = aws_sizing.get('total_monthly_cost', 0)
        st.metric(
            "💰 Total Monthly Cost",
            f"${total_monthly:,.2f}",
            delta="All services included"
        )
    
    with col3:
        pricing_source = aws_sizing.get('pricing_data_source', 'fallback')
        st.metric(
            "📡 Pricing Data",
            "Real-time" if pricing_source == 'aws_api' else "Fallback",
            delta=aws_sizing.get('region', 'us-west-2')
        )
    
    with col4:
        migration_method = config.get('migration_method', 'direct_replication')
        st.metric(
            "🔄 Migration Method",
            migration_method.replace('_', ' ').title(),
            delta=f"Tool: {'DataSync' if migration_method == 'backup_restore' else 'DMS/DataSync'}"
        )
    
    with col5:
        target_platform = config.get('target_platform', 'rds')
        sql_deployment = config.get('sql_server_deployment_type', 'standalone')
        if target_platform == 'ec2' and sql_deployment == 'always_on':
            display_text = "Always On"
            delta_text = "3-Node Cluster"
        else:
            display_text = target_platform.upper()
            delta_text = "Standard deployment"
        
        st.metric(
            "🎯 Target Platform",
            display_text,
            delta=delta_text
        )
    
    # Compute Services Section
    st.markdown("---")
    st.markdown("**🖥️ Compute Services (EC2/RDS)**")
    
    compute_services = aws_sizing.get('compute_services', {})
    if compute_services:
        for service_key, service_data in compute_services.items():
            service_name = service_data.get('service_name', service_key)
            
            with st.expander(f"🖥️ {service_name}", expanded=True):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**📋 Service Details**")
                    st.write(f"**Service Type:** {service_data.get('instance_type', 'N/A')}")
                    st.write(f"**Database Engine:** {service_data.get('database_engine', 'N/A')}")
                    st.write(f"**Quantity:** {service_data.get('quantity', 1)} instance(s)")
                    
                    # Special handling for SQL Server Always On
                    if service_data.get('deployment_type') == 'always_on':
                        st.write(f"**Deployment:** SQL Server Always On (3-Node)")
                        st.write("**Cluster Features:**")
                        cluster_req = service_data.get('cluster_requirements', {})
                        for req_key, req_value in cluster_req.items():
                            if isinstance(req_value, str):
                                st.write(f"  • {req_key.replace('_', ' ').title()}: {req_value}")
                
                with col2:
                    st.markdown("**⚙️ Technical Specifications**")
                    specs = service_data.get('specifications', {})
                    st.write(f"**vCPU:** {specs.get('vcpu', 'N/A')}")
                    st.write(f"**Memory:** {specs.get('memory_gb', 'N/A')} GB")
                    st.write(f"**Network Performance:** {specs.get('network_performance', 'N/A')}")
                    
                    if 'ebs_optimized' in specs:
                        st.write(f"**EBS Optimized:** {'✅' if specs['ebs_optimized'] else '❌'}")
                    if 'enhanced_networking' in specs:
                        st.write(f"**Enhanced Networking:** {'✅' if specs['enhanced_networking'] else '❌'}")
                    if 'deployment_option' in specs:
                        st.write(f"**Deployment:** {specs['deployment_option']}")
                
                with col3:
                    st.markdown("**💰 Cost Information**")
                    pricing = service_data.get('pricing', {})
                    st.write(f"**Cost per Hour:** ${pricing.get('cost_per_hour', 0):.4f}")
                    st.write(f"**Monthly Cost:** ${pricing.get('monthly_cost', 0):,.2f}")
                    st.write(f"**Annual Cost:** ${pricing.get('annual_cost', 0):,.2f}")
                    
                    if service_data.get('quantity', 1) > 1:
                        unit_cost = pricing.get('monthly_cost', 0) / service_data.get('quantity', 1)
                        st.write(f"**Cost per Instance:** ${unit_cost:,.2f}/month")
    else:
        st.info("No compute services configured.")
    
    # Storage Services Section
    st.markdown("---")
    st.markdown("**🗄️ Storage Services (EBS/S3/FSx)**")
    
    storage_services = aws_sizing.get('storage_services', {})
    if storage_services:
        for service_key, service_data in storage_services.items():
            service_name = service_data.get('service_name', service_key)
            
            with st.expander(f"🗄️ {service_name}", expanded=True):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**📋 Storage Details**")
                    st.write(f"**Storage Type:** {service_data.get('storage_type', 'N/A')}")
                    specs = service_data.get('specifications', {})
                    
                    if 'size_gb' in specs:
                        st.write(f"**Storage Size:** {specs['size_gb']:,.0f} GB")
                    elif 'total_size_gb' in specs:
                        st.write(f"**Total Storage:** {specs['total_size_gb']:,.0f} GB")
                        st.write(f"**Per Instance:** {specs.get('size_gb_per_instance', 0):,.0f} GB")
                    
                    if 'purpose' in specs:
                        st.write(f"**Purpose:** {specs['purpose']}")
                
                with col2:
                    st.markdown("**⚙️ Performance Specifications**")
                    
                    if 'baseline_iops' in specs:
                        st.write(f"**Baseline IOPS:** {specs['baseline_iops']:,}")
                    if 'baseline_throughput_mbps' in specs:
                        st.write(f"**Baseline Throughput:** {specs['baseline_throughput_mbps']} MB/s")
                    if 'throughput_capacity_mbps' in specs:
                        st.write(f"**Throughput Capacity:** {specs['throughput_capacity_mbps']} MB/s")
                    if 'durability' in specs:
                        st.write(f"**Durability:** {specs['durability']}")
                    if 'availability' in specs:
                        st.write(f"**Availability:** {specs['availability']}")
                    
                    # Additional features
                    if specs.get('encryption'):
                        st.write("**Encryption:** ✅ Enabled")
                    if specs.get('lifecycle_policies'):
                        st.write("**Lifecycle Policies:** ✅ Enabled")
                
                with col3:
                    st.markdown("**💰 Storage Costs**")
                    pricing = service_data.get('pricing', {})
                    st.write(f"**Cost per GB/Month:** ${pricing.get('cost_per_gb_month', 0):.4f}")
                    st.write(f"**Monthly Cost:** ${pricing.get('monthly_cost', 0):,.2f}")
                    st.write(f"**Annual Cost:** ${pricing.get('annual_cost', 0):,.2f}")
                    
                    # Show backup storage details if applicable
                    if 'backup' in service_name.lower():
                        backup_storage_type = specs.get('backup_storage_type', 'N/A')
                        st.write(f"**Backup Source:** {backup_storage_type.replace('_', ' ').title()}")
    else:
        st.info("No storage services configured.")
    
    # Migration Services Section
    st.markdown("---")
    st.markdown("**🔄 Migration Services (DataSync/DMS)**")
    
    migration_services = aws_sizing.get('migration_services', {})
    if migration_services:
        for service_key, service_data in migration_services.items():
            service_name = service_data.get('service_name', service_key)
            
            with st.expander(f"🔄 {service_name}", expanded=True):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**📋 Migration Service Details**")
                    
                    if 'agent_type' in service_data:
                        st.write(f"**Agent Type:** {service_data['agent_type']}")
                        st.write(f"**Instance Type:** {service_data.get('instance_type', 'N/A')}")
                        st.write(f"**Quantity:** {service_data.get('quantity', 1)} agent(s)")
                    
                    if 'transfer_type' in service_data:
                        st.write(f"**Transfer Type:** {service_data['transfer_type']}")
                        st.write(f"**Migration Type:** {service_data.get('migration_type', 'N/A')}")
                    
                    if 'conversion_type' in service_data:
                        st.write(f"**Conversion:** {service_data['conversion_type']}")
                
                with col2:
                    st.markdown("**⚙️ Service Specifications**")
                    specs = service_data.get('specifications', {})
                    
                    if 'max_throughput_mbps' in specs:
                        st.write(f"**Max Throughput:** {specs['max_throughput_mbps']} Mbps")
                    if 'concurrent_tasks' in specs:
                        st.write(f"**Concurrent Tasks:** {specs['concurrent_tasks']}")
                    if 'data_size_gb' in specs:
                        st.write(f"**Data Size:** {specs['data_size_gb']:,.0f} GB")
                    if 'source_type' in specs:
                        st.write(f"**Source:** {specs['source_type'].replace('_', ' ').title()}")
                    if 'destination_type' in specs:
                        st.write(f"**Destination:** {specs['destination_type']}")
                    
                    # Security features
                    if specs.get('encryption_in_transit'):
                        st.write(f"**Encryption:** {specs['encryption_in_transit']}")
                    if specs.get('compression'):
                        st.write("**Compression:** ✅ Enabled")
                
                with col3:
                    st.markdown("**💰 Migration Costs**")
                    pricing = service_data.get('pricing', {})
                    
                    if pricing.get('cost_per_hour', 0) > 0:
                        st.write(f"**Cost per Hour:** ${pricing['cost_per_hour']:.4f}")
                    if pricing.get('cost_per_gb', 0) > 0:
                        st.write(f"**Cost per GB:** ${pricing['cost_per_gb']:.4f}")
                    if pricing.get('one_time_cost', 0) > 0:
                        st.write(f"**One-time Cost:** ${pricing['one_time_cost']:,.2f}")
                    
                    st.write(f"**Monthly Cost:** ${pricing.get('monthly_cost', 0):,.2f}")
                    st.write(f"**Annual Cost:** ${pricing.get('annual_cost', 0):,.2f}")
                    
                    if pricing.get('note'):
                        st.info(pricing['note'])
    else:
        st.info("No migration services configured.")
    
    # Network Services Section
    st.markdown("---")
    st.markdown("**🌐 Network Services (Direct Connect/VPC)**")
    
    network_services = aws_sizing.get('network_services', {})
    if network_services:
        for service_key, service_data in network_services.items():
            service_name = service_data.get('service_name', service_key)
            
            with st.expander(f"🌐 {service_name}", expanded=True):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**📋 Network Service Details**")
                    st.write(f"**Connection Type:** {service_data.get('connection_type', 'N/A')}")
                    
                    specs = service_data.get('specifications', {})
                    if 'bandwidth' in specs:
                        st.write(f"**Bandwidth:** {specs['bandwidth']}")
                    if 'location' in specs:
                        st.write(f"**Location:** {specs['location']}")
                    if 'redundancy' in specs:
                        st.write(f"**Redundancy:** {specs['redundancy']}")
                
                with col2:
                    st.markdown("**⚙️ Technical Specifications**")
                    
                    if 'port_speed' in specs:
                        st.write(f"**Port Speed:** {specs['port_speed']}")
                    if 'bgp_sessions' in specs:
                        st.write(f"**BGP Sessions:** {specs['bgp_sessions']}")
                    if 'vlan_support' in specs:
                        st.write(f"**VLAN Support:** {specs['vlan_support']}")
                    if 'endpoints_required' in specs:
                        endpoints = specs['endpoints_required']
                        st.write(f"**Endpoints:** {', '.join(endpoints)}")
                    if 'availability_zones' in specs:
                        st.write(f"**Availability Zones:** {specs['availability_zones']}")
                
                with col3:
                    st.markdown("**💰 Network Costs**")
                    pricing = service_data.get('pricing', {})
                    
                    if 'port_hours' in pricing:
                        st.write(f"**Port Cost/Hour:** ${pricing['port_hours']:.4f}")
                    if 'monthly_port_cost' in pricing:
                        st.write(f"**Monthly Port Cost:** ${pricing['monthly_port_cost']:,.2f}")
                    if 'data_transfer_out_per_gb' in pricing:
                        st.write(f"**Data Transfer/GB:** ${pricing['data_transfer_out_per_gb']:.4f}")
                    if 'estimated_monthly_transfer_cost' in pricing:
                        st.write(f"**Est. Transfer Cost:** ${pricing['estimated_monthly_transfer_cost']:,.2f}")
                    
                    st.write(f"**Annual Cost:** ${pricing.get('annual_cost', 0):,.2f}")
    else:
        st.info("No network services configured.")
    
    # Monitoring Services Section
    st.markdown("---")
    st.markdown("**📊 Monitoring Services (CloudWatch)**")
    
    monitoring_services = aws_sizing.get('monitoring_services', {})
    if monitoring_services:
        for service_key, service_data in monitoring_services.items():
            service_name = service_data.get('service_name', service_key)
            
            with st.expander(f"📊 {service_name}", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**📋 Monitoring Details**")
                    specs = service_data.get('specifications', {})
                    
                    if 'estimated_metrics_count' in specs:
                        st.write(f"**Metrics Count:** {specs['estimated_metrics_count']}")
                    if 'estimated_ingestion_gb_month' in specs:
                        st.write(f"**Log Ingestion:** {specs['estimated_ingestion_gb_month']} GB/month")
                    if 'dashboard_count' in specs:
                        st.write(f"**Dashboards:** {specs['dashboard_count']}")
                
                with col2:
                    st.markdown("**⚙️ Configuration**")
                    
                    if 'resolution' in specs:
                        st.write(f"**Resolution:** {specs['resolution']}")
                    if 'retention_period' in specs:
                        st.write(f"**Retention:** {specs['retention_period']}")
                    if 'alarms_included' in specs:
                        st.write(f"**Alarms:** {specs['alarms_included']}")
                    if 'log_groups' in specs:
                        log_groups = specs['log_groups']
                        st.write(f"**Log Groups:** {', '.join(log_groups)}")
                
                with col3:
                    st.markdown("**💰 Monitoring Costs**")
                    pricing = service_data.get('pricing', {})
                    st.write(f"**Monthly Cost:** ${pricing.get('monthly_cost', 0):,.2f}")
                    st.write(f"**Annual Cost:** ${pricing.get('annual_cost', 0):,.2f}")
    else:
        st.info("No monitoring services configured.")
    
    # Backup Services Section
    st.markdown("---")
    st.markdown("**🔒 Backup & DR Services**")
    
    backup_services = aws_sizing.get('backup_services', {})
    if backup_services:
        for service_key, service_data in backup_services.items():
            service_name = service_data.get('service_name', service_key)
            
            with st.expander(f"🔒 {service_name}", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**📋 Backup Details**")
                    specs = service_data.get('specifications', {})
                    
                    if 'backup_frequency' in specs:
                        st.write(f"**Frequency:** {specs['backup_frequency']}")
                    if 'retention_period_days' in specs:
                        st.write(f"**Retention:** {specs['retention_period_days']} days")
                    if 'estimated_backup_size_gb' in specs:
                        st.write(f"**Backup Size:** {specs['estimated_backup_size_gb']:,.0f} GB")
                
                with col2:
                    st.markdown("**⚙️ Features**")
                    
                    if specs.get('compression'):
                        st.write("**Compression:** ✅ Enabled")
                    if specs.get('encryption'):
                        st.write(f"**Encryption:** {specs['encryption']}")
                    if specs.get('cross_region_copy'):
                        st.write("**Cross-Region Copy:** ✅ Enabled")
                    if specs.get('automated_backups'):
                        st.write("**Automated Backups:** ✅ Enabled")
                
                with col3:
                    st.markdown("**💰 Backup Costs**")
                    pricing = service_data.get('pricing', {})
                    
                    if pricing.get('included_in_rds'):
                        st.info("✅ Included in RDS cost")
                    else:
                        st.write(f"**Monthly Cost:** ${pricing.get('monthly_cost', 0):,.2f}")
                        st.write(f"**Annual Cost:** ${pricing.get('annual_cost', 0):,.2f}")
                    
                    if pricing.get('note'):
                        st.info(pricing['note'])
    else:
        st.info("No backup services configured.")
    
    # Service Architecture Summary
    st.markdown("---")
    st.markdown("**🏗️ Architecture Summary**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Service Distribution**")
        service_counts = {
            'Compute': len(compute_services),
            'Storage': len(storage_services),
            'Migration': len(migration_services),
            'Network': len(network_services),
            'Monitoring': len(monitoring_services),
            'Backup': len(backup_services)
        }
        
        # Filter out zero counts for cleaner visualization
        filtered_counts = {k: v for k, v in service_counts.items() if v > 0}
        
        if filtered_counts:
            fig_services = px.bar(
                x=list(filtered_counts.keys()),
                y=list(filtered_counts.values()),
                title="Services by Category",
                labels={'x': 'Service Category', 'y': 'Number of Services'}
            )
            st.plotly_chart(fig_services, use_container_width=True)
    
    with col2:
        st.markdown("**💰 Cost Distribution**")
        cost_by_category = {}
        
        for category, services in [
            ('Compute', compute_services),
            ('Storage', storage_services),
            ('Migration', migration_services),
            ('Network', network_services),
            ('Monitoring', monitoring_services),
            ('Backup', backup_services)
        ]:
            total_cost = 0
            for service_data in services.values():
                if isinstance(service_data, dict):
                    pricing = service_data.get('pricing', {})
                    total_cost += pricing.get('monthly_cost', 0)
            
            if total_cost > 0:
                cost_by_category[category] = total_cost
        
        if cost_by_category:
            fig_costs = px.pie(
                values=list(cost_by_category.values()),
                names=list(cost_by_category.keys()),
                title="Monthly Cost Distribution"
            )
            st.plotly_chart(fig_costs, use_container_width=True)
    
    # Export Options
    st.markdown("---")
    st.markdown("**📥 Export AWS Sizing Analysis**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Export Complete Sizing Analysis", use_container_width=True):
            export_data = {
                'aws_sizing_summary': service_summary,
                'compute_services': compute_services,
                'storage_services': storage_services,
                'migration_services': migration_services,
                'network_services': network_services,
                'monitoring_services': monitoring_services,
                'backup_services': backup_services,
                'total_monthly_cost': total_monthly,
                'pricing_data_source': pricing_source,
                'region': aws_sizing.get('region', 'us-west-2'),
                'analysis_timestamp': datetime.now().isoformat(),
                'configuration': config
            }
            
            st.download_button(
                label="📥 Download AWS Sizing (JSON)",
                data=json.dumps(export_data, indent=2),
                file_name=f"aws_sizing_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col2:
        if st.button("📋 Export Service Specifications", use_container_width=True):
            # Create detailed service specifications export
            specs_data = []
            
            for category, services in [
                ('Compute', compute_services),
                ('Storage', storage_services),
                ('Migration', migration_services),
                ('Network', network_services),
                ('Monitoring', monitoring_services),
                ('Backup', backup_services)
            ]:
                for service_key, service_data in services.items():
                    if isinstance(service_data, dict):
                        specs = service_data.get('specifications', {})
                        pricing = service_data.get('pricing', {})
                        
                        specs_data.append({
                            'Category': category,
                            'Service_Name': service_data.get('service_name', service_key),
                            'Service_Type': service_data.get('instance_type', service_data.get('storage_type', 'N/A')),
                            'Quantity': service_data.get('quantity', 1),
                            'Monthly_Cost': pricing.get('monthly_cost', 0),
                            'Specifications': json.dumps(specs, indent=2)
                        })
            
            if specs_data:
                df_specs = pd.DataFrame(specs_data)
                csv_data = df_specs.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Service Specs (CSV)",
                    data=csv_data,
                    file_name=f"aws_service_specs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

# Enhanced Migration Analyzer Class
class EnhancedMigrationAnalyzer:
    """Enhanced migration analyzer with comprehensive AWS integration"""
    
    def __init__(self):
        self.ai_manager = AnthropicAIManager()
        self.aws_api = EnhancedAWSAPIManager()
        self.service_sizer = ComprehensiveAWSServiceSizer(self.aws_api)
        self.cost_analyzer = ComprehensiveCostAnalyzer(self.service_sizer)
    
    async def comprehensive_migration_analysis(self, config: Dict) -> Dict:
        """Run comprehensive migration analysis with all enhancements"""
        
        try:
            # Generate comprehensive AWS sizing
            aws_sizing = await self.service_sizer.generate_comprehensive_aws_sizing(config, {})
            
            # Generate detailed cost analysis
            cost_analysis = await self.cost_analyzer.generate_detailed_cost_analysis(config, {})
            
            # Basic migration metrics
            migration_metrics = self._calculate_basic_migration_metrics(config, aws_sizing)
            
            return {
                'comprehensive_aws_sizing': aws_sizing,
                'comprehensive_cost_analysis': cost_analysis,
                'migration_metrics': migration_metrics,
                'analysis_timestamp': datetime.now(),
                'configuration_summary': self._generate_configuration_summary(config),
                'api_status': {
                    'anthropic_connected': self.ai_manager.connected,
                    'aws_api_connected': self.aws_api.connected,
                    'pricing_source': aws_sizing.get('pricing_data_source', 'fallback')
                }
            }
            
        except Exception as e:
            logger.error(f"Comprehensive migration analysis failed: {e}")
            raise e
    
    def _calculate_basic_migration_metrics(self, config: Dict, aws_sizing: Dict) -> Dict:
        """Calculate basic migration metrics"""
        
        database_size_gb = config.get('database_size_gb', 1000)
        migration_method = config.get('migration_method', 'direct_replication')
        
        # Estimate migration time based on method
        if migration_method == 'backup_restore':
            backup_size_multiplier = config.get('backup_size_multiplier', 0.7)
            data_size_gb = database_size_gb * backup_size_multiplier
            estimated_throughput_mbps = 500  # Conservative estimate
        else:
            data_size_gb = database_size_gb
            estimated_throughput_mbps = 300  # Conservative estimate for DMS
        
        estimated_time_hours = (data_size_gb * 8 * 1000) / (estimated_throughput_mbps * 3600)
        
        return {
            'database_size_gb': database_size_gb,
            'data_to_migrate_gb': data_size_gb,
            'migration_method': migration_method,
            'estimated_throughput_mbps': estimated_throughput_mbps,
            'estimated_migration_time_hours': estimated_time_hours,
            'estimated_migration_time_days': estimated_time_hours / 24,
            'migration_window_acceptable': estimated_time_hours <= (config.get('downtime_tolerance_minutes', 60) / 60),
            'complexity_score': self._calculate_complexity_score(config),
            'readiness_score': self._calculate_readiness_score(config)
        }
    
    def _calculate_complexity_score(self, config: Dict) -> float:
        """Calculate migration complexity score (1-10)"""
        score = 5.0  # Base score
        
        # Database engine heterogeneity
        if config.get('source_database_engine') != config.get('database_engine'):
            score += 2.0
        
        # Database size
        size_gb = config.get('database_size_gb', 1000)
        if size_gb > 10000:
            score += 1.5
        elif size_gb > 5000:
            score += 1.0
        elif size_gb > 1000:
            score += 0.5
        
        # Environment complexity
        if config.get('environment') == 'production':
            score += 1.0
        
        # SQL Server Always On
        if config.get('sql_server_deployment_type') == 'always_on':
            score += 1.5
        
        # Migration method
        if config.get('migration_method') == 'backup_restore':
            score += 0.5
        
        return min(10.0, score)
    
    def _calculate_readiness_score(self, config: Dict) -> float:
        """Calculate migration readiness score (0-100)"""
        score = 80.0  # Base score
        
        # Performance requirements
        if config.get('performance_requirements') == 'high':
            score -= 10
        
        # Downtime tolerance
        downtime_minutes = config.get('downtime_tolerance_minutes', 60)
        if downtime_minutes < 30:
            score -= 20
        elif downtime_minutes < 60:
            score -= 10
        
        # Database size vs resources
        memory_gb = config.get('current_db_max_memory_gb', 0)
        if memory_gb > 0 and memory_gb < 8:
            score -= 10
        
        return max(0.0, min(100.0, score))
    
    def _generate_configuration_summary(self, config: Dict) -> Dict:
        """Generate configuration summary"""
        return {
            'migration_type': f"{config.get('migration_method', 'direct_replication').replace('_', ' ').title()}",
            'source_platform': f"{config.get('source_database_engine', 'Unknown').upper()} on {config.get('operating_system', 'Unknown').replace('_', ' ').title()}",
            'target_platform': f"AWS {config.get('target_platform', 'RDS').upper()}",
            'target_database': config.get('database_engine', 'Unknown').upper(),
            'environment': config.get('environment', 'Unknown').title(),
            'database_size': f"{config.get('database_size_gb', 0):,} GB",
            'migration_agents': config.get('number_of_agents', 1),
            'destination_storage': config.get('destination_storage_type', 'S3'),
            'special_configurations': self._get_special_configurations(config)
        }
    
    def _get_special_configurations(self, config: Dict) -> List[str]:
        """Get list of special configurations"""
        special_configs = []
        
        if config.get('sql_server_deployment_type') == 'always_on':
            special_configs.append("SQL Server Always On (3-Node Cluster)")
        
        if config.get('migration_method') == 'backup_restore':
            backup_storage = config.get('backup_storage_type', 'nas_drive')
            special_configs.append(f"Backup/Restore via {backup_storage.replace('_', ' ').title()}")
        
        if config.get('destination_storage_type') in ['FSx_Windows', 'FSx_Lustre']:
            special_configs.append(f"High-Performance Storage: {config.get('destination_storage_type')}")
        
        if config.get('number_of_agents', 1) > 3:
            special_configs.append(f"Multi-Agent Setup: {config.get('number_of_agents')} agents")
        
        return special_configs

async def main():
    """Main Streamlit application"""
    
    # Render enhanced header
    render_enhanced_header()
    
    # Enhanced sidebar controls
    config = render_enhanced_sidebar_controls()
    
    # Run analysis when configuration is set
    if st.button("🚀 Run Comprehensive Migration Analysis", type="primary", use_container_width=True):
        
        with st.spinner("🤖 Running comprehensive migration analysis with real-time AWS pricing..."):
            try:
                # Initialize analyzer
                analyzer = EnhancedMigrationAnalyzer()
                
                # Show progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("🔍 Initializing AWS API connections...")
                progress_bar.progress(20)
                
                status_text.text("💰 Fetching real-time AWS pricing...")
                progress_bar.progress(40)
                
                status_text.text("🎯 Calculating comprehensive AWS sizing...")
                progress_bar.progress(60)
                
                status_text.text("📊 Generating detailed cost analysis...")
                progress_bar.progress(80)
                
                # Run comprehensive analysis
                analysis = await analyzer.comprehensive_migration_analysis(config)
                
                status_text.text("✅ Analysis complete!")
                progress_bar.progress(100)
                
                # Store analysis in session state
                st.session_state['analysis'] = analysis
                st.session_state['config'] = config
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                st.success("✅ Comprehensive Migration Analysis Complete!")
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                logger.error(f"Analysis error: {e}")
                return
    
    # Display results if analysis is available
    if 'analysis' in st.session_state and 'config' in st.session_state:
        analysis = st.session_state['analysis']
        config = st.session_state['config']
        
        # Create tabs for different analysis views
        tab1, tab2, tab3 = st.tabs([
            "💰 Complete Cost Analysis", 
            "🎯 AWS Service Sizing",
            "📊 Migration Overview"
        ])
        
        with tab1:
            render_comprehensive_cost_analysis_tab(analysis, config)
        
        with tab2:
            render_comprehensive_aws_sizing_tab(analysis, config)
        
        with tab3:
            # Migration Overview Tab
            st.subheader("📊 Migration Analysis Overview")
            
            migration_metrics = analysis.get('migration_metrics', {})
            config_summary = analysis.get('configuration_summary', {})
            api_status = analysis.get('api_status', {})
            
            # Overview metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric(
                    "🎯 Migration Type",
                    config_summary.get('migration_type', 'Unknown'),
                    delta=f"Method: {migration_metrics.get('migration_method', 'Unknown')}"
                )
            
            with col2:
                st.metric(
                    "📊 Database Size",
                    config_summary.get('database_size', '0 GB'),
                    delta=f"To migrate: {migration_metrics.get('data_to_migrate_gb', 0):,.0f} GB"
                )
            
            with col3:
                st.metric(
                    "⏱️ Est. Time",
                    f"{migration_metrics.get('estimated_migration_time_hours', 0):.1f} hours",
                    delta=f"{migration_metrics.get('estimated_migration_time_days', 0):.1f} days"
                )
            
            with col4:
                complexity_score = migration_metrics.get('complexity_score', 5)
                st.metric(
                    "🔍 Complexity",
                    f"{complexity_score:.1f}/10",
                    delta="Migration complexity"
                )
            
            with col5:
                readiness_score = migration_metrics.get('readiness_score', 80)
                st.metric(
                    "✅ Readiness",
                    f"{readiness_score:.0f}/100",
                    delta="Migration readiness"
                )
            
            # Configuration Summary
            st.markdown("---")
            st.markdown("**🔧 Migration Configuration Summary:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📋 Source Configuration**")
                st.write(f"**Source Platform:** {config_summary.get('source_platform', 'Unknown')}")
                st.write(f"**Current Environment:** {config_summary.get('environment', 'Unknown')}")
                st.write(f"**Database Size:** {config_summary.get('database_size', 'Unknown')}")
                st.write(f"**Migration Agents:** {config_summary.get('migration_agents', 1)}")
            
            with col2:
                st.markdown("**🎯 Target Configuration**")
                st.write(f"**Target Platform:** {config_summary.get('target_platform', 'Unknown')}")
                st.write(f"**Target Database:** {config_summary.get('target_database', 'Unknown')}")
                st.write(f"**Destination Storage:** {config_summary.get('destination_storage', 'Unknown')}")
                
                special_configs = config_summary.get('special_configurations', [])
                if special_configs:
                    st.write("**Special Configurations:**")
                    for special_config in special_configs:
                        st.write(f"• {special_config}")
            
            # API Status
            st.markdown("---")
            st.markdown("**🔌 System Status:**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                ai_status = "✅ Connected" if api_status.get('anthropic_connected') else "❌ Disconnected"
                st.info(f"**Anthropic AI:** {ai_status}")
            
            with col2:
                aws_status = "✅ Connected" if api_status.get('aws_api_connected') else "⚠️ Fallback Mode"
                st.info(f"**AWS API:** {aws_status}")
            
            with col3:
                pricing_source = api_status.get('pricing_source', 'fallback')
                pricing_status = "✅ Real-time" if pricing_source == 'aws_api' else "⚠️ Static Data"
                st.info(f"**Pricing Data:** {pricing_status}")
    
    # Professional footer
    st.markdown("""
    <div class="enterprise-footer">
        <h4>🚀 AWS Enterprise Database Migration Analyzer AI v3.0</h4>
        <p>Professional Migration Analysis • Real-time AWS Pricing • Comprehensive Service Sizing • Enterprise Cost Analysis</p>
        <p style="font-size: 0.9rem; margin-top: 1rem;">
            🔬 Advanced Analytics • 🎯 AI-Driven Insights • 📊 Executive Reporting • 💰 Detailed Cost Breakdown • 🛡️ Enterprise-Grade Analysis
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())