import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import math
import json
import asyncio
from typing import Dict, List, Tuple, Optional
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import anthropic
from anthropic import Anthropic

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

    .ai-card {
        background: linear-gradient(135deg, #fdf4ff 0%, #e879f9 100%);
        padding: 1.5rem;
        border-radius: 6px;
        color: #374151;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border-left: 3px solid #a855f7;
    }

    .aws-card {
        background: linear-gradient(135deg, #fef3c7 0%, #f59e0b 100%);
        padding: 1.5rem;
        border-radius: 6px;
        color: #374151;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border-left: 3px solid #d97706;
    }

    .connection-status {
        padding: 0.5rem 1rem;
        border-radius: 4px;
        margin: 0.5rem 0;
        font-weight: bold;
    }
    
    .status-connected {
        background-color: #d1fae5;
        color: #065f46;
        border: 1px solid #10b981;
    }
    
    .status-disconnected {
        background-color: #fee2e2;
        color: #991b1b;
        border: 1px solid #ef4444;
    }
    
    .status-partial {
        background-color: #fef3c7;
        color: #92400e;
        border: 1px solid #f59e0b;
    }
</style>
""", unsafe_allow_html=True)

class AWSIntegration:
    """AWS API integration for real-time metrics and recommendations"""
    
    def __init__(self):
        self.session = None
        self.cloudwatch = None
        self.datasync = None
        self.dms = None
        self.ec2 = None
        
    def initialize_aws_session(self, aws_access_key: str = None, aws_secret_key: str = None, region: str = 'us-west-2'):
        """Initialize AWS session with credentials"""
        try:
            if aws_access_key and aws_secret_key:
                self.session = boto3.Session(
                    aws_access_key_id=aws_access_key,
                    aws_secret_access_key=aws_secret_key,
                    region_name=region
                )
            else:
                # Use default credentials (IAM roles, ~/.aws/credentials, etc.)
                self.session = boto3.Session(region_name=region)
            
            self.cloudwatch = self.session.client('cloudwatch')
            self.datasync = self.session.client('datasync')
            self.dms = self.session.client('dms')
            self.ec2 = self.session.client('ec2')
            
            return True, "AWS connection established successfully"
        except NoCredentialsError:
            return False, "AWS credentials not found. Please configure credentials."
        except Exception as e:
            return False, f"AWS connection failed: {str(e)}"
    
    def get_datasync_tasks(self) -> List[Dict]:
        """Get existing DataSync tasks"""
        try:
            if not self.datasync:
                return []
                
            response = self.datasync.list_tasks()
            tasks = []
            
            for task in response.get('Tasks', []):
                task_arn = task['TaskArn']
                task_details = self.datasync.describe_task(TaskArn=task_arn)
                
                # Get execution history
                executions = self.datasync.list_task_executions(TaskArn=task_arn, MaxResults=5)
                
                tasks.append({
                    'name': task.get('Name', 'Unnamed Task'),
                    'arn': task_arn,
                    'status': task.get('Status', 'Unknown'),
                    'source_location': task_details.get('SourceLocationArn', 'Unknown'),
                    'destination_location': task_details.get('DestinationLocationArn', 'Unknown'),
                    'executions': executions.get('TaskExecutions', [])
                })
            
            return tasks
        except Exception as e:
            st.error(f"Error fetching DataSync tasks: {str(e)}")
            return []
    
    def get_dms_tasks(self) -> List[Dict]:
        """Get existing DMS replication tasks"""
        try:
            if not self.dms:
                return []
                
            response = self.dms.describe_replication_tasks()
            tasks = []
            
            for task in response.get('ReplicationTasks', []):
                tasks.append({
                    'name': task.get('ReplicationTaskIdentifier', 'Unnamed Task'),
                    'arn': task.get('ReplicationTaskArn', ''),
                    'status': task.get('Status', 'Unknown'),
                    'source_endpoint': task.get('SourceEndpointArn', 'Unknown'),
                    'target_endpoint': task.get('TargetEndpointArn', 'Unknown'),
                    'migration_type': task.get('MigrationType', 'Unknown'),
                    'table_mappings': task.get('TableMappings', '{}')
                })
            
            return tasks
        except Exception as e:
            st.error(f"Error fetching DMS tasks: {str(e)}")
            return []
    
    def get_cloudwatch_metrics(self, service: str, instance_id: str = None) -> Dict:
        """Get CloudWatch metrics for DataSync or DMS"""
        try:
            if not self.cloudwatch:
                return {}
            
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=24)
            
            if service == 'datasync':
                # DataSync metrics
                metrics = {
                    'BytesTransferred': self._get_metric_data('AWS/DataSync', 'BytesTransferred', start_time, end_time),
                    'FilesTransferred': self._get_metric_data('AWS/DataSync', 'FilesTransferred', start_time, end_time),
                }
            elif service == 'dms':
                # DMS metrics
                metrics = {
                    'CDCLatencySource': self._get_metric_data('AWS/DMS', 'CDCLatencySource', start_time, end_time, instance_id),
                    'CDCLatencyTarget': self._get_metric_data('AWS/DMS', 'CDCLatencyTarget', start_time, end_time, instance_id),
                    'CDCThroughputBandwidthSource': self._get_metric_data('AWS/DMS', 'CDCThroughputBandwidthSource', start_time, end_time, instance_id),
                }
            
            return metrics
        except Exception as e:
            st.error(f"Error fetching CloudWatch metrics: {str(e)}")
            return {}
    
    def _get_metric_data(self, namespace: str, metric_name: str, start_time: datetime, 
                        end_time: datetime, instance_id: str = None) -> List[Dict]:
        """Helper method to get metric data from CloudWatch"""
        try:
            dimensions = []
            if instance_id:
                dimensions = [{'Name': 'ReplicationInstanceIdentifier', 'Value': instance_id}]
            
            response = self.cloudwatch.get_metric_statistics(
                Namespace=namespace,
                MetricName=metric_name,
                Dimensions=dimensions,
                StartTime=start_time,
                EndTime=end_time,
                Period=3600,  # 1 hour periods
                Statistics=['Average', 'Maximum', 'Minimum']
            )
            
            return response.get('Datapoints', [])
        except Exception:
            return []

class ClaudeAIIntegration:
    """Claude AI integration for intelligent analysis and recommendations"""
    
    def __init__(self):
        self.client = None
        self.api_key = None
    
    def initialize_claude(self, api_key: str):
        """Initialize Claude AI client"""
        try:
            self.api_key = api_key
            self.client = Anthropic(api_key=api_key)
            
            # Test the connection
            test_response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=50,
                messages=[{"role": "user", "content": "Hello, are you working?"}]
            )
            
            return True, "Claude AI connection established successfully"
        except Exception as e:
            return False, f"Claude AI connection failed: {str(e)}"
    
    def analyze_migration_performance(self, config: Dict, network_perf: Dict, 
                                   agent_perf: Dict, aws_data: Dict = None) -> str:
        """Get Claude AI analysis of migration performance"""
        if not self.client:
            return "Claude AI not initialized"
        
        try:
            # Prepare context for Claude
            context = f"""
            Migration Configuration Analysis:
            
            Hardware Configuration:
            - Operating System: {config['operating_system']}
            - Server Type: {config['server_type']}
            - RAM: {config['ram_gb']} GB
            - CPU Cores: {config['cpu_cores']}
            - NIC: {config['nic_speed']} Mbps {config['nic_type']}
            - Storage Type: {config.get('storage_mount_type', 'Unknown')}
            
            Network Performance:
            - Path: {network_perf['path_name']}
            - Effective Bandwidth: {network_perf['effective_bandwidth_mbps']:.0f} Mbps
            - Total Latency: {network_perf['total_latency_ms']:.1f} ms
            - Reliability: {network_perf['total_reliability']*100:.2f}%
            - Quality Score: {network_perf['network_quality_score']:.1f}/100
            
            Agent Performance:
            - Type: {agent_perf['agent_type']}
            - Size: {agent_perf['agent_size']}
            - Count: {agent_perf['num_agents']}
            - Total Capacity: {agent_perf['total_agent_throughput_mbps']:.0f} Mbps
            - Monthly Cost: ${agent_perf['total_monthly_cost']:.0f}
            
            Migration Details:
            - Database Size: {config['database_size_gb']} GB
            - Migration Type: {config['migration_type']}
            - Environment: {config['environment']}
            
            Performance Differences Observed:
            - Linux NAS typically achieves 15-25% better performance than Windows mapped drives
            - VMware introduces 8-12% overhead compared to physical servers
            - DataSync on physical Linux can achieve near line-rate speeds
            - Windows Server mapped drives suffer from SMB protocol overhead
            
            {f"AWS Real-time Data: {json.dumps(aws_data, indent=2)}" if aws_data else "No real-time AWS data available"}
            """
            
            prompt = f"""
            As an AWS migration expert, analyze this migration configuration and provide:
            
            1. Performance bottleneck analysis
            2. Specific recommendations for optimization
            3. Expected vs actual performance explanation
            4. Cost optimization suggestions
            5. Risk assessment and mitigation strategies
            
            Focus on the real-world performance differences between:
            - Linux NAS vs Windows mapped drives
            - Physical vs VMware deployments
            - DataSync vs DMS for this specific use case
            
            Configuration to analyze:
            {context}
            
            Provide actionable, technical recommendations in a structured format.
            """
            
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.content[0].text
            
        except Exception as e:
            return f"Error getting Claude AI analysis: {str(e)}"
    
    def get_optimization_recommendations(self, bottleneck_type: str, current_config: Dict) -> str:
        """Get specific optimization recommendations based on bottleneck type"""
        if not self.client:
            return "Claude AI not initialized"
        
        try:
            prompt = f"""
            As an AWS migration specialist, the current migration setup has a {bottleneck_type} bottleneck.
            
            Current configuration:
            - Platform: {current_config['server_type']}
            - OS: {current_config['operating_system']}
            - Storage: {current_config.get('storage_mount_type', 'Unknown')}
            - Agent: {current_config['migration_type']}
            
            Provide 3-5 specific, actionable recommendations to resolve this {bottleneck_type} bottleneck.
            Include expected performance improvements and implementation complexity for each recommendation.
            """
            
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=800,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.content[0].text
            
        except Exception as e:
            return f"Error getting optimization recommendations: {str(e)}"

class EnhancedNetworkPathManager:
    """Enhanced network path manager with detailed storage type analysis"""
    
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
                'storage_mount_type': 'nfs',
                'segments': [
                    {
                        'name': 'Linux NAS to Linux Jump Server (NFS)',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 1.5,  # NFS is faster than SMB
                        'reliability': 0.9995,
                        'connection_type': 'internal_lan',
                        'cost_factor': 0.0,
                        'optimization_potential': 0.97,
                        'protocol_efficiency': 0.95  # NFS efficiency
                    },
                    {
                        'name': 'Linux Jump Server to AWS S3 (DX)',
                        'bandwidth_mbps': 2000,
                        'latency_ms': 12,
                        'reliability': 0.998,
                        'connection_type': 'direct_connect',
                        'cost_factor': 2.0,
                        'optimization_potential': 0.94
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
                'storage_mount_type': 'smb',
                'segments': [
                    {
                        'name': 'Windows Share to Windows Jump Server (SMB)',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 4,  # SMB has higher latency
                        'reliability': 0.995,
                        'connection_type': 'internal_lan',
                        'cost_factor': 0.0,
                        'optimization_potential': 0.85,  # Lower optimization potential
                        'protocol_efficiency': 0.78  # SMB overhead
                    },
                    {
                        'name': 'Windows Jump Server to AWS S3 (DX)',
                        'bandwidth_mbps': 2000,
                        'latency_ms': 18,
                        'reliability': 0.998,
                        'connection_type': 'direct_connect',
                        'cost_factor': 2.0,
                        'optimization_potential': 0.88
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
                'storage_mount_type': 'nfs',
                'segments': [
                    {
                        'name': 'San Antonio Linux NAS to Linux Jump Server (NFS)',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 0.8,
                        'reliability': 0.9998,
                        'connection_type': 'internal_lan',
                        'cost_factor': 0.0,
                        'optimization_potential': 0.98,
                        'protocol_efficiency': 0.96
                    },
                    {
                        'name': 'San Antonio to San Jose (Private Line)',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 10,
                        'reliability': 0.9995,
                        'connection_type': 'private_line',
                        'cost_factor': 3.0,
                        'optimization_potential': 0.96
                    },
                    {
                        'name': 'San Jose to AWS Production VPC S3 (DX)',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 6,
                        'reliability': 0.9999,
                        'connection_type': 'direct_connect',
                        'cost_factor': 4.0,
                        'optimization_potential': 0.97
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
                'storage_mount_type': 'smb',
                'segments': [
                    {
                        'name': 'San Antonio Windows Share to Windows Jump Server (SMB)',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 3,
                        'reliability': 0.996,
                        'connection_type': 'internal_lan',
                        'cost_factor': 0.0,
                        'optimization_potential': 0.86,
                        'protocol_efficiency': 0.80
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
                        'bandwidth_mbps': 10000,
                        'latency_ms': 8,
                        'reliability': 0.9999,
                        'connection_type': 'direct_connect',
                        'cost_factor': 4.0,
                        'optimization_potential': 0.95
                    }
                ]
            }
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
    
    def calculate_network_performance(self, path_key: str, time_of_day: int = None) -> Dict:
        """Calculate network performance with enhanced storage type considerations"""
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
            segment_latency = segment['latency_ms']
            segment_bandwidth = segment['bandwidth_mbps']
            segment_reliability = segment['reliability']
            
            # Apply protocol efficiency if present
            protocol_efficiency = segment.get('protocol_efficiency', 1.0)
            effective_bandwidth = segment_bandwidth * protocol_efficiency
            
            # Time-of-day congestion adjustments
            if segment['connection_type'] == 'internal_lan':
                congestion_factor = 1.15 if 9 <= time_of_day <= 17 else 0.92
            elif segment['connection_type'] == 'private_line':
                congestion_factor = 1.25 if 9 <= time_of_day <= 17 else 0.88
            elif segment['connection_type'] == 'direct_connect':
                congestion_factor = 1.08 if 9 <= time_of_day <= 17 else 0.96
            else:
                congestion_factor = 1.0
            
            # Apply congestion
            effective_bandwidth = effective_bandwidth / congestion_factor
            effective_latency = segment_latency * congestion_factor
            
            # Storage type specific adjustments
            if path['storage_mount_type'] == 'smb':
                # Windows SMB has additional overhead
                effective_bandwidth *= 0.82  # SMB protocol overhead
                effective_latency *= 1.3     # SMB latency penalty
            elif path['storage_mount_type'] == 'nfs':
                # Linux NFS is more efficient
                effective_bandwidth *= 0.96  # Minimal NFS overhead
                effective_latency *= 1.05    # Minimal NFS latency
            
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
                'congestion_factor': congestion_factor,
                'protocol_efficiency': protocol_efficiency
            })
        
        # Calculate quality scores
        latency_score = max(0, 100 - (total_latency * 1.5))
        bandwidth_score = min(100, (min_bandwidth / 1000) * 15)
        reliability_score = total_reliability * 100
        
        network_quality = (latency_score * 0.25 + bandwidth_score * 0.45 + reliability_score * 0.30)
        
        result = {
            'path_name': path['name'],
            'destination_storage': path['destination_storage'],
            'environment': path['environment'],
            'os_type': path['os_type'],
            'storage_mount_type': path['storage_mount_type'],
            'total_latency_ms': total_latency,
            'effective_bandwidth_mbps': min_bandwidth,
            'total_reliability': total_reliability,
            'network_quality_score': network_quality,
            'optimization_potential': (1 - optimization_score) * 100,
            'total_cost_factor': total_cost_factor,
            'segments': adjusted_segments
        }
        
        return result

class EnhancedAgentManager:
    """Enhanced agent manager with detailed physical vs VMware analysis"""
    
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
        
        # Physical vs VMware performance characteristics
        self.platform_characteristics = {
            'physical': {
                'base_efficiency': 1.0,
                'cpu_overhead': 0.0,
                'memory_overhead': 0.0,
                'io_efficiency': 1.0,
                'network_efficiency': 1.0
            },
            'vmware': {
                'base_efficiency': 0.92,  # 8% hypervisor overhead
                'cpu_overhead': 0.08,     # CPU overhead
                'memory_overhead': 0.12,  # Memory overhead
                'io_efficiency': 0.88,    # I/O overhead
                'network_efficiency': 0.94 # Network virtualization overhead
            }
        }
    
    def calculate_agent_performance(self, agent_type: str, agent_size: str, num_agents: int, 
                                   platform_type: str = 'vmware', storage_type: str = 'nas',
                                   os_type: str = 'linux') -> Dict:
        """Enhanced agent performance calculation with detailed platform analysis"""
        
        if agent_type == 'datasync':
            base_spec = self.datasync_specs[agent_size]
        else:
            base_spec = self.dms_specs[agent_size]
        
        # Platform characteristics
        platform_char = self.platform_characteristics[platform_type]
        
        # Calculate per-agent performance
        base_throughput = base_spec['throughput_mbps']
        
        # Apply platform efficiency
        platform_throughput = base_throughput * platform_char['base_efficiency']
        
        # Apply I/O efficiency based on storage type and OS
        if storage_type == 'nas' and os_type == 'linux':
            # Linux NAS (NFS) - optimal performance
            io_multiplier = 1.0
        elif storage_type == 'share' and os_type == 'windows':
            # Windows mapped drive (SMB) - reduced performance
            io_multiplier = 0.75  # 25% performance loss due to SMB overhead
        else:
            io_multiplier = 0.9
        
        # Network efficiency
        network_efficiency = platform_char['network_efficiency']
        
        # Final per-agent throughput
        per_agent_throughput = platform_throughput * io_multiplier * network_efficiency
        
        # Calculate scaling efficiency
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
        
        # Enhanced cost calculation
        base_cost_per_hour = base_spec['cost_hour']
        
        # VMware licensing overhead (if applicable)
        if platform_type == 'vmware':
            vmware_licensing_multiplier = 1.15  # 15% licensing overhead
        else:
            vmware_licensing_multiplier = 1.0
        
        per_agent_cost = base_cost_per_hour * 24 * 30 * vmware_licensing_multiplier
        total_monthly_cost = per_agent_cost * num_agents
        
        # Performance loss analysis
        max_theoretical = base_spec['throughput_mbps'] * num_agents
        actual_total = total_agent_throughput
        performance_loss = ((max_theoretical - actual_total) / max_theoretical) * 100
        
        return {
            'agent_type': agent_type,
            'agent_size': agent_size,
            'num_agents': num_agents,
            'platform_type': platform_type,
            'storage_type': storage_type,
            'os_type': os_type,
            'base_throughput_mbps': base_throughput,
            'per_agent_throughput_mbps': per_agent_throughput,
            'total_agent_throughput_mbps': total_agent_throughput,
            'scaling_efficiency': scaling_efficiency,
            'platform_efficiency': platform_char['base_efficiency'],
            'io_multiplier': io_multiplier,
            'network_efficiency': network_efficiency,
            'performance_loss_pct': performance_loss,
            'per_agent_monthly_cost': per_agent_cost,
            'total_monthly_cost': total_monthly_cost,
            'vmware_licensing_multiplier': vmware_licensing_multiplier,
            'base_spec': base_spec,
            'platform_characteristics': platform_char
        }

def get_nic_efficiency(nic_type: str) -> float:
    """Enhanced NIC efficiency based on type"""
    efficiencies = {
        'gigabit_copper': 0.82,
        'gigabit_fiber': 0.87,
        '10g_copper': 0.85,
        '10g_fiber': 0.91,
        '25g_fiber': 0.93,
        '40g_fiber': 0.94
    }
    return efficiencies.get(nic_type, 0.90)

def get_secrets_safely() -> Dict:
    """Safely retrieve secrets with error handling"""
    secrets = {}
    
    try:
        # AWS secrets
        secrets['aws_access_key'] = st.secrets.get("AWS_ACCESS_KEY", None)
        secrets['aws_secret_key'] = st.secrets.get("AWS_SECRET_KEY", None)
        secrets['aws_region'] = st.secrets.get("AWS_REGION", "us-west-2")
        
        # Claude AI secret
        secrets['claude_api_key'] = st.secrets.get("CLAUDE_API_KEY", None)
        
    except Exception as e:
        st.warning(f"Note: Some secrets are not configured in Streamlit Cloud: {str(e)}")
    
    return secrets

def initialize_integrations():
    """Initialize AWS and Claude integrations using secrets"""
    secrets = get_secrets_safely()
    
    # Initialize AWS Integration
    aws_integration = AWSIntegration()
    aws_status = "❌ Not Connected"
    aws_message = "AWS secrets not configured"
    
    if secrets.get('aws_access_key') and secrets.get('aws_secret_key'):
        try:
            success, message = aws_integration.initialize_aws_session(
                secrets['aws_access_key'], 
                secrets['aws_secret_key'], 
                secrets['aws_region']
            )
            if success:
                aws_status = "✅ Connected"
                aws_message = message
            else:
                aws_status = "⚠️ Connection Failed"
                aws_message = message
        except Exception as e:
            aws_status = "⚠️ Connection Failed"
            aws_message = str(e)
    
    # Initialize Claude AI Integration
    claude_integration = ClaudeAIIntegration()
    claude_status = "❌ Not Connected"
    claude_message = "Claude API key not configured"
    
    if secrets.get('claude_api_key'):
        try:
            success, message = claude_integration.initialize_claude(secrets['claude_api_key'])
            if success:
                claude_status = "✅ Connected"
                claude_message = message
            else:
                claude_status = "⚠️ Connection Failed"
                claude_message = message
        except Exception as e:
            claude_status = "⚠️ Connection Failed"
            claude_message = str(e)
    
    return {
        'aws_integration': aws_integration,
        'claude_integration': claude_integration,
        'aws_status': aws_status,
        'aws_message': aws_message,
        'claude_status': claude_status,
        'claude_message': claude_message
    }

def render_connection_status(status_info: Dict):
    """Render connection status in sidebar"""
    st.sidebar.subheader("🔗 API Connection Status")
    
    # AWS Status
    aws_status_class = "status-connected" if "✅" in status_info['aws_status'] else (
        "status-partial" if "⚠️" in status_info['aws_status'] else "status-disconnected"
    )
    
    st.sidebar.markdown(f"""
    <div class="connection-status {aws_status_class}">
        <strong>AWS Integration</strong><br>
        {status_info['aws_status']}<br>
        <small>{status_info['aws_message']}</small>
    </div>
    """, unsafe_allow_html=True)
    
    # Claude AI Status
    claude_status_class = "status-connected" if "✅" in status_info['claude_status'] else (
        "status-partial" if "⚠️" in status_info['claude_status'] else "status-disconnected"
    )
    
    st.sidebar.markdown(f"""
    <div class="connection-status {claude_status_class}">
        <strong>Claude AI Integration</strong><br>
        {status_info['claude_status']}<br>
        <small>{status_info['claude_message']}</small>
    </div>
    """, unsafe_allow_html=True)
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Connections"):
        st.rerun()
    
    # Configuration info
    with st.sidebar.expander("ℹ️ Configuration Info"):
        st.markdown("""
        **Required Streamlit Secrets:**
        
        For AWS Integration:
        - `AWS_ACCESS_KEY`
        - `AWS_SECRET_KEY`
        - `AWS_REGION` (optional, defaults to us-west-2)
        
        For Claude AI Integration:
        - `CLAUDE_API_KEY`
        
        **How to configure:**
        1. Go to your Streamlit Cloud app settings
        2. Navigate to the "Secrets" section
        3. Add the required keys as shown above
        """)

def render_enhanced_bandwidth_waterfall(config: Dict, network_perf: Dict, agent_perf: Dict):
    """Enhanced bandwidth waterfall with storage type analysis"""
    st.markdown("**🌊 Enhanced Bandwidth Waterfall: Complete Performance Analysis**")
    
    user_nic_speed = config['nic_speed']
    nic_type = config['nic_type']
    os_type = config['operating_system']
    platform_type = config['server_type']
    storage_mount = network_perf.get('storage_mount_type', 'unknown')
    
    # Enhanced analysis stages
    stages = ['Hardware\nNIC']
    throughputs = [user_nic_speed]
    descriptions = [f"{user_nic_speed:,.0f} Mbps {nic_type.replace('_', ' ')} NIC"]
    
    # NIC Processing Efficiency
    nic_efficiency = get_nic_efficiency(nic_type)
    after_nic = user_nic_speed * nic_efficiency
    stages.append('NIC\nProcessing')
    throughputs.append(after_nic)
    descriptions.append(f"{nic_efficiency*100:.1f}% NIC efficiency")
    
    # OS Network Stack
    os_efficiency = 0.92 if 'linux' in os_type else 0.86
    after_os = after_nic * os_efficiency
    stages.append('OS Network\nStack')
    throughputs.append(after_os)
    descriptions.append(f"{os_type.replace('_', ' ').title()} stack")
    
    # Platform Virtualization
    if platform_type == 'vmware':
        platform_efficiency = agent_perf['platform_efficiency']
        after_platform = after_os * platform_efficiency
        stages.append('VMware\nOverhead')
        throughputs.append(after_platform)
        descriptions.append(f"{platform_efficiency*100:.1f}% VMware efficiency")
    else:
        after_platform = after_os
        stages.append('Physical\nServer')
        throughputs.append(after_platform)
        descriptions.append('No virtualization overhead')
    
    # Storage Protocol Impact
    storage_efficiency = agent_perf['io_multiplier']
    after_storage = after_platform * storage_efficiency
    stages.append(f'{storage_mount.upper()}\nProtocol')
    throughputs.append(after_storage)
    descriptions.append(f"{storage_efficiency*100:.1f}% {storage_mount.upper()} efficiency")
    
    # Protocol Security Overhead
    protocol_efficiency = 0.85 if config['environment'] == 'production' else 0.88
    after_protocol = after_storage * protocol_efficiency
    stages.append('Security\nProtocols')
    throughputs.append(after_protocol)
    descriptions.append(f"{config['environment'].title()} security")
    
    # Network Path Limitation
    network_bandwidth = network_perf['effective_bandwidth_mbps']
    after_network = min(after_protocol, network_bandwidth)
    stages.append('Network\nBottleneck')
    throughputs.append(after_network)
    descriptions.append(f"{network_bandwidth:,.0f} Mbps network limit")
    
    # Agent Processing
    agent_capacity = agent_perf['total_agent_throughput_mbps']
    final_throughput = min(after_network, agent_capacity)
    stages.append('Final\nThroughput')
    throughputs.append(final_throughput)
    descriptions.append(f"{agent_perf['num_agents']}x {agent_perf['agent_type']} agents")
    
    # Create enhanced visualization
    waterfall_data = pd.DataFrame({
        'Stage': stages,
        'Throughput (Mbps)': throughputs,
        'Description': descriptions
    })
    
    # Create subplot with performance loss
    fig = px.bar(
        waterfall_data,
        x='Stage',
        y='Throughput (Mbps)',
        title=f"Enhanced Analysis: {user_nic_speed:,.0f} Mbps → {final_throughput:.0f} Mbps ({storage_mount.upper()}/{platform_type})",
        text='Throughput (Mbps)',
        color='Throughput (Mbps)',
        color_continuous_scale='RdYlGn'
    )
    
    fig.update_traces(texttemplate='%{text:.0f}', textposition='outside')
    fig.update_layout(height=500, showlegend=False)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Enhanced analysis summary
    total_loss = user_nic_speed - final_throughput
    total_loss_pct = (total_loss / user_nic_speed) * 100
    
    # Storage type impact analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **🔍 Storage Protocol Impact:**
        • **Protocol Type:** {storage_mount.upper()}
        • **Efficiency:** {storage_efficiency*100:.1f}%
        • **Performance Loss:** {(1-storage_efficiency)*100:.1f}%
        • **Bandwidth Impact:** {after_platform - after_storage:.0f} Mbps lost
        """)
    
    with col2:
        st.markdown(f"""
        **⚙️ Platform Impact:**
        • **Platform:** {platform_type.title()}
        • **Efficiency:** {agent_perf['platform_efficiency']*100:.1f}%
        • **Performance Loss:** {(1-agent_perf['platform_efficiency'])*100:.1f}%
        • **Bandwidth Impact:** {after_os - after_platform:.0f} Mbps lost
        """)
    
    # Comparison analysis
    if storage_mount == 'smb':
        nfs_efficiency = 1.0  # Theoretical NFS efficiency
        nfs_throughput = (after_platform / storage_efficiency) * nfs_efficiency
        performance_gain = nfs_throughput - after_storage
        
        st.warning(f"""
        **⚠️ SMB Protocol Performance Impact:**
        • **Current (SMB):** {after_storage:.0f} Mbps
        • **Potential (NFS):** {nfs_throughput:.0f} Mbps  
        • **Performance Gain:** {performance_gain:.0f} Mbps (+{(performance_gain/after_storage)*100:.1f}%)
        
        💡 **Recommendation:** Consider Linux NAS with NFS for optimal performance
        """)
    
    if platform_type == 'vmware':
        physical_throughput = after_os  # No virtualization overhead
        platform_gain = physical_throughput - after_platform
        
        st.info(f"""
        **☁️ VMware Virtualization Impact:**
        • **Current (VMware):** {after_platform:.0f} Mbps
        • **Potential (Physical):** {physical_throughput:.0f} Mbps
        • **Performance Gain:** {platform_gain:.0f} Mbps (+{(platform_gain/after_platform)*100:.1f}%)
        
        💡 **Trade-off:** Physical servers offer better performance but less flexibility
        """)

def render_storage_comparison_analysis(config: Dict):
    """Render detailed storage type comparison"""
    st.markdown("**📊 Storage Protocol Performance Comparison**")
    
    # Create comparison scenarios
    scenarios = []
    
    # Current configuration
    current_os = 'linux' if 'linux' in config['operating_system'].lower() else 'windows'
    current_storage = 'nfs' if current_os == 'linux' else 'smb'
    
    base_throughput = 1000  # Base throughput for comparison
    
    configurations = [
        {'name': 'Linux + NFS + Physical', 'os': 'linux', 'storage': 'nfs', 'platform': 'physical', 'efficiency': 0.96},
        {'name': 'Linux + NFS + VMware', 'os': 'linux', 'storage': 'nfs', 'platform': 'vmware', 'efficiency': 0.88},
        {'name': 'Windows + SMB + Physical', 'os': 'windows', 'storage': 'smb', 'platform': 'physical', 'efficiency': 0.75},
        {'name': 'Windows + SMB + VMware', 'os': 'windows', 'storage': 'smb', 'platform': 'vmware', 'efficiency': 0.69},
    ]
    
    for config_item in configurations:
        is_current = (config_item['os'] == current_os and 
                     config_item['platform'] == config['server_type'])
        
        scenarios.append({
            'Configuration': config_item['name'],
            'Throughput (Mbps)': base_throughput * config_item['efficiency'],
            'Efficiency (%)': config_item['efficiency'] * 100,
            'Current': '✓ Current' if is_current else '',
            'Performance Loss (%)': (1 - config_item['efficiency']) * 100
        })
    
    df_scenarios = pd.DataFrame(scenarios)
    
    # Create comparison chart
    fig = px.bar(
        df_scenarios,
        x='Configuration',
        y='Throughput (Mbps)',
        title='Storage Protocol & Platform Performance Comparison',
        color='Efficiency (%)',
        color_continuous_scale='RdYlGn',
        text='Throughput (Mbps)'
    )
    
    fig.update_traces(texttemplate='%{text:.0f} Mbps', textposition='outside')
    fig.update_layout(height=400, xaxis_tickangle=-45)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance impact table
    st.markdown("**📋 Detailed Performance Analysis:**")
    st.dataframe(df_scenarios.drop('Current', axis=1), use_container_width=True)
    
    # Key insights
    best_config = df_scenarios.loc[df_scenarios['Throughput (Mbps)'].idxmax()]
    worst_config = df_scenarios.loc[df_scenarios['Throughput (Mbps)'].idxmin()]
    
    performance_diff = best_config['Throughput (Mbps)'] - worst_config['Throughput (Mbps)']
    performance_diff_pct = (performance_diff / worst_config['Throughput (Mbps)']) * 100
    
    st.success(f"""
    **🏆 Key Performance Insights:**
    • **Best Configuration:** {best_config['Configuration']} ({best_config['Throughput (Mbps)']:.0f} Mbps)
    • **Worst Configuration:** {worst_config['Configuration']} ({worst_config['Throughput (Mbps)']:.0f} Mbps)
    • **Performance Gap:** {performance_diff:.0f} Mbps ({performance_diff_pct:.1f}% difference)
    • **Linux NFS Advantage:** ~20-25% better than Windows SMB
    • **Physical Server Advantage:** ~8-12% better than VMware
    """)

def render_aws_integration_panel(aws_integration: AWSIntegration):
    """Render AWS integration status and real-time data"""
    st.markdown("**☁️ AWS Real-Time Integration**")
    
    if not aws_integration.session:
        st.info("AWS integration not connected. Check sidebar for connection status.")
        return {}
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 DataSync Tasks**")
        datasync_tasks = aws_integration.get_datasync_tasks()
        
        if datasync_tasks:
            for task in datasync_tasks[:3]:  # Show first 3 tasks
                with st.expander(f"Task: {task['name']}"):
                    st.write(f"**Status:** {task['status']}")
                    st.write(f"**Source:** {task['source_location']}")
                    st.write(f"**Destination:** {task['destination_location']}")
                    if task['executions']:
                        latest_execution = task['executions'][0]
                        st.write(f"**Latest Execution:** {latest_execution.get('Status', 'Unknown')}")
        else:
            st.info("No DataSync tasks found in the current region.")
    
    with col2:
        st.markdown("**🔄 DMS Tasks**")
        dms_tasks = aws_integration.get_dms_tasks()
        
        if dms_tasks:
            for task in dms_tasks[:3]:  # Show first 3 tasks
                with st.expander(f"Task: {task['name']}"):
                    st.write(f"**Status:** {task['status']}")
                    st.write(f"**Migration Type:** {task['migration_type']}")
                    st.write(f"**Source:** {task['source_endpoint']}")
                    st.write(f"**Target:** {task['target_endpoint']}")
        else:
            st.info("No DMS tasks found in the current region.")
    
    return {
        'datasync_tasks': len(datasync_tasks),
        'dms_tasks': len(dms_tasks),
        'active_tasks': len([t for t in datasync_tasks if t['status'] == 'AVAILABLE']) + 
                       len([t for t in dms_tasks if t['status'] == 'ready'])
    }

def render_claude_ai_analysis(claude_integration: ClaudeAIIntegration, config: Dict, 
                            network_perf: Dict, agent_perf: Dict, aws_data: Dict):
    """Render Claude AI analysis panel"""
    st.markdown("**🤖 Claude AI Performance Analysis**")
    
    if not claude_integration.client:
        st.info("Claude AI integration not connected. Check sidebar for connection status.")
        return
    
    with st.spinner("Getting AI analysis..."):
        analysis = claude_integration.analyze_migration_performance(
            config, network_perf, agent_perf, aws_data
        )
    
    st.markdown(f"""
    <div class="ai-card">
        <h4>🧠 AI Performance Analysis</h4>
        <div style="white-space: pre-wrap;">{analysis}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Get specific optimization recommendations
    if network_perf['effective_bandwidth_mbps'] < agent_perf['total_agent_throughput_mbps']:
        bottleneck_type = "network"
    else:
        bottleneck_type = "agent"
    
    with st.expander("🎯 Targeted Optimization Recommendations"):
        with st.spinner("Getting optimization recommendations..."):
            recommendations = claude_integration.get_optimization_recommendations(
                bottleneck_type, config
            )
        st.markdown(recommendations)

def render_enhanced_sidebar():
    """Enhanced sidebar with connection status and configuration"""
    st.sidebar.header("🌐 Enhanced Migration Analyzer")
    
    # Configuration section
    st.sidebar.subheader("💻 System Configuration")
    
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
        index=1,
        format_func=lambda x: "🏢 Physical Server" if x == "physical" else "☁️ VMware Virtual Machine"
    )
    
    # Storage Configuration
    st.sidebar.subheader("💾 Storage Configuration")
    
    # Determine storage type based on OS
    os_lower = operating_system.lower()
    if 'linux' in os_lower:
        storage_options = ["nfs_nas", "iscsi_san", "local_storage"]
        storage_labels = {
            'nfs_nas': '📁 NFS Network Attached Storage',
            'iscsi_san': '🔗 iSCSI Storage Area Network', 
            'local_storage': '💽 Local Direct Attached Storage'
        }
        default_storage = "nfs_nas"
    else:
        storage_options = ["smb_share", "iscsi_san", "local_storage"]
        storage_labels = {
            'smb_share': '📁 SMB/CIFS Network Share',
            'iscsi_san': '🔗 iSCSI Storage Area Network',
            'local_storage': '💽 Local Direct Attached Storage'
        }
        default_storage = "smb_share"
    
    storage_type = st.sidebar.selectbox(
        "Storage Type",
        storage_options,
        index=0,
        format_func=lambda x: storage_labels[x]
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
        'storage_type': storage_type,
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
    """Enhanced main application with automatic secret integration"""
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌐 Enhanced AWS Network Migration Analyzer</h1>
        <p>AI-Powered Analysis • Real-Time AWS Metrics • Physical vs VMware Performance • Storage Protocol Optimization</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize integrations using secrets
    if 'integrations_initialized' not in st.session_state:
        with st.spinner("Initializing API integrations..."):
            integration_status = initialize_integrations()
            st.session_state.update(integration_status)
            st.session_state['integrations_initialized'] = True

    # Get configuration
    config = render_enhanced_sidebar()
    
    # Render connection status in sidebar
    render_connection_status({
        'aws_status': st.session_state.get('aws_status', '❌ Not Connected'),
        'aws_message': st.session_state.get('aws_message', 'Not initialized'),
        'claude_status': st.session_state.get('claude_status', '❌ Not Connected'),
        'claude_message': st.session_state.get('claude_message', 'Not initialized')
    })
    
    # Initialize enhanced managers
    network_manager = EnhancedNetworkPathManager()
    agent_manager = EnhancedAgentManager()
    
    # Get network path
    path_key = network_manager.get_network_path_key(config)
    network_perf = network_manager.calculate_network_performance(path_key)
    
    # Determine storage characteristics
    storage_type_mapping = {
        'nfs_nas': 'nas',
        'smb_share': 'share',
        'iscsi_san': 'san',
        'local_storage': 'local'
    }
    storage_type = storage_type_mapping.get(config['storage_type'], 'nas')
    
    os_type = 'linux' if 'linux' in config['operating_system'].lower() else 'windows'
    
    # Get agent performance
    agent_type = 'datasync' if config['is_homogeneous'] else 'dms'
    agent_perf = agent_manager.calculate_agent_performance(
        agent_type, config['agent_size'], config['number_of_agents'], 
        config['server_type'], storage_type, os_type
    )
    
    # Get AWS real-time data
    aws_data = {}
    if st.session_state.get('aws_integration') and st.session_state['aws_integration'].session:
        aws_data = render_aws_integration_panel(st.session_state['aws_integration'])
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🌊 Enhanced Bandwidth Analysis",
        "📊 Storage Performance Comparison", 
        "🌐 Network Paths",
        "🤖 Agent Performance",
        "☁️ AWS Integration",
        "🧠 AI Analysis"
    ])
    
    with tab1:
        st.subheader("🌊 Enhanced Bandwidth Waterfall Analysis")
        render_enhanced_bandwidth_waterfall(config, network_perf, agent_perf)
        
        # Physical vs VMware comparison
        st.markdown("**⚖️ Physical vs VMware Performance Impact**")
        
        # Calculate both scenarios
        physical_agent_perf = agent_manager.calculate_agent_performance(
            agent_type, config['agent_size'], config['number_of_agents'], 
            'physical', storage_type, os_type
        )
        
        vmware_agent_perf = agent_manager.calculate_agent_performance(
            agent_type, config['agent_size'], config['number_of_agents'], 
            'vmware', storage_type, os_type
        )
        
        comparison_col1, comparison_col2 = st.columns(2)
        
        with comparison_col1:
            st.metric(
                "🏢 Physical Server Performance",
                f"{physical_agent_perf['total_agent_throughput_mbps']:,.0f} Mbps",
                delta=f"+{physical_agent_perf['total_agent_throughput_mbps'] - vmware_agent_perf['total_agent_throughput_mbps']:.0f} Mbps vs VMware"
            )
        
        with comparison_col2:
            st.metric(
                "☁️ VMware Performance", 
                f"{vmware_agent_perf['total_agent_throughput_mbps']:,.0f} Mbps",
                delta=f"{vmware_agent_perf['performance_loss_pct']:.1f}% total loss"
            )
    
    with tab2:
        st.subheader("📊 Storage Protocol Performance Analysis")
        render_storage_comparison_analysis(config)
        
        # Real-world performance insights
        st.markdown("**🔬 Real-World Performance Insights**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="network-card">
                <h4>🐧 Linux NFS Advantages</h4>
                <ul>
                    <li><strong>Lower CPU overhead:</strong> Kernel-level NFS client</li>
                    <li><strong>Better caching:</strong> Page cache optimization</li>
                    <li><strong>Efficient metadata:</strong> Reduced round trips</li>
                    <li><strong>Parallel I/O:</strong> Multiple outstanding requests</li>
                    <li><strong>Network efficiency:</strong> TCP window scaling</li>
                </ul>
                <p><strong>Typical Performance:</strong> 85-95% of line rate</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="warning-card">
                <h4>🪟 Windows SMB Challenges</h4>
                <ul>
                    <li><strong>Protocol overhead:</strong> SMB2/3 authentication</li>
                    <li><strong>Opportunistic locks:</strong> Performance penalties</li>
                    <li><strong>Buffer management:</strong> User-space overhead</li>
                    <li><strong>Latency sensitivity:</strong> Chatty protocol</li>
                    <li><strong>Security overhead:</strong> Encryption impact</li>
                </ul>
                <p><strong>Typical Performance:</strong> 65-80% of line rate</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab3:
        # Existing network paths tab content
        st.subheader("🌐 Network Path Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🎯 Network Quality", f"{network_perf['network_quality_score']:.1f}/100")
        
        with col2:
            st.metric("⚡ Bandwidth", f"{network_perf['effective_bandwidth_mbps']:,.0f} Mbps")
        
        with col3:
            st.metric("🕐 Latency", f"{network_perf['total_latency_ms']:.1f} ms")
        
        with col4:
            st.metric("🛡️ Reliability", f"{network_perf['total_reliability']*100:.2f}%")
    
    with tab4:
        # Enhanced agent performance analysis
        st.subheader("🤖 Enhanced Agent Performance Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "🔧 Configuration",
                f"{agent_perf['num_agents']}x {agent_perf['agent_size'].title()}",
                delta=f"{agent_perf['agent_type'].upper()}"
            )
        
        with col2:
            st.metric(
                "⚡ Total Capacity",
                f"{agent_perf['total_agent_throughput_mbps']:,.0f} Mbps",
                delta=f"{agent_perf['performance_loss_pct']:.1f}% loss from ideal"
            )
        
        with col3:
            st.metric(
                "🎯 Platform Efficiency",
                f"{agent_perf['platform_efficiency']*100:.1f}%",
                delta=f"{config['server_type'].title()}"
            )
        
        with col4:
            st.metric(
                "💰 Monthly Cost",
                f"${agent_perf['total_monthly_cost']:,.0f}",
                delta=f"${agent_perf['per_agent_monthly_cost']:.0f}/agent"
            )
        
        # Detailed performance breakdown
        st.markdown("**📈 Performance Impact Breakdown**")
        
        impact_data = {
            'Factor': ['Base Throughput', 'Platform Efficiency', 'I/O Protocol', 'Network Efficiency', 'Scaling Factor'],
            'Impact (%)': [
                100,
                agent_perf['platform_efficiency'] * 100,
                agent_perf['io_multiplier'] * 100,
                agent_perf['network_efficiency'] * 100,
                agent_perf['scaling_efficiency'] * 100
            ],
            'Cumulative (Mbps)': [
                agent_perf['base_throughput_mbps'] * agent_perf['num_agents'],
                agent_perf['base_throughput_mbps'] * agent_perf['num_agents'] * agent_perf['platform_efficiency'],
                agent_perf['base_throughput_mbps'] * agent_perf['num_agents'] * agent_perf['platform_efficiency'] * agent_perf['io_multiplier'],
                agent_perf['base_throughput_mbps'] * agent_perf['num_agents'] * agent_perf['platform_efficiency'] * agent_perf['io_multiplier'] * agent_perf['network_efficiency'],
                agent_perf['total_agent_throughput_mbps']
            ]
        }
        
        df_impact = pd.DataFrame(impact_data)
        
        fig_impact = px.bar(
            df_impact,
            x='Factor',
            y='Cumulative (Mbps)',
            title='Agent Performance Impact Analysis',
            color='Impact (%)',
            color_continuous_scale='RdYlGn'
        )
        
        st.plotly_chart(fig_impact, use_container_width=True)
    
    with tab5:
        st.subheader("☁️ AWS Real-Time Integration")
        
        aws_integration = st.session_state.get('aws_integration')
        if aws_integration and aws_integration.session:
            aws_data_detailed = render_aws_integration_panel(aws_integration)
            
            # CloudWatch metrics visualization
            st.markdown("**📊 CloudWatch Metrics**")
            
            metrics_col1, metrics_col2 = st.columns(2)
            
            with metrics_col1:
                if agent_type == 'datasync':
                    datasync_metrics = aws_integration.get_cloudwatch_metrics('datasync')
                    if datasync_metrics:
                        st.success(f"Found {len(datasync_metrics)} DataSync metrics")
                    else:
                        st.info("No recent DataSync metrics available")
            
            with metrics_col2:
                dms_metrics = aws_integration.get_cloudwatch_metrics('dms')
                if dms_metrics:
                    st.success(f"Found {len(dms_metrics)} DMS metrics") 
                else:
                    st.info("No recent DMS metrics available")
        else:
            st.warning("AWS integration not connected. Check sidebar for connection status and configure secrets in Streamlit Cloud.")
            aws_data_detailed = {}
    
    with tab6:
        st.subheader("🧠 Claude AI Performance Analysis")
        
        claude_integration = st.session_state.get('claude_integration')
        if claude_integration and claude_integration.client:
            render_claude_ai_analysis(
                claude_integration, 
                config, network_perf, agent_perf, aws_data
            )
        else:
            st.warning("Claude AI integration not connected. Check sidebar for connection status and configure secrets in Streamlit Cloud.")
    
    # Final recommendations summary
    st.markdown("---")
    st.markdown("### 🎯 Executive Summary & Recommendations")
    
    final_throughput = min(
        network_perf['effective_bandwidth_mbps'],
        agent_perf['total_agent_throughput_mbps']
    )
    
    efficiency = (final_throughput / config['nic_speed']) * 100
    migration_time = (config['database_size_gb'] * 8 * 1000) / (final_throughput * 3600)
    
    summary_col1, summary_col2, summary_col3 = st.columns(3)
    
    with summary_col1:
        st.markdown(f"""
        <div class="performance-card">
            <h4>📊 Performance Summary</h4>
            <p><strong>Final Throughput:</strong> {final_throughput:,.0f} Mbps</p>
            <p><strong>Overall Efficiency:</strong> {efficiency:.1f}%</p>
            <p><strong>Migration Time:</strong> {migration_time:.1f} hours</p>
            <p><strong>Platform:</strong> {config['server_type'].title()}</p>
            <p><strong>Storage:</strong> {network_perf.get('storage_mount_type', 'Unknown').upper()}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with summary_col2:
        # Calculate potential improvements
        linux_nfs_improvement = 0.96 / agent_perf['io_multiplier'] if agent_perf['io_multiplier'] < 0.96 else 1.0
        physical_improvement = 1.0 / agent_perf['platform_efficiency'] if agent_perf['platform_efficiency'] < 1.0 else 1.0
        
        potential_throughput = final_throughput * linux_nfs_improvement * physical_improvement
        time_savings = migration_time - ((config['database_size_gb'] * 8 * 1000) / (potential_throughput * 3600))
        
        st.markdown(f"""
        <div class="network-card">
            <h4>🚀 Optimization Potential</h4>
            <p><strong>Current:</strong> {final_throughput:,.0f} Mbps</p>
            <p><strong>Optimized:</strong> {potential_throughput:,.0f} Mbps</p>
            <p><strong>Improvement:</strong> {((potential_throughput/final_throughput)-1)*100:.1f}%</p>
            <p><strong>Time Savings:</strong> {time_savings:.1f} hours</p>
            <p><strong>Recommendations:</strong> Linux NFS + Physical</p>
        </div>
        """, unsafe_allow_html=True)
    
    with summary_col3:
        cost_per_hour = agent_perf['total_monthly_cost'] / (24 * 30)
        migration_cost = cost_per_hour * migration_time
        
        st.markdown(f"""
        <div class="agent-card">
            <h4>💰 Cost Analysis</h4>
            <p><strong>Hourly Cost:</strong> ${cost_per_hour:.2f}/hour</p>
            <p><strong>Migration Cost:</strong> ${migration_cost:.2f}</p>
            <p><strong>Monthly Budget:</strong> ${agent_perf['total_monthly_cost']:,.0f}</p>
            <p><strong>Cost per GB:</strong> ${migration_cost/config['database_size_gb']:.4f}</p>
            <p><strong>Agent Efficiency:</strong> {100-agent_perf['performance_loss_pct']:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()