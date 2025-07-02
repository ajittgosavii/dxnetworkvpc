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


# ADD THIS HELPER FUNCTION at the top of your file (after imports)
def get_storage_impact(storage_technology, database_engine):
    """Get storage technology impact on bandwidth"""
    
    # Storage bandwidth capabilities
    storage_bandwidth = {
        'traditional_san': {'min': 4000, 'max': 8000},
        'ssd_enterprise': {'min': 8000, 'max': 15000},
        'nvme_enterprise': {'min': 20000, 'max': 40000},
        'nas_traditional': {'min': 2000, 'max': 6000},
        'local_ssd': {'min': 4000, 'max': 8000},
        'local_spinning': {'min': 1000, 'max': 3000}
    }
    
    # Protocol efficiency
    if database_engine == 'sqlserver':
        efficiency = 0.75  # SMB overhead
    else:
        efficiency = 0.95  # NFS efficiency
    
    storage_bw = storage_bandwidth.get(storage_technology, {'min': 4000, 'max': 8000})
    effective_min = int(storage_bw['min'] * efficiency)
    effective_max = int(storage_bw['max'] * efficiency)
    
    return {
        'bandwidth_range': f"{effective_min:,} - {effective_max:,} Mbps",
        'min_bandwidth': effective_min,
        'max_bandwidth': effective_max
    }

# Configure page
st.set_page_config(
    page_title="AWS DataSync Database Backup Migration Analyzer",
    page_icon="🗄️",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
                                   agent_perf: Dict, placement_analysis: Dict = None, 
                                   aws_data: Dict = None) -> str:
        """Get Claude AI analysis of database backup migration performance including agent placement"""
        if not self.client:
            return "Claude AI not initialized"
        
        try:
            # Prepare enhanced context for Claude with database backup focus
            context = f"""
            Database Backup Migration Configuration Analysis:
            
            Source Database and Backup Configuration:
            - Database Engine: {config['source_database_engine'].upper()}
            - Backup Location: {config.get('backup_storage_description', 'Unknown')}
            - Backup Size: {config['backup_size_gb']} GB
            - Backup Format: {config.get('backup_format', 'Native database backup')}
            - Storage Protocol: {config.get('storage_protocol', 'Unknown')}
            
            Hardware Configuration:
            - Operating System: {config['operating_system']}
            - Server Type: {config['server_type']}
            - RAM: {config['ram_gb']} GB
            - CPU Cores: {config['cpu_cores']}
            - NIC: {config['nic_speed']} Mbps {config['nic_type']}
            - Storage Type: {config.get('storage_type', 'Unknown')}
            
            Network Performance:
            - Path: {network_perf['path_name']}
            - Effective Bandwidth: {network_perf['effective_bandwidth_mbps']:.0f} Mbps
            - Total Latency: {network_perf['total_latency_ms']:.1f} ms
            - Reliability: {network_perf['total_reliability']*100:.2f}%
            - Quality Score: {network_perf['network_quality_score']:.1f}/100
            
            DataSync Agent Performance:
            - Agent Count: {agent_perf['num_agents']}
            - Agent Size: {agent_perf['agent_size']}
            - Total Capacity: {agent_perf['total_agent_throughput_mbps']:.0f} Mbps
            - Monthly Cost: ${agent_perf['total_monthly_cost']:.0f}
            - Platform Efficiency: {agent_perf['platform_efficiency']*100:.1f}%
            
            Migration Details:
            - Target: AWS S3
            - Environment: {config['environment']}
            - Migration Tool: AWS DataSync (file-based backup transfer)
            
            {f"Agent Placement Analysis: {json.dumps(placement_analysis, indent=2)}" if placement_analysis else "No placement analysis provided"}
            
            Database Backup Migration Considerations:
            - SQL Server backups on Windows Share (SMB) typically have different performance characteristics than Oracle/PostgreSQL backups on Linux NAS (NFS)
            - Large backup files (>100GB) benefit from DataSync's parallel transfer capabilities
            - Backup file compression can impact transfer speeds vs storage efficiency
            - Backup retention policies affect S3 storage class selection
            - DataSync preserves file metadata and timestamps critical for backup integrity
            
            {f"AWS Real-time Data: {json.dumps(aws_data, indent=2)}" if aws_data else "No real-time AWS data available"}
            """
            
            prompt = f"""
            As an AWS migration expert specializing in database backup migrations, analyze this DataSync configuration for transferring database backup files to S3 and provide a structured analysis with the following sections:

            1. DATABASE BACKUP MIGRATION ASSESSMENT
            2. DATASYNC AGENT PLACEMENT OPTIMIZATION FOR BACKUP FILES
            3. BACKUP FILE TRANSFER PERFORMANCE OPTIMIZATION
            4. STORAGE PROTOCOL IMPACT ON BACKUP TRANSFERS
            5. S3 DESTINATION OPTIMIZATION AND LIFECYCLE POLICIES
            6. BACKUP INTEGRITY AND VALIDATION STRATEGIES
            7. COST OPTIMIZATION FOR BACKUP STORAGE AND TRANSFER
            8. RISK ASSESSMENT AND BACKUP RECOVERY CONSIDERATIONS

            Focus on:
            - DataSync optimization for large database backup files
            - Windows Share (SMB) vs Linux NAS (NFS) performance differences for backup files
            - S3 storage class selection for backup retention
            - Backup file validation and integrity checking
            - Cost-effective backup storage strategies
            - Agent placement near backup storage locations

            Configuration to analyze:
            {context}

            Provide specific technical recommendations for DataSync agent placement, backup file transfer optimization, 
            and S3 storage configuration. Include expected transfer times and cost considerations for backup retention.
            """
            
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.content[0].text
            
        except Exception as e:
            return f"Error getting Claude AI analysis: {str(e)}"
    
    def get_optimization_recommendations(self, bottleneck_type: str, current_config: Dict) -> str:
        """Get specific optimization recommendations based on bottleneck type for backup migration"""
        if not self.client:
            return "Claude AI not initialized"
        
        try:
            prompt = f"""
            As an AWS DataSync specialist for database backup migrations, the current backup transfer setup has a {bottleneck_type} bottleneck.
            
            Current backup migration configuration:
            - Platform: {current_config['server_type']}
            - OS: {current_config['operating_system']}
            - Storage: {current_config.get('storage_type', 'Unknown')}
            - Backup Size: {current_config.get('backup_size_gb', 1000)} GB
            - Database: {current_config.get('source_database_engine', 'Unknown')}
            
            Provide 3-5 specific, actionable recommendations to resolve this {bottleneck_type} bottleneck for database backup file transfers to S3.
            Focus on DataSync optimization, backup file handling, and S3 configuration.
            Include expected performance improvements and implementation complexity for each recommendation.
            Format as clear bullet points with technical details.
            """
            
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=800,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.content[0].text
            
        except Exception as e:
            return f"Error getting optimization recommendations: {str(e)}"
    
    def get_placement_recommendations(self, config: Dict, placement_options: List[Dict]) -> str:
        """Get specific DataSync agent placement recommendations for backup migration"""
        if not self.client:
            return "Claude AI not initialized"
        
        try:
            prompt = f"""
            As an AWS DataSync specialist for database backup migrations, analyze these agent placement options and provide specific recommendations:
            
            Current Backup Migration Configuration:
            - Database Engine: {config.get('source_database_engine', 'Unknown')}
            - Backup Storage: {config.get('backup_storage_description', 'Unknown')}
            - Environment: {config['environment']}
            - Platform: {config['server_type']}
            - OS: {config['operating_system']}
            - Backup Size: {config.get('backup_size_gb', 1000)} GB
            
            Placement Options Analysis:
            {json.dumps(placement_options, indent=2)}
            
            Provide:
            1. Recommended DataSync agent placement strategy for backup file transfers
            2. Alternative options with trade-offs for backup scenarios
            3. Implementation considerations for backup storage access
            4. Performance expectations for large backup file transfers
            5. Security implications for backup data handling
            6. S3 destination configuration recommendations
            
            Focus on backup-specific considerations like file size, access patterns, and retention requirements.
            Format as clear bullet points with technical details.
            """
            
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.content[0].text
            
        except Exception as e:
            return f"Error getting placement recommendations: {str(e)}"

class WaterfallBandwidthAnalyzer:
    """Enhanced waterfall bandwidth analysis for detailed end-to-end network path segments"""
    
    def __init__(self):
        self.bottleneck_thresholds = {
            'critical': 0.3,    # Less than 30% of maximum
            'concerning': 0.6,  # Less than 60% of maximum
            'acceptable': 0.85  # Less than 85% of maximum
        }
    
    def analyze_waterfall(self, network_segments: List[Dict]) -> Dict:
        """Analyze bandwidth waterfall through network segments"""
        
        # Calculate cumulative bottlenecks
        waterfall_data = []
        cumulative_bandwidth = float('inf')
        cumulative_latency = 0
        cumulative_reliability = 1.0
        
        for i, segment in enumerate(network_segments):
            effective_bw = segment['effective_bandwidth_mbps']
            latency = segment['effective_latency_ms']
            reliability = segment['reliability']
            
            # Update cumulative metrics
            cumulative_bandwidth = min(cumulative_bandwidth, effective_bw)
            cumulative_latency += latency
            cumulative_reliability *= reliability
            
            # Calculate bottleneck severity
            if i == 0:
                bottleneck_ratio = 1.0
            else:
                previous_min = min([s['effective_bandwidth_mbps'] for s in network_segments[:i]])
                bottleneck_ratio = effective_bw / previous_min if previous_min > 0 else 1.0
            
            # Determine bottleneck status
            if bottleneck_ratio < self.bottleneck_thresholds['critical']:
                bottleneck_status = 'Critical Bottleneck'
                status_color = '#dc2626'
            elif bottleneck_ratio < self.bottleneck_thresholds['concerning']:
                bottleneck_status = 'Performance Concern'
                status_color = '#d97706'
            elif bottleneck_ratio < self.bottleneck_thresholds['acceptable']:
                bottleneck_status = 'Minor Impact'
                status_color = '#059669'
            else:
                bottleneck_status = 'Optimal'
                status_color = '#3b82f6'
            
            waterfall_data.append({
                'segment_name': segment['name'],
                'segment_bandwidth': effective_bw,
                'cumulative_bandwidth': cumulative_bandwidth,
                'cumulative_latency': cumulative_latency,
                'cumulative_reliability': cumulative_reliability,
                'bottleneck_ratio': bottleneck_ratio,
                'bottleneck_status': bottleneck_status,
                'status_color': status_color,
                'latency_contribution': latency,
                'reliability_impact': reliability,
                'connection_type': segment['connection_type']
            })
        
        # Find primary bottleneck
        min_bandwidth_segment = min(waterfall_data, key=lambda x: x['segment_bandwidth'])
        
        analysis = {
            'waterfall_segments': waterfall_data,
            'final_bandwidth': cumulative_bandwidth,
            'final_latency': cumulative_latency,
            'final_reliability': cumulative_reliability,
            'primary_bottleneck': min_bandwidth_segment,
            'bottleneck_count': len([s for s in waterfall_data if 'Bottleneck' in s['bottleneck_status']]),
            'optimization_potential': self._calculate_optimization_potential(waterfall_data)
        }
        
        return analysis
    
    def _calculate_optimization_potential(self, waterfall_data: List[Dict]) -> Dict:
        """Calculate optimization potential for the network path"""
        
        if not waterfall_data:
            return {'total_potential': 0, 'recommendations': []}
        
        max_possible = max([s['segment_bandwidth'] for s in waterfall_data])
        current_final = min([s['segment_bandwidth'] for s in waterfall_data])
        
        potential_improvement = ((max_possible - current_final) / current_final) * 100 if current_final > 0 else 0
        
        recommendations = []
        
        # Identify optimization opportunities
        for segment in waterfall_data:
            if 'Bottleneck' in segment['bottleneck_status']:
                recommendations.append({
                    'segment': segment['segment_name'],
                    'current_bandwidth': segment['segment_bandwidth'],
                    'improvement_type': 'Critical Path Optimization',
                    'priority': 'High'
                })
            elif segment['bottleneck_ratio'] < 0.8:
                recommendations.append({
                    'segment': segment['segment_name'],
                    'current_bandwidth': segment['segment_bandwidth'],
                    'improvement_type': 'Performance Tuning',
                    'priority': 'Medium'
                })
        
        return {
            'total_potential': potential_improvement,
            'recommendations': recommendations,
            'max_possible_bandwidth': max_possible,
            'current_bandwidth': current_final
        }

class EnhancedNetworkPathManager:
    """Enhanced network path manager with detailed end-to-end database backup storage focus"""
    
    def __init__(self):
        self.network_paths = {
            # SQL Server backup paths (Windows Share) - Non-Production with full end-to-end view
            'nonprod_sj_sqlserver_windows_share_s3': {
                'name': 'Non-Prod: SQL Server Backups → AWS S3 (End-to-End)',
                'destination_storage': 'S3',
                'source': 'San Jose',
                'destination': 'AWS US-West-2 S3',
                'environment': 'non-production',
                'database_engine': 'sqlserver',
                'os_type': 'windows',
                'storage_type': 'share',
                'storage_mount_type': 'smb',
                'backup_location': 'Windows File Share',
                'segments': [
                    {
                        'name': 'OS: Windows Server SQL Backup Process',
                        'bandwidth_mbps': 6000,
                        'latency_ms': 0.3,
                        'reliability': 0.9999,
                        'connection_type': 'os_layer',
                        'cost_factor': 0.0,
                        'optimization_potential': 0.85,
                        'protocol_efficiency': 0.75
                    },
                    {
                        'name': 'NIC: 10Gbps Ethernet Interface',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 0.1,
                        'reliability': 0.9998,
                        'connection_type': 'nic_layer',
                        'cost_factor': 0.0,
                        'optimization_potential': 0.88,
                        'protocol_efficiency': 0.85
                    },
                    {
                        'name': 'LAN Switch: Access Layer (SMB)',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 0.8,
                        'reliability': 0.9996,
                        'connection_type': 'lan_switch',
                        'cost_factor': 0.0,
                        'optimization_potential': 0.82,
                        'protocol_efficiency': 0.78
                    },
                    {
                        'name': 'LAN Switch: Distribution Layer',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 1.2,
                        'reliability': 0.9995,
                        'connection_type': 'lan_switch',
                        'cost_factor': 0.0,
                        'optimization_potential': 0.85,
                        'protocol_efficiency': 0.82
                    },
                    {
                        'name': 'Network Link: Campus Backbone',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 2.5,
                        'reliability': 0.9994,
                        'connection_type': 'network_link',
                        'cost_factor': 1.0,
                        'optimization_potential': 0.88,
                        'protocol_efficiency': 0.85
                    },
                    {
                        'name': 'Router: Core Network Router',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 1.8,
                        'reliability': 0.9997,
                        'connection_type': 'router',
                        'cost_factor': 0.5,
                        'optimization_potential': 0.90,
                        'protocol_efficiency': 0.88
                    },
                    {
                        'name': 'Firewall: Enterprise Security Gateway',
                        'bandwidth_mbps': 8000,
                        'latency_ms': 3.2,
                        'reliability': 0.9993,
                        'connection_type': 'firewall',
                        'cost_factor': 1.5,
                        'optimization_potential': 0.75,
                        'protocol_efficiency': 0.72
                    },
                    {
                        'name': 'Network Link: Internet Gateway',
                        'bandwidth_mbps': 2000,
                        'latency_ms': 8.5,
                        'reliability': 0.998,
                        'connection_type': 'internet_gateway',
                        'cost_factor': 2.0,
                        'optimization_potential': 0.85,
                        'protocol_efficiency': 0.82
                    },
                    {
                        'name': 'AWS: Internet Gateway → S3',
                        'bandwidth_mbps': 2000,
                        'latency_ms': 5.2,
                        'reliability': 0.9999,
                        'connection_type': 'aws_service',
                        'cost_factor': 0.0,
                        'optimization_potential': 0.95,
                        'protocol_efficiency': 0.92
                    }
                ]
            },
            # SQL Server backup paths (Windows Share) - Production with full end-to-end view
            'prod_sa_sqlserver_windows_share_s3': {
                'name': 'Prod: SQL Server Backups → San Jose → AWS S3 (End-to-End)',
                'destination_storage': 'S3',
                'source': 'San Antonio',
                'destination': 'AWS US-West-2 Production VPC S3',
                'environment': 'production',
                'database_engine': 'sqlserver',
                'os_type': 'windows',
                'storage_type': 'share',
                'storage_mount_type': 'smb',
                'backup_location': 'Windows File Share',
                'segments': [
                    {
                        'name': 'OS: Windows Server SQL Backup Process',
                        'bandwidth_mbps': 6000,
                        'latency_ms': 0.3,
                        'reliability': 0.9999,
                        'connection_type': 'os_layer',
                        'cost_factor': 0.0,
                        'optimization_potential': 0.85,
                        'protocol_efficiency': 0.75
                    },
                    {
                        'name': 'NIC: 10Gbps Fiber Interface',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 0.08,
                        'reliability': 0.9999,
                        'connection_type': 'nic_layer',
                        'cost_factor': 0.0,
                        'optimization_potential': 0.92,
                        'protocol_efficiency': 0.89
                    },
                    {
                        'name': 'LAN Switch: Access Layer (SMB)',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 0.6,
                        'reliability': 0.9998,
                        'connection_type': 'lan_switch',
                        'cost_factor': 0.0,
                        'optimization_potential': 0.86,
                        'protocol_efficiency': 0.82
                    },
                    {
                        'name': 'LAN Switch: Distribution Layer',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 0.9,
                        'reliability': 0.9997,
                        'connection_type': 'lan_switch',
                        'cost_factor': 0.0,
                        'optimization_potential': 0.88,
                        'protocol_efficiency': 0.85
                    },
                    {
                        'name': 'Network Link: Data Center Backbone',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 1.8,
                        'reliability': 0.9996,
                        'connection_type': 'network_link',
                        'cost_factor': 1.0,
                        'optimization_potential': 0.91,
                        'protocol_efficiency': 0.88
                    },
                    {
                        'name': 'Router: WAN Edge Router (San Antonio)',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 2.2,
                        'reliability': 0.9998,
                        'connection_type': 'router',
                        'cost_factor': 1.5,
                        'optimization_potential': 0.93,
                        'protocol_efficiency': 0.90
                    },
                    {
                        'name': 'Network Link: San Antonio ↔ San Jose (Private Line)',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 8.5,
                        'reliability': 0.9995,
                        'connection_type': 'private_line',
                        'cost_factor': 3.0,
                        'optimization_potential': 0.94,
                        'protocol_efficiency': 0.92
                    },
                    {
                        'name': 'Router: WAN Edge Router (San Jose)',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 1.8,
                        'reliability': 0.9998,
                        'connection_type': 'router',
                        'cost_factor': 1.5,
                        'optimization_potential': 0.93,
                        'protocol_efficiency': 0.90
                    },
                    {
                        'name': 'Firewall: Production Security Gateway',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 2.5,
                        'reliability': 0.9996,
                        'connection_type': 'firewall',
                        'cost_factor': 2.0,
                        'optimization_potential': 0.85,
                        'protocol_efficiency': 0.82
                    },
                    {
                        'name': 'DX Link: AWS Direct Connect (10Gbps)',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 2.8,
                        'reliability': 0.9999,
                        'connection_type': 'direct_connect',
                        'cost_factor': 4.0,
                        'optimization_potential': 0.96,
                        'protocol_efficiency': 0.94
                    },
                    {
                        'name': 'AWS: Direct Connect → Production VPC S3',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 1.2,
                        'reliability': 0.99999,
                        'connection_type': 'aws_service',
                        'cost_factor': 0.0,
                        'optimization_potential': 0.98,
                        'protocol_efficiency': 0.96
                    }
                ]
            },
            # Oracle/PostgreSQL backup paths (Linux NAS) - Non-Production with full end-to-end view
            'nonprod_sj_oracle_linux_nas_s3': {
                'name': 'Non-Prod: Oracle/PostgreSQL Backups → AWS S3 (End-to-End)',
                'destination_storage': 'S3',
                'source': 'San Jose',
                'destination': 'AWS US-West-2 S3',
                'environment': 'non-production',
                'database_engine': 'oracle_postgresql',
                'os_type': 'linux',
                'storage_type': 'nas',
                'storage_mount_type': 'nfs',
                'backup_location': 'Linux NAS',
                'segments': [
                    {
                        'name': 'OS: Linux Database Backup Process', 
                        'bandwidth_mbps': 10000,   # REDUCED from 25000
                        'latency_ms': 0.15,
                        'reliability': 0.99995,
                        'connection_type': 'os_layer',
                        'cost_factor': 0.0,
                        'optimization_potential': 0.92,
                        'protocol_efficiency': 0.95 
                    },
                    {
                        'name': 'NIC: 10Gbps Ethernet Interface',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 0.08,
                        'reliability': 0.9999,
                        'connection_type': 'nic_layer',
                        'cost_factor': 0.0,
                        'optimization_potential': 0.92,
                        'protocol_efficiency': 0.89
                    },
                    {
                        'name': 'LAN Switch: Access Layer (NFS)',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 0.5,
                        'reliability': 0.9998,
                        'connection_type': 'lan_switch',
                        'cost_factor': 0.0,
                        'optimization_potential': 0.94,
                        'protocol_efficiency': 0.91
                    },
                    {
                        'name': 'LAN Switch: Distribution Layer',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 0.7,
                        'reliability': 0.9997,
                        'connection_type': 'lan_switch',
                        'cost_factor': 0.0,
                        'optimization_potential': 0.92,
                        'protocol_efficiency': 0.89
                    },
                    {
                        'name': 'Network Link: Campus Backbone',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 1.8,
                        'reliability': 0.9996,
                        'connection_type': 'network_link',
                        'cost_factor': 1.0,
                        'optimization_potential': 0.90,
                        'protocol_efficiency': 0.87
                    },
                    {
                        'name': 'Router: Core Network Router',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 1.5,
                        'reliability': 0.9998,
                        'connection_type': 'router',
                        'cost_factor': 0.5,
                        'optimization_potential': 0.92,
                        'protocol_efficiency': 0.89
                    },
                    {
                        'name': 'Firewall: Enterprise Security Gateway',
                        'bandwidth_mbps': 8000,
                        'latency_ms': 2.8,
                        'reliability': 0.9994,
                        'connection_type': 'firewall',
                        'cost_factor': 1.5,
                        'optimization_potential': 0.78,
                        'protocol_efficiency': 0.75
                    },
                    {
                        'name': 'Network Link: Internet Gateway',
                        'bandwidth_mbps': 2000,
                        'latency_ms': 7.2,
                        'reliability': 0.998,
                        'connection_type': 'internet_gateway',
                        'cost_factor': 2.0,
                        'optimization_potential': 0.88,
                        'protocol_efficiency': 0.85
                    },
                    {
                        'name': 'AWS: Internet Gateway → S3',
                        'bandwidth_mbps': 2000,
                        'latency_ms': 4.5,
                        'reliability': 0.9999,
                        'connection_type': 'aws_service',
                        'cost_factor': 0.0,
                        'optimization_potential': 0.96,
                        'protocol_efficiency': 0.93
                    }
                ]
            },
            # Oracle/PostgreSQL backup paths (Linux NAS) - Production with full end-to-end view
            'prod_sa_oracle_linux_nas_s3': {
                'name': 'Prod: Oracle/PostgreSQL Backups → San Jose → AWS S3 (End-to-End)',
                'destination_storage': 'S3',
                'source': 'San Antonio',
                'destination': 'AWS US-West-2 Production VPC S3',
                'environment': 'production',
                'database_engine': 'oracle_postgresql',
                'os_type': 'linux',
                'storage_type': 'nas',
                'storage_mount_type': 'nfs',
                'backup_location': 'Linux NAS',
                'segments': [
                    {
                        'name': 'OS: Linux Database Backup Process', 
                        'bandwidth_mbps': 10000,   # REDUCED from 25000
                        'latency_ms': 0.15,
                        'reliability': 0.99995,
                        'connection_type': 'os_layer',
                        'cost_factor': 0.0,
                        'optimization_potential': 0.92,
                        'protocol_efficiency': 0.95 
                    },
                    {
                        'name': 'NIC: 10Gbps Fiber Interface',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 0.06,
                        'reliability': 0.99995,
                        'connection_type': 'nic_layer',
                        'cost_factor': 0.0,
                        'optimization_potential': 0.94,
                        'protocol_efficiency': 0.91
                    },
                    {
                        'name': 'LAN Switch: Access Layer (NFS)',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 0.4,
                        'reliability': 0.9999,
                        'connection_type': 'lan_switch',
                        'cost_factor': 0.0,
                        'optimization_potential': 0.96,
                        'protocol_efficiency': 0.93
                    },
                    {
                        'name': 'LAN Switch: Distribution Layer',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 0.5,
                        'reliability': 0.9998,
                        'connection_type': 'lan_switch',
                        'cost_factor': 0.0,
                        'optimization_potential': 0.94,
                        'protocol_efficiency': 0.91
                    },
                    {
                        'name': 'Network Link: Data Center Backbone',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 1.2,
                        'reliability': 0.9997,
                        'connection_type': 'network_link',
                        'cost_factor': 1.0,
                        'optimization_potential': 0.93,
                        'protocol_efficiency': 0.90
                    },
                    {
                        'name': 'Router: WAN Edge Router (San Antonio)',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 1.5,
                        'reliability': 0.9999,
                        'connection_type': 'router',
                        'cost_factor': 1.5,
                        'optimization_potential': 0.95,
                        'protocol_efficiency': 0.92
                    },
                    {
                        'name': 'Network Link: San Antonio ↔ San Jose (Private Line)',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 7.8,
                        'reliability': 0.9996,
                        'connection_type': 'private_line',
                        'cost_factor': 3.0,
                        'optimization_potential': 0.96,
                        'protocol_efficiency': 0.93
                    },
                    {
                        'name': 'Router: WAN Edge Router (San Jose)',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 1.2,
                        'reliability': 0.9999,
                        'connection_type': 'router',
                        'cost_factor': 1.5,
                        'optimization_potential': 0.95,
                        'protocol_efficiency': 0.92
                    },
                    {
                        'name': 'Firewall: Production Security Gateway',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 2.0,
                        'reliability': 0.9997,
                        'connection_type': 'firewall',
                        'cost_factor': 2.0,
                        'optimization_potential': 0.88,
                        'protocol_efficiency': 0.85
                    },
                    {
                        'name': 'DX Link: AWS Direct Connect (10Gbps)',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 2.2,
                        'reliability': 0.99995,
                        'connection_type': 'direct_connect',
                        'cost_factor': 4.0,
                        'optimization_potential': 0.97,
                        'protocol_efficiency': 0.95
                    },
                    {
                        'name': 'AWS: Direct Connect → Production VPC S3',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 0.8,
                        'reliability': 0.99999,
                        'connection_type': 'aws_service',
                        'cost_factor': 0.0,
                        'optimization_potential': 0.99,
                        'protocol_efficiency': 0.97
                    }
                ]
            }
        }

                # Storage technology bandwidth mapping
        self.storage_bandwidth = {
            'traditional_san': 6000,      # Realistic (was 25,000 in your segments)
            'ssd_enterprise': 12000,      # Enterprise SSD arrays
            'nvme_enterprise': 30000,     # High-end NVMe
            'nas_traditional': 4000,      # Traditional NAS
            'local_ssd': 6000,           # Local SSD
            'local_spinning': 2000        # Spinning disks
        }
        
        # Database protocol efficiency
        self.protocol_efficiency = {
            'sqlserver': 0.75,   # SMB/CIFS overhead
            'oracle': 0.95,      # NFS efficiency
            'postgresql': 0.95,  # NFS efficiency
            'mysql': 0.92        # NFS efficiency
        }      
    
        def get_realistic_os_bandwidth(self, storage_technology, database_engine, backup_size_gb):
            """Calculate realistic OS bandwidth based on storage technology"""
            
            # Get base storage bandwidth
            base_bandwidth = self.storage_bandwidth.get(storage_technology, 6000)
            
            # Apply protocol efficiency
            protocol_eff = self.protocol_efficiency.get(database_engine, 0.90)
            
            # Large file bonus for backup scenarios
            if backup_size_gb > 1000:
                large_file_bonus = 1.1
            elif backup_size_gb > 500:
                large_file_bonus = 1.05
            else:
                large_file_bonus = 1.0
            
            # Calculate effective bandwidth
            effective_bandwidth = int(base_bandwidth * protocol_eff * large_file_bonus)
            
            return {
                'bandwidth_mbps': effective_bandwidth,
                'base_storage_mbps': base_bandwidth,
                'protocol_efficiency': protocol_eff,
                'large_file_bonus': large_file_bonus
            }
        
        def calculate_network_performance(self, path_key: str, config: Dict, time_of_day: int = None) -> Dict:
            """Calculate network performance with REALISTIC storage-based OS bandwidth"""
            path = self.network_paths[path_key]
            
            if time_of_day is None:
                time_of_day = datetime.now().hour
            
            # Get realistic OS bandwidth based on user's storage selection
            storage_technology = config.get('storage_technology', 'traditional_san')
            database_engine = config.get('source_database_engine', 'sqlserver')
            backup_size_gb = config.get('backup_size_gb', 1000)
            
            os_bandwidth_info = self.get_realistic_os_bandwidth(
                storage_technology, database_engine, backup_size_gb
            )
            
            total_latency = 0
            min_bandwidth = float('inf')
            total_reliability = 1.0
            total_cost_factor = 0
            optimization_score = 1.0
            
            adjusted_segments = []
            
            for i, segment in enumerate(path['segments']):
                # OVERRIDE the first segment (OS layer) with realistic bandwidth
                if i == 0 and segment['connection_type'] == 'os_layer':
                    segment_bandwidth = os_bandwidth_info['bandwidth_mbps']
                    # Update the segment name to reflect storage technology
                    segment['name'] = f"OS: {database_engine.upper()} Backup ({storage_technology.replace('_', ' ').title()})"
                else:
                    segment_bandwidth = segment['bandwidth_mbps']
                
                segment_latency = segment['latency_ms']
                segment_reliability = segment['reliability']
                
                # Apply protocol efficiency if present
                protocol_efficiency = segment.get('protocol_efficiency', 1.0)
                if i == 0:  # Use calculated efficiency for OS layer
                    protocol_efficiency = os_bandwidth_info['protocol_efficiency']
                
                effective_bandwidth = segment_bandwidth * protocol_efficiency
                
                # Time-of-day congestion adjustments (existing logic)
                if segment['connection_type'] in ['os_layer', 'nic_layer', 'lan_switch']:
                    congestion_factor = 1.15 if 9 <= time_of_day <= 17 else 0.92
                elif segment['connection_type'] in ['network_link', 'private_line']:
                    congestion_factor = 1.25 if 9 <= time_of_day <= 17 else 0.88
                elif segment['connection_type'] in ['router', 'firewall', 'direct_connect']:
                    congestion_factor = 1.08 if 9 <= time_of_day <= 17 else 0.96
                else:
                    congestion_factor = 1.0
                
                # Apply congestion
                effective_bandwidth = effective_bandwidth / congestion_factor
                effective_latency = segment_latency * congestion_factor
                
                # Database backup specific adjustments (existing logic)
                if path['storage_mount_type'] == 'smb':
                    if 'SMB' in segment['name'] or 'Windows' in segment['name'] or i == 0:
                        effective_bandwidth *= 0.85  # Reduced from 0.78
                        effective_latency *= 1.3     # Reduced from 1.4
                elif path['storage_mount_type'] == 'nfs':
                    if 'NFS' in segment['name'] or 'Linux' in segment['name'] or i == 0:
                        effective_bandwidth *= 0.96  # Slightly better
                        effective_latency *= 1.05    # Reduced from 1.1
                
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
                'database_engine': path['database_engine'],
                'backup_location': path['backup_location'],
                'os_type': path['os_type'],
                'storage_mount_type': path['storage_mount_type'],
                'storage_technology': storage_technology,  # NEW: Add this
                'os_bandwidth_info': os_bandwidth_info,    # NEW: Add this
                'total_latency_ms': total_latency,
                'effective_bandwidth_mbps': min_bandwidth,
                'total_reliability': total_reliability,
                'network_quality_score': network_quality,
                'optimization_potential': (1 - optimization_score) * 100,
                'total_cost_factor': total_cost_factor,
                'segments': adjusted_segments
            }
            
            return result
    
                 
        
    def get_network_path_key(self, config: Dict) -> str:
        """Get network path key based on database engine and configuration"""
        database_engine = config['source_database_engine']
        environment = config['environment']
        
        # Determine path based on database engine
        if database_engine == 'sqlserver':
            if environment == 'non-production':
                return 'nonprod_sj_sqlserver_windows_share_s3'
            else:
                return 'prod_sa_sqlserver_windows_share_s3'
        else:  # Oracle, PostgreSQL, MySQL
            if environment == 'non-production':
                return 'nonprod_sj_oracle_linux_nas_s3'
            else:
                return 'prod_sa_oracle_linux_nas_s3'
    
    def calculate_network_performance(self, path_key: str, time_of_day: int = None) -> Dict:
        """Calculate network performance with backup-specific considerations"""
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
            if segment['connection_type'] in ['os_layer', 'nic_layer', 'lan_switch']:
                congestion_factor = 1.15 if 9 <= time_of_day <= 17 else 0.92
            elif segment['connection_type'] in ['network_link', 'private_line']:
                congestion_factor = 1.25 if 9 <= time_of_day <= 17 else 0.88
            elif segment['connection_type'] in ['router', 'firewall', 'direct_connect']:
                congestion_factor = 1.08 if 9 <= time_of_day <= 17 else 0.96
            else:
                congestion_factor = 1.0
            
            # Apply congestion
            effective_bandwidth = effective_bandwidth / congestion_factor
            effective_latency = segment_latency * congestion_factor
            
            # Database backup specific adjustments
            if path['storage_mount_type'] == 'smb':
                # SMB has higher overhead for large backup files
                if 'SMB' in segment['name'] or 'Windows' in segment['name']:
                    effective_bandwidth *= 0.78  # More aggressive reduction for backup files
                    effective_latency *= 1.4
            elif path['storage_mount_type'] == 'nfs':
                # NFS performs better with large sequential reads (backup files)
                if 'NFS' in segment['name'] or 'Linux' in segment['name']:
                    effective_bandwidth *= 0.94
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
            'database_engine': path['database_engine'],
            'backup_location': path['backup_location'],
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
    """Enhanced agent manager focused on DataSync for database backup migrations"""
    
    def __init__(self):
        # Updated AWS DataSync pricing (per hour) as of 2024 - focused on DataSync since we're doing file transfers
        self.datasync_specs = {
            'small': {'throughput_mbps': 250, 'vcpu': 2, 'memory': 4, 'cost_hour': 0.0416},
            'medium': {'throughput_mbps': 500, 'vcpu': 2, 'memory': 8, 'cost_hour': 0.0832},
            'large': {'throughput_mbps': 1000, 'vcpu': 4, 'memory': 16, 'cost_hour': 0.1664},
            'xlarge': {'throughput_mbps': 2000, 'vcpu': 8, 'memory': 32, 'cost_hour': 0.3328}
        }
        
        # Physical vs VMware performance characteristics for backup file transfers
        self.platform_characteristics = {
            'physical': {
                'base_efficiency': 1.0,
                'cpu_overhead': 0.0,
                'memory_overhead': 0.0,
                'io_efficiency': 1.0,
                'network_efficiency': 1.0,
                'placement_bonus': 1.0,
                'backup_file_efficiency': 1.0  # No overhead for large file operations
            },
            'vmware': {
                'base_efficiency': 0.89,  # Slightly lower for backup file I/O
                'cpu_overhead': 0.11,
                'memory_overhead': 0.15,
                'io_efficiency': 0.85,  # More impact on large file I/O
                'network_efficiency': 0.92,
                'placement_bonus': 0.93,
                'backup_file_efficiency': 0.87  # VMware overhead more pronounced with large files
            }
        }

        # Agent placement characteristics for backup scenarios
        self.placement_characteristics = {
            'source_colocation': {
                'name': 'Co-located with Backup Storage',
                'latency_reduction_ms': -3.5,
                'bandwidth_bonus': 1.25,  # Higher bonus for backup file access
                'reliability_bonus': 1.03,
                'security_score': 0.80,  # Lower for backup security
                'cost_multiplier': 1.0,
                'management_complexity': 0.6,
                'backup_access_efficiency': 1.15  # Direct backup storage access
            },
            'dmz_placement': {
                'name': 'DMZ with Backup Access',
                'latency_reduction_ms': -1.5,
                'bandwidth_bonus': 1.12,
                'reliability_bonus': 1.02,
                'security_score': 0.95,
                'cost_multiplier': 1.15,
                'management_complexity': 0.75,
                'backup_access_efficiency': 1.05
            },
            'centralized_datacenter': {
                'name': 'Centralized Data Center',
                'latency_reduction_ms': 0.0,
                'bandwidth_bonus': 1.0,
                'reliability_bonus': 1.0,
                'security_score': 1.0,
                'cost_multiplier': 1.0,
                'management_complexity': 1.0,
                'backup_access_efficiency': 1.0
            },
            'edge_deployment': {
                'name': 'Edge Near Backup Sources',
                'latency_reduction_ms': -5.0,
                'bandwidth_bonus': 1.35,  # Highest for backup scenarios
                'reliability_bonus': 0.97,
                'security_score': 0.70,
                'cost_multiplier': 1.4,
                'management_complexity': 0.5,
                'backup_access_efficiency': 1.25
            }
        }
    
    def calculate_agent_performance(self, agent_size: str, num_agents: int, 
                                   platform_type: str = 'vmware', storage_type: str = 'nas',
                                   os_type: str = 'linux', placement_type: str = 'centralized_datacenter',
                                   backup_size_gb: int = 1000) -> Dict:
        """Enhanced DataSync agent performance calculation for backup migrations"""
        
        # Always use DataSync for backup file transfers
        base_spec = self.datasync_specs[agent_size]
        
        # Platform characteristics
        platform_char = self.platform_characteristics[platform_type]
        placement_char = self.placement_characteristics[placement_type]
        
        # Calculate per-agent performance
        base_throughput = base_spec['throughput_mbps']
        
        # Apply platform efficiency with backup file considerations
        platform_throughput = base_throughput * platform_char['base_efficiency'] * platform_char['backup_file_efficiency']
        
        # Apply storage protocol efficiency based on backup storage type
        if storage_type == 'nas' and os_type == 'linux':
            io_multiplier = 1.0  # NFS optimal for backup files
        elif storage_type == 'share' and os_type == 'windows':
            io_multiplier = 0.70  # SMB less efficient for large backup files
        else:
            io_multiplier = 0.85
        
        # Apply placement bonus with backup access efficiency
        placement_throughput = platform_throughput * io_multiplier * placement_char['bandwidth_bonus'] * placement_char['backup_access_efficiency']
        
        # Network efficiency
        network_efficiency = platform_char['network_efficiency']
        
        # Backup file size scaling factor (larger files = better efficiency)
        if backup_size_gb >= 1000:
            backup_size_factor = 1.1  # Large files benefit from DataSync optimization
        elif backup_size_gb >= 500:
            backup_size_factor = 1.05
        else:
            backup_size_factor = 1.0
        
        # Final per-agent throughput
        per_agent_throughput = placement_throughput * network_efficiency * backup_size_factor
        
        # Calculate scaling efficiency for backup scenarios
        if num_agents == 1:
            scaling_efficiency = 1.0
        elif num_agents <= 3:
            scaling_efficiency = 0.96  # Slightly better for backup files
        elif num_agents <= 5:
            scaling_efficiency = 0.92
        else:
            scaling_efficiency = 0.87
        
        # Total agent capacity
        total_agent_throughput = per_agent_throughput * num_agents * scaling_efficiency
        
        # Enhanced cost calculation
        base_cost_per_hour = base_spec['cost_hour']
        
        # VMware licensing overhead
        if platform_type == 'vmware':
            vmware_licensing_multiplier = 1.15
        else:
            vmware_licensing_multiplier = 1.0
        
        # Placement cost considerations
        placement_cost_multiplier = placement_char['cost_multiplier']
        
        per_agent_cost = base_cost_per_hour * 24 * 30 * vmware_licensing_multiplier * placement_cost_multiplier
        total_monthly_cost = per_agent_cost * num_agents
        
        # Performance loss analysis
        max_theoretical = base_spec['throughput_mbps'] * num_agents
        actual_total = total_agent_throughput
        performance_loss = ((max_theoretical - actual_total) / max_theoretical) * 100
        
        return {
            'agent_type': 'datasync',  # Always DataSync for backup files
            'agent_size': agent_size,
            'num_agents': num_agents,
            'platform_type': platform_type,
            'storage_type': storage_type,
            'os_type': os_type,
            'placement_type': placement_type,
            'backup_size_gb': backup_size_gb,
            'base_throughput_mbps': base_throughput,
            'per_agent_throughput_mbps': per_agent_throughput,
            'total_agent_throughput_mbps': total_agent_throughput,
            'scaling_efficiency': scaling_efficiency,
            'platform_efficiency': platform_char['base_efficiency'],
            'backup_file_efficiency': platform_char['backup_file_efficiency'],
            'io_multiplier': io_multiplier,
            'network_efficiency': network_efficiency,
            'placement_bonus': placement_char['bandwidth_bonus'],
            'backup_access_efficiency': placement_char['backup_access_efficiency'],
            'backup_size_factor': backup_size_factor,
            'performance_loss_pct': performance_loss,
            'per_agent_monthly_cost': per_agent_cost,
            'total_monthly_cost': total_monthly_cost,
            'vmware_licensing_multiplier': vmware_licensing_multiplier,
            'placement_cost_multiplier': placement_cost_multiplier,
            'base_spec': base_spec,
            'platform_characteristics': platform_char,
            'placement_characteristics': placement_char
        }

class AgentPlacementAnalyzer:
    """Comprehensive DataSync agent placement analysis for database backup migrations"""
    
    def __init__(self):
        self.placement_strategies = {
            'source_colocation': {
                'name': 'Co-located with Backup Storage',
                'description': 'DataSync agents placed directly with database backup storage',
                'pros': [
                    'Direct access to backup files',
                    'Maximum bandwidth to backup storage',
                    'Minimal latency for backup file reads',
                    'Optimal for large backup files'
                ],
                'cons': [
                    'Distributed agent management',
                    'Backup storage security exposure',
                    'Limited scalability per location',
                    'Higher operational complexity'
                ],
                'best_for': ['Large backup files (>500GB)', 'High-speed backup transfers', 'Minimal transfer windows'],
                'avoid_when': ['High security backup requirements', 'Limited backup storage resources']
            },
            'dmz_placement': {
                'name': 'DMZ with Backup Access',
                'description': 'DataSync agents in DMZ with secure backup storage access',
                'pros': [
                    'Balanced security and performance',
                    'Centralized security controls',
                    'Good backup access performance',
                    'Controlled backup data flow'
                ],
                'cons': [
                    'Additional security complexity',
                    'Backup access latency increase',
                    'Firewall configuration overhead',
                    'DMZ resource requirements'
                ],
                'best_for': ['Production backup environments', 'Compliance requirements', 'Secured backup transfers'],
                'avoid_when': ['Ultra-high backup performance needs', 'Simple backup architectures']
            },
            'centralized_datacenter': {
                'name': 'Centralized Data Center',
                'description': 'DataSync agents in central enterprise data center',
                'pros': [
                    'Simplified agent management',
                    'Centralized monitoring',
                    'Standard backup procedures',
                    'Easy maintenance and updates'
                ],
                'cons': [
                    'Network latency to backup storage',
                    'Bandwidth sharing with other services',
                    'Single point of failure',
                    'Distance from backup sources'
                ],
                'best_for': ['Multiple backup sources', 'Standard backup operations', 'Operational simplicity'],
                'avoid_when': ['Performance-critical backup windows', 'Distributed backup storage']
            },
            'edge_deployment': {
                'name': 'Edge Near Backup Sources',
                'description': 'DataSync agents at network edge closest to backup storage',
                'pros': [
                    'Ultra-low latency to backups',
                    'Maximum backup transfer performance',
                    'Optimized backup data paths',
                    'Reduced core network backup load'
                ],
                'cons': [
                    'Complex distributed management',
                    'Edge security challenges',
                    'Higher deployment costs',
                    'Limited monitoring capabilities'
                ],
                'best_for': ['Performance-critical backup transfers', 'Large backup volumes', 'Time-sensitive backup windows'],
                'avoid_when': ['Budget constraints', 'Security-first backup policies']
            }
        }
    
    def analyze_placement_options(self, config: Dict, network_perf: Dict, agent_manager: EnhancedAgentManager) -> List[Dict]:
        """Analyze all placement options for backup migration and score them"""
        placement_options = []
        
        for placement_type, strategy in self.placement_strategies.items():
            # Calculate performance for this placement
            agent_perf = agent_manager.calculate_agent_performance(
                config['agent_size'], 
                config['number_of_agents'],
                config['server_type'],
                self._map_storage_type(config['storage_type']),
                self._determine_os_type(config['operating_system']),
                placement_type,
                config.get('backup_size_gb', 1000)
            )
            
            # Calculate placement score
            score = self._calculate_placement_score(placement_type, config, network_perf, agent_perf)
            
            # Determine implementation complexity
            complexity = self._assess_implementation_complexity(placement_type, config)
            
            placement_options.append({
                'placement_type': placement_type,
                'strategy': strategy,
                'agent_performance': agent_perf,
                'placement_score': score,
                'implementation_complexity': complexity,
                'throughput_mbps': agent_perf['total_agent_throughput_mbps'],
                'monthly_cost': agent_perf['total_monthly_cost'],
                'latency_impact': agent_perf['placement_characteristics']['latency_reduction_ms'],
                'security_score': agent_perf['placement_characteristics']['security_score'],
                'management_complexity': agent_perf['placement_characteristics']['management_complexity'],
                'backup_access_efficiency': agent_perf['backup_access_efficiency']
            })
        
        # Sort by placement score
        placement_options.sort(key=lambda x: x['placement_score'], reverse=True)
        
        return placement_options
    
    def _calculate_placement_score(self, placement_type: str, config: Dict, network_perf: Dict, agent_perf: Dict) -> float:
        """Calculate comprehensive placement score for backup scenarios (0-100)"""
        # Performance score (45% weight - higher for backup performance)
        max_possible_throughput = 2000 * config['number_of_agents']
        performance_score = (agent_perf['total_agent_throughput_mbps'] / max_possible_throughput) * 100
        performance_score = min(performance_score, 100)
        
        # Backup access efficiency bonus (10% weight)
        backup_efficiency_score = agent_perf['backup_access_efficiency'] * 100
        
        # Cost efficiency score (20% weight)
        cost_per_mbps = agent_perf['total_monthly_cost'] / agent_perf['total_agent_throughput_mbps']
        max_cost_per_mbps = 1000
        cost_score = max(0, 100 - (cost_per_mbps / max_cost_per_mbps * 100))
        
        # Security score (15% weight)
        security_score = agent_perf['placement_characteristics']['security_score'] * 100
        
        # Management complexity score (10% weight)
        management_score = agent_perf['placement_characteristics']['management_complexity'] * 100
        
        # Calculate weighted total
        total_score = (
            performance_score * 0.45 +
            backup_efficiency_score * 0.10 +
            cost_score * 0.20 +
            security_score * 0.15 +
            management_score * 0.10
        )
        
        return min(total_score, 100)
    
    def _assess_implementation_complexity(self, placement_type: str, config: Dict) -> Dict:
        """Assess implementation complexity for backup migration placement"""
        complexity_factors = {
            'source_colocation': {
                'setup_time_days': 2,
                'skill_level': 'Medium',
                'infrastructure_changes': 'Minimal',
                'security_review': 'Required',
                'ongoing_maintenance': 'Medium',
                'backup_integration': 'Direct'
            },
            'dmz_placement': {
                'setup_time_days': 5,
                'skill_level': 'High',
                'infrastructure_changes': 'Moderate',
                'security_review': 'Extensive',
                'ongoing_maintenance': 'Moderate',
                'backup_integration': 'Secured'
            },
            'centralized_datacenter': {
                'setup_time_days': 1,
                'skill_level': 'Low',
                'infrastructure_changes': 'Minimal',
                'security_review': 'Standard',
                'ongoing_maintenance': 'Low',
                'backup_integration': 'Standard'
            },
            'edge_deployment': {
                'setup_time_days': 7,
                'skill_level': 'Expert',
                'infrastructure_changes': 'Significant',
                'security_review': 'Extensive',
                'ongoing_maintenance': 'High',
                'backup_integration': 'Optimized'
            }
        }
        
        return complexity_factors.get(placement_type, complexity_factors['centralized_datacenter'])
    
    def _map_storage_type(self, storage_type: str) -> str:
        """Map storage type to simplified categories"""
        mapping = {
            'windows_share': 'share',
            'linux_nas': 'nas',
            'iscsi_san': 'san',
            'local_storage': 'local'
        }
        return mapping.get(storage_type, 'nas')
    
    def _determine_os_type(self, operating_system: str) -> str:
        """Determine OS type from operating system"""
        return 'linux' if 'linux' in operating_system.lower() else 'windows'

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
        secrets['aws_region'] = st.secrets.get("AWS_REGION", "us-east-1")
        
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
                claude_integration = None
        except Exception as e:
            claude_status = "⚠️ Connection Failed"
            claude_message = str(e)
            claude_integration = None
    else:
        claude_integration = None
    
    return {
        'aws_integration': aws_integration,
        'claude_integration': claude_integration,
        'aws_status': aws_status,
        'aws_message': aws_message,
        'claude_status': claude_status,
        'claude_message': claude_message
    }

def render_connection_status(status_info: Dict):
    """Render connection status in sidebar using native Streamlit components"""
    st.sidebar.subheader("🔗 API Connection Status")
    
    # AWS Status
    if "✅" in status_info['aws_status']:
        st.sidebar.success(f"**AWS Integration**\n{status_info['aws_status']}\n{status_info['aws_message']}")
    elif "⚠️" in status_info['aws_status']:
        st.sidebar.warning(f"**AWS Integration**\n{status_info['aws_status']}\n{status_info['aws_message']}")
    else:
        st.sidebar.error(f"**AWS Integration**\n{status_info['aws_status']}\n{status_info['aws_message']}")
    
    # Claude AI Status
    if "✅" in status_info['claude_status']:
        st.sidebar.success(f"**Claude AI Integration**\n{status_info['claude_status']}\n{status_info['claude_message']}")
    elif "⚠️" in status_info['claude_status']:
        st.sidebar.warning(f"**Claude AI Integration**\n{status_info['claude_status']}\n{status_info['claude_message']}")
    else:
        st.sidebar.error(f"**Claude AI Integration**\n{status_info['claude_status']}\n{status_info['claude_message']}")
    
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
        - `AWS_REGION` (optional, defaults to us-east-1)
        
        For Claude AI Integration:
        - `CLAUDE_API_KEY`
        
        **How to configure:**
        1. Go to your Streamlit Cloud app settings
        2. Navigate to the "Secrets" section
        3. Add the required keys as shown above
        """)

def render_enhanced_sidebar():
    """Enhanced sidebar focused on database backup migration scenarios using native Streamlit components"""
    st.sidebar.header("🗄️ Database Backup Migration Analyzer")
    
    # Database Configuration Section
    st.sidebar.subheader("🗃️ Database Configuration")
    
    # Source Database Engine (determines backup storage type)
    source_database_engine = st.sidebar.selectbox(
        "Source Database Engine",
        ["sqlserver", "oracle", "postgresql", "mysql"],
        index=0,
        format_func=lambda x: {
            'sqlserver': '🔵 Microsoft SQL Server',
            'oracle': '🔴 Oracle Database',
            'postgresql': '🐘 PostgreSQL',
            'mysql': '🐬 MySQL'
        }[x]
    )
    
    # Backup Storage Configuration (auto-determined by database engine)
    if source_database_engine == 'sqlserver':
        backup_storage_description = "Windows File Share (SMB/CIFS)"
        storage_type = "windows_share"
        operating_system = "windows_server_2019"
        storage_protocol = "SMB/CIFS"
    else:  # Oracle, PostgreSQL, MySQL
        backup_storage_description = "Linux NAS (NFS)"
        storage_type = "linux_nas"
        operating_system = "rhel_8"
        storage_protocol = "NFS"
    
    st.sidebar.info(f"**Backup Storage:** {backup_storage_description}")
    
    # Backup Configuration
    st.sidebar.subheader("💾 Backup Configuration")
    
    backup_size_gb = st.sidebar.number_input(
        "Total Backup Size (GB)", 
        min_value=100, 
        max_value=50000, 
        value=1000, 
        step=100,
        help="Total size of database backup files to transfer"
    )
    
    backup_format = st.sidebar.selectbox(
        "Backup Format",
        ["native", "compressed", "encrypted"],
        format_func=lambda x: {
            'native': '📦 Native Database Backup',
            'compressed': '🗜️ Compressed Backup',
            'encrypted': '🔒 Encrypted Backup'
        }[x]
    )
    
    # *** ADD THIS SECTION - Storage Technology Selection ***
    st.sidebar.subheader("💽 Backup Storage Technology")
    
    # Default storage technology based on database engine
    if source_database_engine == 'sqlserver':
        default_storage_options = ["traditional_san", "ssd_enterprise", "nvme_enterprise"]
        default_index = 0  # Traditional SAN most common for SQL Server
        default_protocol = "SMB/CIFS"
        default_os = "windows_server_2019"
    else:
        default_storage_options = ["ssd_enterprise", "nvme_enterprise", "traditional_san", "nas_traditional"]
        default_index = 0  # SSD Enterprise most common for modern Oracle/PostgreSQL
        default_protocol = "NFS"
        default_os = "rhel_8"
    
    # Storage technology selection
    storage_technology = st.sidebar.selectbox(
        "Backup Storage Technology",
        default_storage_options,
        index=default_index,
        format_func=lambda x: {
            'traditional_san': '💾 Traditional SAN (6-8 Gbps)',
            'ssd_enterprise': '⚡ Enterprise SSD (10-15 Gbps)', 
            'nvme_enterprise': '🚀 NVMe Enterprise (25-40 Gbps)',
            'nas_traditional': '📁 Traditional NAS (2-6 Gbps)',
            'local_ssd': '💻 Local SSD (4-8 Gbps)',
            'local_spinning': '⏳ Local Spinning Disks (1-3 Gbps)'
        }[x],
        help="Storage technology where database backup files are stored"
    )
    
    # Show storage impact
    storage_impact = get_storage_impact(storage_technology, source_database_engine)
    st.sidebar.info(f"**Expected OS Bandwidth:** {storage_impact['bandwidth_range']}")
    st.sidebar.info(f"**Protocol:** {default_protocol}")
    
    # Storage location description (now dynamic based on technology)
    if source_database_engine == 'sqlserver':
        if storage_technology == 'nvme_enterprise':
            backup_storage_description = "High-Performance Windows Share (SMB3)"
        else:
            backup_storage_description = "Windows File Share (SMB/CIFS)"
        storage_type = "windows_share"
        operating_system = default_os
        storage_protocol = default_protocol
    else:
        if storage_technology == 'nvme_enterprise':
            backup_storage_description = "High-Performance Linux NAS (NFSv4)"
        elif storage_technology == 'nas_traditional':
            backup_storage_description = "Traditional Linux NAS (NFS)"
        else:
            backup_storage_description = "Enterprise Linux NAS (NFS)"
        storage_type = "linux_nas"
        operating_system = default_os
        storage_protocol = default_protocol
    
    
    
    # Environment
    environment = st.sidebar.selectbox(
        "Environment", 
        ["non-production", "production"],
        format_func=lambda x: "🔧 Non-Production" if x == "non-production" else "🏭 Production"
    )
    
    # Hardware Configuration
    st.sidebar.subheader("⚙️ DataSync Agent Hardware")
    
    # Server Platform
    server_type = st.sidebar.selectbox(
        "Agent Platform",
        ["physical", "vmware"],
        index=1,
        format_func=lambda x: "🏢 Physical Server" if x == "physical" else "☁️ VMware Virtual Machine"
    )
    
    ram_gb = st.sidebar.selectbox("RAM (GB)", [8, 16, 32, 64, 128], index=2)
    cpu_cores = st.sidebar.selectbox("CPU Cores", [2, 4, 8, 16, 24], index=2)
    
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
    
    # DataSync Agent Configuration
    st.sidebar.subheader("🤖 DataSync Agent Configuration")
    
    st.sidebar.info("**Migration Tool:** AWS DataSync (optimized for backup file transfers)")
    
    number_of_agents = st.sidebar.number_input(
        "Number of DataSync Agents", 
        min_value=1, 
        max_value=10, 
        value=2, 
        step=1,
        help="Number of DataSync agents for parallel backup file transfer"
    )
    
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
    
    # S3 Destination Configuration
    st.sidebar.subheader("☁️ S3 Destination")
    
    s3_storage_class = st.sidebar.selectbox(
        "S3 Storage Class",
        ["standard", "standard_ia", "glacier", "deep_archive"],
        format_func=lambda x: {
            'standard': '⚡ S3 Standard',
            'standard_ia': '📦 S3 Standard-IA',
            'glacier': '🧊 S3 Glacier',
            'deep_archive': '🗄️ S3 Glacier Deep Archive'
        }[x]
    )
    
    return {
        'source_database_engine': source_database_engine,
        'backup_storage_description': backup_storage_description,
        'backup_size_gb': backup_size_gb,
        'backup_format': backup_format,
        'storage_protocol': storage_protocol,
        'storage_technology': storage_technology,  # NEW: Add this
        'operating_system': operating_system,
        'server_type': server_type,
        'storage_type': storage_type,
        'ram_gb': ram_gb,
        'cpu_cores': cpu_cores,
        'nic_type': nic_type,
        'nic_speed': nic_speed,
        'environment': environment,
        'number_of_agents': number_of_agents,
        'agent_size': agent_size,
        'migration_type': 'datasync',  # Always DataSync for file transfers
        's3_storage_class': s3_storage_class
    }

def render_backup_migration_overview(config: Dict):
    """Render database backup migration overview using native Streamlit components"""
    st.markdown("### 🗄️ Database Backup Migration Overview")
    
    # Migration flow visualization
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.success("**🗃️ Source Database**")
        st.write(f"**Engine:** {config['source_database_engine'].upper()}")
        st.write(f"**Backup Size:** {config['backup_size_gb']:,} GB")
        st.write(f"**Format:** {config['backup_format'].title()}")
        st.write(f"**Environment:** {config['environment'].title()}")
    
    with col2:
        st.info("**📁 Backup Storage**")
        st.write(f"**Type:** {config['backup_storage_description']}")
        st.write(f"**Protocol:** {config['storage_protocol']}")
        st.write(f"**OS:** {config['operating_system'].replace('_', ' ').title()}")
        st.write(f"**Access:** Network Attached")
    
    with col3:
        st.warning("**🤖 DataSync Agents**")
        st.write(f"**Count:** {config['number_of_agents']} agents")
        st.write(f"**Size:** {config['agent_size'].title()}")
        st.write(f"**Platform:** {config['server_type'].title()}")
        st.write(f"**Purpose:** File Transfer")
    
    with col4:
        st.error("**☁️ AWS S3 Destination**", icon="☁️")
        st.write(f"**Service:** Amazon S3")
        st.write(f"**Storage Class:** {config['s3_storage_class'].replace('_', ' ').title()}")
        st.write(f"**Region:** US-West-2")
        st.write(f"**Purpose:** Backup Archive")
    
    # Migration flow diagram
    st.markdown("#### 🔄 Backup Migration Flow")
    
    if config['source_database_engine'] == 'sqlserver':
        st.info("""
        **SQL Server Backup Migration Flow:**
        1. 🗃️ SQL Server creates backup files on Windows File Share (SMB/CIFS)
        2. 🤖 DataSync agents access backup files via SMB protocol
        3. 📤 DataSync transfers backup files to AWS S3 via HTTPS
        4. ☁️ S3 stores backup files with configured storage class
        5. 🔍 DataSync validates file integrity and provides transfer reports
        """)
    else:
        st.info("""
        **Oracle/PostgreSQL/MySQL Backup Migration Flow:**
        1. 🗃️ Database creates backup files on Linux NAS (NFS)
        2. 🤖 DataSync agents access backup files via NFS protocol
        3. 📤 DataSync transfers backup files to AWS S3 via HTTPS
        4. ☁️ S3 stores backup files with configured storage class
        5. 🔍 DataSync validates file integrity and provides transfer reports
        """)

def render_backup_performance_analysis(config: Dict, network_perf: Dict, agent_perf: Dict):
    """Render backup-specific performance analysis using native Streamlit components"""
    st.markdown("### 📊 Backup Transfer Performance Analysis")
    
    # Calculate transfer times
    backup_size_gb = config['backup_size_gb']
    throughput_mbps = min(network_perf['effective_bandwidth_mbps'], agent_perf['total_agent_throughput_mbps'])
    
    # Convert GB to Mb (Gigabytes to Megabits)
    backup_size_mb = backup_size_gb * 8 * 1000  # GB to Mb
    transfer_time_hours = backup_size_mb / (throughput_mbps * 3600)  # Hours
    transfer_time_minutes = transfer_time_hours * 60
    
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "📤 Effective Transfer Rate",
            f"{throughput_mbps:,.0f} Mbps",
            help="End-to-End Performance"
        )
    
    with col2:
        if transfer_time_hours < 1:
            time_display = f"{transfer_time_minutes:.0f} min"
        elif transfer_time_hours < 24:
            time_display = f"{transfer_time_hours:.1f} hrs"
        else:
            time_display = f"{transfer_time_hours/24:.1f} days"
        
        st.metric(
            "⏱️ Estimated Transfer Time",
            time_display,
            help=f"{backup_size_gb:,} GB backup"
        )
    
    with col3:
        # Calculate cost per GB
        transfer_cost = (agent_perf['total_monthly_cost'] / 730) * transfer_time_hours  # Hourly cost
        cost_per_gb = transfer_cost / backup_size_gb
        
        st.metric(
            "💰 Cost per GB",
            f"${cost_per_gb:.4f}",
            help="Transfer cost only"
        )
    
    with col4:
        # Network efficiency for backup files
        storage_efficiency = agent_perf['io_multiplier'] * agent_perf['backup_access_efficiency']
        
        st.metric(
            "🎯 Storage Efficiency",
            f"{storage_efficiency*100:.1f}%",
            help="Backup access optimization"
        )
    
    # Backup-specific insights
    st.markdown("#### 💡 Backup Transfer Insights")
    
    # Performance comparison based on database engine
    if config['source_database_engine'] == 'sqlserver':
        with st.container():
            st.warning("**🔵 SQL Server Backup Characteristics**")
            st.write(f"• **Storage Protocol:** SMB/CIFS on Windows File Share")
            st.write(f"• **Protocol Efficiency:** ~75-80% due to SMB overhead")
            st.write(f"• **Large File Performance:** SMB struggles with files >10GB")
            st.write(f"• **Recommendation:** Consider multiple smaller backup files or compression")
            st.write(f"• **Expected Performance:** {throughput_mbps:,.0f} Mbps ({storage_efficiency*100:.1f}% efficiency)")
    else:
        with st.container():
            st.success("**🐧 Linux Database Backup Characteristics**")
            st.write(f"• **Storage Protocol:** NFS on Linux NAS")
            st.write(f"• **Protocol Efficiency:** ~90-95% optimal for large files")
            st.write(f"• **Large File Performance:** NFS excels with sequential reads")
            st.write(f"• **Recommendation:** Optimize NFS mount options for large files")
            st.write(f"• **Expected Performance:** {throughput_mbps:,.0f} Mbps ({storage_efficiency*100:.1f}% efficiency)")

def render_s3_storage_optimization(config: Dict, agent_perf: Dict):
    """Render S3 storage class optimization for backup retention using native Streamlit components"""
    st.markdown("### ☁️ S3 Storage Optimization for Backup Retention")
    
    backup_size_gb = config['backup_size_gb']
    monthly_transfer_cost = agent_perf['total_monthly_cost']
    
    # S3 storage costs (per GB per month) - 2024 pricing
    storage_costs = {
        'standard': 0.023,
        'standard_ia': 0.0125,
        'glacier': 0.004,
        'deep_archive': 0.00099
    }
    
    # Calculate storage costs for different retention periods
    retention_periods = [1, 3, 6, 12, 24, 36]  # months
    
    storage_analysis = []
    for period in retention_periods:
        for storage_class, cost_per_gb in storage_costs.items():
            total_storage_cost = backup_size_gb * cost_per_gb * period
            retrieval_cost = 0  # Simplified - actual retrieval costs vary
            
            storage_analysis.append({
                'Storage Class': storage_class.replace('_', ' ').title(),
                'Retention (Months)': period,
                'Monthly Storage Cost': backup_size_gb * cost_per_gb,
                'Total Storage Cost': total_storage_cost,
                'Transfer Cost': monthly_transfer_cost,
                'Total Cost': total_storage_cost + monthly_transfer_cost
            })
    
    # Display current selection impact
    selected_class = config['s3_storage_class']
    selected_cost = storage_costs[selected_class]
    
    st.warning("**💰 Current S3 Configuration Impact**")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"• **Selected Storage Class:** {selected_class.replace('_', ' ').title()}")
        st.write(f"• **Storage Cost:** ${selected_cost:.4f} per GB per month")
        st.write(f"• **Monthly Storage Cost:** ${backup_size_gb * selected_cost:.2f} for {backup_size_gb:,} GB")
    with col2:
        st.write(f"• **Annual Storage Cost:** ${backup_size_gb * selected_cost * 12:.2f}")
        st.write(f"• **One-time Transfer Cost:** ${monthly_transfer_cost:.2f}")
    
    # Storage cost comparison chart
    df_storage = pd.DataFrame(storage_analysis)
    df_12_month = df_storage[df_storage['Retention (Months)'] == 12]
    
    fig_storage = px.bar(
        df_12_month,
        x='Storage Class',
        y='Total Cost',
        title=f'S3 Storage Cost Comparison (12-month retention, {backup_size_gb:,} GB)',
        color='Total Cost',
        color_continuous_scale='RdYlGn_r',
        text='Total Cost'
    )
    
    fig_storage.update_traces(
        texttemplate='$%{text:.0f}',
        textposition='outside'
    )
    
    fig_storage.update_layout(
        height=400,
        title=dict(font=dict(size=16)),
        xaxis=dict(title='S3 Storage Class'),
        yaxis=dict(title='Total Cost ($)')
    )
    
    st.plotly_chart(fig_storage, use_container_width=True)
    
    # Storage recommendations
    st.markdown("#### 💡 Storage Class Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("**📦 For Active Backups (0-3 months)**")
        st.write("• **S3 Standard:** Immediate access, higher cost")
        st.write("• **Use Case:** Recent backups, disaster recovery")
        st.write("• **Retrieval:** Instant, no additional cost")
        st.write("• **Best For:** Operational recovery scenarios")
    
    with col2:
        st.info("**🧊 For Archive Backups (>6 months)**")
        st.write("• **S3 Glacier:** Low cost, hours to retrieve")
        st.write("• **S3 Deep Archive:** Lowest cost, 12+ hours")
        st.write("• **Use Case:** Compliance, long-term retention")
        st.write("• **Best For:** Regulatory requirements")

def render_waterfall_analysis(network_perf: Dict):
    """Render detailed end-to-end waterfall bandwidth analysis using native Streamlit components"""
    st.subheader("🌊 End-to-End Network Waterfall Analysis")
    
    # Environment and path information
    env_type = "🏭 Production" if network_perf['environment'] == 'production' else "🔧 Non-Production"
    st.info(f"**Network Path:** {network_perf['path_name']}")
    st.write(f"**Environment:** {env_type} | **Database:** {network_perf['database_engine'].upper()} | **Storage:** {network_perf['backup_location']}")
    
    # Initialize waterfall analyzer
    waterfall_analyzer = WaterfallBandwidthAnalyzer()
    waterfall_analysis = waterfall_analyzer.analyze_waterfall(network_perf['segments'])
    
    # Create enhanced waterfall visualization with network layer icons
    fig = go.Figure()
    
    # Add waterfall bars with detailed labels
    x_labels = []
    y_values = []
    colors = []
    hover_texts = []
    
    # Network layer icons mapping
    layer_icons = {
        'os_layer': '💻',
        'nic_layer': '🔌',
        'lan_switch': '🔀',
        'network_link': '🌐',
        'router': '📡',
        'firewall': '🛡️',
        'private_line': '🚄',
        'internet_gateway': '🌍',
        'direct_connect': '⚡',
        'aws_service': '☁️'
    }
    
    for i, segment in enumerate(waterfall_analysis['waterfall_segments']):
        layer_icon = layer_icons.get(segment['connection_type'], '🔗')
        x_labels.append(f"{layer_icon} {i+1}")
        y_values.append(segment['cumulative_bandwidth'])
        colors.append(segment['status_color'])
        
        hover_text = (f"<b>{segment['segment_name']}</b><br>"
                     f"Individual: {segment['segment_bandwidth']:,.0f} Mbps<br>"
                     f"Cumulative: {segment['cumulative_bandwidth']:,.0f} Mbps<br>"
                     f"Latency: +{segment['latency_contribution']:.1f} ms<br>"
                     f"Status: {segment['bottleneck_status']}")
        hover_texts.append(hover_text)
    
    fig.add_trace(go.Bar(
        x=x_labels,
        y=y_values,
        marker_color=colors,
        text=[f"{val:,.0f}" for val in y_values],
        textposition='outside',
        name='Cumulative Bandwidth (Mbps)',
        hovertemplate='%{hovertext}<extra></extra>',
        hovertext=hover_texts
    ))
    
    fig.update_layout(
        title=f"End-to-End Network Bandwidth Waterfall: {network_perf['database_engine'].upper()} Backup Migration",
        xaxis_title="Network Segments (OS → NIC → LAN → Routers → Firewalls → DX/Internet → AWS)",
        yaxis_title="Cumulative Bandwidth (Mbps)",
        height=500,
        showlegend=False,
        plot_bgcolor='white',
        xaxis=dict(tickangle=45)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Network layer summary
    st.markdown("#### 🏗️ Network Architecture Overview")
    
    layer_summary = {}
    for segment in waterfall_analysis['waterfall_segments']:
        layer_type = segment['connection_type']
        if layer_type not in layer_summary:
            layer_summary[layer_type] = {
                'count': 0,
                'min_bandwidth': float('inf'),
                'total_latency': 0,
                'avg_reliability': []
            }
        layer_summary[layer_type]['count'] += 1
        layer_summary[layer_type]['min_bandwidth'] = min(layer_summary[layer_type]['min_bandwidth'], segment['segment_bandwidth'])
        layer_summary[layer_type]['total_latency'] += segment['latency_contribution']
        layer_summary[layer_type]['avg_reliability'].append(segment['reliability_impact'])
    
    # Display layer summary in columns
    layer_cols = st.columns(3)
    layer_names = {
        'os_layer': '💻 OS Layer',
        'nic_layer': '🔌 NIC Layer', 
        'lan_switch': '🔀 LAN Switches',
        'network_link': '🌐 Network Links',
        'router': '📡 Routers',
        'firewall': '🛡️ Firewalls',
        'private_line': '🚄 Private Lines',
        'internet_gateway': '🌍 Internet Gateway',
        'direct_connect': '⚡ Direct Connect',
        'aws_service': '☁️ AWS Services'
    }
    
    col_idx = 0
    for layer_type, summary in layer_summary.items():
        with layer_cols[col_idx % 3]:
            layer_name = layer_names.get(layer_type, layer_type.replace('_', ' ').title())
            avg_reliability = sum(summary['avg_reliability']) / len(summary['avg_reliability']) if summary['avg_reliability'] else 0
            
            st.metric(
                layer_name,
                f"{summary['min_bandwidth']:,.0f} Mbps",
                help=f"Segments: {summary['count']} | Latency: {summary['total_latency']:.1f}ms | Reliability: {avg_reliability*100:.2f}%"
            )
        col_idx += 1
    
    # Detailed segment analysis with network context
    st.markdown("#### 📋 Detailed End-to-End Segment Analysis")
    
    for i, segment in enumerate(waterfall_analysis['waterfall_segments']):
        with st.container():
            # Get layer icon and type
            layer_icon = layer_icons.get(segment['connection_type'], '🔗')
            layer_name = layer_names.get(segment['connection_type'], segment['connection_type'].replace('_', ' ').title())
            
            col1, col2, col3, col4, col5 = st.columns([4, 2, 2, 1.5, 1.5])
            
            with col1:
                if "Critical" in segment['bottleneck_status']:
                    st.error(f"**{layer_icon} Step {i+1}: {segment['segment_name']}**")
                    st.write(f"**Layer:** {layer_name}")
                    st.write(f"**Status:** {segment['bottleneck_status']}")
                elif "Concern" in segment['bottleneck_status']:
                    st.warning(f"**{layer_icon} Step {i+1}: {segment['segment_name']}**")
                    st.write(f"**Layer:** {layer_name}")
                    st.write(f"**Status:** {segment['bottleneck_status']}")
                else:
                    st.success(f"**{layer_icon} Step {i+1}: {segment['segment_name']}**")
                    st.write(f"**Layer:** {layer_name}")
                    st.write(f"**Status:** {segment['bottleneck_status']}")
            
            with col2:
                st.metric(
                    "Segment BW",
                    f"{segment['segment_bandwidth']:,.0f}",
                    help="Individual segment bandwidth (Mbps)"
                )
            
            with col3:
                st.metric(
                    "Cumulative BW", 
                    f"{segment['cumulative_bandwidth']:,.0f}",
                    help="Bandwidth after this segment (Mbps)"
                )
            
            with col4:
                st.metric(
                    "Latency",
                    f"{segment['latency_contribution']:.1f}ms",
                    help="Latency added by this segment"
                )
            
            with col5:
                st.metric(
                    "Reliability",
                    f"{segment['reliability_impact']*100:.2f}%",
                    help="Segment reliability factor"
                )
            
            # Show flow arrow except for last segment
            if i < len(waterfall_analysis['waterfall_segments']) - 1:
                st.markdown("**⬇️ Data Flow**")
    
    # Enhanced optimization recommendations with layer-specific advice
    optimization = waterfall_analysis['optimization_potential']
    
    st.markdown("#### 🎯 Layer-Specific Optimization Opportunities")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("**📊 Overall Performance Summary**")
        st.write(f"• **Final Bandwidth:** {optimization['current_bandwidth']:,.0f} Mbps")
        st.write(f"• **Maximum Possible:** {optimization['max_possible_bandwidth']:,.0f} Mbps")
        st.write(f"• **Improvement Potential:** {optimization['total_potential']:.1f}%")
        st.write(f"• **Critical Bottlenecks:** {waterfall_analysis['bottleneck_count']}")
        st.write(f"• **Total Network Segments:** {len(waterfall_analysis['waterfall_segments'])}")
        st.write(f"• **End-to-End Latency:** {waterfall_analysis['final_latency']:.1f} ms")
    
    with col2:
        if optimization['recommendations']:
            st.markdown("**🔧 Priority Optimization Targets:**")
            for rec in optimization['recommendations']:
                priority_color = "🔴" if rec['priority'] == 'High' else "🟡"
                # Find the layer type for this segment
                matching_segment = next((s for s in waterfall_analysis['waterfall_segments'] if s['segment_name'] == rec['segment']), None)
                if matching_segment:
                    layer_icon = layer_icons.get(matching_segment['connection_type'], '🔗')
                    st.write(f"{priority_color} {layer_icon} **{rec['segment']}**")
                    st.write(f"   └─ {rec['improvement_type']} ({rec['priority']} Priority)")
                else:
                    st.write(f"{priority_color} **{rec['segment']}** ({rec['priority']} Priority): {rec['improvement_type']}")
        else:
            st.success("✅ End-to-end network path is well optimized!")
            st.write("All network layers are performing within acceptable parameters.")
    
    # Environment-specific insights
    st.markdown("#### 💡 Environment-Specific Network Insights")
    
    if network_perf['environment'] == 'production':
        st.success("**🏭 Production Environment Characteristics**")
        st.write("• **Direct Connect:** Dedicated high-speed AWS connection")
        st.write("• **Redundant Path:** San Antonio → San Jose → AWS for reliability")
        st.write("• **Enhanced Security:** Multiple firewall layers and VPC isolation")
        st.write("• **Higher Reliability:** Enterprise-grade network infrastructure")
        st.write("• **Cost Trade-off:** Higher cost for premium performance and reliability")
    else:
        st.warning("**🔧 Non-Production Environment Characteristics**")
        st.write("• **Internet Gateway:** Cost-effective but shared bandwidth")
        st.write("• **Direct Path:** San Jose → AWS for simplicity")
        st.write("• **Standard Security:** Basic firewall protection")
        st.write("• **Cost Optimized:** Lower cost but potentially variable performance")
        st.write("• **Suitable for:** Development, testing, and non-critical backup transfers")

def main():
    """Enhanced main application with native Streamlit styling and ALL original features preserved"""
    
    # Header using native Streamlit components
    st.title("🗄️ AWS DataSync Database Backup Migration Analyzer")
    st.markdown("### Professional Database Backup Transfer Optimization • DataSync Agent Placement • End-to-End Waterfall Network Analysis • SQL Server & Oracle/PostgreSQL Support")
    
    # Initialize integrations
    if 'integrations_initialized' not in st.session_state:
        with st.spinner("🔄 Initializing API integrations..."):
            try:
                integration_status = initialize_integrations()
                st.session_state.update(integration_status)
                st.session_state['integrations_initialized'] = True
            except Exception as e:
                st.error(f"Error initializing integrations: {str(e)}")
                st.session_state.update({
                    'aws_integration': None,
                    'claude_integration': None,
                    'aws_status': '❌ Initialization Error',
                    'aws_message': str(e),
                    'claude_status': '❌ Initialization Error', 
                    'claude_message': str(e),
                    'integrations_initialized': True
                })

    # Get configuration
    config = render_enhanced_sidebar()
    
    # Render connection status
    render_connection_status({
        'aws_status': st.session_state.get('aws_status', '❌ Not Connected'),
        'aws_message': st.session_state.get('aws_message', 'Not initialized'),
        'claude_status': st.session_state.get('claude_status', '❌ Not Connected'),
        'claude_message': st.session_state.get('claude_message', 'Not initialized')
    })
    
    # Render backup migration overview
    render_backup_migration_overview(config)
    
    # Initialize managers
    network_manager = EnhancedNetworkPathManager()
    agent_manager = EnhancedAgentManager()
    
    # Get network path and performance
    path_key = network_manager.get_network_path_key(config)
    network_perf = network_manager.calculate_network_performance(path_key, config=config)
    
    # Storage characteristics based on database engine
    storage_type_mapping = {
        'windows_share': 'share',
        'linux_nas': 'nas'
    }
    storage_type = storage_type_mapping.get(config['storage_type'], 'nas')
    os_type = 'linux' if 'linux' in config['operating_system'].lower() else 'windows'
    
    # Get DataSync agent performance (default placement)
    agent_perf = agent_manager.calculate_agent_performance(
        config['agent_size'], config['number_of_agents'], 
        config['server_type'], storage_type, os_type, 'centralized_datacenter',
        config['backup_size_gb']
    )
    
    # Enhanced tabs with END-TO-END waterfall analysis tab FIRST
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🌊 End-to-End Waterfall Analysis",  # NOW FIRST with detailed analysis
        "🎯 DataSync Agent Placement",
        "📊 Backup Performance Analysis",
        "☁️ S3 Storage Optimization", 
        "🌐 Network Path Analysis",
        "☁️ AWS Integration",
        "🧠 AI Analysis"
    ])
    
    with tab1:
        # ENHANCED: End-to-End waterfall bandwidth analysis with full detail
        render_waterfall_analysis(network_perf)
    
    with tab2:
        st.subheader("🎯 DataSync Agent Placement for Backup Migration")
        
        # Agent placement analysis specific to backup scenarios
        claude_integration = st.session_state.get('claude_integration')
        
        # Initialize placement analyzer
        placement_analyzer = AgentPlacementAnalyzer()
        placement_options = placement_analyzer.analyze_placement_options(config, network_perf, agent_manager)
        
        # Display placement options with backup focus
        st.markdown("### 📍 DataSync Agent Placement Options for Backup Transfer")
        
        for i, option in enumerate(placement_options):
            is_recommended = i == 0
            
            # Determine status type for native component
            if option['placement_score'] >= 80:
                status_type = "success"
            elif option['placement_score'] >= 65:
                status_type = "warning"
            else:
                status_type = "error"
            
            with st.container():
                if status_type == "success":
                    st.success(f"**{option['strategy']['name']} {'⭐ RECOMMENDED' if is_recommended else ''}** - Score: {option['placement_score']:.1f}/100")
                elif status_type == "warning":
                    st.warning(f"**{option['strategy']['name']} {'⭐ RECOMMENDED' if is_recommended else ''}** - Score: {option['placement_score']:.1f}/100")
                else:
                    st.error(f"**{option['strategy']['name']} {'⭐ RECOMMENDED' if is_recommended else ''}** - Score: {option['placement_score']:.1f}/100")
                
                st.write(option['strategy']['description'])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**Backup Performance:** {option['throughput_mbps']:,.0f} Mbps")
                    st.write(f"Access Efficiency: {option['backup_access_efficiency']*100:.0f}%")
                
                with col2:
                    st.write(f"**Transfer Cost:** ${option['monthly_cost']:,.0f}/month")
                    st.write(f"${option['monthly_cost']/option['throughput_mbps']:.2f}/Mbps")
                
                with col3:
                    st.write(f"**Implementation:** {option['implementation_complexity']['setup_time_days']} days setup")
                    st.write(f"{option['implementation_complexity']['skill_level']} skill level")
            
            # Detailed backup considerations
            with st.expander(f"🔍 {option['strategy']['name']} - Backup-Specific Analysis"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Backup Transfer Advantages:**")
                    for pro in option['strategy']['pros']:
                        st.write(f"• {pro}")
                
                with col2:
                    st.markdown("**Backup Transfer Challenges:**")
                    for con in option['strategy']['cons']:
                        st.write(f"• {con}")
                
                st.write(f"**Backup Integration Details:**")
                st.write(f"- **Setup Time:** {option['implementation_complexity']['setup_time_days']} days")
                st.write(f"- **Backup Integration:** {option['implementation_complexity']['backup_integration']}")
                st.write(f"- **Backup Access Latency:** {option['latency_impact']:+.1f} ms")
                st.write(f"- **Storage Protocol Efficiency:** {option['agent_performance']['io_multiplier']*100:.1f}%")
        
        # AI-powered placement recommendations
        if claude_integration and claude_integration.client:
            st.markdown("### 🧠 AI-Powered Backup Migration Recommendations")
            
            try:
                with st.spinner("🔄 Analyzing DataSync placement for backup migration..."):
                    ai_recommendations = claude_integration.get_placement_recommendations(config, placement_options)
                
                st.info("**🎯 DataSync Agent Placement Strategy for Database Backups**")
                st.write(ai_recommendations)
            except Exception as e:
                st.warning(f"AI analysis error: {str(e)}")
    
    with tab3:
        st.subheader("📊 Database Backup Transfer Performance Analysis")
        
        # Render backup-specific performance analysis
        render_backup_performance_analysis(config, network_perf, agent_perf)
        
        # Protocol comparison for backup files
        st.markdown("#### 🔄 Backup Storage Protocol Comparison")
        
        if config['source_database_engine'] == 'sqlserver':
            st.warning("**🔵 SQL Server Backup on Windows Share (SMB/CIFS)**")
            st.write("**Current Configuration Analysis:**")
            st.write("• Protocol overhead reduces effective bandwidth by ~20-25%")
            st.write("• Large backup files (>10GB) experience additional SMB latency")
            st.write("• Windows file system metadata overhead")
            st.write("• Authentication and session management overhead")
            st.write("")
            st.write("**Optimization Recommendations:**")
            st.write("• Enable SMB3 multichannel if supported")
            st.write("• Increase SMB TCP window size")
            st.write("• Consider backup file compression")
            st.write("• Split large backups into smaller files if possible")
        else:
            st.success("**🐧 Oracle/PostgreSQL Backup on Linux NAS (NFS)**")
            st.write("**Current Configuration Analysis:**")
            st.write("• NFS optimized for large sequential file operations")
            st.write("• Minimal protocol overhead (~5-10% reduction)")
            st.write("• Excellent performance with large backup files")
            st.write("• Efficient client-side caching")
            st.write("")
            st.write("**Optimization Recommendations:**")
            st.write("• Use NFS v4.1 or higher for best performance")
            st.write("• Optimize rsize/wsize parameters (1MB+)")
            st.write("• Enable NFS client-side caching")
            st.write("• Consider dedicated backup network segment")
        
        # Transfer time analysis
        st.markdown("#### ⏱️ Backup Transfer Time Analysis")
        
        backup_sizes = [100, 500, 1000, 2000, 5000, 10000]  # GB
        transfer_scenarios = []
        
        for size in backup_sizes:
            size_mb = size * 8 * 1000  # Convert GB to Mb
            transfer_time = size_mb / (agent_perf['total_agent_throughput_mbps'] * 3600)  # Hours
            transfer_scenarios.append({
                'Backup Size (GB)': size,
                'Transfer Time (Hours)': transfer_time,
                'Transfer Time (Days)': transfer_time / 24
            })
        
        df_transfer = pd.DataFrame(transfer_scenarios)
        
        fig_transfer = px.line(
            df_transfer,
            x='Backup Size (GB)',
            y='Transfer Time (Hours)',
            title=f'DataSync Transfer Time vs Backup Size ({agent_perf["total_agent_throughput_mbps"]:,.0f} Mbps)',
            markers=True
        )
        
        fig_transfer.update_layout(
            height=400,
            xaxis_title="Backup Size (GB)",
            yaxis_title="Transfer Time (Hours)"
        )
        
        st.plotly_chart(fig_transfer, use_container_width=True)
    
    with tab4:
        st.subheader("☁️ S3 Storage Optimization for Database Backups")
        
        # Render S3 storage optimization
        render_s3_storage_optimization(config, agent_perf)
        
        # Backup lifecycle recommendations
        st.markdown("#### 🔄 Backup Lifecycle Management")
        
        lifecycle_col1, lifecycle_col2 = st.columns(2)
        
        with lifecycle_col1:
            st.success("**📅 Recommended Backup Lifecycle**")
            st.write("**0-30 days:** S3 Standard")
            st.write("• Immediate access for recovery")
            st.write("• Full restore capabilities")
            st.write("• Point-in-time recovery")
            st.write("")
            st.write("**30-90 days:** S3 Standard-IA")
            st.write("• Infrequent access, lower cost")
            st.write("• Quick retrieval when needed")
            st.write("• Compliance requirements")
            st.write("")
            st.write("**90+ days:** S3 Glacier")
            st.write("• Long-term archival")
            st.write("• Regulatory compliance")
            st.write("• Cost-effective retention")
        
        with lifecycle_col2:
            st.info(f"**💰 Cost Impact Analysis ({config['backup_size_gb']:,} GB)**")
            st.write(f"**Current Setup:**")
            st.write(f"• Transfer Cost: ${agent_perf['total_monthly_cost']:.2f} (one-time)")
            st.write(f"• S3 Standard (1 year): ${config['backup_size_gb'] * 0.023 * 12:.2f}")
            st.write(f"• S3 Glacier (1 year): ${config['backup_size_gb'] * 0.004 * 12:.2f}")
            st.write("")
            st.write(f"**Annual Savings with Lifecycle:**")
            st.write(f"• Standard to Glacier: ${config['backup_size_gb'] * (0.023 - 0.004) * 12:.2f}")
            st.write(f"• ROI: {((config['backup_size_gb'] * (0.023 - 0.004) * 12) / agent_perf['total_monthly_cost'] * 100):.0f}% of transfer cost")
    
    with tab5:
        st.subheader("🌐 Database Backup Network Path Analysis")
        
        # Network path visualization specific to backup flows
        st.markdown("#### 🗺️ Backup Transfer Network Path")
        
        # Display network segments with backup context
        for i, segment in enumerate(network_perf['segments']):
            with st.expander(f"Segment {i+1}: {segment['name']}", expanded=True):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("**Performance Metrics:**")
                    st.write(f"- Bandwidth: {segment['effective_bandwidth_mbps']:,.0f} Mbps")
                    st.write(f"- Latency: {segment['effective_latency_ms']:.1f} ms")
                    st.write(f"- Reliability: {segment['reliability']*100:.2f}%")
                
                with col2:
                    st.write("**Protocol Details:**")
                    st.write(f"- Connection: {segment['connection_type'].replace('_', ' ').title()}")
                    st.write(f"- Efficiency: {segment.get('protocol_efficiency', 1.0)*100:.1f}%")
                    st.write(f"- Congestion: {segment.get('congestion_factor', 1.0):.2f}x")
                
                with col3:
                    st.write("**Backup Impact:**")
                    st.write(f"- Optimization: {segment['optimization_potential']*100:.1f}%")
                    st.write(f"- Cost Factor: {segment['cost_factor']:.1f}x")
                    st.write(f"- Type: {'Storage Access' if 'Backup' in segment['name'] else 'Network Transit'}")
        
        # Overall network performance
        st.markdown("#### 📊 Network Performance Summary")
        
        perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
        
        with perf_col1:
            st.metric("Total Latency", f"{network_perf['total_latency_ms']:.1f} ms")
        
        with perf_col2:
            st.metric("Effective Bandwidth", f"{network_perf['effective_bandwidth_mbps']:,.0f} Mbps")
        
        with perf_col3:
            st.metric("Network Reliability", f"{network_perf['total_reliability']*100:.2f}%")
        
        with perf_col4:
            st.metric("Quality Score", f"{network_perf['network_quality_score']:.1f}/100")
    
    with tab6:
        st.subheader("☁️ AWS DataSync Integration")
        
        aws_integration = st.session_state.get('aws_integration')
        if aws_integration and aws_integration.session:
            # Real-time DataSync status
            st.markdown("**📊 Current DataSync Tasks**")
            datasync_tasks = aws_integration.get_datasync_tasks()
            
            if datasync_tasks:
                for task in datasync_tasks:
                    with st.expander(f"DataSync Task: {task['name']}", expanded=False):
                        st.write(f"**Task Details:**")
                        st.write(f"- Status: {task['status']}")
                        st.write(f"- Source: {task['source_location'].split('/')[-1] if task['source_location'] != 'Unknown' else 'Unknown'}")
                        st.write(f"- Destination: {task['destination_location'].split('/')[-1] if task['destination_location'] != 'Unknown' else 'Unknown'}")
                        st.write(f"- Executions: {len(task['executions'])} recent runs")
            else:
                st.info("No DataSync tasks found. This tool will help you plan your backup migration configuration.")
            
            # CloudWatch metrics for backup transfers
            st.markdown("**📈 DataSync Metrics (Last 24 Hours)**")
            datasync_metrics = aws_integration.get_cloudwatch_metrics('datasync')
            
            if datasync_metrics.get('BytesTransferred'):
                bytes_data = datasync_metrics['BytesTransferred']
                if bytes_data:
                    st.line_chart(pd.DataFrame(bytes_data).set_index('Timestamp')['Average'])
                else:
                    st.info("No recent DataSync transfer metrics available")
            else:
                st.info("No DataSync metrics available for the selected region")
        else:
            st.warning("AWS integration not connected. Configure AWS credentials to see real-time DataSync information.")
    
    with tab7:
        st.subheader("🧠 AI-Powered Backup Migration Analysis")
        
        claude_integration = st.session_state.get('claude_integration')
        if claude_integration and claude_integration.client:
            try:
                with st.spinner("🔄 Analyzing database backup migration configuration..."):
                    analysis = claude_integration.analyze_migration_performance(
                        config, network_perf, agent_perf, {}, {}
                    )
                
                st.info("**🧠 Comprehensive Database Backup Migration Analysis**")
                st.write(analysis)
                
                # Get specific optimization recommendations
                if network_perf['effective_bandwidth_mbps'] < agent_perf['total_agent_throughput_mbps']:
                    bottleneck_type = "network"
                else:
                    bottleneck_type = "agent"
                
                with st.expander("🎯 Specific Backup Migration Optimizations", expanded=False):
                    with st.spinner("🔄 Getting backup-specific recommendations..."):
                        recommendations = claude_integration.get_optimization_recommendations(
                            bottleneck_type, config
                        )
                    
                    st.info(f"**🔧 {bottleneck_type.title()} Optimization for Backup Transfer**")
                    st.write(recommendations)
                
            except Exception as e:
                st.warning(f"AI analysis error: {str(e)}")
        else:
            st.info("Claude AI integration not connected. Configure Claude API key for intelligent backup migration analysis.")
    
    # Executive Summary for Backup Migration (preserved and enhanced)
    st.markdown("---")
    st.markdown("### 🎯 Database Backup Migration Executive Summary")
    
    # Calculate key metrics
    placement_analyzer = AgentPlacementAnalyzer()
    placement_options = placement_analyzer.analyze_placement_options(config, network_perf, agent_manager)
    recommended_option = placement_options[0]
    
    final_throughput = recommended_option['throughput_mbps']
    backup_size_gb = config['backup_size_gb']
    backup_size_mb = backup_size_gb * 8 * 1000
    transfer_time_hours = backup_size_mb / (final_throughput * 3600)
    
    summary_col1, summary_col2, summary_col3 = st.columns(3)
    
    with summary_col1:
        st.success("**🗄️ Backup Migration Plan**")
        st.write(f"**Source:** {config['source_database_engine'].upper()}")
        st.write(f"**Backup Size:** {backup_size_gb:,} GB")
        st.write(f"**Storage:** {config['backup_storage_description']}")
        st.write(f"**Transfer Time:** {transfer_time_hours:.1f} hours")
        st.write(f"**Agent Strategy:** {recommended_option['strategy']['name']}")
    
    with summary_col2:
        st.warning("**⚡ Performance Summary**")
        st.write(f"**Transfer Rate:** {final_throughput:,.0f} Mbps")
        st.write(f"**DataSync Agents:** {config['number_of_agents']}x {config['agent_size'].title()}")
        st.write(f"**Platform:** {config['server_type'].title()}")
        st.write(f"**Efficiency Score:** {recommended_option['placement_score']:.1f}/100")
        st.write(f"**Backup Access:** {recommended_option['backup_access_efficiency']*100:.0f}%")
    
    with summary_col3:
        transfer_cost = (agent_perf['total_monthly_cost'] / 730) * transfer_time_hours
        storage_cost_annual = backup_size_gb * 0.023 * 12  # S3 Standard
        
        st.error("**💰 Cost Analysis**", icon="💰")
        st.write(f"**Transfer Cost:** ${transfer_cost:.2f} (one-time)")
        st.write(f"**Monthly Agent Cost:** ${agent_perf['total_monthly_cost']:,.0f}")
        st.write(f"**S3 Storage (Annual):** ${storage_cost_annual:.2f}")
        st.write(f"**Cost per GB:** ${transfer_cost/backup_size_gb:.4f}")
        st.write(f"**Setup Time:** {recommended_option['implementation_complexity']['setup_time_days']} days")

if __name__ == "__main__":
    main()