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
    page_title="AWS DataSync Database Migration Analyzer",
    page_icon="🗄️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS styling with database migration specific components
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 50%, #1d4ed8 100%);
        padding: 2rem;
        border-radius: 8px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(30,58,138,0.2);
    }
    
    .database-card {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        padding: 1.5rem;
        border-radius: 8px;
        color: #1f2937;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        border-left: 4px solid #22c55e;
        font-size: 14px;
        line-height: 1.6;
    }
    
    .backup-card {
        background: linear-gradient(135deg, #f0f9ff 0%, #dbeafe 100%);
        padding: 1.5rem;
        border-radius: 8px;
        color: #1f2937;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        border-left: 4px solid #3b82f6;
        font-size: 14px;
        line-height: 1.6;
    }
    
    .network-card {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        padding: 1.5rem;
        border-radius: 8px;
        color: #1f2937;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        border-left: 4px solid #22c55e;
        font-size: 14px;
        line-height: 1.6;
    }
    
    .performance-card {
        background: linear-gradient(135deg, #fef7f0 0%, #fed7aa 100%);
        padding: 1.5rem;
        border-radius: 8px;
        color: #1f2937;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        border-left: 4px solid #f97316;
        font-size: 14px;
        line-height: 1.6;
    }
    
    .agent-card {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        padding: 1.5rem;
        border-radius: 8px;
        color: #1f2937;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        border-left: 4px solid #64748b;
        font-size: 14px;
        line-height: 1.6;
    }
    
    
    .decision-matrix {
        background: linear-gradient(135deg, #f0f9ff 0%, #dbeafe 100%);
        padding: 1.5rem;
        border-radius: 8px;
        color: #1f2937;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        border-left: 4px solid #3b82f6;
        font-size: 14px;
        line-height: 1.6;
    }
    
    .warning-card {
        background: linear-gradient(135deg, #fefce8 0%, #fef3c7 100%);
        padding: 1.5rem;
        border-radius: 8px;
        color: #1f2937;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        border-left: 4px solid #f59e0b;
        font-size: 14px;
        line-height: 1.6;
    }

    .professional-ai-card {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        padding: 2rem;
        border-radius: 10px;
        color: #1e293b;
        margin: 1rem 0;
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
        border-left: 5px solid #3b82f6;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 15px;
        line-height: 1.7;
    }

    .aws-card {
        background: linear-gradient(135deg, #fef3c7 0%, #f59e0b 100%);
        padding: 1.5rem;
        border-radius: 8px;
        color: #1f2937;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        border-left: 4px solid #d97706;
        font-size: 14px;
        line-height: 1.6;
    }

    .connection-status {
        padding: 0.75rem 1.25rem;
        border-radius: 6px;
        margin: 0.75rem 0;
        font-weight: 600;
        font-size: 13px;
    }
    
    .status-connected {
        background-color: #d1fae5;
        color: #065f46;
        border: 2px solid #10b981;
    }
    
    .status-disconnected {
        background-color: #fee2e2;
        color: #991b1b;
        border: 2px solid #ef4444;
    }
    
    .status-partial {
        background-color: #fef3c7;
        color: #92400e;
        border: 2px solid #f59e0b;
    }

    .network-segment-card {
        background: white;
        border: 2px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.25rem;
        margin: 0.75rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        font-size: 14px;
        line-height: 1.6;
    }

    .segment-performance {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 0.75rem;
        font-size: 13px;
        font-weight: 500;
    }

    .ai-section {
        background: white;
        border: 2px solid #e2e8f0;
        border-radius: 10px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
        font-size: 15px;
        line-height: 1.7;
    }

    .ai-section h4 {
        color: #1e293b;
        font-weight: 700;
        margin-bottom: 1.5rem;
        padding-bottom: 0.75rem;
        border-bottom: 3px solid #f1f5f9;
        font-size: 18px;
    }

    .ai-section p, .ai-section ul, .ai-section ol {
        color: #475569;
        line-height: 1.8;
        margin-bottom: 1rem;
        font-size: 15px;
    }

    .ai-section ul li {
        margin-bottom: 0.6rem;
        font-size: 14px;
    }

    .ai-highlight {
        background: #f8fafc;
        border-left: 4px solid #3b82f6;
        padding: 1.5rem;
        margin: 1.5rem 0;
        border-radius: 0 8px 8px 0;
        font-size: 15px;
    }

    .metric-container {
        background: white;
        border: 2px solid #e5e7eb;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
    }

    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }

    .metric-label {
        font-size: 14px;
        color: #6b7280;
        font-weight: 500;
    }


    .sql-server-card {
        background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%);
        padding: 1.5rem;
        border-radius: 8px;
        color: #1f2937;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        border-left: 4px solid #0ea5e9;
        font-size: 14px;
        line-height: 1.6;
    }

    .linux-db-card {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        padding: 1.5rem;
        border-radius: 8px;
        color: #1f2937;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        border-left: 4px solid #22c55e;
        font-size: 14px;
        line-height: 1.6;
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
    
    def analyze_database_backup_migration(self, config: Dict, network_perf: Dict, 
                                        agent_perf: Dict, placement_analysis: Dict = None, 
                                        aws_data: Dict = None) -> str:
        """Get Claude AI analysis of database backup migration performance including agent placement"""
        if not self.client:
            return "Claude AI not initialized"
        
        try:
            # Prepare enhanced context for Claude
            context = f"""
            Database Backup Migration Analysis:
            
            Database Configuration:
            - Source Database: {config['source_database_engine'].upper()}
            - Backup Size: {config['backup_size_gb']} GB
            - Backup Type: {config['backup_type']}
            - Compression: {config['backup_compression']}
            - Operating System: {config['operating_system']}
            - Storage Type: {config['storage_type']}
            
            Hardware Configuration:
            - Server Type: {config['server_type']}
            - RAM: {config['ram_gb']} GB
            - CPU Cores: {config['cpu_cores']}
            - NIC: {config['nic_speed']} Mbps {config['nic_type']}
            
            Network Performance:
            - Path: {network_perf['path_name']}
            - Effective Bandwidth: {network_perf['effective_bandwidth_mbps']:.0f} Mbps
            - Total Latency: {network_perf['total_latency_ms']:.1f} ms
            - Reliability: {network_perf['total_reliability']*100:.2f}%
            - Quality Score: {network_perf['network_quality_score']:.1f}/100
            - Storage Protocol: {network_perf.get('storage_mount_type', 'Unknown').upper()}
            
            DataSync Agent Performance:
            - Agent Count: {agent_perf['num_agents']}
            - Agent Size: {agent_perf['agent_size']}
            - Total Capacity: {agent_perf['total_agent_throughput_mbps']:.0f} Mbps
            - Monthly Cost: ${agent_perf['total_monthly_cost']:.0f}
            - Platform Efficiency: {agent_perf['platform_efficiency']*100:.1f}%
            
            Migration Details:
            - Environment: {config['environment']}
            - Migration Strategy: DataSync backup file transfer
            - Target: AWS S3
            
            {f"Agent Placement Analysis: {json.dumps(placement_analysis, indent=2)}" if placement_analysis else "No placement analysis provided"}
            
            Database-Specific Performance Considerations:
            - SQL Server backup files (.bak) typically achieve good compression ratios
            - Oracle backup files (.dmp) with EXPDP can be highly compressed
            - PostgreSQL backup files (.sql/.dump) benefit from gzip compression
            - Linux-based database backups on NAS typically perform 20-25% better than Windows SMB
            - Large backup files (>100GB) benefit significantly from parallel DataSync agents
            
            {f"AWS Real-time Data: {json.dumps(aws_data, indent=2)}" if aws_data else "No real-time AWS data available"}
            """
            
            prompt = f"""
            As an AWS database migration expert specializing in DataSync backup file transfers, analyze this configuration and provide a structured analysis with the following sections:

            1. DATABASE BACKUP MIGRATION STRATEGY ANALYSIS
            2. DATASYNC AGENT PLACEMENT FOR BACKUP FILES  
            3. STORAGE PROTOCOL OPTIMIZATION FOR DATABASE BACKUPS
            4. BACKUP FILE SIZE AND COMPRESSION IMPACT
            5. NETWORK PERFORMANCE OPTIMIZATION
            6. COST OPTIMIZATION FOR BACKUP TRANSFERS
            7. IMPLEMENTATION TIMELINE AND RISK ASSESSMENT

            Focus on database backup file migration scenarios:
            - How backup file characteristics affect DataSync performance
            - Optimal agent placement for accessing backup storage
            - Storage protocol efficiency for large backup files
            - Compression impact on transfer times
            - Best practices for SQL Server backups on Windows Share vs Oracle/Postgres backups on Linux NAS

            Configuration to analyze:
            {context}

            Provide specific technical recommendations for DataSync agent configuration, backup file handling, 
            and infrastructure optimization for database backup migrations to S3.
            """
            
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.content[0].text
            
        except Exception as e:
            return f"Error getting Claude AI analysis: {str(e)}"
    
    def get_database_specific_recommendations(self, database_type: str, current_config: Dict) -> str:
        """Get database-specific optimization recommendations"""
        if not self.client:
            return "Claude AI not initialized"
        
        try:
            prompt = f"""
            As an AWS migration specialist, provide specific recommendations for migrating {database_type.upper()} database backups using DataSync.
            
            Current configuration:
            - Database: {database_type.upper()}
            - Platform: {current_config['server_type']}
            - OS: {current_config['operating_system']}
            - Storage: {current_config.get('storage_type', 'Unknown')}
            - Backup Size: {current_config.get('backup_size_gb', 'Unknown')} GB
            
            Provide 5-7 specific, actionable recommendations for optimizing {database_type.upper()} backup file transfers to S3.
            Include expected performance improvements and implementation steps.
            Focus on backup file characteristics, compression options, and DataSync configuration.
            Format as clear bullet points with technical details.
            """
            
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=800,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.content[0].text
            
        except Exception as e:
            return f"Error getting database-specific recommendations: {str(e)}"
    
    def get_placement_recommendations(self, config: Dict, placement_options: List[Dict]) -> str:
        """Get specific agent placement recommendations for database backup migrations"""
        if not self.client:
            return "Claude AI not initialized"
        
        try:
            prompt = f"""
            As an AWS migration specialist, analyze these DataSync agent placement options for database backup file migration:
            
            Current Configuration:
            - Database: {config['source_database_engine'].upper()}
            - Backup Type: {config.get('backup_type', 'Full backup')}
            - Environment: {config['environment']}
            - Platform: {config['server_type']}
            - OS: {config['operating_system']}
            - Backup Size: {config.get('backup_size_gb', 'Unknown')} GB
            - Storage: {config.get('storage_type', 'Unknown')}
            
            Placement Options Analysis:
            {json.dumps(placement_options, indent=2)}
            
            Provide:
            1. Recommended placement strategy for database backup file access
            2. Storage protocol optimization recommendations
            3. Backup file handling considerations
            4. Performance expectations for large backup files
            5. Security considerations for database backup access
            
            Format as clear bullet points with technical details specific to database backup scenarios.
            """
            
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.content[0].text
            
        except Exception as e:
            return f"Error getting placement recommendations: {str(e)}"

class DatabaseBackupNetworkPathManager:
    """Enhanced network path manager for database backup migrations"""
    
    def __init__(self):
        self.network_paths = {
            'nonprod_sj_sqlserver_backup_s3': {
                'name': 'Non-Prod: San Jose SQL Server Backup (Windows Share) → AWS S3',
                'destination_storage': 'S3',
                'source': 'San Jose',
                'destination': 'AWS US-West-2 S3',
                'environment': 'non-production',
                'database_type': 'sqlserver',
                'os_type': 'windows',
                'storage_type': 'share',
                'storage_mount_type': 'smb',
                'backup_location': 'windows_share_drive',
                'segments': [
                    {
                        'name': 'SQL Server Backup Files → Windows Share Drive (SMB)',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 2.5,
                        'reliability': 0.995,
                        'connection_type': 'internal_lan',
                        'cost_factor': 0.0,
                        'optimization_potential': 0.85,
                        'protocol_efficiency': 0.78,
                        'backup_file_impact': 1.2  # Backup files are larger, more impact
                    },
                    {
                        'name': 'Windows Share to DataSync Agent (SMB)',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 1.5,
                        'reliability': 0.995,
                        'connection_type': 'internal_lan',
                        'cost_factor': 0.0,
                        'optimization_potential': 0.82,
                        'protocol_efficiency': 0.75
                    },
                    {
                        'name': 'DataSync Agent to AWS S3 (DX)',
                        'bandwidth_mbps': 2000,
                        'latency_ms': 18,
                        'reliability': 0.998,
                        'connection_type': 'direct_connect',
                        'cost_factor': 2.0,
                        'optimization_potential': 0.88
                    }
                ]
            },
            'nonprod_sj_oracle_backup_s3': {
                'name': 'Non-Prod: San Jose Oracle Backup (Linux NAS) → AWS S3',
                'destination_storage': 'S3',
                'source': 'San Jose',
                'destination': 'AWS US-West-2 S3',
                'environment': 'non-production',
                'database_type': 'oracle',
                'os_type': 'linux',
                'storage_type': 'nas',
                'storage_mount_type': 'nfs',
                'backup_location': 'linux_nas_drive',
                'segments': [
                    {
                        'name': 'Oracle Backup Files → Linux NAS (NFS)',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 1.2,
                        'reliability': 0.9995,
                        'connection_type': 'internal_lan',
                        'cost_factor': 0.0,
                        'optimization_potential': 0.97,
                        'protocol_efficiency': 0.95,
                        'backup_file_impact': 1.0  # Oracle dumps compress well
                    },
                    {
                        'name': 'Linux NAS to DataSync Agent (NFS)',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 1.0,
                        'reliability': 0.9995,
                        'connection_type': 'internal_lan',
                        'cost_factor': 0.0,
                        'optimization_potential': 0.94,
                        'protocol_efficiency': 0.92
                    },
                    {
                        'name': 'DataSync Agent to AWS S3 (DX)',
                        'bandwidth_mbps': 2000,
                        'latency_ms': 12,
                        'reliability': 0.998,
                        'connection_type': 'direct_connect',
                        'cost_factor': 2.0,
                        'optimization_potential': 0.94
                    }
                ]
            },
            'nonprod_sj_postgres_backup_s3': {
                'name': 'Non-Prod: San Jose PostgreSQL Backup (Linux NAS) → AWS S3',
                'destination_storage': 'S3',
                'source': 'San Jose',
                'destination': 'AWS US-West-2 S3',
                'environment': 'non-production',
                'database_type': 'postgresql',
                'os_type': 'linux',
                'storage_type': 'nas',
                'storage_mount_type': 'nfs',
                'backup_location': 'linux_nas_drive',
                'segments': [
                    {
                        'name': 'PostgreSQL Backup Files → Linux NAS (NFS)',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 1.0,
                        'reliability': 0.9995,
                        'connection_type': 'internal_lan',
                        'cost_factor': 0.0,
                        'optimization_potential': 0.98,
                        'protocol_efficiency': 0.96,
                        'backup_file_impact': 0.9  # Text-based dumps compress very well
                    },
                    {
                        'name': 'Linux NAS to DataSync Agent (NFS)',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 0.8,
                        'reliability': 0.9995,
                        'connection_type': 'internal_lan',
                        'cost_factor': 0.0,
                        'optimization_potential': 0.95,
                        'protocol_efficiency': 0.93
                    },
                    {
                        'name': 'DataSync Agent to AWS S3 (DX)',
                        'bandwidth_mbps': 2000,
                        'latency_ms': 12,
                        'reliability': 0.998,
                        'connection_type': 'direct_connect',
                        'cost_factor': 2.0,
                        'optimization_potential': 0.94
                    }
                ]
            },
            'prod_sa_sqlserver_backup_s3': {
                'name': 'Prod: San Antonio SQL Server Backup (Windows Share) → San Jose → AWS S3',
                'destination_storage': 'S3',
                'source': 'San Antonio',
                'destination': 'AWS US-West-2 Production VPC S3',
                'environment': 'production',
                'database_type': 'sqlserver',
                'os_type': 'windows',
                'storage_type': 'share',
                'storage_mount_type': 'smb',
                'backup_location': 'windows_share_drive',
                'segments': [
                    {
                        'name': 'SQL Server Backup Files → Windows Share Drive (SMB)',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 3.5,
                        'reliability': 0.996,
                        'connection_type': 'internal_lan',
                        'cost_factor': 0.0,
                        'optimization_potential': 0.84,
                        'protocol_efficiency': 0.76,
                        'backup_file_impact': 1.3
                    },
                    {
                        'name': 'Windows Share to DataSync Agent (SMB)',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 2.0,
                        'reliability': 0.996,
                        'connection_type': 'internal_lan',
                        'cost_factor': 0.0,
                        'optimization_potential': 0.80,
                        'protocol_efficiency': 0.73
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
            },
            'prod_sa_oracle_backup_s3': {
                'name': 'Prod: San Antonio Oracle Backup (Linux NAS) → San Jose → AWS S3',
                'destination_storage': 'S3',
                'source': 'San Antonio',
                'destination': 'AWS US-West-2 Production VPC S3',
                'environment': 'production',
                'database_type': 'oracle',
                'os_type': 'linux',
                'storage_type': 'nas',
                'storage_mount_type': 'nfs',
                'backup_location': 'linux_nas_drive',
                'segments': [
                    {
                        'name': 'Oracle Backup Files → Linux NAS (NFS)',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 0.8,
                        'reliability': 0.9998,
                        'connection_type': 'internal_lan',
                        'cost_factor': 0.0,
                        'optimization_potential': 0.98,
                        'protocol_efficiency': 0.96,
                        'backup_file_impact': 1.0
                    },
                    {
                        'name': 'Linux NAS to DataSync Agent (NFS)',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 0.6,
                        'reliability': 0.9998,
                        'connection_type': 'internal_lan',
                        'cost_factor': 0.0,
                        'optimization_potential': 0.96,
                        'protocol_efficiency': 0.94
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
            'prod_sa_postgres_backup_s3': {
                'name': 'Prod: San Antonio PostgreSQL Backup (Linux NAS) → San Jose → AWS S3',
                'destination_storage': 'S3',
                'source': 'San Antonio',
                'destination': 'AWS US-West-2 Production VPC S3',
                'environment': 'production',
                'database_type': 'postgresql',
                'os_type': 'linux',
                'storage_type': 'nas',
                'storage_mount_type': 'nfs',
                'backup_location': 'linux_nas_drive',
                'segments': [
                    {
                        'name': 'PostgreSQL Backup Files → Linux NAS (NFS)',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 0.6,
                        'reliability': 0.9998,
                        'connection_type': 'internal_lan',
                        'cost_factor': 0.0,
                        'optimization_potential': 0.99,
                        'protocol_efficiency': 0.97,
                        'backup_file_impact': 0.85
                    },
                    {
                        'name': 'Linux NAS to DataSync Agent (NFS)',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 0.5,
                        'reliability': 0.9998,
                        'connection_type': 'internal_lan',
                        'cost_factor': 0.0,
                        'optimization_potential': 0.97,
                        'protocol_efficiency': 0.95
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
            }
        }
    
    def get_network_path_key(self, config: Dict) -> str:
        """Get network path key based on database and configuration"""
        database_type = config['source_database_engine'].lower()
        environment = config['environment']
        
        # Map database types to backup storage patterns
        if database_type == 'sqlserver':
            backup_pattern = 'sqlserver_backup'
        elif database_type in ['oracle', 'postgresql']:
            backup_pattern = f"{database_type}_backup"
        else:
            # Default to postgres pattern for other databases
            backup_pattern = 'postgres_backup'
        
        if environment == 'non-production':
            return f'nonprod_sj_{backup_pattern}_s3'
        else:
            return f'prod_sa_{backup_pattern}_s3'
    
    def calculate_network_performance(self, path_key: str, time_of_day: int = None, 
                                    backup_compression: float = 1.0) -> Dict:
        """Calculate network performance with backup file considerations"""
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
            
            # Apply backup file impact (larger files, different I/O patterns)
            backup_file_impact = segment.get('backup_file_impact', 1.0)
            effective_bandwidth = effective_bandwidth / backup_file_impact
            
            # Apply compression impact (compressed backup files transfer faster)
            effective_bandwidth = effective_bandwidth * backup_compression
            
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
            
            # Storage type specific adjustments for backup files
            if path['storage_mount_type'] == 'smb':
                effective_bandwidth *= 0.82  # SMB overhead for large backup files
                effective_latency *= 1.3
            elif path['storage_mount_type'] == 'nfs':
                effective_bandwidth *= 0.96  # NFS is more efficient for large files
                effective_latency *= 1.05
            
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
                'protocol_efficiency': protocol_efficiency,
                'backup_file_impact': backup_file_impact
            })
        
        # Calculate quality scores with backup file considerations
        latency_score = max(0, 100 - (total_latency * 1.5))
        bandwidth_score = min(100, (min_bandwidth / 1000) * 15)
        reliability_score = total_reliability * 100
        
        network_quality = (latency_score * 0.25 + bandwidth_score * 0.45 + reliability_score * 0.30)
        
        result = {
            'path_name': path['name'],
            'destination_storage': path['destination_storage'],
            'environment': path['environment'],
            'database_type': path['database_type'],
            'os_type': path['os_type'],
            'storage_mount_type': path['storage_mount_type'],
            'backup_location': path['backup_location'],
            'total_latency_ms': total_latency,
            'effective_bandwidth_mbps': min_bandwidth,
            'total_reliability': total_reliability,
            'network_quality_score': network_quality,
            'optimization_potential': (1 - optimization_score) * 100,
            'total_cost_factor': total_cost_factor,
            'segments': adjusted_segments
        }
        
        return result

class EnhancedDataSyncAgentManager:
    """Enhanced DataSync agent manager focused on backup file transfers"""
    
    def __init__(self):
        # Updated AWS DataSync pricing (per hour) as of 2024
        self.datasync_specs = {
            'small': {'throughput_mbps': 250, 'vcpu': 2, 'memory': 4, 'cost_hour': 0.0416, 'backup_efficiency': 0.85},
            'medium': {'throughput_mbps': 500, 'vcpu': 2, 'memory': 8, 'cost_hour': 0.0832, 'backup_efficiency': 0.90},
            'large': {'throughput_mbps': 1000, 'vcpu': 4, 'memory': 16, 'cost_hour': 0.1664, 'backup_efficiency': 0.93},
            'xlarge': {'throughput_mbps': 2000, 'vcpu': 8, 'memory': 32, 'cost_hour': 0.3328, 'backup_efficiency': 0.95}
        }
        
        # Physical vs VMware performance characteristics for backup transfers
        self.platform_characteristics = {
            'physical': {
                'base_efficiency': 1.0,
                'cpu_overhead': 0.0,
                'memory_overhead': 0.0,
                'io_efficiency': 1.0,
                'network_efficiency': 1.0,
                'backup_file_handling': 1.0
            },
            'vmware': {
                'base_efficiency': 0.92,
                'cpu_overhead': 0.08,
                'memory_overhead': 0.12,
                'io_efficiency': 0.88,
                'network_efficiency': 0.94,
                'backup_file_handling': 0.85  # VMware has more overhead with large backup files
            }
        }

        # Agent placement characteristics for backup file access
        self.placement_characteristics = {
            'backup_source_colocation': {
                'latency_reduction_ms': -3.0,
                'bandwidth_bonus': 1.20,
                'reliability_bonus': 1.03,
                'security_score': 0.80,
                'cost_multiplier': 1.0,
                'management_complexity': 0.7,
                'backup_access_efficiency': 1.15
            },
            'dmz_placement': {
                'latency_reduction_ms': -1.5,
                'bandwidth_bonus': 1.10,
                'reliability_bonus': 1.02,
                'security_score': 0.95,
                'cost_multiplier': 1.1,
                'management_complexity': 0.8,
                'backup_access_efficiency': 1.05
            },
            'centralized_datacenter': {
                'latency_reduction_ms': 0.0,
                'bandwidth_bonus': 1.0,
                'reliability_bonus': 1.0,
                'security_score': 1.0,
                'cost_multiplier': 1.0,
                'management_complexity': 1.0,
                'backup_access_efficiency': 1.0
            },
            'edge_deployment': {
                'latency_reduction_ms': -5.0,
                'bandwidth_bonus': 1.30,
                'reliability_bonus': 0.97,
                'security_score': 0.70,
                'cost_multiplier': 1.4,
                'management_complexity': 0.6,
                'backup_access_efficiency': 1.25
            }
        }
        
        # Database-specific backup file characteristics
        self.database_backup_characteristics = {
            'sqlserver': {
                'compression_ratio': 0.7,  # SQL Server backups compress well
                'file_size_factor': 1.2,   # Larger backup files
                'io_pattern': 'sequential', # Good sequential I/O
                'preferred_storage': 'smb_share'
            },
            'oracle': {
                'compression_ratio': 0.6,  # Oracle dumps compress very well
                'file_size_factor': 1.0,   # Moderate backup files
                'io_pattern': 'sequential',
                'preferred_storage': 'nfs_nas'
            },
            'postgresql': {
                'compression_ratio': 0.5,  # Text-based dumps compress excellently
                'file_size_factor': 0.8,   # Smaller compressed backups
                'io_pattern': 'sequential',
                'preferred_storage': 'nfs_nas'
            },
            'mysql': {
                'compression_ratio': 0.6,
                'file_size_factor': 0.9,
                'io_pattern': 'sequential',
                'preferred_storage': 'nfs_nas'
            }
        }
    
    def calculate_agent_performance(self, agent_size: str, num_agents: int, 
                                   platform_type: str = 'vmware', database_type: str = 'sqlserver',
                                   storage_type: str = 'share', os_type: str = 'windows', 
                                   placement_type: str = 'centralized_datacenter',
                                   backup_compression: float = 1.0) -> Dict:
        """Enhanced agent performance calculation for backup file transfers"""
        
        base_spec = self.datasync_specs[agent_size]
        db_characteristics = self.database_backup_characteristics.get(database_type, self.database_backup_characteristics['sqlserver'])
        
        # Platform characteristics
        platform_char = self.platform_characteristics[platform_type]
        placement_char = self.placement_characteristics[placement_type]
        
        # Calculate per-agent performance
        base_throughput = base_spec['throughput_mbps']
        
        # Apply platform efficiency
        platform_throughput = base_throughput * platform_char['base_efficiency']
        
        # Apply backup file handling efficiency
        backup_file_throughput = platform_throughput * platform_char['backup_file_handling']
        
        # Apply database-specific backup characteristics
        db_throughput = backup_file_throughput * (2.0 - db_characteristics['file_size_factor'])
        
        # Apply I/O efficiency based on storage type and OS
        if storage_type == 'nas' and os_type == 'linux':
            io_multiplier = 1.0
        elif storage_type == 'share' and os_type == 'windows':
            io_multiplier = 0.75
        else:
            io_multiplier = 0.9
        
        # Apply placement bonus for backup file access
        placement_throughput = db_throughput * io_multiplier * placement_char['backup_access_efficiency']
        
        # Network efficiency
        network_efficiency = platform_char['network_efficiency']
        
        # DataSync agent efficiency for backup files
        agent_backup_efficiency = base_spec['backup_efficiency']
        
        # Final per-agent throughput
        per_agent_throughput = placement_throughput * network_efficiency * agent_backup_efficiency
        
        # Apply compression impact (compressed backups transfer faster)
        per_agent_throughput = per_agent_throughput * backup_compression
        
        # Calculate scaling efficiency for backup file transfers
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
        
        # Enhanced cost calculation with placement considerations
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
            'agent_type': 'datasync',
            'agent_size': agent_size,
            'num_agents': num_agents,
            'platform_type': platform_type,
            'database_type': database_type,
            'storage_type': storage_type,
            'os_type': os_type,
            'placement_type': placement_type,
            'base_throughput_mbps': base_throughput,
            'per_agent_throughput_mbps': per_agent_throughput,
            'total_agent_throughput_mbps': total_agent_throughput,
            'scaling_efficiency': scaling_efficiency,
            'platform_efficiency': platform_char['base_efficiency'],
            'backup_file_efficiency': platform_char['backup_file_handling'],
            'io_multiplier': io_multiplier,
            'network_efficiency': network_efficiency,
            'agent_backup_efficiency': agent_backup_efficiency,
            'placement_bonus': placement_char['backup_access_efficiency'],
            'performance_loss_pct': performance_loss,
            'per_agent_monthly_cost': per_agent_cost,
            'total_monthly_cost': total_monthly_cost,
            'vmware_licensing_multiplier': vmware_licensing_multiplier,
            'placement_cost_multiplier': placement_cost_multiplier,
            'compression_impact': backup_compression,
            'base_spec': base_spec,
            'platform_characteristics': platform_char,
            'placement_characteristics': placement_char,
            'database_characteristics': db_characteristics
        }

class DatabasePlacementAnalyzer:
    """Database backup specific placement analysis and optimization"""
    
    def __init__(self):
        self.placement_strategies = {
            'backup_source_colocation': {
                'name': 'Backup Source Co-location',
                'description': 'DataSync agents placed directly with backup storage',
                'pros': [
                    'Direct access to backup files',
                    'Minimal latency for large file reads',
                    'Maximum bandwidth utilization',
                    'No intermediate network hops for backup access'
                ],
                'cons': [
                    'Distributed agent management',
                    'Security exposure at backup source',
                    'Limited backup storage scalability',
                    'Agent maintenance at multiple sites'
                ],
                'best_for': ['Large backup files (>500GB)', 'High-throughput requirements', 'SQL Server backup files'],
                'avoid_when': ['High security requirements', 'Limited backup server resources']
            },
            'dmz_placement': {
                'name': 'DMZ Backup Processing',
                'description': 'DataSync agents in DMZ with backup file staging',
                'pros': [
                    'Controlled backup file access',
                    'Centralized security management',
                    'Good performance with staging',
                    'Backup file preprocessing capability'
                ],
                'cons': [
                    'Additional backup file copy overhead',
                    'DMZ storage requirements',
                    'Network latency for backup reads',
                    'Complex firewall configurations'
                ],
                'best_for': ['Production environments', 'Compliance requirements', 'Oracle/PostgreSQL backups'],
                'avoid_when': ['Ultra-high performance needs', 'Simple backup scenarios']
            },
            'centralized_datacenter': {
                'name': 'Centralized Data Center',
                'description': 'DataSync agents in central enterprise data center',
                'pros': [
                    'Simplified agent management',
                    'Centralized monitoring and maintenance',
                    'Standard security controls',
                    'Easy backup scheduling coordination'
                ],
                'cons': [
                    'Network latency for backup file access',
                    'Bandwidth sharing with other workloads',
                    'Distance from backup sources',
                    'Potential backup file transfer bottlenecks'
                ],
                'best_for': ['Multiple database backup sources', 'Standard deployments', 'Operational simplicity'],
                'avoid_when': ['Performance-critical transfers', 'Very large backup files']
            },
            'edge_deployment': {
                'name': 'Edge Backup Processing',
                'description': 'DataSync agents at network edge closest to backup storage',
                'pros': [
                    'Ultra-low latency backup access',
                    'Maximum backup transfer performance',
                    'Optimized backup file paths',
                    'Reduced core network load'
                ],
                'cons': [
                    'Complex distributed management',
                    'Security challenges at edge',
                    'Higher operational costs',
                    'Limited monitoring capabilities'
                ],
                'best_for': ['Performance-critical backup transfers', 'Very large databases (>5TB)', 'Time-sensitive migrations'],
                'avoid_when': ['Budget constraints', 'Security-first environments']
            }
        }
    
    def analyze_placement_options(self, config: Dict, network_perf: Dict, 
                                agent_manager: EnhancedDataSyncAgentManager) -> List[Dict]:
        """Analyze all placement options for database backup scenarios"""
        placement_options = []
        
        for placement_type, strategy in self.placement_strategies.items():
            # Calculate performance for this placement
            agent_perf = agent_manager.calculate_agent_performance(
                config['agent_size'], 
                config['number_of_agents'],
                config['server_type'],
                config['source_database_engine'],
                self._map_storage_type(config['storage_type']),
                self._determine_os_type(config['operating_system']),
                placement_type,
                config.get('backup_compression', 1.0)
            )
            
            # Calculate placement score for backup scenarios
            score = self._calculate_backup_placement_score(placement_type, config, network_perf, agent_perf)
            
            # Determine implementation complexity
            complexity = self._assess_backup_implementation_complexity(placement_type, config)
            
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
    
    def _calculate_backup_placement_score(self, placement_type: str, config: Dict, 
                                        network_perf: Dict, agent_perf: Dict) -> float:
        """Calculate comprehensive placement score for backup scenarios (0-100)"""
        # Performance score (45% weight - higher for backup scenarios)
        max_possible_throughput = 2000 * config['number_of_agents']
        performance_score = (agent_perf['total_agent_throughput_mbps'] / max_possible_throughput) * 100
        performance_score = min(performance_score, 100)
        
        # Cost efficiency score (20% weight)
        cost_per_mbps = agent_perf['total_monthly_cost'] / agent_perf['total_agent_throughput_mbps']
        max_cost_per_mbps = 1000
        cost_score = max(0, 100 - (cost_per_mbps / max_cost_per_mbps * 100))
        
        # Security score (15% weight)
        security_score = agent_perf['placement_characteristics']['security_score'] * 100
        
        # Management complexity score (10% weight)
        management_score = agent_perf['placement_characteristics']['management_complexity'] * 100
        
        # Backup access efficiency (10% weight - specific to backup scenarios)
        backup_access_score = agent_perf['backup_access_efficiency'] * 100
        
        # Calculate weighted total
        total_score = (
            performance_score * 0.45 +
            cost_score * 0.20 +
            security_score * 0.15 +
            management_score * 0.10 +
            backup_access_score * 0.10
        )
        
        return min(total_score, 100)
    
    def _assess_backup_implementation_complexity(self, placement_type: str, config: Dict) -> Dict:
        """Assess implementation complexity for backup placement scenarios"""
        complexity_factors = {
            'backup_source_colocation': {
                'setup_time_days': 2,
                'skill_level': 'Medium',
                'infrastructure_changes': 'Minimal',
                'security_review': 'Standard',
                'ongoing_maintenance': 'Medium',
                'backup_coordination': 'Simple'
            },
            'dmz_placement': {
                'setup_time_days': 5,
                'skill_level': 'High',
                'infrastructure_changes': 'Moderate',
                'security_review': 'Extensive',
                'ongoing_maintenance': 'High',
                'backup_coordination': 'Complex'
            },
            'centralized_datacenter': {
                'setup_time_days': 1,
                'skill_level': 'Low',
                'infrastructure_changes': 'Minimal',
                'security_review': 'Standard',
                'ongoing_maintenance': 'Low',
                'backup_coordination': 'Standard'
            },
            'edge_deployment': {
                'setup_time_days': 7,
                'skill_level': 'Expert',
                'infrastructure_changes': 'Significant',
                'security_review': 'Extensive',
                'ongoing_maintenance': 'Very High',
                'backup_coordination': 'Very Complex'
            }
        }
        
        return complexity_factors.get(placement_type, complexity_factors['centralized_datacenter'])
    
    def _map_storage_type(self, storage_type: str) -> str:
        """Map storage type to simplified categories"""
        mapping = {
            'windows_share_drive': 'share',
            'linux_nas_drive': 'nas',
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

def get_database_backup_info(database_type: str) -> Dict:
    """Get database-specific backup information"""
    backup_info = {
        'sqlserver': {
            'typical_backup_extensions': ['.bak', '.trn'],
            'compression_available': True,
            'typical_compression_ratio': 0.7,
            'preferred_storage': 'Windows Share Drive (SMB)',
            'backup_tools': ['SQL Server Management Studio', 'T-SQL BACKUP', 'PowerShell'],
            'considerations': [
                'SQL Server backups work well with SMB shares',
                'Consider backup compression for faster transfers',
                'Transaction log backups require frequent transfers',
                'Large database backups benefit from parallel DataSync agents'
            ]
        },
        'oracle': {
            'typical_backup_extensions': ['.dmp', '.log'],
            'compression_available': True,
            'typical_compression_ratio': 0.6,
            'preferred_storage': 'Linux NAS Drive (NFS)',
            'backup_tools': ['RMAN', 'Data Pump (expdp)', 'Cold Backup'],
            'considerations': [
                'Oracle works efficiently with NFS storage',
                'Data Pump exports compress very well',
                'RMAN backups can be parallelized',
                'Archive log backups need frequent synchronization'
            ]
        },
        'postgresql': {
            'typical_backup_extensions': ['.sql', '.dump', '.tar'],
            'compression_available': True,
            'typical_compression_ratio': 0.5,
            'preferred_storage': 'Linux NAS Drive (NFS)',
            'backup_tools': ['pg_dump', 'pg_dumpall', 'pg_basebackup'],
            'considerations': [
                'PostgreSQL dumps are text-based and compress excellently',
                'Custom format dumps provide best compression',
                'Parallel dumps improve performance significantly',
                'WAL files require continuous archiving'
            ]
        },
        'mysql': {
            'typical_backup_extensions': ['.sql', '.dump'],
            'compression_available': True,
            'typical_compression_ratio': 0.6,
            'preferred_storage': 'Linux NAS Drive (NFS)',
            'backup_tools': ['mysqldump', 'mysqlpump', 'Percona XtraBackup'],
            'considerations': [
                'MySQL dumps compress well with gzip',
                'Binary backups transfer faster than logical dumps',
                'Point-in-time recovery requires binary log transfers',
                'Large databases benefit from physical backups'
            ]
        }
    }
    return backup_info.get(database_type, backup_info['mysql'])

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
        - `AWS_REGION` (optional, defaults to us-east-1)
        
        For Claude AI Integration:
        - `CLAUDE_API_KEY`
        
        **How to configure:**
        1. Go to your Streamlit Cloud app settings
        2. Navigate to the "Secrets" section
        3. Add the required keys as shown above
        """)

def render_database_backup_overview(config: Dict):
    """Render database-specific backup overview"""
    st.markdown("**🗄️ Database Backup Migration Overview**")
    
    database_type = config['source_database_engine']
    backup_info = get_database_backup_info(database_type)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if database_type == 'sqlserver':
            card_class = "sql-server-card"
            icon = "🔵"
        else:
            card_class = "linux-db-card" 
            icon = "🐧"
            
        st.markdown(f"""
        <div class="{card_class}">
            <h4 style="font-size: 18px; font-weight: bold; margin-bottom: 15px;">{icon} {database_type.upper()} Backup Configuration</h4>
            <div style="font-size: 15px; line-height: 1.7;">
                <p style="margin: 8px 0;"><strong>Backup Size:</strong> {config.get('backup_size_gb', 1000):,} GB</p>
                <p style="margin: 8px 0;"><strong>Backup Type:</strong> {config.get('backup_type', 'Full backup')}</p>
                <p style="margin: 8px 0;"><strong>Compression:</strong> {config.get('backup_compression_ratio', 0.7)*100:.0f}% ratio</p>
                <p style="margin: 8px 0;"><strong>Storage:</strong> {backup_info['preferred_storage']}</p>
                <p style="margin: 8px 0;"><strong>Extensions:</strong> {', '.join(backup_info['typical_backup_extensions'])}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="backup-card">
            <h4 style="font-size: 18px; font-weight: bold; margin-bottom: 15px;">🛠️ Recommended Tools & Considerations</h4>
            <div style="font-size: 14px; line-height: 1.6;">
                <p style="margin: 8px 0;"><strong>Backup Tools:</strong></p>
                <ul style="margin: 0 0 12px 20px; padding: 0;">
                    {chr(10).join([f'<li style="margin: 4px 0;">{tool}</li>' for tool in backup_info['backup_tools']])}
                </ul>
                <p style="margin: 8px 0;"><strong>Key Considerations:</strong></p>
                <ul style="margin: 0; padding: 0 0 0 20px;">
                    {chr(10).join([f'<li style="margin: 4px 0; font-size: 13px;">{consideration}</li>' for consideration in backup_info['considerations'][:2]])}
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

def render_backup_transfer_metrics(config: Dict, network_perf: Dict, agent_perf: Dict):
    """Render backup-specific transfer metrics"""
    st.markdown("**📊 Backup Transfer Performance Metrics**")
    
    backup_size_gb = config.get('backup_size_gb', 1000)
    compression_ratio = config.get('backup_compression_ratio', 1.0)
    effective_backup_size = backup_size_gb * compression_ratio
    
    # Calculate transfer times
    throughput_mbps = min(network_perf['effective_bandwidth_mbps'], agent_perf['total_agent_throughput_mbps'])
    
    # Convert GB to Mbits for calculation
    backup_size_mbits = effective_backup_size * 8 * 1000
    transfer_time_seconds = backup_size_mbits / throughput_mbps
    transfer_time_hours = transfer_time_seconds / 3600
    
    # Calculate costs
    transfer_cost = (agent_perf['total_monthly_cost'] / (30 * 24)) * transfer_time_hours
    cost_per_gb = transfer_cost / backup_size_gb
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">📦 {backup_size_gb:,} GB</div>
            <div class="metric-label">Original Backup Size</div>
            <div style="font-size: 12px; color: #6b7280;">Uncompressed</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">🗜️ {effective_backup_size:,.0f} GB</div>
            <div class="metric-label">Compressed Size</div>
            <div style="font-size: 12px; color: #059669;">{(1-compression_ratio)*100:.0f}% reduction</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">⚡ {throughput_mbps:,.0f} Mbps</div>
            <div class="metric-label">Effective Throughput</div>
            <div style="font-size: 12px; color: #6b7280;">Backup transfer rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">⏱️ {transfer_time_hours:.1f} hrs</div>
            <div class="metric-label">Transfer Time</div>
            <div style="font-size: 12px; color: #6b7280;">{transfer_time_seconds/60:.0f} minutes</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">💰 ${transfer_cost:.2f}</div>
            <div class="metric-label">Transfer Cost</div>
            <div style="font-size: 12px; color: #6b7280;">${cost_per_gb:.4f}/GB</div>
        </div>
        """, unsafe_allow_html=True)

def render_database_storage_comparison(config: Dict):
    """Render database-specific storage comparison"""
    st.markdown("**📊 Database Backup Storage Performance Comparison**")
    
    # Create comparison scenarios based on database types
    scenarios = []
    
    database_configs = [
        {'name': 'SQL Server → Windows Share (SMB)', 'db': 'sqlserver', 'os': 'windows', 'storage': 'smb', 'efficiency': 0.75},
        {'name': 'Oracle → Linux NAS (NFS)', 'db': 'oracle', 'os': 'linux', 'storage': 'nfs', 'efficiency': 0.96},
        {'name': 'PostgreSQL → Linux NAS (NFS)', 'db': 'postgresql', 'os': 'linux', 'storage': 'nfs', 'efficiency': 0.98},
        {'name': 'MySQL → Linux NAS (NFS)', 'db': 'mysql', 'os': 'linux', 'storage': 'nfs', 'efficiency': 0.92},
    ]
    
    base_throughput = 1000  # Base throughput for comparison
    current_db = config['source_database_engine']
    
    for config_item in configs_item:
        is_current = config_item['db'] == current_db
        backup_info = get_database_backup_info(config_item['db'])
        
        # Apply database-specific backup characteristics
        backup_efficiency = base_throughput * config_item['efficiency'] * backup_info['typical_compression_ratio']
        
        scenarios.append({
            'Configuration': config_item['name'],
            'Database': config_item['db'].upper(),
            'Throughput (Mbps)': backup_efficiency,
            'Efficiency (%)': config_item['efficiency'] * 100,
            'Compression Ratio': backup_info['typical_compression_ratio'],
            'Current': '✓ Current' if is_current else '',
            'Storage Protocol': config_item['storage'].upper()
        })
    
    df_scenarios = pd.DataFrame(scenarios)
    
    # Create comparison chart
    fig = px.bar(
        df_scenarios,
        x='Configuration',
        y='Throughput (Mbps)',
        title='Database Backup Storage Performance Comparison',
        color='Efficiency (%)',
        color_continuous_scale='RdYlGn',
        text='Throughput (Mbps)',
        hover_data=['Compression Ratio', 'Storage Protocol']
    )
    
    fig.update_traces(
        texttemplate='%{text:.0f} Mbps', 
        textposition='outside',
        textfont=dict(size=14, family="Arial Black")
    )
    
    fig.update_layout(
        height=500,
        xaxis_tickangle=-45,
        title=dict(
            font=dict(size=18, family="Arial Black")
        ),
        xaxis=dict(
            title=dict(text='Database & Storage Configuration', font=dict(size=14)),
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            title=dict(text='Effective Throughput (Mbps)', font=dict(size=14)),
            tickfont=dict(size=12)
        ),
        font=dict(size=12)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance insights
    best_config = df_scenarios.loc[df_scenarios['Throughput (Mbps)'].idxmax()]
    worst_config = df_scenarios.loc[df_scenarios['Throughput (Mbps)'].idxmin()]
    
    performance_diff = best_config['Throughput (Mbps)'] - worst_config['Throughput (Mbps)']
    performance_diff_pct = (performance_diff / worst_config['Throughput (Mbps)']) * 100
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border: 3px solid #10b981; border-radius: 12px; padding: 20px; margin: 20px 0; font-size: 15px; line-height: 1.8;">
        <strong style="color: #065f46; font-size: 18px;">🏆 Database Backup Performance Insights:</strong><br><br>
        • <strong>Best Configuration:</strong> {best_config['Configuration']} ({best_config['Throughput (Mbps)']:.0f} Mbps)<br>
        • <strong>Worst Configuration:</strong> {worst_config['Configuration']} ({worst_config['Throughput (Mbps)']:.0f} Mbps)<br>
        • <strong>Performance Gap:</strong> {performance_diff:.0f} Mbps ({performance_diff_pct:.1f}% difference)<br>
        • <strong>Linux NFS Advantage:</strong> Significant performance benefit for Oracle/PostgreSQL<br>
        • <strong>Compression Impact:</strong> PostgreSQL text dumps achieve best compression ratios
    </div>
    """, unsafe_allow_html=True)

def render_datasync_backup_configuration(config: Dict, agent_perf: Dict):
    """Render DataSync-specific configuration for backup transfers"""
    st.markdown("**⚙️ DataSync Agent Configuration for Database Backups**")
    
    database_type = config['source_database_engine']
    backup_info = get_database_backup_info(database_type)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="agent-card">
            <h4 style="font-size: 17px; font-weight: bold; margin-bottom: 15px;">🤖 DataSync Agent Configuration</h4>
            <div style="font-size: 14px; line-height: 1.7;">
                <p style="margin: 8px 0;"><strong>Agent Count:</strong> {agent_perf['num_agents']}</p>
                <p style="margin: 8px 0;"><strong>Agent Size:</strong> {agent_perf['agent_size'].title()}</p>
                <p style="margin: 8px 0;"><strong>Total Capacity:</strong> {agent_perf['total_agent_throughput_mbps']:,.0f} Mbps</p>
                <p style="margin: 8px 0;"><strong>Platform:</strong> {agent_perf['platform_type'].title()}</p>
                <p style="margin: 8px 0;"><strong>Backup Efficiency:</strong> {agent_perf['agent_backup_efficiency']*100:.1f}%</p>
                <p style="margin: 8px 0;"><strong>Monthly Cost:</strong> ${agent_perf['total_monthly_cost']:,.0f}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="backup-card">
            <h4 style="font-size: 17px; font-weight: bold; margin-bottom: 15px;">📁 Backup File Handling</h4>
            <div style="font-size: 14px; line-height: 1.7;">
                <p style="margin: 8px 0;"><strong>Database Type:</strong> {database_type.upper()}</p>
                <p style="margin: 8px 0;"><strong>File Extensions:</strong> {', '.join(backup_info['typical_backup_extensions'])}</p>
                <p style="margin: 8px 0;"><strong>Compression Available:</strong> {'Yes' if backup_info['compression_available'] else 'No'}</p>
                <p style="margin: 8px 0;"><strong>Typical Compression:</strong> {backup_info['typical_compression_ratio']*100:.0f}%</p>
                <p style="margin: 8px 0;"><strong>Storage Access:</strong> {backup_info['preferred_storage']}</p>
                <p style="margin: 8px 0;"><strong>Transfer Optimization:</strong> Sequential I/O</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

def render_agent_placement_analysis(config: Dict, network_perf: Dict, agent_manager, claude_integration):
    """Render comprehensive agent placement analysis using native Streamlit components"""
    st.markdown("## 🎯 Agent Placement Strategy Analysis")
    
    # Initialize placement analyzer
    placement_analyzer = DatabasePlacementAnalyzer()
    
    try:
        # Analyze placement options
        placement_options = placement_analyzer.analyze_placement_options(config, network_perf, agent_manager)
    except Exception as e:
        st.error(f"Error analyzing placement options: {str(e)}")
        return
    
    # Render placement options
    st.markdown("### 📍 Placement Strategy Comparison")
    
    try:
        # Create placement comparison chart
        placement_df = pd.DataFrame([
            {
                'Strategy': opt['strategy']['name'],
                'Performance Score': opt['placement_score'],
                'Throughput (Mbps)': opt['throughput_mbps'],
                'Monthly Cost ($)': opt['monthly_cost'],
                'Security Score': opt['security_score'] * 100,
                'Management Complexity': (1 - opt['management_complexity']) * 100
            }
            for opt in placement_options
        ])
        
        # Placement score chart
        fig_placement = px.bar(
            placement_df,
            x='Strategy',
            y='Performance Score',
            title='Agent Placement Strategy Scores',
            color='Performance Score',
            color_continuous_scale='RdYlGn',
            text='Performance Score'
        )
        
        fig_placement.update_traces(
            texttemplate='%{text:.1f}',
            textposition='outside',
            textfont=dict(size=14, family="Arial Black")
        )
        
        fig_placement.update_layout(
            height=400,
            title=dict(font=dict(size=18, family="Arial Black")),
            xaxis=dict(title=dict(text='Placement Strategy', font=dict(size=14))),
            yaxis=dict(title=dict(text='Overall Score (0-100)', font=dict(size=14))),
            font=dict(size=12)
        )
        
        st.plotly_chart(fig_placement, use_container_width=True)
        
        # Display placement options using native Streamlit components
        st.markdown("### 📋 Detailed Placement Analysis")
        
        for i, option in enumerate(placement_options):
            is_recommended = i == 0
            
            # Use different Streamlit container types based on score
            if option['placement_score'] >= 80:
                container = st.success if is_recommended else st.info
            elif option['placement_score'] >= 65:
                container = st.info
            else:
                container = st.warning
            
            with st.container():
                # Header with score
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.subheader(f"{'🏆 ' if is_recommended else ''}{option['strategy']['name']}")
                    if is_recommended:
                        st.success("Recommended Strategy")
                with col2:
                    st.metric("Score", f"{option['placement_score']:.1f}/100")
                
                # Description
                st.write(option['strategy']['description'])
                
                # Key metrics in columns
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Backup Access Performance",
                        f"{option['throughput_mbps']:,.0f} Mbps",
                        f"{option['backup_access_efficiency']*100:.0f}% efficiency"
                    )
                
                with col2:
                    cost_per_mbps = option['monthly_cost'] / option['throughput_mbps']
                    st.metric(
                        "Transfer Cost",
                        f"${option['monthly_cost']:,.0f}/month",
                        f"${cost_per_mbps:.2f}/Mbps"
                    )
                
                with col3:
                    st.metric(
                        "Security",
                        f"{option['security_score']*100:.0f}%",
                        "rating"
                    )
                
                with col4:
                    st.metric(
                        "Management",
                        f"{option['management_complexity']*100:.0f}%",
                        "complexity"
                    )
                
                # Expandable details
                with st.expander(f"📋 {option['strategy']['name']} - Implementation Details", expanded=is_recommended):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**✅ Advantages for Backup Scenarios:**")
                        for pro in option['strategy']['pros']:
                            st.markdown(f"• {pro}")
                    
                    with col2:
                        st.markdown("**⚠️ Challenges:**")
                        for con in option['strategy']['cons']:
                            st.markdown(f"• {con}")
                    
                    # Implementation timeline using info box
                    complexity = option['implementation_complexity']
                    st.info(f"""
                    **🔧 Implementation Details:**
                    - **Setup Time:** {complexity['setup_time_days']} days
                    - **Skill Level Required:** {complexity['skill_level']}
                    - **Backup Coordination:** {complexity['backup_coordination']}
                    """)
                
                st.divider()  # Add separator between options
        
        # Summary comparison table
        st.markdown("### 📊 Strategy Comparison Summary")
        
        # Create a more readable comparison dataframe
        comparison_data = []
        for opt in placement_options:
            comparison_data.append({
                'Strategy': opt['strategy']['name'],
                'Score': f"{opt['placement_score']:.1f}/100",
                'Throughput': f"{opt['throughput_mbps']:,.0f} Mbps",
                'Monthly Cost': f"${opt['monthly_cost']:,.0f}",
                'Security': f"{opt['security_score']*100:.0f}%",
                'Setup Time': f"{opt['implementation_complexity']['setup_time_days']} days",
                'Backup Access': f"{opt['backup_access_efficiency']*100:.0f}%"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Best option highlight
        best_option = placement_options[0]
        st.success(f"""
        **🏆 Recommended: {best_option['strategy']['name']}**
        
        This strategy provides the best balance of performance ({best_option['placement_score']:.1f}/100), 
        cost efficiency (${best_option['monthly_cost']:,.0f}/month), and backup access efficiency 
        ({best_option['backup_access_efficiency']*100:.0f}%).
        """)
        
    except Exception as e:
        st.error(f"Error creating placement analysis: {str(e)}")
        st.info("Placement analysis data is still available in the summary above.")

def render_infrastructure_topology_with_placement(config: Dict, network_perf: Dict, agent_perf: Dict, placement_type: str):
    """Render detailed infrastructure topology showing agent placement using native Streamlit"""
    st.markdown("### 🏗️ Infrastructure Topology with Agent Placement")
    
    # Create topology based on placement type
    topology_components = []
    
    # Base infrastructure components
    source_storage_type = network_perf.get('storage_mount_type', 'nfs').upper()
    source_os = network_perf.get('os_type', 'linux').title()
    
    # Source tier
    topology_components.extend([
        {
            'tier': 'Source',
            'component': f'{source_storage_type} Storage Server',
            'details': f"{config.get('backup_size_gb', 1000)} GB Database Backup",
            'status': 'Active',
            'icon': '🗄️',
            'has_agent': placement_type == 'backup_source_colocation'
        },
        {
            'tier': 'Source',
            'component': f'{source_os} File System',
            'details': f"{source_storage_type} Protocol",
            'status': 'Active',
            'icon': '📁',
            'has_agent': False
        }
    ])
    
    # Network tier
    if placement_type == 'edge_deployment':
        topology_components.append({
            'tier': 'Edge',
            'component': 'Edge Network Node',
            'details': 'Agent Deployment Location',
            'status': 'Agent Deployed',
            'icon': '🌐',
            'has_agent': True
        })
    
    # Security tier
    if placement_type == 'dmz_placement':
        topology_components.append({
            'tier': 'Security',
            'component': 'DMZ Security Zone',
            'details': 'Controlled Agent Environment',
            'status': 'Agent Deployed',
            'icon': '🛡️',
            'has_agent': True
        })
    
    # Data center tier
    if placement_type == 'centralized_datacenter':
        topology_components.append({
            'tier': 'Data Center',
            'component': 'Central Data Center',
            'details': 'Standard Agent Deployment',
            'status': 'Agent Deployed',
            'icon': '🏢',
            'has_agent': True
        })
    
    # AWS tier
    topology_components.extend([
        {
            'tier': 'AWS Edge',
            'component': 'Direct Connect Gateway',
            'details': 'AWS Entry Point',
            'status': 'Connected',
            'icon': '🌉',
            'has_agent': False
        },
        {
            'tier': 'AWS',
            'component': 'Target S3 Bucket',
            'details': f"Backup files destination",
            'status': 'Ready',
            'icon': '🗃️',
            'has_agent': False
        }
    ])
    
    # Create visual topology using native Streamlit
    st.markdown("#### 🗺️ Network Topology with Agent Placement")
    
    # Group by tier
    tiers = {}
    for comp in topology_components:
        tier = comp['tier']
        if tier not in tiers:
            tiers[tier] = []
        tiers[tier].append(comp)
    
    # Display tiers using native Streamlit columns and containers
    for tier_name, components in tiers.items():
        st.markdown(f"**{tier_name} Tier:**")
        
        # Create columns for components
        if len(components) <= 4:
            cols = st.columns(len(components))
        else:
            cols = st.columns(4)
        
        for i, comp in enumerate(components):
            col_idx = i % len(cols)
            
            with cols[col_idx]:
                # Use appropriate Streamlit component based on agent deployment
                if comp['has_agent']:
                    with st.container():
                        st.success(f"""
                        {comp['icon']} **{comp['component']}**
                        
                        {comp['details']}
                        
                        Status: {comp['status']}
                        
                        🤖 **AGENT DEPLOYED**
                        """)
                else:
                    with st.container():
                        st.info(f"""
                        {comp['icon']} **{comp['component']}**
                        
                        {comp['details']}
                        
                        Status: {comp['status']}
                        """)
        
        st.divider()

def render_placement_decision_matrix(config: Dict, placement_options: List[Dict]):
    """Render decision matrix for placement options using native Streamlit"""
    st.markdown("### 📊 Placement Decision Matrix")
    
    # Create decision matrix data
    matrix_data = []
    criteria = ['Performance', 'Cost Efficiency', 'Security', 'Management', 'Implementation']
    
    for option in placement_options:
        # Calculate individual criterion scores
        performance_score = (option['throughput_mbps'] / 2000) * 100  # Normalized to 2000 Mbps max
        cost_efficiency = max(0, 100 - (option['monthly_cost'] / option['throughput_mbps']))
        security_score = option['security_score'] * 100
        management_score = option['management_complexity'] * 100
        impl_score = 100 - (option['implementation_complexity']['setup_time_days'] * 10)  # Inverse of setup time
        
        matrix_data.append({
            'Strategy': option['strategy']['name'],
            'Performance': min(performance_score, 100),
            'Cost Efficiency': min(cost_efficiency, 100),
            'Security': security_score,
            'Management': management_score,
            'Implementation': max(impl_score, 0),
            'Overall Score': option['placement_score']
        })
    
    # Create heatmap visualization (keep this as it's already native Plotly)
    df_matrix = pd.DataFrame(matrix_data)
    
    # Prepare data for heatmap
    heatmap_data = df_matrix.set_index('Strategy')[criteria].values
    strategies = df_matrix['Strategy'].tolist()
    
    fig_matrix = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=criteria,
        y=strategies,
        colorscale='RdYlGn',
        text=heatmap_data,
        texttemplate='%{text:.1f}',
        textfont={"size": 12},
        hoverongaps=False
    ))
    
    fig_matrix.update_layout(
        title=dict(
            text='Placement Strategy Decision Matrix',
            font=dict(size=18, family="Arial Black")
        ),
        xaxis_title="Evaluation Criteria",
        yaxis_title="Placement Strategies",
        height=400,
        font=dict(size=12)
    )
    
    st.plotly_chart(fig_matrix, use_container_width=True)
    
    # Summary table using native Streamlit dataframe
    st.markdown("**📋 Detailed Scoring Matrix:**")
    
    # Format the dataframe for better display
    formatted_df = df_matrix.copy()
    for col in criteria + ['Overall Score']:
        formatted_df[col] = formatted_df[col].round(1)
    
    # Use Streamlit's native dataframe with styling
    st.dataframe(
        formatted_df,
        use_container_width=True,
        column_config={
            "Strategy": st.column_config.TextColumn("Strategy", width="medium"),
            "Performance": st.column_config.ProgressColumn("Performance", min_value=0, max_value=100),
            "Cost Efficiency": st.column_config.ProgressColumn("Cost Efficiency", min_value=0, max_value=100),
            "Security": st.column_config.ProgressColumn("Security", min_value=0, max_value=100),
            "Management": st.column_config.ProgressColumn("Management", min_value=0, max_value=100),
            "Implementation": st.column_config.ProgressColumn("Implementation", min_value=0, max_value=100),
            "Overall Score": st.column_config.ProgressColumn("Overall Score", min_value=0, max_value=100),
        }
    )
    
    # Highlight the best option
    best_strategy = df_matrix.loc[df_matrix['Overall Score'].idxmax()]
    st.success(f"""
    **🏆 Best Strategy: {best_strategy['Strategy']}**
    
    Overall Score: {best_strategy['Overall Score']:.1f}/100
    - Performance: {best_strategy['Performance']:.1f}/100
    - Cost Efficiency: {best_strategy['Cost Efficiency']:.1f}/100
    - Security: {best_strategy['Security']:.1f}/100
    """)

def render_network_path_visualization(network_perf: Dict, config: Dict, agent_perf: Dict):
    """Render comprehensive network path visualization with detailed infrastructure components"""
    st.markdown("**🌐 Comprehensive Network Infrastructure & Migration Path Analysis**")
    
    segments = network_perf['segments']
    agent_type = config['migration_type']
    num_agents = config['number_of_agents']
    agent_size = config['agent_size']
    
    # Create detailed network topology diagram
    st.markdown("### 🏗️ Complete Infrastructure Topology")
    
    # Build comprehensive network flow with all components
    topology_components = []
    
    # Source Infrastructure
    source_storage_type = network_perf.get('storage_mount_type', 'nfs').upper()
    source_os = network_perf.get('os_type', 'linux').title()
    
    topology_components.extend([
        {
            'layer': 'Source Storage',
            'component': f'{source_storage_type} Storage Server',
            'details': f"{config.get('backup_size_gb', 1000)} GB {config.get('source_database_engine', 'Database').upper()} Backup",
            'bandwidth_in': 10000,
            'bandwidth_out': 10000,
            'latency_ms': 0.5,
            'status': 'Operational',
            'icon': '🗄️'
        },
        {
            'layer': 'Source Network',
            'component': f'{source_os} File System Layer',
            'details': f"{source_storage_type} Protocol Handler",
            'bandwidth_in': 10000,
            'bandwidth_out': segments[0]['effective_bandwidth_mbps'] if segments else 9500,
            'latency_ms': 1.0,
            'status': 'Active',
            'icon': '📁'
        }
    ])
    
    # Network Infrastructure Components
    if len(segments) > 1:  # Multi-segment path
        topology_components.extend([
            {
                'layer': 'Local Network',
                'component': 'Source Site Core Switch',
                'details': '10GbE Aggregation Layer',
                'bandwidth_in': segments[0]['effective_bandwidth_mbps'] if segments else 9500,
                'bandwidth_out': segments[0]['effective_bandwidth_mbps'] if segments else 9400,
                'latency_ms': 0.2,
                'status': 'Operational',
                'icon': '🔀'
            },
            {
                'layer': 'Local Network',
                'component': 'Site Border Router/Firewall',
                'details': 'Security & Routing Layer',
                'bandwidth_in': segments[0]['effective_bandwidth_mbps'] if segments else 9400,
                'bandwidth_out': segments[0]['effective_bandwidth_mbps'] if segments else 9200,
                'latency_ms': 1.5,
                'status': 'Protected',
                'icon': '🛡️'
            },
            {
                'layer': 'WAN',
                'component': 'Private Line/MPLS Network',
                'details': f"Segment: {segments[1]['name'] if len(segments) > 1 else 'Direct'}",
                'bandwidth_in': segments[1]['effective_bandwidth_mbps'] if len(segments) > 1 else 9200,
                'bandwidth_out': segments[1]['effective_bandwidth_mbps'] if len(segments) > 1 else 8800,
                'latency_ms': segments[1]['effective_latency_ms'] if len(segments) > 1 else 10,
                'status': 'Connected',
                'icon': '🌐'
            }
        ])
    
    # Jump Server & Migration Agents
    platform_type = config.get('server_type', 'vmware').title()
    
    topology_components.extend([
        {
            'layer': 'Migration Platform',
            'component': f'{platform_type} Jump Server',
            'details': f"{config.get('ram_gb', 32)}GB RAM, {config.get('cpu_cores', 8)} vCPU",
            'bandwidth_in': segments[-1]['effective_bandwidth_mbps'] if segments else 8800,
            'bandwidth_out': agent_perf['total_agent_throughput_mbps'],
            'latency_ms': 2.0 if platform_type == 'Vmware' else 1.0,
            'status': 'Ready',
            'icon': '🖥️'
        },
        {
            'layer': 'Migration Agents',
            'component': f'{num_agents}x DataSync {agent_size.title()} Agents',
            'details': f"Total Capacity: {agent_perf['total_agent_throughput_mbps']:,.0f} Mbps",
            'bandwidth_in': agent_perf['total_agent_throughput_mbps'],
            'bandwidth_out': agent_perf['total_agent_throughput_mbps'],
            'latency_ms': 1.5,
            'status': f"Scaling: {agent_perf['scaling_efficiency']*100:.0f}%",
            'icon': '🤖'
        }
    ])
    
    # AWS Infrastructure
    aws_region = "US-West-2"  # Default region
    final_bandwidth = min(network_perf['effective_bandwidth_mbps'], agent_perf['total_agent_throughput_mbps'])
    
    topology_components.extend([
        {
            'layer': 'AWS Edge',
            'component': 'AWS Direct Connect Gateway',
            'details': f'Region: {aws_region}',
            'bandwidth_in': agent_perf['total_agent_throughput_mbps'],
            'bandwidth_out': final_bandwidth,
            'latency_ms': 5.0,
            'status': 'Connected',
            'icon': '🌉'
        },
        {
            'layer': 'AWS VPC',
            'component': 'VPC Transit Gateway',
            'details': 'Multi-AZ Routing',
            'bandwidth_in': final_bandwidth,
            'bandwidth_out': final_bandwidth * 0.98,
            'latency_ms': 1.0,
            'status': 'Routing',
            'icon': '🔗'
        },
        {
            'layer': 'AWS Services',
            'component': f'Target: S3 Bucket',
            'details': f"Backup storage in {aws_region}",
            'bandwidth_in': final_bandwidth * 0.98,
            'bandwidth_out': final_bandwidth * 0.98,
            'latency_ms': 2.0,
            'status': 'Receiving',
            'icon': '🗃️'
        }
    ])
    
    # Create interactive network flow diagram
    create_detailed_network_diagram(topology_components)
    
    # Performance metrics dashboard
    render_network_performance_dashboard(topology_components, network_perf, agent_perf)
    
    # Detailed component analysis
    render_infrastructure_component_analysis(topology_components, config)
    
    # Network bottleneck identification
    identify_network_bottlenecks(topology_components, network_perf, agent_perf)

def create_detailed_network_diagram(components):
    """Create detailed network topology diagram with enhanced visibility"""
    st.markdown("#### 🗺️ Network Topology Flow Diagram")
    
    try:
        # Prepare data for Sankey diagram with proper structure
        labels = []
        sources = []
        targets = []
        values = []
        colors = []
        
        color_map = {
            'Source Storage': '#ef4444',
            'Source Network': '#f97316', 
            'Local Network': '#eab308',
            'WAN': '#06b6d4',
            'Migration Platform': '#3b82f6',
            'Migration Agents': '#8b5cf6',
            'AWS Edge': '#6b7280',
            'AWS VPC': '#ef4444',
            'AWS Services': '#22c55e'
        }
        
        # Build labels and connections
        for i, comp in enumerate(components):
            labels.append(f"{comp['icon']} {comp['component']}")
            colors.append(color_map.get(comp['layer'], '#6b7280'))
            
            if i > 0:
                sources.append(i-1)
                targets.append(i)
                # Use bandwidth_out from previous component as the flow value
                flow_value = components[i-1]['bandwidth_out']
                values.append(flow_value)
        
        # Only create diagram if we have valid data
        if len(sources) > 0 and len(targets) > 0 and len(values) > 0:
            # Create Sankey diagram with fixed positioning
            fig_sankey = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=2),
                    label=labels,
                    color=colors
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    color='rgba(128, 128, 128, 0.3)'
                )
            )])
            
            fig_sankey.update_layout(
                title=dict(
                    text="Network Infrastructure Flow - Bandwidth Progression",
                    font=dict(size=18, family="Arial Black")
                ),
                font=dict(size=12, family="Arial"),
                height=600,
                margin=dict(l=20, r=20, t=80, b=20)
            )
            
            st.plotly_chart(fig_sankey, use_container_width=True)
        else:
            st.warning("Unable to create network flow diagram - insufficient data points")
    
    except Exception as e:
        st.error(f"Error creating network diagram: {str(e)}")
        st.info("Displaying component details instead:")
    
    # Create detailed component grid (always show this as fallback)
    st.markdown("#### 🧩 Infrastructure Component Details")
    
    # Group components by layer
    layers = {}
    for comp in components:
        layer = comp['layer']
        if layer not in layers:
            layers[layer] = []
        layers[layer].append(comp)
    
    # Display each layer
    for layer_name, layer_comps in layers.items():
        with st.expander(f"🔍 {layer_name} Layer Details", expanded=False):
            cols = st.columns(min(len(layer_comps), 3))
            
            for idx, comp in enumerate(layer_comps):
                col_idx = idx % 3
                with cols[col_idx]:
                    bandwidth_change = comp['bandwidth_out'] - comp['bandwidth_in']
                    bandwidth_pct = (bandwidth_change / comp['bandwidth_in']) * 100 if comp['bandwidth_in'] > 0 else 0
                    
                    st.markdown(f"""
                    <div class="network-segment-card">
                        <h5 style="font-size: 16px; font-weight: bold; color: #1f2937;">{comp['icon']} {comp['component']}</h5>
                        <p style="font-size: 14px; margin: 8px 0;"><strong>Function:</strong> {comp['details']}</p>
                        <div class="segment-performance">
                            <div style="font-size: 13px; line-height: 1.5;">
                                <strong>Input:</strong> {comp['bandwidth_in']:,.0f} Mbps<br>
                                <strong>Output:</strong> {comp['bandwidth_out']:,.0f} Mbps<br>
                                <strong>Loss:</strong> {abs(bandwidth_change):,.0f} Mbps ({abs(bandwidth_pct):.1f}%)<br>
                                <strong>Latency:</strong> {comp['latency_ms']:.1f} ms<br>
                                <strong>Status:</strong> <span style="color: #059669; font-weight: bold;">{comp['status']}</span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

def render_network_performance_dashboard(components, network_perf, agent_perf):
    """Render comprehensive performance dashboard with enhanced visibility"""
    st.markdown("#### 📊 Network Performance Dashboard")
    
    # Key metrics row with improved styling
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_latency = sum(comp['latency_ms'] for comp in components)
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">🕐 {total_latency:.1f} ms</div>
            <div class="metric-label">Total Latency</div>
            <div style="font-size: 12px; color: #6b7280;">vs {network_perf['total_latency_ms']:.1f} ms calculated</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        initial_bw = components[0]['bandwidth_in'] if components else 0
        final_bw = components[-1]['bandwidth_out'] if components else 0
        total_loss = initial_bw - final_bw
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">📉 {total_loss:,.0f} Mbps</div>
            <div class="metric-label">Total BW Loss</div>
            <div style="font-size: 12px; color: #6b7280;">{(total_loss/initial_bw)*100:.1f}% total</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        bottleneck_bw = min(comp['bandwidth_out'] for comp in components)
        bottleneck_comp = next(comp for comp in components if comp['bandwidth_out'] == bottleneck_bw)
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">🚧 {bottleneck_bw:,.0f} Mbps</div>
            <div class="metric-label">Bottleneck</div>
            <div style="font-size: 12px; color: #6b7280;">{bottleneck_comp['component'][:25]}...</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        agent_efficiency = (agent_perf['total_agent_throughput_mbps'] / agent_perf.get('base_throughput_mbps', 1000)) * 100 if agent_perf.get('base_throughput_mbps') else 0
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">🤖 {agent_efficiency:.1f}%</div>
            <div class="metric-label">Agent Efficiency</div>
            <div style="font-size: 12px; color: #6b7280;">{agent_perf['num_agents']} agents</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        overall_efficiency = (final_bw / initial_bw) * 100 if initial_bw > 0 else 0
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">🎯 {overall_efficiency:.1f}%</div>
            <div class="metric-label">Overall Efficiency</div>
            <div style="font-size: 12px; color: #6b7280;">End-to-End</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Performance timeline chart with enhanced visibility
    st.markdown("#### ⏱️ Bandwidth Progression Through Infrastructure")
    
    timeline_data = []
    cumulative_latency = 0
    
    for i, comp in enumerate(components):
        cumulative_latency += comp['latency_ms']
        timeline_data.append({
            'Step': i + 1,
            'Component': comp['component'],
            'Layer': comp['layer'],
            'Bandwidth_In': comp['bandwidth_in'],
            'Bandwidth_Out': comp['bandwidth_out'],
            'Cumulative_Latency': cumulative_latency,
            'Component_Latency': comp['latency_ms']
        })
    
    df_timeline = pd.DataFrame(timeline_data)
    
    # Create dual-axis chart for bandwidth and latency with improved visibility
    fig = go.Figure()
    
    # Bandwidth progression with enhanced styling
    fig.add_trace(go.Scatter(
        x=df_timeline['Step'],
        y=df_timeline['Bandwidth_Out'],
        mode='lines+markers',
        name='Bandwidth (Mbps)',
        line=dict(color='#3b82f6', width=4),
        marker=dict(size=10, color='#1e40af'),
        text=df_timeline['Component'],
        hovertemplate='<b>%{text}</b><br>Bandwidth: %{y:,.0f} Mbps<extra></extra>'
    ))
    
    # Add latency on secondary y-axis with enhanced styling
    fig.add_trace(go.Scatter(
        x=df_timeline['Step'],
        y=df_timeline['Cumulative_Latency'],
        mode='lines+markers',
        name='Cumulative Latency (ms)',
        line=dict(color='#ef4444', width=3, dash='dash'),
        marker=dict(size=8, color='#dc2626'),
        yaxis='y2',
        text=df_timeline['Component'],
        hovertemplate='<b>%{text}</b><br>Latency: %{y:.1f} ms<extra></extra>'
    ))
    
    # Update layout for dual axis with enhanced visibility
    fig.update_layout(
        title=dict(
            text='Network Performance Progression Through Infrastructure',
            font=dict(size=18, family="Arial Black")
        ),
        xaxis=dict(
            title=dict(text='Infrastructure Component Sequence', font=dict(size=14)),
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            title=dict(text='Bandwidth (Mbps)', font=dict(size=14, color='#3b82f6')), 
            side='left',
            tickfont=dict(size=12, color='#3b82f6')
        ),
        yaxis2=dict(
            title=dict(text='Cumulative Latency (ms)', font=dict(size=14, color='#ef4444')), 
            side='right', 
            overlaying='y',
            tickfont=dict(size=12, color='#ef4444')
        ),
        height=500,
        hovermode='x unified',
        font=dict(size=12),
        legend=dict(font=dict(size=12))
    )
    
    # Add component labels on x-axis with better formatting
    fig.update_xaxes(
        tickmode='array',
        tickvals=df_timeline['Step'],
        ticktext=[f"{comp[:18]}..." if len(comp) > 18 else comp for comp in df_timeline['Component']],
        tickangle=45,
        tickfont=dict(size=11)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_infrastructure_component_analysis(components, config):
    """Render detailed infrastructure component analysis with enhanced visibility"""
    st.markdown("#### 🔬 Infrastructure Component Deep Dive")
    
    # Migration-specific components analysis
    agent_type = config['migration_type']
    
    st.markdown("##### 📦 AWS DataSync Agent Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="network-card">
            <h6 style="font-size: 16px; font-weight: bold; margin-bottom: 12px;">🔄 DataSync Agent Characteristics</h6>
            <ul style="font-size: 14px; line-height: 1.6; margin: 0; padding-left: 20px;">
                <li style="margin-bottom: 6px;"><strong>Protocol:</strong> Custom AWS protocol over HTTPS</li>
                <li style="margin-bottom: 6px;"><strong>Compression:</strong> Real-time data compression</li>
                <li style="margin-bottom: 6px;"><strong>Verification:</strong> Checksum validation</li>
                <li style="margin-bottom: 6px;"><strong>Encryption:</strong> TLS 1.2 in transit</li>
                <li style="margin-bottom: 6px;"><strong>Optimization:</strong> Sparse file detection</li>
                <li style="margin-bottom: 6px;"><strong>Throttling:</strong> Configurable bandwidth control</li>
                <li style="margin-bottom: 6px;"><strong>Backup Files:</strong> Large file optimization</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="performance-card">
            <h6 style="font-size: 16px; font-weight: bold; margin-bottom: 12px;">⚡ Current DataSync Configuration</h6>
            <div style="font-size: 14px; line-height: 1.6;">
                <p style="margin: 8px 0;"><strong>Agent Count:</strong> {config['number_of_agents']}</p>
                <p style="margin: 8px 0;"><strong>Agent Size:</strong> {config['agent_size'].title()}</p>
                <p style="margin: 8px 0;"><strong>Platform:</strong> {config['server_type'].title()}</p>
                <p style="margin: 8px 0;"><strong>Database:</strong> {config.get('source_database_engine', 'Unknown').upper()}</p>
                <p style="margin: 8px 0;"><strong>Backup Size:</strong> {config.get('backup_size_gb', 1000):,} GB</p>
                <p style="margin: 8px 0;"><strong>Expected Transfer:</strong> Backup file synchronization</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

def identify_network_bottlenecks(components, network_perf, agent_perf):
    """Identify and analyze network bottlenecks with enhanced visibility"""
    st.markdown("#### 🚧 Bottleneck Analysis & Optimization Opportunities")
    
    # Find bandwidth bottlenecks
    bandwidth_drops = []
    for i in range(len(components) - 1):
        current_comp = components[i]
        next_comp = components[i + 1]
        
        bandwidth_drop = current_comp['bandwidth_out'] - next_comp['bandwidth_in']
        if bandwidth_drop > 10:  # Significant drop
            bandwidth_drops.append({
                'from_component': current_comp['component'],
                'to_component': next_comp['component'],
                'bandwidth_loss': bandwidth_drop,
                'loss_percentage': (bandwidth_drop / current_comp['bandwidth_out']) * 100
            })
    
    # Find latency hotspots
    latency_hotspots = [comp for comp in components if comp['latency_ms'] > 5.0]
    
    # Analysis results with enhanced visibility
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### 📉 Bandwidth Bottlenecks")
        
        if bandwidth_drops:
            for drop in bandwidth_drops:
                st.markdown(f"""
                <div style="background: #fef3c7; border: 2px solid #f59e0b; border-radius: 8px; padding: 15px; margin: 10px 0; font-size: 14px;">
                    <strong style="color: #92400e; font-size: 15px;">⚠️ Bandwidth Drop Detected:</strong><br>
                    <div style="margin-top: 8px; line-height: 1.6;">
                        • <strong>From:</strong> {drop['from_component']}<br>
                        • <strong>To:</strong> {drop['to_component']}<br>
                        • <strong>Loss:</strong> {drop['bandwidth_loss']:,.0f} Mbps ({drop['loss_percentage']:.1f}%)
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: #d1fae5; border: 2px solid #10b981; border-radius: 8px; padding: 15px; margin: 10px 0; font-size: 14px;">
                <strong style="color: #065f46; font-size: 15px;">✅ No significant bandwidth drops detected</strong>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("##### 🕐 Latency Hotspots")
        
        if latency_hotspots:
            for hotspot in latency_hotspots:
                st.markdown(f"""
                <div style="background: #fef3c7; border: 2px solid #f59e0b; border-radius: 8px; padding: 15px; margin: 10px 0; font-size: 14px;">
                    <strong style="color: #92400e; font-size: 15px;">⚠️ High Latency Component:</strong><br>
                    <div style="margin-top: 8px; line-height: 1.6;">
                        • <strong>Component:</strong> {hotspot['component']}<br>
                        • <strong>Latency:</strong> {hotspot['latency_ms']:.1f} ms<br>
                        • <strong>Layer:</strong> {hotspot['layer']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: #d1fae5; border: 2px solid #10b981; border-radius: 8px; padding: 15px; margin: 10px 0; font-size: 14px;">
                <strong style="color: #065f46; font-size: 15px;">✅ All components within acceptable latency ranges</strong>
            </div>
            """, unsafe_allow_html=True)

def render_enhanced_bandwidth_waterfall(config: Dict, network_perf: Dict, agent_perf: Dict):
    """Enhanced bandwidth waterfall with storage type analysis and improved visibility"""
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
    
    # Backup File Handling Impact
    backup_efficiency = agent_perf.get('backup_file_efficiency', 1.0)
    after_backup = after_storage * backup_efficiency
    stages.append('Backup File\nHandling')
    throughputs.append(after_backup)
    descriptions.append(f"{backup_efficiency*100:.1f}% backup file efficiency")
    
    # Protocol Security Overhead
    protocol_efficiency = 0.85 if config['environment'] == 'production' else 0.88
    after_protocol = after_backup * protocol_efficiency
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
    descriptions.append(f"{agent_perf['num_agents']}x DataSync agents")
    
    # Create enhanced visualization with improved visibility
    waterfall_data = pd.DataFrame({
        'Stage': stages,
        'Throughput (Mbps)': throughputs,
        'Description': descriptions
    })
    
    # Create enhanced bar chart with better visibility
    fig = px.bar(
        waterfall_data,
        x='Stage',
        y='Throughput (Mbps)',
        title=f"Enhanced Analysis: {user_nic_speed:,.0f} Mbps → {final_throughput:.0f} Mbps ({storage_mount.upper()}/{platform_type})",
        text='Throughput (Mbps)',
        color='Throughput (Mbps)',
        color_continuous_scale='RdYlGn',
        hover_data=['Description']
    )
    
    fig.update_traces(
        texttemplate='%{text:.0f}', 
        textposition='outside',
        textfont=dict(size=14, family="Arial Black")
    )
    
    fig.update_layout(
        height=600,
        showlegend=False,
        title=dict(
            font=dict(size=18, family="Arial Black")
        ),
        xaxis=dict(
            title=dict(text='Analysis Stages', font=dict(size=14)),
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            title=dict(text='Throughput (Mbps)', font=dict(size=14)),
            tickfont=dict(size=12)
        ),
        font=dict(size=12)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Enhanced analysis summary with improved visibility
    total_loss = user_nic_speed - final_throughput
    total_loss_pct = (total_loss / user_nic_speed) * 100
    
    # Storage type impact analysis with enhanced cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="network-card">
            <h4 style="font-size: 16px; font-weight: bold; margin-bottom: 12px;">🔍 Storage Protocol Impact</h4>
            <div style="font-size: 14px; line-height: 1.7;">
                • <strong>Protocol Type:</strong> {storage_mount.upper()}<br>
                • <strong>Efficiency:</strong> {storage_efficiency*100:.1f}%<br>
                • <strong>Performance Loss:</strong> {(1-storage_efficiency)*100:.1f}%<br>
                • <strong>Bandwidth Impact:</strong> {after_platform - after_storage:.0f} Mbps lost
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="performance-card">
            <h4 style="font-size: 16px; font-weight: bold; margin-bottom: 12px;">⚙️ Platform Impact</h4>
            <div style="font-size: 14px; line-height: 1.7;">
                • <strong>Platform:</strong> {platform_type.title()}<br>
                • <strong>Efficiency:</strong> {agent_perf['platform_efficiency']*100:.1f}%<br>
                • <strong>Performance Loss:</strong> {(1-agent_perf['platform_efficiency'])*100:.1f}%<br>
                • <strong>Bandwidth Impact:</strong> {after_os - after_platform:.0f} Mbps lost
            </div>
        </div>
        """, unsafe_allow_html=True)

def render_aws_integration_panel(aws_integration):
    """Render AWS integration status and real-time data with enhanced visibility"""
    st.markdown("**☁️ AWS Real-Time Integration**")
    
    if not aws_integration.session:
        st.markdown("""
        <div style="background: #e0f2fe; border: 2px solid #0288d1; border-radius: 8px; padding: 15px; margin: 10px 0; font-size: 14px;">
            <strong style="color: #01579b;">ℹ️ AWS integration not connected. Check sidebar for connection status.</strong>
        </div>
        """, unsafe_allow_html=True)
        return {}
    
    # Show current region with enhanced styling
    current_region = aws_integration.session.region_name
    st.markdown(f"""
    <div style="background: #e0f2fe; border: 2px solid #0288d1; border-radius: 8px; padding: 15px; margin: 10px 0; font-size: 15px;">
        <strong style="color: #01579b;">📍 Connected to AWS Region: {current_region}</strong>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 DataSync Tasks**")
        datasync_tasks = aws_integration.get_datasync_tasks()
        
        if datasync_tasks:
            st.markdown(f"""
            <div style="background: #d1fae5; border: 2px solid #10b981; border-radius: 8px; padding: 15px; margin: 10px 0; font-size: 14px;">
                <strong style="color: #065f46;">✅ Found {len(datasync_tasks)} DataSync tasks in {current_region}</strong>
            </div>
            """, unsafe_allow_html=True)
            
            for task in datasync_tasks[:3]:  # Show first 3 tasks
                with st.expander(f"Task: {task['name']}", expanded=False):
                    st.markdown(f"""
                    <div style="font-size: 14px; line-height: 1.6;">
                        <strong>Status:</strong> {task['status']}<br>
                    """)
                    # Extract location names from ARNs for better readability
                    source_loc = task['source_location'].split('/')[-1] if task['source_location'] != 'Unknown' else 'Unknown'
                    dest_loc = task['destination_location'].split('/')[-1] if task['destination_location'] != 'Unknown' else 'Unknown'
                    st.write(f"**Source Location:** {source_loc}")
                    st.write(f"**Destination Location:** {dest_loc}")
                    if task['executions']:
                        latest_execution = task['executions'][0]
                        st.write(f"**Latest Execution:** {latest_execution.get('Status', 'Unknown')}")
        else:
            st.markdown(f"""
            <div style="background: #fef3c7; border: 2px solid #f59e0b; border-radius: 8px; padding: 15px; margin: 10px 0; font-size: 14px;">
                <strong style="color: #92400e;">⚠️ No DataSync tasks found in {current_region}</strong>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("**🔄 DMS Tasks**")
        dms_tasks = aws_integration.get_dms_tasks()
        
        if dms_tasks:
            st.markdown(f"""
            <div style="background: #d1fae5; border: 2px solid #10b981; border-radius: 8px; padding: 15px; margin: 10px 0; font-size: 14px;">
                <strong style="color: #065f46;">✅ Found {len(dms_tasks)} DMS tasks in {current_region}</strong>
            </div>
            """, unsafe_allow_html=True)
            
            for task in dms_tasks[:3]:  # Show first 3 tasks
                with st.expander(f"Task: {task['name']}", expanded=False):
                    st.write(f"**Status:** {task['status']}")
                    st.write(f"**Migration Type:** {task['migration_type']}")
                    # Extract endpoint names from ARNs for better readability
                    source_ep = task['source_endpoint'].split('/')[-1] if task['source_endpoint'] != 'Unknown' else 'Unknown'
                    target_ep = task['target_endpoint'].split('/')[-1] if task['target_endpoint'] != 'Unknown' else 'Unknown'
                    st.write(f"**Source Endpoint:** {source_ep}")
                    st.write(f"**Target Endpoint:** {target_ep}")
        else:
            st.markdown(f"""
            <div style="background: #fef3c7; border: 2px solid #f59e0b; border-radius: 8px; padding: 15px; margin: 10px 0; font-size: 14px;">
                <strong style="color: #92400e;">⚠️ No DMS tasks found in {current_region}</strong>
            </div>
            """, unsafe_allow_html=True)
    
    return {
        'datasync_tasks': len(datasync_tasks),
        'dms_tasks': len(dms_tasks),
        'active_tasks': len([t for t in datasync_tasks if t['status'] == 'AVAILABLE']) + 
                       len([t for t in dms_tasks if t['status'] == 'ready']),
        'region': current_region
    }

def render_enhanced_sidebar():
    """Enhanced sidebar with database backup migration configuration"""
    st.sidebar.header("🗄️ Database Backup Migration Analyzer")
    
    # Database Configuration section
    st.sidebar.subheader("🗃️ Database Configuration")
    
    # Source Database
    source_database_engine = st.sidebar.selectbox(
        "Source Database",
        ["sqlserver", "oracle", "postgresql", "mysql"],
        index=0,
        format_func=lambda x: {
            'sqlserver': '🔵 Microsoft SQL Server',
            'oracle': '🔴 Oracle Database',
            'postgresql': '🐘 PostgreSQL',
            'mysql': '🐬 MySQL'
        }[x]
    )
    
    # Auto-determine storage based on database type
    if source_database_engine == 'sqlserver':
        default_storage = "windows_share_drive"
        default_os = "windows_server_2019"
        storage_options = ["windows_share_drive", "local_storage"]
        storage_labels = {
            'windows_share_drive': '📁 Windows Share Drive (SMB)',
            'local_storage': '💽 Local Storage'
        }
    else:
        default_storage = "linux_nas_drive"
        default_os = "rhel_8"
        storage_options = ["linux_nas_drive", "local_storage"]
        storage_labels = {
            'linux_nas_drive': '📁 Linux NAS Drive (NFS)',
            'local_storage': '💽 Local Storage'
        }
    
    # Backup Configuration
    st.sidebar.subheader("💾 Backup Configuration")
    
    backup_size_gb = st.sidebar.number_input(
        "Backup Size (GB)", 
        min_value=100, 
        max_value=50000, 
        value=1000, 
        step=100,
        help="Size of the database backup files to transfer"
    )
    
    backup_type = st.sidebar.selectbox(
        "Backup Type",
        ["full_backup", "differential_backup", "transaction_log", "archive_log"],
        format_func=lambda x: {
            'full_backup': '📦 Full Database Backup',
            'differential_backup': '📝 Differential Backup',
            'transaction_log': '📋 Transaction Log Backup',
            'archive_log': '📚 Archive Log Backup'
        }[x]
    )
    
    # Compression settings
    backup_info = get_database_backup_info(source_database_engine)
    default_compression = backup_info['typical_compression_ratio']
    
    compression_enabled = st.sidebar.checkbox("Enable Backup Compression", value=True)
    if compression_enabled:
        backup_compression_ratio = st.sidebar.slider(
            "Compression Ratio",
            min_value=0.3,
            max_value=1.0,
            value=default_compression,
            step=0.1,
            format="%.1f",
            help="Lower values = better compression"
        )
    else:
        backup_compression_ratio = 1.0
    
    # Storage Type
    storage_type = st.sidebar.selectbox(
        "Backup Storage Location",
        storage_options,
        index=0,
        format_func=lambda x: storage_labels[x],
        help=f"Recommended for {source_database_engine.upper()}: {backup_info['preferred_storage']}"
    )
    
    # Operating System (auto-determined but adjustable)
    st.sidebar.subheader("💻 System Configuration")
    
    if source_database_engine == 'sqlserver':
        os_options = ["windows_server_2019", "windows_server_2022"]
    else:
        os_options = ["rhel_8", "rhel_9", "ubuntu_20_04", "ubuntu_22_04"]
    
    operating_system = st.sidebar.selectbox(
        "Operating System",
        os_options,
        index=0 if default_os in os_options else 0,
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
    
    # Environment
    environment = st.sidebar.selectbox("Environment", ["non-production", "production"])
    
    # DataSync Agent Configuration
    st.sidebar.subheader("🤖 DataSync Agent Configuration")
    number_of_agents = st.sidebar.number_input("Number of DataSync Agents", min_value=1, max_value=10, value=2, step=1)
    
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
    
    # Database-specific recommendations
    with st.sidebar.expander(f"💡 {source_database_engine.upper()} Recommendations"):
        st.markdown(f"""
        **Backup Tools:**
        {chr(10).join([f'• {tool}' for tool in backup_info['backup_tools']])}
        
        **Key Considerations:**
        {chr(10).join([f'• {consideration}' for consideration in backup_info['considerations'][:3]])}
        """)
    
    return {
        'source_database_engine': source_database_engine,
        'backup_size_gb': backup_size_gb,
        'backup_type': backup_type,
        'backup_compression': compression_enabled,
        'backup_compression_ratio': backup_compression_ratio,
        'storage_type': storage_type,
        'operating_system': operating_system,
        'server_type': server_type,
        'ram_gb': ram_gb,
        'cpu_cores': cpu_cores,
        'nic_type': nic_type,
        'nic_speed': nic_speed,
        'environment': environment,
        'number_of_agents': number_of_agents,
        'agent_size': agent_size,
        'migration_type': 'datasync'  # Always DataSync for backup file transfers
    }

def parse_ai_analysis(analysis_text: str) -> Dict:
    """Parse Claude AI analysis into structured sections"""
    sections = {
        'performance_bottleneck': '',
        'agent_placement': '',
        'storage_optimization': '',
        'backup_migration_strategy': '',
        'cost_optimization': '',
        'implementation_timeline': '',
        'risk_assessment': ''
    }
    
    current_section = None
    lines = analysis_text.split('\n')
    
    for line in lines:
        line = line.strip()
        
        # Identify section headers
        if 'PERFORMANCE BOTTLENECK' in line.upper() or 'BOTTLENECK ANALYSIS' in line.upper():
            current_section = 'performance_bottleneck'
        elif 'AGENT PLACEMENT' in line.upper():
            current_section = 'agent_placement'
        elif 'STORAGE' in line.upper() and 'OPTIMIZATION' in line.upper():
            current_section = 'storage_optimization'
        elif 'BACKUP' in line.upper() and ('STRATEGY' in line.upper() or 'MIGRATION' in line.upper()):
            current_section = 'backup_migration_strategy'
        elif 'COST OPTIMIZATION' in line.upper():
            current_section = 'cost_optimization'
        elif 'IMPLEMENTATION' in line.upper() and ('TIMELINE' in line.upper() or 'STRATEGY' in line.upper()):
            current_section = 'implementation_timeline'
        elif 'RISK ASSESSMENT' in line.upper() or 'RISK' in line.upper():
            current_section = 'risk_assessment'
        elif current_section and line:
            sections[current_section] += line + '\n'
    
    # If sections aren't clearly defined, put everything in performance_bottleneck
    if not any(sections.values()):
        sections['performance_bottleneck'] = analysis_text
    
    return sections

def render_professional_ai_analysis(claude_integration, config: Dict, network_perf: Dict, agent_perf: Dict, aws_data: Dict):
    """Render professionally formatted Claude AI analysis with enhanced visibility"""
    st.markdown("**🤖 AI-Powered Performance Analysis**")
    
    if not claude_integration or not claude_integration.client:
        st.markdown("""
        <div style="background: #e0f2fe; border: 2px solid #0288d1; border-radius: 8px; padding: 15px; margin: 10px 0; font-size: 14px;">
            <strong style="color: #01579b;">ℹ️ Claude AI integration not connected. Check sidebar for connection status.</strong>
        </div>
        """, unsafe_allow_html=True)
        return
    
    with st.spinner("🔄 Analyzing database backup migration configuration..."):
        try:
            analysis = claude_integration.analyze_database_backup_migration(
                config, network_perf, agent_perf, {}, aws_data
            )
        except Exception as e:
            st.error(f"Error getting AI analysis: {str(e)}")
            return
    
    # Parse the analysis into sections
    sections = parse_ai_analysis(analysis)
    
    # Render sections professionally with enhanced visibility
    if sections['performance_bottleneck']:
        st.markdown(f"""
        <div class="ai-section">
            <h4 style="font-size: 20px; color: #1e293b; font-weight: bold;">🔍 Performance Bottleneck Analysis</h4>
            <div style="font-size: 15px; line-height: 1.8; color: #374151;">{sections['performance_bottleneck'].replace(chr(10), '<br>')}</div>
        </div>
        """, unsafe_allow_html=True)
    
    if sections['agent_placement']:
        st.markdown(f"""
        <div class="ai-section">
            <h4 style="font-size: 20px; color: #1e293b; font-weight: bold;">🎯 DataSync Agent Placement Strategy</h4>
            <div style="font-size: 15px; line-height: 1.8; color: #374151;">{sections['agent_placement'].replace(chr(10), '<br>')}</div>
        </div>
        """, unsafe_allow_html=True)
    
    if sections['storage_optimization']:
        st.markdown(f"""
        <div class="ai-section">
            <h4 style="font-size: 20px; color: #1e293b; font-weight: bold;">📁 Storage Protocol Optimization</h4>
            <div style="font-size: 15px; line-height: 1.8; color: #374151;">{sections['storage_optimization'].replace(chr(10), '<br>')}</div>
        </div>
        """, unsafe_allow_html=True)
    
    if sections['backup_migration_strategy']:
        st.markdown(f"""
        <div class="ai-section">
            <h4 style="font-size: 20px; color: #1e293b; font-weight: bold;">🗄️ Database Backup Migration Strategy</h4>
            <div style="font-size: 15px; line-height: 1.8; color: #374151;">{sections['backup_migration_strategy'].replace(chr(10), '<br>')}</div>
        </div>
        """, unsafe_allow_html=True)
    
    if sections['cost_optimization']:
        st.markdown(f"""
        <div class="ai-section">
            <h4 style="font-size: 20px; color: #1e293b; font-weight: bold;">💰 Cost Optimization Strategies</h4>
            <div style="font-size: 15px; line-height: 1.8; color: #374151;">{sections['cost_optimization'].replace(chr(10), '<br>')}</div>
        </div>
        """, unsafe_allow_html=True)
    
    if sections['implementation_timeline']:
        st.markdown(f"""
        <div class="ai-section">
            <h4 style="font-size: 20px; color: #1e293b; font-weight: bold;">🛣️ Implementation Timeline & Strategy</h4>
            <div style="font-size: 15px; line-height: 1.8; color: #374151;">{sections['implementation_timeline'].replace(chr(10), '<br>')}</div>
        </div>
        """, unsafe_allow_html=True)
    
    if sections['risk_assessment']:
        st.markdown(f"""
        <div class="ai-section">
            <h4 style="font-size: 20px; color: #1e293b; font-weight: bold;">⚠️ Risk Assessment & Mitigation</h4>
            <div style="font-size: 15px; line-height: 1.8; color: #374151;">{sections['risk_assessment'].replace(chr(10), '<br>')}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # If no sections were parsed, show the full analysis
    if not any(sections.values()):
        st.markdown(f"""
        <div class="ai-section">
            <h4 style="font-size: 20px; color: #1e293b; font-weight: bold;">🧠 Complete Analysis</h4>
            <div style="font-size: 15px; line-height: 1.8; color: #374151;">{analysis.replace(chr(10), '<br>')}</div>
        </div>
        """, unsafe_allow_html=True)

def render_storage_comparison_analysis(config: Dict):
    """Render detailed storage type comparison with enhanced visibility"""
    st.markdown("**📊 Storage Protocol Performance Comparison**")
    
    # Create comparison scenarios
    scenarios = []
    
    # Current configuration
    current_db = config['source_database_engine']
    current_storage = config['storage_type']
    
    base_throughput = 1000  # Base throughput for comparison
    
    configurations = [
        {'name': 'SQL Server + Windows Share + Physical', 'db': 'sqlserver', 'storage': 'windows_share_drive', 'platform': 'physical', 'efficiency': 0.78},
        {'name': 'SQL Server + Windows Share + VMware', 'db': 'sqlserver', 'storage': 'windows_share_drive', 'platform': 'vmware', 'efficiency': 0.71},
        {'name': 'Oracle + Linux NAS + Physical', 'db': 'oracle', 'storage': 'linux_nas_drive', 'platform': 'physical', 'efficiency': 0.96},
        {'name': 'Oracle + Linux NAS + VMware', 'db': 'oracle', 'storage': 'linux_nas_drive', 'platform': 'vmware', 'efficiency': 0.88},
        {'name': 'PostgreSQL + Linux NAS + Physical', 'db': 'postgresql', 'storage': 'linux_nas_drive', 'platform': 'physical', 'efficiency': 0.98},
        {'name': 'PostgreSQL + Linux NAS + VMware', 'db': 'postgresql', 'storage': 'linux_nas_drive', 'platform': 'vmware', 'efficiency': 0.90},
    ]
    
    for config_item in configurations:
        is_current = (config_item['db'] == current_db and 
                     config_item['storage'] == current_storage and
                     config_item['platform'] == config['server_type'])
        
        backup_info = get_database_backup_info(config_item['db'])
        
        scenarios.append({
            'Configuration': config_item['name'],
            'Database': config_item['db'].upper(),
            'Throughput (Mbps)': base_throughput * config_item['efficiency'],
            'Efficiency (%)': config_item['efficiency'] * 100,
            'Current': '✓ Current' if is_current else '',
            'Compression Ratio': backup_info['typical_compression_ratio'],
            'Performance Loss (%)': (1 - config_item['efficiency']) * 100
        })
    
    df_scenarios = pd.DataFrame(scenarios)
    
    # Create comparison chart with enhanced visibility
    fig = px.bar(
        df_scenarios,
        x='Configuration',
        y='Throughput (Mbps)',
        title='Database & Storage Protocol Performance Comparison',
        color='Efficiency (%)',
        color_continuous_scale='RdYlGn',
        text='Throughput (Mbps)',
        hover_data=['Compression Ratio', 'Performance Loss (%)']
    )
    
    fig.update_traces(
        texttemplate='%{text:.0f} Mbps', 
        textposition='outside',
        textfont=dict(size=14, family="Arial Black")
    )
    
    fig.update_layout(
        height=500,
        xaxis_tickangle=-45,
        title=dict(
            font=dict(size=18, family="Arial Black")
        ),
        xaxis=dict(
            title=dict(text='Database & Storage Configuration', font=dict(size=14)),
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            title=dict(text='Throughput (Mbps)', font=dict(size=14)),
            tickfont=dict(size=12)
        ),
        font=dict(size=12)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance impact table with enhanced styling
    st.markdown("**📋 Detailed Performance Analysis:**")
    
    # Style the dataframe for better visibility
    styled_df = df_scenarios.drop('Current', axis=1).style.format({
        'Throughput (Mbps)': '{:.0f}',
        'Efficiency (%)': '{:.1f}%',
        'Compression Ratio': '{:.1f}',
        'Performance Loss (%)': '{:.1f}%'
    }).background_gradient(subset=['Throughput (Mbps)'], cmap='RdYlGn')
    
    st.dataframe(styled_df, use_container_width=True, height=200)
    
    # Key insights with enhanced visibility
    best_config = df_scenarios.loc[df_scenarios['Throughput (Mbps)'].idxmax()]
    worst_config = df_scenarios.loc[df_scenarios['Throughput (Mbps)'].idxmin()]
    
    performance_diff = best_config['Throughput (Mbps)'] - worst_config['Throughput (Mbps)']
    performance_diff_pct = (performance_diff / worst_config['Throughput (Mbps)']) * 100
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border: 3px solid #10b981; border-radius: 12px; padding: 20px; margin: 20px 0; font-size: 15px; line-height: 1.8;">
        <strong style="color: #065f46; font-size: 18px;">🏆 Key Performance Insights:</strong><br><br>
        • <strong>Best Configuration:</strong> {best_config['Configuration']} ({best_config['Throughput (Mbps)']:.0f} Mbps)<br>
        • <strong>Worst Configuration:</strong> {worst_config['Configuration']} ({worst_config['Throughput (Mbps)']:.0f} Mbps)<br>
        • <strong>Performance Gap:</strong> {performance_diff:.0f} Mbps ({performance_diff_pct:.1f}% difference)<br>
        • <strong>Linux NFS Advantage:</strong> Significant benefit for Oracle/PostgreSQL backup transfers<br>
        • <strong>Physical Server Advantage:</strong> ~8-12% better than VMware for backup file handling<br>
        • <strong>SQL Server Considerations:</strong> Windows SMB acceptable but consider NFS alternatives
    </div>
    """, unsafe_allow_html=True)

def render_bandwidth_waterfall_comparison(config: Dict, network_perf: Dict, agent_perf: Dict):
    """Render bandwidth waterfall comparison between platforms"""
    st.markdown("**🌊 Platform Performance Comparison: Physical vs VMware**")
    
    # Calculate both physical and VMware scenarios
    base_nic_speed = config['nic_speed']
    
    scenarios = ['Physical Server', 'VMware Platform']
    scenario_data = []
    
    for scenario in scenarios:
        platform_type = 'physical' if scenario == 'Physical Server' else 'vmware'
        
        # Calculate waterfall stages for each scenario
        stages = []
        values = []
        
        # Start with NIC speed
        current_bw = base_nic_speed
        stages.append('NIC Speed')
        values.append(current_bw)
        
        # NIC efficiency
        nic_efficiency = get_nic_efficiency(config['nic_type'])
        current_bw *= nic_efficiency
        stages.append('NIC Processing')
        values.append(current_bw)
        
        # OS efficiency
        os_efficiency = 0.92 if 'linux' in config['operating_system'] else 0.86
        current_bw *= os_efficiency
        stages.append('OS Stack')
        values.append(current_bw)
        
        # Platform efficiency
        platform_efficiency = 1.0 if platform_type == 'physical' else 0.92
        current_bw *= platform_efficiency
        stages.append('Platform')
        values.append(current_bw)
        
        # Storage protocol
        storage_efficiency = agent_perf['io_multiplier']
        current_bw *= storage_efficiency
        stages.append('Storage Protocol')
        values.append(current_bw)
        
        # Agent processing
        agent_efficiency = agent_perf.get('agent_backup_efficiency', 0.9)
        current_bw *= agent_efficiency
        stages.append('DataSync Agent')
        values.append(current_bw)
        
        for stage, value in zip(stages, values):
            scenario_data.append({
                'Platform': scenario,
                'Stage': stage,
                'Throughput (Mbps)': value
            })
    
    df_comparison = pd.DataFrame(scenario_data)
    
    # Create grouped bar chart
    fig_comparison = px.bar(
        df_comparison,
        x='Stage',
        y='Throughput (Mbps)',
        color='Platform',
        title='Performance Degradation: Physical vs VMware Platform',
        barmode='group',
        text='Throughput (Mbps)'
    )
    
    fig_comparison.update_traces(
        texttemplate='%{text:.0f}',
        textposition='outside',
        textfont=dict(size=12)
    )
    
    fig_comparison.update_layout(
        height=500,
        title=dict(font=dict(size=18, family="Arial Black")),
        xaxis=dict(title=dict(text='Performance Stages', font=dict(size=14))),
        yaxis=dict(title=dict(text='Throughput (Mbps)', font=dict(size=14))),
        font=dict(size=12),
        legend=dict(font=dict(size=12))
    )
    
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Summary comparison
    physical_final = df_comparison[(df_comparison['Platform'] == 'Physical Server') & 
                                  (df_comparison['Stage'] == 'DataSync Agent')]['Throughput (Mbps)'].iloc[0]
    vmware_final = df_comparison[(df_comparison['Platform'] == 'VMware Platform') & 
                               (df_comparison['Stage'] == 'DataSync Agent')]['Throughput (Mbps)'].iloc[0]
    
    performance_gain = physical_final - vmware_final
    performance_gain_pct = (performance_gain / vmware_final) * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="network-card">
            <h4>🏢 Physical Server Performance</h4>
            <p><strong>Final Throughput:</strong> {physical_final:.0f} Mbps</p>
            <p><strong>Efficiency:</strong> {(physical_final/base_nic_speed)*100:.1f}%</p>
            <p><strong>Performance Gain:</strong> +{performance_gain:.0f} Mbps vs VMware</p>
            <p><strong>Relative Gain:</strong> +{performance_gain_pct:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="performance-card">
            <h4>☁️ VMware Platform Performance</h4>
            <p><strong>Final Throughput:</strong> {vmware_final:.0f} Mbps</p>
            <p><strong>Efficiency:</strong> {(vmware_final/base_nic_speed)*100:.1f}%</p>
            <p><strong>Virtualization Overhead:</strong> -{(base_nic_speed-vmware_final):.0f} Mbps</p>
            <p><strong>Total Loss:</strong> -{(1-(vmware_final/base_nic_speed))*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

def main():
    """Enhanced main application for database backup migrations using DataSync"""
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style="font-size: 32px; margin-bottom: 15px;">🗄️ AWS DataSync Database Backup Migration Analyzer</h1>
        <p style="font-size: 18px; margin: 0;">Database Backup File Transfers • SQL Server Windows Share • Oracle/PostgreSQL Linux NAS • DataSync Optimization</p>
    </div>
    """, unsafe_allow_html=True)
    
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
    
    # Initialize managers
    network_manager = DatabaseBackupNetworkPathManager()
    agent_manager = EnhancedDataSyncAgentManager()
    
    # Get network path and performance
    path_key = network_manager.get_network_path_key(config)
    network_perf = network_manager.calculate_network_performance(
        path_key, 
        backup_compression=config['backup_compression_ratio']
    )
    
    # Storage characteristics
    storage_type_mapping = {
        'windows_share_drive': 'share',
        'linux_nas_drive': 'nas',
        'local_storage': 'local'
    }
    storage_type = storage_type_mapping.get(config['storage_type'], 'nas')
    os_type = 'linux' if 'linux' in config['operating_system'].lower() else 'windows'
    
    # Get agent performance
    agent_perf = agent_manager.calculate_agent_performance(
        config['agent_size'], 
        config['number_of_agents'], 
        config['server_type'],
        config['source_database_engine'],
        storage_type, 
        os_type, 
        'centralized_datacenter',
        config['backup_compression_ratio']
    )
    
    # Display database backup overview
    render_database_backup_overview(config)
    
    # Display backup transfer metrics
    render_backup_transfer_metrics(config, network_perf, agent_perf)
    
    # Enhanced tabs for database backup migration
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🎯 DataSync Agent Placement",
        "🗄️ Database Backup Analysis",
        "📊 Storage Performance Comparison", 
        "🌐 Network Path Visualization",
        "🤖 DataSync Configuration",
        "☁️ AWS Integration",
        "🧠 AI Analysis"
    ])
    
    with tab1:
        st.subheader("🎯 DataSync Agent Placement for Database Backups")
        
        # Agent placement analysis for backup scenarios
        placement_analyzer = DatabasePlacementAnalyzer()
        placement_options = placement_analyzer.analyze_placement_options(config, network_perf, agent_manager)
        
        # Render placement options specific to backup scenarios
        st.markdown("### 📍 Backup Access Strategy Comparison")
        
        for i, option in enumerate(placement_options):
            is_recommended = i == 0
            
            if option['placement_score'] >= 80:
                score_class = "excellent"
                card_class = "database-card"
            elif option['placement_score'] >= 65:
                score_class = "good"
                card_class = "backup-card"
            else:
                score_class = "poor"
                card_class = "warning-card"
            
            st.markdown(f"""
            <div class="placement-option {'recommended' if is_recommended else ''}">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                    <h4 style="font-size: 18px; font-weight: bold; margin: 0;">{option['strategy']['name']}</h4>
                    <span class="placement-score {score_class}">{option['placement_score']:.1f}/100</span>
                </div>
                
                <p style="font-size: 15px; color: #6b7280; margin-bottom: 1rem;">{option['strategy']['description']}</p>
                
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 1rem;">
                    <div>
                        <strong>Backup Access Performance:</strong><br>
                        {option['throughput_mbps']:,.0f} Mbps<br>
                        <small>Backup Access Efficiency: {option['backup_access_efficiency']*100:.0f}%</small>
                    </div>
                    <div>
                        <strong>Transfer Cost:</strong><br>
                        ${option['monthly_cost']:,.0f}/month<br>
                        <small>${option['monthly_cost']/option['throughput_mbps']:.2f}/Mbps</small>
                    </div>
                    <div>
                        <strong>Security:</strong><br>
                        {option['security_score']*100:.0f}% rating<br>
                        <small>Backup access security</small>
                    </div>
                    <div>
                        <strong>Management:</strong><br>
                        {option['management_complexity']*100:.0f}% complexity<br>
                        <small>Operational overhead</small>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Expandable details
            with st.expander(f"📋 {option['strategy']['name']} - Implementation Details", expanded=is_recommended):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**✅ Advantages for Backup Scenarios:**")
                    for pro in option['strategy']['pros']:
                        st.markdown(f"• {pro}")
                
                with col2:
                    st.markdown("**⚠️ Challenges:**")
                    for con in option['strategy']['cons']:
                        st.markdown(f"• {con}")
                
                # Implementation timeline
                complexity = option['implementation_complexity']
                st.markdown(f"""
                **🔧 Implementation Details:**
                - **Setup Time:** {complexity['setup_time_days']} days
                - **Skill Level Required:** {complexity['skill_level']}
                - **Backup Coordination:** {complexity['backup_coordination']}
                """)
    
    with tab2:
        st.subheader("🗄️ Database Backup Analysis & Optimization")
        
        # DataSync configuration for backups
        render_datasync_backup_configuration(config, agent_perf)
        
        # Database-specific storage comparison
        render_database_storage_comparison(config)
        
        # Backup file transfer timeline
        st.markdown("**⏱️ Backup Transfer Timeline Analysis**")
        
        backup_size_gb = config['backup_size_gb']
        compression_ratio = config['backup_compression_ratio']
        effective_size = backup_size_gb * compression_ratio
        throughput_mbps = min(network_perf['effective_bandwidth_mbps'], agent_perf['total_agent_throughput_mbps'])
        
        # Calculate timeline for different backup sizes
        backup_sizes = [100, 500, 1000, 2500, 5000, 10000]  # GB
        timeline_data = []
        
        for size in backup_sizes:
            effective_backup_size = size * compression_ratio
            backup_size_mbits = effective_backup_size * 8 * 1000
            transfer_seconds = backup_size_mbits / throughput_mbps
            transfer_hours = transfer_seconds / 3600
            
            timeline_data.append({
                'Backup Size (GB)': size,
                'Effective Size (GB)': effective_backup_size,
                'Transfer Time (Hours)': transfer_hours,
                'Transfer Time (Minutes)': transfer_seconds / 60
            })
        
        df_timeline = pd.DataFrame(timeline_data)
        
        fig_timeline = px.line(
            df_timeline,
            x='Backup Size (GB)',
            y='Transfer Time (Hours)',
            title=f'Backup Transfer Time vs Size ({config["source_database_engine"].upper()} with {compression_ratio*100:.0f}% compression)',
            markers=True
        )
        
        fig_timeline.update_layout(
            height=400,
            title=dict(font=dict(size=18, family="Arial Black")),
            xaxis=dict(title=dict(text='Backup Size (GB)', font=dict(size=14))),
            yaxis=dict(title=dict(text='Transfer Time (Hours)', font=dict(size=14))),
            font=dict(size=12)
        )
        
        st.plotly_chart(fig_timeline, use_container_width=True)
    
    with tab3:
        st.subheader("📊 Database Storage Performance Comparison")
        
        # Render database-specific storage comparison (moved to tab2, but kept for backwards compatibility)
        render_database_storage_comparison(config)
        
        # SQL Server vs Linux database comparison
        st.markdown("**⚖️ SQL Server (Windows) vs Linux Database Comparison**")
        
        comparison_scenarios = [
            {
                'scenario': 'SQL Server on Windows Share (SMB)',
                'throughput': 750,
                'compression': 0.7,
                'efficiency': 75,
                'pros': ['Native Windows integration', 'Familiar tools', 'Enterprise support'],
                'cons': ['SMB protocol overhead', 'Lower compression ratios', 'Windows licensing costs']
            },
            {
                'scenario': 'Oracle on Linux NAS (NFS)',
                'throughput': 960,
                'compression': 0.6,
                'efficiency': 96,
                'pros': ['Excellent NFS performance', 'Superior compression', 'Lower overhead'],
                'cons': ['Linux expertise required', 'More complex backup tools', 'Initial setup complexity']
            },
            {
                'scenario': 'PostgreSQL on Linux NAS (NFS)',
                'throughput': 980,
                'compression': 0.5,
                'efficiency': 98,
                'pros': ['Best compression ratios', 'Text-based dumps', 'Excellent performance'],
                'cons': ['Custom format complexity', 'Parallel dump setup', 'Recovery planning']
            }
        ]
        
        for scenario in comparison_scenarios:
            is_current = config['source_database_engine'] in scenario['scenario'].lower()
            card_class = "database-card" if is_current else "backup-card"
            
            st.markdown(f"""
            <div class="{card_class}">
                <h4 style="font-size: 17px; font-weight: bold; margin-bottom: 15px;">
                    {'🎯 ' if is_current else ''}{"scenario['scenario']}"} 
                    {'(Current Configuration)' if is_current else ''}
                </h4>
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; margin-bottom: 15px;">
                    <div><strong>Throughput:</strong> {scenario['throughput']} Mbps</div>
                    <div><strong>Compression:</strong> {scenario['compression']*100:.0f}% ratio</div>
                    <div><strong>Efficiency:</strong> {scenario['efficiency']}%</div>
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                    <div>
                        <strong>Advantages:</strong>
                        <ul style="margin: 5px 0; padding-left: 20px;">
                            {chr(10).join([f'<li style="font-size: 13px;">{pro}</li>' for pro in scenario['pros']])}
                        </ul>
                    </div>
                    <div>
                        <strong>Considerations:</strong>
                        <ul style="margin: 5px 0; padding-left: 20px;">
                            {chr(10).join([f'<li style="font-size: 13px;">{con}</li>' for con in scenario['cons']])}
                        </ul>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with tab4:
        st.subheader("🌐 Network Path Visualization for Database Backups")
        
        # Network path specific to backup scenarios
        st.markdown(f"**🗺️ {network_perf['path_name']}**")
        
        # Enhanced network segments for backup scenarios
        segments_data = []
        for segment in network_perf['segments']:
            segments_data.append({
                'Segment': segment['name'],
                'Bandwidth (Mbps)': segment['effective_bandwidth_mbps'],
                'Latency (ms)': segment['effective_latency_ms'],
                'Reliability (%)': segment['reliability'] * 100,
                'Connection Type': segment['connection_type'].replace('_', ' ').title()
            })
        
        df_segments = pd.DataFrame(segments_data)
        
        # Create network performance chart
        fig_network = px.bar(
            df_segments,
            x='Segment',
            y='Bandwidth (Mbps)',
            title='Database Backup Network Path Performance',
            color='Reliability (%)',
            color_continuous_scale='RdYlGn',
            text='Bandwidth (Mbps)',
            hover_data=['Latency (ms)', 'Connection Type']
        )
        
        fig_network.update_traces(
            texttemplate='%{text:.0f}',
            textposition='outside'
        )
        
        fig_network.update_layout(
            height=500,
            xaxis_tickangle=-45,
            title=dict(font=dict(size=18, family="Arial Black")),
            xaxis=dict(title=dict(text='Network Segments', font=dict(size=14))),
            yaxis=dict(title=dict(text='Effective Bandwidth (Mbps)', font=dict(size=14))),
            font=dict(size=12)
        )
        
        st.plotly_chart(fig_network, use_container_width=True)
        
        # Network path details
        st.markdown("**📋 Network Path Analysis:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="network-card">
                <h4>🌐 Path Characteristics</h4>
                <p><strong>Database:</strong> {network_perf['database_type'].upper()}</p>
                <p><strong>Storage Protocol:</strong> {network_perf['storage_mount_type'].upper()}</p>
                <p><strong>Backup Location:</strong> {network_perf['backup_location'].replace('_', ' ').title()}</p>
                <p><strong>Total Latency:</strong> {network_perf['total_latency_ms']:.1f} ms</p>
                <p><strong>Effective Bandwidth:</strong> {network_perf['effective_bandwidth_mbps']:,.0f} Mbps</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="performance-card">
                <h4>📊 Performance Metrics</h4>
                <p><strong>Network Quality:</strong> {network_perf['network_quality_score']:.1f}/100</p>
                <p><strong>Reliability:</strong> {network_perf['total_reliability']*100:.2f}%</p>
                <p><strong>Optimization Potential:</strong> {network_perf['optimization_potential']:.1f}%</p>
                <p><strong>Environment:</strong> {network_perf['environment'].title()}</p>
                <p><strong>Source Location:</strong> {network_perf.get('source', 'Unknown')}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab5:
        st.subheader("🤖 DataSync Agent Configuration & Performance")
        
        # Enhanced DataSync metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">🤖 {agent_perf['num_agents']}x {agent_perf['agent_size'].title()}</div>
                <div class="metric-label">DataSync Agents</div>
                <div style="font-size: 12px; color: #6b7280;">Backup file optimized</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">⚡ {agent_perf['total_agent_throughput_mbps']:,.0f} Mbps</div>
                <div class="metric-label">Total Capacity</div>
                <div style="font-size: 12px; color: #dc2626;">{agent_perf['performance_loss_pct']:.1f}% loss from ideal</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">📁 {agent_perf['backup_file_efficiency']*100:.1f}%</div>
                <div class="metric-label">Backup File Efficiency</div>
                <div style="font-size: 12px; color: #6b7280;">Large file handling</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">💰 ${agent_perf['total_monthly_cost']:,.0f}</div>
                <div class="metric-label">Monthly Cost</div>
                <div style="font-size: 12px; color: #6b7280;">${agent_perf['per_agent_monthly_cost']:.0f}/agent</div>
            </div>
            """, unsafe_allow_html=True)
        
        # DataSync agent performance breakdown
        st.markdown("**📈 DataSync Performance Factor Analysis**")
        
        performance_factors = {
            'Factor': [
                'Base Throughput',
                'Platform Efficiency', 
                'Backup File Handling',
                'I/O Protocol',
                'Network Efficiency',
                'Agent Backup Optimization',
                'Compression Impact',
                'Scaling Factor'
            ],
            'Impact (%)': [
                100,
                agent_perf['platform_efficiency'] * 100,
                agent_perf['backup_file_efficiency'] * 100,
                agent_perf['io_multiplier'] * 100,
                agent_perf['network_efficiency'] * 100,
                agent_perf['agent_backup_efficiency'] * 100,
                agent_perf['compression_impact'] * 100,
                agent_perf['scaling_efficiency'] * 100
            ]
        }
        
        df_factors = pd.DataFrame(performance_factors)
        
        fig_factors = px.bar(
            df_factors,
            x='Factor',
            y='Impact (%)',
            title='DataSync Agent Performance Factor Analysis',
            color='Impact (%)',
            color_continuous_scale='RdYlGn',
            text='Impact (%)'
        )
        
        fig_factors.update_traces(
            texttemplate='%{text:.1f}%',
            textposition='outside'
        )
        
        fig_factors.update_layout(
            height=500,
            xaxis_tickangle=-45,
            title=dict(font=dict(size=18, family="Arial Black")),
            xaxis=dict(title=dict(text='Performance Factors', font=dict(size=14))),
            yaxis=dict(title=dict(text='Efficiency Impact (%)', font=dict(size=14))),
            font=dict(size=12)
        )
        
        st.plotly_chart(fig_factors, use_container_width=True)
        
        # Database-specific recommendations
        backup_info = get_database_backup_info(config['source_database_engine'])
        
        st.markdown(f"""
        <div class="database-card">
            <h4>💡 {config['source_database_engine'].upper()}-Specific DataSync Recommendations</h4>
            <div style="font-size: 14px; line-height: 1.7;">
                <p><strong>Recommended Tools:</strong> {', '.join(backup_info['backup_tools'][:3])}</p>
                <p><strong>Optimal File Extensions:</strong> {', '.join(backup_info['typical_backup_extensions'])}</p>
                <p><strong>Compression Recommendation:</strong> {'Enabled' if backup_info['compression_available'] else 'Not Available'} 
                   ({backup_info['typical_compression_ratio']*100:.0f}% typical ratio)</p>
                <p><strong>Storage Preference:</strong> {backup_info['preferred_storage']}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with tab6:
        st.subheader("☁️ AWS Integration & Real-Time Monitoring")
        
        aws_integration = st.session_state.get('aws_integration')
        if aws_integration and aws_integration.session:
            # Render AWS integration panel (reuse existing function)
            aws_data_detailed = render_aws_integration_panel(aws_integration)
            
            # DataSync-specific insights for backup transfers
            st.markdown("**📊 DataSync Backup Transfer Insights**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="aws-card">
                    <h4>🔄 DataSync Backup Configuration</h4>
                    <div style="font-size: 14px; line-height: 1.7;">
                        <p><strong>Source Type:</strong> {config['source_database_engine'].upper()} Backup Files</p>
                        <p><strong>Source Storage:</strong> {config['storage_type'].replace('_', ' ').title()}</p>
                        <p><strong>Target:</strong> AWS S3</p>
                        <p><strong>Transfer Method:</strong> AWS DataSync</p>
                        <p><strong>Compression:</strong> {'Enabled' if config['backup_compression'] else 'Disabled'}</p>
                        <p><strong>Agent Count:</strong> {config['number_of_agents']} {config['agent_size']} agents</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                final_throughput = min(network_perf['effective_bandwidth_mbps'], agent_perf['total_agent_throughput_mbps'])
                transfer_time_hours = (config['backup_size_gb'] * config['backup_compression_ratio'] * 8 * 1000) / (final_throughput * 3600)
                
                st.markdown(f"""
                <div class="backup-card">
                    <h4>⏱️ Transfer Estimates</h4>
                    <div style="font-size: 14px; line-height: 1.7;">
                        <p><strong>Backup Size:</strong> {config['backup_size_gb']:,} GB</p>
                        <p><strong>Compressed Size:</strong> {config['backup_size_gb'] * config['backup_compression_ratio']:,.0f} GB</p>
                        <p><strong>Effective Throughput:</strong> {final_throughput:,.0f} Mbps</p>
                        <p><strong>Estimated Transfer Time:</strong> {transfer_time_hours:.1f} hours</p>
                        <p><strong>Daily Backup Window:</strong> {'Fits in' if transfer_time_hours <= 8 else 'Exceeds'} 8-hour window</p>
                        <p><strong>Weekly Full Backup:</strong> {'Feasible' if transfer_time_hours <= 24 else 'Challenging'}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="warning-card">
                <strong>⚠️ AWS integration not connected.</strong><br>
                Configure AWS credentials in the sidebar to monitor real-time DataSync tasks and get live performance metrics.
            </div>
            """, unsafe_allow_html=True)
    
    with tab7:
        st.subheader("🧠 AI-Powered Database Backup Migration Analysis")
        
        claude_integration = st.session_state.get('claude_integration')
        if claude_integration and claude_integration.client:
            try:
                placement_analyzer = DatabasePlacementAnalyzer()
                placement_options = placement_analyzer.analyze_placement_options(config, network_perf, agent_manager)
                
                with st.spinner("🔄 Analyzing database backup migration configuration..."):
                    analysis = claude_integration.analyze_database_backup_migration(
                        config, network_perf, agent_perf, placement_options[0], {}
                    )
                
                st.markdown(f"""
                <div class="ai-section">
                    <h4>🧠 Comprehensive Database Backup Migration Analysis</h4>
                    <div style="font-size: 15px; line-height: 1.8; color: #374151;">{analysis.replace(chr(10), '<br>')}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Database-specific recommendations
                with st.expander(f"💡 {config['source_database_engine'].upper()}-Specific Optimization Recommendations", expanded=False):
                    with st.spinner("🔄 Getting database-specific recommendations..."):
                        db_recommendations = claude_integration.get_database_specific_recommendations(
                            config['source_database_engine'], config
                        )
                    
                    st.markdown(f"""
                    <div class="ai-section">
                        <h4>🎯 {config['source_database_engine'].upper()} Backup Migration Optimization</h4>
                        <div style="font-size: 15px; line-height: 1.8; color: #374151;">{db_recommendations.replace(chr(10), '<br>')}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Agent placement recommendations
                with st.expander("📍 DataSync Agent Placement Recommendations", expanded=False):
                    with st.spinner("🔄 Analyzing optimal agent placement..."):
                        placement_recommendations = claude_integration.get_placement_recommendations(config, placement_options)
                    
                    st.markdown(f"""
                    <div class="ai-section">
                        <h4>🎯 Strategic Agent Placement for Backup Access</h4>
                        <div style="font-size: 15px; line-height: 1.8; color: #374151;">{placement_recommendations.replace(chr(10), '<br>')}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
            except Exception as e:
                st.markdown(f"""
                <div class="warning-card">
                    <strong>⚠️ Error during AI analysis: {str(e)}</strong><br>
                    <small>Claude AI analysis may not be fully available. Check your API configuration.</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="warning-card">
                <strong>ℹ️ Enhanced AI Analysis</strong><br>
                Claude AI integration not connected. Connect Claude AI to get intelligent database backup migration analysis, 
                including database-specific recommendations and optimal DataSync agent placement strategies.
            </div>
            """, unsafe_allow_html=True)
    
    # Enhanced executive summary for database backup scenarios
    st.markdown("---")
    st.markdown("### 🎯 Executive Summary: Database Backup Migration Strategy")
    
    placement_analyzer = DatabasePlacementAnalyzer()
    placement_options = placement_analyzer.analyze_placement_options(config, network_perf, agent_manager)
    recommended_option = placement_options[0]
    
    final_throughput = recommended_option['throughput_mbps']
    backup_size_compressed = config['backup_size_gb'] * config['backup_compression_ratio']
    transfer_time_hours = (backup_size_compressed * 8 * 1000) / (final_throughput * 3600)
    transfer_cost = (recommended_option['monthly_cost'] / (30 * 24)) * transfer_time_hours
    
    summary_col1, summary_col2, summary_col3 = st.columns(3)
    
    with summary_col1:
        backup_window_status = "✅ Fits" if transfer_time_hours <= 8 else "⚠️ Exceeds"
        daily_backup_feasible = transfer_time_hours <= 4
        
        st.markdown(f"""
        <div class="database-card">
            <h4>🗄️ Database Backup Summary</h4>
            <div style="font-size: 15px; line-height: 1.8;">
                <p><strong>Database:</strong> {config['source_database_engine'].upper()}</p>
                <p><strong>Backup Size:</strong> {config['backup_size_gb']:,} GB → {backup_size_compressed:,.0f} GB</p>
                <p><strong>Storage:</strong> {config['storage_type'].replace('_', ' ').title()}</p>
                <p><strong>Transfer Time:</strong> {transfer_time_hours:.1f} hours</p>
                <p><strong>8-Hour Window:</strong> {backup_window_status}</p>
                <p><strong>Daily Backup:</strong> {'✅ Feasible' if daily_backup_feasible else '⚠️ Challenging'}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with summary_col2:
        st.markdown(f"""
        <div class="backup-card">
            <h4>🎯 Recommended Strategy</h4>
            <div style="font-size: 15px; line-height: 1.8;">
                <p><strong>Placement:</strong> {recommended_option['strategy']['name']}</p>
                <p><strong>DataSync Agents:</strong> {config['number_of_agents']}x {config['agent_size']}</p>
                <p><strong>Expected Throughput:</strong> {final_throughput:,.0f} Mbps</p>
                <p><strong>Performance Score:</strong> {recommended_option['placement_score']:.1f}/100</p>
                <p><strong>Backup Access Efficiency:</strong> {recommended_option['backup_access_efficiency']*100:.0f}%</p>
                <p><strong>Security Rating:</strong> {recommended_option['security_score']*100:.0f}%</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with summary_col3:
        cost_per_gb = transfer_cost / config['backup_size_gb']
        monthly_backup_cost = transfer_cost * 30  # Assuming daily backups
        
        st.markdown(f"""
        <div class="performance-card">
            <h4>💰 Cost Analysis</h4>
            <div style="font-size: 15px; line-height: 1.8;">
                <p><strong>Transfer Cost:</strong> ${transfer_cost:.2f}</p>
                <p><strong>Cost per GB:</strong> ${cost_per_gb:.4f}</p>
                <p><strong>Monthly (Daily Backups):</strong> ${monthly_backup_cost:.0f}</p>
                <p><strong>Annual Backup Costs:</strong> ${monthly_backup_cost * 12:.0f}</p>
                <p><strong>Setup Time:</strong> {recommended_option['implementation_complexity']['setup_time_days']} days</p>
                <p><strong>ROI Timeline:</strong> {'Immediate' if transfer_time_hours <= 8 else 'After optimization'}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Implementation roadmap
    st.markdown("### 🛣️ Implementation Roadmap")
    
    roadmap_phases = [
        {
            'phase': 'Phase 1: Infrastructure Setup',
            'duration': '1-2 days',
            'tasks': [
                f'Configure {config["source_database_engine"].upper()} backup processes',
                f'Set up {config["storage_type"].replace("_", " ").title()} storage access',
                'Install and configure DataSync agents',
                'Establish network connectivity to AWS S3'
            ]
        },
        {
            'phase': 'Phase 2: Testing & Validation',
            'duration': '2-3 days', 
            'tasks': [
                'Test backup file access and transfer speeds',
                'Validate compression and encryption settings',
                'Perform test transfers with sample backup files',
                'Monitor performance and adjust agent configurations'
            ]
        },
        {
            'phase': 'Phase 3: Production Implementation',
            'duration': '1-2 days',
            'tasks': [
                'Schedule production backup transfers',
                'Implement monitoring and alerting',
                'Document operational procedures',
                'Train database team on new processes'
            ]
        }
    ]
    
    for i, phase in enumerate(roadmap_phases):
        with st.expander(f"📋 {phase['phase']} ({phase['duration']})", expanded=i==0):
            st.markdown("**Key Tasks:**")
            for task in phase['tasks']:
                st.markdown(f"• {task}")
    
    # Final recommendations banner
    database_type = config['source_database_engine']
    storage_type = config['storage_type']
    
    if database_type == 'sqlserver' and 'windows' in storage_type:
        recommendation_color = "#0ea5e9"
        recommendation_text = "SQL Server backups on Windows Share Drive are well-supported. Consider enabling backup compression and using multiple DataSync agents for large databases."
    elif database_type in ['oracle', 'postgresql'] and 'linux' in storage_type:
        recommendation_color = "#22c55e"
        recommendation_text = f"{database_type.upper()} backups on Linux NAS provide excellent performance. Take advantage of superior compression ratios and NFS efficiency."
    else:
        recommendation_color = "#f59e0b"
        recommendation_text = "Consider aligning your database type with the recommended storage platform for optimal performance."
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid {recommendation_color}; border-radius: 12px; padding: 25px; margin: 25px 0; text-align: center;">
        <h3 style="color: {recommendation_color}; font-size: 20px; margin-bottom: 15px;">🚀 Final Recommendation</h3>
        <p style="font-size: 16px; line-height: 1.8; color: #1e293b; margin: 0;">{recommendation_text}</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()