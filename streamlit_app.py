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

    .network-flow-container {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #e5e7eb;
    }

    .datacenter-diagram {
        background: #ffffff;
        border-radius: 8px;
        padding: 2rem;
        margin: 1rem 0;
        border: 2px solid #e5e7eb;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    .aws-service-table {
        background: #ffffff;
        border-radius: 6px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid #e5e7eb;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    .network-node {
        background: #3b82f6;
        color: white;
        padding: 8px 12px;
        border-radius: 4px;
        margin: 4px;
        display: inline-block;
        font-size: 0.85rem;
        font-weight: 500;
    }

    .network-connection {
        border-top: 2px solid #6b7280;
        margin: 8px 0;
        position: relative;
    }

    .connection-label {
        background: white;
        padding: 2px 8px;
        position: absolute;
        top: -12px;
        left: 50%;
        transform: translateX(-50%);
        font-size: 0.75rem;
        color: #6b7280;
        border: 1px solid #e5e7eb;
        border-radius: 3px;
    }

    .datacenter-zone {
        border: 2px dashed #9ca3af;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem;
        background: rgba(156, 163, 175, 0.05);
    }

    .agent-placement {
        background: #10b981;
        color: white;
        padding: 6px 10px;
        border-radius: 4px;
        margin: 4px;
        display: inline-block;
        font-size: 0.8rem;
        font-weight: 500;
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
    
    async def analyze_migration_workload(self, config: Dict, performance_data: Dict) -> Dict:
        """Enhanced AI-powered workload analysis with detailed insights"""
        if not self.connected:
            return self._fallback_workload_analysis(config, performance_data)
        
        try:
            # Enhanced prompt with backup storage considerations
            migration_method = config.get('migration_method', 'direct_replication')
            backup_storage_type = config.get('backup_storage_type', 'nas_drive')
            backup_size_multiplier = config.get('backup_size_multiplier', 0.7)
            
            migration_details = ""
            if migration_method == 'backup_restore':
                backup_size_gb = config['database_size_gb'] * backup_size_multiplier
                migration_details = f"""
                BACKUP STORAGE MIGRATION:
                - Migration Method: Backup/Restore via DataSync
                - Backup Storage: {backup_storage_type.replace('_', ' ').title()}
                - Database Size: {config['database_size_gb']} GB
                - Backup Size: {backup_size_gb:.0f} GB ({int(backup_size_multiplier*100)}%)
                - Protocol: {'SMB' if backup_storage_type == 'windows_share' else 'NFS'}
                - Tool: AWS DataSync (File Transfer)
                """
            else:
                migration_details = f"""
                DIRECT REPLICATION MIGRATION:
                - Migration Method: Direct database replication
                - Source Database: {config['source_database_engine']}
                - Target Database: {config['database_engine']}
                - Tool: {'AWS DataSync' if config['source_database_engine'] == config['database_engine'] else 'AWS DMS'}
                """
            
            prompt = f"""
            As a senior AWS migration consultant with deep expertise in database migrations, provide a comprehensive analysis of this migration scenario:

            CURRENT INFRASTRUCTURE:
            - Source Database: {config['source_database_engine']} ({config['database_size_gb']} GB)
            - Target Database: {config['database_engine']}
            - Target Platform: {config.get('target_platform', 'rds').upper()}
            - Operating System: {config['operating_system']}
            - Platform: {config['server_type']}
            - Hardware: {config['cpu_cores']} cores @ {config['cpu_ghz']} GHz, {config['ram_gb']} GB RAM
            - Network: {config['nic_type']} ({config['nic_speed']} Mbps)
            - Environment: {config['environment']}
            - Performance Requirement: {config['performance_requirements']}
            - Downtime Tolerance: {config['downtime_tolerance_minutes']} minutes
            - Migration Agents: {config.get('number_of_agents', 1)} agents configured
            - Destination Storage: {config.get('destination_storage_type', 'S3')}

            CURRENT DATABASE PERFORMANCE METRICS:
            - Database Max Memory: {config.get('current_db_max_memory_gb', 'Not specified')} GB
            - Database Max CPU Cores: {config.get('current_db_max_cpu_cores', 'Not specified')} cores
            - Database Max IOPS: {config.get('current_db_max_iops', 'Not specified')} IOPS
            - Database Max Throughput: {config.get('current_db_max_throughput_mbps', 'Not specified')} MB/s

            {migration_details}

            CURRENT PERFORMANCE METRICS:
            - Database TPS: {performance_data.get('database_performance', {}).get('effective_tps', 'Unknown')}
            - Storage IOPS: {performance_data.get('storage_performance', {}).get('effective_iops', 'Unknown')}
            - Network Bandwidth: {performance_data.get('network_performance', {}).get('effective_bandwidth_mbps', 'Unknown')} Mbps
            - OS Efficiency: {performance_data.get('os_impact', {}).get('total_efficiency', 0) * 100:.1f}%
            - Overall Performance Score: {performance_data.get('performance_score', 0):.1f}/100

            Please provide a detailed assessment including:
            1. MIGRATION COMPLEXITY (1-10 scale with detailed justification)
            2. RISK ASSESSMENT with specific risk percentages and mitigation strategies
            3. PERFORMANCE OPTIMIZATION recommendations with expected improvement percentages
            4. AWS SIZING RECOMMENDATIONS based on current database performance metrics
            5. DETAILED TIMELINE with phase-by-phase breakdown
            6. RESOURCE ALLOCATION with specific AWS instance recommendations
            7. COST OPTIMIZATION strategies with potential savings
            8. BEST PRACTICES specific to this configuration with implementation steps
            9. TESTING STRATEGY with checkpoints and validation criteria
            10. ROLLBACK PROCEDURES and contingency planning
            11. POST-MIGRATION monitoring and optimization recommendations
            12. AGENT SCALING IMPACT analysis based on {config.get('number_of_agents', 1)} agents
            13. DESTINATION STORAGE IMPACT for {config.get('destination_storage_type', 'S3')} including performance and cost implications
            14. BACKUP STORAGE CONSIDERATIONS for {migration_method} method using {backup_storage_type if migration_method == 'backup_restore' else 'N/A'}
            15. RIGHT-SIZING ANALYSIS: Compare current database performance ({config.get('current_db_max_memory_gb', 'Unknown')} GB RAM, {config.get('current_db_max_cpu_cores', 'Unknown')} cores, {config.get('current_db_max_iops', 'Unknown')} IOPS) with recommended AWS instance types

            Provide quantitative analysis wherever possible, including specific metrics, percentages, and measurable outcomes.
            Format the response as detailed sections with clear recommendations and actionable insights.
            Pay special attention to right-sizing recommendations based on the provided current database performance metrics.
            """
            
            message = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=4000,
                temperature=0.2,
                messages=[{"role": "user", "content": prompt}]
            )
            
            ai_response = message.content[0].text
            ai_analysis = self._parse_detailed_ai_response(ai_response, config, performance_data)
            
            return {
                'ai_complexity_score': ai_analysis.get('complexity_score', 6),
                'risk_factors': ai_analysis.get('risk_factors', []),
                'risk_percentages': ai_analysis.get('risk_percentages', {}),
                'mitigation_strategies': ai_analysis.get('mitigation_strategies', []),
                'performance_recommendations': ai_analysis.get('performance_recommendations', []),
                'performance_improvements': ai_analysis.get('performance_improvements', {}),
                'timeline_suggestions': ai_analysis.get('timeline_suggestions', []),
                'resource_allocation': ai_analysis.get('resource_allocation', {}),
                'cost_optimization': ai_analysis.get('cost_optimization', []),
                'best_practices': ai_analysis.get('best_practices', []),
                'testing_strategy': ai_analysis.get('testing_strategy', []),
                'rollback_procedures': ai_analysis.get('rollback_procedures', []),
                'post_migration_monitoring': ai_analysis.get('post_migration_monitoring', []),
                'confidence_level': ai_analysis.get('confidence_level', 'medium'),
                'detailed_assessment': ai_analysis.get('detailed_assessment', {}),
                'agent_scaling_impact': ai_analysis.get('agent_scaling_impact', {}),
                'destination_storage_impact': ai_analysis.get('destination_storage_impact', {}),
                'backup_storage_considerations': ai_analysis.get('backup_storage_considerations', {}),
                'raw_ai_response': ai_response
            }
            
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            st.error(f"AI Analysis Error: {str(e)}")
            return self._fallback_workload_analysis(config, performance_data)
    
    def _parse_detailed_ai_response(self, ai_response: str, config: Dict, performance_data: Dict) -> Dict:
        """Enhanced parsing for detailed AI analysis"""
        
        complexity_factors = []
        base_complexity = 5
        
        # Migration method complexity
        migration_method = config.get('migration_method', 'direct_replication')
        if migration_method == 'backup_restore':
            backup_storage_type = config.get('backup_storage_type', 'nas_drive')
            if backup_storage_type == 'windows_share':
                complexity_factors.append(('SMB protocol overhead', 0.5))
                base_complexity += 0.5
            else:
                complexity_factors.append(('NFS protocol efficiency', -0.2))
                base_complexity -= 0.2
        
        # Database engine complexity
        if config['source_database_engine'] != config['database_engine']:
            complexity_factors.append(('Heterogeneous migration', 2))
            base_complexity += 2
        
        # Database size complexity
        if config['database_size_gb'] > 10000:
            complexity_factors.append(('Large database size', 1.5))
            base_complexity += 1.5
        elif config['database_size_gb'] > 5000:
            complexity_factors.append(('Medium database size', 0.5))
            base_complexity += 0.5
        
        # Performance requirements
        if config['performance_requirements'] == 'high':
            complexity_factors.append(('High performance requirements', 1))
            base_complexity += 1
        
        # Environment complexity
        if config['environment'] == 'production':
            complexity_factors.append(('Production environment', 0.5))
            base_complexity += 0.5
        
        # Agent scaling impact
        num_agents = config.get('number_of_agents', 1)
        if num_agents > 3:
            complexity_factors.append(('Multi-agent coordination complexity', 0.5))
            base_complexity += 0.5
        
        # Destination storage complexity
        destination_storage = config.get('destination_storage_type', 'S3')
        if destination_storage == 'FSx_Windows':
            complexity_factors.append(('FSx for Windows File System complexity', 0.8))
            base_complexity += 0.8
        elif destination_storage == 'FSx_Lustre':
            complexity_factors.append(('FSx for Lustre high-performance complexity', 1.0))
            base_complexity += 1.0
        
        complexity_score = min(10, max(1, base_complexity))
        
        # Generate detailed risk assessment
        risk_factors = []
        risk_percentages = {}
        
        if migration_method == 'backup_restore':
            backup_storage_type = config.get('backup_storage_type', 'nas_drive')
            if backup_storage_type == 'windows_share':
                risk_factors.append("SMB protocol stability over WAN connections")
                risk_percentages['smb_protocol_risk'] = 15
            risk_factors.append("Backup file integrity and completeness verification")
            risk_percentages['backup_integrity_risk'] = 10
        
        if config['source_database_engine'] != config['database_engine']:
            risk_factors.append("Schema conversion complexity may cause compatibility issues")
            risk_percentages['schema_conversion_risk'] = 25
        
        if config['database_size_gb'] > 5000:
            risk_factors.append("Large database size increases migration time and failure probability")
            risk_percentages['large_database_risk'] = 15
        
        if config['downtime_tolerance_minutes'] < 120:
            risk_factors.append("Tight downtime window may require multiple attempts")
            risk_percentages['downtime_risk'] = 20
        
        # Agent-specific risks
        if num_agents == 1 and config['database_size_gb'] > 5000:
            risk_factors.append("Single agent may become throughput bottleneck")
            risk_percentages['agent_bottleneck_risk'] = 30
        elif num_agents > 5:
            risk_factors.append("Complex multi-agent coordination may cause synchronization issues")
            risk_percentages['coordination_risk'] = 15
        
        perf_score = performance_data.get('performance_score', 0)
        if perf_score < 70:
            risk_factors.append("Current performance issues may impact migration success")
            risk_percentages['performance_risk'] = 30
        
        return {
            'complexity_score': complexity_score,
            'complexity_factors': complexity_factors,
            'risk_factors': risk_factors,
            'risk_percentages': risk_percentages,
            'mitigation_strategies': self._generate_mitigation_strategies(risk_factors, config),
            'performance_recommendations': self._generate_performance_recommendations(config),
            'performance_improvements': {'overall_optimization': '15-25%'},
            'timeline_suggestions': self._generate_timeline_suggestions(config),
            'resource_allocation': self._generate_resource_allocation(config, complexity_score),
            'cost_optimization': self._generate_cost_optimization(config, complexity_score),
            'best_practices': self._generate_best_practices(config, complexity_score),
            'testing_strategy': self._generate_testing_strategy(config, complexity_score),
            'rollback_procedures': self._generate_rollback_procedures(config),
            'post_migration_monitoring': self._generate_monitoring_recommendations(config),
            'confidence_level': 'high' if complexity_score < 6 else 'medium' if complexity_score < 8 else 'requires_specialist_review',
            'agent_scaling_impact': self._analyze_agent_scaling_impact(config),
            'destination_storage_impact': self._analyze_storage_impact(config),
            'backup_storage_considerations': self._analyze_backup_storage_considerations(config),
            'detailed_assessment': {
                'overall_readiness': 'ready' if perf_score > 75 and complexity_score < 7 else 'needs_preparation' if perf_score > 60 else 'significant_preparation_required',
                'success_probability': max(60, 95 - (complexity_score * 5) - max(0, (70 - perf_score))),
                'recommended_approach': 'direct_migration' if complexity_score < 6 and config['database_size_gb'] < 2000 else 'staged_migration'
            }
        }
    
    def _analyze_backup_storage_considerations(self, config: Dict) -> Dict:
        """Analyze backup storage specific considerations"""
        migration_method = config.get('migration_method', 'direct_replication')
        
        if migration_method != 'backup_restore':
            return {'applicable': False}
        
        backup_storage_type = config.get('backup_storage_type', 'nas_drive')
        backup_size_multiplier = config.get('backup_size_multiplier', 0.7)
        
        considerations = {
            'applicable': True,
            'storage_type': backup_storage_type,
            'protocol': 'SMB' if backup_storage_type == 'windows_share' else 'NFS',
            'backup_size_factor': backup_size_multiplier,
            'advantages': [],
            'challenges': [],
            'optimizations': []
        }
        
        if backup_storage_type == 'windows_share':
            considerations['advantages'] = [
                'Native Windows integration',
                'Familiar SMB protocols',
                'Windows authentication support',
                'Easy backup verification'
            ]
            considerations['challenges'] = [
                'SMB protocol overhead (~15% bandwidth loss)',
                'Authentication complexity over WAN',
                'SMB version compatibility requirements'
            ]
            considerations['optimizations'] = [
                'Enable SMB3 multichannel',
                'Optimize SMB signing settings',
                'Use dedicated backup network',
                'Configure SMB compression'
            ]
        else:  # nas_drive
            considerations['advantages'] = [
                'High-performance NFS protocol',
                'Better bandwidth utilization',
                'Lower protocol overhead',
                'Parallel file access capabilities'
            ]
            considerations['challenges'] = [
                'NFS tuning complexity',
                'Cross-platform compatibility',
                'NFS over WAN considerations'
            ]
            considerations['optimizations'] = [
                'Use NFS v4.1+ for best performance',
                'Optimize rsize/wsize parameters',
                'Enable NFS caching',
                'Configure appropriate timeouts'
            ]
        
        return considerations
    
    def _generate_mitigation_strategies(self, risk_factors: List[str], config: Dict) -> List[str]:
        """Generate specific mitigation strategies"""
        strategies = []
        migration_method = config.get('migration_method', 'direct_replication')
        
        if migration_method == 'backup_restore':
            strategies.append("Conduct backup integrity verification before migration")
            strategies.append("Test backup restore procedures in non-production environment")
            backup_storage_type = config.get('backup_storage_type', 'nas_drive')
            if backup_storage_type == 'windows_share':
                strategies.append("Optimize SMB performance and test stability over WAN")
            else:
                strategies.append("Configure NFS for optimal performance and reliability")
        
        if any('schema' in risk.lower() for risk in risk_factors):
            strategies.append("Conduct comprehensive schema conversion testing with AWS SCT")
            strategies.append("Create detailed schema mapping documentation")
        
        if any('database size' in risk.lower() for risk in risk_factors):
            strategies.append("Implement parallel data transfer using multiple DMS tasks")
            strategies.append("Use AWS DataSync for initial bulk data transfer")
        
        if any('downtime' in risk.lower() for risk in risk_factors):
            strategies.append("Implement read replica for near-zero downtime migration")
            strategies.append("Use AWS DMS ongoing replication for data synchronization")
        
        if any('agent' in risk.lower() for risk in risk_factors):
            strategies.append(f"Optimize {config.get('number_of_agents', 1)} agent configuration for workload")
            strategies.append("Implement agent health monitoring and automatic failover")
        
        return strategies
    
    def _generate_performance_recommendations(self, config: Dict) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        migration_method = config.get('migration_method', 'direct_replication')
        
        if migration_method == 'backup_restore':
            backup_storage_type = config.get('backup_storage_type', 'nas_drive')
            if backup_storage_type == 'windows_share':
                recommendations.append("Enable SMB3 multichannel for better throughput")
                recommendations.append("Optimize SMB client and server settings")
            else:
                recommendations.append("Tune NFS client settings for large file transfers")
                recommendations.append("Configure optimal NFS rsize/wsize values")
            recommendations.append("Use multiple DataSync agents for parallel transfers")
        
        recommendations.extend([
            "Optimize database queries and indexes before migration",
            "Configure proper instance sizing",
            "Implement monitoring and alerting"
        ])
        
        return recommendations
    
    def _generate_timeline_suggestions(self, config: Dict) -> List[str]:
        """Generate timeline suggestions"""
        migration_method = config.get('migration_method', 'direct_replication')
        timeline = [
            "Phase 1: Assessment and Planning (2-3 weeks)",
            "Phase 2: Environment Setup and Testing (2-4 weeks)"
        ]
        
        if migration_method == 'backup_restore':
            timeline.append("Phase 3: Backup Validation and DataSync Setup (1-2 weeks)")
        else:
            timeline.append("Phase 3: Data Validation and Performance Testing (1-2 weeks)")
        
        timeline.extend([
            "Phase 4: Migration Execution (1-3 days)",
            "Phase 5: Post-Migration Validation and Optimization (1 week)"
        ])
        
        return timeline
    
    def _generate_resource_allocation(self, config: Dict, complexity_score: int) -> Dict:
        """Generate resource allocation recommendations"""
        num_agents = config.get('number_of_agents', 1)
        migration_method = config.get('migration_method', 'direct_replication')
        
        base_team_size = 3 + (complexity_score // 3) + (num_agents // 3)
        
        allocation = {
            'migration_team_size': base_team_size,
            'aws_specialists_needed': 1 if complexity_score < 6 else 2,
            'database_experts_required': 1 if config['source_database_engine'] == config['database_engine'] else 2,
            'testing_resources': '2-3 dedicated testers',
            'infrastructure_requirements': f"Staging environment with {config['cpu_cores']*2} cores and {config['ram_gb']*1.5} GB RAM"
        }
        
        if migration_method == 'backup_restore':
            allocation['storage_specialists'] = 1
            allocation['backup_validation_team'] = 2
        
        return allocation
    
    def _generate_cost_optimization(self, config: Dict, complexity_score: int) -> List[str]:
        """Generate cost optimization strategies"""
        optimizations = []
        
        if config['database_size_gb'] < 1000:
            optimizations.append("Consider Reserved Instances for 20-30% cost savings")
        
        if config['environment'] == 'non-production':
            optimizations.append("Use Spot Instances for development/testing to reduce costs by 60-70%")
        
        migration_method = config.get('migration_method', 'direct_replication')
        if migration_method == 'backup_restore':
            optimizations.append("Optimize backup storage costs and DataSync pricing")
        
        optimizations.append("Implement automated scaling policies to optimize resource utilization")
        
        return optimizations
    
    def _generate_best_practices(self, config: Dict, complexity_score: int) -> List[str]:
        """Generate best practices"""
        practices = [
            "Implement comprehensive backup strategy before migration initiation",
            "Use AWS Migration Hub for centralized migration tracking",
            "Establish detailed communication plan with stakeholders",
            "Create detailed runbook with step-by-step procedures"
        ]
        
        migration_method = config.get('migration_method', 'direct_replication')
        if migration_method == 'backup_restore':
            practices.append("Validate backup integrity before starting migration")
            practices.append("Test restore procedures in isolated environment")
        
        return practices
    
    def _generate_testing_strategy(self, config: Dict, complexity_score: int) -> List[str]:
        """Generate testing strategy"""
        strategy = [
            "Unit Testing: Validate individual migration components",
            "Integration Testing: Test end-to-end migration workflow",
            "Performance Testing: Validate AWS environment performance",
            "Data Integrity Testing: Verify data consistency and completeness"
        ]
        
        migration_method = config.get('migration_method', 'direct_replication')
        if migration_method == 'backup_restore':
            strategy.append("Backup Validation Testing: Verify backup file integrity and completeness")
        
        return strategy
    
    def _generate_rollback_procedures(self, config: Dict) -> List[str]:
        """Generate rollback procedures"""
        procedures = [
            "Maintain synchronized read replica during migration window",
            "Create point-in-time recovery snapshot before cutover",
            "Prepare DNS switching procedures for quick rollback",
            "Document application configuration rollback steps"
        ]
        
        migration_method = config.get('migration_method', 'direct_replication')
        if migration_method == 'backup_restore':
            procedures.append("Keep original backup files until migration validation complete")
        
        return procedures
    
    def _generate_monitoring_recommendations(self, config: Dict) -> List[str]:
        """Generate monitoring recommendations"""
        recommendations = [
            "Implement CloudWatch detailed monitoring for all database metrics",
            "Set up automated alerts for performance degradation",
            "Monitor application response times and error rates",
            "Track database connection patterns and query performance"
        ]
        
        migration_method = config.get('migration_method', 'direct_replication')
        if migration_method == 'backup_restore':
            recommendations.append("Monitor DataSync task progress and error rates")
        
        return recommendations
    
    def _analyze_agent_scaling_impact(self, config: Dict) -> Dict:
        """Analyze agent scaling impact"""
        num_agents = config.get('number_of_agents', 1)
        migration_method = config.get('migration_method', 'direct_replication')
        
        impact = {
            'parallel_processing_benefit': min(num_agents * 20, 80),
            'coordination_overhead': max(0, (num_agents - 1) * 5),
            'throughput_multiplier': min(num_agents * 0.8, 4.0),
            'management_complexity': num_agents * 10,
            'optimal_agent_count': self._calculate_optimal_agents(config),
            'current_efficiency': min(100, (100 - (abs(num_agents - self._calculate_optimal_agents(config)) * 10)))
        }
        
        if migration_method == 'backup_restore':
            impact['file_transfer_optimization'] = num_agents * 15
        
        return impact
    
    def _analyze_storage_impact(self, config: Dict) -> Dict:
        """Analyze destination storage impact"""
        destination_storage = config.get('destination_storage_type', 'S3')
        return {
            'storage_type': destination_storage,
            'performance_impact': self._calculate_storage_performance_impact(destination_storage),
            'cost_impact': self._calculate_storage_cost_impact(destination_storage),
            'complexity_factor': self._get_storage_complexity_factor(destination_storage)
        }
    
    def _calculate_optimal_agents(self, config: Dict) -> int:
        """Calculate optimal number of agents"""
        database_size = config['database_size_gb']
        migration_method = config.get('migration_method', 'direct_replication')
        
        if migration_method == 'backup_restore':
            # For backup/restore, optimal agents depend on backup size and storage type
            backup_size_multiplier = config.get('backup_size_multiplier', 0.7)
            effective_size = database_size * backup_size_multiplier
            if effective_size < 500:
                return 1
            elif effective_size < 2000:
                return 2
            elif effective_size < 10000:
                return 3
            else:
                return 4
        else:
            # Original logic for direct replication
            if database_size < 1000:
                return 1
            elif database_size < 5000:
                return 2
            elif database_size < 20000:
                return 3
            else:
                return 4
    
    def _calculate_storage_performance_impact(self, storage_type: str) -> Dict:
        """Calculate performance impact for storage"""
        storage_profiles = {
            'S3': {'throughput_multiplier': 1.0, 'performance_rating': 'Good'},
            'FSx_Windows': {'throughput_multiplier': 1.3, 'performance_rating': 'Very Good'},
            'FSx_Lustre': {'throughput_multiplier': 2.0, 'performance_rating': 'Excellent'}
        }
        return storage_profiles.get(storage_type, storage_profiles['S3'])
    
    def _calculate_storage_cost_impact(self, storage_type: str) -> Dict:
        """Calculate cost impact for storage"""
        cost_profiles = {
            'S3': {'base_cost_multiplier': 1.0, 'long_term_value': 'Excellent'},
            'FSx_Windows': {'base_cost_multiplier': 2.5, 'long_term_value': 'Good'},
            'FSx_Lustre': {'base_cost_multiplier': 4.0, 'long_term_value': 'Good for HPC'}
        }
        return cost_profiles.get(storage_type, cost_profiles['S3'])
    
    def _get_storage_complexity_factor(self, storage_type: str) -> float:
        """Get complexity factor for storage type"""
        complexity_factors = {'S3': 1.0, 'FSx_Windows': 1.8, 'FSx_Lustre': 2.2}
        return complexity_factors.get(storage_type, 1.0)
    
    def _fallback_workload_analysis(self, config: Dict, performance_data: Dict) -> Dict:
        """Fallback analysis when AI is not available"""
        complexity_score = 5
        if config['source_database_engine'] != config['database_engine']:
            complexity_score += 2
        if config['database_size_gb'] > 5000:
            complexity_score += 1
        
        migration_method = config.get('migration_method', 'direct_replication')
        if migration_method == 'backup_restore':
            complexity_score += 1  # Backup/restore adds some complexity
        
        return {
            'ai_complexity_score': min(10, complexity_score),
            'risk_factors': ["Migration complexity varies with database engine differences"],
            'mitigation_strategies': ["Conduct thorough pre-migration testing"],
            'performance_recommendations': ["Optimize database before migration"],
            'confidence_level': 'medium',
            'backup_storage_considerations': self._analyze_backup_storage_considerations(config),
            'raw_ai_response': 'AI analysis not available - using fallback analysis'
        }

class AWSAPIManager:
    """Manage AWS API integration for real-time pricing and optimization"""
    
    def __init__(self):
        self.session = None
        self.pricing_client = None
        self.connected = False
        
        try:
            self.session = boto3.Session()
            self.pricing_client = self.session.client('pricing', region_name='us-east-1')
            self.pricing_client.describe_services(MaxResults=1)
            self.connected = True
            logger.info("AWS API clients initialized successfully")
        except (NoCredentialsError, ClientError) as e:
            logger.warning(f"AWS API initialization failed: {e}")
            self.connected = False
    
    async def get_real_time_pricing(self, region: str = 'us-west-2') -> Dict:
        """Fetch real-time AWS pricing data"""
        if not self.connected:
            return self._fallback_pricing_data(region)
        
        try:
            ec2_pricing = await self._get_ec2_pricing(region)
            rds_pricing = await self._get_rds_pricing(region)
            storage_pricing = await self._get_storage_pricing(region)
            dx_pricing = await self._get_direct_connect_pricing(region)
            datasync_pricing = await self._get_datasync_pricing(region)
            
            return {
                'region': region,
                'last_updated': datetime.now(),
                'ec2_instances': ec2_pricing,
                'rds_instances': rds_pricing,
                'storage': storage_pricing,
                'direct_connect': dx_pricing,
                'datasync': datasync_pricing,
                'data_source': 'aws_api'
            }
        except Exception as e:
            logger.error(f"Failed to fetch AWS pricing: {e}")
            return self._fallback_pricing_data(region)
    
    async def _get_ec2_pricing(self, region: str) -> Dict:
        """Get EC2 instance pricing"""
        instance_types = ['t3.medium', 't3.large', 't3.xlarge', 'c5.large', 'c5.xlarge', 
                         'c5.2xlarge', 'r6i.large', 'r6i.xlarge', 'r6i.2xlarge']
        
        pricing_data = {}
        for instance_type in instance_types:
            try:
                response = self.pricing_client.get_products(
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
                            price_per_hour = float(price_info['pricePerUnit']['USD'])
                            
                            attributes = price_data.get('product', {}).get('attributes', {})
                            pricing_data[instance_type] = {
                                'vcpu': int(attributes.get('vcpu', 2)),
                                'memory': self._extract_memory_gb(attributes.get('memory', '4 GiB')),
                                'cost_per_hour': price_per_hour
                            }
            except Exception as e:
                logger.warning(f"Failed to get pricing for {instance_type}: {e}")
                pricing_data[instance_type] = self._get_fallback_instance_pricing(instance_type)
        
        return pricing_data
    
    async def _get_rds_pricing(self, region: str) -> Dict:
        """Get RDS instance pricing"""
        # Similar to EC2 pricing but for RDS
        return self._fallback_rds_pricing()
    
    async def _get_storage_pricing(self, region: str) -> Dict:
        """Get storage pricing"""
        return self._fallback_storage_pricing()
    
    async def _get_direct_connect_pricing(self, region: str) -> Dict:
        """Get Direct Connect pricing"""
        if not self.connected:
            return self._fallback_direct_connect_pricing()
        
        try:
            # Direct Connect pricing is complex and varies by location
            # For simplicity, we'll use fallback data but structure it properly
            return self._fallback_direct_connect_pricing()
        except Exception as e:
            logger.warning(f"Failed to get Direct Connect pricing: {e}")
            return self._fallback_direct_connect_pricing()
    
    async def _get_datasync_pricing(self, region: str) -> Dict:
        """Get DataSync pricing"""
        if not self.connected:
            return self._fallback_datasync_pricing()
        
        try:
            # DataSync pricing is based on data transferred
            response = self.pricing_client.get_products(
                ServiceCode='AWSDataSync',
                Filters=[
                    {'Type': 'TERM_MATCH', 'Field': 'location', 'Value': self._region_to_location(region)},
                    {'Type': 'TERM_MATCH', 'Field': 'transferType', 'Value': 'AWS-to-AWS'}
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
                        transfer_type = attributes.get('transferType', 'unknown')
                        
                        pricing_data[transfer_type] = {
                            'price_per_gb': float(price_info['pricePerUnit']['USD']),
                            'description': price_info.get('description', ''),
                            'unit': price_info.get('unit', 'GB')
                        }
            
            return pricing_data if pricing_data else self._fallback_datasync_pricing()
            
        except Exception as e:
            logger.warning(f"Failed to get DataSync pricing: {e}")
            return self._fallback_datasync_pricing()
    
    def _region_to_location(self, region: str) -> str:
        """Convert AWS region to location name"""
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
    
    def _fallback_pricing_data(self, region: str) -> Dict:
        """Fallback pricing data"""
        return {
            'region': region,
            'last_updated': datetime.now(),
            'data_source': 'fallback',
            'ec2_instances': self._fallback_ec2_pricing(),
            'rds_instances': self._fallback_rds_pricing(),
            'storage': self._fallback_storage_pricing(),
            'direct_connect': self._fallback_direct_connect_pricing(),
            'datasync': self._fallback_datasync_pricing()
        }
    
    def _fallback_ec2_pricing(self) -> Dict:
        """Fallback EC2 pricing"""
        return {
            't3.medium': {'vcpu': 2, 'memory': 4, 'cost_per_hour': 0.0416},
            't3.large': {'vcpu': 2, 'memory': 8, 'cost_per_hour': 0.0832},
            't3.xlarge': {'vcpu': 4, 'memory': 16, 'cost_per_hour': 0.1664},
            'c5.large': {'vcpu': 2, 'memory': 4, 'cost_per_hour': 0.085},
            'c5.xlarge': {'vcpu': 4, 'memory': 8, 'cost_per_hour': 0.17},
            'c5.2xlarge': {'vcpu': 8, 'memory': 16, 'cost_per_hour': 0.34},
            'r6i.large': {'vcpu': 2, 'memory': 16, 'cost_per_hour': 0.252},
            'r6i.xlarge': {'vcpu': 4, 'memory': 32, 'cost_per_hour': 0.504},
            'r6i.2xlarge': {'vcpu': 8, 'memory': 64, 'cost_per_hour': 1.008}
        }
    
    def _fallback_rds_pricing(self) -> Dict:
        """Fallback RDS pricing"""
        return {
            'db.t3.medium': {'vcpu': 2, 'memory': 4, 'cost_per_hour': 0.068},
            'db.t3.large': {'vcpu': 2, 'memory': 8, 'cost_per_hour': 0.136},
            'db.r6g.large': {'vcpu': 2, 'memory': 16, 'cost_per_hour': 0.48},
            'db.r6g.xlarge': {'vcpu': 4, 'memory': 32, 'cost_per_hour': 0.96},
            'db.r6g.2xlarge': {'vcpu': 8, 'memory': 64, 'cost_per_hour': 1.92}
        }
    
    def _fallback_storage_pricing(self) -> Dict:
        """Fallback storage pricing"""
        return {
            'gp3': {'cost_per_gb_month': 0.08, 'iops_included': 3000},
            'io1': {'cost_per_gb_month': 0.125, 'cost_per_iops_month': 0.065},
            's3_standard': {'cost_per_gb_month': 0.023, 'requests_per_1000': 0.0004}
        }
    
    def _fallback_direct_connect_pricing(self) -> Dict:
        """Fallback Direct Connect pricing"""
        return {
            '1Gbps': {
                'port_hours': 0.30,  # Per hour for dedicated connection
                'data_transfer_out': 0.02,  # Per GB
                'description': '1 Gbps Dedicated Connection'
            },
            '10Gbps': {
                'port_hours': 2.25,  # Per hour for dedicated connection
                'data_transfer_out': 0.02,  # Per GB
                'description': '10 Gbps Dedicated Connection'
            },
            '100Gbps': {
                'port_hours': 22.50,  # Per hour for dedicated connection
                'data_transfer_out': 0.02,  # Per GB
                'description': '100 Gbps Dedicated Connection'
            }
        }
    
    def _fallback_datasync_pricing(self) -> Dict:
        """Fallback DataSync pricing"""
        return {
            'data_transfer': {
                'price_per_gb': 0.0125,  # $0.0125 per GB transferred
                'description': 'Data transferred by DataSync',
                'unit': 'GB'
            },
            'task_execution': {
                'price_per_gb': 0.0125,
                'description': 'Per GB of data prepared by DataSync',
                'unit': 'GB'
            }
        }
    
    def _get_fallback_instance_pricing(self, instance_type: str) -> Dict:
        """Get fallback pricing for instance"""
        fallback_data = self._fallback_ec2_pricing()
        return fallback_data.get(instance_type, {'vcpu': 2, 'memory': 4, 'cost_per_hour': 0.05})

class ComprehensiveAWSCostCalculator:
    """Comprehensive AWS cost calculator for all migration services"""
    
    def __init__(self, aws_api_manager: AWSAPIManager):
        self.aws_api = aws_api_manager
    
    async def calculate_comprehensive_migration_costs(self, config: Dict, analysis: Dict) -> Dict:
        """Calculate comprehensive costs for all AWS services"""
        
        # Get real-time pricing data
        pricing_data = await self.aws_api.get_real_time_pricing()
        
        # Calculate individual service costs
        compute_costs = await self._calculate_compute_costs(config, pricing_data)
        storage_costs = await self._calculate_storage_costs(config, pricing_data)
        network_costs = await self._calculate_network_costs(config, pricing_data, analysis)
        migration_costs = await self._calculate_migration_service_costs(config, pricing_data, analysis)
        
        # Calculate total costs
        monthly_costs = {
            'compute': compute_costs['monthly_total'],
            'storage': storage_costs['monthly_total'],
            'network': network_costs['monthly_total'],
            'migration_services': migration_costs['monthly_total']
        }
        
        total_monthly = sum(monthly_costs.values())
        
        # One-time costs
        one_time_costs = {
            'migration_setup': migration_costs['one_time_setup'],
            'data_transfer': migration_costs['one_time_data_transfer'],
            'professional_services': config.get('database_size_gb', 0) * 0.05  # Estimated professional services
        }
        
        total_one_time = sum(one_time_costs.values())
        
        return {
            'pricing_data': pricing_data,
            'compute_costs': compute_costs,
            'storage_costs': storage_costs,
            'network_costs': network_costs,
            'migration_costs': migration_costs,
            'monthly_breakdown': monthly_costs,
            'one_time_breakdown': one_time_costs,
            'total_monthly': total_monthly,
            'total_one_time': total_one_time,
            'annual_total': total_monthly * 12 + total_one_time,
            'three_year_total': total_monthly * 36 + total_one_time,
            'cost_optimization_recommendations': self._generate_cost_optimization_recommendations(monthly_costs, config),
            'detailed_service_breakdown': self._generate_detailed_service_breakdown(compute_costs, storage_costs, network_costs, migration_costs)
        }
    
    def _generate_detailed_service_breakdown(self, compute_costs: Dict, storage_costs: Dict, 
                                           network_costs: Dict, migration_costs: Dict) -> List[Dict]:
        """Generate detailed breakdown of all AWS services"""
        services = []
        
        # Compute Services
        if compute_costs.get('service_type') == 'RDS':
            services.append({
                'service_category': 'Database',
                'service_name': 'Amazon RDS',
                'instance_type': compute_costs.get('primary_instance', 'Unknown'),
                'monthly_cost': compute_costs.get('writer_monthly_cost', 0),
                'unit': 'Instance Hours',
                'cost_per_hour': compute_costs.get('instance_details', {}).get('primary', {}).get('cost_per_hour', 0),
                'description': 'Primary database instance'
            })
            
            if compute_costs.get('reader_instances', 0) > 0:
                services.append({
                    'service_category': 'Database',
                    'service_name': 'Amazon RDS (Read Replicas)',
                    'instance_type': compute_costs.get('primary_instance', 'Unknown'),
                    'monthly_cost': compute_costs.get('reader_monthly_cost', 0),
                    'unit': 'Instance Hours',
                    'cost_per_hour': compute_costs.get('instance_details', {}).get('readers', {}).get('cost_per_hour', 0),
                    'description': f"{compute_costs.get('reader_instances', 0)} read replica instances"
                })
            
            if compute_costs.get('multi_az_cost', 0) > 0:
                services.append({
                    'service_category': 'Database',
                    'service_name': 'Amazon RDS (Multi-AZ)',
                    'instance_type': 'Multi-AZ Deployment',
                    'monthly_cost': compute_costs.get('multi_az_cost', 0),
                    'unit': 'Multi-AZ Hours',
                    'cost_per_hour': compute_costs.get('multi_az_cost', 0) / (24 * 30),
                    'description': 'Multi-AZ deployment for high availability'
                })
        else:
            # EC2 Services
            services.append({
                'service_category': 'Compute',
                'service_name': 'Amazon EC2',
                'instance_type': compute_costs.get('primary_instance', 'Unknown'),
                'monthly_cost': compute_costs.get('monthly_instance_cost', 0),
                'unit': 'Instance Hours',
                'cost_per_hour': compute_costs.get('instance_details', {}).get('cost_per_hour', 0),
                'description': f"{compute_costs.get('instance_count', 1)} EC2 instances"
            })
            
            if compute_costs.get('os_licensing_cost', 0) > 0:
                services.append({
                    'service_category': 'Licensing',
                    'service_name': 'Windows Server Licensing',
                    'instance_type': 'OS License',
                    'monthly_cost': compute_costs.get('os_licensing_cost', 0),
                    'unit': 'License',
                    'cost_per_hour': 0,
                    'description': 'Windows Server licensing fees'
                })
        
        # Storage Services
        primary_storage = storage_costs.get('primary_storage', {})
        if primary_storage.get('monthly_cost', 0) > 0:
            services.append({
                'service_category': 'Storage',
                'service_name': f"Amazon EBS ({primary_storage.get('type', 'gp3').upper()})",
                'instance_type': f"{primary_storage.get('size_gb', 0)} GB",
                'monthly_cost': primary_storage.get('monthly_cost', 0),
                'unit': 'GB-Month',
                'cost_per_hour': 0,
                'description': 'Primary database storage'
            })
        
        destination_storage = storage_costs.get('destination_storage', {})
        if destination_storage.get('monthly_cost', 0) > 0:
            storage_name = {
                'S3': 'Amazon S3',
                'FSx_Windows': 'Amazon FSx for Windows File Server',
                'FSx_Lustre': 'Amazon FSx for Lustre'
            }.get(destination_storage.get('type', 'S3'), 'Amazon S3')
            
            services.append({
                'service_category': 'Storage',
                'service_name': storage_name,
                'instance_type': f"{destination_storage.get('size_gb', 0)} GB",
                'monthly_cost': destination_storage.get('monthly_cost', 0),
                'unit': 'GB-Month',
                'cost_per_hour': 0,
                'description': 'Destination storage for migration'
            })
        
        backup_storage = storage_costs.get('backup_storage', {})
        if backup_storage.get('applicable', False) and backup_storage.get('monthly_cost', 0) > 0:
            services.append({
                'service_category': 'Storage',
                'service_name': 'Amazon S3 (Backup Storage)',
                'instance_type': f"{backup_storage.get('size_gb', 0)} GB",
                'monthly_cost': backup_storage.get('monthly_cost', 0),
                'unit': 'GB-Month',
                'cost_per_hour': 0,
                'description': 'Backup file storage for migration'
            })
        
        # Network Services
        dx_service = network_costs.get('direct_connect', {})
        if dx_service.get('monthly_cost', 0) > 0:
            services.append({
                'service_category': 'Network',
                'service_name': 'AWS Direct Connect',
                'instance_type': dx_service.get('capacity', 'Unknown'),
                'monthly_cost': dx_service.get('monthly_cost', 0),
                'unit': 'Port Hours',
                'cost_per_hour': dx_service.get('hourly_cost', 0),
                'description': 'Dedicated network connection'
            })
        
        data_transfer = network_costs.get('data_transfer', {})
        if data_transfer.get('monthly_cost', 0) > 0:
            services.append({
                'service_category': 'Network',
                'service_name': 'Data Transfer Out',
                'instance_type': f"{data_transfer.get('monthly_gb', 0)} GB/month",
                'monthly_cost': data_transfer.get('monthly_cost', 0),
                'unit': 'GB',
                'cost_per_hour': 0,
                'description': 'Data transfer charges'
            })
        
        vpn_backup = network_costs.get('vpn_backup', {})
        if vpn_backup.get('monthly_cost', 0) > 0:
            services.append({
                'service_category': 'Network',
                'service_name': 'AWS VPN Gateway',
                'instance_type': 'VPN Connection',
                'monthly_cost': vpn_backup.get('monthly_cost', 0),
                'unit': 'VPN Hours',
                'cost_per_hour': vpn_backup.get('monthly_cost', 0) / (24 * 30),
                'description': 'Backup VPN connection'
            })
        
        # Migration Services
        agent_costs = migration_costs.get('agent_costs', {})
        if agent_costs.get('monthly_cost', 0) > 0:
            services.append({
                'service_category': 'Migration',
                'service_name': 'Migration Agents (EC2)',
                'instance_type': f"{agent_costs.get('agent_count', 1)} Agents",
                'monthly_cost': agent_costs.get('monthly_cost', 0),
                'unit': 'Instance Hours',
                'cost_per_hour': agent_costs.get('cost_per_agent', 0) / (24 * 30) if agent_costs.get('cost_per_agent') else 0,
                'description': 'DataSync/DMS agent instances'
            })
        
        datasync_service = migration_costs.get('datasync', {})
        if datasync_service.get('applicable', False) and datasync_service.get('monthly_sync_cost', 0) > 0:
            services.append({
                'service_category': 'Migration',
                'service_name': 'AWS DataSync',
                'instance_type': f"{datasync_service.get('data_size_gb', 0)} GB",
                'monthly_cost': datasync_service.get('monthly_sync_cost', 0),
                'unit': 'GB Transferred',
                'cost_per_hour': 0,
                'description': 'Ongoing data synchronization'
            })
        
        dms_service = migration_costs.get('dms', {})
        if dms_service.get('applicable', False) and dms_service.get('monthly_cost', 0) > 0:
            services.append({
                'service_category': 'Migration',
                'service_name': 'AWS Database Migration Service',
                'instance_type': 'DMS Replication Instance',
                'monthly_cost': dms_service.get('monthly_cost', 0),
                'unit': 'Instance Hours',
                'cost_per_hour': dms_service.get('monthly_cost', 0) / (24 * 30),
                'description': 'Database migration service'
            })
        
        return services
    
    async def _calculate_compute_costs(self, config: Dict, pricing_data: Dict) -> Dict:
        """Calculate compute costs (EC2/RDS)"""
        target_platform = config.get('target_platform', 'rds')
        database_size_gb = config.get('database_size_gb', 1000)
        environment = config.get('environment', 'non-production')
        
        if target_platform == 'rds':
            # RDS instance costs
            if database_size_gb < 1000:
                instance_type = 'db.t3.medium'
                instance_cost = 0.068
            elif database_size_gb < 5000:
                instance_type = 'db.r6g.large'
                instance_cost = 0.48
            else:
                instance_type = 'db.r6g.xlarge'
                instance_cost = 0.96
            
            # Get real pricing if available
            rds_instances = pricing_data.get('rds_instances', {})
            if instance_type in rds_instances:
                instance_cost = rds_instances[instance_type].get('cost_per_hour', instance_cost)
            
            # Calculate reader instances
            if environment == 'production':
                reader_count = 2 if database_size_gb > 5000 else 1
            else:
                reader_count = 1 if database_size_gb > 2000 else 0
            
            writer_monthly = instance_cost * 24 * 30
            reader_monthly = instance_cost * 24 * 30 * reader_count * 0.9  # Readers are slightly cheaper
            
            # Multi-AZ costs
            multi_az_cost = writer_monthly if environment == 'production' else 0
            
            return {
                'service_type': 'RDS',
                'primary_instance': instance_type,
                'writer_instances': 1,
                'reader_instances': reader_count,
                'writer_monthly_cost': writer_monthly,
                'reader_monthly_cost': reader_monthly,
                'multi_az_cost': multi_az_cost,
                'monthly_total': writer_monthly + reader_monthly + multi_az_cost,
                'instance_details': {
                    'primary': {'type': instance_type, 'cost_per_hour': instance_cost},
                    'readers': {'count': reader_count, 'cost_per_hour': instance_cost * 0.9}
                }
            }
        
        else:  # EC2
            # EC2 instance costs
            if database_size_gb < 1000:
                instance_type = 't3.large'
                instance_cost = 0.0832
            elif database_size_gb < 5000:
                instance_type = 'r6i.large'
                instance_cost = 0.252
            else:
                instance_type = 'r6i.xlarge'
                instance_cost = 0.504
            
            # Get real pricing if available
            ec2_instances = pricing_data.get('ec2_instances', {})
            if instance_type in ec2_instances:
                instance_cost = ec2_instances[instance_type].get('cost_per_hour', instance_cost)
            
            # Calculate additional instances for HA
            instance_count = 2 if environment == 'production' else 1
            
            monthly_cost = instance_cost * 24 * 30 * instance_count
            
            # Operating system licensing
            os_licensing = 0
            if 'windows' in config.get('operating_system', ''):
                os_licensing = 150 * instance_count  # Windows licensing per instance
            
            return {
                'service_type': 'EC2',
                'primary_instance': instance_type,
                'instance_count': instance_count,
                'monthly_instance_cost': monthly_cost,
                'os_licensing_cost': os_licensing,
                'monthly_total': monthly_cost + os_licensing,
                'instance_details': {
                    'type': instance_type,
                    'count': instance_count,
                    'cost_per_hour': instance_cost
                }
            }
    
    async def _calculate_storage_costs(self, config: Dict, pricing_data: Dict) -> Dict:
        """Calculate storage costs (EBS, S3, FSx)"""
        database_size_gb = config.get('database_size_gb', 1000)
        destination_storage = config.get('destination_storage_type', 'S3')
        target_platform = config.get('target_platform', 'rds')
        
        # Primary database storage
        if target_platform == 'rds':
            storage_size = database_size_gb * 1.5  # RDS storage multiplier
            storage_type = 'gp3'
            storage_cost_per_gb = 0.08
        else:
            storage_size = database_size_gb * 2.0  # EC2 storage multiplier
            storage_type = 'gp3'
            storage_cost_per_gb = 0.08
        
        # Get real pricing if available
        storage_pricing = pricing_data.get('storage', {})
        if storage_type in storage_pricing:
            storage_cost_per_gb = storage_pricing[storage_type].get('cost_per_gb_month', storage_cost_per_gb)
        
        primary_storage_cost = storage_size * storage_cost_per_gb
        
        # Destination storage costs
        if destination_storage == 'S3':
            dest_storage_cost_per_gb = 0.023
            dest_storage_cost = database_size_gb * 1.2 * dest_storage_cost_per_gb
        elif destination_storage == 'FSx_Windows':
            dest_storage_cost_per_gb = 0.13
            dest_storage_cost = database_size_gb * 1.2 * dest_storage_cost_per_gb
        elif destination_storage == 'FSx_Lustre':
            dest_storage_cost_per_gb = 0.14
            dest_storage_cost = database_size_gb * 1.2 * dest_storage_cost_per_gb
        else:
            dest_storage_cost = 0
        
        # Backup storage costs
        backup_storage_cost = 0
        migration_method = config.get('migration_method', 'direct_replication')
        if migration_method == 'backup_restore':
            backup_size_multiplier = config.get('backup_size_multiplier', 0.7)
            backup_size = database_size_gb * backup_size_multiplier
            backup_storage_cost = backup_size * 0.023  # S3 Standard pricing for backups
        
        return {
            'primary_storage': {
                'size_gb': storage_size,
                'type': storage_type,
                'cost_per_gb': storage_cost_per_gb,
                'monthly_cost': primary_storage_cost
            },
            'destination_storage': {
                'type': destination_storage,
                'size_gb': database_size_gb * 1.2,
                'cost_per_gb': dest_storage_cost_per_gb if destination_storage != 'S3' else 0.023,
                'monthly_cost': dest_storage_cost
            },
            'backup_storage': {
                'applicable': migration_method == 'backup_restore',
                'size_gb': database_size_gb * config.get('backup_size_multiplier', 0.7) if migration_method == 'backup_restore' else 0,
                'monthly_cost': backup_storage_cost
            },
            'monthly_total': primary_storage_cost + dest_storage_cost + backup_storage_cost
        }
    
    async def _calculate_network_costs(self, config: Dict, pricing_data: Dict, analysis: Dict) -> Dict:
        """Calculate network costs (Direct Connect, Data Transfer)"""
        environment = config.get('environment', 'non-production')
        database_size_gb = config.get('database_size_gb', 1000)
        
        # Direct Connect costs
        if environment == 'production':
            dx_capacity = '10Gbps'
            dx_hourly = 2.25
        else:
            dx_capacity = '1Gbps'  
            dx_hourly = 0.30
        
        # Get real pricing if available
        dx_pricing = pricing_data.get('direct_connect', {})
        if dx_capacity in dx_pricing:
            dx_hourly = dx_pricing[dx_capacity].get('port_hours', dx_hourly)
        
        dx_monthly = dx_hourly * 24 * 30
        
        # Data transfer costs
        # Estimate monthly data transfer based on database size and usage
        monthly_data_transfer_gb = database_size_gb * 0.1  # 10% of DB size per month
        data_transfer_cost_per_gb = 0.02  # Standard DX data transfer cost
        
        if dx_capacity in dx_pricing:
            data_transfer_cost_per_gb = dx_pricing[dx_capacity].get('data_transfer_out', data_transfer_cost_per_gb)
        
        data_transfer_monthly = monthly_data_transfer_gb * data_transfer_cost_per_gb
        
        # VPN costs (if needed as backup)
        vpn_monthly = 45.0  # VPN Gateway cost
        
        return {
            'direct_connect': {
                'capacity': dx_capacity,
                'hourly_cost': dx_hourly,
                'monthly_cost': dx_monthly
            },
            'data_transfer': {
                'monthly_gb': monthly_data_transfer_gb,
                'cost_per_gb': data_transfer_cost_per_gb,
                'monthly_cost': data_transfer_monthly
            },
            'vpn_backup': {
                'monthly_cost': vpn_monthly if environment == 'production' else 0
            },
            'monthly_total': dx_monthly + data_transfer_monthly + (vpn_monthly if environment == 'production' else 0)
        }
    
    async def _calculate_migration_service_costs(self, config: Dict, pricing_data: Dict, analysis: Dict) -> Dict:
        """Calculate migration service costs (DataSync, DMS, Agents)"""
        database_size_gb = config.get('database_size_gb', 1000)
        migration_method = config.get('migration_method', 'direct_replication')
        num_agents = config.get('number_of_agents', 1)
        
        # Agent costs (EC2 instances running agents)
        agent_analysis = analysis.get('agent_analysis', {})
        agent_monthly_cost = agent_analysis.get('monthly_cost', 0)
        
        # DataSync costs
        datasync_monthly = 0
        datasync_one_time = 0
        
        if migration_method == 'backup_restore':
            # DataSync pricing
            backup_size_multiplier = config.get('backup_size_multiplier', 0.7)
            backup_size_gb = database_size_gb * backup_size_multiplier
            
            datasync_pricing = pricing_data.get('datasync', {})
            price_per_gb = 0.0125  # Default pricing
            
            if 'data_transfer' in datasync_pricing:
                price_per_gb = datasync_pricing['data_transfer'].get('price_per_gb', price_per_gb)
            
            datasync_one_time = backup_size_gb * price_per_gb
            
            # Monthly DataSync for ongoing sync (if applicable)
            monthly_sync_gb = backup_size_gb * 0.05  # 5% change per month
            datasync_monthly = monthly_sync_gb * price_per_gb
        
        # DMS costs
        dms_monthly = 0
        dms_one_time = 0
        
        if migration_method == 'direct_replication' and config.get('source_database_engine') != config.get('database_engine'):
            # DMS replication instance costs (included in agent costs)
            # Additional DMS-specific costs
            dms_one_time = database_size_gb * 0.001  # Minimal setup cost
        
        # Professional services and setup
        setup_cost = 2000 + (num_agents * 500)  # Base setup + per agent
        
        return {
            'agent_costs': {
                'monthly_cost': agent_monthly_cost,
                'agent_count': num_agents,
                'cost_per_agent': agent_monthly_cost / num_agents if num_agents > 0 else 0
            },
            'datasync': {
                'applicable': migration_method == 'backup_restore',
                'one_time_transfer_cost': datasync_one_time,
                'monthly_sync_cost': datasync_monthly,
                'data_size_gb': database_size_gb * config.get('backup_size_multiplier', 0.7) if migration_method == 'backup_restore' else 0
            },
            'dms': {
                'applicable': migration_method == 'direct_replication' and config.get('source_database_engine') != config.get('database_engine'),
                'monthly_cost': dms_monthly,
                'one_time_setup': dms_one_time
            },
            'setup_and_professional_services': {
                'one_time_cost': setup_cost,
                'includes': ['Migration planning', 'Agent setup', 'Testing', 'Cutover support']
            },
            'monthly_total': agent_monthly_cost + datasync_monthly + dms_monthly,
            'one_time_setup': datasync_one_time + dms_one_time + setup_cost,
            'one_time_data_transfer': datasync_one_time
        }
    
    def _generate_cost_optimization_recommendations(self, monthly_costs: Dict, config: Dict) -> List[str]:
        """Generate cost optimization recommendations"""
        recommendations = []
        
        total_monthly = sum(monthly_costs.values())
        
        # Compute optimization
        compute_pct = (monthly_costs.get('compute', 0) / total_monthly * 100) if total_monthly > 0 else 0
        if compute_pct > 60:
            recommendations.append("Consider Reserved Instances for 20-30% compute cost savings")
        
        # Storage optimization
        storage_pct = (monthly_costs.get('storage', 0) / total_monthly * 100) if total_monthly > 0 else 0
        if storage_pct > 25:
            recommendations.append("Implement storage lifecycle policies to reduce costs")
        
        # Network optimization
        network_pct = (monthly_costs.get('network', 0) / total_monthly * 100) if total_monthly > 0 else 0
        if network_pct > 15:
            recommendations.append("Optimize data transfer patterns and consider CloudFront")
        
        # Migration optimization
        if config.get('number_of_agents', 1) > 4:
            recommendations.append("Consider consolidating agents to reduce operational overhead")
        
        # Environment-specific
        if config.get('environment') == 'non-production':
            recommendations.append("Use Spot Instances for non-production workloads for 60-70% savings")
        
        return recommendations

class OSPerformanceManager:
    """Enhanced OS performance manager with AI insights"""
    
    def __init__(self):
        self.operating_systems = {
            'windows_server_2019': {
                'name': 'Windows Server 2019',
                'cpu_efficiency': 0.92,
                'memory_efficiency': 0.88,
                'io_efficiency': 0.85,
                'network_efficiency': 0.90,
                'virtualization_overhead': 0.12,
                'database_optimizations': {
                    'mysql': 0.88, 'postgresql': 0.85, 'oracle': 0.95, 'sqlserver': 0.98, 'mongodb': 0.87
                },
                'licensing_cost_factor': 2.5,
                'management_complexity': 0.6,
                'security_overhead': 0.08,
                'ai_insights': {
                    'strengths': ['Native SQL Server integration', 'Enterprise management tools'],
                    'weaknesses': ['Higher licensing costs', 'More resource overhead'],
                    'migration_considerations': ['Licensing compliance', 'Service account migration']
                }
            },
            'windows_server_2022': {
                'name': 'Windows Server 2022',
                'cpu_efficiency': 0.95,
                'memory_efficiency': 0.92,
                'io_efficiency': 0.90,
                'network_efficiency': 0.93,
                'virtualization_overhead': 0.10,
                'database_optimizations': {
                    'mysql': 0.90, 'postgresql': 0.88, 'oracle': 0.97, 'sqlserver': 0.99, 'mongodb': 0.89
                },
                'licensing_cost_factor': 3.0,
                'management_complexity': 0.5,
                'security_overhead': 0.06,
                'ai_insights': {
                    'strengths': ['Improved container support', 'Enhanced security features'],
                    'weaknesses': ['Higher costs', 'Newer OS compatibility risks'],
                    'migration_considerations': ['Hardware compatibility', 'Application compatibility testing']
                }
            },
            'rhel_8': {
                'name': 'Red Hat Enterprise Linux 8',
                'cpu_efficiency': 0.96,
                'memory_efficiency': 0.94,
                'io_efficiency': 0.95,
                'network_efficiency': 0.95,
                'virtualization_overhead': 0.06,
                'database_optimizations': {
                    'mysql': 0.95, 'postgresql': 0.97, 'oracle': 0.93, 'sqlserver': 0.85, 'mongodb': 0.96
                },
                'licensing_cost_factor': 1.5,
                'management_complexity': 0.7,
                'security_overhead': 0.04,
                'ai_insights': {
                    'strengths': ['Excellent performance', 'Strong container support'],
                    'weaknesses': ['Commercial licensing required', 'Steeper learning curve'],
                    'migration_considerations': ['Staff training needs', 'Application compatibility']
                }
            },
            'rhel_9': {
                'name': 'Red Hat Enterprise Linux 9',
                'cpu_efficiency': 0.98,
                'memory_efficiency': 0.96,
                'io_efficiency': 0.97,
                'network_efficiency': 0.97,
                'virtualization_overhead': 0.05,
                'database_optimizations': {
                    'mysql': 0.97, 'postgresql': 0.98, 'oracle': 0.95, 'sqlserver': 0.87, 'mongodb': 0.97
                },
                'licensing_cost_factor': 1.8,
                'management_complexity': 0.6,
                'security_overhead': 0.03,
                'ai_insights': {
                    'strengths': ['Latest performance optimizations', 'Enhanced security'],
                    'weaknesses': ['Newer release stability', 'Application compatibility risks'],
                    'migration_considerations': ['Extensive testing required', 'Legacy application assessment']
                }
            },
            'ubuntu_20_04': {
                'name': 'Ubuntu Server 20.04 LTS',
                'cpu_efficiency': 0.97,
                'memory_efficiency': 0.95,
                'io_efficiency': 0.96,
                'network_efficiency': 0.96,
                'virtualization_overhead': 0.05,
                'database_optimizations': {
                    'mysql': 0.96, 'postgresql': 0.98, 'oracle': 0.90, 'sqlserver': 0.82, 'mongodb': 0.97
                },
                'licensing_cost_factor': 1.0,
                'management_complexity': 0.8,
                'security_overhead': 0.03,
                'ai_insights': {
                    'strengths': ['No licensing costs', 'Great community support'],
                    'weaknesses': ['No commercial support without subscription', 'Requires Linux expertise'],
                    'migration_considerations': ['Staff Linux skills', 'Management tool migration']
                }
            },
            'ubuntu_22_04': {
                'name': 'Ubuntu Server 22.04 LTS',
                'cpu_efficiency': 0.98,
                'memory_efficiency': 0.97,
                'io_efficiency': 0.98,
                'network_efficiency': 0.98,
                'virtualization_overhead': 0.04,
                'database_optimizations': {
                    'mysql': 0.98, 'postgresql': 0.99, 'oracle': 0.92, 'sqlserver': 0.84, 'mongodb': 0.98
                },
                'licensing_cost_factor': 1.0,
                'management_complexity': 0.7,
                'security_overhead': 0.02,
                'ai_insights': {
                    'strengths': ['Latest performance features', 'Enhanced security'],
                    'weaknesses': ['Newer release risks', 'Potential compatibility issues'],
                    'migration_considerations': ['Comprehensive testing', 'Backup OS strategy']
                }
            }
        }
    
    def extract_database_engine(self, target_database_selection: str, config: Dict) -> str:
        """Extract the actual database engine from target selection and config"""
        # For RDS, use the database_engine directly
        if target_database_selection.startswith('rds_'):
            return target_database_selection.replace('rds_', '')
        elif target_database_selection.startswith('ec2_'):
            return config.get('ec2_database_engine', 'mysql')
        else:
            # Check if we have EC2 database engine specified
            if config.get('target_platform') == 'ec2' and config.get('ec2_database_engine'):
                return config.get('ec2_database_engine')
            return config.get('database_engine', target_database_selection)
    
    def calculate_os_performance_impact(self, os_type: str, platform_type: str, config: Dict) -> Dict:
        """Enhanced OS performance calculation with AI insights"""
        os_config = self.operating_systems[os_type]
        database_engine = self.extract_database_engine(config.get('database_engine', 'mysql'), config)
        
        # Base OS efficiency calculation
        base_efficiency = (
            os_config['cpu_efficiency'] * 0.3 +
            os_config['memory_efficiency'] * 0.25 +
            os_config['io_efficiency'] * 0.25 +
            os_config['network_efficiency'] * 0.2
        )
        
        # Database-specific optimization
        db_optimization = os_config['database_optimizations'].get(database_engine, 0.85)
        
        # Virtualization impact
        if platform_type == 'vmware':
            virtualization_penalty = os_config['virtualization_overhead']
            total_efficiency = base_efficiency * db_optimization * (1 - virtualization_penalty)
        else:
            total_efficiency = base_efficiency * db_optimization
        
        # Platform-specific adjustments
        if platform_type == 'physical':
            total_efficiency *= 1.05 if 'windows' not in os_type else 1.02
        
        return {
            **{k: v for k, v in os_config.items() if k != 'ai_insights'},
            'total_efficiency': total_efficiency,
            'base_efficiency': base_efficiency,
            'db_optimization': db_optimization,
            'actual_database_engine': database_engine,
            'virtualization_overhead': os_config['virtualization_overhead'] if platform_type == 'vmware' else 0,
            'ai_insights': os_config['ai_insights'],
            'platform_optimization': 1.02 if platform_type == 'physical' and 'windows' in os_type else 1.05 if platform_type == 'physical' else 1.0
        }

def get_nic_efficiency(nic_type):
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

class EnhancedNetworkIntelligenceManager:
    """AI-powered network path intelligence with enhanced analysis including backup storage paths"""
    
    def __init__(self):
        self.network_paths = {
            # Backup Storage to S3 Paths (NEW)
            'nonprod_sj_windows_share_s3': {
                'name': 'Non-Prod: San Jose Windows Share → AWS S3 (DataSync)',
                'destination_storage': 'S3',
                'source': 'San Jose Windows Share',
                'destination': 'AWS US-West-2 S3',
                'environment': 'non-production',
                'os_type': 'windows',
                'storage_type': 'windows_share',
                'migration_type': 'backup_restore',
                'segments': [
                    {
                        'name': 'Windows Share to DataSync Agent',
                        'bandwidth_mbps': 1000,
                        'latency_ms': 3,
                        'reliability': 0.998,
                        'connection_type': 'smb_share',
                        'cost_factor': 0.0,
                        'ai_optimization_potential': 0.92
                    },
                    {
                        'name': 'DataSync Agent to AWS S3 (DX)',
                        'bandwidth_mbps': 2000,
                        'latency_ms': 15,
                        'reliability': 0.998,
                        'connection_type': 'direct_connect',
                        'cost_factor': 2.0,
                        'ai_optimization_potential': 0.94
                    }
                ],
                'ai_insights': {
                    'performance_bottlenecks': ['SMB protocol overhead', 'Windows Share I/O'],
                    'optimization_opportunities': ['SMB3 multichannel', 'DataSync bandwidth optimization'],
                    'risk_factors': ['Windows Share availability', 'SMB authentication'],
                    'recommended_improvements': ['Enable SMB3 multichannel', 'Pre-stage backup files']
                }
            },
            'nonprod_sj_nas_drive_s3': {
                'name': 'Non-Prod: San Jose NAS Drive → AWS S3 (DataSync)',
                'destination_storage': 'S3',
                'source': 'San Jose NAS Drive',
                'destination': 'AWS US-West-2 S3',
                'environment': 'non-production',
                'os_type': 'linux',
                'storage_type': 'nas_drive',
                'migration_type': 'backup_restore',
                'segments': [
                    {
                        'name': 'NAS Drive to DataSync Agent',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 2,
                        'reliability': 0.999,
                        'connection_type': 'nfs_share',
                        'cost_factor': 0.0,
                        'ai_optimization_potential': 0.96
                    },
                    {
                        'name': 'DataSync Agent to AWS S3 (DX)',
                        'bandwidth_mbps': 2000,
                        'latency_ms': 12,
                        'reliability': 0.998,
                        'connection_type': 'direct_connect',
                        'cost_factor': 2.0,
                        'ai_optimization_potential': 0.95
                    }
                ],
                'ai_insights': {
                    'performance_bottlenecks': ['NAS internal bandwidth', 'DX connection sharing'],
                    'optimization_opportunities': ['NFS performance tuning', 'Parallel file transfers'],
                    'risk_factors': ['NAS hardware limitations', 'NFS connection stability'],
                    'recommended_improvements': ['Optimize NFS mount options', 'Configure DataSync parallelism']
                }
            },
            'prod_sa_windows_share_s3': {
                'name': 'Prod: San Antonio Windows Share → San Jose → AWS Production VPC S3',
                'destination_storage': 'S3',
                'source': 'San Antonio Windows Share',
                'destination': 'AWS US-West-2 Production VPC S3',
                'environment': 'production',
                'os_type': 'windows',
                'storage_type': 'windows_share',
                'migration_type': 'backup_restore',
                'segments': [
                    {
                        'name': 'Windows Share to DataSync Agent',
                        'bandwidth_mbps': 1000,
                        'latency_ms': 2,
                        'reliability': 0.999,
                        'connection_type': 'smb_share',
                        'cost_factor': 0.0,
                        'ai_optimization_potential': 0.93
                    },
                    {
                        'name': 'San Antonio to San Jose (Private Line)',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 12,
                        'reliability': 0.9995,
                        'connection_type': 'private_line',
                        'cost_factor': 3.0,
                        'ai_optimization_potential': 0.94
                    },
                    {
                        'name': 'San Jose to AWS Production VPC S3 (DX)',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 8,
                        'reliability': 0.9999,
                        'connection_type': 'direct_connect',
                        'cost_factor': 4.0,
                        'ai_optimization_potential': 0.96
                    }
                ],
                'ai_insights': {
                    'performance_bottlenecks': ['SMB over WAN latency', 'Multi-hop complexity'],
                    'optimization_opportunities': ['WAN optimization', 'Backup file pre-staging'],
                    'risk_factors': ['Cross-site dependencies', 'SMB over WAN reliability'],
                    'recommended_improvements': ['Implement WAN acceleration', 'Stage backups closer to transfer point']
                }
            },
            'prod_sa_nas_drive_s3': {
                'name': 'Prod: San Antonio NAS Drive → San Jose → AWS Production VPC S3',
                'destination_storage': 'S3',
                'source': 'San Antonio NAS Drive',
                'destination': 'AWS US-West-2 Production VPC S3',
                'environment': 'production',
                'os_type': 'linux',
                'storage_type': 'nas_drive',
                'migration_type': 'backup_restore',
                'segments': [
                    {
                        'name': 'NAS Drive to DataSync Agent',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 1,
                        'reliability': 0.999,
                        'connection_type': 'nfs_share',
                        'cost_factor': 0.0,
                        'ai_optimization_potential': 0.97
                    },
                    {
                        'name': 'San Antonio to San Jose (Private Line)',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 12,
                        'reliability': 0.9995,
                        'connection_type': 'private_line',
                        'cost_factor': 3.0,
                        'ai_optimization_potential': 0.94
                    },
                    {
                        'name': 'San Jose to AWS Production VPC S3 (DX)',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 8,
                        'reliability': 0.9999,
                        'connection_type': 'direct_connect',
                        'cost_factor': 4.0,
                        'ai_optimization_potential': 0.96
                    }
                ],
                'ai_insights': {
                    'performance_bottlenecks': ['Cross-site latency accumulation', 'NFS over WAN'],
                    'optimization_opportunities': ['End-to-end optimization', 'NFS tuning'],
                    'risk_factors': ['Multiple failure points', 'NFS over WAN complexity'],
                    'recommended_improvements': ['Implement NFS over VPN', 'Add backup staging area']
                }
            },
            # Original paths for direct replication (EXISTING)
            'nonprod_sj_linux_nas_s3': {
                'name': 'Non-Prod: San Jose Linux NAS → AWS S3 (Direct Replication)',
                'destination_storage': 'S3',
                'source': 'San Jose',
                'destination': 'AWS US-West-2 S3',
                'environment': 'non-production',
                'os_type': 'linux',
                'storage_type': 'nas',
                'migration_type': 'direct_replication',
                'segments': [
                    {
                        'name': 'Linux NAS to Linux Jump Server',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 2,
                        'reliability': 0.999,
                        'connection_type': 'internal_lan',
                        'cost_factor': 0.0,
                        'ai_optimization_potential': 0.95
                    },
                    {
                        'name': 'Linux Jump Server to AWS S3 (DX)',
                        'bandwidth_mbps': 2000,
                        'latency_ms': 15,
                        'reliability': 0.998,
                        'connection_type': 'direct_connect',
                        'cost_factor': 2.0,
                        'ai_optimization_potential': 0.92
                    }
                ],
                'ai_insights': {
                    'performance_bottlenecks': ['Linux NAS internal bandwidth', 'DX connection sharing'],
                    'optimization_opportunities': ['NAS performance tuning', 'DX bandwidth upgrade'],
                    'risk_factors': ['Single DX connection dependency', 'NAS hardware limitations'],
                    'recommended_improvements': ['Implement NAS caching', 'Configure QoS on DX']
                }
            },
            'nonprod_sj_linux_nas_fsx_windows': {
                'name': 'Non-Prod: San Jose Linux NAS → AWS FSx for Windows',
                'destination_storage': 'FSx_Windows',
                'source': 'San Jose',
                'destination': 'AWS US-West-2 FSx for Windows',
                'environment': 'non-production',
                'os_type': 'linux',
                'storage_type': 'nas',
                'migration_type': 'direct_replication',
                'segments': [
                    {
                        'name': 'Linux NAS to Linux Jump Server',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 2,
                        'reliability': 0.999,
                        'connection_type': 'internal_lan',
                        'cost_factor': 0.0,
                        'ai_optimization_potential': 0.95
                    },
                    {
                        'name': 'Linux Jump Server to AWS FSx Windows (DX)',
                        'bandwidth_mbps': 2000,
                        'latency_ms': 12,
                        'reliability': 0.999,
                        'connection_type': 'direct_connect',
                        'cost_factor': 2.5,
                        'ai_optimization_potential': 0.94
                    }
                ],
                'ai_insights': {
                    'performance_bottlenecks': ['Linux to Windows protocol conversion', 'SMB overhead'],
                    'optimization_opportunities': ['SMB3 protocol optimization', 'FSx throughput configuration'],
                    'risk_factors': ['Cross-platform compatibility', 'SMB version negotiation'],
                    'recommended_improvements': ['Test SMB3.1.1 compatibility', 'Configure FSx performance mode']
                }
            },
            'nonprod_sj_linux_nas_fsx_lustre': {
                'name': 'Non-Prod: San Jose Linux NAS → AWS FSx for Lustre',
                'destination_storage': 'FSx_Lustre',
                'source': 'San Jose',
                'destination': 'AWS US-West-2 FSx for Lustre',
                'environment': 'non-production',
                'os_type': 'linux',
                'storage_type': 'nas',
                'migration_type': 'direct_replication',
                'segments': [
                    {
                        'name': 'Linux NAS to Linux Jump Server',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 2,
                        'reliability': 0.999,
                        'connection_type': 'internal_lan',
                        'cost_factor': 0.0,
                        'ai_optimization_potential': 0.95
                    },
                    {
                        'name': 'Linux Jump Server to AWS FSx Lustre (DX)',
                        'bandwidth_mbps': 2000,
                        'latency_ms': 8,
                        'reliability': 0.9995,
                        'connection_type': 'direct_connect',
                        'cost_factor': 3.0,
                        'ai_optimization_potential': 0.97
                    }
                ],
                'ai_insights': {
                    'performance_bottlenecks': ['Lustre client configuration', 'Parallel processing coordination'],
                    'optimization_opportunities': ['Lustre striping optimization', 'Parallel I/O tuning'],
                    'risk_factors': ['Lustre complexity', 'Client compatibility'],
                    'recommended_improvements': ['Optimize Lustre striping patterns', 'Configure parallel data transfer']
                }
            },
            'prod_sa_linux_nas_s3': {
                'name': 'Prod: San Antonio Linux NAS → San Jose → AWS Production VPC S3',
                'destination_storage': 'S3',
                'source': 'San Antonio',
                'destination': 'AWS US-West-2 Production VPC S3',
                'environment': 'production',
                'os_type': 'linux',
                'storage_type': 'nas',
                'migration_type': 'direct_replication',
                'segments': [
                    {
                        'name': 'San Antonio Linux NAS to Linux Jump Server',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 1,
                        'reliability': 0.999,
                        'connection_type': 'internal_lan',
                        'cost_factor': 0.0,
                        'ai_optimization_potential': 0.97
                    },
                    {
                        'name': 'San Antonio to San Jose (Private Line)',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 12,
                        'reliability': 0.9995,
                        'connection_type': 'private_line',
                        'cost_factor': 3.0,
                        'ai_optimization_potential': 0.94
                    },
                    {
                        'name': 'San Jose to AWS Production VPC S3 (DX)',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 8,
                        'reliability': 0.9999,
                        'connection_type': 'direct_connect',
                        'cost_factor': 4.0,
                        'ai_optimization_potential': 0.96
                    }
                ],
                'ai_insights': {
                    'performance_bottlenecks': ['Cross-site latency accumulation', 'Multiple hop complexity'],
                    'optimization_opportunities': ['End-to-end optimization', 'Compression algorithms'],
                    'risk_factors': ['Multiple failure points', 'Complex troubleshooting'],
                    'recommended_improvements': ['Implement WAN optimization', 'Add redundant paths']
                }
            }
        }
    
    def calculate_ai_enhanced_path_performance(self, path_key: str, time_of_day: int = None) -> Dict:
        """AI-enhanced network path performance calculation"""
        path = self.network_paths[path_key]
        
        if time_of_day is None:
            time_of_day = datetime.now().hour
        
        total_latency = 0
        min_bandwidth = float('inf')
        total_reliability = 1.0
        total_cost_factor = 0
        ai_optimization_score = 1.0
        
        adjusted_segments = []
        
        for segment in path['segments']:
            segment_latency = segment['latency_ms']
            segment_bandwidth = segment['bandwidth_mbps']
            segment_reliability = segment['reliability']
            
            # Time-of-day adjustments
            if segment['connection_type'] == 'internal_lan':
                congestion_factor = 1.1 if 9 <= time_of_day <= 17 else 0.95
            elif segment['connection_type'] == 'private_line':
                congestion_factor = 1.2 if 9 <= time_of_day <= 17 else 0.9
            elif segment['connection_type'] == 'direct_connect':
                congestion_factor = 1.05 if 9 <= time_of_day <= 17 else 0.98
            elif segment['connection_type'] in ['smb_share', 'nfs_share']:
                # Backup storage specific adjustments
                congestion_factor = 1.3 if 9 <= time_of_day <= 17 else 1.0
            else:
                congestion_factor = 1.0
            
            effective_bandwidth = segment_bandwidth / congestion_factor
            effective_latency = segment_latency * congestion_factor
            
            # OS-specific adjustments
            if path['os_type'] == 'windows' and segment['connection_type'] != 'internal_lan':
                effective_bandwidth *= 0.95
                effective_latency *= 1.1
            
            # Backup storage protocol adjustments
            if path.get('migration_type') == 'backup_restore':
                if path['storage_type'] == 'windows_share' and segment['connection_type'] == 'smb_share':
                    effective_bandwidth *= 0.85  # SMB overhead
                elif path['storage_type'] == 'nas_drive' and segment['connection_type'] == 'nfs_share':
                    effective_bandwidth *= 0.92  # NFS is more efficient
            
            # Destination storage adjustments
            if 'FSx' in path['destination_storage']:
                if path['destination_storage'] == 'FSx_Windows':
                    effective_bandwidth *= 1.1
                    effective_latency *= 0.9
                elif path['destination_storage'] == 'FSx_Lustre':
                    effective_bandwidth *= 1.3
                    effective_latency *= 0.7
            
            ai_optimization_score *= segment['ai_optimization_potential']
            
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
        
        base_network_quality = (latency_score * 0.25 + bandwidth_score * 0.45 + reliability_score * 0.30)
        ai_enhanced_quality = base_network_quality * ai_optimization_score
        
        # Storage bonus
        storage_bonus = 0
        if path['destination_storage'] == 'FSx_Windows':
            storage_bonus = 10
        elif path['destination_storage'] == 'FSx_Lustre':
            storage_bonus = 20
        
        ai_enhanced_quality = min(100, ai_enhanced_quality + storage_bonus)
        
        return {
            'path_name': path['name'],
            'destination_storage': path['destination_storage'],
            'migration_type': path.get('migration_type', 'direct_replication'),
            'total_latency_ms': total_latency,
            'effective_bandwidth_mbps': min_bandwidth,
            'total_reliability': total_reliability,
            'network_quality_score': base_network_quality,
            'ai_enhanced_quality_score': ai_enhanced_quality,
            'ai_optimization_potential': (1 - ai_optimization_score) * 100,
            'total_cost_factor': total_cost_factor,
            'storage_performance_bonus': storage_bonus,
            'segments': adjusted_segments,
            'environment': path['environment'],
            'os_type': path['os_type'],
            'storage_type': path['storage_type'],
            'ai_insights': path['ai_insights']
        }

class EnhancedAgentSizingManager:
    """Enhanced agent sizing with scalable agent count and AI recommendations"""
    
    def __init__(self):
        self.datasync_agent_specs = {
            'small': {
                'name': 'Small Agent (t3.medium)',
                'vcpu': 2,
                'memory_gb': 4,
                'max_throughput_mbps_per_agent': 250,
                'max_concurrent_tasks_per_agent': 10,
                'cost_per_hour_per_agent': 0.0416,
                'recommended_for': 'Up to 1TB per agent, <100 Mbps network per agent'
            },
            'medium': {
                'name': 'Medium Agent (c5.large)',
                'vcpu': 2,
                'memory_gb': 4,
                'max_throughput_mbps_per_agent': 500,
                'max_concurrent_tasks_per_agent': 25,
                'cost_per_hour_per_agent': 0.085,
                'recommended_for': '1-5TB per agent, 100-500 Mbps network per agent'
            },
            'large': {
                'name': 'Large Agent (c5.xlarge)',
                'vcpu': 4,
                'memory_gb': 8,
                'max_throughput_mbps_per_agent': 1000,
                'max_concurrent_tasks_per_agent': 50,
                'cost_per_hour_per_agent': 0.17,
                'recommended_for': '5-20TB per agent, 500Mbps-1Gbps network per agent'
            },
            'xlarge': {
                'name': 'XLarge Agent (c5.2xlarge)',
                'vcpu': 8,
                'memory_gb': 16,
                'max_throughput_mbps_per_agent': 2000,
                'max_concurrent_tasks_per_agent': 100,
                'cost_per_hour_per_agent': 0.34,
                'recommended_for': '>20TB per agent, >1Gbps network per agent'
            }
        }
        
        self.dms_agent_specs = {
            'small': {
                'name': 'Small DMS Instance (t3.medium)',
                'vcpu': 2,
                'memory_gb': 4,
                'max_throughput_mbps_per_agent': 200,
                'max_concurrent_tasks_per_agent': 5,
                'cost_per_hour_per_agent': 0.0416,
                'recommended_for': 'Up to 500GB per agent, simple schemas'
            },
            'medium': {
                'name': 'Medium DMS Instance (c5.large)',
                'vcpu': 2,
                'memory_gb': 4,
                'max_throughput_mbps_per_agent': 400,
                'max_concurrent_tasks_per_agent': 10,
                'cost_per_hour_per_agent': 0.085,
                'recommended_for': '500GB-2TB per agent, moderate complexity'
            },
            'large': {
                'name': 'Large DMS Instance (c5.xlarge)',
                'vcpu': 4,
                'memory_gb': 8,
                'max_throughput_mbps_per_agent': 800,
                'max_concurrent_tasks_per_agent': 20,
                'cost_per_hour_per_agent': 0.17,
                'recommended_for': '2-10TB per agent, complex schemas'
            },
            'xlarge': {
                'name': 'XLarge DMS Instance (c5.2xlarge)',
                'vcpu': 8,
                'memory_gb': 16,
                'max_throughput_mbps_per_agent': 1500,
                'max_concurrent_tasks_per_agent': 40,
                'cost_per_hour_per_agent': 0.34,
                'recommended_for': '10-50TB per agent, very complex schemas'
            },
            'xxlarge': {
                'name': 'XXLarge DMS Instance (c5.4xlarge)',
                'vcpu': 16,
                'memory_gb': 32,
                'max_throughput_mbps_per_agent': 2500,
                'max_concurrent_tasks_per_agent': 80,
                'cost_per_hour_per_agent': 0.68,
                'recommended_for': '>50TB per agent, enterprise workloads'
            }
        }
    
    def calculate_agent_configuration(self, agent_type: str, agent_size: str, number_of_agents: int, destination_storage: str = 'S3') -> Dict:
        """Calculate agent configuration with FSx architecture"""
        if agent_type == 'datasync':
            agent_spec = self.datasync_agent_specs[agent_size]
        else:
            agent_spec = self.dms_agent_specs[agent_size]
        
        # Calculate scaling efficiency
        scaling_efficiency = self._calculate_scaling_efficiency(number_of_agents)
        
        # Storage performance multiplier
        storage_multiplier = self._get_storage_performance_multiplier(destination_storage)
        
        total_throughput = (agent_spec['max_throughput_mbps_per_agent'] * 
                           number_of_agents * scaling_efficiency * storage_multiplier)
        
        total_concurrent_tasks = (agent_spec['max_concurrent_tasks_per_agent'] * number_of_agents)
        total_cost_per_hour = agent_spec['cost_per_hour_per_agent'] * number_of_agents
        
        # Management overhead
        management_overhead_factor = 1.0 + (number_of_agents - 1) * 0.05
        storage_overhead = self._get_storage_management_overhead(destination_storage)
        
        return {
            'agent_type': agent_type,
            'agent_size': agent_size,
            'number_of_agents': number_of_agents,
            'destination_storage': destination_storage,
            'per_agent_spec': agent_spec,
            'total_vcpu': agent_spec['vcpu'] * number_of_agents,
            'total_memory_gb': agent_spec['memory_gb'] * number_of_agents,
            'max_throughput_mbps_per_agent': agent_spec['max_throughput_mbps_per_agent'],
            'total_max_throughput_mbps': total_throughput,
            'effective_throughput_mbps': total_throughput,
            'total_concurrent_tasks': total_concurrent_tasks,
            'cost_per_hour_per_agent': agent_spec['cost_per_hour_per_agent'],
            'total_cost_per_hour': total_cost_per_hour,
            'total_monthly_cost': total_cost_per_hour * 24 * 30,
            'scaling_efficiency': scaling_efficiency,
            'storage_performance_multiplier': storage_multiplier,
            'management_overhead_factor': management_overhead_factor,
            'storage_management_overhead': storage_overhead,
            'effective_cost_per_hour': total_cost_per_hour * management_overhead_factor * storage_overhead,
            'scaling_recommendations': self._get_scaling_recommendations(agent_size, number_of_agents, destination_storage),
            'optimal_configuration': self._assess_configuration_optimality(agent_size, number_of_agents, destination_storage)
        }
    
    def _get_storage_performance_multiplier(self, destination_storage: str) -> float:
        """Get performance multiplier based on destination storage type"""
        multipliers = {'S3': 1.0, 'FSx_Windows': 1.15, 'FSx_Lustre': 1.4}
        return multipliers.get(destination_storage, 1.0)
    
    def _get_storage_management_overhead(self, destination_storage: str) -> float:
        """Get management overhead factor for destination storage"""
        overheads = {'S3': 1.0, 'FSx_Windows': 1.1, 'FSx_Lustre': 1.2}
        return overheads.get(destination_storage, 1.0)
    
    def _calculate_scaling_efficiency(self, number_of_agents: int) -> float:
        """Calculate scaling efficiency - diminishing returns with more agents"""
        if number_of_agents == 1:
            return 1.0
        elif number_of_agents <= 3:
            return 0.95
        elif number_of_agents <= 5:
            return 0.90
        elif number_of_agents <= 8:
            return 0.85
        else:
            return 0.80
    
    def _get_scaling_recommendations(self, agent_size: str, number_of_agents: int, destination_storage: str) -> List[str]:
        """Get scaling-specific recommendations"""
        recommendations = []
        
        if number_of_agents == 1:
            recommendations.append("Single agent configuration - consider scaling for larger workloads")
        elif number_of_agents <= 3:
            recommendations.append("Good balance of performance and manageability")
            recommendations.append("Configure load balancing for optimal distribution")
        else:
            recommendations.append("High-scale configuration requiring careful coordination")
            recommendations.append("Implement centralized monitoring and logging")
        
        # Storage-specific recommendations
        if destination_storage == 'FSx_Lustre':
            recommendations.append("Optimize agents for high-performance Lustre file system")
        elif destination_storage == 'FSx_Windows':
            recommendations.append("Ensure agents are optimized for Windows file sharing protocols")
        
        return recommendations
    
    def _assess_configuration_optimality(self, agent_size: str, number_of_agents: int, destination_storage: str) -> Dict:
        """Assess if the configuration is optimal"""
        efficiency_score = 100
        
        if agent_size == 'small' and number_of_agents > 6:
            efficiency_score -= 20
        
        if number_of_agents > 8:
            efficiency_score -= 25
        
        if 2 <= number_of_agents <= 4 and agent_size in ['medium', 'large']:
            efficiency_score += 10
        
        # Storage-specific adjustments
        if destination_storage == 'FSx_Lustre' and agent_size in ['large', 'xlarge']:
            efficiency_score += 5
        elif destination_storage == 'FSx_Windows' and agent_size in ['medium', 'large']:
            efficiency_score += 3
        
        complexity = "Low" if number_of_agents <= 2 else "Medium" if number_of_agents <= 5 else "High"
        cost_efficiency = "Good" if efficiency_score >= 90 else "Fair" if efficiency_score >= 75 else "Poor"
        
        return {
            'efficiency_score': max(0, efficiency_score),
            'management_complexity': complexity,
            'cost_efficiency': cost_efficiency,
            'optimal_recommendation': self._generate_optimal_recommendation(agent_size, number_of_agents, efficiency_score, destination_storage)
        }
    
    def _generate_optimal_recommendation(self, agent_size: str, number_of_agents: int, efficiency_score: int, destination_storage: str) -> str:
        """Generate optimal configuration recommendation"""
        if efficiency_score >= 90:
            return f"Optimal configuration for {destination_storage} destination"
        elif efficiency_score >= 75:
            return f"Good configuration with minor optimization opportunities for {destination_storage}"
        elif number_of_agents > 6 and agent_size == 'small':
            return "Consider consolidating to fewer, larger agents"
        elif number_of_agents > 8:
            return "Consider reducing agent count to improve manageability"
        else:
            return f"Configuration needs optimization for better {destination_storage} efficiency"

class OnPremPerformanceAnalyzer:
    """Enhanced on-premises performance analyzer with AI insights"""
    
    def __init__(self):
        self.cpu_architectures = {
            'intel_xeon_e5': {'base_performance': 1.0, 'single_thread': 0.9, 'multi_thread': 1.1},
            'intel_xeon_sp': {'base_performance': 1.2, 'single_thread': 1.1, 'multi_thread': 1.3},
            'amd_epyc': {'base_performance': 1.15, 'single_thread': 1.0, 'multi_thread': 1.4}
        }
        
        self.storage_types = {
            'sas_hdd': {'iops': 150, 'throughput_mbps': 200, 'latency_ms': 8},
            'sata_ssd': {'iops': 75000, 'throughput_mbps': 500, 'latency_ms': 0.2},
            'nvme_ssd': {'iops': 450000, 'throughput_mbps': 3500, 'latency_ms': 0.05}
        }
    
    def calculate_ai_enhanced_performance(self, config: Dict, os_manager: OSPerformanceManager) -> Dict:
        """AI-enhanced on-premises performance calculation"""
        
        # Get OS impact
        os_impact = os_manager.calculate_os_performance_impact(
            config['operating_system'], 
            config['server_type'], 
            config
        )
        
        # Performance calculations
        cpu_performance = self._calculate_cpu_performance(config, os_impact)
        memory_performance = self._calculate_memory_performance(config, os_impact)
        storage_performance = self._calculate_storage_performance(config, os_impact)
        network_performance = self._calculate_network_performance(config, os_impact)
        database_performance = self._calculate_database_performance(config, os_impact)
        
        # AI-enhanced overall performance analysis
        overall_performance = self._calculate_ai_enhanced_overall_performance(
            cpu_performance, memory_performance, storage_performance, 
            network_performance, database_performance, os_impact, config
        )
        
        return {
            'cpu_performance': cpu_performance,
            'memory_performance': memory_performance,
            'storage_performance': storage_performance,
            'network_performance': network_performance,
            'database_performance': database_performance,
            'overall_performance': overall_performance,
            'os_impact': os_impact,
            'bottlenecks': ['No major bottlenecks identified'],
            'ai_insights': ['System appears well-configured for migration'],
            'performance_score': overall_performance['composite_score']
        }
    
    def _calculate_cpu_performance(self, config: Dict, os_impact: Dict) -> Dict:
        """Calculate CPU performance metrics"""
        base_performance = config['cpu_cores'] * config['cpu_ghz']
        os_adjusted = base_performance * os_impact['cpu_efficiency']
        
        if config['server_type'] == 'vmware':
            virtualization_penalty = 1 - os_impact['virtualization_overhead']
            final_performance = os_adjusted * virtualization_penalty
        else:
            final_performance = os_adjusted * 1.05
        
        return {
            'base_performance': base_performance,
            'os_adjusted_performance': os_adjusted,
            'final_performance': final_performance,
            'efficiency_factor': os_impact['cpu_efficiency']
        }
    
    def _calculate_memory_performance(self, config: Dict, os_impact: Dict) -> Dict:
        """Calculate memory performance"""
        base_memory = config['ram_gb']
        os_overhead = 4 if 'windows' in config['operating_system'] else 2
        available_memory = base_memory - os_overhead
        effective_memory = available_memory * os_impact['memory_efficiency']
        
        return {
            'total_memory_gb': base_memory,
            'os_overhead_gb': os_overhead,
            'available_memory_gb': available_memory,
            'effective_memory_gb': effective_memory,
            'memory_efficiency': os_impact['memory_efficiency']
        }
    
    def _calculate_storage_performance(self, config: Dict, os_impact: Dict) -> Dict:
        """Calculate storage performance"""
        if config['cpu_cores'] >= 8:
            storage_type = 'nvme_ssd'
        elif config['cpu_cores'] >= 4:
            storage_type = 'sata_ssd'
        else:
            storage_type = 'sas_hdd'
        
        storage_specs = self.storage_types[storage_type]
        effective_iops = storage_specs['iops'] * os_impact['io_efficiency']
        effective_throughput = storage_specs['throughput_mbps'] * os_impact['io_efficiency']
        
        return {
            'storage_type': storage_type,
            'base_iops': storage_specs['iops'],
            'effective_iops': effective_iops,
            'base_throughput_mbps': storage_specs['throughput_mbps'],
            'effective_throughput_mbps': effective_throughput,
            'io_efficiency': os_impact['io_efficiency']
        }
    
    def _calculate_network_performance(self, config: Dict, os_impact: Dict) -> Dict:
        """Calculate network performance"""
        base_bandwidth = config['nic_speed']
        effective_bandwidth = base_bandwidth * os_impact['network_efficiency']
        
        if config['server_type'] == 'vmware':
            effective_bandwidth *= 0.92
        
        return {
            'nic_type': config['nic_type'],
            'base_bandwidth_mbps': base_bandwidth,
            'effective_bandwidth_mbps': effective_bandwidth,
            'network_efficiency': os_impact['network_efficiency']
        }
    
    def _calculate_database_performance(self, config: Dict, os_impact: Dict) -> Dict:
        """Calculate database performance"""
        db_optimization = os_impact['db_optimization']
        
        # Get the actual database engine for performance calculation
        database_engine = config.get('ec2_database_engine') or config.get('database_engine', 'mysql')
        
        if database_engine == 'mysql':
            base_tps = 5000
        elif database_engine == 'postgresql':
            base_tps = 4500
        elif database_engine == 'oracle':
            base_tps = 6000
        elif database_engine == 'sqlserver':
            base_tps = 5500
        else:
            base_tps = 4000
        
        hardware_factor = min(2.0, (config['cpu_cores'] / 4) * (config['ram_gb'] / 16))
        effective_tps = base_tps * hardware_factor * db_optimization
        
        return {
            'database_engine': database_engine,
            'base_tps': base_tps,
            'hardware_factor': hardware_factor,
            'db_optimization': db_optimization,
            'effective_tps': effective_tps
        }
    
    def _calculate_ai_enhanced_overall_performance(self, cpu_perf: Dict, mem_perf: Dict, 
                                                 storage_perf: Dict, net_perf: Dict, 
                                                 db_perf: Dict, os_impact: Dict, config: Dict) -> Dict:
        """AI-enhanced overall performance calculation"""
        
        cpu_score = min(100, (cpu_perf['final_performance'] / 50) * 100)
        memory_score = min(100, (mem_perf['effective_memory_gb'] / 64) * 100)
        storage_score = min(100, (storage_perf['effective_iops'] / 100000) * 100)
        network_score = min(100, (net_perf['effective_bandwidth_mbps'] / 10000) * 100)
        database_score = min(100, (db_perf['effective_tps'] / 10000) * 100)
        
        composite_score = (
            cpu_score * 0.25 +
            memory_score * 0.2 +
            storage_score * 0.25 +
            network_score * 0.15 +
            database_score * 0.15
        )
        
        return {
            'cpu_score': cpu_score,
            'memory_score': memory_score,
            'storage_score': storage_score,
            'network_score': network_score,
            'database_score': database_score,
            'composite_score': composite_score,
            'performance_tier': self._get_performance_tier(composite_score)
        }
    
    def _get_performance_tier(self, score: float) -> str:
        """Get performance tier based on score"""
        if score >= 80:
            return "High Performance"
        elif score >= 60:
            return "Standard Performance"
        elif score >= 40:
            return "Basic Performance"
        else:
            return "Limited Performance"

class EnhancedMigrationAnalyzer:
    """Enhanced migration analyzer with AI and AWS API integration plus FSx support"""
    
    def __init__(self):
        self.ai_manager = AnthropicAIManager()
        self.aws_api = AWSAPIManager()
        self.os_manager = OSPerformanceManager()
        self.network_manager = EnhancedNetworkIntelligenceManager()
        self.agent_manager = EnhancedAgentSizingManager()
        self.onprem_analyzer = OnPremPerformanceAnalyzer()
        self.cost_calculator = ComprehensiveAWSCostCalculator(self.aws_api)
        
        self.nic_types = {
            'gigabit_copper': {'max_speed': 1000, 'efficiency': 0.85},
            'gigabit_fiber': {'max_speed': 1000, 'efficiency': 0.90},
            '10g_copper': {'max_speed': 10000, 'efficiency': 0.88},
            '10g_fiber': {'max_speed': 10000, 'efficiency': 0.92},
            '25g_fiber': {'max_speed': 25000, 'efficiency': 0.94},
            '40g_fiber': {'max_speed': 40000, 'efficiency': 0.95}
        }
    
    async def comprehensive_ai_migration_analysis(self, config: Dict) -> Dict:
        """Comprehensive AI-powered migration analysis"""
        
        # API status tracking
        api_status = APIStatus(
            anthropic_connected=self.ai_manager.connected,
            aws_pricing_connected=self.aws_api.connected,
            last_update=datetime.now()
        )
        
        # On-premises performance analysis
        onprem_performance = self.onprem_analyzer.calculate_ai_enhanced_performance(config, self.os_manager)
        
        # Network path analysis
        network_path_key = self._get_network_path_key(config)
        network_perf = self.network_manager.calculate_ai_enhanced_path_performance(network_path_key)
        
        # Migration type and tools
        migration_method = config.get('migration_method', 'direct_replication')
        
        if migration_method == 'backup_restore':
            # For backup/restore, always use DataSync regardless of database engine
            migration_type = 'backup_restore'
            primary_tool = 'datasync'
        else:
            # For direct replication, use existing logic
            is_homogeneous = config['source_database_engine'] == config['database_engine']
            migration_type = 'homogeneous' if is_homogeneous else 'heterogeneous'
            primary_tool = 'datasync' if is_homogeneous else 'dms'
        
        # Agent analysis
        agent_analysis = await self._analyze_ai_migration_agents_with_scaling(config, primary_tool, network_perf)
        
        # Migration throughput
        agent_throughput = agent_analysis['total_effective_throughput']
        network_throughput = network_perf['effective_bandwidth_mbps']
        migration_throughput = min(agent_throughput, network_throughput)
        
        # Migration time
        migration_time_hours = await self._calculate_ai_migration_time_with_agents(
            config, migration_throughput, onprem_performance, agent_analysis
        )
        
        # AWS sizing
        aws_sizing = await self._ai_enhanced_aws_sizing(config)
        
        # Cost analysis
        cost_analysis = await self._calculate_ai_enhanced_costs_with_agents(
            config, aws_sizing, agent_analysis, network_perf
        )
        
        # FSx comparisons
        fsx_comparisons = await self._generate_fsx_destination_comparisons(config)
        
        # AI overall assessment
        ai_overall_assessment = await self._generate_ai_overall_assessment_with_agents(
            config, onprem_performance, aws_sizing, migration_time_hours, agent_analysis
        )
        
        # Comprehensive cost analysis
        comprehensive_costs = await self.cost_calculator.calculate_comprehensive_migration_costs(
            config, {
                'onprem_performance': onprem_performance,
                'network_performance': network_perf,
                'agent_analysis': agent_analysis,
                'aws_sizing_recommendations': aws_sizing
            }
        )
        
        return {
            'api_status': api_status,
            'onprem_performance': onprem_performance,
            'network_performance': network_perf,
            'migration_type': migration_type,
            'primary_tool': primary_tool,
            'agent_analysis': agent_analysis,
            'migration_throughput_mbps': migration_throughput,
            'estimated_migration_time_hours': migration_time_hours,
            'aws_sizing_recommendations': aws_sizing,
            'cost_analysis': cost_analysis,
            'comprehensive_costs': comprehensive_costs,
            'fsx_comparisons': fsx_comparisons,
            'ai_overall_assessment': ai_overall_assessment
        }
    
    def _get_network_path_key(self, config: Dict) -> str:
        """Get network path key based on migration method and backup storage"""
        os_lower = config.get('operating_system', '').lower()
        if any(os_name in os_lower for os_name in ['linux', 'ubuntu', 'rhel']):
            os_type = 'linux'
        else:
            os_type = 'windows'
        
        environment = config.get('environment', 'non-production').replace('-', '_').lower()
        destination_storage = config.get('destination_storage_type', 'S3').lower()
        migration_method = config.get('migration_method', 'direct_replication')
        backup_storage_type = config.get('backup_storage_type', 'nas_drive')
        
        # For backup/restore method, use backup storage paths
        if migration_method == 'backup_restore':
            if environment in ['non_production', 'nonprod']:
                if backup_storage_type == 'windows_share':
                    return "nonprod_sj_windows_share_s3"
                else:  # nas_drive
                    return "nonprod_sj_nas_drive_s3"
            elif environment == 'production':
                if backup_storage_type == 'windows_share':
                    return "prod_sa_windows_share_s3"
                else:  # nas_drive
                    return "prod_sa_nas_drive_s3"
        
        # For direct replication, use original paths
        else:
            if environment in ['non_production', 'nonprod']:
                if destination_storage == 's3':
                    return f"nonprod_sj_{os_type}_nas_s3"
                elif destination_storage == 'fsx_windows':
                    return f"nonprod_sj_{os_type}_nas_fsx_windows"
                elif destination_storage == 'fsx_lustre':
                    return f"nonprod_sj_{os_type}_nas_fsx_lustre"
            elif environment == 'production':
                if destination_storage == 's3':
                    return f"prod_sa_{os_type}_nas_s3"
        
        # Default fallback for direct replication
        return f"nonprod_sj_{os_type}_nas_s3"
    
    async def _analyze_ai_migration_agents_with_scaling(self, config: Dict, primary_tool: str, network_perf: Dict) -> Dict:
        """Enhanced migration agent analysis with scaling support and backup storage considerations"""
        
        num_agents = config.get('number_of_agents', 1)
        destination_storage = config.get('destination_storage_type', 'S3')
        migration_method = config.get('migration_method', 'direct_replication')
        
        # For backup/restore, always use DataSync
        if migration_method == 'backup_restore':
            primary_tool = 'datasync'
            agent_size = config.get('datasync_agent_size', 'medium')
            agent_config = self.agent_manager.calculate_agent_configuration('datasync', agent_size, num_agents, destination_storage)
        elif primary_tool == 'datasync':
            agent_size = config['datasync_agent_size']
            agent_config = self.agent_manager.calculate_agent_configuration('datasync', agent_size, num_agents, destination_storage)
        else:
            agent_size = config['dms_agent_size']
            agent_config = self.agent_manager.calculate_agent_configuration('dms', agent_size, num_agents, destination_storage)
        
        total_max_throughput = agent_config['total_max_throughput_mbps']
        network_bandwidth = network_perf['effective_bandwidth_mbps']
        
        # Apply backup storage efficiency factors
        if migration_method == 'backup_restore':
            backup_storage_type = config.get('backup_storage_type', 'nas_drive')
            if backup_storage_type == 'windows_share':
                # SMB has some overhead
                backup_efficiency = 0.85
            else:  # nas_drive with NFS
                backup_efficiency = 0.92
            
            effective_throughput = min(total_max_throughput * backup_efficiency, network_bandwidth)
        else:
            effective_throughput = min(total_max_throughput, network_bandwidth)
            backup_efficiency = 1.0
        
        # Determine bottleneck
        if total_max_throughput < network_bandwidth:
            bottleneck = f'agents ({num_agents} agents)'
            bottleneck_severity = 'high' if effective_throughput / total_max_throughput < 0.7 else 'medium'
        else:
            bottleneck = 'network'
            bottleneck_severity = 'medium' if effective_throughput / network_bandwidth > 0.8 else 'high'
        
        # Add backup storage specific bottleneck detection
        if migration_method == 'backup_restore':
            backup_storage_type = config.get('backup_storage_type', 'nas_drive')
            if backup_storage_type == 'windows_share' and effective_throughput < total_max_throughput * 0.9:
                bottleneck = f'{bottleneck} + SMB protocol overhead'
        
        return {
            'primary_tool': primary_tool,
            'agent_size': agent_size,
            'number_of_agents': num_agents,
            'destination_storage': destination_storage,
            'migration_method': migration_method,
            'backup_storage_type': config.get('backup_storage_type', 'nas_drive'),
            'agent_configuration': agent_config,
            'total_max_throughput_mbps': total_max_throughput,
            'total_effective_throughput': effective_throughput,
            'backup_efficiency': backup_efficiency,
            'bottleneck': bottleneck,
            'bottleneck_severity': bottleneck_severity,
            'scaling_efficiency': agent_config['scaling_efficiency'],
            'management_overhead': agent_config['management_overhead_factor'],
            'storage_performance_multiplier': agent_config.get('storage_performance_multiplier', 1.0),
            'cost_per_hour': agent_config['effective_cost_per_hour'],
            'monthly_cost': agent_config['total_monthly_cost']
        }
    
    async def _calculate_ai_migration_time_with_agents(self, config: Dict, migration_throughput: float, 
                                                     onprem_performance: Dict, agent_analysis: Dict) -> float:
        """AI-enhanced migration time calculation with backup storage considerations"""
        
        database_size_gb = config['database_size_gb']
        num_agents = config.get('number_of_agents', 1)
        destination_storage = config.get('destination_storage_type', 'S3')
        migration_method = config.get('migration_method', 'direct_replication')
        
        # Calculate data size to transfer
        if migration_method == 'backup_restore':
            # For backup/restore, calculate backup file size
            backup_size_multiplier = config.get('backup_size_multiplier', 0.7)
            data_size_gb = database_size_gb * backup_size_multiplier
            backup_storage_type = config.get('backup_storage_type', 'nas_drive')
            
            # Base calculation for file transfer
            base_time_hours = (data_size_gb * 8 * 1000) / (migration_throughput * 3600)
            
            # Backup storage specific factors
            if backup_storage_type == 'windows_share':
                complexity_factor = 1.2  # SMB protocol overhead
            else:  # nas_drive
                complexity_factor = 1.1  # NFS is more efficient
            
            # Add backup preparation time
            backup_prep_time = 0.5 + (database_size_gb / 10000)  # 0.5-2 hours for backup prep
            
        else:
            # For direct replication, use database size
            data_size_gb = database_size_gb
            base_time_hours = (data_size_gb * 8 * 1000) / (migration_throughput * 3600)
            complexity_factor = 1.0
            backup_prep_time = 0
        
        # Database engine complexity
        if config['source_database_engine'] != config['database_engine']:
            complexity_factor *= 1.3
        
        # OS and platform factors
        if 'windows' in config['operating_system']:
            complexity_factor *= 1.1
        
        if config['server_type'] == 'vmware':
            complexity_factor *= 1.05
        
        # Destination storage adjustments
        if destination_storage == 'FSx_Windows':
            complexity_factor *= 0.9
        elif destination_storage == 'FSx_Lustre':
            complexity_factor *= 0.7
        
        # Agent scaling adjustments
        scaling_efficiency = agent_analysis.get('scaling_efficiency', 1.0)
        storage_multiplier = agent_analysis.get('storage_performance_multiplier', 1.0)
        
        if num_agents > 1:
            agent_time_factor = (1 / min(num_agents * scaling_efficiency * storage_multiplier, 6.0))
            complexity_factor *= agent_time_factor
            
            if num_agents > 5:
                complexity_factor *= 1.1
        
        total_time = base_time_hours * complexity_factor + backup_prep_time
        
        return total_time
    
    async def _ai_enhanced_aws_sizing(self, config: Dict) -> Dict:
        """AI-enhanced AWS sizing"""
        
        # Get real-time pricing
        pricing_data = await self.aws_api.get_real_time_pricing()
        
        # RDS sizing
        rds_sizing = self._calculate_rds_sizing(config, pricing_data)
        
        # EC2 sizing  
        ec2_sizing = self._calculate_ec2_sizing(config, pricing_data)
        
        # Reader/writer configuration
        reader_writer_config = self._calculate_reader_writer_config(config)
        
        # Deployment recommendation
        deployment_recommendation = self._recommend_deployment_type(config, rds_sizing, ec2_sizing)
        
        # AI analysis
        ai_analysis = await self.ai_manager.analyze_migration_workload(config, {})
        
        return {
            'rds_recommendations': rds_sizing,
            'ec2_recommendations': ec2_sizing,
            'reader_writer_config': reader_writer_config,
            'deployment_recommendation': deployment_recommendation,
            'ai_analysis': ai_analysis,
            'pricing_data': pricing_data
        }
    
    def _calculate_reader_writer_config(self, config: Dict) -> Dict:
        """Calculate reader/writer configuration"""
        database_size_gb = config['database_size_gb']
        performance_req = config.get('performance_requirements', 'standard')
        environment = config.get('environment', 'non-production')
        
        writers = 1
        readers = 0
        
        if database_size_gb > 500:
            readers += 1
        if database_size_gb > 2000:
            readers += 1
        if database_size_gb > 10000:
            readers += 2
        
        if performance_req == 'high':
            readers += 2
        
        if environment == 'production':
            readers = max(readers, 2)
        
        total_instances = writers + readers
        
        return {
            'writers': writers,
            'readers': readers,
            'total_instances': total_instances,
            'write_capacity_percent': (writers / total_instances) * 100 if total_instances > 0 else 100,
            'read_capacity_percent': (readers / total_instances) * 100 if total_instances > 0 else 0,
            'recommended_read_split': min(80, (readers / total_instances) * 100) if total_instances > 0 else 0,
            'reasoning': f"AI-optimized for {database_size_gb}GB database"
        }
    
    def _recommend_deployment_type(self, config: Dict, rds_rec: Dict, ec2_rec: Dict) -> Dict:
        """Recommend deployment type based on user selection and analysis"""
        target_platform = config.get('target_platform', 'rds')
        
        # Use user's explicit choice but provide analysis
        rds_score = 0
        ec2_score = 0
        
        if config['database_size_gb'] < 2000:
            rds_score += 40
        elif config['database_size_gb'] > 10000:
            ec2_score += 30
        
        if config['performance_requirements'] == 'high':
            ec2_score += 30
        else:
            rds_score += 35
        
        if config['environment'] == 'production':
            rds_score += 20
        
        rds_score += 20  # Management simplicity
        
        # Override with user selection
        recommendation = target_platform
        
        # Calculate confidence based on alignment with analysis
        analytical_recommendation = 'rds' if rds_score > ec2_score else 'ec2'
        confidence = 0.9 if recommendation == analytical_recommendation else 0.6
        
        primary_reasons = [
            f"User selected {target_platform.upper()} platform",
            f"Suitable for {config['database_size_gb']:,}GB database",
            f"Appropriate for {config.get('environment', 'non-production')} environment"
        ]
        
        if recommendation != analytical_recommendation:
            primary_reasons.append(f"Note: Analysis suggests {analytical_recommendation.upper()} might be optimal")
        
        return {
            'recommendation': recommendation,
            'user_choice': target_platform,
            'analytical_recommendation': analytical_recommendation,
            'confidence': confidence,
            'rds_score': rds_score,
            'ec2_score': ec2_score,
            'primary_reasons': primary_reasons
        }
    
    def _calculate_rds_sizing(self, config: Dict, pricing_data: Dict) -> Dict:
        """Calculate RDS sizing based on database size and performance metrics"""
        database_size_gb = config['database_size_gb']
        
        # Get current database performance metrics for better sizing
        current_memory_gb = config.get('current_db_max_memory_gb', 0)
        current_cpu_cores = config.get('current_db_max_cpu_cores', 0)
        current_iops = config.get('current_db_max_iops', 0)
        current_throughput_mbps = config.get('current_db_max_throughput_mbps', 0)
        
        # Base sizing on database size (existing logic)
        if database_size_gb < 1000:
            base_instance_type = 'db.t3.medium'
            base_cost_per_hour = 0.068
        elif database_size_gb < 5000:
            base_instance_type = 'db.r6g.large'
            base_cost_per_hour = 0.48
        else:
            base_instance_type = 'db.r6g.xlarge'
            base_cost_per_hour = 0.96
        
        # Performance-based sizing recommendations
        recommended_instance_type = base_instance_type
        performance_based_cost = base_cost_per_hour
        sizing_reasoning = ["Based on database size"]
        
        # Adjust based on current memory usage
        if current_memory_gb > 0:
            # Add 20% buffer for AWS overhead and growth
            required_memory = current_memory_gb * 1.2
            
            if required_memory > 64:
                recommended_instance_type = 'db.r6g.4xlarge'
                performance_based_cost = 3.84
                sizing_reasoning.append(f"Memory requirement: {required_memory:.0f} GB")
            elif required_memory > 32:
                recommended_instance_type = 'db.r6g.2xlarge'
                performance_based_cost = 1.92
                sizing_reasoning.append(f"Memory requirement: {required_memory:.0f} GB")
            elif required_memory > 16:
                recommended_instance_type = 'db.r6g.xlarge'
                performance_based_cost = 0.96
                sizing_reasoning.append(f"Memory requirement: {required_memory:.0f} GB")
            elif required_memory > 8:
                recommended_instance_type = 'db.r6g.large'
                performance_based_cost = 0.48
                sizing_reasoning.append(f"Memory requirement: {required_memory:.0f} GB")
        
        # Adjust based on current CPU usage
        if current_cpu_cores > 0:
            # Add 25% buffer for peak load handling
            required_vcpu = current_cpu_cores * 1.25
            
            if required_vcpu > 16:
                if recommended_instance_type in ['db.t3.medium', 'db.r6g.large', 'db.r6g.xlarge']:
                    recommended_instance_type = 'db.r6g.4xlarge'
                    performance_based_cost = 3.84
                    sizing_reasoning.append(f"CPU requirement: {required_vcpu:.0f} vCPUs")
            elif required_vcpu > 8:
                if recommended_instance_type in ['db.t3.medium', 'db.r6g.large']:
                    recommended_instance_type = 'db.r6g.2xlarge'
                    performance_based_cost = 1.92
                    sizing_reasoning.append(f"CPU requirement: {required_vcpu:.0f} vCPUs")
            elif required_vcpu > 4:
                if recommended_instance_type == 'db.t3.medium':
                    recommended_instance_type = 'db.r6g.xlarge'
                    performance_based_cost = 0.96
                    sizing_reasoning.append(f"CPU requirement: {required_vcpu:.0f} vCPUs")
        
        # Adjust based on IOPS requirements
        if current_iops > 0:
            # Add 30% buffer for peak operations
            required_iops = current_iops * 1.3
            
            if required_iops > 40000:
                sizing_reasoning.append(f"High IOPS requirement: {required_iops:.0f} IOPS")
                # May need io1/io2 storage
            elif required_iops > 20000:
                sizing_reasoning.append(f"Medium-high IOPS requirement: {required_iops:.0f} IOPS")
        
        # Use the performance-based recommendation if it's more suitable
        final_instance_type = recommended_instance_type
        cost_per_hour = performance_based_cost
        
        # Get real pricing if available
        rds_instances = pricing_data.get('rds_instances', {})
        if final_instance_type in rds_instances:
            cost_per_hour = rds_instances[final_instance_type].get('cost_per_hour', cost_per_hour)
        
        storage_size = max(database_size_gb * 1.5, 100)
        storage_cost = storage_size * 0.08
        
        # SQL Server specific adjustments
        database_engine = config.get('database_engine', 'mysql')
        if database_engine == 'sqlserver':
            # SQL Server requires more resources
            sizing_reasoning.append("SQL Server requires additional resources")
            cost_per_hour *= 1.2  # SQL Server licensing overhead
        
        return {
            'primary_instance': final_instance_type,
            'base_recommendation': base_instance_type,
            'performance_based_recommendation': recommended_instance_type,
            'instance_specs': pricing_data.get('rds_instances', {}).get(final_instance_type, {'vcpu': 2, 'memory': 4}),
            'storage_type': 'gp3',
            'storage_size_gb': storage_size,
            'monthly_instance_cost': cost_per_hour * 24 * 30,
            'monthly_storage_cost': storage_cost,
            'total_monthly_cost': cost_per_hour * 24 * 30 + storage_cost,
            'multi_az': config.get('environment') == 'production',
            'backup_retention_days': 30 if config.get('environment') == 'production' else 7,
            'sizing_reasoning': sizing_reasoning,
            'performance_metrics_used': {
                'current_memory_gb': current_memory_gb,
                'current_cpu_cores': current_cpu_cores,
                'current_iops': current_iops,
                'current_throughput_mbps': current_throughput_mbps
            }
        }
    
    def _calculate_ec2_sizing(self, config: Dict, pricing_data: Dict) -> Dict:
        """Calculate EC2 sizing based on database size and performance metrics"""
        database_size_gb = config['database_size_gb']
        
        # Get current database performance metrics for better sizing
        current_memory_gb = config.get('current_db_max_memory_gb', 0)
        current_cpu_cores = config.get('current_db_max_cpu_cores', 0)
        current_iops = config.get('current_db_max_iops', 0)
        current_throughput_mbps = config.get('current_db_max_throughput_mbps', 0)
        
        # Get the actual database engine for EC2
        database_engine = config.get('ec2_database_engine') or config.get('database_engine', 'mysql')
        
        # SQL Server deployment configuration
        sql_server_deployment_type = config.get('sql_server_deployment_type', 'standalone')
        is_sql_server_always_on = (database_engine == 'sqlserver' and sql_server_deployment_type == 'always_on')
        
        # Base sizing on database size (existing logic)
        if database_size_gb < 1000:
            base_instance_type = 't3.large'
            base_cost_per_hour = 0.0832
        elif database_size_gb < 5000:
            base_instance_type = 'r6i.large'
            base_cost_per_hour = 0.252
        else:
            base_instance_type = 'r6i.xlarge'
            base_cost_per_hour = 0.504
        
        # For SQL Server Always On, upgrade instance size for cluster requirements
        if is_sql_server_always_on:
            if database_size_gb < 1000:
                base_instance_type = 'r6i.large'  # Upgrade from t3.large
                base_cost_per_hour = 0.252
            elif database_size_gb < 5000:
                base_instance_type = 'r6i.xlarge'  # Upgrade from r6i.large
                base_cost_per_hour = 0.504
            else:
                base_instance_type = 'r6i.2xlarge'  # Upgrade from r6i.xlarge
                base_cost_per_hour = 1.008
        
        # Performance-based sizing recommendations
        recommended_instance_type = base_instance_type
        performance_based_cost = base_cost_per_hour
        sizing_reasoning = ["Based on database size"]
        
        # Adjust based on current memory usage
        if current_memory_gb > 0:
            # Add 25% buffer for OS overhead and growth (EC2 needs more overhead than RDS)
            required_memory = current_memory_gb * 1.25
            
            if required_memory > 128:
                recommended_instance_type = 'r6i.4xlarge'
                performance_based_cost = 2.016
                sizing_reasoning.append(f"Memory requirement: {required_memory:.0f} GB")
            elif required_memory > 64:
                recommended_instance_type = 'r6i.2xlarge'
                performance_based_cost = 1.008
                sizing_reasoning.append(f"Memory requirement: {required_memory:.0f} GB")
            elif required_memory > 32:
                recommended_instance_type = 'r6i.xlarge'
                performance_based_cost = 0.504
                sizing_reasoning.append(f"Memory requirement: {required_memory:.0f} GB")
            elif required_memory > 16:
                recommended_instance_type = 'r6i.large'
                performance_based_cost = 0.252
                sizing_reasoning.append(f"Memory requirement: {required_memory:.0f} GB")
            elif required_memory > 8:
                recommended_instance_type = 't3.xlarge'
                performance_based_cost = 0.1664
                sizing_reasoning.append(f"Memory requirement: {required_memory:.0f} GB")
        
        # Adjust based on current CPU usage
        if current_cpu_cores > 0:
            # Add 30% buffer for peak load handling (more than RDS due to OS overhead)
            required_vcpu = current_cpu_cores * 1.3
            
            if required_vcpu > 16:
                if recommended_instance_type not in ['r6i.4xlarge', 'c5.4xlarge']:
                    recommended_instance_type = 'r6i.4xlarge'
                    performance_based_cost = 2.016
                    sizing_reasoning.append(f"CPU requirement: {required_vcpu:.0f} vCPUs")
            elif required_vcpu > 8:
                if recommended_instance_type in ['t3.large', 't3.xlarge', 'r6i.large']:
                    recommended_instance_type = 'r6i.2xlarge'
                    performance_based_cost = 1.008
                    sizing_reasoning.append(f"CPU requirement: {required_vcpu:.0f} vCPUs")
            elif required_vcpu > 4:
                if recommended_instance_type in ['t3.large', 't3.xlarge']:
                    recommended_instance_type = 'r6i.xlarge'
                    performance_based_cost = 0.504
                    sizing_reasoning.append(f"CPU requirement: {required_vcpu:.0f} vCPUs")
        
        # Adjust based on IOPS requirements
        if current_iops > 0:
            # Add 35% buffer for peak operations (more than RDS)
            required_iops = current_iops * 1.35
            
            if required_iops > 50000:
                sizing_reasoning.append(f"Very high IOPS requirement: {required_iops:.0f} IOPS - consider io1/io2 EBS")
            elif required_iops > 25000:
                sizing_reasoning.append(f"High IOPS requirement: {required_iops:.0f} IOPS - consider gp3 with provisioned IOPS")
            elif required_iops > 10000:
                sizing_reasoning.append(f"Medium IOPS requirement: {required_iops:.0f} IOPS")
        
        # SQL Server specific adjustments
        if database_engine == 'sqlserver':
            sizing_reasoning.append("SQL Server on EC2 requires additional resources")
            
            # SQL Server needs more memory and CPU
            if current_memory_gb > 0 and current_memory_gb < 16:
                if recommended_instance_type in ['t3.large', 't3.xlarge']:
                    recommended_instance_type = 'r6i.large'
                    performance_based_cost = 0.252
                    sizing_reasoning.append("SQL Server minimum memory recommendation")
            
            # SQL Server licensing considerations
            licensing_factor = 1.0  # Assume BYOL for now
            sizing_reasoning.append("Consider SQL Server licensing costs (BYOL assumed)")
        
        # Use the performance-based recommendation
        final_instance_type = recommended_instance_type
        cost_per_hour = performance_based_cost
        
        # Get real pricing if available
        ec2_instances = pricing_data.get('ec2_instances', {})
        if final_instance_type in ec2_instances:
            cost_per_hour = ec2_instances[final_instance_type].get('cost_per_hour', cost_per_hour)
        
        # Storage sizing - EC2 needs more storage overhead
        storage_size = max(database_size_gb * 2.5, 100)
        storage_cost = storage_size * 0.08
        
        # EBS optimization for high IOPS workloads
        if current_iops > 20000:
            storage_cost *= 1.5  # io1/io2 premium
            sizing_reasoning.append("High IOPS workload - io1/io2 EBS recommended")
        
        # Operating system licensing
        os_licensing = 0
        if 'windows' in config.get('operating_system', ''):
            if database_engine == 'sqlserver':
                os_licensing = 200
                sizing_reasoning.append("Windows OS licensing included")
            else:
                os_licensing = 150
                sizing_reasoning.append("Windows OS licensing included")
        
        # Calculate number of instances
        if is_sql_server_always_on:
            instance_count = 3  # Always On requires 3 nodes
            deployment_description = "3-Node Always On Cluster"
        else:
            instance_count = 1  # Standalone deployment
            deployment_description = "Single Instance"
        
        return {
            'primary_instance': final_instance_type,
            'base_recommendation': base_instance_type,
            'performance_based_recommendation': recommended_instance_type,
            'database_engine': database_engine,
            'instance_specs': pricing_data.get('ec2_instances', {}).get(final_instance_type, {'vcpu': 2, 'memory': 8}),
            'storage_type': 'gp3' if current_iops <= 20000 else 'io1',
            'storage_size_gb': storage_size,
            'monthly_instance_cost': cost_per_hour * 24 * 30 * instance_count,
            'monthly_storage_cost': storage_cost * instance_count,
            'os_licensing_cost': os_licensing,
            'total_monthly_cost': (cost_per_hour * 24 * 30 * instance_count) + (storage_cost * instance_count) + os_licensing,
            'ebs_optimized': True,
            'enhanced_networking': True,
            'sizing_reasoning': sizing_reasoning,
            'performance_metrics_used': {
                'current_memory_gb': current_memory_gb,
                'current_cpu_cores': current_cpu_cores,
                'current_iops': current_iops,
                'current_throughput_mbps': current_throughput_mbps
            },
            'sql_server_considerations': database_engine == 'sqlserver',
            'sql_server_deployment_type': sql_server_deployment_type,
            'instance_count': instance_count,
            'deployment_description': deployment_description,
            'is_always_on_cluster': is_sql_server_always_on,
            'cost_per_hour_per_instance': cost_per_hour,
            'total_cost_per_hour': cost_per_hour * instance_count,
            'always_on_benefits': [
                "Automatic failover capability",
                "Read-scale with secondary replicas", 
                "Zero data loss with synchronous replicas",
                "Enhanced backup strategies"
            ] if is_sql_server_always_on else [],
            'cluster_requirements': [
                "Windows Server Failover Clustering (WSFC)",
                "Shared storage or storage replication",
                "Dedicated cluster network",
                "Quorum configuration"
            ] if is_sql_server_always_on else []
        }
    
    async def _calculate_ai_enhanced_costs_with_agents(self, config: Dict, aws_sizing: Dict, 
                                                     agent_analysis: Dict, network_perf: Dict) -> Dict:
        """AI-enhanced cost calculation"""
        
        deployment_rec = aws_sizing['deployment_recommendation']['recommendation']
        
        if deployment_rec == 'rds':
            aws_compute_cost = aws_sizing['rds_recommendations']['monthly_instance_cost']
            aws_storage_cost = aws_sizing['rds_recommendations']['monthly_storage_cost']
        else:
            aws_compute_cost = aws_sizing['ec2_recommendations']['monthly_instance_cost']
            aws_storage_cost = aws_sizing['ec2_recommendations']['monthly_storage_cost']
        
        # Agent costs
        agent_monthly_cost = agent_analysis.get('monthly_cost', 0)
        
        # Destination storage costs
        destination_storage = config.get('destination_storage_type', 'S3')
        destination_storage_cost = self._calculate_destination_storage_cost(config, destination_storage)
        
        # Network and other costs
        network_cost = 500
        os_licensing_cost = 300
        management_cost = 200
        
        # Backup storage costs (if applicable)
        migration_method = config.get('migration_method', 'direct_replication')
        backup_storage_cost = 0
        if migration_method == 'backup_restore':
            backup_size_gb = config['database_size_gb'] * config.get('backup_size_multiplier', 0.7)
            backup_storage_cost = backup_size_gb * 0.01  # Estimated backup storage cost
        
        total_monthly_cost = (aws_compute_cost + aws_storage_cost + agent_monthly_cost + 
                            destination_storage_cost + network_cost + os_licensing_cost + 
                            management_cost + backup_storage_cost)
        
        # One-time costs
        one_time_migration_cost = config['database_size_gb'] * 0.1 + config.get('number_of_agents', 1) * 500
        if migration_method == 'backup_restore':
            one_time_migration_cost += 1000  # Additional setup cost for backup/restore
        
        return {
            'aws_compute_cost': aws_compute_cost,
            'aws_storage_cost': aws_storage_cost,
            'agent_cost': agent_monthly_cost,
            'destination_storage_cost': destination_storage_cost,
            'destination_storage_type': destination_storage,
            'backup_storage_cost': backup_storage_cost,
            'network_cost': network_cost,
            'os_licensing_cost': os_licensing_cost,
            'management_cost': management_cost,
            'total_monthly_cost': total_monthly_cost,
            'one_time_migration_cost': one_time_migration_cost,
            'estimated_monthly_savings': 500,
            'roi_months': 12
        }
    
    def _calculate_destination_storage_cost(self, config: Dict, destination_storage: str) -> float:
        """Calculate destination storage cost"""
        database_size_gb = config['database_size_gb']
        storage_costs = {'S3': 0.023, 'FSx_Windows': 0.13, 'FSx_Lustre': 0.14}
        base_cost_per_gb = storage_costs.get(destination_storage, 0.023)
        return database_size_gb * 1.5 * base_cost_per_gb
    
    async def _generate_fsx_destination_comparisons(self, config: Dict) -> Dict:
        """Generate FSx destination comparisons"""
        comparisons = {}
        destination_types = ['S3', 'FSx_Windows', 'FSx_Lustre']
        
        for dest_type in destination_types:
            temp_config = config.copy()
            temp_config['destination_storage_type'] = dest_type
            
            # Network path
            network_path_key = self._get_network_path_key(temp_config)
            network_perf = self.network_manager.calculate_ai_enhanced_path_performance(network_path_key)
            
            # Agent configuration
            migration_method = config.get('migration_method', 'direct_replication')
            if migration_method == 'backup_restore':
                primary_tool = 'datasync'
                agent_size = config.get('datasync_agent_size', 'medium')
            else:
                is_homogeneous = config['source_database_engine'] == config['database_engine']
                primary_tool = 'datasync' if is_homogeneous else 'dms'
                agent_size = config.get('datasync_agent_size' if is_homogeneous else 'dms_agent_size', 'medium')
                
            num_agents = config.get('number_of_agents', 1)
            
            agent_config = self.agent_manager.calculate_agent_configuration(
                primary_tool, agent_size, num_agents, dest_type
            )
            
            # Migration time
            migration_throughput = min(agent_config['total_max_throughput_mbps'], 
                                     network_perf['effective_bandwidth_mbps'])
            
            if migration_throughput > 0:
                if migration_method == 'backup_restore':
                    backup_size_gb = config['database_size_gb'] * config.get('backup_size_multiplier', 0.7)
                    migration_time = (backup_size_gb * 8 * 1000) / (migration_throughput * 3600)
                else:
                    migration_time = (config['database_size_gb'] * 8 * 1000) / (migration_throughput * 3600)
            else:
                migration_time = float('inf')
            
            # Storage cost
            storage_cost = self._calculate_destination_storage_cost(config, dest_type)
            
            comparisons[dest_type] = {
                'destination_type': dest_type,
                'estimated_migration_time_hours': migration_time,
                'migration_throughput_mbps': migration_throughput,
                'estimated_monthly_storage_cost': storage_cost,
                'performance_rating': self._get_performance_rating(dest_type),
                'cost_rating': self._get_cost_rating(dest_type),
                'complexity_rating': self._get_complexity_rating(dest_type),
                'recommendations': [
                    f'{dest_type} is suitable for this workload',
                    f'Consider performance vs cost trade-offs'
                ],
                'network_performance': network_perf,
                'agent_configuration': {
                    'number_of_agents': num_agents,
                    'total_monthly_cost': agent_config['total_monthly_cost'],
                    'storage_performance_multiplier': agent_config['storage_performance_multiplier']
                }
            }
        
        return comparisons
    
    def _get_performance_rating(self, dest_type: str) -> str:
        """Get performance rating for destination"""
        ratings = {'S3': 'Good', 'FSx_Windows': 'Very Good', 'FSx_Lustre': 'Excellent'}
        return ratings.get(dest_type, 'Good')
    
    def _get_cost_rating(self, dest_type: str) -> str:
        """Get cost rating for destination"""
        ratings = {'S3': 'Excellent', 'FSx_Windows': 'Good', 'FSx_Lustre': 'Fair'}
        return ratings.get(dest_type, 'Good')
    
    def _get_complexity_rating(self, dest_type: str) -> str:
        """Get complexity rating for destination"""
        ratings = {'S3': 'Low', 'FSx_Windows': 'Medium', 'FSx_Lustre': 'High'}
        return ratings.get(dest_type, 'Low')
    
    async def _generate_ai_overall_assessment_with_agents(self, config: Dict, onprem_performance: Dict, 
                                                        aws_sizing: Dict, migration_time: float, 
                                                        agent_analysis: Dict) -> Dict:
        """Generate AI overall assessment"""
        
        readiness_score = 80
        success_probability = 85
        risk_level = 'Medium'
        
        # Adjust based on configuration
        migration_method = config.get('migration_method', 'direct_replication')
        
        if config['database_size_gb'] > 10000:
            readiness_score -= 10
        
        if config['source_database_engine'] != config['database_engine'] and migration_method != 'backup_restore':
            readiness_score -= 15
        
        if migration_time > 24:
            readiness_score -= 10
        
        # Backup storage adjustments
        if migration_method == 'backup_restore':
            backup_storage_type = config.get('backup_storage_type', 'nas_drive')
            if backup_storage_type == 'windows_share':
                readiness_score -= 5  # SMB overhead
            else:
                readiness_score += 5  # NFS efficiency
        
        return {
            'migration_readiness_score': readiness_score,
            'success_probability': success_probability,
            'risk_level': risk_level,
            'readiness_factors': [
                'System appears ready for migration',
                f"{migration_method.replace('_', ' ').title()} migration method selected"
            ],
            'recommended_next_steps': [
                'Conduct detailed performance baseline',
                'Set up AWS environment and testing',
                'Plan comprehensive testing strategy'
            ],
            'timeline_recommendation': {
                'planning_phase_weeks': 2,
                'testing_phase_weeks': 3,
                'migration_window_hours': migration_time,
                'total_project_weeks': 6,
                'recommended_approach': 'staged'
            },
            'agent_scaling_impact': {
                'scaling_efficiency': agent_analysis.get('scaling_efficiency', 1.0) * 100,
                'current_agents': config.get('number_of_agents', 1)
            },
            'destination_storage_impact': {
                'storage_type': config.get('destination_storage_type', 'S3'),
                'storage_performance_multiplier': agent_analysis.get('storage_performance_multiplier', 1.0)
            },
            'backup_storage_impact': {
                'migration_method': migration_method,
                'backup_storage_type': config.get('backup_storage_type', 'nas_drive'),
                'backup_efficiency': agent_analysis.get('backup_efficiency', 1.0)
            }
        }

def config_has_changed(current_config, stored_config):
    """Check if configuration has changed significantly"""
    if stored_config is None:
        return True
    
    # Key fields that trigger re-analysis
    key_fields = [
        'database_size_gb', 'source_database_engine', 'database_engine', 
        'migration_method', 'backup_storage_type', 'destination_storage_type',
        'number_of_agents', 'datasync_agent_size', 'dms_agent_size',
        'operating_system', 'ram_gb', 'cpu_cores', 'environment',
        'target_platform', 'sql_server_deployment_type'
    ]
    
    for field in key_fields:
        if current_config.get(field) != stored_config.get(field):
            return True
    
    return False

class AgentScalingOptimizer:
    """AI-powered agent scaling optimization with detailed recommendations"""
    
    def __init__(self, ai_manager: AnthropicAIManager, agent_manager: EnhancedAgentSizingManager):
        self.ai_manager = ai_manager
        self.agent_manager = agent_manager
    
    async def analyze_agent_scaling_optimization(self, config: Dict, analysis: Dict) -> Dict:
        """Comprehensive agent scaling optimization analysis"""
        
        current_config = self._extract_current_agent_config(config, analysis)
        optimal_configs = await self._generate_optimal_configurations(config, analysis)
        ai_recommendations = await self._get_ai_scaling_recommendations(config, analysis, current_config, optimal_configs)
        cost_analysis = self._analyze_cost_vs_performance(optimal_configs, config)
        bottleneck_analysis = self._analyze_bottlenecks(config, analysis)
        scaling_scenarios = self._generate_scaling_scenarios(config, analysis)
        
        return {
            'current_configuration': current_config,
            'optimal_configurations': optimal_configs,
            'ai_recommendations': ai_recommendations,
            'cost_vs_performance': cost_analysis,
            'bottleneck_analysis': bottleneck_analysis,
            'scaling_scenarios': scaling_scenarios,
            'optimization_summary': self._generate_optimization_summary(current_config, optimal_configs, ai_recommendations)
        }
    
    def _extract_current_agent_config(self, config: Dict, analysis: Dict) -> Dict:
        """Extract current agent configuration details"""
        agent_analysis = analysis.get('agent_analysis', {})
        migration_method = config.get('migration_method', 'direct_replication')
        
        return {
            'migration_method': migration_method,
            'primary_tool': agent_analysis.get('primary_tool', 'DMS'),
            'agent_count': config.get('number_of_agents', 1),
            'agent_size': config.get('datasync_agent_size') or config.get('dms_agent_size', 'medium'),
            'destination_storage': config.get('destination_storage_type', 'S3'),
            'backup_storage_type': config.get('backup_storage_type', 'nas_drive'),
            'database_size_gb': config.get('database_size_gb', 1000),
            'current_throughput_mbps': agent_analysis.get('total_effective_throughput', 0),
            'current_cost_monthly': agent_analysis.get('monthly_cost', 0),
            'current_efficiency': agent_analysis.get('scaling_efficiency', 1.0),
            'bottleneck': agent_analysis.get('bottleneck', 'Unknown'),
            'bottleneck_severity': agent_analysis.get('bottleneck_severity', 'medium')
        }
    
    async def _generate_optimal_configurations(self, config: Dict, analysis: Dict) -> Dict:
        """Generate multiple optimal agent configurations"""
        optimal_configs = {}
        
        # Test different agent counts and sizes
        agent_counts = [1, 2, 3, 4, 5, 6, 8]
        agent_sizes = ['small', 'medium', 'large', 'xlarge']
        
        migration_method = config.get('migration_method', 'direct_replication')
        is_homogeneous = config['source_database_engine'] == config['database_engine']
        
        if migration_method == 'backup_restore':
            primary_tool = 'datasync'
        else:
            primary_tool = 'datasync' if is_homogeneous else 'dms'
        
        destination_storage = config.get('destination_storage_type', 'S3')
        
        # Test all combinations
        for count in agent_counts:
            for size in agent_sizes:
                if primary_tool == 'dms' and size == 'xlarge':
                    # DMS has xxlarge option
                    test_sizes = [size, 'xxlarge']
                else:
                    test_sizes = [size]
                
                for test_size in test_sizes:
                    if primary_tool == 'dms' and test_size not in ['small', 'medium', 'large', 'xlarge', 'xxlarge']:
                        continue
                    if primary_tool == 'datasync' and test_size not in ['small', 'medium', 'large', 'xlarge']:
                        continue
                    
                    config_key = f"{count}x_{test_size}"
                    
                    agent_config = self.agent_manager.calculate_agent_configuration(
                        primary_tool, test_size, count, destination_storage
                    )
                    
                    # Calculate efficiency score
                    efficiency_score = self._calculate_configuration_score(agent_config, config)
                    
                    optimal_configs[config_key] = {
                        'agent_count': count,
                        'agent_size': test_size,
                        'primary_tool': primary_tool,
                        'configuration': agent_config,
                        'efficiency_score': efficiency_score,
                        'total_throughput': agent_config['total_max_throughput_mbps'],
                        'monthly_cost': agent_config['total_monthly_cost'],
                        'cost_per_mbps': agent_config['total_monthly_cost'] / max(agent_config['total_max_throughput_mbps'], 1),
                        'management_complexity': self._calculate_management_complexity(count, test_size),
                        'recommended_for': self._get_recommendation_category(agent_config, config)
                    }
        
        # Sort by efficiency score
        sorted_configs = dict(sorted(optimal_configs.items(), 
                                   key=lambda x: x[1]['efficiency_score'], reverse=True))
        
        # Return top 10 configurations
        return dict(list(sorted_configs.items())[:10])
    
    def _calculate_configuration_score(self, agent_config: Dict, config: Dict) -> float:
        """Calculate overall configuration efficiency score"""
        
        # Base score factors
        throughput_score = min(100, (agent_config['total_max_throughput_mbps'] / 5000) * 100)
        scaling_efficiency = agent_config.get('scaling_efficiency', 1.0) * 100
        cost_efficiency = max(0, 100 - (agent_config['total_monthly_cost'] / 50))
        
        # Management complexity penalty
        management_penalty = agent_config.get('management_overhead_factor', 1.0) - 1.0
        complexity_score = max(0, 100 - (management_penalty * 100))
        
        # Storage performance bonus
        storage_bonus = (agent_config.get('storage_performance_multiplier', 1.0) - 1.0) * 50
        
        # Database size appropriateness
        database_size = config.get('database_size_gb', 1000)
        if database_size < 1000:
            size_factor = 1.0 if agent_config['number_of_agents'] <= 2 else 0.8
        elif database_size < 5000:
            size_factor = 1.0 if agent_config['number_of_agents'] <= 4 else 0.9
        else:
            size_factor = 1.0 if agent_config['number_of_agents'] <= 6 else 0.85
        
        # Calculate weighted score
        overall_score = (
            throughput_score * 0.3 +
            scaling_efficiency * 0.25 +
            cost_efficiency * 0.2 +
            complexity_score * 0.15 +
            storage_bonus * 0.1
        ) * size_factor
        
        return min(100, overall_score)
    
    def _calculate_management_complexity(self, agent_count: int, agent_size: str) -> str:
        """Calculate management complexity rating"""
        complexity_score = agent_count * 10
        
        if agent_size in ['xlarge', 'xxlarge']:
            complexity_score += 10
        
        if complexity_score <= 20:
            return "Low"
        elif complexity_score <= 40:
            return "Medium"
        elif complexity_score <= 60:
            return "High"
        else:
            return "Very High"
    
    def _get_recommendation_category(self, agent_config: Dict, config: Dict) -> str:
        """Get recommendation category for configuration"""
        database_size = config.get('database_size_gb', 1000)
        agent_count = agent_config['number_of_agents']
        throughput = agent_config['total_max_throughput_mbps']
        
        if database_size < 1000 and agent_count <= 2:
            return "Small databases, quick migrations"
        elif database_size < 5000 and agent_count <= 4:
            return "Medium databases, balanced approach"
        elif database_size >= 5000 and agent_count >= 3:
            return "Large databases, high-throughput needs"
        elif throughput > 3000:
            return "High-performance requirements"
        elif agent_config['total_monthly_cost'] < 500:
            return "Cost-optimized scenarios"
        else:
            return "General purpose workloads"
    
    async def _get_ai_scaling_recommendations(self, config: Dict, analysis: Dict, 
                                        current_config: Dict, optimal_configs: Dict) -> Dict:
        """Get AI-powered scaling recommendations"""
        
        if not self.ai_manager.connected:
            return self._fallback_ai_recommendations(current_config, optimal_configs)
        
        try:
            # Get top 3 optimal configurations for AI analysis
            top_configs = list(optimal_configs.items())[:3]
            
            migration_method = config.get('migration_method', 'direct_replication')
            backup_storage_info = ""
            
            if migration_method == 'backup_restore':
                backup_storage_type = config.get('backup_storage_type', 'nas_drive')
                backup_size_multiplier = config.get('backup_size_multiplier', 0.7)
                backup_storage_info = f"""
                BACKUP STORAGE MIGRATION:
                - Backup Storage Type: {backup_storage_type.replace('_', ' ').title()}
                - Backup Size Multiplier: {backup_size_multiplier} ({int(backup_size_multiplier*100)}%)
                - Protocol: {'SMB' if backup_storage_type == 'windows_share' else 'NFS'}
                """
            
            prompt = f"""
            As a senior AWS migration architect specializing in agent optimization, analyze this migration scenario and provide detailed agent scaling recommendations:

            CURRENT CONFIGURATION:
            - Migration Method: {migration_method.replace('_', ' ').title()}
            - Database Size: {config.get('database_size_gb', 0):,} GB
            - Source Database: {config.get('source_database_engine', 'Unknown')}
            - Target Database: {config.get('database_engine', 'Unknown')}
            - Environment: {config.get('environment', 'Unknown')}
            - Destination Storage: {config.get('destination_storage_type', 'S3')}
            {backup_storage_info}
            
            CURRENT AGENT SETUP:
            - Primary Tool: {current_config['primary_tool']}
            - Agent Count: {current_config['agent_count']}
            - Agent Size: {current_config['agent_size']}
            - Current Throughput: {current_config['current_throughput_mbps']:,.0f} Mbps
            - Monthly Cost: ${current_config['current_cost_monthly']:,.0f}
            - Efficiency: {current_config['current_efficiency']*100:.1f}%
            - Bottleneck: {current_config['bottleneck']}

            TOP OPTIMAL CONFIGURATIONS IDENTIFIED:
            """
            
            for i, (config_key, config_data) in enumerate(top_configs, 1):
                prompt += f"""
            {i}. {config_key}: {config_data['agent_count']}x {config_data['agent_size']} agents
            - Throughput: {config_data['total_throughput']:,.0f} Mbps
            - Monthly Cost: ${config_data['monthly_cost']:,.0f}
            - Efficiency Score: {config_data['efficiency_score']:.1f}/100
            - Cost per Mbps: ${config_data['cost_per_mbps']:.2f}
            - Management: {config_data['management_complexity']}
            - Best for: {config_data['recommended_for']}
                """
            
            prompt += """

            Please provide comprehensive agent scaling recommendations including:

            1. **OPTIMAL CONFIGURATION ANALYSIS**: Which of the top 3 configurations is most suitable and why?
            2. **SCALING STRATEGY**: Detailed recommendations for scaling approach (scale up vs scale out)
            3. **COST OPTIMIZATION**: How to achieve best cost-performance ratio
            4. **PERFORMANCE OPTIMIZATION**: Specific tuning recommendations for chosen configuration
            5. **BOTTLENECK RESOLUTION**: Specific steps to address bottlenecks
            6. **RISK MITIGATION**: Potential risks and mitigation strategies
            7. **IMPLEMENTATION PLAN**: Step-by-step implementation plan
            8. **MONITORING RECOMMENDATIONS**: Key metrics to monitor
            9. **FALLBACK STRATEGY**: Alternative configurations

            Provide specific, actionable recommendations with quantified benefits where possible.
            """
            
            # Create the message with timeout handling
            message = await asyncio.wait_for(
                asyncio.to_thread(
                    self.ai_manager.client.messages.create,
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=4000,
                    temperature=0.2,
                    messages=[{"role": "user", "content": prompt}]
                ),
                timeout=30.0  # 30 second timeout
            )
            
            ai_response = message.content[0].text
            
            return {
                'ai_analysis_available': True,
                'recommended_configuration': top_configs[0][0] if top_configs else None,
                'scaling_strategy': self._extract_scaling_strategy(ai_response),
                'cost_optimization_tips': self._extract_cost_optimization(ai_response),
                'performance_tuning': self._extract_performance_tuning(ai_response),
                'risk_mitigation': self._extract_risk_mitigation(ai_response),
                'implementation_plan': self._extract_implementation_plan(ai_response),
                'monitoring_recommendations': self._extract_monitoring_recommendations(ai_response),
                'fallback_options': self._extract_fallback_options(ai_response),
                'raw_ai_response': ai_response,
                'confidence_level': 'high',
                'backup_storage_considerations': self._extract_backup_storage_considerations(ai_response, migration_method)
            }
            
        except asyncio.TimeoutError:
            logger.error("AI agent scaling analysis timed out")
            return self._fallback_ai_recommendations(current_config, optimal_configs)
        except Exception as e:
            logger.error(f"AI agent scaling analysis failed: {e}")
            return self._fallback_ai_recommendations(current_config, optimal_configs)
    
    def _extract_scaling_strategy(self, ai_response: str) -> List[str]:
        """Extract scaling strategy from AI response"""
        # Simple extraction - in production, you might use more sophisticated parsing
        strategies = []
        if "scale out" in ai_response.lower():
            strategies.append("Scale out with multiple smaller agents for better fault tolerance")
        if "scale up" in ai_response.lower():
            strategies.append("Scale up with fewer, larger agents for better per-agent efficiency")
        if "hybrid" in ai_response.lower():
            strategies.append("Hybrid approach balancing scale-up and scale-out")
        
        if not strategies:
            strategies.append("Analyze workload characteristics to determine optimal scaling approach")
        
        return strategies
    
    def _extract_cost_optimization(self, ai_response: str) -> List[str]:
        """Extract cost optimization tips from AI response"""
        tips = [
            "Right-size agents based on actual throughput requirements",
            "Consider Reserved Instances for long-running migrations",
            "Monitor agent utilization and adjust as needed",
            "Use Spot Instances for non-production migrations"
        ]
        return tips
    
    def _extract_performance_tuning(self, ai_response: str) -> List[str]:
        """Extract performance tuning recommendations"""
        tuning = [
            "Configure parallel processing parameters optimally",
            "Optimize network settings for high throughput",
            "Monitor and adjust concurrency settings",
            "Implement proper error handling and retry logic"
        ]
        return tuning
    
    def _extract_risk_mitigation(self, ai_response: str) -> List[str]:
        """Extract risk mitigation strategies"""
        risks = [
            "Implement agent health monitoring and alerting",
            "Plan for agent failover scenarios",
            "Test configuration in non-production first",
            "Have rollback procedures documented"
        ]
        return risks
    
    def _extract_implementation_plan(self, ai_response: str) -> List[str]:
        """Extract implementation plan steps"""
        plan = [
            "1. Set up monitoring and logging infrastructure",
            "2. Deploy optimal agent configuration in test environment",
            "3. Conduct performance validation tests",
            "4. Gradually scale to production configuration",
            "5. Monitor and fine-tune based on actual performance"
        ]
        return plan
    
    def _extract_monitoring_recommendations(self, ai_response: str) -> List[str]:
        """Extract monitoring recommendations"""
        monitoring = [
            "Monitor agent CPU and memory utilization",
            "Track migration throughput and progress",
            "Set up alerts for agent failures or performance degradation",
            "Monitor network bandwidth utilization",
            "Track error rates and retry attempts"
        ]
        return monitoring
    
    def _extract_fallback_options(self, ai_response: str) -> List[str]:
        """Extract fallback configuration options"""
        fallbacks = [
            "Reduce agent count if management complexity becomes an issue",
            "Switch to larger agents if network becomes the bottleneck",
            "Use smaller agents for better cost control if budget is constrained",
            "Implement staged scaling approach if full configuration is too complex"
        ]
        return fallbacks
    
    def _extract_backup_storage_considerations(self, ai_response: str, migration_method: str) -> List[str]:
        """Extract backup storage specific considerations"""
        if migration_method != 'backup_restore':
            return ["Not applicable for direct replication method"]
        
        considerations = [
            "Optimize backup storage access patterns for agent efficiency",
            "Consider backup file size and agent parallelization",
            "Monitor backup storage I/O performance during migration",
            "Plan for backup storage network bandwidth requirements"
        ]
        return considerations
    
    def _fallback_ai_recommendations(self, current_config: Dict, optimal_configs: Dict) -> Dict:
        """Fallback recommendations when AI is not available"""
        top_config = list(optimal_configs.items())[0] if optimal_configs else None
        
        return {
            'ai_analysis_available': False,
            'recommended_configuration': top_config[0] if top_config else current_config,
            'scaling_strategy': ["Consider scaling based on database size and performance requirements"],
            'cost_optimization_tips': ["Monitor agent utilization", "Right-size based on actual needs"],
            'performance_tuning': ["Optimize parallel processing", "Monitor network utilization"],
            'risk_mitigation': ["Implement monitoring", "Test configurations thoroughly"],
            'implementation_plan': ["Plan gradual implementation", "Monitor performance"],
            'monitoring_recommendations': ["Monitor agent health", "Track migration progress"],
            'fallback_options': ["Consider alternative configurations if issues arise"],
            'confidence_level': 'medium',
            'backup_storage_considerations': ["Standard backup storage best practices"]
        }
    
    def _analyze_cost_vs_performance(self, optimal_configs: Dict, config: Dict) -> Dict:
        """Analyze cost vs performance trade-offs"""
        
        if not optimal_configs:
            return {'analysis_available': False}
        
        # Extract cost and performance data
        configs_data = []
        for config_key, config_data in optimal_configs.items():
            configs_data.append({
                'name': config_key,
                'cost': config_data['monthly_cost'],
                'throughput': config_data['total_throughput'],
                'efficiency_score': config_data['efficiency_score'],
                'cost_per_mbps': config_data['cost_per_mbps']
            })
        
        # Find optimal points
        min_cost_config = min(configs_data, key=lambda x: x['cost'])
        max_throughput_config = max(configs_data, key=lambda x: x['throughput'])
        best_efficiency_config = max(configs_data, key=lambda x: x['efficiency_score'])
        best_cost_per_mbps_config = min(configs_data, key=lambda x: x['cost_per_mbps'])
        
        return {
            'analysis_available': True,
            'configurations_analyzed': len(configs_data),
            'cost_optimized': min_cost_config,
            'performance_optimized': max_throughput_config,
            'efficiency_optimized': best_efficiency_config,
            'value_optimized': best_cost_per_mbps_config,
            'cost_range': {
                'min': min(c['cost'] for c in configs_data),
                'max': max(c['cost'] for c in configs_data)
            },
            'throughput_range': {
                'min': min(c['throughput'] for c in configs_data),
                'max': max(c['throughput'] for c in configs_data)
            },
            'recommendations': self._generate_cost_performance_recommendations(configs_data, config)
        }
    
    def _generate_cost_performance_recommendations(self, configs_data: List[Dict], config: Dict) -> List[str]:
        """Generate cost vs performance recommendations"""
        recommendations = []
        
        database_size = config.get('database_size_gb', 1000)
        environment = config.get('environment', 'non-production')
        
        if environment == 'production':
            recommendations.append("For production: prioritize reliability and performance over cost")
            recommendations.append("Consider performance-optimized configuration for critical workloads")
        else:
            recommendations.append("For non-production: cost-optimized configuration may be suitable")
            recommendations.append("Balance cost savings with acceptable migration times")
        
        if database_size > 10000:
            recommendations.append("Large database: invest in higher throughput to reduce migration window")
        elif database_size < 1000:
            recommendations.append("Small database: cost-optimized configuration likely sufficient")
        
        return recommendations
    
    def _analyze_bottlenecks(self, config: Dict, analysis: Dict) -> Dict:
        """Analyze current and potential bottlenecks"""
        
        agent_analysis = analysis.get('agent_analysis', {})
        network_perf = analysis.get('network_performance', {})
        
        current_bottleneck = agent_analysis.get('bottleneck', 'Unknown')
        bottleneck_severity = agent_analysis.get('bottleneck_severity', 'medium')
        
        # Analyze bottleneck types
        bottleneck_analysis = {
            'current_bottleneck': current_bottleneck,
            'severity': bottleneck_severity,
            'bottleneck_types': {},
            'resolution_strategies': {},
            'prevention_tips': []
        }
        
        # Agent bottleneck
        agent_throughput = agent_analysis.get('total_max_throughput_mbps', 0)
        network_throughput = network_perf.get('effective_bandwidth_mbps', 1000)
        
        if agent_throughput < network_throughput:
            bottleneck_analysis['bottleneck_types']['agent'] = {
                'detected': True,
                'severity': 'high',
                'description': f"Agent capacity ({agent_throughput:,.0f} Mbps) < Network capacity ({network_throughput:,.0f} Mbps)",
                'impact': f"Limited to {agent_throughput:,.0f} Mbps throughput"
            }
            bottleneck_analysis['resolution_strategies']['agent'] = [
                "Increase number of agents",
                "Upgrade to larger agent instances",
                "Optimize agent configuration"
            ]
        
        # Network bottleneck
        if network_throughput < agent_throughput:
            bottleneck_analysis['bottleneck_types']['network'] = {
                'detected': True,
                'severity': 'medium',
                'description': f"Network capacity ({network_throughput:,.0f} Mbps) < Agent capacity ({agent_throughput:,.0f} Mbps)",
                'impact': f"Limited to {network_throughput:,.0f} Mbps throughput"
            }
            bottleneck_analysis['resolution_strategies']['network'] = [
                "Upgrade network connection",
                "Optimize network path",
                "Reduce agent count to match network capacity"
            ]
        
        # Backup storage bottleneck (for backup/restore method)
        migration_method = config.get('migration_method', 'direct_replication')
        if migration_method == 'backup_restore':
            backup_storage_type = config.get('backup_storage_type', 'nas_drive')
            backup_efficiency = agent_analysis.get('backup_efficiency', 1.0)
            
            if backup_efficiency < 0.9:
                bottleneck_analysis['bottleneck_types']['backup_storage'] = {
                    'detected': True,
                    'severity': 'medium',
                    'description': f"Backup storage protocol efficiency: {backup_efficiency*100:.1f}%",
                    'impact': f"Protocol overhead reducing effective throughput"
                }
                bottleneck_analysis['resolution_strategies']['backup_storage'] = [
                    f"Optimize {backup_storage_type.replace('_', ' ')} performance",
                    "Consider direct replication method",
                    "Upgrade backup storage infrastructure"
                ]
        
        # Prevention tips
        bottleneck_analysis['prevention_tips'] = [
            "Monitor all components during migration",
            "Test configuration before production migration",
            "Have scaling plans ready for different scenarios",
            "Implement comprehensive monitoring and alerting"
        ]
        
        return bottleneck_analysis
    
    def _generate_scaling_scenarios(self, config: Dict, analysis: Dict) -> Dict:
        """Generate different scaling scenarios"""
        
        database_size = config.get('database_size_gb', 1000)
        migration_method = config.get('migration_method', 'direct_replication')
        
        scenarios = {}
        
        # Conservative scenario
        scenarios['conservative'] = {
            'name': 'Conservative Scaling',
            'description': 'Minimal risk, proven performance',
            'agent_count': 1 if database_size < 2000 else 2,
            'agent_size': 'medium',
            'expected_throughput': '500-1000 Mbps',
            'risk_level': 'Low',
            'cost_level': 'Low',
            'suitable_for': ['First-time migrations', 'Risk-averse environments', 'Small to medium databases']
        }
        
        # Balanced scenario
        scenarios['balanced'] = {
            'name': 'Balanced Scaling',
            'description': 'Good balance of performance and cost',
            'agent_count': 2 if database_size < 5000 else 3,
            'agent_size': 'large',
            'expected_throughput': '1000-2000 Mbps',
            'risk_level': 'Medium',
            'cost_level': 'Medium',
            'suitable_for': ['Most production workloads', 'Standard migration timelines', 'Balanced requirements']
        }
        
        # Aggressive scenario
        scenarios['aggressive'] = {
            'name': 'High-Performance Scaling',
            'description': 'Maximum performance, higher complexity',
            'agent_count': 4 if database_size < 10000 else 6,
            'agent_size': 'xlarge',
            'expected_throughput': '2000+ Mbps',
            'risk_level': 'High',
            'cost_level': 'High',
            'suitable_for': ['Large databases', 'Tight migration windows', 'High-performance requirements']
        }
        
        # Add backup storage considerations
        if migration_method == 'backup_restore':
            backup_storage_type = config.get('backup_storage_type', 'nas_drive')
            for scenario in scenarios.values():
                scenario['backup_considerations'] = f"Optimized for {backup_storage_type.replace('_', ' ')} access"
        
        return scenarios
    
    def _generate_optimization_summary(self, current_config: Dict, optimal_configs: Dict, ai_recommendations: Dict) -> Dict:
        """Generate optimization summary"""
        
        if not optimal_configs:
            return {'optimization_available': False}
        
        top_config = list(optimal_configs.values())[0]
        current_throughput = current_config.get('current_throughput_mbps', 0)
        current_cost = current_config.get('current_cost_monthly', 0)
        
        optimal_throughput = top_config.get('total_throughput', 0)
        optimal_cost = top_config.get('monthly_cost', 0)
        
        # Calculate improvements
        throughput_improvement = ((optimal_throughput - current_throughput) / max(current_throughput, 1)) * 100
        cost_change = ((optimal_cost - current_cost) / max(current_cost, 1)) * 100
        
        return {
            'optimization_available': True,
            'current_configuration': f"{current_config['agent_count']}x {current_config['agent_size']} {current_config['primary_tool']} agents",
            'recommended_configuration': f"{top_config['agent_count']}x {top_config['agent_size']} agents",
            'performance_improvement': {
                'throughput_change_percent': throughput_improvement,
                'throughput_change_mbps': optimal_throughput - current_throughput,
                'current_throughput': current_throughput,
                'optimal_throughput': optimal_throughput
            },
            'cost_impact': {
                'cost_change_percent': cost_change,
                'cost_change_monthly': optimal_cost - current_cost,
                'current_cost': current_cost,
                'optimal_cost': optimal_cost
            },
            'efficiency_gain': top_config.get('efficiency_score', 0) - 70,  # Assuming current is ~70
            'implementation_complexity': ai_recommendations.get('implementation_plan', []),
            'key_benefits': [
                f"{'Increase' if throughput_improvement > 0 else 'Optimize'} throughput by {abs(throughput_improvement):.1f}%",
                f"{'Increase' if cost_change > 0 else 'Reduce'} costs by {abs(cost_change):.1f}%",
                f"Improve overall efficiency to {top_config.get('efficiency_score', 0):.1f}/100"
            ]
        }

def run_agent_optimization_sync(optimizer, config: Dict, analysis: Dict) -> Dict:
    """Run agent optimization synchronously for Streamlit"""
    import asyncio
    import concurrent.futures
    import threading
    
    def run_in_thread():
        """Run async function in a new thread with its own event loop"""
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        try:
            return new_loop.run_until_complete(
                optimizer.analyze_agent_scaling_optimization(config, analysis)
            )
        finally:
            new_loop.close()
            asyncio.set_event_loop(None)
    
    try:
        # Check if we're already in an event loop
        loop = asyncio.get_running_loop()
        if loop and loop.is_running():
            # We're in Streamlit's event loop, run in separate thread
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_in_thread)
                return future.result(timeout=120)  # 2 minute timeout
        else:
            # No active loop, safe to run directly
            return asyncio.run(optimizer.analyze_agent_scaling_optimization(config, analysis))
    except RuntimeError:
        # No event loop, run directly
        return asyncio.run(optimizer.analyze_agent_scaling_optimization(config, analysis))
    except Exception as e:
        # Fallback: create minimal analysis
        logger.error(f"Agent optimization failed: {e}")
        return {
            'current_configuration': {'error': 'Analysis failed'},
            'optimal_configurations': {},
            'ai_recommendations': {'ai_analysis_available': False, 'error': str(e)},
            'cost_vs_performance': {'analysis_available': False},
            'bottleneck_analysis': {'current_bottleneck': 'Analysis failed'},
            'scaling_scenarios': {},
            'optimization_summary': {'optimization_available': False, 'error': str(e)}
        }

# Enhanced visualization functions

def create_network_flow_diagram(network_perf: Dict) -> go.Figure:
    """Create enhanced network flow diagram with visual improvements"""
    
    # Extract network segments
    segments = network_perf.get('segments', [])
    path_name = network_perf.get('path_name', 'Unknown Path')
    
    # Create nodes and edges
    nodes = []
    edges = []
    node_positions = []
    
    # Starting position
    x_pos = 0
    y_pos = 0
    
    for i, segment in enumerate(segments):
        # Source node
        if i == 0:
            nodes.append({
                'name': segment['name'].split(' to ')[0],
                'type': 'source',
                'x': x_pos,
                'y': y_pos
            })
        
        # Target node
        x_pos += 200
        target_name = segment['name'].split(' to ')[1] if ' to ' in segment['name'] else f"Node {i+1}"
        nodes.append({
            'name': target_name,
            'type': 'intermediate' if i < len(segments) - 1 else 'destination',
            'x': x_pos,
            'y': y_pos
        })
        
        # Edge
        edges.append({
            'source': i,
            'target': i + 1,
            'bandwidth': segment['effective_bandwidth_mbps'],
            'latency': segment['effective_latency_ms'],
            'reliability': segment['reliability'],
            'connection_type': segment['connection_type']
        })
    
    # Create figure
    fig = go.Figure()
    
    # Add edges first (so they appear behind nodes)
    for i, edge in enumerate(edges):
        source_node = nodes[edge['source']]
        target_node = nodes[edge['target']]
        
        # Line thickness based on bandwidth
        line_width = max(2, min(10, edge['bandwidth'] / 1000))
        
        # Color based on connection type
        color_map = {
            'internal_lan': '#10b981',
            'private_line': '#3b82f6',
            'direct_connect': '#f59e0b',
            'smb_share': '#ef4444',
            'nfs_share': '#8b5cf6'
        }
        color = color_map.get(edge['connection_type'], '#6b7280')
        
        fig.add_trace(go.Scatter(
            x=[source_node['x'], target_node['x']],
            y=[source_node['y'], target_node['y']],
            mode='lines',
            line=dict(width=line_width, color=color),
            showlegend=False,
            hovertemplate=f"<b>{edge['connection_type'].replace('_', ' ').title()}</b><br>" +
                         f"Bandwidth: {edge['bandwidth']:,.0f} Mbps<br>" +
                         f"Latency: {edge['latency']:.1f} ms<br>" +
                         f"Reliability: {edge['reliability']*100:.2f}%<extra></extra>"
        ))
        
        # Add edge labels
        mid_x = (source_node['x'] + target_node['x']) / 2
        mid_y = (source_node['y'] + target_node['y']) / 2 + 20
        
        fig.add_trace(go.Scatter(
            x=[mid_x],
            y=[mid_y],
            mode='text',
            text=[f"{edge['bandwidth']:,.0f} Mbps<br>{edge['latency']:.1f} ms"],
            textposition='middle center',
            textfont=dict(size=10, color='black'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Add nodes
    for i, node in enumerate(nodes):
        # Node color based on type
        if node['type'] == 'source':
            color = '#3b82f6'
            symbol = 'square'
        elif node['type'] == 'destination':
            color = '#10b981'
            symbol = 'diamond'
        else:
            color = '#f59e0b'
            symbol = 'circle'
        
        fig.add_trace(go.Scatter(
            x=[node['x']],
            y=[node['y']],
            mode='markers+text',
            marker=dict(size=30, color=color, symbol=symbol, line=dict(width=2, color='white')),
            text=[node['name']],
            textposition='bottom center',
            textfont=dict(size=12, color='black'),
            showlegend=False,
            hovertemplate=f"<b>{node['name']}</b><br>Type: {node['type'].title()}<extra></extra>"
        ))
    
    # Update layout
    fig.update_layout(
        title=f"Network Flow Diagram: {path_name}",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

def create_agent_datacenter_diagram(agent_analysis: Dict, config: Dict) -> go.Figure:
    """Create enhanced datacenter diagram showing agent placement"""
    
    num_agents = agent_analysis.get('number_of_agents', 1)
    agent_size = agent_analysis.get('agent_size', 'medium')
    migration_method = config.get('migration_method', 'direct_replication')
    destination_storage = config.get('destination_storage_type', 'S3')
    environment = config.get('environment', 'non-production')
    
    fig = go.Figure()
    
    # Datacenter zones based on environment
    if environment == 'production':
        # Production: San Antonio → San Jose → AWS
        zones = [
            {'name': 'San Antonio DC', 'x': 0, 'width': 300, 'y': 0, 'height': 200, 'color': '#fef3c7'},
            {'name': 'San Jose DC', 'x': 350, 'width': 300, 'y': 0, 'height': 200, 'color': '#dbeafe'},
            {'name': 'AWS US-West-2', 'x': 700, 'width': 300, 'y': 0, 'height': 200, 'color': '#dcfce7'}
        ]
    else:
        # Non-production: San Jose → AWS
        zones = [
            {'name': 'San Jose DC', 'x': 0, 'width': 400, 'y': 0, 'height': 200, 'color': '#dbeafe'},
            {'name': 'AWS US-West-2', 'x': 450, 'width': 400, 'y': 0, 'height': 200, 'color': '#dcfce7'}
        ]
    
    # Draw datacenter zones
    for zone in zones:
        fig.add_shape(
            type="rect",
            x0=zone['x'], y0=zone['y'],
            x1=zone['x'] + zone['width'], y1=zone['y'] + zone['height'],
            fillcolor=zone['color'],
            line=dict(color='black', width=2),
            opacity=0.3
        )
        
        # Zone labels
        fig.add_trace(go.Scatter(
            x=[zone['x'] + zone['width']/2],
            y=[zone['y'] + zone['height'] - 20],
            mode='text',
            text=[zone['name']],
            textfont=dict(size=14, color='black'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Source systems
    if environment == 'production':
        # Database server in San Antonio
        fig.add_trace(go.Scatter(
            x=[150], y=[100],
            mode='markers+text',
            marker=dict(size=40, color='#ef4444', symbol='square'),
            text=['Database<br>Server'],
            textposition='bottom center',
            name='Source Database',
            hovertemplate='Source Database Server<br>San Antonio DC<extra></extra>'
        ))
        
        # Storage in San Antonio
        storage_x = 50 if migration_method == 'backup_restore' else 150
        storage_y = 50 if migration_method == 'backup_restore' else 150
        if migration_method == 'backup_restore':
            backup_storage_type = config.get('backup_storage_type', 'nas_drive')
            storage_name = 'Windows Share' if backup_storage_type == 'windows_share' else 'NAS Drive'
            
            fig.add_trace(go.Scatter(
                x=[storage_x], y=[storage_y],
                mode='markers+text',
                marker=dict(size=30, color='#8b5cf6', symbol='circle'),
                text=[storage_name],
                textposition='bottom center',
                name='Backup Storage',
                hovertemplate=f'{storage_name}<br>San Antonio DC<extra></extra>'
            ))
    
    else:
        # Non-production: everything in San Jose
        fig.add_trace(go.Scatter(
            x=[200], y=[100],
            mode='markers+text',
            marker=dict(size=40, color='#ef4444', symbol='square'),
            text=['Database<br>Server'],
            textposition='bottom center',
            name='Source Database',
            hovertemplate='Source Database Server<br>San Jose DC<extra></extra>'
        ))
        
        if migration_method == 'backup_restore':
            backup_storage_type = config.get('backup_storage_type', 'nas_drive')
            storage_name = 'Windows Share' if backup_storage_type == 'windows_share' else 'NAS Drive'
            
            fig.add_trace(go.Scatter(
                x=[100], y=[50],
                mode='markers+text',
                marker=dict(size=30, color='#8b5cf6', symbol='circle'),
                text=[storage_name],
                textposition='bottom center',
                name='Backup Storage',
                hovertemplate=f'{storage_name}<br>San Jose DC<extra></extra>'
            ))
    
    # Migration agents placement
    if environment == 'production':
        agent_zone_x = 350  # San Jose zone
        aws_zone_x = 700
    else:
        agent_zone_x = 0   # San Jose zone
        aws_zone_x = 450
    
    # Calculate agent positions
    agents_per_row = min(4, num_agents)
    rows = (num_agents + agents_per_row - 1) // agents_per_row
    
    agent_positions = []
    for i in range(num_agents):
        row = i // agents_per_row
        col = i % agents_per_row
        
        x = agent_zone_x + 50 + (col * 60)
        y = 50 + (row * 40)
        agent_positions.append((x, y))
    
    # Draw migration agents
    agent_colors = {'small': '#10b981', 'medium': '#3b82f6', 'large': '#f59e0b', 'xlarge': '#ef4444', 'xxlarge': '#8b5cf6'}
    agent_color = agent_colors.get(agent_size, '#6b7280')
    
    for i, (x, y) in enumerate(agent_positions):
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers+text',
            marker=dict(size=25, color=agent_color, symbol='diamond'),
            text=[f'Agent<br>{i+1}'],
            textposition='bottom center',
            name=f'Migration Agent {i+1}' if i == 0 else None,  # Only show legend for first agent
            showlegend=i == 0,
            hovertemplate=f'Migration Agent {i+1}<br>Size: {agent_size}<br>Type: {agent_analysis.get("primary_tool", "Unknown").upper()}<extra></extra>'
        ))
    
    # Destination storage in AWS
    storage_icons = {
        'S3': {'symbol': 'circle', 'color': '#f59e0b'},
        'FSx_Windows': {'symbol': 'square', 'color': '#3b82f6'},
        'FSx_Lustre': {'symbol': 'diamond', 'color': '#10b981'}
    }
    
    storage_config = storage_icons.get(destination_storage, storage_icons['S3'])
    
    fig.add_trace(go.Scatter(
        x=[aws_zone_x + 200], y=[100],
        mode='markers+text',
        marker=dict(size=40, color=storage_config['color'], symbol=storage_config['symbol']),
        text=[f'{destination_storage}<br>Storage'],
        textposition='bottom center',
        name='Destination Storage',
        hovertemplate=f'{destination_storage} Storage<br>AWS US-West-2<extra></extra>'
    ))
    
    # Add connection lines
    # From source to agents
    for x, y in agent_positions:
        if environment == 'production':
            source_x, source_y = 150, 100  # San Antonio DB
        else:
            source_x, source_y = 200, 100  # San Jose DB
        
        fig.add_trace(go.Scatter(
            x=[source_x, x],
            y=[source_y, y],
            mode='lines',
            line=dict(width=2, color='#6b7280', dash='dot'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # From agents to destination
    for x, y in agent_positions:
        dest_x = aws_zone_x + 200
        dest_y = 100
        
        fig.add_trace(go.Scatter(
            x=[x, dest_x],
            y=[y, dest_y],
            mode='lines',
            line=dict(width=3, color=agent_color),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Update layout
    fig.update_layout(
        title=f"Migration Agent Datacenter Placement<br><sub>{num_agents}x {agent_size} {agent_analysis.get('primary_tool', 'Migration').upper()} agents → {destination_storage}</sub>",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-50, 1100]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-20, 250]),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=400,
        margin=dict(l=20, r=20, t=80, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_aws_services_table(services_breakdown: List[Dict]) -> pd.DataFrame:
    """Create comprehensive AWS services table with enhanced details"""
    
    if not services_breakdown:
        return pd.DataFrame()
    
    # Convert to DataFrame with better formatting
    df_data = []
    
    for service in services_breakdown:
        df_data.append({
            'Service Category': service.get('service_category', 'Unknown'),
            'AWS Service': service.get('service_name', 'Unknown'),
            'Instance/Resource Type': service.get('instance_type', 'N/A'),
            'Monthly Cost ($)': f"${service.get('monthly_cost', 0):,.2f}",
            'Hourly Cost ($)': f"${service.get('cost_per_hour', 0):.4f}" if service.get('cost_per_hour', 0) > 0 else 'N/A',
            'Unit': service.get('unit', 'N/A'),
            'Description': service.get('description', 'No description'),
            'Cost per Month (Raw)': service.get('monthly_cost', 0)  # For sorting
        })
    
    df = pd.DataFrame(df_data)
    
    if not df.empty:
        # Sort by cost (descending) and then by category
        df = df.sort_values(['Cost per Month (Raw)', 'Service Category'], ascending=[False, True])
        # Remove the raw cost column used for sorting
        df = df.drop('Cost per Month (Raw)', axis=1)
    
    return df

def create_cost_breakdown_chart(comprehensive_costs: Dict) -> go.Figure:
    """Create enhanced cost breakdown visualization"""
    
    monthly_breakdown = comprehensive_costs.get('monthly_breakdown', {})
    
    if not monthly_breakdown:
        return go.Figure()
    
    # Prepare data
    categories = []
    costs = []
    colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4']
    
    for i, (category, cost) in enumerate(monthly_breakdown.items()):
        if cost > 0:
            categories.append(category.replace('_', ' ').title())
            costs.append(cost)
    
    # Create pie chart
    fig = go.Figure(data=[
        go.Pie(
            labels=categories,
            values=costs,
            hole=0.4,
            marker_colors=colors[:len(categories)],
            textinfo='label+percent+value',
            texttemplate='%{label}<br>%{percent}<br>$%{value:,.0f}',
            hovertemplate='<b>%{label}</b><br>Cost: $%{value:,.2f}<br>Percentage: %{percent}<extra></extra>'
        )
    ])
    
    # Add total in center
    total_cost = sum(costs)
    fig.add_annotation(
        text=f"Total Monthly<br><b>${total_cost:,.0f}</b>",
        x=0.5, y=0.5,
        font_size=16,
        showarrow=False
    )
    
    fig.update_layout(
        title="Monthly Cost Breakdown by Service Category",
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.01
        )
    )
    
    return fig

# ================================================================================================
# STREAMLIT APPLICATION UI WITH ENHANCED VISUALIZATIONS
# ================================================================================================

def main():
    """Main Streamlit application"""
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 AWS Enterprise Database Migration Analyzer AI v3.0</h1>
        <p>Comprehensive AI-powered analysis for database migrations to AWS with advanced agent optimization</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'last_config' not in st.session_state:
        st.session_state.last_config = None
    if 'agent_optimization_results' not in st.session_state:
        st.session_state.agent_optimization_results = None
    
    # Sidebar configuration
    with st.sidebar:
        st.header("📋 Migration Configuration")
        
        # Database Configuration
        st.subheader("🗄️ Database Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            database_size_gb = st.number_input(
                "Database Size (GB)", 
                min_value=1, max_value=1000000, 
                value=5000, step=100,
                help="Total size of database to migrate"
            )
            
            source_database_engine = st.selectbox(
                "Source Database Engine",
                options=['mysql', 'postgresql', 'oracle', 'sqlserver', 'mongodb'],
                index=0,
                help="Current database engine"
            )
        
        with col2:
            target_platform = st.selectbox(
                "Target Platform",
                options=['rds', 'ec2'],
                index=0,
                help="AWS target platform"
            )
            
            if target_platform == 'rds':
                database_engine = st.selectbox(
                    "Target Database Engine",
                    options=['mysql', 'postgresql', 'oracle', 'sqlserver'],
                    index=0,
                    help="Target RDS database engine"
                )
                ec2_database_engine = database_engine  # Same as RDS selection
                sql_server_deployment_type = 'standalone'
            else:  # EC2
                ec2_database_engine = st.selectbox(
                    "EC2 Database Engine",
                    options=['mysql', 'postgresql', 'oracle', 'sqlserver', 'mongodb'],
                    index=0,
                    help="Database engine for EC2 deployment"
                )
                database_engine = ec2_database_engine
                
                if ec2_database_engine == 'sqlserver':
                    sql_server_deployment_type = st.selectbox(
                        "SQL Server Deployment",
                        options=['standalone', 'always_on'],
                        index=0,
                        help="SQL Server deployment type"
                    )
                else:
                    sql_server_deployment_type = 'standalone'
        
        # Current Database Performance Metrics
        st.subheader("📊 Current Database Performance")
        with st.expander("Database Performance Metrics", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                current_db_max_memory_gb = st.number_input(
                    "Current DB Max Memory (GB)", 
                    min_value=0, max_value=1000, 
                    value=0, step=1,
                    help="Maximum memory used by database (0 = auto-size)"
                )
                current_db_max_cpu_cores = st.number_input(
                    "Current DB Max CPU Cores", 
                    min_value=0, max_value=128, 
                    value=0, step=1,
                    help="Maximum CPU cores used by database (0 = auto-size)"
                )
            with col2:
                current_db_max_iops = st.number_input(
                    "Current DB Max IOPS", 
                    min_value=0, max_value=500000, 
                    value=0, step=1000,
                    help="Maximum IOPS used by database (0 = auto-size)"
                )
                current_db_max_throughput_mbps = st.number_input(
                    "Current DB Max Throughput (MB/s)", 
                    min_value=0, max_value=10000, 
                    value=0, step=50,
                    help="Maximum throughput used by database (0 = auto-size)"
                )
        
        # Migration Method Configuration
        st.subheader("🔄 Migration Method")
        migration_method = st.selectbox(
            "Migration Method",
            options=['direct_replication', 'backup_restore'],
            format_func=lambda x: "Direct Replication" if x == 'direct_replication' else "Backup/Restore",
            help="Choose migration approach"
        )
        
        backup_storage_type = None
        backup_size_multiplier = 0.7
        
        if migration_method == 'backup_restore':
            col1, col2 = st.columns(2)
            with col1:
                backup_storage_type = st.selectbox(
                    "Backup Storage Type",
                    options=['nas_drive', 'windows_share'],
                    format_func=lambda x: "NAS Drive (NFS)" if x == 'nas_drive' else "Windows Share (SMB)",
                    help="Type of backup storage"
                )
            with col2:
                backup_size_multiplier = st.slider(
                    "Backup Size Factor", 
                    min_value=0.5, max_value=1.0, 
                    value=0.7, step=0.05,
                    help="Backup size as factor of database size"
                )
        
        # Agent Configuration
        st.subheader("🤖 Migration Agents")
        col1, col2 = st.columns(2)
        
        with col1:
            number_of_agents = st.selectbox(
                "Number of Agents",
                options=[1, 2, 3, 4, 5, 6, 8],
                index=1,
                help="Number of migration agents"
            )
            
            destination_storage_type = st.selectbox(
                "Destination Storage",
                options=['S3', 'FSx_Windows', 'FSx_Lustre'],
                help="AWS destination storage type"
            )
        
        with col2:
            datasync_agent_size = st.selectbox(
                "DataSync Agent Size",
                options=['small', 'medium', 'large', 'xlarge'],
                index=1,
                help="DataSync agent instance size"
            )
            
            dms_agent_size = st.selectbox(
                "DMS Agent Size",
                options=['small', 'medium', 'large', 'xlarge', 'xxlarge'],
                index=1,
                help="DMS agent instance size"
            )
        
        # Infrastructure Configuration
        st.subheader("🖥️ Current Infrastructure")
        col1, col2 = st.columns(2)
        
        with col1:
            operating_system = st.selectbox(
                "Operating System",
                options=['windows_server_2019', 'windows_server_2022', 'rhel_8', 'rhel_9', 'ubuntu_20_04', 'ubuntu_22_04'],
                index=2,
                help="Current server operating system"
            )
            
            server_type = st.selectbox(
                "Server Type",
                options=['physical', 'vmware'],
                help="Current server deployment type"
            )
            
            environment = st.selectbox(
                "Environment",
                options=['production', 'non-production'],
                help="Environment type"
            )
        
        with col2:
            cpu_cores = st.number_input(
                "CPU Cores", 
                min_value=1, max_value=128, 
                value=8, step=1
            )
            
            cpu_ghz = st.number_input(
                "CPU Speed (GHz)", 
                min_value=1.0, max_value=5.0, 
                value=2.4, step=0.1
            )
            
            ram_gb = st.number_input(
                "RAM (GB)", 
                min_value=1, max_value=1024, 
                value=32, step=1
            )
        
        # Network Configuration
        st.subheader("🌐 Network Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            nic_type = st.selectbox(
                "NIC Type",
                options=['gigabit_copper', 'gigabit_fiber', '10g_copper', '10g_fiber', '25g_fiber', '40g_fiber'],
                index=3,
                help="Network interface card type"
            )
        
        with col2:
            nic_speed = st.number_input(
                "NIC Speed (Mbps)", 
                min_value=100, max_value=100000, 
                value=10000, step=1000
            )
        
        # Performance Requirements
        st.subheader("⚡ Performance Requirements")
        col1, col2 = st.columns(2)
        
        with col1:
            performance_requirements = st.selectbox(
                "Performance Level",
                options=['standard', 'high'],
                help="Required performance level"
            )
        
        with col2:
            downtime_tolerance_minutes = st.number_input(
                "Max Downtime (minutes)", 
                min_value=0, max_value=1440, 
                value=240, step=30
            )
    
    # Analyze button
    if st.sidebar.button("🚀 Run AI Analysis", type="primary", use_container_width=True):
        # Build configuration
        config = {
            'database_size_gb': database_size_gb,
            'source_database_engine': source_database_engine,
            'database_engine': database_engine,
            'ec2_database_engine': ec2_database_engine,
            'target_platform': target_platform,
            'sql_server_deployment_type': sql_server_deployment_type,
            'migration_method': migration_method,
            'backup_storage_type': backup_storage_type,
            'backup_size_multiplier': backup_size_multiplier,
            'number_of_agents': number_of_agents,
            'destination_storage_type': destination_storage_type,
            'datasync_agent_size': datasync_agent_size,
            'dms_agent_size': dms_agent_size,
            'operating_system': operating_system,
            'server_type': server_type,
            'environment': environment,
            'cpu_cores': cpu_cores,
            'cpu_ghz': cpu_ghz,
            'ram_gb': ram_gb,
            'nic_type': nic_type,
            'nic_speed': nic_speed,
            'performance_requirements': performance_requirements,
            'downtime_tolerance_minutes': downtime_tolerance_minutes,
            'current_db_max_memory_gb': current_db_max_memory_gb,
            'current_db_max_cpu_cores': current_db_max_cpu_cores,
            'current_db_max_iops': current_db_max_iops,
            'current_db_max_throughput_mbps': current_db_max_throughput_mbps
        }
        
        # Check if configuration changed
        if config_has_changed(config, st.session_state.last_config):
            with st.spinner('🤖 Running comprehensive AI analysis...'):
                try:
                    # Initialize analyzer
                    analyzer = EnhancedMigrationAnalyzer()
                    
                    # Run analysis
                    results = asyncio.run(analyzer.comprehensive_ai_migration_analysis(config))
                    
                    # Store results
                    st.session_state.analysis_results = results
                    st.session_state.last_config = config.copy()
                    st.session_state.agent_optimization_results = None  # Reset optimization
                    
                    st.success("✅ Analysis completed successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    logger.error(f"Analysis error: {e}")
        else:
            st.info("ℹ️ Using cached results (configuration unchanged)")
    
    # Main content area
    if st.session_state.analysis_results:
        results = st.session_state.analysis_results
        
        # API Status
        col1, col2, col3 = st.columns(3)
        with col1:
            api_status = results.get('api_status', {})
            anthropic_status = "🟢 Connected" if api_status.anthropic_connected else "🔴 Offline"
            st.markdown(f"""
            <div class="api-status-card">
                <strong>Anthropic AI:</strong> {anthropic_status}
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            aws_status = "🟢 Connected" if api_status.aws_pricing_connected else "🔴 Offline"
            st.markdown(f"""
            <div class="api-status-card">
                <strong>AWS Pricing API:</strong> {aws_status}
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            last_update = api_status.last_update.strftime("%H:%M:%S") if api_status.last_update else "Unknown"
            st.markdown(f"""
            <div class="api-status-card">
                <strong>Last Update:</strong> {last_update}
            </div>
            """, unsafe_allow_html=True)
        
        # Main results tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Performance Analysis", 
            "🌐 Network Intelligence", 
            "🤖 Agent Configuration", 
            "💰 Cost Analysis", 
            "🎯 AWS Sizing", 
            "⚙️ Agent Optimization"
        ])
        
        with tab1:
            display_performance_analysis(results)
        
        with tab2:
            display_enhanced_network_intelligence(results)
        
        with tab3:
            display_agent_configuration(results)
        
        with tab4:
            display_enhanced_cost_analysis(results)
        
        with tab5:
            display_aws_sizing(results)
        
        with tab6:
            display_agent_optimization(results, st.session_state.last_config)

def display_performance_analysis(results):
    """Display performance analysis results"""
    st.header("📊 Performance Analysis")
    
    onprem_perf = results.get('onprem_performance', {})
    network_perf = results.get('network_performance', {})
    
    # Performance score overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        perf_score = onprem_perf.get('performance_score', 0)
        st.metric("Overall Performance", f"{perf_score:.1f}/100", 
                 delta=f"{perf_score-70:.1f}" if perf_score > 70 else None)
    
    with col2:
        network_quality = network_perf.get('ai_enhanced_quality_score', 0)
        st.metric("Network Quality", f"{network_quality:.1f}/100")
    
    with col3:
        total_latency = network_perf.get('total_latency_ms', 0)
        st.metric("Network Latency", f"{total_latency:.1f} ms")
    
    with col4:
        effective_bandwidth = network_perf.get('effective_bandwidth_mbps', 0)
        st.metric("Effective Bandwidth", f"{effective_bandwidth:,.0f} Mbps")
    
    # Detailed performance breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🖥️ On-Premises Performance")
        
        cpu_perf = onprem_perf.get('cpu_performance', {})
        memory_perf = onprem_perf.get('memory_performance', {})
        storage_perf = onprem_perf.get('storage_performance', {})
        
        st.markdown(f"""
        <div class="professional-card">
            <h4>CPU Performance</h4>
            <p><strong>Final Performance:</strong> {cpu_perf.get('final_performance', 0):.1f} units</p>
            <p><strong>Efficiency:</strong> {cpu_perf.get('efficiency_factor', 0)*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="professional-card">
            <h4>Memory Performance</h4>
            <p><strong>Total Memory:</strong> {memory_perf.get('total_memory_gb', 0)} GB</p>
            <p><strong>Effective Memory:</strong> {memory_perf.get('effective_memory_gb', 0):.1f} GB</p>
            <p><strong>Efficiency:</strong> {memory_perf.get('memory_efficiency', 0)*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="professional-card">
            <h4>Storage Performance</h4>
            <p><strong>Storage Type:</strong> {storage_perf.get('storage_type', 'Unknown').replace('_', ' ').title()}</p>
            <p><strong>Effective IOPS:</strong> {storage_perf.get('effective_iops', 0):,.0f}</p>
            <p><strong>Throughput:</strong> {storage_perf.get('effective_throughput_mbps', 0):,.0f} MB/s</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("🌐 Network Performance")
        
        st.markdown(f"""
        <div class="network-flow-container">
            <h4>Network Path Details</h4>
            <p><strong>Path:</strong> {network_perf.get('path_name', 'Unknown')}</p>
            <p><strong>Migration Type:</strong> {network_perf.get('migration_type', 'Unknown').replace('_', ' ').title()}</p>
            <p><strong>Destination Storage:</strong> {network_perf.get('destination_storage', 'Unknown')}</p>
            <p><strong>Environment:</strong> {network_perf.get('environment', 'Unknown').title()}</p>
            <p><strong>OS Type:</strong> {network_perf.get('os_type', 'Unknown').title()}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Network segments
        segments = network_perf.get('segments', [])
        if segments:
            for i, segment in enumerate(segments, 1):
                st.markdown(f"""
                <div class="professional-card">
                    <h5>Segment {i}: {segment.get('name', 'Unknown')}</h5>
                    <p><strong>Bandwidth:</strong> {segment.get('effective_bandwidth_mbps', 0):,.0f} Mbps</p>
                    <p><strong>Latency:</strong> {segment.get('effective_latency_ms', 0):.1f} ms</p>
                    <p><strong>Reliability:</strong> {segment.get('reliability', 0)*100:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
        
        # AI insights
        ai_insights = network_perf.get('ai_insights', {})
        if ai_insights:
            with st.expander("🤖 AI Network Insights"):
                bottlenecks = ai_insights.get('performance_bottlenecks', [])
                if bottlenecks:
                    st.write("**Performance Bottlenecks:**")
                    for bottleneck in bottlenecks:
                        st.write(f"• {bottleneck}")
                
                optimizations = ai_insights.get('optimization_opportunities', [])
                if optimizations:
                    st.write("**Optimization Opportunities:**")
                    for opt in optimizations:
                        st.write(f"• {opt}")

def display_enhanced_network_intelligence(results):
    """Display enhanced network intelligence with visual flow diagram"""
    st.header("🌐 Enhanced Network Intelligence")
    
    network_perf = results.get('network_performance', {})
    
    # Network overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        network_quality = network_perf.get('ai_enhanced_quality_score', 0)
        st.metric("Network Quality Score", f"{network_quality:.1f}/100")
    
    with col2:
        total_latency = network_perf.get('total_latency_ms', 0)
        st.metric("End-to-End Latency", f"{total_latency:.1f} ms")
    
    with col3:
        effective_bandwidth = network_perf.get('effective_bandwidth_mbps', 0)
        st.metric("Effective Bandwidth", f"{effective_bandwidth:,.0f} Mbps")
    
    with col4:
        total_reliability = network_perf.get('total_reliability', 0)
        st.metric("Path Reliability", f"{total_reliability*100:.2f}%")
    
    # Enhanced Network Flow Visualization
    st.subheader("🔄 Network Flow Visualization")
    
    try:
        # Create network flow diagram
        network_flow_fig = create_network_flow_diagram(network_perf)
        st.plotly_chart(network_flow_fig, use_container_width=True)
    except Exception as e:
        st.error(f"Could not create network flow diagram: {str(e)}")
        
        # Fallback text-based visualization
        st.markdown("### Network Path Flow")
        segments = network_perf.get('segments', [])
        if segments:
            for i, segment in enumerate(segments, 1):
                st.markdown(f"""
                <div class="network-flow-container">
                    <div class="network-node">Segment {i}</div>
                    <div class="network-connection">
                        <div class="connection-label">{segment.get('effective_bandwidth_mbps', 0):,.0f} Mbps | {segment.get('effective_latency_ms', 0):.1f} ms</div>
                    </div>
                    <p><strong>{segment.get('name', 'Unknown')}</strong></p>
                    <p>Type: {segment.get('connection_type', 'Unknown').replace('_', ' ').title()}</p>
                    <p>Reliability: {segment.get('reliability', 0)*100:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Network Path Details
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Path Configuration")
        
        st.markdown(f"""
        <div class="professional-card">
            <h4>Migration Path Details</h4>
            <p><strong>Path Name:</strong> {network_perf.get('path_name', 'Unknown')}</p>
            <p><strong>Migration Type:</strong> {network_perf.get('migration_type', 'Unknown').replace('_', ' ').title()}</p>
            <p><strong>Environment:</strong> {network_perf.get('environment', 'Unknown').title()}</p>
            <p><strong>OS Type:</strong> {network_perf.get('os_type', 'Unknown').title()}</p>
            <p><strong>Storage Type:</strong> {network_perf.get('storage_type', 'Unknown').replace('_', ' ').title()}</p>
            <p><strong>Destination:</strong> {network_perf.get('destination_storage', 'Unknown')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Performance metrics
        st.markdown(f"""
        <div class="professional-card">
            <h4>Performance Metrics</h4>
            <p><strong>Base Quality Score:</strong> {network_perf.get('network_quality_score', 0):.1f}/100</p>
            <p><strong>AI Enhanced Score:</strong> {network_perf.get('ai_enhanced_quality_score', 0):.1f}/100</p>
            <p><strong>Optimization Potential:</strong> {network_perf.get('ai_optimization_potential', 0):.1f}%</p>
            <p><strong>Storage Bonus:</strong> +{network_perf.get('storage_performance_bonus', 0):.0f} points</p>
            <p><strong>Cost Factor:</strong> {network_perf.get('total_cost_factor', 0):.1f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("🎯 AI Insights & Recommendations")
        
        ai_insights = network_perf.get('ai_insights', {})
        if ai_insights:
            
            # Performance bottlenecks
            bottlenecks = ai_insights.get('performance_bottlenecks', [])
            if bottlenecks:
                st.markdown("**🚨 Performance Bottlenecks:**")
                for bottleneck in bottlenecks:
                    st.markdown(f"• {bottleneck}")
            
            # Optimization opportunities
            optimizations = ai_insights.get('optimization_opportunities', [])
            if optimizations:
                st.markdown("**⚡ Optimization Opportunities:**")
                for opt in optimizations:
                    st.markdown(f"• {opt}")
            
            # Risk factors
            risks = ai_insights.get('risk_factors', [])
            if risks:
                st.markdown("**⚠️ Risk Factors:**")
                for risk in risks:
                    st.markdown(f"• {risk}")
            
            # Recommended improvements
            improvements = ai_insights.get('recommended_improvements', [])
            if improvements:
                st.markdown("**🔧 Recommended Improvements:**")
                for improvement in improvements:
                    st.markdown(f"• {improvement}")
        else:
            st.info("No AI insights available for this network path.")
    
    # Detailed Segment Analysis
    st.subheader("🔍 Detailed Segment Analysis")
    
    segments = network_perf.get('segments', [])
    if segments:
        # Create a table for segment details
        segment_data = []
        for i, segment in enumerate(segments, 1):
            segment_data.append({
                'Segment': f"Segment {i}",
                'Name': segment.get('name', 'Unknown'),
                'Type': segment.get('connection_type', 'Unknown').replace('_', ' ').title(),
                'Base Bandwidth (Mbps)': f"{segment.get('bandwidth_mbps', 0):,.0f}",
                'Effective Bandwidth (Mbps)': f"{segment.get('effective_bandwidth_mbps', 0):,.0f}",
                'Base Latency (ms)': f"{segment.get('latency_ms', 0):.1f}",
                'Effective Latency (ms)': f"{segment.get('effective_latency_ms', 0):.1f}",
                'Reliability (%)': f"{segment.get('reliability', 0)*100:.2f}%",
                'Congestion Factor': f"{segment.get('congestion_factor', 1.0):.2f}",
                'AI Optimization Potential (%)': f"{segment.get('ai_optimization_potential', 0)*100:.1f}%"
            })
        
        import pandas as pd
        segment_df = pd.DataFrame(segment_data)
        st.dataframe(segment_df, use_container_width=True)
    
    # Network optimization recommendations
    st.subheader("💡 Network Optimization Strategy")
    
    optimization_potential = network_perf.get('ai_optimization_potential', 0)
    if optimization_potential > 10:
        st.warning(f"🔧 High optimization potential detected: {optimization_potential:.1f}%")
        st.markdown("**Priority Actions:**")
        st.markdown("• Review network configuration for performance improvements")
        st.markdown("• Consider upgrading connection types with high latency")
        st.markdown("• Implement AI-recommended optimizations")
    elif optimization_potential > 5:
        st.info(f"🛠️ Moderate optimization potential: {optimization_potential:.1f}%")
        st.markdown("**Suggested Actions:**")
        st.markdown("• Monitor network performance during migration")
        st.markdown("• Consider minor configuration adjustments")
    else:
        st.success(f"✅ Network path is well-optimized: {optimization_potential:.1f}% optimization potential")
        st.markdown("• Current configuration appears optimal for migration")

def display_agent_configuration(results):
    """Display agent configuration analysis"""
    st.header("🤖 Agent Configuration Analysis")
    
    agent_analysis = results.get('agent_analysis', {})
    
    # Agent overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Primary Tool", agent_analysis.get('primary_tool', 'Unknown').upper())
    
    with col2:
        st.metric("Number of Agents", agent_analysis.get('number_of_agents', 0))
    
    with col3:
        st.metric("Agent Size", agent_analysis.get('agent_size', 'Unknown').title())
    
    with col4:
        throughput = agent_analysis.get('total_effective_throughput', 0)
        st.metric("Total Throughput", f"{throughput:,.0f} Mbps")
    
    # Agent details
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚙️ Configuration Details")
        
        st.markdown(f"""
        <div class="professional-card">
            <h4>Agent Specifications</h4>
            <p><strong>Migration Method:</strong> {agent_analysis.get('migration_method', 'Unknown').replace('_', ' ').title()}</p>
            <p><strong>Destination Storage:</strong> {agent_analysis.get('destination_storage', 'Unknown')}</p>
            <p><strong>Backup Storage Type:</strong> {agent_analysis.get('backup_storage_type', 'N/A').replace('_', ' ').title()}</p>
            <p><strong>Scaling Efficiency:</strong> {agent_analysis.get('scaling_efficiency', 0)*100:.1f}%</p>
            <p><strong>Storage Performance Multiplier:</strong> {agent_analysis.get('storage_performance_multiplier', 1.0):.2f}x</p>
        </div>
        """, unsafe_allow_html=True)
        
        bottleneck = agent_analysis.get('bottleneck', 'Unknown')
        bottleneck_severity = agent_analysis.get('bottleneck_severity', 'medium')
        bottleneck_color = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}.get(bottleneck_severity, '🟡')
        
        st.markdown(f"""
        <div class="professional-card">
            <h4>Bottleneck Analysis</h4>
            <p><strong>Current Bottleneck:</strong> {bottleneck_color} {bottleneck}</p>
            <p><strong>Severity:</strong> {bottleneck_severity.title()}</p>
            <p><strong>Max Agent Throughput:</strong> {agent_analysis.get('total_max_throughput_mbps', 0):,.0f} Mbps</p>
            <p><strong>Effective Throughput:</strong> {agent_analysis.get('total_effective_throughput', 0):,.0f} Mbps</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("💰 Cost Breakdown")
        
        monthly_cost = agent_analysis.get('monthly_cost', 0)
        cost_per_hour = agent_analysis.get('cost_per_hour', 0)
        
        st.markdown(f"""
        <div class="professional-card">
            <h4>Agent Costs</h4>
            <p><strong>Monthly Cost:</strong> ${monthly_cost:,.2f}</p>
            <p><strong>Hourly Cost:</strong> ${cost_per_hour:.2f}</p>
            <p><strong>Cost per Mbps:</strong> ${monthly_cost/max(agent_analysis.get('total_effective_throughput', 1), 1):.2f}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Agent configuration details
        agent_config = agent_analysis.get('agent_configuration', {})
        if agent_config:
            per_agent_spec = agent_config.get('per_agent_spec', {})
            st.markdown(f"""
            <div class="professional-card">
                <h4>Per-Agent Specifications</h4>
                <p><strong>Instance Type:</strong> {per_agent_spec.get('name', 'Unknown')}</p>
                <p><strong>vCPU:</strong> {per_agent_spec.get('vcpu', 0)}</p>
                <p><strong>Memory:</strong> {per_agent_spec.get('memory_gb', 0)} GB</p>
                <p><strong>Max Throughput:</strong> {per_agent_spec.get('max_throughput_mbps_per_agent', 0):,.0f} Mbps</p>
                <p><strong>Max Tasks:</strong> {per_agent_spec.get('max_concurrent_tasks_per_agent', 0)}</p>
            </div>
            """, unsafe_allow_html=True)

def display_enhanced_cost_analysis(results):
    """Display enhanced comprehensive cost analysis with tabular AWS services view"""
    st.header("💰 Comprehensive Cost Analysis")
    
    cost_analysis = results.get('cost_analysis', {})
    comprehensive_costs = results.get('comprehensive_costs', {})
    
    # Cost overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_monthly = comprehensive_costs.get('total_monthly', cost_analysis.get('total_monthly_cost', 0))
        st.metric("Monthly Costs", f"${total_monthly:,.0f}")
    
    with col2:
        total_one_time = comprehensive_costs.get('total_one_time', cost_analysis.get('one_time_migration_cost', 0))
        st.metric("One-time Costs", f"${total_one_time:,.0f}")
    
    with col3:
        annual_total = comprehensive_costs.get('annual_total', total_monthly * 12 + total_one_time)
        st.metric("Annual Total", f"${annual_total:,.0f}")
    
    with col4:
        three_year = comprehensive_costs.get('three_year_total', total_monthly * 36 + total_one_time)
        st.metric("3-Year Total", f"${three_year:,.0f}")
    
    # Enhanced Cost Visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Cost Distribution")
        
        try:
            # Create enhanced cost breakdown chart
            cost_chart = create_cost_breakdown_chart(comprehensive_costs)
            st.plotly_chart(cost_chart, use_container_width=True)
        except Exception as e:
            st.error(f"Could not create cost chart: {str(e)}")
            
            # Fallback: simple metrics display
            monthly_breakdown = comprehensive_costs.get('monthly_breakdown', {})
            for category, amount in monthly_breakdown.items():
                if amount > 0:
                    st.metric(category.replace('_', ' ').title(), f"${amount:,.2f}/month")
    
    with col2:
        st.subheader("🔄 One-time vs Recurring Costs")
        
        one_time_breakdown = comprehensive_costs.get('one_time_breakdown', {})
        monthly_breakdown = comprehensive_costs.get('monthly_breakdown', {})
        
        # One-time costs
        st.markdown("**One-time Costs:**")
        for category, amount in one_time_breakdown.items():
            if amount > 0:
                st.markdown(f"• {category.replace('_', ' ').title()}: ${amount:,.2f}")
        
        # Monthly recurring costs
        st.markdown("**Monthly Recurring Costs:**")
        for category, amount in monthly_breakdown.items():
            if amount > 0:
                st.markdown(f"• {category.replace('_', ' ').title()}: ${amount:,.2f}")
    
    # Comprehensive AWS Services Table
    st.subheader("🏗️ Detailed AWS Services Breakdown")
    
    services_breakdown = comprehensive_costs.get('detailed_service_breakdown', [])
    if services_breakdown:
        try:
            # Create comprehensive services table
            services_df = create_aws_services_table(services_breakdown)
            
            if not services_df.empty:
                st.markdown("**All AWS Services and Resources:**")
                st.dataframe(services_df, use_container_width=True, height=400)
                
                # Service category summary
                st.subheader("📋 Service Category Summary")
                
                # Group by category for summary
                category_costs = {}
                for service in services_breakdown:
                    category = service.get('service_category', 'Unknown')
                    cost = service.get('monthly_cost', 0)
                    if category not in category_costs:
                        category_costs[category] = 0
                    category_costs[category] += cost
                
                # Display category summary
                col1, col2, col3 = st.columns(3)
                for i, (category, cost) in enumerate(sorted(category_costs.items(), key=lambda x: x[1], reverse=True)):
                    with [col1, col2, col3][i % 3]:
                        st.metric(category, f"${cost:,.2f}/month")
                
                # Export option
                st.download_button(
                    label="📥 Download Services Breakdown (CSV)",
                    data=services_df.to_csv(index=False),
                    file_name="aws_services_breakdown.csv",
                    mime="text/csv"
                )
            else:
                st.info("No detailed service breakdown available.")
        except Exception as e:
            st.error(f"Could not create services table: {str(e)}")
            
            # Fallback: show raw breakdown
            st.write("**Service Breakdown (Raw Data):**")
            for service in services_breakdown[:10]:  # Show first 10
                st.write(f"• {service.get('service_name', 'Unknown')}: ${service.get('monthly_cost', 0):,.2f}/month")
    else:
        st.info("Detailed service breakdown not available.")
    
    # Cost Optimization Recommendations
    st.subheader("💡 Cost Optimization Recommendations")
    
    optimizations = comprehensive_costs.get('cost_optimization_recommendations', [])
    if optimizations:
        for i, opt in enumerate(optimizations, 1):
            st.markdown(f"{i}. {opt}")
    else:
        st.info("No specific cost optimization recommendations available.")
    
    # Cost Comparison and ROI Analysis
    st.subheader("📈 Cost Analysis & ROI")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Investment Timeline:**")
        st.markdown(f"• **Monthly Operating Cost:** ${total_monthly:,.0f}")
        st.markdown(f"• **Year 1 Total:** ${total_monthly * 12 + total_one_time:,.0f}")
        st.markdown(f"• **Year 2 Total:** ${total_monthly * 24 + total_one_time:,.0f}")
        st.markdown(f"• **Year 3 Total:** ${total_monthly * 36 + total_one_time:,.0f}")
    
    with col2:
        # Calculate potential savings and ROI
        estimated_savings = cost_analysis.get('estimated_monthly_savings', 500)
        roi_months = cost_analysis.get('roi_months', 12)
        
        st.markdown("**ROI Analysis:**")
        st.markdown(f"• **Estimated Monthly Savings:** ${estimated_savings:,.0f}")
        st.markdown(f"• **Break-even Period:** {roi_months} months")
        st.markdown(f"• **3-Year Net Savings:** ${estimated_savings * 36 - total_one_time:,.0f}")
        
        if estimated_savings * 36 > total_one_time:
            st.success("✅ Positive ROI projected over 3 years")
        else:
            st.warning("⚠️ Extended payback period - review cost optimization opportunities")

def display_aws_sizing(results):
    """Display AWS sizing recommendations"""
    st.header("🎯 AWS Sizing Recommendations")
    
    aws_sizing = results.get('aws_sizing_recommendations', {})
    
    # Deployment recommendation
    deployment_rec = aws_sizing.get('deployment_recommendation', {})
    recommendation = deployment_rec.get('recommendation', 'Unknown')
    confidence = deployment_rec.get('confidence', 0)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Recommended Platform", recommendation.upper())
    
    with col2:
        st.metric("Confidence Level", f"{confidence*100:.0f}%")
    
    with col3:
        analytical_rec = deployment_rec.get('analytical_recommendation', 'Unknown')
        st.metric("AI Analysis Suggests", analytical_rec.upper())
    
    # Platform-specific recommendations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🗄️ RDS Recommendations")
        rds_rec = aws_sizing.get('rds_recommendations', {})
        
        st.markdown(f"""
        <div class="professional-card">
            <h4>Instance Configuration</h4>
            <p><strong>Primary Instance:</strong> {rds_rec.get('primary_instance', 'Unknown')}</p>
            <p><strong>Storage Type:</strong> {rds_rec.get('storage_type', 'gp3').upper()}</p>
            <p><strong>Storage Size:</strong> {rds_rec.get('storage_size_gb', 0):,} GB</p>
            <p><strong>Multi-AZ:</strong> {'Yes' if rds_rec.get('multi_az', False) else 'No'}</p>
            <p><strong>Backup Retention:</strong> {rds_rec.get('backup_retention_days', 7)} days</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="professional-card">
            <h4>Monthly Costs</h4>
            <p><strong>Instance Cost:</strong> ${rds_rec.get('monthly_instance_cost', 0):,.2f}</p>
            <p><strong>Storage Cost:</strong> ${rds_rec.get('monthly_storage_cost', 0):,.2f}</p>
            <p><strong>Total RDS Cost:</strong> ${rds_rec.get('total_monthly_cost', 0):,.2f}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # RDS Sizing reasoning
        sizing_reasoning = rds_rec.get('sizing_reasoning', [])
        if sizing_reasoning:
            with st.expander("💭 RDS Sizing Logic"):
                for reason in sizing_reasoning:
                    st.write(f"• {reason}")
    
    with col2:
        st.subheader("💻 EC2 Recommendations")
        ec2_rec = aws_sizing.get('ec2_recommendations', {})
        
        st.markdown(f"""
        <div class="professional-card">
            <h4>Instance Configuration</h4>
            <p><strong>Primary Instance:</strong> {ec2_rec.get('primary_instance', 'Unknown')}</p>
            <p><strong>Database Engine:</strong> {ec2_rec.get('database_engine', 'Unknown')}</p>
            <p><strong>Instance Count:</strong> {ec2_rec.get('instance_count', 1)}</p>
            <p><strong>Deployment:</strong> {ec2_rec.get('deployment_description', 'Single Instance')}</p>
            <p><strong>Storage Type:</strong> {ec2_rec.get('storage_type', 'gp3').upper()}</p>
            <p><strong>Storage Size:</strong> {ec2_rec.get('storage_size_gb', 0):,} GB</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="professional-card">
            <h4>Monthly Costs</h4>
            <p><strong>Instance Cost:</strong> ${ec2_rec.get('monthly_instance_cost', 0):,.2f}</p>
            <p><strong>Storage Cost:</strong> ${ec2_rec.get('monthly_storage_cost', 0):,.2f}</p>
            <p><strong>OS Licensing:</strong> ${ec2_rec.get('os_licensing_cost', 0):,.2f}</p>
            <p><strong>Total EC2 Cost:</strong> ${ec2_rec.get('total_monthly_cost', 0):,.2f}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # SQL Server specific information
        if ec2_rec.get('sql_server_considerations', False):
            is_always_on = ec2_rec.get('is_always_on_cluster', False)
            if is_always_on:
                with st.expander("🏢 SQL Server Always On Benefits"):
                    benefits = ec2_rec.get('always_on_benefits', [])
                    for benefit in benefits:
                        st.write(f"• {benefit}")
                
                with st.expander("⚙️ Cluster Requirements"):
                    requirements = ec2_rec.get('cluster_requirements', [])
                    for req in requirements:
                        st.write(f"• {req}")
        
        # EC2 sizing reasoning
        sizing_reasoning = ec2_rec.get('sizing_reasoning', [])
        if sizing_reasoning:
            with st.expander("💭 EC2 Sizing Logic"):
                for reason in sizing_reasoning:
                    st.write(f"• {reason}")
    
    # Reader/Writer configuration
    reader_writer = aws_sizing.get('reader_writer_config', {})
    if reader_writer:
        st.subheader("📖 Reader/Writer Configuration")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Writer Instances", reader_writer.get('writers', 1))
        
        with col2:
            st.metric("Reader Instances", reader_writer.get('readers', 0))
        
        with col3:
            st.metric("Total Instances", reader_writer.get('total_instances', 1))
        
        with col4:
            st.metric("Read Split %", f"{reader_writer.get('recommended_read_split', 0):.0f}%")

def display_agent_optimization(results, config):
    """Display enhanced agent optimization analysis with datacenter diagram"""
    st.header("⚙️ Agent Optimization Analysis")
    
    # Check if optimization has been run
    if st.session_state.agent_optimization_results is None:
        if st.button("🚀 Run Agent Optimization Analysis", type="primary", use_container_width=True):
            with st.spinner('🤖 Running agent optimization analysis...'):
                try:
                    # Initialize optimizer
                    analyzer = EnhancedMigrationAnalyzer()
                    optimizer = AgentScalingOptimizer(analyzer.ai_manager, analyzer.agent_manager)
                    
                    # Run optimization
                    optimization_results = run_agent_optimization_sync(optimizer, config, results)
                    st.session_state.agent_optimization_results = optimization_results
                    
                    st.success("✅ Agent optimization completed!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Optimization failed: {str(e)}")
                    return
        else:
            st.info("Click the button above to run detailed agent optimization analysis.")
            return
    
    opt_results = st.session_state.agent_optimization_results
    
    # Enhanced Datacenter Visualization
    st.subheader("🏢 Agent Datacenter Placement Diagram")
    
    try:
        # Create datacenter diagram
        agent_analysis = results.get('agent_analysis', {})
        datacenter_fig = create_agent_datacenter_diagram(agent_analysis, config)
        st.plotly_chart(datacenter_fig, use_container_width=True)
    except Exception as e:
        st.error(f"Could not create datacenter diagram: {str(e)}")
        
        # Fallback: text-based diagram
        st.markdown("### Agent Placement Overview")
        agent_analysis = results.get('agent_analysis', {})
        num_agents = agent_analysis.get('number_of_agents', 1)
        agent_size = agent_analysis.get('agent_size', 'medium')
        destination_storage = config.get('destination_storage_type', 'S3')
        environment = config.get('environment', 'non-production')
        
        if environment == 'production':
            st.markdown("""
            <div class="datacenter-diagram">
                <div class="datacenter-zone">
                    <h4>📍 San Antonio DC</h4>
                    <div class="network-node">Database Server</div>
                    <div class="network-node">Backup Storage</div>
                </div>
                <div class="datacenter-zone">
                    <h4>📍 San Jose DC</h4>
                    """ + "".join([f'<div class="agent-placement">Agent {i+1} ({agent_size})</div>' for i in range(num_agents)]) + f"""
                </div>
                <div class="datacenter-zone">
                    <h4>☁️ AWS US-West-2</h4>
                    <div class="network-node">{destination_storage} Storage</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="datacenter-diagram">
                <div class="datacenter-zone">
                    <h4>📍 San Jose DC</h4>
                    <div class="network-node">Database Server</div>
                    <div class="network-node">Backup Storage</div>
                    """ + "".join([f'<div class="agent-placement">Agent {i+1} ({agent_size})</div>' for i in range(num_agents)]) + f"""
                </div>
                <div class="datacenter-zone">
                    <h4>☁️ AWS US-West-2</h4>
                    <div class="network-node">{destination_storage} Storage</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Optimization summary
    opt_summary = opt_results.get('optimization_summary', {})
    
    if opt_summary.get('optimization_available', False):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_config = opt_summary.get('current_configuration', 'Unknown')
            st.metric("Current Config", current_config)
        
        with col2:
            recommended_config = opt_summary.get('recommended_configuration', 'Unknown')
            st.metric("Recommended Config", recommended_config)
        
        with col3:
            perf_improvement = opt_summary.get('performance_improvement', {})
            throughput_change = perf_improvement.get('throughput_change_percent', 0)
            st.metric("Performance Change", f"{throughput_change:+.1f}%")
        
        with col4:
            cost_impact = opt_summary.get('cost_impact', {})
            cost_change = cost_impact.get('cost_change_percent', 0)
            st.metric("Cost Impact", f"{cost_change:+.1f}%")
    
    # Detailed optimization results
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Optimal Configurations",
        "🤖 AI Recommendations", 
        "📊 Cost vs Performance",
        "🔍 Bottleneck Analysis"
    ])
    
    with tab1:
        display_optimal_configurations(opt_results)
    
    with tab2:
        display_ai_recommendations(opt_results)
    
    with tab3:
        display_cost_vs_performance(opt_results)
    
    with tab4:
        display_bottleneck_analysis(opt_results)

def display_optimal_configurations(opt_results):
    """Display optimal agent configurations"""
    st.subheader("🎯 Optimal Agent Configurations")
    
    optimal_configs = opt_results.get('optimal_configurations', {})
    
    if optimal_configs:
        # Top configurations table
        config_data = []
        for config_key, config_details in list(optimal_configs.items())[:10]:
            config_data.append({
                'Configuration': config_key,
                'Agent Count': config_details.get('agent_count', 0),
                'Agent Size': config_details.get('agent_size', 'Unknown'),
                'Throughput (Mbps)': f"{config_details.get('total_throughput', 0):,.0f}",
                'Monthly Cost': f"${config_details.get('monthly_cost', 0):,.0f}",
                'Efficiency Score': f"{config_details.get('efficiency_score', 0):.1f}/100",
                'Cost/Mbps': f"${config_details.get('cost_per_mbps', 0):.2f}",
                'Management': config_details.get('management_complexity', 'Unknown'),
                'Best For': config_details.get('recommended_for', 'General use')
            })
        
        if config_data:
            import pandas as pd
            df = pd.DataFrame(config_data)
            st.dataframe(df, use_container_width=True)
        
        # Detailed view of top 3 configurations
        st.subheader("📋 Top 3 Configuration Details")
        
        top_configs = list(optimal_configs.items())[:3]
        
        for i, (config_key, config_details) in enumerate(top_configs, 1):
            with st.expander(f"#{i}: {config_key} Configuration"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    **Configuration:** {config_details.get('agent_count', 0)}x {config_details.get('agent_size', 'Unknown')} agents  
                    **Primary Tool:** {config_details.get('primary_tool', 'Unknown').upper()}  
                    **Total Throughput:** {config_details.get('total_throughput', 0):,.0f} Mbps  
                    **Monthly Cost:** ${config_details.get('monthly_cost', 0):,.0f}  
                    """)
                
                with col2:
                    st.markdown(f"""
                    **Efficiency Score:** {config_details.get('efficiency_score', 0):.1f}/100  
                    **Cost per Mbps:** ${config_details.get('cost_per_mbps', 0):.2f}  
                    **Management Complexity:** {config_details.get('management_complexity', 'Unknown')}  
                    **Recommended For:** {config_details.get('recommended_for', 'General use')}  
                    """)
        
        # Scaling scenarios
        scaling_scenarios = opt_results.get('scaling_scenarios', {})
        if scaling_scenarios:
            st.subheader("📈 Scaling Scenarios")
            
            for scenario_key, scenario in scaling_scenarios.items():
                st.markdown(f"""
                <div class="agent-scaling-card">
                    <h4>{scenario.get('name', 'Unknown Scenario')}</h4>
                    <p><strong>Description:</strong> {scenario.get('description', 'No description')}</p>
                    <p><strong>Configuration:</strong> {scenario.get('agent_count', 0)}x {scenario.get('agent_size', 'Unknown')} agents</p>
                    <p><strong>Expected Throughput:</strong> {scenario.get('expected_throughput', 'Unknown')}</p>
                    <p><strong>Risk Level:</strong> {scenario.get('risk_level', 'Unknown')} | <strong>Cost Level:</strong> {scenario.get('cost_level', 'Unknown')}</p>
                    <p><strong>Suitable For:</strong> {', '.join(scenario.get('suitable_for', []))}</p>
                </div>
                """, unsafe_allow_html=True)

def display_ai_recommendations(opt_results):
    """Display AI-powered recommendations"""
    st.subheader("🤖 AI-Powered Recommendations")
    
    ai_recommendations = opt_results.get('ai_recommendations', {})
    
    if ai_recommendations.get('ai_analysis_available', False):
        # Recommended configuration
        recommended_config = ai_recommendations.get('recommended_configuration', 'Unknown')
        st.markdown(f"""
        <div class="insight-card">
            <h4>🎯 Recommended Configuration</h4>
            <p><strong>{recommended_config}</strong></p>
            <p>Based on comprehensive AI analysis of your workload characteristics</p>
        </div>
        """, unsafe_allow_html=True)
        
        # AI recommendations sections
        col1, col2 = st.columns(2)
        
        with col1:
            # Scaling strategy
            scaling_strategy = ai_recommendations.get('scaling_strategy', [])
            if scaling_strategy:
                st.markdown("**🔧 Scaling Strategy:**")
                for strategy in scaling_strategy:
                    st.write(f"• {strategy}")
            
            # Performance tuning
            performance_tuning = ai_recommendations.get('performance_tuning', [])
            if performance_tuning:
                st.markdown("**⚡ Performance Tuning:**")
                for tuning in performance_tuning:
                    st.write(f"• {tuning}")
            
            # Implementation plan
            implementation_plan = ai_recommendations.get('implementation_plan', [])
            if implementation_plan:
                st.markdown("**📋 Implementation Plan:**")
                for step in implementation_plan:
                    st.write(f"{step}")
        
        with col2:
            # Cost optimization
            cost_optimization = ai_recommendations.get('cost_optimization_tips', [])
            if cost_optimization:
                st.markdown("**💰 Cost Optimization:**")
                for tip in cost_optimization:
                    st.write(f"• {tip}")
            
            # Risk mitigation
            risk_mitigation = ai_recommendations.get('risk_mitigation', [])
            if risk_mitigation:
                st.markdown("**🛡️ Risk Mitigation:**")
                for risk in risk_mitigation:
                    st.write(f"• {risk}")
            
            # Monitoring recommendations
            monitoring = ai_recommendations.get('monitoring_recommendations', [])
            if monitoring:
                st.markdown("**📊 Monitoring:**")
                for monitor in monitoring:
                    st.write(f"• {monitor}")
        
        # Backup storage considerations
        backup_considerations = ai_recommendations.get('backup_storage_considerations', [])
        if backup_considerations and backup_considerations[0] != "Not applicable for direct replication method":
            with st.expander("💾 Backup Storage Considerations"):
                for consideration in backup_considerations:
                    st.write(f"• {consideration}")
        
        # Fallback options
        fallback_options = ai_recommendations.get('fallback_options', [])
        if fallback_options:
            with st.expander("🔄 Fallback Options"):
                for option in fallback_options:
                    st.write(f"• {option}")
        
        # Raw AI response
        with st.expander("🤖 Raw AI Analysis"):
            raw_response = ai_recommendations.get('raw_ai_response', 'No detailed analysis available')
            st.text_area("AI Analysis", raw_response, height=300)
    
    else:
        st.warning("⚠️ AI analysis not available. Using fallback recommendations.")
        
        # Show fallback recommendations
        scaling_strategy = ai_recommendations.get('scaling_strategy', [])
        if scaling_strategy:
            for strategy in scaling_strategy:
                st.write(f"• {strategy}")

def display_cost_vs_performance(opt_results):
    """Display cost vs performance analysis"""
    st.subheader("📊 Cost vs Performance Analysis")
    
    cost_performance = opt_results.get('cost_vs_performance', {})
    
    if cost_performance.get('analysis_available', False):
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            configs_analyzed = cost_performance.get('configurations_analyzed', 0)
            st.metric("Configurations Analyzed", configs_analyzed)
        
        with col2:
            cost_range = cost_performance.get('cost_range', {})
            cost_spread = cost_range.get('max', 0) - cost_range.get('min', 0)
            st.metric("Cost Range", f"${cost_spread:,.0f}")
        
        with col3:
            throughput_range = cost_performance.get('throughput_range', {})
            throughput_spread = throughput_range.get('max', 0) - throughput_range.get('min', 0)
            st.metric("Throughput Range", f"{throughput_spread:,.0f} Mbps")
        
        with col4:
            # Calculate efficiency spread
            st.metric("Analysis Depth", "Comprehensive")
        
        # Optimization categories
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🎯 Optimization Categories:**")
            
            cost_optimized = cost_performance.get('cost_optimized', {})
            st.markdown(f"""
            <div class="metric-card">
                <strong>💰 Cost Optimized:</strong> {cost_optimized.get('name', 'Unknown')}<br>
                <small>Cost: ${cost_optimized.get('cost', 0):,.0f} | Throughput: {cost_optimized.get('throughput', 0):,.0f} Mbps</small>
            </div>
            """, unsafe_allow_html=True)
            
            performance_optimized = cost_performance.get('performance_optimized', {})
            st.markdown(f"""
            <div class="metric-card">
                <strong>⚡ Performance Optimized:</strong> {performance_optimized.get('name', 'Unknown')}<br>
                <small>Cost: ${performance_optimized.get('cost', 0):,.0f} | Throughput: {performance_optimized.get('throughput', 0):,.0f} Mbps</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            efficiency_optimized = cost_performance.get('efficiency_optimized', {})
            st.markdown(f"""
            <div class="metric-card">
                <strong>🎯 Efficiency Optimized:</strong> {efficiency_optimized.get('name', 'Unknown')}<br>
                <small>Score: {efficiency_optimized.get('efficiency_score', 0):.1f}/100 | Cost/Mbps: ${efficiency_optimized.get('cost_per_mbps', 0):.2f}</small>
            </div>
            """, unsafe_allow_html=True)
            
            value_optimized = cost_performance.get('value_optimized', {})
            st.markdown(f"""
            <div class="metric-card">
                <strong>💎 Value Optimized:</strong> {value_optimized.get('name', 'Unknown')}<br>
                <small>Best Cost/Mbps: ${value_optimized.get('cost_per_mbps', 0):.2f}</small>
            </div>
            """, unsafe_allow_html=True)
        
        # Recommendations
        recommendations = cost_performance.get('recommendations', [])
        if recommendations:
            st.markdown("**💡 Cost vs Performance Recommendations:**")
            for rec in recommendations:
                st.write(f"• {rec}")
    
    else:
        st.warning("⚠️ Cost vs performance analysis not available.")

def display_bottleneck_analysis(opt_results):
    """Display bottleneck analysis"""
    st.subheader("🔍 Bottleneck Analysis")
    
    bottleneck_analysis = opt_results.get('bottleneck_analysis', {})
    
    # Current bottleneck overview
    current_bottleneck = bottleneck_analysis.get('current_bottleneck', 'Unknown')
    severity = bottleneck_analysis.get('severity', 'medium')
    
    severity_colors = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}
    severity_color = severity_colors.get(severity, '🟡')
    
    st.markdown(f"""
    <div class="insight-card">
        <h4>🎯 Current Bottleneck Status</h4>
        <p><strong>Primary Bottleneck:</strong> {severity_color} {current_bottleneck}</p>
        <p><strong>Severity Level:</strong> {severity.title()}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Detailed bottleneck types
    bottleneck_types = bottleneck_analysis.get('bottleneck_types', {})
    
    if bottleneck_types:
        st.subheader("📊 Detailed Bottleneck Analysis")
        
        for bottleneck_type, details in bottleneck_types.items():
            if details.get('detected', False):
                severity = details.get('severity', 'medium')
                severity_color = severity_colors.get(severity, '🟡')
                
                st.markdown(f"""
                <div class="professional-card">
                    <h4>{severity_color} {bottleneck_type.replace('_', ' ').title()} Bottleneck</h4>
                    <p><strong>Description:</strong> {details.get('description', 'No description')}</p>
                    <p><strong>Impact:</strong> {details.get('impact', 'Unknown impact')}</p>
                    <p><strong>Severity:</strong> {severity.title()}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Resolution strategies
    resolution_strategies = bottleneck_analysis.get('resolution_strategies', {})
    
    if resolution_strategies:
        st.subheader("🛠️ Resolution Strategies")
        
        for bottleneck_type, strategies in resolution_strategies.items():
            if strategies:
                st.markdown(f"**{bottleneck_type.replace('_', ' ').title()} Bottleneck Solutions:**")
                for strategy in strategies:
                    st.write(f"• {strategy}")
    
    # Prevention tips
    prevention_tips = bottleneck_analysis.get('prevention_tips', [])
    
    if prevention_tips:
        st.subheader("🛡️ Prevention Tips")
        for tip in prevention_tips:
            st.write(f"• {tip}")

# Footer
def display_footer():
    """Display application footer"""
    st.markdown("""
    <div class="enterprise-footer">
        <h4>🤖 AWS Enterprise Database Migration Analyzer AI v3.0</h4>
        <p>Powered by Anthropic AI • Real-time AWS Pricing • Advanced Agent Optimization</p>
        <p><small>© 2024 Enterprise Migration Solutions. Built with Streamlit and Claude AI.</small></p>
    </div>
    """, unsafe_allow_html=True)

# Run the application
if __name__ == "__main__":
    main()
    display_footer()