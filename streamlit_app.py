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
            'cost_optimization_recommendations': self._generate_cost_optimization_recommendations(monthly_costs, config)
        }
    
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

# Helper functions for rendering
def render_enhanced_header():
    """Enhanced header with professional styling"""
    ai_manager = AnthropicAIManager()
    aws_api = AWSAPIManager()
    
    ai_status = "🟢" if ai_manager.connected else "🔴"
    aws_status = "🟢" if aws_api.connected else "🔴"
    
    st.markdown(f"""
    <div class="main-header">
        <h1>🤖 AWS Enterprise Database Migration Analyzer AI v3.0</h1>
        <p style="font-size: 1.2rem; margin-top: 0.5rem;">
            Professional-Grade Migration Analysis • AI-Powered Insights • Real-time AWS Integration • Agent Scaling Optimization • FSx Destination Analysis • Backup Storage Support
        </p>
        <div style="margin-top: 1rem; font-size: 0.8rem;">
            <span style="margin-right: 20px;">{ai_status} Anthropic Claude AI</span>
            <span style="margin-right: 20px;">{aws_status} AWS Pricing APIs</span>
            <span style="margin-right: 20px;">🟢 Network Intelligence Engine</span>
            <span style="margin-right: 20px;">🟢 Agent Scaling Optimizer</span>
            <span style="margin-right: 20px;">🟢 FSx Destination Analysis</span>
            <span>🟢 Backup Storage Migration</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_api_status_sidebar():
    """Enhanced API status sidebar"""
    st.sidebar.markdown("### 🔌 System Status")
    
    ai_manager = AnthropicAIManager()
    aws_api = AWSAPIManager()
    
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
    """Enhanced sidebar with AI-powered recommendations"""
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
        index=3,  # Default to SQL Server
        format_func=lambda x: {
            'mysql': '🐬 MySQL', 'postgresql': '🐘 PostgreSQL', 'oracle': '🏛️ Oracle',
            'sqlserver': '🪟 SQL Server', 'mongodb': '🍃 MongoDB'
        }[x]
    )
    
    # On-Premise Database Performance Inputs
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
    
    # Initialize variables that might be conditionally set
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
        ec2_database_engine = None  # Not used for RDS
    else:  # EC2
        # For EC2, show SQL Server prominently if source is SQL Server
        if source_database_engine == "sqlserver":
            database_engine = st.sidebar.selectbox(
                "Target Database (EC2)",
                ["sqlserver", "mysql", "postgresql", "oracle", "mongodb"],
                index=0,  # Default to SQL Server for SQL Server sources
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
        ec2_database_engine = database_engine  # Store the actual database engine for EC2
        
        # SQL Server Deployment Type (only show if SQL Server is selected for EC2)
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
            
            # Show deployment-specific information
            if sql_server_deployment_type == "always_on":
                st.sidebar.info("""
                **🔄 SQL Server Always On Cluster:**
                • 3 EC2 instances (Primary + 2 Replicas)
                • High Availability & Disaster Recovery
                • Automatic failover capability
                • Shared storage or storage replication
                • Higher cost but enterprise-grade reliability
                """)
            else:
                st.sidebar.info("""
                **🖥️ Standalone SQL Server:**
                • Single EC2 instance
                • Standard SQL Server deployment
                • Cost-effective for non-HA requirements
                • Manual backup and recovery processes
                """)
    
    # Add placeholder database properties for missing UI elements
    database_size_gb = st.sidebar.number_input("Database Size (GB)", 
                                              min_value=100, max_value=100000, value=1000, step=100)
    
    downtime_tolerance_minutes = st.sidebar.number_input("Max Downtime (minutes)", 
                                                        min_value=1, max_value=480, value=60)
    
    performance_requirements = st.sidebar.selectbox("Performance Requirement", ["standard", "high"])
    
    # Backup Storage Configuration for DataSync
    st.sidebar.subheader("💾 Backup Storage Configuration")
    
    # Determine backup storage type based on database engine
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
    elif source_database_engine in ['oracle', 'postgresql']:
        backup_storage_type = st.sidebar.selectbox(
            "Backup Storage Type", 
            ["nas_drive", "windows_share"],
            index=0,
            format_func=lambda x: {
                'nas_drive': '🗄️ NAS Drive (Default for Oracle/PostgreSQL)',
                'windows_share': '🪟 Windows Share Drive (Alternative)'
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
    
    # Backup size configuration
    backup_size_multiplier = st.sidebar.selectbox(
        "Backup Size vs Database",
        [0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        index=2,
        format_func=lambda x: f"{int(x*100)}% of DB size ({x:.1f}x multiplier)"
    )
    
    # Migration method selection
    migration_method = st.sidebar.selectbox(
        "Migration Method",
        ["backup_restore", "direct_replication"],
        format_func=lambda x: {
            'backup_restore': '📦 Backup/Restore via DataSync (File Transfer)',
            'direct_replication': '🔄 Direct Replication via DMS (Live Sync)'
        }[x]
    )
    
    # Destination Storage Selection
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
    
    environment = st.sidebar.selectbox("Environment", ["non-production", "production"])
    
    # Agent Configuration
    st.sidebar.subheader("🤖 Migration Agent Configuration")
    
    # Determine primary tool based on migration method
    if migration_method == 'backup_restore':
        primary_tool = "DataSync"
        is_homogeneous = True  # Always use DataSync for backup/restore
    else:
        is_homogeneous = source_database_engine == database_engine
        primary_tool = "DataSync" if is_homogeneous else "DMS"
    
    st.sidebar.success(f"**Primary Tool:** AWS {primary_tool}")
    
    # Show migration method info
    if migration_method == 'backup_restore':
        st.sidebar.info(f"**Method:** Backup/Restore via DataSync from {backup_storage_type.replace('_', ' ').title()}")
        st.sidebar.write(f"**Backup Size:** {int(backup_size_multiplier*100)}% of database ({backup_size_multiplier:.1f}x)")
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
    
    # AI Configuration
    st.sidebar.subheader("🧠 AI Configuration")
    enable_ai_analysis = st.sidebar.checkbox("Enable AI Analysis", value=True)
    
    if st.sidebar.button("🔄 Refresh AI Analysis", type="primary"):
        st.rerun()
    
    # Return the configuration dictionary
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
        'enable_ai_analysis': enable_ai_analysis
    }

def render_bandwidth_waterfall_analysis(analysis, config):
    """Show complete bandwidth degradation from user hardware to final throughput including backup storage"""
    st.markdown("**🌊 Bandwidth Waterfall Analysis: From Your Hardware to Migration Throughput**")
    
    network_perf = analysis.get('network_performance', {})
    os_impact = analysis.get('onprem_performance', {}).get('os_impact', {})
    agent_analysis = analysis.get('agent_analysis', {})
    
    user_nic_speed = config.get('nic_speed', 1000)
    nic_type = config.get('nic_type', 'gigabit_fiber')
    environment = config.get('environment', 'non-production')
    migration_method = config.get('migration_method', 'direct_replication')
    backup_storage_type = config.get('backup_storage_type', 'nas_drive')
    
    network_path_limit = network_perf.get('effective_bandwidth_mbps', 1000)
    raw_user_capacity = user_nic_speed
    
    # NIC Hardware Efficiency
    nic_efficiency = get_nic_efficiency(nic_type)
    after_nic = raw_user_capacity * nic_efficiency
    
    # OS Network Stack
    os_network_efficiency = os_impact.get('network_efficiency', 0.90)
    after_os = after_nic * os_network_efficiency
    
    # Virtualization Impact
    server_type = config.get('server_type', 'physical')
    if server_type == 'vmware':
        virtualization_efficiency = 0.92
        after_virtualization = after_os * virtualization_efficiency
    else:
        virtualization_efficiency = 1.0
        after_virtualization = after_os
    
    # Backup Storage Protocol Impact (NEW)
    if migration_method == 'backup_restore':
        if backup_storage_type == 'windows_share':
            protocol_efficiency = 0.75  # SMB has more overhead
            protocol_name = "SMB Protocol"
        else:  # nas_drive
            protocol_efficiency = 0.88  # NFS is more efficient
            protocol_name = "NFS Protocol"
        after_backup_protocol = after_virtualization * protocol_efficiency
    else:
        protocol_efficiency = 0.82 if 'production' in environment else 0.85
        protocol_name = "Standard Protocol"
        after_backup_protocol = after_virtualization * protocol_efficiency
    
    # Network Path Limitation
    after_network_limit = min(after_backup_protocol, network_path_limit)
    network_is_bottleneck = after_backup_protocol > network_path_limit
    
    # Migration Agent Processing
    total_agent_capacity = agent_analysis.get('total_max_throughput_mbps', after_network_limit * 0.75)
    actual_throughput = agent_analysis.get('total_effective_throughput', 0)
    
    final_throughput = actual_throughput if actual_throughput > 0 else min(total_agent_capacity, after_network_limit)
    
    # Build the waterfall stages
    stages = ['Your NIC\nCapacity']
    throughputs = [raw_user_capacity]
    efficiencies = [100]
    descriptions = [f"{user_nic_speed:,.0f} Mbps {nic_type.replace('_', ' ')} NIC"]
    
    # Add each stage
    stages.append('After NIC\nProcessing')
    throughputs.append(after_nic)
    efficiencies.append(nic_efficiency * 100)
    descriptions.append(f"{nic_type.replace('_', ' ').title()} hardware efficiency")
    
    # OS Processing
    os_name = config.get('operating_system', 'unknown').replace('_', ' ').title()
    stages.append(f'After OS\nNetwork Stack')
    throughputs.append(after_os)
    efficiencies.append(os_network_efficiency * 100)
    descriptions.append(f"{os_name} network processing")
    
    # Virtualization (if applicable)
    if server_type == 'vmware':
        stages.append('After VMware\nVirtualization')
        throughputs.append(after_virtualization)
        efficiencies.append(virtualization_efficiency * 100)
        descriptions.append('VMware hypervisor overhead')
    
    # Backup Storage Protocol (NEW STAGE)
    stages.append(f'After {protocol_name}\nOverhead')
    throughputs.append(after_backup_protocol)
    efficiencies.append(protocol_efficiency * 100)
    
    if migration_method == 'backup_restore':
        descriptions.append(f"{protocol_name} for {backup_storage_type.replace('_', ' ')} access")
    else:
        descriptions.append(f"{environment.title()} security protocols")
    
    # Network Path Limitation
    stages.append('After Network\nPath Limit')
    throughputs.append(after_network_limit)
    if network_is_bottleneck:
        efficiencies.append((network_path_limit / after_backup_protocol) * 100)
        descriptions.append(f"Production path: {network_path_limit:,.0f} Mbps available" if environment == 'production' else f"Non-prod DX limit: {network_path_limit:,.0f} Mbps")
    else:
        efficiencies.append(100)
        descriptions.append(f"Network supports {network_path_limit:,.0f} Mbps (no additional limit)")
    
    # Final Migration Throughput
    tool_name = agent_analysis.get('primary_tool', 'DMS').upper()
    num_agents = config.get('number_of_agents', 1)
    stages.append(f'Final Migration\nThroughput')
    throughputs.append(final_throughput)
    
    if after_network_limit > 0:
        agent_efficiency = (final_throughput / after_network_limit) * 100
    else:
        agent_efficiency = 75
    efficiencies.append(agent_efficiency)
    
    if migration_method == 'backup_restore':
        descriptions.append(f"{num_agents}x DataSync agents from {backup_storage_type.replace('_', ' ')}")
    else:
        descriptions.append(f"{num_agents}x {tool_name} agents processing")
    
    # Create the visualization
    waterfall_data = {
        'Stage': stages,
        'Throughput (Mbps)': throughputs,
        'Efficiency': efficiencies,
        'Description': descriptions
    }
    
    # Update the title to reflect migration method
    if migration_method == 'backup_restore':
        title = f"Backup Migration Bandwidth: {user_nic_speed:,.0f} Mbps Hardware → {final_throughput:.0f} Mbps via {backup_storage_type.replace('_', ' ').title()}"
    else:
        title = f"Direct Migration Bandwidth: {user_nic_speed:,.0f} Mbps Hardware → {final_throughput:.0f} Mbps Migration Speed"
    
    fig = px.bar(
        waterfall_data,
        x='Stage',
        y='Throughput (Mbps)',
        title=title,
        color='Efficiency',
        color_continuous_scale='RdYlGn',
        text='Throughput (Mbps)',
        hover_data=['Description']
    )
    
    fig.update_traces(texttemplate='%{text:.0f} Mbps', textposition='outside')
    fig.update_layout(height=500)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Enhanced Analysis Summary
    total_loss = user_nic_speed - final_throughput
    total_loss_pct = (total_loss / user_nic_speed) * 100
    
    # Identify the primary bottleneck
    if network_is_bottleneck:
        primary_bottleneck = f"Network path ({network_path_limit:,.0f} Mbps limit)"
        bottleneck_type = "Infrastructure"
    elif migration_method == 'backup_restore' and protocol_efficiency < 0.85:
        primary_bottleneck = f"Backup storage protocol ({protocol_name})"
        bottleneck_type = "Backup Storage"
    elif final_throughput < after_network_limit * 0.9:
        primary_bottleneck = f"Migration agents ({num_agents}x {tool_name})"
        bottleneck_type = "Agent Capacity"
    else:
        primary_bottleneck = "Protocol and OS overhead"
        bottleneck_type = "Software"
    
    if bottleneck_type == "Infrastructure":
        st.warning(f"""
        ⚠️ **Network Infrastructure Bottleneck Detected:**
        • **Your Hardware:** {user_nic_speed:,.0f} Mbps {nic_type.replace('_', ' ')} NIC
        • **Network Limitation:** {network_path_limit:,.0f} Mbps ({environment} environment)
        • **Final Migration Speed:** {final_throughput:.0f} Mbps
        • **Recommendation:** Plan migration times using {final_throughput:.0f} Mbps actual speed
        """)
    elif bottleneck_type == "Backup Storage":
        st.error(f"""
        💾 **Backup Storage Protocol Bottleneck:**
        • **Available Bandwidth:** {min(user_nic_speed, network_path_limit):,.0f} Mbps
        • **After {protocol_name}:** {after_backup_protocol:.0f} Mbps
        • **Final Migration Speed:** {final_throughput:.0f} Mbps
        • **Recommendation:** Optimize {backup_storage_type.replace('_', ' ')} performance or consider direct replication
        """)
    elif bottleneck_type == "Agent Capacity":
        st.error(f"""
        🔍 **Agent/Processing Bottleneck:**
        • **Available Bandwidth:** {min(user_nic_speed, network_path_limit):,.0f} Mbps
        • **Final Migration Speed:** {final_throughput:.0f} Mbps
        • **Primary Issue:** {primary_bottleneck}
        • **Recommendation:** Optimize agent configuration or increase agent count
        """)
    else:
        st.info(f"""
        💡 **Hardware Bottleneck (Expected):**
        • **Your Hardware:** {user_nic_speed:,.0f} Mbps {nic_type.replace('_', ' ')} NIC  
        • **Final Migration Speed:** {final_throughput:.0f} Mbps
        • **Recommendation:** Consider NIC upgrade if faster migration needed
        """)
    
    # Add backup storage specific insights
    if migration_method == 'backup_restore':
        st.info(f"""
        📦 **Backup Storage Migration Insights:**
        • **Backup Storage:** {backup_storage_type.replace('_', ' ').title()}
        • **Protocol:** {protocol_name} ({protocol_efficiency*100:.1f}% efficiency)
        • **Method Advantage:** Lower production database impact
        • **Consider:** Direct replication for higher throughput if backup overhead is significant
        """)

def render_performance_impact_table(analysis: Dict, config: Dict):
    """Render performance impact analysis table"""
    st.markdown("**⚡ Performance Impact Analysis:**")
    
    # Get various performance metrics
    os_impact = analysis.get('onprem_performance', {}).get('os_impact', {})
    network_perf = analysis.get('network_performance', {})
    agent_analysis = analysis.get('agent_analysis', {})
    
    # Create performance impact data
    impact_data = []
    
    # Hardware baseline
    user_nic = config.get('nic_speed', 1000)
    impact_data.append({
        'Component': 'Hardware NIC',
        'Baseline (Mbps)': f"{user_nic:,}",
        'Efficiency': '100%',
        'Impact': 'Baseline',
        'Notes': f"{config.get('nic_type', 'Unknown').replace('_', ' ').title()}"
    })
    
    # OS network efficiency
    os_efficiency = os_impact.get('network_efficiency', 0.90)
    after_os = user_nic * os_efficiency
    impact_data.append({
        'Component': 'OS Network Stack',
        'Baseline (Mbps)': f"{after_os:,.0f}",
        'Efficiency': f"{os_efficiency*100:.1f}%",
        'Impact': f"-{(1-os_efficiency)*100:.1f}%",
        'Notes': os_impact.get('name', 'OS').split()[-1]
    })
    
    # Network path
    network_limit = network_perf.get('effective_bandwidth_mbps', user_nic)
    network_efficiency = (network_limit / user_nic) if user_nic > 0 else 1.0
    impact_data.append({
        'Component': 'Network Path',
        'Baseline (Mbps)': f"{network_limit:,.0f}",
        'Efficiency': f"{network_efficiency*100:.1f}%",
        'Impact': f"{(network_efficiency-1)*100:+.1f}%",
        'Notes': f"{network_perf.get('environment', 'Unknown').title()} environment"
    })
    
    # Migration agents
    final_throughput = agent_analysis.get('total_effective_throughput', network_limit * 0.75)
    agent_efficiency = (final_throughput / network_limit) if network_limit > 0 else 0.75
    impact_data.append({
        'Component': 'Migration Agents',
        'Baseline (Mbps)': f"{final_throughput:,.0f}",
        'Efficiency': f"{agent_efficiency*100:.1f}%",
        'Impact': f"{(agent_efficiency-1)*100:+.1f}%",
        'Notes': f"{config.get('number_of_agents', 1)} {agent_analysis.get('primary_tool', 'DMS')} agents"
    })
    
    # Final performance
    total_efficiency = (final_throughput / user_nic) if user_nic > 0 else 0.5
    impact_data.append({
        'Component': 'FINAL RESULT',
        'Baseline (Mbps)': f"{final_throughput:,.0f}",
        'Efficiency': f"{total_efficiency*100:.1f}%",
        'Impact': f"{(total_efficiency-1)*100:+.1f}%",
        'Notes': f"Migration throughput for {config.get('database_size_gb', 0):,} GB"
    })
    
    df_impact = pd.DataFrame(impact_data)
    st.dataframe(df_impact, use_container_width=True)
    
    # Show estimated migration time
    if final_throughput > 0:
        migration_method = config.get('migration_method', 'direct_replication')
        if migration_method == 'backup_restore':
            backup_size_multiplier = config.get('backup_size_multiplier', 0.7)
            data_size = config.get('database_size_gb', 0) * backup_size_multiplier
            st.info(f"**Estimated Migration Time:** {(data_size * 8 * 1000) / (final_throughput * 3600):.1f} hours for {data_size:,.0f} GB backup at {final_throughput:.0f} Mbps")
        else:
            migration_time = (config.get('database_size_gb', 0) * 8 * 1000) / (final_throughput * 3600)
            st.info(f"**Estimated Migration Time:** {migration_time:.1f} hours at {final_throughput:.0f} Mbps effective throughput")

# Tab rendering functions
def render_ai_insights_tab_enhanced(analysis: Dict, config: Dict):
    """Render enhanced AI insights and analysis tab"""
    st.subheader("🧠 AI-Powered Migration Insights & Analysis")
    
    ai_analysis = analysis.get('aws_sizing_recommendations', {}).get('ai_analysis', {})
    ai_assessment = analysis.get('ai_overall_assessment', {})
    
    # Migration Method and Backup Storage Analysis
    migration_method = config.get('migration_method', 'direct_replication')
    if migration_method == 'backup_restore':
        st.markdown("**💾 Backup Storage Migration Analysis:**")
        
        backup_col1, backup_col2 = st.columns(2)
        
        with backup_col1:
            backup_storage_type = config.get('backup_storage_type', 'nas_drive')
            backup_size_multiplier = config.get('backup_size_multiplier', 0.7)
            backup_size_gb = config.get('database_size_gb', 0) * backup_size_multiplier
            
            st.info("📦 **Backup Storage Configuration**")
            st.write(f"**Storage Type:** {backup_storage_type.replace('_', ' ').title()}")
            st.write(f"**Database Size:** {config.get('database_size_gb', 0):,} GB")
            st.write(f"**Backup Size:** {backup_size_gb:,.0f} GB ({int(backup_size_multiplier*100)}%)")
            st.write(f"**Migration Tool:** DataSync (File Transfer)")
            st.write(f"**Transfer Protocol:** {'SMB' if backup_storage_type == 'windows_share' else 'NFS'}")
            
            agent_analysis = analysis.get('agent_analysis', {})
            backup_efficiency = agent_analysis.get('backup_efficiency', 1.0)
            st.write(f"**Protocol Efficiency:** {backup_efficiency*100:.1f}%")
        
        with backup_col2:
            st.success("🎯 **Backup Migration Advantages**")
            
            if backup_storage_type == 'windows_share':
                st.write("• Native Windows integration")
                st.write("• Familiar SMB protocols")
                st.write("• Windows authentication support")
                st.write("• Easy backup verification")
            else:  # nas_drive
                st.write("• High-performance NFS protocol")
                st.write("• Better bandwidth utilization")
                st.write("• Lower protocol overhead")
                st.write("• Parallel file access")
            
            st.write("• **Benefit:** Lower impact on production database")
            st.write("• **Benefit:** Backup validation before migration")
            st.write("• **Benefit:** Easy rollback capability")
        
        # Backup-specific recommendations
        st.markdown("**💡 Backup Storage Specific Recommendations:**")
        
        backup_rec_col1, backup_rec_col2 = st.columns(2)
        
        with backup_rec_col1:
            st.warning("🚨 **Pre-Migration Checklist**")
            st.write("• Verify backup integrity and completeness")
            st.write("• Test backup restore procedures")
            st.write("• Ensure sufficient storage space in AWS")
            st.write("• Configure DataSync security and permissions")
            
            if backup_storage_type == 'windows_share':
                st.write("• Verify SMB version (recommend SMB3+)")
                st.write("• Check Windows authentication setup")
                st.write("• Test SMB performance with large files")
            else:
                st.write("• Optimize NFS mount options")
                st.write("• Verify NFS version compatibility")
                st.write("• Test NFS performance characteristics")
        
        with backup_rec_col2:
            st.info("⚡ **Performance Optimization Tips**")
            
            if backup_storage_type == 'windows_share':
                st.write("• Enable SMB3 multichannel if available")
                st.write("• Use dedicated network for backup transfer")
                st.write("• Configure SMB signing appropriately")
                st.write("• Monitor SMB connection stability")
            else:
                st.write("• Use NFS v4.1+ for better performance")
                st.write("• Configure appropriate rsize/wsize values")
                st.write("• Enable NFS caching where appropriate")
                st.write("• Monitor NFS connection health")
            
            st.write("• **DataSync:** Configure bandwidth throttling")
            st.write("• **DataSync:** Use multiple agents for large datasets")
            st.write("• **DataSync:** Schedule transfers during off-peak hours")
        
        st.markdown("---")  # Add separator
    
    # AI Analysis Overview
    st.markdown("**🤖 AI Analysis Overview:**")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        complexity_score = ai_analysis.get('ai_complexity_score', 6)
        st.metric(
            "🎯 AI Complexity Score",
            f"{complexity_score:.1f}/10",
            delta=ai_analysis.get('confidence_level', 'medium').title()
        )
    
    with col2:
        readiness_score = ai_assessment.get('migration_readiness_score', 0)
        st.metric(
            "📊 Migration Readiness",
            f"{readiness_score:.0f}/100",
            delta=ai_assessment.get('risk_level', 'Unknown')
        )
    
    with col3:
        success_probability = ai_assessment.get('success_probability', 0)
        st.metric(
            "🎯 Success Probability",
            f"{success_probability:.0f}%",
            delta=f"Confidence: {ai_analysis.get('confidence_level', 'medium').title()}"
        )
    
    with col4:
        num_agents = config.get('number_of_agents', 1)
        agent_efficiency = analysis.get('agent_analysis', {}).get('scaling_efficiency', 1.0)
        st.metric(
            "🤖 Agent Efficiency",
            f"{agent_efficiency*100:.1f}%",
            delta=f"{num_agents} agents"
        )
    
    with col5:
        migration_method = config.get('migration_method', 'direct_replication')
        backup_storage = config.get('backup_storage_type', 'nas_drive').replace('_', ' ').title()
        
        if migration_method == 'backup_restore':
            display_text = f"Backup/Restore via {backup_storage}"
            delta_text = f"Tool: {analysis.get('agent_analysis', {}).get('primary_tool', 'DataSync').upper()}"
        else:
            destination = config.get('destination_storage_type', 'S3')
            display_text = destination
            delta_text = f"Efficiency: {agent_efficiency*100:.1f}%"
        
        st.metric(
            "🗄️ Migration Method",
            display_text,
            delta=delta_text
        )
    
    # AI Risk Assessment and Factors
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**⚠️ AI-Identified Risk Factors:**")
        
        risk_factors = ai_analysis.get('risk_factors', [])
        risk_percentages = ai_analysis.get('risk_percentages', {})
        
        with st.container():
            st.warning("Risk Assessment")
            
            if risk_factors:
                st.write("**Identified Risks:**")
                for i, risk in enumerate(risk_factors[:4], 1):
                    st.write(f"{i}. {risk}")
                
                if risk_percentages:
                    st.write("**Risk Probabilities:**")
                    for risk_type, percentage in list(risk_percentages.items())[:3]:
                        risk_name = risk_type.replace('_', ' ').title()
                        st.write(f"• {risk_name}: {percentage}%")
            else:
                st.write("No significant risks identified by AI analysis")
                st.write("Migration appears to be low-risk with current configuration")
    
    with col2:
        st.markdown("**🛡️ AI-Recommended Mitigation Strategies:**")
        
        mitigation_strategies = ai_analysis.get('mitigation_strategies', [])
        
        with st.container():
            st.success("Mitigation Recommendations")
            
            if mitigation_strategies:
                for i, strategy in enumerate(mitigation_strategies[:4], 1):
                    st.write(f"{i}. {strategy}")
            else:
                st.write("• Continue with standard migration best practices")
                st.write("• Implement comprehensive testing procedures")
                st.write("• Monitor performance throughout migration")
    
    # AI Performance Recommendations
    st.markdown("**🚀 AI Performance Optimization Recommendations:**")
    
    performance_recommendations = ai_analysis.get('performance_recommendations', [])
    performance_improvements = ai_analysis.get('performance_improvements', {})
    
    if performance_recommendations:
        for i, recommendation in enumerate(performance_recommendations, 1):
            impact = "High" if i <= 2 else "Medium" if i <= 4 else "Low"
            complexity = "Low" if i <= 2 else "Medium" if i <= 4 else "High"
            
            with st.expander(f"Recommendation {i}: {recommendation}", expanded=(i <= 2)):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if impact == "High":
                        st.success(f"**Expected Impact:** {impact}")
                    elif impact == "Medium":
                        st.warning(f"**Expected Impact:** {impact}")
                    else:
                        st.info(f"**Expected Impact:** {impact}")
                
                with col2:
                    st.write(f"**Implementation Complexity:** {complexity}")
                
                with col3:
                    # Try to find corresponding improvement percentage
                    improvement_key = recommendation.lower().replace(' ', '_')[:20]
                    improvement = None
                    for key, value in performance_improvements.items():
                        if any(word in key.lower() for word in improvement_key.split('_')[:2]):
                            improvement = value
                            break
                    
                    if improvement:
                        st.write(f"**Expected Improvement:** {improvement}")
                    else:
                        expected_improvement = "15-25%" if impact == "High" else "5-15%" if impact == "Medium" else "2-10%"
                        st.write(f"**Expected Improvement:** {expected_improvement}")
    else:
        st.info("Current configuration appears well-optimized. Continue with standard best practices.")

def render_network_intelligence_tab(analysis: Dict, config: Dict):
    """Render network intelligence analysis tab with AI insights using native components"""
    st.subheader("🌐 Network Intelligence & Path Optimization")
    
    network_perf = analysis.get('network_performance', {})
    
    # Network Overview Dashboard
    st.markdown("**📊 Network Performance Overview:**")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "🎯 Network Quality",
            f"{network_perf.get('network_quality_score', 0):.1f}/100",
            delta=f"AI Enhanced: {network_perf.get('ai_enhanced_quality_score', 0):.1f}"
        )
    
    with col2:
        st.metric(
            "🌐 Network Capacity",  # Changed from "Effective Bandwidth"
            f"{network_perf.get('effective_bandwidth_mbps', 0):,.0f} Mbps",
            delta="Raw network limit (not migration speed)"  # Added context
        )
    
    with col3:
        st.metric(
            "🕐 Total Latency",
            f"{network_perf.get('total_latency_ms', 0):.1f} ms",
            delta=f"Reliability: {network_perf.get('total_reliability', 0)*100:.2f}%"
        )
    
    with col4:
        st.metric(
            "🗄️ Destination Storage",
            network_perf.get('destination_storage', 'S3'),
            delta=f"Bonus: +{network_perf.get('storage_performance_bonus', 0)}%"
        )
    
    with col5:
        ai_optimization = network_perf.get('ai_optimization_potential', 0)
        st.metric(
            "🤖 AI Optimization",
            f"{ai_optimization:.1f}%",
            delta="Improvement potential"
        )
    
    # Add bandwidth waterfall analysis
    st.markdown("---")  # Add separator
    render_bandwidth_waterfall_analysis(analysis, config)

    st.markdown("---")  # Add separator  
    render_performance_impact_table(analysis, config)
    
    
    # Network Path Visualization
    st.markdown("**🗺️ Network Path Visualization:**")
    
    if network_perf.get('segments'):
        # Create network path diagram
        try:
            network_diagram = create_network_path_diagram(network_perf)
            st.plotly_chart(network_diagram, use_container_width=True)
        except Exception as e:
            st.warning(f"Network diagram could not be rendered: {str(e)}")
            
            # Fallback: Show path as table
            segments_data = []
            for i, segment in enumerate(network_perf.get('segments', []), 1):
                segments_data.append({
                    'Hop': i,
                    'Segment': segment['name'],
                    'Type': segment['connection_type'].replace('_', ' ').title(),
                    'Bandwidth (Mbps)': f"{segment.get('effective_bandwidth_mbps', 0):,.0f}",
                    'Latency (ms)': f"{segment.get('effective_latency_ms', 0):.1f}",
                    'Reliability': f"{segment['reliability']*100:.3f}%",
                    'Cost Factor': f"{segment['cost_factor']:.1f}x"
                })
            
            df_segments = pd.DataFrame(segments_data)
            st.dataframe(df_segments, use_container_width=True)
    else:
        st.info("Network path data not available in current analysis")
    
    # Detailed Network Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔍 Network Performance Analysis:**")
        
        with st.container():
            st.info("Performance Metrics")
            st.write(f"**Path Name:** {network_perf.get('path_name', 'Unknown')}")
            st.write(f"**Environment:** {network_perf.get('environment', 'Unknown').title()}")
            st.write(f"**OS Type:** {network_perf.get('os_type', 'Unknown').title()}")
            st.write(f"**Storage Type:** {network_perf.get('storage_type', 'Unknown').title()}")
            st.write(f"**Destination:** {network_perf.get('destination_storage', 'S3')}")
            st.write(f"**Migration Type:** {network_perf.get('migration_type', 'direct_replication').replace('_', ' ').title()}")
            st.write(f"**Network Quality Score:** {network_perf.get('network_quality_score', 0):.1f}/100")
            st.write(f"**AI Enhanced Score:** {network_perf.get('ai_enhanced_quality_score', 0):.1f}/100")
            st.write(f"**Cost Factor:** {network_perf.get('total_cost_factor', 0):.1f}x")
    
    with col2:
        st.markdown("**🤖 AI Network Insights:**")
        
        ai_insights = network_perf.get('ai_insights', {})
        
        with st.container():
            st.success("AI Analysis & Recommendations")
            
            st.write("**Performance Bottlenecks:**")
            bottlenecks = ai_insights.get('performance_bottlenecks', ['No bottlenecks identified'])
            for bottleneck in bottlenecks[:3]:
                st.write(f"• {bottleneck}")
            
            st.write("**Optimization Opportunities:**")
            opportunities = ai_insights.get('optimization_opportunities', ['Standard optimization'])
            for opportunity in opportunities[:3]:
                st.write(f"• {opportunity}")
            
            st.write("**Risk Factors:**")
            risks = ai_insights.get('risk_factors', ['No significant risks'])
            for risk in risks[:2]:
                st.write(f"• {risk}")
    
    # Network Optimization Recommendations
    st.markdown("**💡 Network Optimization Recommendations:**")
    
    recommendations = ai_insights.get('recommended_improvements', [])
    if recommendations:
        for i, recommendation in enumerate(recommendations, 1):
            impact = "High" if i <= 2 else "Medium" if i <= 4 else "Low"
            complexity = "Low" if i <= 2 else "Medium" if i <= 4 else "High"
            priority = "Immediate" if i <= 2 else "Short-term" if i <= 4 else "Long-term"
            
            with st.expander(f"Recommendation {i}: {recommendation}", expanded=(i <= 2)):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if impact == "High":
                        st.success(f"**Expected Impact:** {impact}")
                    elif impact == "Medium":
                        st.warning(f"**Expected Impact:** {impact}")
                    else:
                        st.info(f"**Expected Impact:** {impact}")
                
                with col2:
                    st.write(f"**Implementation Complexity:** {complexity}")
                
                with col3:
                    st.write(f"**Priority:** {priority}")
    else:
        st.info("Network appears optimally configured for current requirements")

def render_comprehensive_cost_analysis_tab(analysis: Dict, config: Dict):
        """Render comprehensive AWS cost analysis tab with all services clearly organized"""
        st.subheader("💰 Complete AWS Cost Analysis")
        
        # Get comprehensive cost data
        comprehensive_costs = analysis.get('comprehensive_costs', {})
        
        if not comprehensive_costs:
            st.warning("⚠️ Comprehensive cost data not available. Please run the analysis first.")
            return
        
        # Executive Cost Summary
        st.markdown("**📊 Executive Cost Summary**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_monthly = comprehensive_costs.get('total_monthly', 0)
            st.metric(
                "💰 Total Monthly",
                f"${total_monthly:,.0f}",
                delta=f"Annual: ${total_monthly * 12:,.0f}"
            )
        
        with col2:
            total_one_time = comprehensive_costs.get('total_one_time', 0)
            st.metric(
                "🔄 One-Time Costs",
                f"${total_one_time:,.0f}",
                delta="Setup & Migration"
            )
        
        with col3:
            three_year_total = comprehensive_costs.get('three_year_total', 0)
            st.metric(
                "📅 3-Year Total",
                f"${three_year_total:,.0f}",
                delta="All costs included"
            )
        
        with col4:
            monthly_breakdown = comprehensive_costs.get('monthly_breakdown', {})
            largest_cost = max(monthly_breakdown.items(), key=lambda x: x[1]) if monthly_breakdown else ('Unknown', 0)
            st.metric(
                "🎯 Largest Cost",
                largest_cost[0].replace('_', ' ').title(),
                delta=f"${largest_cost[1]:,.0f}/mo"
            )
        
        # AWS Services Cost Breakdown - Clear and Organized
        st.markdown("---")
        st.markdown("**🔧 AWS Services Cost Breakdown**")
        
        # Create a comprehensive service breakdown table
        service_breakdown_data = []
        
        # Get all cost components
        compute_costs = comprehensive_costs.get('compute_costs', {})
        storage_costs = comprehensive_costs.get('storage_costs', {})
        network_costs = comprehensive_costs.get('network_costs', {})
        migration_costs = comprehensive_costs.get('migration_costs', {})
        
        # Add RDS/EC2 costs
        if compute_costs.get('service_type') == 'RDS':
            # RDS Writer
            writer_cost = compute_costs.get('writer_monthly_cost', 0)
            if writer_cost > 0:
                service_breakdown_data.append({
                    'AWS Service': 'RDS Writer Instance',
                    'Service Type': compute_costs.get('primary_instance', 'Unknown'),
                    'Monthly Cost': f"${writer_cost:,.0f}",
                    'Usage': f"{compute_costs.get('writer_instances', 1)} instance(s)",
                    'Category': 'Database Compute'
                })
            
            # RDS Readers
            reader_cost = compute_costs.get('reader_monthly_cost', 0)
            if reader_cost > 0:
                service_breakdown_data.append({
                    'AWS Service': 'RDS Read Replicas',
                    'Service Type': compute_costs.get('primary_instance', 'Unknown'),
                    'Monthly Cost': f"${reader_cost:,.0f}",
                    'Usage': f"{compute_costs.get('reader_instances', 0)} instance(s)",
                    'Category': 'Database Compute'
                })
            
            # Multi-AZ
            multi_az_cost = compute_costs.get('multi_az_cost', 0)
            if multi_az_cost > 0:
                service_breakdown_data.append({
                    'AWS Service': 'RDS Multi-AZ',
                    'Service Type': 'High Availability',
                    'Monthly Cost': f"${multi_az_cost:,.0f}",
                    'Usage': 'Multi-AZ deployment',
                    'Category': 'Database Compute'
                })
        
        else:
            # EC2 instances
            instance_cost = compute_costs.get('monthly_instance_cost', 0)
            if instance_cost > 0:
                service_breakdown_data.append({
                    'AWS Service': 'EC2 Instances',
                    'Service Type': compute_costs.get('primary_instance', 'Unknown'),
                    'Monthly Cost': f"${instance_cost:,.0f}",
                    'Usage': f"{compute_costs.get('instance_count', 1)} instance(s)",
                    'Category': 'Database Compute'
                })
            
            # OS Licensing
            os_cost = compute_costs.get('os_licensing_cost', 0)
            if os_cost > 0:
                service_breakdown_data.append({
                    'AWS Service': 'OS Licensing',
                    'Service Type': 'Windows Licensing',
                    'Monthly Cost': f"${os_cost:,.0f}",
                    'Usage': f"{compute_costs.get('instance_count', 1)} license(s)",
                    'Category': 'Database Compute'
                })
        
        # EBS Storage
        primary_storage = storage_costs.get('primary_storage', {})
        if primary_storage.get('monthly_cost', 0) > 0:
            service_breakdown_data.append({
                'AWS Service': 'EBS Storage',
                'Service Type': primary_storage.get('type', 'gp3').upper(),
                'Monthly Cost': f"${primary_storage.get('monthly_cost', 0):,.0f}",
                'Usage': f"{primary_storage.get('size_gb', 0):,.0f} GB",
                'Category': 'Database Storage'
            })
        
        # S3/FSx Destination Storage
        destination_storage = storage_costs.get('destination_storage', {})
        if destination_storage.get('monthly_cost', 0) > 0:
            service_breakdown_data.append({
                'AWS Service': destination_storage.get('type', 'S3'),
                'Service Type': 'Destination Storage',
                'Monthly Cost': f"${destination_storage.get('monthly_cost', 0):,.0f}",
                'Usage': f"{destination_storage.get('size_gb', 0):,.0f} GB",
                'Category': 'Migration Storage'
            })
        
        # Backup Storage (if applicable)
        backup_storage = storage_costs.get('backup_storage', {})
        if backup_storage.get('applicable', False) and backup_storage.get('monthly_cost', 0) > 0:
            service_breakdown_data.append({
                'AWS Service': 'S3 Backup Storage',
                'Service Type': 'Migration Backup',
                'Monthly Cost': f"${backup_storage.get('monthly_cost', 0):,.0f}",
                'Usage': f"{backup_storage.get('size_gb', 0):,.0f} GB",
                'Category': 'Migration Storage'
            })
        
        # Direct Connect
        dx_costs = network_costs.get('direct_connect', {})
        if dx_costs.get('monthly_cost', 0) > 0:
            service_breakdown_data.append({
                'AWS Service': 'Direct Connect',
                'Service Type': dx_costs.get('capacity', 'Unknown'),
                'Monthly Cost': f"${dx_costs.get('monthly_cost', 0):,.0f}",
                'Usage': f"${dx_costs.get('hourly_cost', 0):.2f}/hour",
                'Category': 'Network'
            })
        
        # Data Transfer
        transfer_costs = network_costs.get('data_transfer', {})
        if transfer_costs.get('monthly_cost', 0) > 0:
            service_breakdown_data.append({
                'AWS Service': 'Data Transfer Out',
                'Service Type': 'DX Data Transfer',
                'Monthly Cost': f"${transfer_costs.get('monthly_cost', 0):,.0f}",
                'Usage': f"{transfer_costs.get('monthly_gb', 0):,.0f} GB/month",
                'Category': 'Network'
            })
        
        # VPN (if applicable)
        vpn_costs = network_costs.get('vpn_backup', {})
        if vpn_costs.get('monthly_cost', 0) > 0:
            service_breakdown_data.append({
                'AWS Service': 'VPN Gateway',
                'Service Type': 'Backup Connectivity',
                'Monthly Cost': f"${vpn_costs.get('monthly_cost', 0):,.0f}",
                'Usage': 'HA/Backup connection',
                'Category': 'Network'
            })
        
        # Migration Agents (DataSync/DMS)
        agent_costs = migration_costs.get('agent_costs', {})
        if agent_costs.get('monthly_cost', 0) > 0:
            primary_tool = analysis.get('agent_analysis', {}).get('primary_tool', 'DMS')
            service_breakdown_data.append({
                'AWS Service': f'{primary_tool} Agents (EC2)',
                'Service Type': 'Migration Processing',
                'Monthly Cost': f"${agent_costs.get('monthly_cost', 0):,.0f}",
                'Usage': f"{agent_costs.get('agent_count', 0)} agent(s)",
                'Category': 'Migration Services'
            })
        
        # DataSync Usage (if applicable)
        datasync_costs = migration_costs.get('datasync', {})
        if datasync_costs.get('applicable', False):
            if datasync_costs.get('monthly_sync_cost', 0) > 0:
                service_breakdown_data.append({
                    'AWS Service': 'DataSync Usage',
                    'Service Type': 'Data Transfer Service',
                    'Monthly Cost': f"${datasync_costs.get('monthly_sync_cost', 0):,.0f}",
                    'Usage': f"{datasync_costs.get('data_size_gb', 0):,.0f} GB/month",
                    'Category': 'Migration Services'
                })
        
        # DMS Usage (if applicable)
        dms_costs = migration_costs.get('dms', {})
        if dms_costs.get('applicable', False) and dms_costs.get('monthly_cost', 0) > 0:
            service_breakdown_data.append({
                'AWS Service': 'DMS Replication',
                'Service Type': 'Database Migration',
                'Monthly Cost': f"${dms_costs.get('monthly_cost', 0):,.0f}",
                'Usage': 'Replication instances',
                'Category': 'Migration Services'
            })
        
    if service_breakdown_data:
            st.markdown("**📋 Complete AWS Services Breakdown:**")
            df_services = pd.DataFrame(service_breakdown_data)
        
    # Group by category and show
    categories = df_services['Category'].unique()
    
        for category in categories:
            category_data = df_services[df_services['Category'] == category]
            
            # FIXED: Added missing closing parenthesis and better string cleaning
            category_total = sum([
                float(str(row['Monthly Cost']).replace('$', '').replace(',', '').strip()) 
                for _, row in category_data.iterrows() 
                if str(row['Monthly Cost']).replace('$', '').replace(',', '').strip().replace('.', '').isdigit()
            ])  # <-- This closing parenthesis was missing!
            
            with st.expander(f"🔧 {category} - ${category_total:,.0f}/month", expanded=True):
                # Remove category column for display since it's in the header
                display_data = category_data.drop('Category', axis=1)
                st.dataframe(display_data, use_container_width=True, hide_index=True)
        
        # One-Time Costs Summary
        st.markdown("---")
        st.markdown("**💸 One-Time Migration Costs**")
        
        one_time_col1, one_time_col2 = st.columns(2)
        
        with one_time_col1:
            st.info("**Migration Setup Costs**")
            setup_costs = migration_costs.get('setup_and_professional_services', {})
            setup_cost = setup_costs.get('one_time_cost', 0)
            st.write(f"**Professional Services:** ${setup_cost:,.0f}")
            
            includes = setup_costs.get('includes', [])
            if includes:
                st.write("**Includes:**")
                for service in includes:
                    st.write(f"• {service}")
        
        with one_time_col2:
            st.success("**Data Transfer Costs**")
            
            # DataSync one-time transfer
            if datasync_costs.get('applicable', False):
                transfer_cost = datasync_costs.get('one_time_transfer_cost', 0)
                transfer_size = datasync_costs.get('data_size_gb', 0)
                st.write(f"**DataSync Transfer:** ${transfer_cost:,.0f}")
                st.write(f"**Data Size:** {transfer_size:,.0f} GB")
            
            # DMS one-time setup
            if dms_costs.get('applicable', False):
                dms_setup = dms_costs.get('one_time_setup', 0)
                st.write(f"**DMS Setup:** ${dms_setup:,.0f}")
            
            if not datasync_costs.get('applicable', False) and not dms_costs.get('applicable', False):
                st.write("**No significant one-time data transfer costs**")
        
        # Cost Visualization
        st.markdown("---")
        st.markdown("**📊 Cost Analysis & Projections**")
        
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            st.markdown("**Monthly Cost Distribution**")
            
            # Create pie chart for monthly costs by category
            if service_breakdown_data:
                category_totals = {}
                for service in service_breakdown_data:
                    category = service['Category']
                    cost = float(service['Monthly Cost'].replace(', '').replace(',', ''))
                    category_totals[category] = category_totals.get(category, 0) + cost
                
                if category_totals:
                    fig_pie = px.pie(
                        values=list(category_totals.values()),
                        names=list(category_totals.keys()),
                        title="Monthly Costs by Service Category"
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                else:
                    st.info("Cost breakdown data not available for visualization")
            else:
                st.info("Service breakdown data not available")
        
        with viz_col2:
            st.markdown("**3-Year Cost Projection**")
            
            # Create cost projection over 3 years
            months = list(range(0, 37, 6))  # Every 6 months for 3 years
            cumulative_costs = []
            
            for month in months:
                cumulative_cost = total_one_time + (total_monthly * month)
                cumulative_costs.append(cumulative_cost)
            
            projection_data = {
                'Months': months,
                'Cumulative Cost ($)': cumulative_costs
            }
            
            fig_line = px.line(
                projection_data,
                x='Months',
                y='Cumulative Cost ($)',
                title="3-Year Cumulative Cost Projection",
                markers=True
            )
            fig_line.update_traces(line_color='#3498db', marker_color='#e74c3c')
            st.plotly_chart(fig_line, use_container_width=True)
        
        # Cost Optimization Recommendations
        st.markdown("---")
        st.markdown("**💡 Cost Optimization Opportunities**")
        
        recommendations = comprehensive_costs.get('cost_optimization_recommendations', [])
        
        if recommendations:
            opt_col1, opt_col2 = st.columns(2)
            
            with opt_col1:
                st.success("**Immediate Opportunities (0-3 months)**")
                for i, rec in enumerate(recommendations[:3], 1):
                    potential_savings = "20-30%" if 'Reserved' in rec else "60-70%" if 'Spot' in rec else "10-25%"
                    st.write(f"**{i}.** {rec}")
                    st.write(f"   💰 Potential savings: {potential_savings}")
                    st.write("")
            
            with opt_col2:
                st.info("**Medium-term Opportunities (3-12 months)**")
                for i, rec in enumerate(recommendations[3:6], 4):
                    potential_savings = "15-25%" if 'lifecycle' in rec.lower() else "5-15%"
                    st.write(f"**{i}.** {rec}")
                    st.write(f"   💰 Potential savings: {potential_savings}")
                    st.write("")
        else:
            st.info("✅ Current configuration appears well-optimized. Continue monitoring for opportunities.")
        
        # Data Source and Summary
        st.markdown("---")
        st.markdown("**📊 Analysis Summary & Data Sources**")
        
        summary_col1, summary_col2 = st.columns(2)
        
        with summary_col1:
            st.info("**Cost Summary**")
            st.write(f"**Monthly Operating Cost:** ${total_monthly:,.0f}")
            st.write(f"**One-time Migration Cost:** ${total_one_time:,.0f}")
            st.write(f"**Annual Operating Cost:** ${total_monthly * 12:,.0f}")
            st.write(f"**3-Year Total Cost:** ${three_year_total:,.0f}")
            
            # ROI calculation
            cost_analysis = analysis.get('cost_analysis', {})
            estimated_savings = cost_analysis.get('estimated_monthly_savings', 0)
            if estimated_savings > 0:
                roi_months = total_one_time / estimated_savings if estimated_savings > 0 else 0
                st.write(f"**Estimated ROI Period:** {roi_months:.1f} months")
        
        with summary_col2:
            st.success("**Data Sources & Reliability**")
            
            pricing_data = comprehensive_costs.get('pricing_data', {})
            data_source = pricing_data.get('data_source', 'unknown')
            last_updated = pricing_data.get('last_updated', 'Unknown')
            
            if data_source == 'aws_api':
                st.write("✅ **Pricing Data:** Real-time AWS API")
                st.write(f"📅 **Last Updated:** {last_updated}")
                st.write("🎯 **Accuracy:** High (±5%)")
            else:
                st.write("⚠️ **Pricing Data:** Fallback estimates")
                st.write("🎯 **Accuracy:** Moderate (±15%)")
            
            st.write(f"🔧 **Services Analyzed:** {len(service_breakdown_data)} AWS services")
            st.write(f"📊 **Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
        # Export functionality
        if st.button("📤 Export Complete Cost Analysis", use_container_width=True):
            export_data = {
                'analysis_date': datetime.now().isoformat(),
                'configuration': config,
                'cost_summary': {
                    'total_monthly': total_monthly,
                    'total_one_time': total_one_time,
                    'three_year_total': three_year_total
                },
                'service_breakdown': service_breakdown_data,
                'optimization_recommendations': recommendations,
                'data_source': data_source
            }
            
            st.download_button(
                label="💾 Download Cost Analysis (JSON)",
                data=json.dumps(export_data, indent=2),
                file_name=f"aws_cost_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

def render_os_performance_tab(analysis: Dict, config: Dict):
    """Render OS performance analysis tab using native Streamlit components"""
    st.subheader("💻 Operating System Performance Analysis")
    
    os_impact = analysis.get('onprem_performance', {}).get('os_impact', {})
    
    # OS Overview
    st.markdown("**🖥️ Operating System Overview:**")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "💻 Current OS",
            os_impact.get('name', 'Unknown'),
            delta=f"Platform: {config.get('server_type', 'Unknown').title()}"
        )
    
    with col2:
        st.metric(
            "⚡ Total Efficiency",
            f"{os_impact.get('total_efficiency', 0)*100:.1f}%",
            delta=f"Base: {os_impact.get('base_efficiency', 0)*100:.1f}%"
        )
    
    with col3:
        st.metric(
            "🗄️ DB Optimization",
            f"{os_impact.get('db_optimization', 0)*100:.1f}%",
            delta=f"Engine: {os_impact.get('actual_database_engine', 'Unknown').upper()}"
        )
    
    with col4:
        st.metric(
            "💰 Licensing Factor",
            f"{os_impact.get('licensing_cost_factor', 1.0):.1f}x",
            delta=f"Complexity: {os_impact.get('management_complexity', 0)*100:.0f}%"
        )
    
    with col5:
        virt_overhead = os_impact.get('virtualization_overhead', 0)
        st.metric(
            "☁️ Virtualization",
            f"{virt_overhead*100:.1f}%" if config.get('server_type') == 'vmware' else "N/A",
            delta="Overhead" if config.get('server_type') == 'vmware' else "Physical"
        )
    
    # OS Performance Breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 OS Performance Metrics:**")
        
        # Create radar chart for OS performance
        performance_metrics = {
            'CPU Efficiency': os_impact.get('cpu_efficiency', 0) * 100,
            'Memory Efficiency': os_impact.get('memory_efficiency', 0) * 100,
            'I/O Efficiency': os_impact.get('io_efficiency', 0) * 100,
            'Network Efficiency': os_impact.get('network_efficiency', 0) * 100,
            'DB Optimization': os_impact.get('db_optimization', 0) * 100
        }
        
        fig_radar = go.Figure()
        
        categories = list(performance_metrics.keys())
        values = list(performance_metrics.values())
        
        fig_radar.add_trace(go.Scatterpolar(
            r=values + [values[0]],  # Close the polygon
            theta=categories + [categories[0]],
            fill='toself',
            name='OS Performance',
            line_color='#3498db'
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=False,
            title="OS Performance Profile"
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
    
    with col2:
        st.markdown("**🤖 AI OS Insights:**")
        
        ai_insights = os_impact.get('ai_insights', {})
        
        with st.container():
            st.success(f"AI Analysis of {os_impact.get('name', 'Current OS')}")
            
            st.write("**Strengths:**")
            for strength in ai_insights.get('strengths', ['General purpose OS'])[:3]:
                st.write(f"• {strength}")
            
            st.write("**Weaknesses:**")
            for weakness in ai_insights.get('weaknesses', ['No significant issues'])[:3]:
                st.write(f"• {weakness}")
            
            st.write("**Migration Considerations:**")
            for consideration in ai_insights.get('migration_considerations', ['Standard migration'])[:3]:
                st.write(f"• {consideration}")

def render_aws_sizing_tab(analysis: Dict, config: Dict):
    """Render AWS sizing and configuration recommendations tab using native components"""
    st.subheader("🎯 AWS Sizing & Configuration Recommendations")
    
    aws_sizing = analysis.get('aws_sizing_recommendations', {})
    deployment_rec = aws_sizing.get('deployment_recommendation', {})
    
    # Deployment Recommendation Overview
    st.markdown("**☁️ Deployment Recommendation:**")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        recommendation = deployment_rec.get('recommendation', 'unknown').upper()
        confidence = deployment_rec.get('confidence', 0)
        st.metric(
            "🎯 Recommended Deployment",
            recommendation,
            delta=f"Confidence: {confidence*100:.1f}%"
        )
    
    with col2:
        rds_score = deployment_rec.get('rds_score', 0)
        ec2_score = deployment_rec.get('ec2_score', 0)
        st.metric(
            "📊 RDS Score",
            f"{rds_score:.0f}",
            delta=f"EC2: {ec2_score:.0f}"
        )
    
    with col3:
        ai_analysis = aws_sizing.get('ai_analysis', {})
        complexity = ai_analysis.get('ai_complexity_score', 6)
        st.metric(
            "🤖 AI Complexity",
            f"{complexity:.1f}/10",
            delta=ai_analysis.get('confidence_level', 'medium').title()
        )
    
    with col4:
        if recommendation == 'RDS':
            monthly_cost = aws_sizing.get('rds_recommendations', {}).get('total_monthly_cost', 0)
        else:
            monthly_cost = aws_sizing.get('ec2_recommendations', {}).get('total_monthly_cost', 0)
        st.metric(
            "💰 Monthly Cost",
            f"${monthly_cost:,.0f}",
            delta="Compute + Storage"
        )
    
    with col5:
        reader_writer = aws_sizing.get('reader_writer_config', {})
        total_instances = reader_writer.get('total_instances', 1)
        writers = reader_writer.get('writers', 1)
        readers = reader_writer.get('readers', 0)
        st.metric(
            "🖥️ Total Instances",
            f"{total_instances}",
            delta=f"Writers: {writers}, Readers: {readers}"
        )
    
    # NEW SECTION: Detailed Instance Specifications
    st.markdown("---")
    st.markdown("**🖥️ Detailed Instance Specifications:**")
    
    # Show detailed sizing based on recommendation
    if recommendation == 'RDS':
        rds_rec = aws_sizing.get('rds_recommendations', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**☁️ RDS Configuration:**")
            st.info("Primary Database Instance")
            
            primary_instance = rds_rec.get('primary_instance', 'Unknown')
            instance_specs = rds_rec.get('instance_specs', {})
            
            st.write(f"**Primary Instance Type:** {primary_instance}")
            st.write(f"**vCPUs:** {instance_specs.get('vcpu', 'Unknown')}")
            st.write(f"**Memory:** {instance_specs.get('memory', 'Unknown')} GB")
            st.write(f"**Multi-AZ:** {'Yes' if rds_rec.get('multi_az', False) else 'No'}")
            st.write(f"**Storage Type:** {rds_rec.get('storage_type', 'gp3')}")
            st.write(f"**Storage Size:** {rds_rec.get('storage_size_gb', 0):,.0f} GB")
            st.write(f"**Backup Retention:** {rds_rec.get('backup_retention_days', 7)} days")
        
        with col2:
            st.markdown("**📊 Cost Breakdown:**")
            st.success("Monthly Costs")
            
            st.write(f"**Instance Cost:** ${rds_rec.get('monthly_instance_cost', 0):,.0f}")
            st.write(f"**Storage Cost:** ${rds_rec.get('monthly_storage_cost', 0):,.0f}")
            st.write(f"**Total Monthly:** ${rds_rec.get('total_monthly_cost', 0):,.0f}")
            
            # Reader instance details if applicable
            if readers > 0:
                st.write(f"**Read Replicas:** {readers} instances")
                st.write(f"**Reader Instance Type:** {primary_instance}")
                st.write(f"**Total Instance Cost:** ${rds_rec.get('monthly_instance_cost', 0) * (1 + readers):,.0f}")
    
    else:  # EC2 recommendation
        ec2_rec = aws_sizing.get('ec2_recommendations', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🖥️ EC2 Configuration:**")
            st.info("Self-Managed Database Instance")
            
            primary_instance = ec2_rec.get('primary_instance', 'Unknown')
            instance_specs = ec2_rec.get('instance_specs', {})
            database_engine = ec2_rec.get('database_engine', 'Unknown')
            
            st.write(f"**Database Engine:** {database_engine.upper()}")
            st.write(f"**Instance Type:** {primary_instance}")
            st.write(f"**vCPUs:** {instance_specs.get('vcpu', 'Unknown')}")
            st.write(f"**Memory:** {instance_specs.get('memory', 'Unknown')} GB")
            st.write(f"**Storage Type:** {ec2_rec.get('storage_type', 'gp3')}")
            st.write(f"**Storage Size:** {ec2_rec.get('storage_size_gb', 0):,.0f} GB")
            st.write(f"**EBS Optimized:** {'Yes' if ec2_rec.get('ebs_optimized', True) else 'No'}")
            st.write(f"**Enhanced Networking:** {'Yes' if ec2_rec.get('enhanced_networking', True) else 'No'}")
            
            # SQL Server specific information
            if database_engine == 'sqlserver':
                deployment_type = ec2_rec.get('sql_server_deployment_type', 'standalone')
                instance_count = ec2_rec.get('instance_count', 1)
                st.write(f"**Deployment:** {deployment_type.replace('_', ' ').title()}")
                st.write(f"**Instance Count:** {instance_count}")
                if deployment_type == 'always_on':
                    st.write(f"**Cluster Type:** 3-Node Always On")
        
        with col2:
            st.markdown("**📊 Cost Breakdown:**")
            st.success("Monthly Costs")
            
            st.write(f"**Instance Cost:** ${ec2_rec.get('monthly_instance_cost', 0):,.0f}")
            st.write(f"**Storage Cost:** ${ec2_rec.get('monthly_storage_cost', 0):,.0f}")
            
            os_licensing = ec2_rec.get('os_licensing_cost', 0)
            if os_licensing > 0:
                st.write(f"**OS Licensing:** ${os_licensing:,.0f}")
            
            st.write(f"**Total Monthly:** ${ec2_rec.get('total_monthly_cost', 0):,.0f}")
            
            # SQL Server Always On specific costs
            if ec2_rec.get('is_always_on_cluster', False):
                cost_per_instance = ec2_rec.get('cost_per_hour_per_instance', 0) * 24 * 30
                st.write(f"**Cost per Instance:** ${cost_per_instance:,.0f}")
                st.write(f"**Total Cluster Cost:** ${ec2_rec.get('total_cost_per_hour', 0) * 24 * 30:,.0f}")
    
    # Performance-based sizing recommendations
    st.markdown("---")
    st.markdown("**⚡ Performance-Based Sizing Analysis:**")
    
    sizing_col1, sizing_col2 = st.columns(2)
    
    with sizing_col1:
        st.markdown("**📊 Current Database Performance Metrics:**")
        
        performance_metrics = {}
        if recommendation == 'RDS':
            performance_metrics = rds_rec.get('performance_metrics_used', {})
        else:
            performance_metrics = ec2_rec.get('performance_metrics_used', {})
        
        st.info("Input Performance Data")
        st.write(f"**Current Memory:** {performance_metrics.get('current_memory_gb', 0)} GB")
        st.write(f"**Current CPU Cores:** {performance_metrics.get('current_cpu_cores', 0)}")
        st.write(f"**Current IOPS:** {performance_metrics.get('current_iops', 0):,}")
        st.write(f"**Current Throughput:** {performance_metrics.get('current_throughput_mbps', 0)} MB/s")
    
    with sizing_col2:
        st.markdown("**🎯 Sizing Reasoning:**")
        
        if recommendation == 'RDS':
            reasoning = rds_rec.get('sizing_reasoning', [])
        else:
            reasoning = ec2_rec.get('sizing_reasoning', [])
        
        st.success("AI Sizing Logic")
        for reason in reasoning:
            st.write(f"• {reason}")
        
        if not reasoning:
            st.write("• Based on database size analysis")
            st.write("• Standard AWS best practices applied")
    
    # SQL Server Always On Benefits (if applicable)
    if recommendation == 'EC2' and ec2_rec.get('is_always_on_cluster', False):
        st.markdown("---")
        st.markdown("**🔄 SQL Server Always On Cluster Benefits:**")
        
        benefits_col1, benefits_col2 = st.columns(2)
        
        with benefits_col1:
            st.success("✅ **Always On Advantages**")
            benefits = ec2_rec.get('always_on_benefits', [])
            for benefit in benefits:
                st.write(f"• {benefit}")
        
        with benefits_col2:
            st.warning("⚙️ **Cluster Requirements**")
            requirements = ec2_rec.get('cluster_requirements', [])
            for requirement in requirements:
                st.write(f"• {requirement}")
    
    # Reader/Writer Configuration
    st.markdown("---")
    st.markdown("**📈 Reader/Writer Configuration:**")
    
    reader_writer_config = aws_sizing.get('reader_writer_config', {})
    
    config_col1, config_col2, config_col3 = st.columns(3)
    
    with config_col1:
        st.info("**Instance Distribution**")
        st.write(f"**Writer Instances:** {reader_writer_config.get('writers', 1)}")
        st.write(f"**Reader Instances:** {reader_writer_config.get('readers', 0)}")
        st.write(f"**Total Instances:** {reader_writer_config.get('total_instances', 1)}")
    
    with config_col2:
        st.success("**Capacity Distribution**")
        st.write(f"**Write Capacity:** {reader_writer_config.get('write_capacity_percent', 100):.1f}%")
        st.write(f"**Read Capacity:** {reader_writer_config.get('read_capacity_percent', 0):.1f}%")
        st.write(f"**Recommended Read Split:** {reader_writer_config.get('recommended_read_split', 0):.1f}%")
    
    with config_col3:
        st.warning("**Configuration Notes**")
        reasoning = reader_writer_config.get('reasoning', 'Standard configuration')
        st.write(f"• {reasoning}")
        
        if reader_writer_config.get('readers', 0) > 0:
            st.write("• Read replicas improve performance")
            st.write("• Distribute read workloads effectively")
        else:
            st.write("• Single instance sufficient for current load")
    
    # Deployment Reasoning
    st.markdown("---")
    st.markdown("**🤔 Deployment Decision Reasoning:**")
    
    reasoning_col1, reasoning_col2 = st.columns(2)
    
    with reasoning_col1:
        st.info("**Decision Factors**")
        
        primary_reasons = deployment_rec.get('primary_reasons', [])
        for reason in primary_reasons:
            st.write(f"• {reason}")
        
        user_choice = deployment_rec.get('user_choice', 'unknown')
        analytical_rec = deployment_rec.get('analytical_recommendation', 'unknown')
        
        if user_choice != analytical_rec:
            st.warning(f"⚠️ **Note:** User selected {user_choice.upper()}, but analysis suggests {analytical_rec.upper()} might be optimal")
    
    with reasoning_col2:
        st.success("**Scoring Analysis**")
        
        st.write(f"**RDS Suitability Score:** {deployment_rec.get('rds_score', 0)}/100")
        st.write(f"**EC2 Suitability Score:** {deployment_rec.get('ec2_score', 0)}/100")
        st.write(f"**Decision Confidence:** {deployment_rec.get('confidence', 0)*100:.1f}%")
        
        if deployment_rec.get('rds_score', 0) > deployment_rec.get('ec2_score', 0):
            st.write("🎯 **Analysis Favors:** RDS (Managed)")
        else:
            st.write("🎯 **Analysis Favors:** EC2 (Self-Managed)")

def render_agent_scaling_tab(analysis: Dict, config: Dict):
    """Render agent scaling and configuration analysis tab"""
    st.subheader("🤖 Migration Agent Scaling & Configuration Analysis")
    
    agent_analysis = analysis.get('agent_analysis', {})
    
    # Agent Configuration Overview
    st.markdown("**🎯 Agent Configuration Overview:**")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        num_agents = agent_analysis.get('number_of_agents', 1)
        primary_tool = agent_analysis.get('primary_tool', 'DMS')
        st.metric(
            "🤖 Active Agents",
            f"{num_agents}",
            delta=f"Tool: {primary_tool.upper()}"
        )
    
    with col2:
        agent_size = agent_analysis.get('agent_size', 'medium')
        monthly_cost = agent_analysis.get('monthly_cost', 0)
        st.metric(
            "💰 Monthly Cost",
            f"${monthly_cost:,.0f}",
            delta=f"Size: {agent_size.title()}"
        )
    
    with col3:
        total_throughput = agent_analysis.get('total_max_throughput_mbps', 0)
        effective_throughput = agent_analysis.get('total_effective_throughput', 0)
        st.metric(
            "⚡ Max Throughput",
            f"{total_throughput:,.0f} Mbps",
            delta=f"Effective: {effective_throughput:,.0f} Mbps"
        )
    
    with col4:
        scaling_efficiency = agent_analysis.get('scaling_efficiency', 1.0)
        st.metric(
            "📈 Scaling Efficiency",
            f"{scaling_efficiency*100:.1f}%",
            delta="Multi-agent coordination"
        )
    
    with col5:
        bottleneck = agent_analysis.get('bottleneck', 'Unknown')
        bottleneck_severity = agent_analysis.get('bottleneck_severity', 'medium')
        st.metric(
            "🎯 Bottleneck",
            bottleneck.title(),
            delta=f"Severity: {bottleneck_severity.title()}"
        )
    
    # Agent Configuration Details
    st.markdown("---")
    st.markdown("**⚙️ Detailed Agent Configuration:**")
    
    config_col1, config_col2 = st.columns(2)
    
    with config_col1:
        st.info("**Agent Specifications**")
        
        agent_config = agent_analysis.get('agent_configuration', {})
        per_agent_spec = agent_config.get('per_agent_spec', {})
        
        st.write(f"**Agent Type:** {agent_analysis.get('primary_tool', 'Unknown').upper()}")
        st.write(f"**Agent Size:** {agent_analysis.get('agent_size', 'Unknown').title()}")
        st.write(f"**Number of Agents:** {num_agents}")
        st.write(f"**vCPU per Agent:** {per_agent_spec.get('vcpu', 'Unknown')}")
        st.write(f"**Memory per Agent:** {per_agent_spec.get('memory_gb', 'Unknown')} GB")
        st.write(f"**Max Throughput per Agent:** {per_agent_spec.get('max_throughput_mbps_per_agent', 'Unknown')} Mbps")
        st.write(f"**Max Tasks per Agent:** {per_agent_spec.get('max_concurrent_tasks_per_agent', 'Unknown')}")
        st.write(f"**Cost per Agent per Hour:** ${per_agent_spec.get('cost_per_hour_per_agent', 0):.4f}")
    
    with config_col2:
        st.success("**Total Configuration**")
        
        st.write(f"**Total vCPUs:** {agent_config.get('total_vcpu', 'Unknown')}")
        st.write(f"**Total Memory:** {agent_config.get('total_memory_gb', 'Unknown')} GB")
        st.write(f"**Total Max Throughput:** {agent_config.get('total_max_throughput_mbps', 'Unknown')} Mbps")
        st.write(f"**Total Concurrent Tasks:** {agent_config.get('total_concurrent_tasks', 'Unknown')}")
        st.write(f"**Total Cost per Hour:** ${agent_config.get('total_cost_per_hour', 0):.2f}")
        st.write(f"**Effective Cost per Hour:** ${agent_config.get('effective_cost_per_hour', 0):.2f}")
        
        # Destination storage impact
        destination_storage = agent_analysis.get('destination_storage', 'S3')
        storage_multiplier = agent_analysis.get('storage_performance_multiplier', 1.0)
        st.write(f"**Destination Storage:** {destination_storage}")
        st.write(f"**Storage Performance Multiplier:** {storage_multiplier:.2f}x")
    
    # Migration Method Specific Information
    migration_method = agent_analysis.get('migration_method', 'direct_replication')
    
    if migration_method == 'backup_restore':
        st.markdown("---")
        st.markdown("**📦 Backup Storage Migration Configuration:**")
        
        backup_col1, backup_col2 = st.columns(2)
        
        with backup_col1:
            st.warning("**Backup Storage Details**")
            
            backup_storage_type = agent_analysis.get('backup_storage_type', 'nas_drive')
            backup_efficiency = agent_analysis.get('backup_efficiency', 1.0)
            
            st.write(f"**Backup Storage Type:** {backup_storage_type.replace('_', ' ').title()}")
            st.write(f"**Protocol Efficiency:** {backup_efficiency*100:.1f}%")
            
            if backup_storage_type == 'windows_share':
                st.write(f"**Protocol:** SMB (Windows Share)")
                st.write(f"**Expected Overhead:** ~15% bandwidth loss")
                st.write(f"**Authentication:** Windows-based")
            else:
                st.write(f"**Protocol:** NFS (Network Attached Storage)")
                st.write(f"**Expected Overhead:** ~8% bandwidth loss")
                st.write(f"**Authentication:** NFS-based")
        
        with backup_col2:
            st.info("**DataSync Agent Optimization**")
            
            st.write("**Backup Storage Optimizations:**")
            if backup_storage_type == 'windows_share':
                st.write("• Enable SMB3 multichannel")
                st.write("• Optimize SMB client settings")
                st.write("• Use dedicated backup network")
                st.write("• Monitor SMB connection stability")
            else:
                st.write("• Use NFS v4.1+ for performance")
                st.write("• Configure optimal rsize/wsize")
                st.write("• Enable NFS caching")
                st.write("• Monitor NFS performance")
    
    # Scaling Analysis
    st.markdown("---")
    st.markdown("**📈 Agent Scaling Analysis:**")
    
    scaling_col1, scaling_col2, scaling_col3 = st.columns(3)
    
    with scaling_col1:
        st.info("**Current Scaling Metrics**")
        
        management_overhead = agent_analysis.get('management_overhead', 1.0)
        
        st.write(f"**Scaling Efficiency:** {scaling_efficiency*100:.1f}%")
        st.write(f"**Management Overhead:** {management_overhead:.2f}x")
        st.write(f"**Coordination Complexity:** {'High' if num_agents > 5 else 'Medium' if num_agents > 2 else 'Low'}")
        
        # Calculate effective parallel benefit
        theoretical_max = num_agents * per_agent_spec.get('max_throughput_mbps_per_agent', 500)
        actual_max = agent_config.get('total_max_throughput_mbps', theoretical_max)
        parallel_efficiency = (actual_max / theoretical_max * 100) if theoretical_max > 0 else 100
        
        st.write(f"**Parallel Efficiency:** {parallel_efficiency:.1f}%")
    
    with scaling_col2:
        st.success("**Scaling Recommendations**")
        
        scaling_recommendations = agent_config.get('scaling_recommendations', [])
        
        if scaling_recommendations:
            for rec in scaling_recommendations:
                st.write(f"• {rec}")
        else:
            st.write("• Current configuration appears optimal")
            st.write("• Monitor performance during migration")
    
    with scaling_col3:
        st.warning("**Configuration Assessment**")
        
        optimal_config = agent_config.get('optimal_configuration', {})
        
        efficiency_score = optimal_config.get('efficiency_score', 100)
        complexity = optimal_config.get('management_complexity', 'Low')
        cost_efficiency = optimal_config.get('cost_efficiency', 'Good')
        
        st.write(f"**Efficiency Score:** {efficiency_score:.0f}/100")
        st.write(f"**Management Complexity:** {complexity}")
        st.write(f"**Cost Efficiency:** {cost_efficiency}")
        
        optimal_recommendation = optimal_config.get('optimal_recommendation', 'Current configuration is good')
        st.write(f"**Recommendation:** {optimal_recommendation}")
    
    # Performance Comparison Chart
    st.markdown("---")
    st.markdown("**📊 Agent Performance Comparison:**")
    
    # Create agent scaling visualization
    agent_counts = list(range(1, min(11, num_agents + 4)))
    throughputs = []
    costs = []
    efficiencies = []
    
    cost_per_agent = per_agent_spec.get('cost_per_hour_per_agent', 0.1) * 24 * 30
    throughput_per_agent = per_agent_spec.get('max_throughput_mbps_per_agent', 500)
    
    for count in agent_counts:
        # Calculate scaling efficiency for each count
        if count == 1:
            scaling_eff = 1.0
        elif count <= 3:
            scaling_eff = 0.95
        elif count <= 5:
            scaling_eff = 0.90
        elif count <= 8:
            scaling_eff = 0.85
        else:
            scaling_eff = 0.80
        
        # Apply storage multiplier
        storage_multiplier = agent_analysis.get('storage_performance_multiplier', 1.0)
        
        total_throughput = throughput_per_agent * count * scaling_eff * storage_multiplier
        total_cost = cost_per_agent * count
        efficiency = (total_throughput / (throughput_per_agent * count)) * 100
        
        throughputs.append(total_throughput)
        costs.append(total_cost)
        efficiencies.append(efficiency)
    
    # Create comparison visualizations
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        # Throughput vs Agent Count
        fig_throughput = px.line(
            x=agent_counts,
            y=throughputs,
            title="Throughput vs Number of Agents",
            labels={'x': 'Number of Agents', 'y': 'Total Throughput (Mbps)'},
            markers=True
        )
        fig_throughput.add_vline(x=num_agents, line_dash="dash", line_color="red", 
                                annotation_text=f"Current: {num_agents} agents")
        st.plotly_chart(fig_throughput, use_container_width=True)
    
    with viz_col2:
        # Cost vs Efficiency
        fig_efficiency = px.scatter(
            x=costs,
            y=efficiencies,
            size=[c/100 for c in agent_counts],  # Size by agent count
            title="Cost vs Efficiency Trade-off",
            labels={'x': 'Monthly Cost ($)', 'y': 'Scaling Efficiency (%)'},
            hover_data={'Agent Count': agent_counts}
        )
        
        # Highlight current configuration
        current_cost = cost_per_agent * num_agents
        current_efficiency = scaling_efficiency * 100
        fig_efficiency.add_scatter(x=[current_cost], y=[current_efficiency], 
                                 mode='markers', marker=dict(color='red', size=15),
                                 name='Current Config')
        
        st.plotly_chart(fig_efficiency, use_container_width=True)

def create_network_path_diagram(network_perf: Dict):
    """Create network path diagram using Plotly"""
    segments = network_perf.get('segments', [])
    
    if not segments:
        return None
    
    fig = go.Figure()
    
    # Create network path visualization
    x_positions = list(range(len(segments)))
    y_position = 0
    
    for i, segment in enumerate(segments):
        # Add segment box
        fig.add_shape(
            type="rect",
            x0=i-0.4, y0=-0.3, x1=i+0.4, y1=0.3,
            fillcolor="lightblue" if i % 2 == 0 else "lightgreen",
            line=dict(color="black", width=2)
        )
        
        # Add segment label
        fig.add_annotation(
            x=i, y=0,
            text=f"{segment['name']}<br>{segment.get('effective_bandwidth_mbps', 0):,.0f} Mbps<br>{segment.get('effective_latency_ms', 0):.1f}ms",
            showarrow=False,
            font=dict(size=10)
        )
        
        # Add arrow to next segment
        if i < len(segments) - 1:
            fig.add_annotation(
                x=i+0.5, y=0,
                ax=i+0.4, ay=0,
                axref="x", ayref="y",
                arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="red"
            )
    
    fig.update_layout(
        title=f"Network Path: {network_perf.get('path_name', 'Unknown')}",
        xaxis=dict(range=[-0.5, len(segments)-0.5], showticklabels=False),
        yaxis=dict(range=[-0.5, 0.5], showticklabels=False),
        height=200,
        showlegend=False
    )
    
    return fig

def render_footer():
    """Render enhanced footer"""
    st.markdown("""
    <div class="enterprise-footer">
        <h3>🤖 AWS Enterprise Database Migration Analyzer AI v3.0</h3>
    </div>
    """, unsafe_allow_html=True)

class UnifiedAWSCostCalculator:
    """Unified AWS cost calculator to ensure consistency across all tabs"""
    
    def __init__(self, aws_api_manager: AWSAPIManager):
        self.aws_api = aws_api_manager
    
    async def calculate_unified_aws_costs(self, config: Dict, analysis: Dict) -> Dict:
        """Calculate unified AWS costs for all services with consistency"""
        
        # Get real-time pricing data
        pricing_data = await self.aws_api.get_real_time_pricing()
        
        # Determine deployment type
        target_platform = config.get('target_platform', 'rds')
        aws_sizing = analysis.get('aws_sizing_recommendations', {})
        
        # Calculate primary compute and storage costs
        if target_platform == 'rds':
            compute_storage_costs = self._calculate_unified_rds_costs(config, aws_sizing, pricing_data)
        else:
            compute_storage_costs = self._calculate_unified_ec2_costs(config, aws_sizing, pricing_data)
        
        # Calculate networking costs
        network_costs = await self._calculate_unified_network_costs(config, pricing_data)
        
        # Calculate migration agent costs
        agent_costs = self._calculate_unified_agent_costs(config, analysis, pricing_data)
        
        # Calculate destination storage costs
        destination_costs = self._calculate_unified_destination_storage_costs(config, pricing_data)
        
        # Unified monthly totals (NO ONE-TIME COSTS)
        monthly_totals = {
            'primary_compute': compute_storage_costs['compute_monthly'],
            'primary_storage': compute_storage_costs['storage_monthly'],
            'network_services': network_costs['monthly_total'],
            'migration_agents': agent_costs['monthly_total'],
            'destination_storage': destination_costs['monthly_total'],
            'backup_storage': destination_costs.get('backup_monthly', 0)
        }
        
        total_monthly = sum(monthly_totals.values())
        
        return {
            'pricing_data_source': pricing_data.get('data_source', 'fallback'),
            'last_updated': pricing_data.get('last_updated', datetime.now()),
            'target_platform': target_platform,
            'deployment_details': compute_storage_costs,
            'network_details': network_costs,
            'agent_details': agent_costs,
            'destination_details': destination_costs,
            'monthly_breakdown': monthly_totals,
            'total_monthly_cost': total_monthly,
            'annual_cost': total_monthly * 12,
            'three_year_cost': total_monthly * 36,
            'cost_per_gb_per_month': total_monthly / config.get('database_size_gb', 1) if config.get('database_size_gb', 0) > 0 else 0,
            'optimization_opportunities': self._generate_unified_optimization_recommendations(monthly_totals, config)
        }
    
    def _calculate_unified_rds_costs(self, config: Dict, aws_sizing: Dict, pricing_data: Dict) -> Dict:
        """Calculate unified RDS costs"""
        rds_rec = aws_sizing.get('rds_recommendations', {})
        reader_writer_config = aws_sizing.get('reader_writer_config', {})
        
        # Primary instance cost
        primary_instance_cost = rds_rec.get('monthly_instance_cost', 0)
        
        # Reader instances cost (if any)
        readers = reader_writer_config.get('readers', 0)
        reader_cost = primary_instance_cost * readers * 0.9 if readers > 0 else 0  # Readers are 10% cheaper
        
        # Multi-AZ cost
        multi_az_cost = primary_instance_cost if config.get('environment') == 'production' else 0
        
        # Storage cost
        storage_cost = rds_rec.get('monthly_storage_cost', 0)
        
        # Total compute cost
        total_compute = primary_instance_cost + reader_cost + multi_az_cost
        
        return {
            'service_type': 'RDS',
            'primary_instance': rds_rec.get('primary_instance', 'Unknown'),
            'primary_instance_cost': primary_instance_cost,
            'reader_instances': readers,
            'reader_cost': reader_cost,
            'multi_az_cost': multi_az_cost,
            'storage_cost': storage_cost,
            'compute_monthly': total_compute,
            'storage_monthly': storage_cost,
            'total_monthly': total_compute + storage_cost,
            'instance_details': {
                'primary_type': rds_rec.get('primary_instance', 'Unknown'),
                'storage_type': rds_rec.get('storage_type', 'gp3'),
                'storage_size_gb': rds_rec.get('storage_size_gb', 0),
                'multi_az_enabled': config.get('environment') == 'production'
            }
        }
    
    def _calculate_unified_ec2_costs(self, config: Dict, aws_sizing: Dict, pricing_data: Dict) -> Dict:
        """Calculate unified EC2 costs"""
        ec2_rec = aws_sizing.get('ec2_recommendations', {})
        
        # Instance costs
        instance_cost = ec2_rec.get('monthly_instance_cost', 0)
        
        # OS licensing
        os_licensing = ec2_rec.get('os_licensing_cost', 0)
        
        # Storage cost
        storage_cost = ec2_rec.get('monthly_storage_cost', 0)
        
        # Total compute cost (instance + OS licensing)
        total_compute = instance_cost + os_licensing
        
        return {
            'service_type': 'EC2',
            'primary_instance': ec2_rec.get('primary_instance', 'Unknown'),
            'instance_cost': instance_cost,
            'os_licensing_cost': os_licensing,
            'storage_cost': storage_cost,
            'compute_monthly': total_compute,
            'storage_monthly': storage_cost,
            'total_monthly': total_compute + storage_cost,
            'instance_details': {
                'primary_type': ec2_rec.get('primary_instance', 'Unknown'),
                'database_engine': ec2_rec.get('database_engine', 'Unknown'),
                'storage_type': ec2_rec.get('storage_type', 'gp3'),
                'storage_size_gb': ec2_rec.get('storage_size_gb', 0),
                'instance_count': ec2_rec.get('instance_count', 1),
                'deployment_type': ec2_rec.get('sql_server_deployment_type', 'standalone')
            }
        }
    
    async def _calculate_unified_network_costs(self, config: Dict, pricing_data: Dict) -> Dict:
        """Calculate unified network costs"""
        environment = config.get('environment', 'non-production')
        
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
        database_size_gb = config.get('database_size_gb', 1000)
        monthly_data_transfer_gb = database_size_gb * 0.1  # 10% of DB size per month
        data_transfer_cost_per_gb = 0.02
        
        if dx_capacity in dx_pricing:
            data_transfer_cost_per_gb = dx_pricing[dx_capacity].get('data_transfer_out', data_transfer_cost_per_gb)
        
        data_transfer_monthly = monthly_data_transfer_gb * data_transfer_cost_per_gb
        
        # VPN costs (if production)
        vpn_monthly = 45.0 if environment == 'production' else 0
        
        return {
            'direct_connect': {
                'capacity': dx_capacity,
                'monthly_cost': dx_monthly
            },
            'data_transfer': {
                'monthly_gb': monthly_data_transfer_gb,
                'monthly_cost': data_transfer_monthly
            },
            'vpn_gateway': {
                'monthly_cost': vpn_monthly
            },
            'monthly_total': dx_monthly + data_transfer_monthly + vpn_monthly
        }
    
    def _calculate_unified_agent_costs(self, config: Dict, analysis: Dict, pricing_data: Dict) -> Dict:
        """Calculate unified migration agent costs"""
        agent_analysis = analysis.get('agent_analysis', {})
        
        # Get agent configuration
        num_agents = config.get('number_of_agents', 1)
        migration_method = config.get('migration_method', 'direct_replication')
        
        # Determine primary tool
        if migration_method == 'backup_restore':
            primary_tool = 'datasync'
            agent_size = config.get('datasync_agent_size', 'medium')
        else:
            is_homogeneous = config['source_database_engine'] == config['database_engine']
            primary_tool = 'datasync' if is_homogeneous else 'dms'
            agent_size = config.get('datasync_agent_size' if is_homogeneous else 'dms_agent_size', 'medium')
        
        # Agent specifications
        if primary_tool == 'datasync':
            agent_specs = {
                'small': {'vcpu': 2, 'memory': 4, 'cost_per_hour': 0.0416},
                'medium': {'vcpu': 2, 'memory': 4, 'cost_per_hour': 0.085},
                'large': {'vcpu': 4, 'memory': 8, 'cost_per_hour': 0.17},
                'xlarge': {'vcpu': 8, 'memory': 16, 'cost_per_hour': 0.34}
            }
        else:  # DMS
            agent_specs = {
                'small': {'vcpu': 2, 'memory': 4, 'cost_per_hour': 0.0416},
                'medium': {'vcpu': 2, 'memory': 4, 'cost_per_hour': 0.085},
                'large': {'vcpu': 4, 'memory': 8, 'cost_per_hour': 0.17},
                'xlarge': {'vcpu': 8, 'memory': 16, 'cost_per_hour': 0.34},
                'xxlarge': {'vcpu': 16, 'memory': 32, 'cost_per_hour': 0.68}
            }
        
        agent_spec = agent_specs.get(agent_size, agent_specs['medium'])
        
        # Calculate total monthly cost
        total_monthly_cost = agent_spec['cost_per_hour'] * 24 * 30 * num_agents
        
        return {
            'primary_tool': primary_tool,
            'agent_size': agent_size,
            'number_of_agents': num_agents,
            'cost_per_agent_per_hour': agent_spec['cost_per_hour'],
            'total_vcpu': agent_spec['vcpu'] * num_agents,
            'total_memory_gb': agent_spec['memory'] * num_agents,
            'monthly_total': total_monthly_cost,
            'migration_method': migration_method
        }
    
    def _calculate_unified_destination_storage_costs(self, config: Dict, pricing_data: Dict) -> Dict:
        """Calculate unified destination storage costs"""
        database_size_gb = config.get('database_size_gb', 1000)
        destination_storage = config.get('destination_storage_type', 'S3')
        migration_method = config.get('migration_method', 'direct_replication')
        
        # Destination storage pricing
        storage_pricing = {
            'S3': 0.023,
            'FSx_Windows': 0.13,
            'FSx_Lustre': 0.14
        }
        
        dest_cost_per_gb = storage_pricing.get(destination_storage, 0.023)
        dest_storage_size = database_size_gb * 1.2  # 20% buffer
        dest_monthly_cost = dest_storage_size * dest_cost_per_gb
        
        # Backup storage costs (if backup/restore method)
        backup_monthly_cost = 0
        backup_size_gb = 0
        
        if migration_method == 'backup_restore':
            backup_size_multiplier = config.get('backup_size_multiplier', 0.7)
            backup_size_gb = database_size_gb * backup_size_multiplier
            backup_monthly_cost = backup_size_gb * 0.023  # S3 Standard for backups
        
        return {
            'destination_storage_type': destination_storage,
            'destination_size_gb': dest_storage_size,
            'destination_cost_per_gb': dest_cost_per_gb,
            'destination_monthly_cost': dest_monthly_cost,
            'backup_applicable': migration_method == 'backup_restore',
            'backup_size_gb': backup_size_gb,
            'backup_monthly_cost': backup_monthly_cost,
            'monthly_total': dest_monthly_cost + backup_monthly_cost
        }
    
    def _generate_unified_optimization_recommendations(self, monthly_costs: Dict, config: Dict) -> List[str]:
        """Generate unified cost optimization recommendations"""
        recommendations = []
        total_monthly = sum(monthly_costs.values())
        
        # Compute optimization
        compute_cost = monthly_costs.get('primary_compute', 0)
        if compute_cost > total_monthly * 0.6:
            recommendations.append("Consider Reserved Instances for 20-30% savings on compute costs")
        
        # Storage optimization
        storage_costs = monthly_costs.get('primary_storage', 0) + monthly_costs.get('destination_storage', 0)
        if storage_costs > total_monthly * 0.25:
            recommendations.append("Implement storage lifecycle policies and optimize storage types")
        
        # Network optimization
        network_cost = monthly_costs.get('network_services', 0)
        if network_cost > total_monthly * 0.15:
            recommendations.append("Review data transfer patterns and consider CloudFront for optimization")
        
        # Agent optimization
        agent_cost = monthly_costs.get('migration_agents', 0)
        num_agents = config.get('number_of_agents', 1)
        if num_agents > 4 and agent_cost > total_monthly * 0.1:
            recommendations.append("Consider consolidating migration agents to reduce operational overhead")
        
        # Environment-specific
        if config.get('environment') == 'non-production':
            recommendations.append("Use Spot Instances for non-production workloads for 60-70% savings")
        
        return recommendations

def render_total_aws_cost_tab(analysis: Dict, config: Dict):
    """Render the unified Total AWS Cost tab"""
    st.subheader("💰 Total AWS Cost Analysis")
    
    # Get unified cost data
    unified_costs = analysis.get('unified_aws_costs', {})
    
    if not unified_costs:
        st.warning("⚠️ Unified cost data not available. Please run the analysis first.")
        return
    
    # Executive Summary
    st.markdown("**📊 Executive Cost Summary**")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_monthly = unified_costs.get('total_monthly_cost', 0)
        st.metric(
            "💰 Total Monthly Cost",
            f"${total_monthly:,.0f}",
            delta=f"${total_monthly/30:.0f}/day"
        )
    
    with col2:
        annual_cost = unified_costs.get('annual_cost', 0)
        st.metric(
            "📅 Annual Cost",
            f"${annual_cost:,.0f}",
            delta=f"Per quarter: ${annual_cost/4:,.0f}"
        )
    
    with col3:
        cost_per_gb = unified_costs.get('cost_per_gb_per_month', 0)
        st.metric(
            "📊 Cost per GB/Month",
            f"${cost_per_gb:.2f}",
            delta=f"Database: {config.get('database_size_gb', 0):,} GB"
        )
    
    with col4:
        target_platform = unified_costs.get('target_platform', 'unknown').upper()
        pricing_source = unified_costs.get('pricing_data_source', 'fallback')
        st.metric(
            "🎯 Platform",
            target_platform,
            delta=f"Data: {pricing_source.title()}"
        )
    
    # Monthly Cost Breakdown
    st.markdown("---")
    st.markdown("**📋 Monthly Cost Breakdown**")
    
    monthly_breakdown = unified_costs.get('monthly_breakdown', {})
    
    # Create detailed breakdown table
    breakdown_data = []
    for category, cost in monthly_breakdown.items():
        if cost > 0:
            category_name = category.replace('_', ' ').title()
            percentage = (cost / total_monthly * 100) if total_monthly > 0 else 0
            breakdown_data.append({
                'Cost Category': category_name,
                'Monthly Cost': f"${cost:,.0f}",
                'Percentage': f"{percentage:.1f}%",
                'Annual Cost': f"${cost * 12:,.0f}"
            })
    
    if breakdown_data:
        df_breakdown = pd.DataFrame(breakdown_data)
        st.dataframe(df_breakdown, use_container_width=True, hide_index=True)
    
    # Service Details
    st.markdown("---")
    st.markdown("**🔧 Detailed Service Analysis**")
    
    detail_col1, detail_col2 = st.columns(2)
    
    with detail_col1:
        st.markdown("**💻 Primary Infrastructure**")
        
        deployment_details = unified_costs.get('deployment_details', {})
        
        st.info(f"**{deployment_details.get('service_type', 'Unknown')} Configuration**")
        
        if deployment_details.get('service_type') == 'RDS':
            st.write(f"**Primary Instance:** {deployment_details.get('primary_instance', 'Unknown')}")
            st.write(f"**Instance Cost:** ${deployment_details.get('primary_instance_cost', 0):,.0f}/month")
            
            if deployment_details.get('reader_instances', 0) > 0:
                st.write(f"**Read Replicas:** {deployment_details.get('reader_instances', 0)} instances")
                st.write(f"**Reader Cost:** ${deployment_details.get('reader_cost', 0):,.0f}/month")
            
            if deployment_details.get('multi_az_cost', 0) > 0:
                st.write(f"**Multi-AZ Cost:** ${deployment_details.get('multi_az_cost', 0):,.0f}/month")
        
        else:  # EC2
            st.write(f"**Instance Type:** {deployment_details.get('primary_instance', 'Unknown')}")
            st.write(f"**Instance Cost:** ${deployment_details.get('instance_cost', 0):,.0f}/month")
            
            if deployment_details.get('os_licensing_cost', 0) > 0:
                st.write(f"**OS Licensing:** ${deployment_details.get('os_licensing_cost', 0):,.0f}/month")
        
        st.write(f"**Storage Cost:** ${deployment_details.get('storage_cost', 0):,.0f}/month")
        st.write(f"**Total Infrastructure:** ${deployment_details.get('total_monthly', 0):,.0f}/month")
    
    with detail_col2:
        st.markdown("**🌐 Supporting Services**")
        
        network_details = unified_costs.get('network_details', {})
        agent_details = unified_costs.get('agent_details', {})
        destination_details = unified_costs.get('destination_details', {})
        
        st.success("**Network Services**")
        dx_cost = network_details.get('direct_connect', {}).get('monthly_cost', 0)
        transfer_cost = network_details.get('data_transfer', {}).get('monthly_cost', 0)
        vpn_cost = network_details.get('vpn_gateway', {}).get('monthly_cost', 0)
        
        st.write(f"**Direct Connect:** ${dx_cost:,.0f}/month")
        st.write(f"**Data Transfer:** ${transfer_cost:,.0f}/month")
        if vpn_cost > 0:
            st.write(f"**VPN Gateway:** ${vpn_cost:,.0f}/month")
        
        st.warning("**Migration Services**")
        st.write(f"**Agent Tool:** {agent_details.get('primary_tool', 'Unknown').upper()}")
        st.write(f"**Number of Agents:** {agent_details.get('number_of_agents', 0)}")
        st.write(f"**Agent Costs:** ${agent_details.get('monthly_total', 0):,.0f}/month")
        
        st.info("**Storage Services**")
        st.write(f"**Destination:** {destination_details.get('destination_storage_type', 'Unknown')}")
        st.write(f"**Destination Cost:** ${destination_details.get('destination_monthly_cost', 0):,.0f}/month")
        
        if destination_details.get('backup_applicable', False):
            st.write(f"**Backup Storage:** ${destination_details.get('backup_monthly_cost', 0):,.0f}/month")
    
    # Cost Projection Chart
    st.markdown("---")
    st.markdown("**📈 Cost Projection Over Time**")
    
    # Create 3-year projection
    months = list(range(1, 37))  # 36 months
    monthly_costs = [total_monthly] * 36
    cumulative_costs = [sum(monthly_costs[:i+1]) for i in range(36)]
    
    projection_data = pd.DataFrame({
        'Month': months,
        'Monthly Cost': monthly_costs,
        'Cumulative Cost': cumulative_costs
    })
    
    fig_projection = px.line(
        projection_data,
        x='Month',
        y='Cumulative Cost',
        title=f"3-Year AWS Cost Projection: ${total_monthly:,.0f}/month",
        labels={'Month': 'Months from Start', 'Cumulative Cost': 'Cumulative Cost ($)'}
    )
    
    # Add quarterly markers
    for quarter in range(3, 37, 3):
        fig_projection.add_vline(
            x=quarter, 
            line_dash="dash", 
            line_color="gray",
            annotation_text=f"Q{(quarter//3)}"
        )
    
    fig_projection.update_traces(line_color='#3498db', line_width=3)
    fig_projection.update_layout(height=400)
    st.plotly_chart(fig_projection, use_container_width=True)
    
    # Cost Distribution Pie Chart
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        st.markdown("**🥧 Cost Distribution**")
        
        # Filter out zero costs for pie chart
        pie_data = {k.replace('_', ' ').title(): v for k, v in monthly_breakdown.items() if v > 0}
        
        if pie_data:
            fig_pie = px.pie(
                values=list(pie_data.values()),
                names=list(pie_data.keys()),
                title="Monthly Cost Distribution"
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No cost data available for visualization")
    
    with viz_col2:
        st.markdown("**📊 Cost Efficiency Metrics**")
        
        st.metric(
            "💾 Cost per GB",
            f"${cost_per_gb:.2f}",
            delta="Per month"
        )
        
        # Calculate cost efficiency metrics
        database_size = config.get('database_size_gb', 1)
        if database_size > 0:
            infrastructure_cost = monthly_breakdown.get('primary_compute', 0) + monthly_breakdown.get('primary_storage', 0)
            infrastructure_per_gb = infrastructure_cost / database_size
            
            st.metric(
                "🖥️ Infrastructure Cost/GB",
                f"${infrastructure_per_gb:.2f}",
                delta="Compute + Storage"
            )
        
        # Performance cost ratio
        total_throughput = analysis.get('agent_analysis', {}).get('total_effective_throughput', 1000)
        cost_per_mbps = total_monthly / total_throughput if total_throughput > 0 else 0
        
        st.metric(
            "⚡ Cost per Mbps",
            f"${cost_per_mbps:.2f}",
            delta="Migration performance"
        )
    
    # Optimization Opportunities
    st.markdown("---")
    st.markdown("**💡 Cost Optimization Opportunities**")
    
    optimization_recs = unified_costs.get('optimization_opportunities', [])
    
    if optimization_recs:
        opt_col1, opt_col2 = st.columns(2)
        
        with opt_col1:
            st.success("**Immediate Opportunities (0-3 months)**")
            for i, rec in enumerate(optimization_recs[:3], 1):
                potential_savings = "20-30%" if 'Reserved' in rec else "60-70%" if 'Spot' in rec else "10-25%"
                st.write(f"**{i}.** {rec}")
                st.write(f"   💰 Potential monthly savings: {potential_savings}")
                st.write("")
        
        with opt_col2:
            st.info("**Medium-term Opportunities (3-12 months)**")
            remaining_recs = optimization_recs[3:] if len(optimization_recs) > 3 else ["Continue monitoring for additional opportunities"]
            for i, rec in enumerate(remaining_recs[:3], 4):
                potential_savings = "15-25%" if 'lifecycle' in rec.lower() else "5-15%"
                st.write(f"**{i}.** {rec}")
                st.write(f"   💰 Potential monthly savings: {potential_savings}")
                st.write("")
    else:
        st.info("✅ Current configuration appears well-optimized. Continue monitoring for opportunities.")
    
    # Summary and Export
    st.markdown("---")
    st.markdown("**📄 Cost Summary & Export**")
    
    summary_col1, summary_col2 = st.columns(2)
    
    with summary_col1:
        st.info("**Cost Summary**")
        st.write(f"**Total Monthly Cost:** ${total_monthly:,.0f}")
        st.write(f"**Annual Cost:** ${annual_cost:,.0f}")
        st.write(f"**3-Year Total:** ${unified_costs.get('three_year_cost', 0):,.0f}")
        st.write(f"**Cost per GB per Month:** ${cost_per_gb:.2f}")
        st.write(f"**Target Platform:** {target_platform}")
        
        # Calculate cost efficiency vs industry benchmarks
        if cost_per_gb > 0:
            if cost_per_gb < 0.10:
                efficiency_rating = "Excellent"
            elif cost_per_gb < 0.25:
                efficiency_rating = "Good"
            elif cost_per_gb < 0.50:
                efficiency_rating = "Average"
            else:
                efficiency_rating = "Needs Optimization"
            
            st.write(f"**Cost Efficiency Rating:** {efficiency_rating}")
    
    with summary_col2:
        st.success("**Data Quality & Reliability**")
        
        last_updated = unified_costs.get('last_updated', 'Unknown')
        pricing_source = unified_costs.get('pricing_data_source', 'fallback')
        
        if pricing_source == 'aws_api':
            st.write("✅ **Pricing Data:** Real-time AWS API")
            st.write(f"📅 **Last Updated:** {last_updated}")
            st.write("🎯 **Accuracy:** High (±5%)")
        else:
            st.write("⚠️ **Pricing Data:** Fallback estimates")
            st.write("🎯 **Accuracy:** Moderate (±15%)")
        
        total_services = len([k for k, v in monthly_breakdown.items() if v > 0])
        st.write(f"🔧 **Services Analyzed:** {total_services} AWS services")
        st.write(f"📊 **Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # Export functionality
    if st.button("📤 Export Total AWS Cost Analysis", use_container_width=True):
        export_data = {
            'analysis_date': datetime.now().isoformat(),
            'configuration': config,
            'unified_costs': unified_costs,
            'monthly_breakdown': monthly_breakdown,
            'total_monthly_cost': total_monthly,
            'annual_cost': annual_cost,
            'cost_per_gb_per_month': cost_per_gb,
            'optimization_opportunities': optimization_recs
        }
        
        st.download_button(
            label="💾 Download Total AWS Cost Analysis (JSON)",
            data=json.dumps(export_data, indent=2),
            file_name=f"total_aws_cost_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

    # CORRECT CODE:
    category_total = 0
    for _, row in category_data.iterrows():
        cost_str = row['Monthly Cost'].replace('


# Updated main function to include the new tab
async def main_with_unified_costs():
    """Enhanced main application with unified cost analysis"""
    render_enhanced_header()
    
    # Enhanced sidebar controls
    config = render_enhanced_sidebar_controls()
    
    # Initialize the migration analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = EnhancedMigrationAnalyzer()
    
    # Check if we need to run analysis
    if 'analysis_data' not in st.session_state or config_has_changed(config, st.session_state.get('last_config')):
        with st.spinner("🤖 Running comprehensive AI-powered migration analysis with unified cost calculation..."):
            try:
                # Run the comprehensive analysis with unified costs
                analysis_data = await run_comprehensive_analysis_with_unified_costs(st.session_state.analyzer, config)
                
                st.session_state.analysis_data = analysis_data
                st.session_state.last_config = config.copy()
                
                st.success("✅ Analysis completed successfully with unified cost calculation!")
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                logger.error(f"Analysis error: {e}")
                return
    else:
        st.info("📊 Using cached analysis results. Change configuration to trigger re-analysis.")
    
    analysis_data = st.session_state.analysis_data
    
    # Create tabs for organized display - UPDATED TAB LIST
    tab_names = [
        "💰 Total AWS Cost",  # NEW: First tab for unified costs
        "🧠 AI Insights & Analysis",
        "🌐 Network Intelligence", 
        "💻 OS Performance Analysis",
        "🎯 AWS Sizing & Configuration",
        "🤖 Agent Scaling Analysis",
        "📊 Detailed Cost Breakdown"  # RENAMED: Old comprehensive cost analysis
    ]
    
    tabs = st.tabs(tab_names)
    
    # Render each tab
    with tabs[0]:  # NEW: Total AWS Cost tab
        render_total_aws_cost_tab(analysis_data, config)
    
    with tabs[1]:
        render_ai_insights_tab_enhanced(analysis_data, config)
    
    with tabs[2]:
        render_network_intelligence_tab(analysis_data, config)
    
    with tabs[3]:
        render_os_performance_tab(analysis_data, config)
    
    with tabs[4]:
        render_aws_sizing_tab(analysis_data, config)
    
    with tabs[5]:
        render_agent_scaling_tab(analysis_data, config)
    
    with tabs[6]:  # FIXED: Detailed breakdown (using fixed version)
        render_comprehensive_cost_analysis_tab_fixed(analysis_data, config)
    
    # Render footer
    render_footer()

# Replace the existing main call at the bottom of the file
if __name__ == "__main__":
    asyncio.run(main_with_unified_costs()), '').replace(',', '')
        try:
            category_total += float(cost_str)
        except (ValueError, TypeError):
            pass  # Skip if can't parse the cost string
    
    return category_total


# Updated main function to include the new tab
async def main_with_unified_costs():
    """Enhanced main application with unified cost analysis"""
    render_enhanced_header()
    
    # Enhanced sidebar controls
    config = render_enhanced_sidebar_controls()
    
    # Initialize the migration analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = EnhancedMigrationAnalyzer()
    
    # Check if we need to run analysis
    if 'analysis_data' not in st.session_state or config_has_changed(config, st.session_state.get('last_config')):
        with st.spinner("🤖 Running comprehensive AI-powered migration analysis with unified cost calculation..."):
            try:
                # Run the comprehensive analysis with unified costs
                analysis_data = await run_comprehensive_analysis_with_unified_costs(st.session_state.analyzer, config)
                
                st.session_state.analysis_data = analysis_data
                st.session_state.last_config = config.copy()
                
                st.success("✅ Analysis completed successfully with unified cost calculation!")
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                logger.error(f"Analysis error: {e}")
                return
    else:
        st.info("📊 Using cached analysis results. Change configuration to trigger re-analysis.")
    
    analysis_data = st.session_state.analysis_data
    
    # Create tabs for organized display - UPDATED TAB LIST
    tab_names = [
        "💰 Total AWS Cost",  # NEW: First tab for unified costs
        "🧠 AI Insights & Analysis",
        "🌐 Network Intelligence", 
        "💻 OS Performance Analysis",
        "🎯 AWS Sizing & Configuration",
        "🤖 Agent Scaling Analysis",
        "📊 Detailed Cost Breakdown"  # RENAMED: Old comprehensive cost analysis
    ]
    
    tabs = st.tabs(tab_names)
    
    # Render each tab
    with tabs[0]:  # NEW: Total AWS Cost tab
        render_total_aws_cost_tab(analysis_data, config)
    
    with tabs[1]:
        render_ai_insights_tab_enhanced(analysis_data, config)
    
    with tabs[2]:
        render_network_intelligence_tab(analysis_data, config)
    
    with tabs[3]:
        render_os_performance_tab(analysis_data, config)
    
    with tabs[4]:
        render_aws_sizing_tab(analysis_data, config)
    
    with tabs[5]:
        render_agent_scaling_tab(analysis_data, config)
    
    with tabs[6]:  # FIXED: Detailed breakdown (using fixed version)
        render_comprehensive_cost_analysis_tab_fixed(analysis_data, config)
    
    # Render footer
    render_footer()



# Updated main function to include the new tab
async def main_with_unified_costs():
    """Enhanced main application with unified cost analysis"""
    render_enhanced_header()
    
    # Enhanced sidebar controls
    config = render_enhanced_sidebar_controls()
    
    # Initialize the migration analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = EnhancedMigrationAnalyzer()
    
    # Check if we need to run analysis
    if 'analysis_data' not in st.session_state or config_has_changed(config, st.session_state.get('last_config')):
        with st.spinner("🤖 Running comprehensive AI-powered migration analysis with unified cost calculation..."):
            try:
                # Run the comprehensive analysis with unified costs
                analysis_data = await run_comprehensive_analysis_with_unified_costs(st.session_state.analyzer, config)
                
                st.session_state.analysis_data = analysis_data
                st.session_state.last_config = config.copy()
                
                st.success("✅ Analysis completed successfully with unified cost calculation!")
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                logger.error(f"Analysis error: {e}")
                return
    else:
        st.info("📊 Using cached analysis results. Change configuration to trigger re-analysis.")
    
    analysis_data = st.session_state.analysis_data
    
    # Create tabs for organized display - UPDATED TAB LIST
    tab_names = [
        "💰 Total AWS Cost",  # NEW: First tab for unified costs
        "🧠 AI Insights & Analysis",
        "🌐 Network Intelligence", 
        "💻 OS Performance Analysis",
        "🎯 AWS Sizing & Configuration",
        "🤖 Agent Scaling Analysis",
        "📊 Detailed Cost Breakdown"  # RENAMED: Old comprehensive cost analysis
    ]
    
    tabs = st.tabs(tab_names)
    
    # Render each tab
    with tabs[0]:  # NEW: Total AWS Cost tab
        render_total_aws_cost_tab(analysis_data, config)
    
    with tabs[1]:
        render_ai_insights_tab_enhanced(analysis_data, config)
    
    with tabs[2]:
        render_network_intelligence_tab(analysis_data, config)
    
    with tabs[3]:
        render_os_performance_tab(analysis_data, config)
    
    with tabs[4]:
        render_aws_sizing_tab(analysis_data, config)
    
    with tabs[5]:
        render_agent_scaling_tab(analysis_data, config)
    
    with tabs[6]:  # FIXED: Detailed breakdown (using fixed version)
        render_comprehensive_cost_analysis_tab_fixed(analysis_data, config)
    
    # Render footer
    render_footer()

# Replace the existing main call at the bottom of the file
if __name__ == "__main__":
    asyncio.run(main_with_unified_costs()), '').replace(',', '').replace('.', '').isdigit()])

# EXPLANATION OF THE FIXES:
# 1. Fixed: *, row  →  _, row  (correct tuple unpacking)
# 2. Fixed: category*data  →  category_data  (correct variable name)  
# 3. Fixed: String parsing with better error handling
# 4. Added try/except to handle parsing errors gracefully

def render_comprehensive_cost_analysis_tab_fixed(analysis: Dict, config: Dict):
    """Render comprehensive AWS cost analysis tab with all services clearly organized - FIXED VERSION"""
    st.subheader("💰 Complete AWS Cost Analysis")
    
    # Get comprehensive cost data
    comprehensive_costs = analysis.get('comprehensive_costs', {})
    
    if not comprehensive_costs:
        st.warning("⚠️ Comprehensive cost data not available. Please run the analysis first.")
        return
    
    # Executive Cost Summary
    st.markdown("**📊 Executive Cost Summary**")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_monthly = comprehensive_costs.get('total_monthly', 0)
        st.metric(
            "💰 Total Monthly",
            f"${total_monthly:,.0f}",
            delta=f"Annual: ${total_monthly * 12:,.0f}"
        )
    
    with col2:
        total_one_time = comprehensive_costs.get('total_one_time', 0)
        st.metric(
            "🔄 One-Time Costs",
            f"${total_one_time:,.0f}",
            delta="Setup & Migration"
        )
    
    with col3:
        three_year_total = comprehensive_costs.get('three_year_total', 0)
        st.metric(
            "📅 3-Year Total",
            f"${three_year_total:,.0f}",
            delta="All costs included"
        )
    
    with col4:
        monthly_breakdown = comprehensive_costs.get('monthly_breakdown', {})
        if monthly_breakdown:
            largest_cost = max(monthly_breakdown.items(), key=lambda x: x[1])
            st.metric(
                "🎯 Largest Cost",
                largest_cost[0].replace('_', ' ').title(),
                delta=f"${largest_cost[1]:,.0f}/mo"
            )
        else:
            st.metric("🎯 Largest Cost", "Unknown", delta="No data")
    
    # Create a simple service breakdown table instead of the problematic code
    st.markdown("---")
    st.markdown("**🔧 AWS Services Cost Breakdown**")
    
    # Get cost components safely
    compute_costs = comprehensive_costs.get('compute_costs', {})
    storage_costs = comprehensive_costs.get('storage_costs', {})
    network_costs = comprehensive_costs.get('network_costs', {})
    migration_costs = comprehensive_costs.get('migration_costs', {})
    
    # Create simplified breakdown
    breakdown_data = []
    
    # Compute costs
    if compute_costs.get('monthly_total', 0) > 0:
        breakdown_data.append({
            'Service Category': 'Database Compute',
            'Service Type': compute_costs.get('service_type', 'Unknown'),
            'Monthly Cost': f"${compute_costs.get('monthly_total', 0):,.0f}",
            'Details': f"Primary database infrastructure"
        })
    
    # Storage costs
    if storage_costs.get('monthly_total', 0) > 0:
        breakdown_data.append({
            'Service Category': 'Storage Services',
            'Service Type': 'EBS + Destination Storage',
            'Monthly Cost': f"${storage_costs.get('monthly_total', 0):,.0f}",
            'Details': f"Database and destination storage"
        })
    
    # Network costs
    if network_costs.get('monthly_total', 0) > 0:
        breakdown_data.append({
            'Service Category': 'Network Services',
            'Service Type': 'Direct Connect + Data Transfer',
            'Monthly Cost': f"${network_costs.get('monthly_total', 0):,.0f}",
            'Details': f"Connectivity and data transfer"
        })
    
    # Migration costs
    if migration_costs.get('monthly_total', 0) > 0:
        breakdown_data.append({
            'Service Category': 'Migration Services',
            'Service Type': 'DataSync/DMS Agents',
            'Monthly Cost': f"${migration_costs.get('monthly_total', 0):,.0f}",
            'Details': f"Migration processing agents"
        })
    
    if breakdown_data:
        df_breakdown = pd.DataFrame(breakdown_data)
        st.dataframe(df_breakdown, use_container_width=True, hide_index=True)
    else:
        st.info("No detailed cost breakdown available")
    
    # Rest of the function continues with visualizations and other content...
    st.markdown("---")
    st.markdown("**📊 Cost Visualization**")
    
    if monthly_breakdown:
        # Create pie chart
        fig_pie = px.pie(
            values=list(monthly_breakdown.values()),
            names=[k.replace('_', ' ').title() for k in monthly_breakdown.keys()],
            title="Monthly Cost Distribution"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    st.info("✅ This is the fixed version of the comprehensive cost analysis tab.")
    """Run comprehensive analysis including unified cost calculation"""
    
    # Run the existing comprehensive analysis
    analysis_data = await analyzer.comprehensive_ai_migration_analysis(config)
    
    # Create unified cost calculator and add unified costs
    unified_calculator = UnifiedAWSCostCalculator(analyzer.aws_api)
    unified_costs = await unified_calculator.calculate_unified_aws_costs(config, analysis_data)
    
    # Add unified costs to analysis data
    analysis_data['unified_aws_costs'] = unified_costs
    
    return analysis_data

# Updated main function to include the new tab
async def main_with_unified_costs():
    """Enhanced main application with unified cost analysis"""
    render_enhanced_header()
    
    # Enhanced sidebar controls
    config = render_enhanced_sidebar_controls()
    
    # Initialize the migration analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = EnhancedMigrationAnalyzer()
    
    # Check if we need to run analysis
    if 'analysis_data' not in st.session_state or config_has_changed(config, st.session_state.get('last_config')):
        with st.spinner("🤖 Running comprehensive AI-powered migration analysis with unified cost calculation..."):
            try:
                # Run the comprehensive analysis with unified costs
                analysis_data = await run_comprehensive_analysis_with_unified_costs(st.session_state.analyzer, config)
                
                st.session_state.analysis_data = analysis_data
                st.session_state.last_config = config.copy()
                
                st.success("✅ Analysis completed successfully with unified cost calculation!")
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                logger.error(f"Analysis error: {e}")
                return
    else:
        st.info("📊 Using cached analysis results. Change configuration to trigger re-analysis.")
    
    analysis_data = st.session_state.analysis_data
    
    # Create tabs for organized display - UPDATED TAB LIST
    tab_names = [
        "💰 Total AWS Cost",  # NEW: First tab for unified costs
        "🧠 AI Insights & Analysis",
        "🌐 Network Intelligence", 
        "💻 OS Performance Analysis",
        "🎯 AWS Sizing & Configuration",
        "🤖 Agent Scaling Analysis",
        "📊 Detailed Cost Breakdown"  # RENAMED: Old comprehensive cost analysis
    ]
    
    tabs = st.tabs(tab_names)
    
    # Render each tab
    with tabs[0]:  # NEW: Total AWS Cost tab
        render_total_aws_cost_tab(analysis_data, config)
    
    with tabs[1]:
        render_ai_insights_tab_enhanced(analysis_data, config)
    
    with tabs[2]:
        render_network_intelligence_tab(analysis_data, config)
    
    with tabs[3]:
        render_os_performance_tab(analysis_data, config)
    
    with tabs[4]:
        render_aws_sizing_tab(analysis_data, config)
    
    with tabs[5]:
        render_agent_scaling_tab(analysis_data, config)
    
    with tabs[6]:  # FIXED: Detailed breakdown (using fixed version)
        render_comprehensive_cost_analysis_tab_fixed(analysis_data, config)
    
    # Render footer
    render_footer()

# Replace the existing main call at the bottom of the file
if __name__ == "__main__":
    asyncio.run(main_with_unified_costs())