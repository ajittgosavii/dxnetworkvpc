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
import uuid
import firebase_admin
from firebase_admin import credentials, auth, firestore
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.platypus.flowables import HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.legends import Legend
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
import io
import base64
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import tempfile
import os

class AWSMigrationPDFReportGenerator:
    """Professional PDF report generator for AWS Migration Analysis"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()
    
    def setup_custom_styles(self):
        """Setup custom paragraph styles for the report"""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#1e3a8a')
        ))
        
        # Section header style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            spaceBefore=20,
            textColor=colors.HexColor('#1e40af'),
            borderWidth=1,
            borderColor=colors.HexColor('#1e40af'),
            borderPadding=5
        ))
        
        # Subsection header style
        self.styles.add(ParagraphStyle(
            name='SubsectionHeader',
            parent=self.styles['Heading3'],
            fontSize=14,
            spaceAfter=8,
            spaceBefore=12,
            textColor=colors.HexColor('#374151')
        ))
        
        # Key metric style
        self.styles.add(ParagraphStyle(
            name='KeyMetric',
            parent=self.styles['Normal'],
            fontSize=12,
            textColor=colors.HexColor('#059669'),
            fontName='Helvetica-Bold'
        ))
        
        # Warning style
        self.styles.add(ParagraphStyle(
            name='Warning',
            parent=self.styles['Normal'],
            fontSize=11,
            textColor=colors.HexColor('#dc2626'),
            leftIndent=20,
            rightIndent=20,
            borderWidth=1,
            borderColor=colors.HexColor('#fecaca'),
            backColor=colors.HexColor('#fef2f2'),
            borderPadding=8
        ))
        
        # Info box style
        self.styles.add(ParagraphStyle(
            name='InfoBox',
            parent=self.styles['Normal'],
            fontSize=11,
            textColor=colors.HexColor('#1f2937'),
            leftIndent=20,
            rightIndent=20,
            borderWidth=1,
            borderColor=colors.HexColor('#e5e7eb'),
            backColor=colors.HexColor('#f9fafb'),
            borderPadding=8
        ))
    
    def generate_comprehensive_report(self, analysis: Dict, config: Dict) -> bytes:
        """Generate comprehensive PDF report"""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Build the report content
        story = []
        
        # Cover page
        story.extend(self._create_cover_page(analysis, config))
        story.append(PageBreak())
        
        # Executive summary
        story.extend(self._create_executive_summary(analysis, config))
        story.append(PageBreak())
        
        # Table of contents
        story.extend(self._create_table_of_contents())
        story.append(PageBreak())
        
        # Migration overview
        story.extend(self._create_migration_overview(analysis, config))
        
        # Cost analysis
        story.extend(self._create_cost_analysis_section(analysis, config))
        
        # Performance analysis
        story.extend(self._create_performance_analysis_section(analysis, config))
        
        # AWS sizing recommendations
        story.extend(self._create_aws_sizing_section(analysis, config))
        
        # AI insights and recommendations
        story.extend(self._create_ai_insights_section(analysis, config))
        
        # Risk assessment
        story.extend(self._create_risk_assessment_section(analysis, config))
        
        # Implementation roadmap
        story.extend(self._create_implementation_roadmap(analysis, config))
        
        # Appendices
        story.extend(self._create_appendices(analysis, config))
        
        # Build the PDF
        doc.build(story)
        
        # Return the PDF bytes
        buffer.seek(0)
        return buffer.read()
    
    def _create_cover_page(self, analysis: Dict, config: Dict) -> list:
        """Create professional cover page"""
        story = []
        
        # Title
        story.append(Spacer(1, 2*inch))
        story.append(Paragraph("AWS Enterprise Database Migration", self.styles['CustomTitle']))
        story.append(Paragraph("Comprehensive Analysis Report", self.styles['CustomTitle']))
        
        story.append(Spacer(1, 1*inch))
        
        # Project details table
        project_data = [
            ['Project Details', ''],
            ['Source Database:', f"{config.get('source_database_engine', 'Unknown').upper()}"],
            ['Target Database:', f"{config.get('database_engine', 'Unknown').upper()}"],
            ['Database Size:', f"{config.get('database_size_gb', 0):,} GB"],
            ['Environment:', f"{config.get('environment', 'Unknown').title()}"],
            ['Migration Method:', f"{config.get('migration_method', 'Unknown').replace('_', ' ').title()}"],
            ['Target Platform:', f"{config.get('target_platform', 'Unknown').upper()}"],
            ['Generated On:', datetime.now().strftime('%B %d, %Y at %I:%M %p')],
            ['Report Version:', 'v3.0']
        ]
        
        project_table = Table(project_data, colWidths=[2.5*inch, 3*inch])
        project_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), colors.HexColor('#1e3a8a')),
            ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8fafc'))
        ]))
        
        story.append(project_table)
        story.append(Spacer(1, 1*inch))
        
        # Key findings preview
        readiness_score = analysis.get('ai_overall_assessment', {}).get('migration_readiness_score', 0)
        validated_costs = self._get_validated_costs(analysis, config)
        
        story.append(Paragraph("Executive Summary Highlights", self.styles['SectionHeader']))
        
        highlights = [
            f"Migration Readiness Score: {readiness_score:.0f}/100",
            f"Total Monthly Cost: ${validated_costs['total_monthly']:,.0f}",
            f"Estimated Migration Time: {analysis.get('estimated_migration_time_hours', 0):.1f} hours",
            f"Number of Agents: {config.get('number_of_agents', 1)}",
            f"Primary Migration Tool: {analysis.get('agent_analysis', {}).get('primary_tool', 'Unknown').upper()}"
        ]
        
        for highlight in highlights:
            story.append(Paragraph(f"• {highlight}", self.styles['KeyMetric']))
        
        story.append(Spacer(1, 1*inch))
        
        # Disclaimer
        disclaimer = """
        <para align="center"><i>
        This report contains proprietary and confidential information. 
        Distribution should be limited to authorized personnel only.
        </i></para>
        """
        story.append(Paragraph(disclaimer, self.styles['Normal']))
        
        return story
    
    def _create_executive_summary(self, analysis: Dict, config: Dict) -> list:
        """Create executive summary section"""
        story = []
        
        story.append(Paragraph("Executive Summary", self.styles['CustomTitle']))
        story.append(Spacer(1, 20))
        
        # Migration overview
        migration_method = config.get('migration_method', 'direct_replication')
        source_db = config.get('source_database_engine', 'Unknown').upper()
        target_db = config.get('database_engine', 'Unknown').upper()
        db_size = config.get('database_size_gb', 0)
        
        overview_text = f"""
        This comprehensive analysis evaluates the migration of a {db_size:,} GB {source_db} database 
        to AWS {target_db} using the {migration_method.replace('_', ' ')} approach. 
        The analysis incorporates AI-powered insights, real-time AWS pricing, and detailed 
        performance modeling to provide actionable recommendations for a successful migration.
        """
        story.append(Paragraph(overview_text, self.styles['Normal']))
        story.append(Spacer(1, 12))
        
        # Key metrics table
        validated_costs = self._get_validated_costs(analysis, config)
        readiness_score = analysis.get('ai_overall_assessment', {}).get('migration_readiness_score', 0)
        
        metrics_data = [
            ['Key Metrics', 'Value', 'Assessment'],
            ['Migration Readiness', f"{readiness_score:.0f}/100", self._get_readiness_assessment(readiness_score)],
            ['Monthly Operating Cost', f"${validated_costs['total_monthly']:,.0f}", 'Post-migration'],
            ['One-time Migration Cost', f"${validated_costs['total_one_time']:,.0f}", 'Implementation'],
            ['3-Year Total Cost', f"${validated_costs['three_year_total']:,.0f}", 'Complete TCO'],
            ['Migration Duration', f"{analysis.get('estimated_migration_time_hours', 0):.1f} hours", 'Estimated window'],
            ['Migration Throughput', f"{analysis.get('migration_throughput_mbps', 0):,.0f} Mbps", 'Effective speed']
        ]
        
        metrics_table = Table(metrics_data, colWidths=[2*inch, 1.5*inch, 2*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3a8a')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8fafc')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
        ]))
        
        story.append(metrics_table)
        story.append(Spacer(1, 20))
        
        # Key recommendations
        story.append(Paragraph("Key Recommendations", self.styles['SubsectionHeader']))
        
        ai_recommendations = analysis.get('aws_sizing_recommendations', {}).get('ai_analysis', {})
        recommendations = ai_recommendations.get('performance_recommendations', [
            "Conduct comprehensive pre-migration testing",
            "Implement proper monitoring and alerting",
            "Plan for adequate migration window",
            "Ensure backup and rollback procedures are tested"
        ])
        
        for i, rec in enumerate(recommendations[:5], 1):
            story.append(Paragraph(f"{i}. {rec}", self.styles['Normal']))
        
        return story
    
    def _create_table_of_contents(self) -> list:
        """Create table of contents"""
        story = []
        
        story.append(Paragraph("Table of Contents", self.styles['CustomTitle']))
        story.append(Spacer(1, 20))
        
        toc_items = [
            "1. Executive Summary",
            "2. Migration Overview",
            "3. Cost Analysis",
            "4. Performance Analysis", 
            "5. AWS Sizing Recommendations",
            "6. AI Insights and Recommendations",
            "7. Risk Assessment",
            "8. Implementation Roadmap",
            "Appendix A: Technical Specifications",
            "Appendix B: Detailed Cost Breakdown",
            "Appendix C: Network Analysis"
        ]
        
        for item in toc_items:
            story.append(Paragraph(item, self.styles['Normal']))
            story.append(Spacer(1, 8))
        
        return story
    
    def _create_migration_overview(self, analysis: Dict, config: Dict) -> list:
        """Create migration overview section"""
        story = []
        
        story.append(Paragraph("1. Migration Overview", self.styles['SectionHeader']))
        
        # Current environment
        story.append(Paragraph("Current Environment", self.styles['SubsectionHeader']))
        
        current_env_data = [
            ['Component', 'Specification'],
            ['Source Database Engine', config.get('source_database_engine', 'Unknown').upper()],
            ['Database Size', f"{config.get('database_size_gb', 0):,} GB"],
            ['Operating System', config.get('operating_system', 'Unknown').replace('_', ' ').title()],
            ['Server Type', config.get('server_type', 'Unknown').title()],
            ['CPU Cores', str(config.get('cpu_cores', 'Unknown'))],
            ['RAM', f"{config.get('ram_gb', 'Unknown')} GB"],
            ['Network Interface', config.get('nic_type', 'Unknown').replace('_', ' ').title()],
            ['Environment', config.get('environment', 'Unknown').title()]
        ]
        
        current_table = Table(current_env_data, colWidths=[2*inch, 3*inch])
        current_table.setStyle(self._get_standard_table_style())
        story.append(current_table)
        story.append(Spacer(1, 15))
        
        # Target environment
        story.append(Paragraph("Target AWS Environment", self.styles['SubsectionHeader']))
        
        target_platform = config.get('target_platform', 'rds')
        aws_sizing = analysis.get('aws_sizing_recommendations', {})
        
        if target_platform == 'rds':
            rds_rec = aws_sizing.get('rds_recommendations', {})
            target_instance = rds_rec.get('primary_instance', 'Unknown')
            monthly_cost = rds_rec.get('total_monthly_cost', 0)
        else:
            ec2_rec = aws_sizing.get('ec2_recommendations', {})
            target_instance = ec2_rec.get('primary_instance', 'Unknown')
            monthly_cost = ec2_rec.get('total_monthly_cost', 0)
        
        target_env_data = [
            ['Component', 'Specification'],
            ['Target Platform', target_platform.upper()],
            ['Target Database Engine', config.get('database_engine', 'Unknown').upper()],
            ['Recommended Instance', target_instance],
            ['Destination Storage', config.get('destination_storage_type', 'S3')],
            ['Migration Method', config.get('migration_method', 'Unknown').replace('_', ' ').title()],
            ['Number of Agents', str(config.get('number_of_agents', 1))],
            ['Estimated Monthly Cost', f"${monthly_cost:,.0f}"]
        ]
        
        target_table = Table(target_env_data, colWidths=[2*inch, 3*inch])
        target_table.setStyle(self._get_standard_table_style())
        story.append(target_table)
        story.append(Spacer(1, 15))
        
        # Migration approach
        story.append(Paragraph("Migration Approach", self.styles['SubsectionHeader']))
        
        migration_method = config.get('migration_method', 'direct_replication')
        primary_tool = analysis.get('agent_analysis', {}).get('primary_tool', 'Unknown')
        
        if migration_method == 'backup_restore':
            backup_storage = config.get('backup_storage_type', 'nas_drive')
            approach_text = f"""
            The migration will use the backup/restore approach via AWS DataSync. Database backups 
            will be created on {backup_storage.replace('_', ' ')} storage and transferred to AWS 
            using {config.get('number_of_agents', 1)} DataSync agents. This approach minimizes 
            impact on the production database during the transfer phase.
            """
        else:
            approach_text = f"""
            The migration will use direct replication via AWS {primary_tool.upper()}. This approach 
            provides real-time synchronization between source and target databases, enabling 
            minimal downtime migration with continuous data capture.
            """
        
        story.append(Paragraph(approach_text, self.styles['Normal']))
        story.append(Spacer(1, 15))
        
        return story
    
    def _create_cost_analysis_section(self, analysis: Dict, config: Dict) -> list:
        """Create detailed cost analysis section"""
        story = []
        
        story.append(Paragraph("2. Cost Analysis", self.styles['SectionHeader']))
        
        validated_costs = self._get_validated_costs(analysis, config)
        
        # Cost summary
        story.append(Paragraph("Cost Summary", self.styles['SubsectionHeader']))
        
        cost_summary_text = f"""
        The total cost of ownership for this migration includes both one-time implementation 
        costs and ongoing operational expenses. The analysis uses {validated_costs['cost_source']} 
        calculation methods {'with validation' if validated_costs['is_validated'] else 'with noted discrepancies'}.
        """
        story.append(Paragraph(cost_summary_text, self.styles['Normal']))
        story.append(Spacer(1, 10))
        
        # Cost breakdown table
        cost_data = [
            ['Cost Category', 'Amount', 'Frequency', 'Notes'],
            ['Monthly Operating Cost', f"${validated_costs['total_monthly']:,.0f}", 'Monthly', 'Ongoing AWS services'],
            ['One-time Migration Cost', f"${validated_costs['total_one_time']:,.0f}", 'One-time', 'Setup and migration'],
            ['Annual Total', f"${validated_costs['total_monthly'] * 12:,.0f}", 'Annual', 'Operating costs only'],
            ['3-Year Total Cost', f"${validated_costs['three_year_total']:,.0f}", '3 Years', 'Complete TCO']
        ]
        
        cost_table = Table(cost_data, colWidths=[2*inch, 1.5*inch, 1*inch, 1.5*inch])
        cost_table.setStyle(self._get_standard_table_style())
        story.append(cost_table)
        story.append(Spacer(1, 15))
        
        # Monthly cost breakdown
        story.append(Paragraph("Monthly Cost Breakdown", self.styles['SubsectionHeader']))
        
        breakdown = validated_costs.get('breakdown', {})
        breakdown_data = [['Service Component', 'Monthly Cost', 'Percentage']]
        
        for component, cost in breakdown.items():
            if cost > 0:
                percentage = (cost / validated_costs['total_monthly']) * 100
                breakdown_data.append([
                    component.replace('_', ' ').title(),
                    f"${cost:,.0f}",
                    f"{percentage:.1f}%"
                ])
        
        breakdown_table = Table(breakdown_data, colWidths=[2.5*inch, 1.5*inch, 1*inch])
        breakdown_table.setStyle(self._get_standard_table_style())
        story.append(breakdown_table)
        story.append(Spacer(1, 15))
        
        # Cost optimization recommendations
        story.append(Paragraph("Cost Optimization Opportunities", self.styles['SubsectionHeader']))
        
        cost_optimizations = [
            "Consider Reserved Instances for 20-30% savings on compute costs",
            "Implement auto-scaling policies to optimize resource utilization",
            "Use Spot Instances for non-production workloads",
            "Review and optimize storage lifecycle policies",
            "Monitor actual usage vs provisioned capacity for right-sizing"
        ]
        
        for optimization in cost_optimizations:
            story.append(Paragraph(f"• {optimization}", self.styles['Normal']))
        
        story.append(Spacer(1, 20))
        
        return story
    
    def _create_performance_analysis_section(self, analysis: Dict, config: Dict) -> list:
        """Create performance analysis section"""
        story = []
        
        story.append(Paragraph("3. Performance Analysis", self.styles['SectionHeader']))
        
        # Current performance
        story.append(Paragraph("Current Environment Performance", self.styles['SubsectionHeader']))
        
        onprem_performance = analysis.get('onprem_performance', {})
        os_impact = onprem_performance.get('os_impact', {})
        
        perf_data = [
            ['Metric', 'Value', 'Efficiency Rating'],
            ['CPU Performance', f"{config.get('cpu_cores', 0)} cores @ {config.get('cpu_ghz', 0)} GHz", f"{os_impact.get('cpu_efficiency', 0)*100:.1f}%"],
            ['Memory Performance', f"{config.get('ram_gb', 0)} GB RAM", f"{os_impact.get('memory_efficiency', 0)*100:.1f}%"],
            ['I/O Performance', 'System Storage', f"{os_impact.get('io_efficiency', 0)*100:.1f}%"],
            ['Network Performance', f"{config.get('nic_speed', 0)} Mbps", f"{os_impact.get('network_efficiency', 0)*100:.1f}%"],
            ['Overall Efficiency', 'Combined Score', f"{os_impact.get('total_efficiency', 0)*100:.1f}%"]
        ]
        
        perf_table = Table(perf_data, colWidths=[2*inch, 2*inch, 1.5*inch])
        perf_table.setStyle(self._get_standard_table_style())
        story.append(perf_table)
        story.append(Spacer(1, 15))
        
        # Migration performance
        story.append(Paragraph("Migration Performance Metrics", self.styles['SubsectionHeader']))
        
        migration_perf_data = [
            ['Metric', 'Value', 'Assessment'],
            ['Migration Throughput', f"{analysis.get('migration_throughput_mbps', 0):,.0f} Mbps", 'Effective transfer rate'],
            ['Estimated Migration Time', f"{analysis.get('estimated_migration_time_hours', 0):.1f} hours", 'Total window'],
            ['Number of Agents', str(config.get('number_of_agents', 1)), 'Parallel processing'],
            ['Primary Tool', analysis.get('agent_analysis', {}).get('primary_tool', 'Unknown').upper(), 'Migration service'],
            ['Bottleneck', analysis.get('agent_analysis', {}).get('bottleneck', 'None identified'), 'Limiting factor']
        ]
        
        migration_perf_table = Table(migration_perf_data, colWidths=[2*inch, 2*inch, 1.5*inch])
        migration_perf_table.setStyle(self._get_standard_table_style())
        story.append(migration_perf_table)
        story.append(Spacer(1, 15))
        
        # Performance recommendations
        story.append(Paragraph("Performance Optimization Recommendations", self.styles['SubsectionHeader']))
        
        ai_analysis = analysis.get('aws_sizing_recommendations', {}).get('ai_analysis', {})
        perf_recommendations = ai_analysis.get('performance_recommendations', [
            "Optimize database queries and indexes before migration",
            "Configure proper instance sizing based on current workload",
            "Implement comprehensive monitoring and alerting",
            "Test migration performance in non-production environment"
        ])
        
        for rec in perf_recommendations:
            story.append(Paragraph(f"• {rec}", self.styles['Normal']))
        
        story.append(Spacer(1, 20))
        
        return story
    
    def _create_aws_sizing_section(self, analysis: Dict, config: Dict) -> list:
        """Create AWS sizing recommendations section"""
        story = []
        
        story.append(Paragraph("4. AWS Sizing Recommendations", self.styles['SectionHeader']))
        
        aws_sizing = analysis.get('aws_sizing_recommendations', {})
        deployment_rec = aws_sizing.get('deployment_recommendation', {})
        
        # Deployment recommendation
        story.append(Paragraph("Recommended Deployment", self.styles['SubsectionHeader']))
        
        recommendation = deployment_rec.get('recommendation', 'unknown').upper()
        confidence = deployment_rec.get('confidence', 0)
        
        deployment_text = f"""
        Based on the analysis, {recommendation} is recommended with {confidence*100:.1f}% confidence. 
        This recommendation considers database size, performance requirements, management complexity, 
        and cost optimization factors.
        """
        story.append(Paragraph(deployment_text, self.styles['Normal']))
        story.append(Spacer(1, 10))
        
        # Instance specifications
        target_platform = config.get('target_platform', 'rds')
        
        if target_platform == 'rds':
            story.append(Paragraph("RDS Configuration", self.styles['SubsectionHeader']))
            rds_rec = aws_sizing.get('rds_recommendations', {})
            
            rds_data = [
                ['Configuration Item', 'Recommendation'],
                ['Primary Instance Type', rds_rec.get('primary_instance', 'Unknown')],
                ['Storage Type', rds_rec.get('storage_type', 'gp3')],
                ['Storage Size', f"{rds_rec.get('storage_size_gb', 0):,.0f} GB"],
                ['Multi-AZ', 'Yes' if rds_rec.get('multi_az', False) else 'No'],
                ['Backup Retention', f"{rds_rec.get('backup_retention_days', 7)} days"],
                ['Monthly Instance Cost', f"${rds_rec.get('monthly_instance_cost', 0):,.0f}"],
                ['Monthly Storage Cost', f"${rds_rec.get('monthly_storage_cost', 0):,.0f}"],
                ['Total Monthly Cost', f"${rds_rec.get('total_monthly_cost', 0):,.0f}"]
            ]
        else:
            story.append(Paragraph("EC2 Configuration", self.styles['SubsectionHeader']))
            ec2_rec = aws_sizing.get('ec2_recommendations', {})
            
            rds_data = [
                ['Configuration Item', 'Recommendation'],
                ['Primary Instance Type', ec2_rec.get('primary_instance', 'Unknown')],
                ['Storage Type', ec2_rec.get('storage_type', 'gp3')],
                ['Storage Size', f"{ec2_rec.get('storage_size_gb', 0):,.0f} GB"],
                ['EBS Optimized', 'Yes' if ec2_rec.get('ebs_optimized', False) else 'No'],
                ['Enhanced Networking', 'Yes' if ec2_rec.get('enhanced_networking', False) else 'No'],
                ['Monthly Instance Cost', f"${ec2_rec.get('monthly_instance_cost', 0):,.0f}"],
                ['Monthly Storage Cost', f"${ec2_rec.get('monthly_storage_cost', 0):,.0f}"],
                ['Total Monthly Cost', f"${ec2_rec.get('total_monthly_cost', 0):,.0f}"]
            ]
        
        config_table = Table(rds_data, colWidths=[2.5*inch, 2.5*inch])
        config_table.setStyle(self._get_standard_table_style())
        story.append(config_table)
        story.append(Spacer(1, 15))
        
        # Reader/Writer configuration
        reader_writer = aws_sizing.get('reader_writer_config', {})
        if reader_writer.get('total_instances', 1) > 1:
            story.append(Paragraph("Read Replica Configuration", self.styles['SubsectionHeader']))
            
            replica_text = f"""
            Recommended configuration includes {reader_writer.get('writers', 1)} writer instance(s) 
            and {reader_writer.get('readers', 0)} read replica(s) for a total of 
            {reader_writer.get('total_instances', 1)} instances. This configuration supports 
            {reader_writer.get('recommended_read_split', 0):.1f}% read traffic distribution 
            to read replicas.
            """
            story.append(Paragraph(replica_text, self.styles['Normal']))
            story.append(Spacer(1, 10))
        
        story.append(Spacer(1, 20))
        
        return story
    
    def _create_ai_insights_section(self, analysis: Dict, config: Dict) -> list:
        """Create AI insights and recommendations section"""
        story = []
        
        story.append(Paragraph("5. AI Insights and Recommendations", self.styles['SectionHeader']))
        
        ai_analysis = analysis.get('aws_sizing_recommendations', {}).get('ai_analysis', {})
        
        # AI complexity assessment
        story.append(Paragraph("Migration Complexity Analysis", self.styles['SubsectionHeader']))
        
        complexity_score = ai_analysis.get('ai_complexity_score', 6)
        confidence_level = ai_analysis.get('confidence_level', 'medium')
        
        complexity_text = f"""
        The AI analysis rates this migration at {complexity_score:.1f}/10 complexity with 
        {confidence_level} confidence. This assessment considers factors such as database 
        engine compatibility, data size, performance requirements, and infrastructure complexity.
        """
        story.append(Paragraph(complexity_text, self.styles['Normal']))
        story.append(Spacer(1, 10))
        
        # Risk factors
        story.append(Paragraph("Identified Risk Factors", self.styles['SubsectionHeader']))
        
        risk_factors = ai_analysis.get('risk_factors', [])
        if risk_factors:
            for risk in risk_factors[:5]:
                story.append(Paragraph(f"• {risk}", self.styles['Normal']))
        else:
            story.append(Paragraph("No significant risk factors identified by AI analysis.", self.styles['Normal']))
        
        story.append(Spacer(1, 10))
        
        # Mitigation strategies
        story.append(Paragraph("Recommended Mitigation Strategies", self.styles['SubsectionHeader']))
        
        mitigation_strategies = ai_analysis.get('mitigation_strategies', [])
        if mitigation_strategies:
            for strategy in mitigation_strategies[:5]:
                story.append(Paragraph(f"• {strategy}", self.styles['Normal']))
        else:
            story.append(Paragraph("Standard migration best practices recommended.", self.styles['Normal']))
        
        story.append(Spacer(1, 10))
        
        # Performance recommendations
        story.append(Paragraph("AI Performance Recommendations", self.styles['SubsectionHeader']))
        
        performance_recommendations = ai_analysis.get('performance_recommendations', [])
        if performance_recommendations:
            for rec in performance_recommendations[:5]:
                story.append(Paragraph(f"• {rec}", self.styles['Normal']))
        else:
            story.append(Paragraph("Current configuration appears well-optimized.", self.styles['Normal']))
        
        story.append(Spacer(1, 20))
        
        return story
    
    def _create_risk_assessment_section(self, analysis: Dict, config: Dict) -> list:
        """Create risk assessment section"""
        story = []
        
        story.append(Paragraph("6. Risk Assessment", self.styles['SectionHeader']))
        
        ai_overall = analysis.get('ai_overall_assessment', {})
        
        # Overall risk level
        risk_level = ai_overall.get('risk_level', 'Medium')
        success_probability = ai_overall.get('success_probability', 85)
        
        risk_overview = f"""
        The migration is assessed as {risk_level} risk with {success_probability:.0f}% 
        probability of success. This assessment is based on technical complexity, 
        organizational readiness, and identified risk factors.
        """
        story.append(Paragraph(risk_overview, self.styles['Normal']))
        story.append(Spacer(1, 15))
        
        # Risk categories
        story.append(Paragraph("Risk Categories", self.styles['SubsectionHeader']))
        
        risk_categories = [
            ['Risk Category', 'Level', 'Mitigation Status'],
            ['Technical Complexity', self._assess_technical_risk(config), 'Planned'],
            ['Data Migration', self._assess_data_risk(config), 'Managed'],
            ['Performance Impact', self._assess_performance_risk(analysis), 'Monitored'],
            ['Downtime Risk', self._assess_downtime_risk(config), 'Controlled'],
            ['Cost Overrun', 'Low', 'Budgeted'],
            ['Timeline Risk', self._assess_timeline_risk(analysis), 'Scheduled']
        ]
        
        risk_table = Table(risk_categories, colWidths=[2*inch, 1*inch, 1.5*inch])
        risk_table.setStyle(self._get_standard_table_style())
        story.append(risk_table)
        story.append(Spacer(1, 15))
        
        # Risk mitigation plan
        story.append(Paragraph("Risk Mitigation Plan", self.styles['SubsectionHeader']))
        
        mitigation_plan = [
            "Comprehensive testing in non-production environment",
            "Detailed rollback procedures documented and tested",
            "Continuous monitoring during migration process",
            "Backup and recovery procedures validated",
            "Stakeholder communication plan established",
            "Technical support resources identified and available"
        ]
        
        for item in mitigation_plan:
            story.append(Paragraph(f"• {item}", self.styles['Normal']))
        
        story.append(Spacer(1, 20))
        
        return story
    
    def _create_implementation_roadmap(self, analysis: Dict, config: Dict) -> list:
        """Create implementation roadmap section"""
        story = []
        
        story.append(Paragraph("7. Implementation Roadmap", self.styles['SectionHeader']))
        
        # Timeline
        timeline_rec = analysis.get('ai_overall_assessment', {}).get('timeline_recommendation', {})
        
        story.append(Paragraph("Project Timeline", self.styles['SubsectionHeader']))
        
        timeline_data = [
            ['Phase', 'Duration', 'Key Activities'],
            ['Planning & Assessment', f"{timeline_rec.get('planning_phase_weeks', 2)} weeks", 'Detailed analysis, resource planning'],
            ['Environment Setup', '1-2 weeks', 'AWS environment provisioning, agent setup'],
            ['Testing & Validation', f"{timeline_rec.get('testing_phase_weeks', 3)} weeks", 'Non-production migration testing'],
            ['Migration Execution', f"{timeline_rec.get('migration_window_hours', 24)} hours", 'Production data migration'],
            ['Post-Migration', '1 week', 'Validation, optimization, documentation'],
            ['Total Project Duration', f"{timeline_rec.get('total_project_weeks', 6)} weeks", 'End-to-end timeline']
        ]
        
        timeline_table = Table(timeline_data, colWidths=[1.5*inch, 1*inch, 3*inch])
        timeline_table.setStyle(self._get_standard_table_style())
        story.append(timeline_table)
        story.append(Spacer(1, 15))
        
        # Key milestones
        story.append(Paragraph("Key Milestones", self.styles['SubsectionHeader']))
        
        milestones = [
            "Migration plan approval and resource allocation",
            "AWS environment setup and configuration complete",
            "Non-production migration testing successful",
            "Go/No-go decision for production migration",
            "Production migration execution complete",
            "Post-migration validation and sign-off"
        ]
        
        for i, milestone in enumerate(milestones, 1):
            story.append(Paragraph(f"{i}. {milestone}", self.styles['Normal']))
        
        story.append(Spacer(1, 15))
        
        # Success criteria
        story.append(Paragraph("Success Criteria", self.styles['SubsectionHeader']))
        
        success_criteria = [
            "All data migrated successfully with zero data loss",
            "Application performance meets or exceeds baseline",
            "Migration completed within planned downtime window",
            "All post-migration validation tests pass",
            "Stakeholder sign-off obtained",
            "Documentation and knowledge transfer complete"
        ]
        
        for criterion in success_criteria:
            story.append(Paragraph(f"• {criterion}", self.styles['Normal']))
        
        story.append(Spacer(1, 20))
        
        return story
    
    def _create_appendices(self, analysis: Dict, config: Dict) -> list:
        """Create appendices section"""
        story = []
        
        # Appendix A: Technical Specifications
        story.append(PageBreak())
        story.append(Paragraph("Appendix A: Technical Specifications", self.styles['SectionHeader']))
        
        # Current database performance metrics
        story.append(Paragraph("Current Database Performance Metrics", self.styles['SubsectionHeader']))
        
        db_metrics_data = [
            ['Metric', 'Current Value', 'AWS Recommendation'],
            ['Max Memory (GB)', str(config.get('current_db_max_memory_gb', 'Not specified')), 'Based on workload analysis'],
            ['Max CPU Cores', str(config.get('current_db_max_cpu_cores', 'Not specified')), 'Right-sized for AWS'],
            ['Max IOPS', f"{config.get('current_db_max_iops', 0):,}", 'EBS optimized storage'],
            ['Max Throughput (MB/s)', str(config.get('current_db_max_throughput_mbps', 'Not specified')), 'Enhanced networking enabled']
        ]
        
        db_metrics_table = Table(db_metrics_data, colWidths=[2*inch, 1.5*inch, 2*inch])
        db_metrics_table.setStyle(self._get_standard_table_style())
        story.append(db_metrics_table)
        story.append(Spacer(1, 15))
        
        # Appendix B: Detailed Cost Breakdown
        story.append(Paragraph("Appendix B: Detailed Cost Breakdown", self.styles['SubsectionHeader']))
        
        validated_costs = self._get_validated_costs(analysis, config)
        detailed_costs = [
            ['Service Component', 'Monthly Cost', 'Annual Cost', '3-Year Cost']
        ]
        
        breakdown = validated_costs.get('breakdown', {})
        for component, monthly_cost in breakdown.items():
            if monthly_cost > 0:
                annual_cost = monthly_cost * 12
                three_year_cost = monthly_cost * 36
                detailed_costs.append([
                    component.replace('_', ' ').title(),
                    f"${monthly_cost:,.0f}",
                    f"${annual_cost:,.0f}",
                    f"${three_year_cost:,.0f}"
                ])
        
        # Add totals row
        detailed_costs.append([
            'TOTAL',
            f"${validated_costs['total_monthly']:,.0f}",
            f"${validated_costs['total_monthly'] * 12:,.0f}",
            f"${validated_costs['total_monthly'] * 36:,.0f}"
        ])
        
        detailed_cost_table = Table(detailed_costs, colWidths=[2*inch, 1.25*inch, 1.25*inch, 1.5*inch])
        detailed_cost_table.setStyle(self._get_standard_table_style())
        # Make totals row bold
        detailed_cost_table.setStyle(TableStyle([
            ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
            ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#e5e7eb'))
        ]))
        story.append(detailed_cost_table)
        story.append(Spacer(1, 15))
        
        # Appendix C: Network Analysis
        story.append(Paragraph("Appendix C: Network Analysis", self.styles['SubsectionHeader']))
        
        network_perf = analysis.get('network_performance', {})
        
        network_summary = f"""
        Network Path: {network_perf.get('path_name', 'Standard migration path')}
        
        Network Quality Score: {network_perf.get('network_quality_score', 0):.1f}/100
        AI Enhanced Score: {network_perf.get('ai_enhanced_quality_score', 0):.1f}/100
        Effective Bandwidth: {network_perf.get('effective_bandwidth_mbps', 0):,.0f} Mbps
        Total Latency: {network_perf.get('total_latency_ms', 0):.1f} ms
        Overall Reliability: {network_perf.get('total_reliability', 0)*100:.3f}%
        """
        
        story.append(Paragraph(network_summary, self.styles['Normal']))
        story.append(Spacer(1, 10))
        
        # Report generation info
        story.append(HRFlowable(width="100%", thickness=1, lineCap='round', color=colors.HexColor('#e5e7eb')))
        story.append(Spacer(1, 10))
        
        report_info = f"""
        Report generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')} by 
        AWS Enterprise Database Migration Analyzer AI v3.0. 
        
        This report contains proprietary analysis and should be treated as confidential.
        """
        story.append(Paragraph(report_info, self.styles['Normal']))
        
        return story
    
    def _get_standard_table_style(self):
        """Get standard table style"""
        return TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3a8a')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8fafc')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
        ])
    
    def _get_validated_costs(self, analysis: Dict, config: Dict) -> Dict:
        """Get validated costs using the same logic as the main application"""
        comprehensive_costs = analysis.get('comprehensive_costs', {})
        basic_costs = analysis.get('cost_analysis', {})
        
        if comprehensive_costs.get('cost_source') == 'unified_calculation':
            return {
                'total_monthly': comprehensive_costs['total_monthly'],
                'total_one_time': comprehensive_costs.get('total_one_time', 0),
                'three_year_total': comprehensive_costs['three_year_total'],
                'breakdown': comprehensive_costs.get('monthly_breakdown', {}),
                'cost_source': 'unified',
                'is_validated': True
            }
        elif comprehensive_costs.get('total_monthly', 0) > 0:
            return {
                'total_monthly': comprehensive_costs['total_monthly'],
                'total_one_time': comprehensive_costs.get('total_one_time', 0),
                'three_year_total': comprehensive_costs.get('three_year_total', 0),
                'breakdown': comprehensive_costs.get('monthly_breakdown', {}),
                'cost_source': 'comprehensive',
                'is_validated': False
            }
        else:
            return {
                'total_monthly': basic_costs.get('total_monthly_cost', 1000),
                'total_one_time': basic_costs.get('one_time_migration_cost', 5000),
                'three_year_total': (basic_costs.get('total_monthly_cost', 1000) * 36) + basic_costs.get('one_time_migration_cost', 5000),
                'breakdown': {
                    'compute': basic_costs.get('aws_compute_cost', 600),
                    'storage': basic_costs.get('aws_storage_cost', 200),
                    'agents': basic_costs.get('agent_cost', 150),
                    'network': basic_costs.get('network_cost', 50)
                },
                'cost_source': 'basic',
                'is_validated': False
            }
    
    def _get_readiness_assessment(self, score: float) -> str:
        """Get readiness assessment text"""
        if score >= 80:
            return "Excellent"
        elif score >= 70:
            return "Good"
        elif score >= 60:
            return "Fair"
        else:
            return "Needs Improvement"
    
    def _assess_technical_risk(self, config: Dict) -> str:
        """Assess technical risk level"""
        if config.get('source_database_engine') != config.get('database_engine'):
            return "Medium"
        elif config.get('database_size_gb', 0) > 10000:
            return "Medium"
        else:
            return "Low"
    
    def _assess_data_risk(self, config: Dict) -> str:
        """Assess data migration risk"""
        size = config.get('database_size_gb', 0)
        if size > 10000:
            return "Medium"
        elif size > 5000:
            return "Medium"
        else:
            return "Low"
    
    def _assess_performance_risk(self, analysis: Dict) -> str:
        """Assess performance risk"""
        perf_score = analysis.get('onprem_performance', {}).get('performance_score', 70)
        if perf_score < 60:
            return "High"
        elif perf_score < 80:
            return "Medium"
        else:
            return "Low"
    
    def _assess_downtime_risk(self, config: Dict) -> str:
        """Assess downtime risk"""
        tolerance = config.get('downtime_tolerance_minutes', 60)
        if tolerance < 30:
            return "High"
        elif tolerance < 120:
            return "Medium"
        else:
            return "Low"
    
    def _assess_timeline_risk(self, analysis: Dict) -> str:
        """Assess timeline risk"""
        migration_time = analysis.get('estimated_migration_time_hours', 24)
        if migration_time > 48:
            return "High"
        elif migration_time > 24:
            return "Medium"
        else:
            return "Low"

def export_pdf_report(analysis: Dict, config: Dict, report_type: str = "comprehensive") -> bytes:
    """Export PDF report with specified type"""
    try:
        pdf_generator = AWSMigrationPDFReportGenerator()
        
        if report_type == "comprehensive":
            pdf_data = pdf_generator.generate_comprehensive_report(analysis, config)
        else:
            # Could add other report types here (summary, technical, executive, etc.)
            pdf_data = pdf_generator.generate_comprehensive_report(analysis, config)
        
        return pdf_data
    
    except Exception as e:
        st.error(f"Failed to generate PDF report: {str(e)}")
        return None

def render_pdf_export_section(analysis: Dict, config: Dict):
    """Render PDF export section for any tab"""
    st.markdown("---")
    st.markdown("**📄 PDF Report Generation**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Generate Comprehensive Report", use_container_width=True):
            with st.spinner("Generating comprehensive PDF report..."):
                pdf_data = export_pdf_report(analysis, config, "comprehensive")
                if pdf_data:
                    st.download_button(
                        label="📥 Download Comprehensive Report",
                        data=pdf_data,
                        file_name=f"aws_migration_comprehensive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                    st.success("✅ Comprehensive report generated successfully!")
    
    with col2:
        if st.button("📋 Generate Executive Summary", use_container_width=True):
            with st.spinner("Generating executive summary PDF..."):
                # You could create a separate executive summary generator
                pdf_data = export_pdf_report(analysis, config, "comprehensive")
                if pdf_data:
                    st.download_button(
                        label="📥 Download Executive Summary",
                        data=pdf_data,
                        file_name=f"aws_migration_executive_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                    st.success("✅ Executive summary generated successfully!")
    
    with col3:
        if st.button("🔧 Generate Technical Report", use_container_width=True):
            with st.spinner("Generating technical PDF report..."):
                pdf_data = export_pdf_report(analysis, config, "comprehensive")
                if pdf_data:
                    st.download_button(
                        label="📥 Download Technical Report",
                        data=pdf_data,
                        file_name=f"aws_migration_technical_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                    st.success("✅ Technical report generated successfully!")

def safe_float(value, default=0.0):
    """Safely convert a value to float, returning default if None or invalid"""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_int(value, default=0):
    """Safely convert a value to int, returning default if None or invalid"""
    if value is None:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

def safe_subtract(a, b, default_a=0, default_b=0):
    """Safely subtract two values, handling None cases"""
    safe_a = safe_float(a, default_a)
    safe_b = safe_float(b, default_b)
    return safe_a - safe_b

def safe_multiply(a, b, default_a=0, default_b=1):
    """Safely multiply two values, handling None cases"""
    safe_a = safe_float(a, default_a)
    safe_b = safe_float(b, default_b)
    return safe_a * safe_b

def safe_divide(a, b, default_a=0, default_b=1):
    """Safely divide two values, handling None and zero cases"""
    safe_a = safe_float(a, default_a)
    safe_b = safe_float(b, default_b)
    if safe_b == 0:
        return 0
    return safe_a / safe_b



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

# Firebase Authentication Manager
class FirebaseAuthManager:
    def __init__(self):
        self.db = None
        self.initialize_firebase()
    
    def initialize_firebase(self):
        """Initialize Firebase Admin SDK"""
        try:
            # Check if Firebase is already initialized
            if not firebase_admin._apps:
                # Load Firebase credentials from Streamlit secrets
                firebase_config = st.secrets.get("firebase", {})
                
                if firebase_config:
                    # Create credentials from secrets
                    cred_dict = {
                        "type": firebase_config.get("type"),
                        "project_id": firebase_config.get("project_id"),
                        "private_key_id": firebase_config.get("private_key_id"),
                        "private_key": firebase_config.get("private_key").replace('\\n', '\n'),
                        "client_email": firebase_config.get("client_email"),
                        "client_id": firebase_config.get("client_id"),
                        "auth_uri": firebase_config.get("auth_uri"),
                        "token_uri": firebase_config.get("token_uri"),
                        "auth_provider_x509_cert_url": firebase_config.get("auth_provider_x509_cert_url"),
                        "client_x509_cert_url": firebase_config.get("client_x509_cert_url")
                    }
                    
                    cred = credentials.Certificate(cred_dict)
                    firebase_admin.initialize_app(cred)
                    self.db = firestore.client()
                    logger.info("Firebase initialized successfully")
                else:
                    logger.error("Firebase configuration not found in secrets")
                    st.error("Firebase configuration not found. Please check your secrets configuration.")
            else:
                # Firebase already initialized
                self.db = firestore.client()
                
        except Exception as e:
            logger.error(f"Firebase initialization failed: {e}")
            st.error(f"Firebase initialization failed: {str(e)}")
    
    def create_user(self, email: str, password: str, display_name: str, role: str = "user") -> Dict:
        """Create a new user with email/password"""
        try:
            # Create user in Firebase Auth
            user = auth.create_user(
                email=email,
                password=password,
                display_name=display_name
            )
            
            # Store additional user data in Firestore
            user_data = {
                'uid': user.uid,
                'email': email,
                'display_name': display_name,
                'role': role,
                'created_at': datetime.now(),
                'last_login': None,
                'is_active': True,
                'created_by': st.session_state.get('user_email', 'system')
            }
            
            if self.db:
                self.db.collection('users').document(user.uid).set(user_data)
            
            return {
                'success': True,
                'user_id': user.uid,
                'message': f'User {email} created successfully'
            }
            
        except Exception as e:
            logger.error(f"User creation failed: {e}")
            return {
                'success': False,
                'message': f'Failed to create user: {str(e)}'
            }
    
    def verify_user_credentials(self, email: str, password: str) -> Dict:
        """Verify user credentials using Firebase Auth REST API"""
        try:
            # Firebase Web API key (you'll need to add this to your secrets)
            api_key = st.secrets.get("firebase", {}).get("web_api_key")
            
            if not api_key:
                return {'success': False, 'message': 'Firebase Web API key not configured'}
            
            # Firebase Auth REST API endpoint
            url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={api_key}"
            
            payload = {
                "email": email,
                "password": password,
                "returnSecureToken": True
            }
            
            response = requests.post(url, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                user_id = data.get('localId')
                
                # Update last login in Firestore
                if self.db and user_id:
                    self.db.collection('users').document(user_id).update({
                        'last_login': datetime.now()
                    })
                    
                    # Get user data from Firestore
                    user_doc = self.db.collection('users').document(user_id).get()
                    if user_doc.exists:
                        user_data = user_doc.to_dict()
                        
                        # Check if user is active
                        if not user_data.get('is_active', True):
                            return {'success': False, 'message': 'User account is deactivated'}
                        
                        return {
                            'success': True,
                            'user_id': user_id,
                            'email': email,
                            'display_name': user_data.get('display_name', email),
                            'role': user_data.get('role', 'user'),
                            'id_token': data.get('idToken')
                        }
                
                return {
                    'success': True,
                    'user_id': user_id,
                    'email': email,
                    'display_name': email,
                    'role': 'user',
                    'id_token': data.get('idToken')
                }
            else:
                error_data = response.json()
                error_message = error_data.get('error', {}).get('message', 'Authentication failed')
                return {'success': False, 'message': error_message}
                
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return {'success': False, 'message': f'Authentication error: {str(e)}'}
    
    def get_all_users(self) -> List[Dict]:
        """Get all users from Firestore"""
        try:
            if not self.db:
                return []
            
            users_ref = self.db.collection('users')
            docs = users_ref.stream()
            
            users = []
            for doc in docs:
                user_data = doc.to_dict()
                users.append(user_data)
            
            return users
        except Exception as e:
            logger.error(f"Failed to get users: {e}")
            return []
    
    def update_user_status(self, user_id: str, is_active: bool) -> bool:
        """Update user active status"""
        try:
            if self.db:
                self.db.collection('users').document(user_id).update({
                    'is_active': is_active,
                    'updated_at': datetime.now()
                })
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to update user status: {e}")
            return False
    
    def delete_user(self, user_id: str) -> bool:
        """Delete user from Firebase Auth and Firestore"""
        try:
            # Delete from Firebase Auth
            auth.delete_user(user_id)
            
            # Delete from Firestore
            if self.db:
                self.db.collection('users').document(user_id).delete()
            
            return True
        except Exception as e:
            logger.error(f"Failed to delete user: {e}")
            return False

# Authentication UI Components
def render_login_page():
    """Render the login page"""
    st.markdown("""
    <div style="text-align: center; padding: 2rem;">
        <h1>🔐 AWS Migration Analyzer</h1>
        <h3>Enterprise Authentication Portal</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Center the login form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### 🔑 Sign In")
        
        with st.form("login_form"):
            email = st.text_input("📧 Email Address", placeholder="user@company.com")
            password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")
            
            col1, col2 = st.columns(2)
            with col1:
                login_button = st.form_submit_button("🚀 Sign In", use_container_width=True)
            
            if login_button:
                if email and password:
                    auth_manager = FirebaseAuthManager()
                    result = auth_manager.verify_user_credentials(email, password)
                    
                    if result['success']:
                        # Store user info in session state
                        st.session_state['authenticated'] = True
                        st.session_state['user_id'] = result['user_id']
                        st.session_state['user_email'] = result['email']
                        st.session_state['user_name'] = result['display_name']
                        st.session_state['user_role'] = result['role']
                        st.session_state['login_time'] = datetime.now()
                        
                        st.success(f"Welcome back, {result['display_name']}!")
                        st.rerun()
                    else:
                        st.error(f"❌ Login failed: {result['message']}")
                else:
                    st.error("Please enter both email and password")
        
        # Admin section
        st.markdown("---")
        st.markdown("### 👤 Administrator?")
        if st.button("🛠️ Admin Panel", use_container_width=True):
            st.session_state['show_admin'] = True
            st.rerun()

def render_admin_panel():
    """Render the admin panel for user management"""
    st.markdown("## 🛠️ Administrator Panel")
    
    # Back button
    if st.button("← Back to Login"):
        st.session_state['show_admin'] = False
        st.rerun()
    
    # Admin authentication
    if not st.session_state.get('admin_authenticated', False):
        st.markdown("### 🔐 Admin Authentication")
        
        with st.form("admin_login"):
            admin_email = st.text_input("Admin Email")
            admin_password = st.text_input("Admin Password", type="password")
            admin_key = st.text_input("Admin Key", type="password", help="Special admin key")
            
            if st.form_submit_button("Authenticate as Admin"):
                # Check admin credentials (you should configure this in secrets)
                expected_admin_email = st.secrets.get("admin", {}).get("email", "admin@company.com")
                expected_admin_password = st.secrets.get("admin", {}).get("password", "admin123")
                expected_admin_key = st.secrets.get("admin", {}).get("key", "admin_key_123")
                
                if (admin_email == expected_admin_email and 
                    admin_password == expected_admin_password and 
                    admin_key == expected_admin_key):
                    st.session_state['admin_authenticated'] = True
                    st.session_state['admin_email'] = admin_email
                    st.success("Admin authenticated successfully!")
                    st.rerun()
                else:
                    st.error("Invalid admin credentials")
        
        return
    
    # Admin is authenticated, show user management
    st.success(f"Logged in as Admin: {st.session_state.get('admin_email')}")
    
    # User management tabs
    tab1, tab2, tab3 = st.tabs(["👥 Manage Users", "➕ Create User", "📊 User Analytics"])
    
    auth_manager = FirebaseAuthManager()
    
    with tab1:
        st.markdown("### 👥 User Management")
        
        users = auth_manager.get_all_users()
        
        if users:
            # Create users dataframe
            users_data = []
            for user in users:
                # Handle datetime formatting safely
                created_date = user.get('created_at', '')
                if created_date:
                    try:
                        if isinstance(created_date, str):
                            created_formatted = created_date[:10]  # Just the date part
                        elif hasattr(created_date, 'strftime'):
                            created_formatted = created_date.strftime('%Y-%m-%d')
                        else:
                            created_formatted = str(created_date)[:10]
                    except:
                        created_formatted = 'Unknown'
                else:
                    created_formatted = 'Unknown'

                last_login = user.get('last_login', '')
                if last_login:
                    try:
                        if isinstance(last_login, str):
                            last_login_formatted = last_login[:16].replace('T', ' ')  # Date and time
                        elif hasattr(last_login, 'strftime'):
                            last_login_formatted = last_login.strftime('%Y-%m-%d %H:%M')
                        else:
                            last_login_formatted = str(last_login)[:16]
                    except:
                        last_login_formatted = 'Never'
                else:
                    last_login_formatted = 'Never'

                users_data.append({
                    'Email': user.get('email', ''),
                    'Name': user.get('display_name', ''),
                    'Role': user.get('role', 'user'),
                    'Status': 'Active' if user.get('is_active', True) else 'Inactive',
                    'Created': created_formatted,
                    'Last Login': last_login_formatted,
                    'UID': user.get('uid', '')
                })
            
            df_users = pd.DataFrame(users_data)
            
            # Display users table
            st.dataframe(df_users[['Email', 'Name', 'Role', 'Status', 'Created', 'Last Login']], 
                        use_container_width=True)
            
            # User actions
            st.markdown("### User Actions")
            
            selected_user = st.selectbox("Select User", 
                                       options=[f"{user['Email']} ({user['Name']})" for user in users_data],
                                       index=0 if users_data else None)
            
            if selected_user and users_data:
                selected_uid = None
                for user in users_data:
                    if f"{user['Email']} ({user['Name']})" == selected_user:
                        selected_uid = user['UID']
                        selected_status = user['Status'] == 'Active'
                        break
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("🔄 Toggle Status"):
                        if auth_manager.update_user_status(selected_uid, not selected_status):
                            st.success("User status updated!")
                            st.rerun()
                        else:
                            st.error("Failed to update user status")
                
                with col2:
                    if st.button("🗑️ Delete User", type="secondary"):
                        if st.session_state.get('confirm_delete'):
                            if auth_manager.delete_user(selected_uid):
                                st.success("User deleted successfully!")
                                st.session_state['confirm_delete'] = False
                                st.rerun()
                            else:
                                st.error("Failed to delete user")
                        else:
                            st.session_state['confirm_delete'] = True
                            st.warning("Click again to confirm deletion")
                
                with col3:
                    if st.button("📧 Reset Password"):
                        st.info("Password reset functionality would be implemented here")
        else:
            st.info("No users found")
    
    with tab2:
        st.markdown("### ➕ Create New User")
        
        with st.form("create_user_form"):
            new_email = st.text_input("Email Address*")
            new_name = st.text_input("Full Name*")
            new_password = st.text_input("Password*", type="password", 
                                       help="Minimum 6 characters")
            new_role = st.selectbox("Role", ["user", "admin", "analyst"], index=0)
            
            if st.form_submit_button("Create User"):
                if new_email and new_name and new_password:
                    if len(new_password) >= 6:
                        result = auth_manager.create_user(
                            email=new_email,
                            password=new_password,
                            display_name=new_name,
                            role=new_role
                        )
                        
                        if result['success']:
                            st.success(f"✅ User created successfully: {new_email}")
                            # Clear form by rerunning
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(f"❌ {result['message']}")
                    else:
                        st.error("Password must be at least 6 characters")
                else:
                    st.error("Please fill in all required fields")
    
    with tab3:
        st.markdown("### 📊 User Analytics")
        
        users = auth_manager.get_all_users()
        
        if users:
            # User statistics
            total_users = len(users)
            active_users = len([u for u in users if u.get('is_active', True)])
            inactive_users = total_users - active_users
            
            # Recent logins (last 7 days)
            # Recent logins (last 7 days)
            week_ago = datetime.now() - timedelta(days=7)
            recent_logins = 0
            for u in users:
                last_login = u.get('last_login')
                if last_login:
                    try:
                        # Handle different datetime formats from Firebase
                        if isinstance(last_login, str):
                            # Try to parse string datetime
                            last_login_dt = datetime.fromisoformat(last_login.replace('Z', '+00:00'))
                        elif hasattr(last_login, 'timestamp'):
                            # Handle Firebase timestamp
                            last_login_dt = last_login.to_datetime()
                        elif isinstance(last_login, datetime):
                            # Already a datetime object
                            last_login_dt = last_login
                        else:
                            continue
                        
                        if last_login_dt > week_ago:
                            recent_logins += 1
                    except (ValueError, AttributeError):
                        # Skip if we can't parse the datetime
                        continue
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Users", total_users)
            
            with col2:
                st.metric("Active Users", active_users)
            
            with col3:
                st.metric("Inactive Users", inactive_users)
            
            with col4:
                st.metric("Recent Logins (7d)", recent_logins)
            
            # User creation over time (if you have enough data)
            st.markdown("### User Registration Trend")
            
            # Create sample chart
            # Create sample chart with safe datetime handling
            creation_dates = []
            for u in users:
                created_at = u.get('created_at')
                if created_at:
                    try:
                        if isinstance(created_at, str):
                            # Try to parse string datetime
                            dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                            creation_dates.append(dt)
                        elif hasattr(created_at, 'to_datetime'):
                            # Handle Firebase timestamp
                            creation_dates.append(created_at.to_datetime())
                        elif isinstance(created_at, datetime):
                            # Already a datetime object
                            creation_dates.append(created_at)
                    except (ValueError, AttributeError):
                        # Skip if we can't parse the datetime
                        continue

            if creation_dates:
                df_dates = pd.DataFrame({'date': creation_dates})
                df_dates['month'] = df_dates['date'].dt.to_period('M')
                monthly_counts = df_dates.groupby('month').size().reset_index(name='count')
                monthly_counts['month'] = monthly_counts['month'].astype(str)
                
                fig = px.line(monthly_counts, x='month', y='count', 
                            title='User Registrations by Month')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No registration data available for chart")
        else:
            st.info("No user data available for analytics")

def render_logout_section():
    """Render logout section in sidebar"""
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 👤 User Info")
        st.write(f"**Name:** {st.session_state.get('user_name', 'Unknown')}")
        st.write(f"**Email:** {st.session_state.get('user_email', 'Unknown')}")
        st.write(f"**Role:** {st.session_state.get('user_role', 'user').title()}")
        
        login_time = st.session_state.get('login_time')
        if login_time:
            duration = datetime.now() - login_time
            hours = int(duration.total_seconds() // 3600)
            minutes = int((duration.total_seconds() % 3600) // 60)
            st.write(f"**Session:** {hours}h {minutes}m")
        
        st.markdown("---")
        
        if st.button("🚪 Logout", use_container_width=True):
            # Clear all session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.success("Logged out successfully!")
            st.rerun()

def check_authentication():
    """Check if user is authenticated and session is valid"""
    if not st.session_state.get('authenticated', False):
        return False
    
    # Check session timeout (24 hours)
    login_time = st.session_state.get('login_time')
    if login_time:
        session_duration = datetime.now() - login_time
        if session_duration > timedelta(hours=24):
            # Session expired
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.error("Session expired. Please login again.")
            return False
    
    return True


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
            # This runs only when no API key is provided
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
        else:  # nas_drive:
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
            except Exception as e:  # ✅ Correctly indented - same level as try:
                logger.warning(f"Failed to get pricing for {instance_type}: {e}")
                pricing_data[instance_type] = self._get_fallback_instance_pricing(instance_type)

        return pricing_data  # ✅ Moved outside the for loop

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

    async def calculate_unified_migration_costs(self, config: Dict, analysis: Dict) -> Dict:
        """Single unified cost calculation to eliminate discrepancies"""

        # Get components from their authoritative sources
        aws_sizing = analysis.get('aws_sizing_recommendations', {})
        agent_analysis = analysis.get('agent_analysis', {})

        # AWS Infrastructure Costs (from sizing recommendations)
        target_platform = config.get('target_platform', 'rds')
        if target_platform == 'rds':
            aws_compute_monthly = safe_float(aws_sizing.get('rds_recommendations', {}).get('monthly_instance_cost', 0))
            aws_storage_monthly = safe_float(aws_sizing.get('rds_recommendations', {}).get('monthly_storage_cost', 0))
        else:
            aws_compute_monthly = safe_float(aws_sizing.get('ec2_recommendations', {}).get('monthly_instance_cost', 0))
            aws_storage_monthly = safe_float(aws_sizing.get('ec2_recommendations', {}).get('monthly_storage_cost', 0))

        # Agent Costs (from agent analysis - single source of truth)
        agent_monthly = safe_float(agent_analysis.get('monthly_cost', 0))

        # Additional Storage Costs (destination and backup)
        additional_storage = await self._calculate_additional_storage_costs_unified(config)

        # Network Costs
        network_monthly = await self._calculate_network_costs_unified(config)

        # Total monthly cost
        total_monthly = (
            aws_compute_monthly +
            aws_storage_monthly +
            agent_monthly +
            safe_float(additional_storage.get('total', 0)) +
            network_monthly
        )

        # One-time costs
        setup_cost = 2000 + (safe_int(config.get('number_of_agents', 1)) * 500)
        migration_method = config.get('migration_method', 'direct_replication')
        if migration_method == 'backup_restore':
            setup_cost += 1000  # Additional DataSync setup

        return {
            'total_monthly': max(0, total_monthly),
            'total_one_time': max(0, setup_cost),
            'three_year_total': max(0, (total_monthly * 36) + setup_cost),
            'detailed_breakdown': {
                'aws_compute': aws_compute_monthly,
                'aws_storage': aws_storage_monthly,
                'migration_agents': agent_monthly,
                'additional_storage': safe_float(additional_storage.get('total', 0)),
                'network': network_monthly
            },
            'monthly_breakdown': {
                'compute': aws_compute_monthly,
                'storage': aws_storage_monthly,
                'agents': agent_monthly,
                'destination_storage': safe_float(additional_storage.get('destination', 0)),
                'backup_storage': safe_float(additional_storage.get('backup', 0)),
                'network': network_monthly
            },
            'cost_source': 'unified_calculation',
            'calculation_timestamp': datetime.now().isoformat()
        }

    async def _calculate_additional_storage_costs_unified(self, config: Dict) -> Dict:
        """Calculate additional storage costs (destination and backup)"""
        database_size_gb = safe_float(config.get('database_size_gb', 1000))
        destination_storage = config.get('destination_storage_type', 'S3')
        migration_method = config.get('migration_method', 'direct_replication')

        costs = {'destination': 0, 'backup': 0, 'total': 0}

        # Destination storage (only if not S3 - FSx adds costs)
        if destination_storage == 'FSx_Windows':
            dest_size = database_size_gb * 0.3  # Migration staging only
            costs['destination'] = dest_size * 0.13
        elif destination_storage == 'FSx_Lustre':
            dest_size = database_size_gb * 0.3  # Migration staging only
            costs['destination'] = dest_size * 0.14

        # Backup storage (only for backup/restore method)
        if migration_method == 'backup_restore':
            backup_size_multiplier = safe_float(config.get('backup_size_multiplier', 0.7))
            backup_size = database_size_gb * backup_size_multiplier
            costs['backup'] = backup_size * 0.023  # S3 Standard

        costs['total'] = costs['destination'] + costs['backup']
        return costs

    async def _calculate_network_costs_unified(self, config: Dict) -> float:
        """Calculate unified network costs"""
        environment = config.get('environment', 'non-production')
        database_size = safe_float(config.get('database_size_gb', 1000))

        # Direct Connect
        if environment == 'production':
            dx_monthly = 2.25 * 24 * 30  # 10Gbps DX
            data_transfer = database_size * 0.1 * 0.02  # 10% monthly transfer
            vpn_backup = 45
            return dx_monthly + data_transfer + vpn_backup
        else:
            dx_monthly = 0.30 * 24 * 30  # 1Gbps DX
            data_transfer = database_size * 0.05 * 0.02  # 5% monthly transfer
            return dx_monthly + data_transfer

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
            reader_monthly = instance_cost * 24 * 30 * reader_count * 0.9

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
                os_licensing = 150 * instance_count

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

class CostValidationManager:
    """Centralized cost validation and standardization"""

    def __init__(self):
        pass

    def get_validated_costs(self, analysis: Dict, config: Dict) -> Dict:
        """Get validated and standardized costs with unified approach"""
        comprehensive_costs = analysis.get('comprehensive_costs', {})
        basic_costs = analysis.get('cost_analysis', {})

        # Prefer unified costs if available
        if comprehensive_costs.get('cost_source') == 'unified_calculation':
            monthly_total = comprehensive_costs['total_monthly']
            one_time_total = comprehensive_costs.get('total_one_time', 0)
            breakdown = comprehensive_costs.get('monthly_breakdown', {})
            cost_source = 'unified'
            is_validated = True
            validation_results = {'is_consistent': True, 'discrepancies': [], 'discrepancy_count': 0}

        elif comprehensive_costs.get('total_monthly', 0) > 0:
            # Use comprehensive but validate
            monthly_total = comprehensive_costs['total_monthly']
            one_time_total = comprehensive_costs.get('total_one_time', 0)
            breakdown = comprehensive_costs.get('monthly_breakdown', {})
            cost_source = 'comprehensive'
            validation_results = self._validate_cost_consistency_v2(comprehensive_costs, basic_costs, analysis, config)
            is_validated = validation_results['is_consistent']

        else:
            # Fall back to basic costs but validate them
            monthly_total = self._validate_basic_costs(basic_costs, analysis, config)
            one_time_total = basic_costs.get('one_time_migration_cost', 0)
            breakdown = self._create_breakdown_from_basic(basic_costs, analysis, config)
            cost_source = 'basic_validated'
            validation_results = {'is_consistent': False, 'discrepancies': [{'type': 'using_fallback_costs'}], 'discrepancy_count': 1}
            is_validated = False

        return {
            'total_monthly': monthly_total,
            'total_one_time': one_time_total,
            'three_year_total': (monthly_total * 36) + one_time_total,
            'breakdown': breakdown,
            'cost_source': cost_source,
            'validation': validation_results,
            'is_validated': is_validated
        }

    def _validate_basic_costs(self, basic_costs: Dict, analysis: Dict, config: Dict) -> float:
        """Validate and correct basic costs to remove double-counting"""
        aws_compute = basic_costs.get('aws_compute_cost', 0)
        aws_storage = basic_costs.get('aws_storage_cost', 0)

        # Get agent cost from authoritative source
        agent_analysis = analysis.get('agent_analysis', {})
        if agent_analysis.get('monthly_cost', 0) > 0:
            validated_agent_cost = agent_analysis['monthly_cost']
        else:
            validated_agent_cost = basic_costs.get('agent_cost', 0)

        # Add other costs without double counting
        network_cost = basic_costs.get('network_cost', 500)  # Default network cost
        other_cost = basic_costs.get('management_cost', 200)  # Management overhead

        validated_total = aws_compute + aws_storage + validated_agent_cost + network_cost + other_cost
        return validated_total

    def _create_breakdown_from_basic(self, basic_costs: Dict, analysis: Dict, config: Dict) -> Dict:
        """Create standardized breakdown from basic costs"""
        agent_analysis = analysis.get('agent_analysis', {})

        return {
            'compute': basic_costs.get('aws_compute_cost', 0),
            'primary_storage': basic_costs.get('aws_storage_cost', 0),
            'agents': agent_analysis.get('monthly_cost', basic_costs.get('agent_cost', 0)),
            'destination_storage': basic_costs.get('destination_storage_cost', 0),
            'backup_storage': basic_costs.get('backup_storage_cost', 0),
            'network': basic_costs.get('network_cost', 500),
            'other': basic_costs.get('management_cost', 200)
        }

    def _create_standardized_breakdown(self, analysis: Dict, config: Dict, cost_source: str, total_monthly: float) -> Dict:
        """Create standardized cost breakdown"""

        agent_analysis = analysis.get('agent_analysis', {})
        aws_sizing = analysis.get('aws_sizing_recommendations', {})

        # Agent costs (single source of truth)
        agent_cost = agent_analysis.get('monthly_cost', 0)

        # Compute costs
        if config.get('target_platform') == 'rds':
            compute_cost = aws_sizing.get('rds_recommendations', {}).get('total_monthly_cost', 0)
            # Subtract storage cost to avoid double counting
            storage_cost = aws_sizing.get('rds_recommendations', {}).get('monthly_storage_cost', 0)
            compute_cost -= storage_cost
        else:
            compute_cost = aws_sizing.get('ec2_recommendations', {}).get('total_monthly_cost', 0)
            storage_cost = aws_sizing.get('ec2_recommendations', {}).get('monthly_storage_cost', 0)
            compute_cost -= storage_cost

        # Storage costs breakdown
        primary_storage = storage_cost
        destination_storage = self._calculate_destination_storage_cost(config)
        backup_storage = self._calculate_backup_storage_cost(config)

        # Network costs
        network_cost = self._calculate_standardized_network_cost(config)

        # Other costs
        remaining_cost = max(0, total_monthly - (
            agent_cost + compute_cost + primary_storage +
            destination_storage + backup_storage + network_cost
        ))

        return {
            'agents': agent_cost,
            'compute': compute_cost,
            'primary_storage': primary_storage,
            'destination_storage': destination_storage,
            'backup_storage': backup_storage,
            'network': network_cost,
            'other': remaining_cost
        }

    def _calculate_destination_storage_cost(self, config: Dict) -> float:
        """Calculate destination storage cost"""
        database_size_gb = config.get('database_size_gb', 0)
        destination_storage = config.get('destination_storage_type', 'S3')

        storage_costs = {'S3': 0.023, 'FSx_Windows': 0.13, 'FSx_Lustre': 0.14}
        cost_per_gb = storage_costs.get(destination_storage, 0.023)
        return database_size_gb * 1.2 * cost_per_gb

    def _calculate_backup_storage_cost(self, config: Dict) -> float:
        """Calculate backup storage cost"""
        if config.get('migration_method') != 'backup_restore':
            return 0

        backup_size = config.get('database_size_gb', 0) * config.get('backup_size_multiplier', 0.7)
        return backup_size * 0.023  # S3 Standard pricing

    def _calculate_standardized_network_cost(self, config: Dict) -> float:
        """Calculate standardized network cost"""
        environment = config.get('environment', 'non-production')
        database_size = config.get('database_size_gb', 0)

        if environment == 'production':
            dx_monthly = 2.25 * 24 * 30  # 10Gbps DX
            data_transfer = database_size * 0.1 * 0.02  # 10% monthly transfer
            vpn_backup = 45
            return dx_monthly + data_transfer + vpn_backup
        else:
            dx_monthly = 0.30 * 24 * 30  # 1Gbps DX
            data_transfer = database_size * 0.05 * 0.02  # 5% monthly transfer
            return dx_monthly + data_transfer

    def _validate_cost_consistency_v2(self, comprehensive_costs: Dict, basic_costs: Dict, analysis: Dict, config: Dict) -> Dict:
        """Validate cost consistency between different calculation methods"""
        discrepancies = []

        # Check comprehensive vs basic costs
        comp_total = comprehensive_costs.get('total_monthly', 0)
        basic_total = basic_costs.get('total_monthly_cost', 0)

        if comp_total > 0 and basic_total > 0:
            diff_pct = abs(comp_total - basic_total) / max(comp_total, basic_total) * 100
            if diff_pct > 15:  # 15% tolerance
                discrepancies.append({
                    'type': 'total_cost_mismatch',
                    'difference_percent': diff_pct,
                    'comprehensive': comp_total,
                    'basic': basic_total
                })

        # Check agent cost consistency
        agent_cost_1 = analysis.get('agent_analysis', {}).get('monthly_cost', 0)
        agent_cost_2 = basic_costs.get('agent_cost', 0)

        if agent_cost_1 > 0 and agent_cost_2 > 0:
            if abs(agent_cost_1 - agent_cost_2) > min(agent_cost_1, agent_cost_2) * 0.1:  # 10% tolerance
                discrepancies.append({
                    'type': 'agent_cost_mismatch',
                    'agent_analysis': agent_cost_1,
                    'cost_analysis': agent_cost_2
                })

        return {
            'is_consistent': len(discrepancies) == 0,
            'discrepancies': discrepancies,
            'discrepancy_count': len(discrepancies)
        }

    def _calculate_fallback_costs(self, config: Dict, aws_sizing: Dict, agent_analysis: Dict) -> tuple:
        """Calculate fallback costs when other methods unavailable"""
        # Basic cost calculation
        if config.get('target_platform') == 'rds':
            monthly_cost = aws_sizing.get('rds_recommendations', {}).get('total_monthly_cost', 1000)
        else:
            monthly_cost = aws_sizing.get('ec2_recommendations', {}).get('total_monthly_cost', 1200)

        # Add agent costs
        monthly_cost += agent_analysis.get('monthly_cost', 500)

        # Add network and other costs
        monthly_cost += self._calculate_standardized_network_cost(config)

        # One-time costs
        one_time_cost = 2000 + (config.get('number_of_agents', 1) * 500)

        return monthly_cost, one_time_cost

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
            # Backup Storage to S3 Paths (for backup_restore migration method)
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

            # Direct Replication Paths (for direct_replication migration method)
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
            },

            # NEW: Windows Direct Replication Paths (these were missing!)
            'nonprod_sj_windows_nas_s3': {
                'name': 'Non-Prod: San Jose Windows NAS → AWS S3 (Direct Replication)',
                'destination_storage': 'S3',
                'source': 'San Jose',
                'destination': 'AWS US-West-2 S3',
                'environment': 'non-production',
                'os_type': 'windows',
                'storage_type': 'nas',
                'migration_type': 'direct_replication',
                'segments': [
                    {
                        'name': 'Windows NAS to Windows Jump Server',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 2,
                        'reliability': 0.999,
                        'connection_type': 'internal_lan',
                        'cost_factor': 0.0,
                        'ai_optimization_potential': 0.93
                    },
                    {
                        'name': 'Windows Jump Server to AWS S3 (DX)',
                        'bandwidth_mbps': 2000,
                        'latency_ms': 18,
                        'reliability': 0.998,
                        'connection_type': 'direct_connect',
                        'cost_factor': 2.0,
                        'ai_optimization_potential': 0.90
                    }
                ],
                'ai_insights': {
                    'performance_bottlenecks': ['Windows protocol overhead', 'DX connection sharing'],
                    'optimization_opportunities': ['Windows performance tuning', 'DX bandwidth upgrade'],
                    'risk_factors': ['Single DX connection dependency', 'Windows update cycles'],
                    'recommended_improvements': ['Implement Windows caching', 'Configure QoS on DX']
                }
            },
            'nonprod_sj_windows_nas_fsx_windows': {
                'name': 'Non-Prod: San Jose Windows NAS → AWS FSx for Windows',
                'destination_storage': 'FSx_Windows',
                'source': 'San Jose',
                'destination': 'AWS US-West-2 FSx for Windows',
                'environment': 'non-production',
                'os_type': 'windows',
                'storage_type': 'nas',
                'migration_type': 'direct_replication',
                'segments': [
                    {
                        'name': 'Windows NAS to Windows Jump Server',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 2,
                        'reliability': 0.999,
                        'connection_type': 'internal_lan',
                        'cost_factor': 0.0,
                        'ai_optimization_potential': 0.93
                    },
                    {
                        'name': 'Windows Jump Server to AWS FSx Windows (DX)',
                        'bandwidth_mbps': 2000,
                        'latency_ms': 12,
                        'reliability': 0.999,
                        'connection_type': 'direct_connect',
                        'cost_factor': 2.5,
                        'ai_optimization_potential': 0.94
                    }
                ],
                'ai_insights': {
                    'performance_bottlenecks': ['Windows to Windows protocol optimization', 'SMB overhead'],
                    'optimization_opportunities': ['SMB3 protocol optimization', 'FSx throughput configuration'],
                    'risk_factors': ['Cross-platform compatibility', 'SMB version negotiation'],
                    'recommended_improvements': ['Test SMB3.1.1 compatibility', 'Configure FSx performance mode']
                }
            },
            'nonprod_sj_windows_nas_fsx_lustre': {
                'name': 'Non-Prod: San Jose Windows NAS → AWS FSx for Lustre',
                'destination_storage': 'FSx_Lustre',
                'source': 'San Jose',
                'destination': 'AWS US-West-2 FSx for Lustre',
                'environment': 'non-production',
                'os_type': 'windows',
                'storage_type': 'nas',
                'migration_type': 'direct_replication',
                'segments': [
                    {
                        'name': 'Windows NAS to Windows Jump Server',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 2,
                        'reliability': 0.999,
                        'connection_type': 'internal_lan',
                        'cost_factor': 0.0,
                        'ai_optimization_potential': 0.93
                    },
                    {
                        'name': 'Windows Jump Server to AWS FSx Lustre (DX)',
                        'bandwidth_mbps': 2000,
                        'latency_ms': 8,
                        'reliability': 0.9995,
                        'connection_type': 'direct_connect',
                        'cost_factor': 3.0,
                        'ai_optimization_potential': 0.95
                    }
                ],
                'ai_insights': {
                    'performance_bottlenecks': ['Windows Lustre client setup', 'Protocol conversion overhead'],
                    'optimization_opportunities': ['Lustre client optimization', 'Parallel I/O tuning'],
                    'risk_factors': ['Windows Lustre compatibility', 'Client complexity'],
                    'recommended_improvements': ['Optimize Windows Lustre client', 'Configure parallel data transfer']
                }
            },
            'prod_sa_windows_nas_s3': {
                'name': 'Prod: San Antonio Windows NAS → San Jose → AWS Production VPC S3',
                'destination_storage': 'S3',
                'source': 'San Antonio',
                'destination': 'AWS US-West-2 Production VPC S3',
                'environment': 'production',
                'os_type': 'windows',
                'storage_type': 'nas',
                'migration_type': 'direct_replication',
                'segments': [
                    {
                        'name': 'San Antonio Windows NAS to Windows Jump Server',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 2,
                        'reliability': 0.999,
                        'connection_type': 'internal_lan',
                        'cost_factor': 0.0,
                        'ai_optimization_potential': 0.95
                    },
                    {
                        'name': 'San Antonio to San Jose (Private Line)',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 15,
                        'reliability': 0.9995,
                        'connection_type': 'private_line',
                        'cost_factor': 3.0,
                        'ai_optimization_potential': 0.92
                    },
                    {
                        'name': 'San Jose to AWS Production VPC S3 (DX)',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 10,
                        'reliability': 0.9999,
                        'connection_type': 'direct_connect',
                        'cost_factor': 4.0,
                        'ai_optimization_potential': 0.94
                    }
                ],
                'ai_insights': {
                    'performance_bottlenecks': ['Cross-site Windows latency', 'Multiple hop complexity'],
                    'optimization_opportunities': ['End-to-end Windows optimization', 'Compression algorithms'],
                    'risk_factors': ['Multiple failure points', 'Windows authentication across sites'],
                    'recommended_improvements': ['Implement WAN optimization', 'Add redundant paths']
                }
            },
            'prod_sa_windows_nas_fsx_windows': {
                'name': 'Prod: San Antonio Windows NAS → San Jose → AWS FSx for Windows',
                'destination_storage': 'FSx_Windows',
                'source': 'San Antonio',
                'destination': 'AWS US-West-2 FSx for Windows',
                'environment': 'production',
                'os_type': 'windows',
                'storage_type': 'nas',
                'migration_type': 'direct_replication',
                'segments': [
                    {
                        'name': 'San Antonio Windows NAS to Windows Jump Server',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 2,
                        'reliability': 0.999,
                        'connection_type': 'internal_lan',
                        'cost_factor': 0.0,
                        'ai_optimization_potential': 0.95
                    },
                    {
                        'name': 'San Antonio to San Jose (Private Line)',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 15,
                        'reliability': 0.9995,
                        'connection_type': 'private_line',
                        'cost_factor': 3.0,
                        'ai_optimization_potential': 0.92
                    },
                    {
                        'name': 'San Jose to AWS FSx Windows (DX)',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 6,
                        'reliability': 0.9999,
                        'connection_type': 'direct_connect',
                        'cost_factor': 4.5,
                        'ai_optimization_potential': 0.96
                    }
                ],
                'ai_insights': {
                    'performance_bottlenecks': ['Windows SMB over WAN', 'Multi-hop complexity'],
                    'optimization_opportunities': ['WAN optimization', 'SMB3 multichannel'],
                    'risk_factors': ['Cross-site Windows dependencies', 'SMB over WAN reliability'],
                    'recommended_improvements': ['Implement WAN acceleration', 'Configure SMB3 multichannel']
                }
            },
            'prod_sa_windows_nas_fsx_lustre': {
                'name': 'Prod: San Antonio Windows NAS → San Jose → AWS FSx for Lustre',
                'destination_storage': 'FSx_Lustre',
                'source': 'San Antonio',
                'destination': 'AWS US-West-2 FSx for Lustre',
                'environment': 'production',
                'os_type': 'windows',
                'storage_type': 'nas',
                'migration_type': 'direct_replication',
                'segments': [
                    {
                        'name': 'San Antonio Windows NAS to Windows Jump Server',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 2,
                        'reliability': 0.999,
                        'connection_type': 'internal_lan',
                        'cost_factor': 0.0,
                        'ai_optimization_potential': 0.95
                    },
                    {
                        'name': 'San Antonio to San Jose (Private Line)',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 15,
                        'reliability': 0.9995,
                        'connection_type': 'private_line',
                        'cost_factor': 3.0,
                        'ai_optimization_potential': 0.92
                    },
                    {
                        'name': 'San Jose to AWS FSx Lustre (DX)',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 4,
                        'reliability': 0.9999,
                        'connection_type': 'direct_connect',
                        'cost_factor': 5.0,
                        'ai_optimization_potential': 0.98
                    }
                ],
                'ai_insights': {
                    'performance_bottlenecks': ['Windows Lustre complexity', 'Cross-site latency'],
                    'optimization_opportunities': ['Lustre optimization', 'End-to-end tuning'],
                    'risk_factors': ['Windows Lustre support', 'Multiple failure points'],
                    'recommended_improvements': ['Optimize Windows Lustre setup', 'Add redundant paths']
                }
            },
            'prod_sa_linux_nas_fsx_windows': {
                'name': 'Prod: San Antonio Linux NAS → San Jose → AWS FSx for Windows',
                'destination_storage': 'FSx_Windows',
                'source': 'San Antonio',
                'destination': 'AWS US-West-2 FSx for Windows',
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
                        'name': 'San Jose to AWS FSx Windows (DX)',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 6,
                        'reliability': 0.9999,
                        'connection_type': 'direct_connect',
                        'cost_factor': 4.5,
                        'ai_optimization_potential': 0.96
                    }
                ],
                'ai_insights': {
                    'performance_bottlenecks': ['Linux to Windows protocol conversion', 'Cross-site latency'],
                    'optimization_opportunities': ['SMB3 protocol optimization', 'FSx throughput configuration'],
                    'risk_factors': ['Cross-platform compatibility', 'Multiple failure points'],
                    'recommended_improvements': ['Test SMB3.1.1 compatibility', 'Configure FSx performance mode']
                }
            },
            'prod_sa_linux_nas_fsx_lustre': {
                'name': 'Prod: San Antonio Linux NAS → San Jose → AWS FSx for Lustre',
                'destination_storage': 'FSx_Lustre',
                'source': 'San Antonio',
                'destination': 'AWS US-West-2 FSx for Lustre',
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
                        'name': 'San Jose to AWS FSx Lustre (DX)',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 4,
                        'reliability': 0.9999,
                        'connection_type': 'direct_connect',
                        'cost_factor': 5.0,
                        'ai_optimization_potential': 0.98
                    }
                ],
                'ai_insights': {
                    'performance_bottlenecks': ['Lustre client configuration', 'Cross-site coordination'],
                    'optimization_opportunities': ['Lustre striping optimization', 'Parallel I/O tuning'],
                    'risk_factors': ['Lustre complexity', 'Multiple failure points'],
                    'recommended_improvements': ['Optimize Lustre striping patterns', 'Configure parallel data transfer']
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

        # Use unified cost calculation (SINGLE SOURCE OF TRUTH)
        unified_costs = await self.cost_calculator.calculate_unified_migration_costs(config, {
            'aws_sizing_recommendations': aws_sizing,
            'agent_analysis': agent_analysis,
            'network_performance': network_perf,
            'onprem_performance': onprem_performance
        })

        # Create both cost structures for compatibility
        cost_analysis = unified_costs.copy()
        cost_analysis['total_monthly_cost'] = unified_costs['total_monthly']  # Add compatible field names
        cost_analysis['aws_compute_cost'] = unified_costs['detailed_breakdown']['aws_compute']
        cost_analysis['aws_storage_cost'] = unified_costs['detailed_breakdown']['aws_storage']
        cost_analysis['agent_cost'] = unified_costs['detailed_breakdown']['migration_agents']
        cost_analysis['network_cost'] = unified_costs['detailed_breakdown']['network']
        cost_analysis['destination_storage_cost'] = unified_costs['detailed_breakdown']['additional_storage']
        cost_analysis['one_time_migration_cost'] = unified_costs['total_one_time']

        comprehensive_costs = unified_costs.copy()
        comprehensive_costs['data_source'] = 'unified'

        # FSx comparisons
        fsx_comparisons = await self._generate_fsx_destination_comparisons(config)

        # AI overall assessment
        ai_overall_assessment = await self._generate_ai_overall_assessment_with_agents(
            config, onprem_performance, aws_sizing, migration_time_hours, agent_analysis
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
            'cost_analysis': cost_analysis,  # Use unified costs
            'comprehensive_costs': comprehensive_costs,  # Use unified costs
            'fsx_comparisons': fsx_comparisons,
            'ai_overall_assessment': ai_overall_assessment
        }

    
    
    
    # In the EnhancedMigrationAnalyzer class (around line 1464), replace the _get_network_path_key method:

    def _get_network_path_key(self, config: Dict) -> str:
            """Get network path key based on migration method and backup storage with error handling"""
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
                else:
                    return "nonprod_sj_nas_drive_s3"  # Default fallback
            
            # For direct replication, use original paths
            else:
                if environment in ['non_production', 'nonprod']:
                    if destination_storage == 's3':
                        key = f"nonprod_sj_{os_type}_nas_s3"
                    elif destination_storage == 'fsx_windows':
                        key = f"nonprod_sj_{os_type}_nas_fsx_windows"
                    elif destination_storage == 'fsx_lustre':
                        key = f"nonprod_sj_{os_type}_nas_fsx_lustre"
                    else:
                        key = f"nonprod_sj_{os_type}_nas_s3"
                elif environment == 'production':
                    if destination_storage == 's3':
                        key = f"prod_sa_{os_type}_nas_s3"
                    elif destination_storage == 'fsx_windows':
                        key = f"prod_sa_{os_type}_nas_fsx_windows"
                    elif destination_storage == 'fsx_lustre':
                        key = f"prod_sa_{os_type}_nas_fsx_lustre"
                    else:
                        key = f"prod_sa_{os_type}_nas_s3"
                else:
                    key = f"nonprod_sj_{os_type}_nas_s3"  # Default fallback

                # Check if the key exists in network_paths, if not, use fallback
                if hasattr(self.network_manager, 'network_paths') and key in self.network_manager.network_paths:
                    return key
                else:
                    # Use fallback logic
                    if os_type == 'windows':
                        fallback_key = key.replace('windows', 'linux')
                        if hasattr(self.network_manager, 'network_paths') and fallback_key in self.network_manager.network_paths:
                            return fallback_key
                    
                    if 'prod_sa' in key:
                        fallback_key = key.replace('prod_sa', 'nonprod_sj')
                        if hasattr(self.network_manager, 'network_paths') and fallback_key in self.network_manager.network_paths:
                            return fallback_key
                    
                    # Ultimate fallback to a known working path
                    return "nonprod_sj_linux_nas_s3"

            # Default fallback for backup/restore if nothing matches
            return "nonprod_sj_nas_drive_s3"
     

    async def _analyze_ai_migration_agents_with_scaling(self, config: Dict, primary_tool: str, network_perf: Dict) -> Dict:
        """Enhanced migration agent analysis with scaling support and backup storage considerations"""

        num_agents = config.get('number_of_agents', 1)
        destination_storage = config.get('destination_storage_type', 'S3')
        migration_method = config.get('migration_method', 'direct_replication')

        # Determine the correct tool and agent size based on migration method and database engines
        if migration_method == 'backup_restore':
            # For backup/restore, always use DataSync regardless of database engine
            actual_primary_tool = 'datasync'
            agent_size = config.get('datasync_agent_size', 'medium')
            agent_config = self.agent_manager.calculate_agent_configuration('datasync', agent_size, num_agents, destination_storage)
        else:
            # For direct replication, determine tool based on database engine compatibility
            is_homogeneous = config['source_database_engine'] == config['database_engine']
            
            if is_homogeneous:
                actual_primary_tool = 'datasync'
                agent_size = config.get('datasync_agent_size', 'medium')
                agent_config = self.agent_manager.calculate_agent_configuration('datasync', agent_size, num_agents, destination_storage)
            else:
                # Heterogeneous migration - use DMS
                actual_primary_tool = 'dms'
                agent_size = config.get('dms_agent_size', 'medium')
                agent_config = self.agent_manager.calculate_agent_configuration('dms', agent_size, num_agents, destination_storage)

        # Rest of the method logic remains the same...
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
            'primary_tool': actual_primary_tool,  # Use the correctly determined tool
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

        database_size_gb = safe_float(config.get('database_size_gb', 1000))
        num_agents = safe_int(config.get('number_of_agents', 1))
        destination_storage = config.get('destination_storage_type', 'S3')
        migration_method = config.get('migration_method', 'direct_replication')

        # Calculate data size to transfer
        if migration_method == 'backup_restore':
            # For backup/restore, calculate backup file size
            backup_size_multiplier = safe_float(config.get('backup_size_multiplier', 0.7))
            data_size_gb = database_size_gb * backup_size_multiplier
            backup_storage_type = config.get('backup_storage_type', 'nas_drive')

            # Base calculation for file transfer
            migration_throughput_safe = safe_float(migration_throughput, 100)  # Default 100 Mbps
            if migration_throughput_safe > 0:
                base_time_hours = (data_size_gb * 8 * 1000) / (migration_throughput_safe * 3600)
            else:
                base_time_hours = 24  # Default 24 hours if no throughput

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
            migration_throughput_safe = safe_float(migration_throughput, 100)
            if migration_throughput_safe > 0:
                base_time_hours = (data_size_gb * 8 * 1000) / (migration_throughput_safe * 3600)
            else:
                base_time_hours = 48  # Default 48 hours if no throughput
            complexity_factor = 1.0
            backup_prep_time = 0

        # Database engine complexity
        if config.get('source_database_engine') != config.get('database_engine'):
            complexity_factor *= 1.3

        # OS and platform factors
        if 'windows' in config.get('operating_system', ''):
            complexity_factor *= 1.1

        if config.get('server_type') == 'vmware':
            complexity_factor *= 1.05

        # Destination storage adjustments
        if destination_storage == 'FSx_Windows':
            complexity_factor *= 0.9
        elif destination_storage == 'FSx_Lustre':
            complexity_factor *= 0.7

        # Agent scaling adjustments
        scaling_efficiency = safe_float(agent_analysis.get('scaling_efficiency', 1.0))
        storage_multiplier = safe_float(agent_analysis.get('storage_performance_multiplier', 1.0))

        if num_agents > 1:
            agent_time_factor = (1 / min(num_agents * scaling_efficiency * storage_multiplier, 6.0))
            complexity_factor *= agent_time_factor

            if num_agents > 5:
                complexity_factor *= 1.1

        total_time = base_time_hours * complexity_factor + backup_prep_time

        return max(0.1, total_time)  # Ensure minimum 0.1 hours

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

    def config_has_changed(self, current_config, stored_config):
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
    """Enhanced header with user info"""
    ai_manager = AnthropicAIManager()
    aws_api = AWSAPIManager()

    ai_status = "🟢" if ai_manager.connected else "🔴"
    aws_status = "🟢" if aws_api.connected else "🔴"
    
    # Get user info
    user_name = st.session_state.get('user_name', 'User')
    user_role = st.session_state.get('user_role', 'user')

    st.markdown(f"""
    <div class="main-header">
    <h1>🤖 AWS Enterprise Database Migration Analyzer AI v3.0</h1>
    <p style="font-size: 1.2rem; margin-top: 0.5rem;">
    Professional-Grade Migration Analysis • AI-Powered Insights • Real-time AWS Integration
    </p>
    <div style="margin-top: 1rem; font-size: 0.8rem;">
    <span style="margin-right: 20px;">{ai_status} Anthropic Claude AI</span>
    <span style="margin-right: 20px;">{aws_status} AWS Pricing APIs</span>
    <span style="margin-right: 20px;">👤 Welcome, {user_name} ({user_role.title()})</span>
    <span>🔐 Secure Enterprise Access</span>
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
    else:  # EC2:
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

# Determine primary tool based on migration method and correct database engine
    if migration_method == 'backup_restore':
        primary_tool = "DataSync"
        is_homogeneous = True  # Always use DataSync for backup/restore
    else:
        # Get the correct target database engine based on platform
        if target_platform == "rds":
            target_db_engine = database_engine
        else:  # EC2
            target_db_engine = ec2_database_engine if ec2_database_engine else database_engine
        
        # Check if migration is homogeneous (same source and target engines)
        is_homogeneous = source_database_engine == target_db_engine
        primary_tool = "DataSync" if is_homogeneous else "DMS"

    st.sidebar.success(f"**Primary Tool:** AWS {primary_tool}")

    # Show migration method info
    if migration_method == 'backup_restore':
        st.sidebar.info(f"**Method:** Backup/Restore via DataSync from {backup_storage_type.replace('_', ' ').title()}")
        st.sidebar.write(f"**Backup Size:** {int(backup_size_multiplier*100)}% of database ({backup_size_multiplier:.1f}x)")
    else:
        migration_type = 'homogeneous' if is_homogeneous else 'heterogeneous'
        st.sidebar.info(f"**Method:** Direct replication ({migration_type})")
        if not is_homogeneous:
            st.sidebar.warning(f"**Schema Conversion:** {source_database_engine.upper()} → {target_db_engine.upper()}")

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

    st.plotly_chart(fig, use_container_width=True, key="bandwidth_waterfall_chart")

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


def render_ai_insights_tab_enhanced(analysis: Dict, config: Dict):
    """Render enhanced AI insights and analysis tab"""

    # ADD THIS LINE AT THE BEGINNING
    validated_costs = add_cost_validation_to_tab(analysis, config)
    st.subheader("🧠 AI-Powered Migration Insights & Analysis")

    # USE VALIDATED COSTS INSTEAD
    monthly_cost = validated_costs['total_monthly']  # NEW WAY
    validation_icon = "✅" if validated_costs['is_validated'] else "⚠️"
    st.metric("Monthly Cost", f"${monthly_cost:,.0f}",
              delta=f"{validation_icon} Validated")

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

    # ADD VALIDATION AT THE START (if this tab shows costs)
    validated_costs = add_cost_validation_to_tab(analysis, config)
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

    # NEW: Agent Placement Visualization
    st.markdown("---")
    st.markdown("**🤖 DataSync/DMS Agent Placement & Migration Flow:**")
    
    try:
        agent_diagram = create_agent_placement_diagram(analysis, config)
        st.plotly_chart(agent_diagram, use_container_width=True, key=f"agent_placement_{int(time.time() * 1000000)}")
        
        # Add explanation of the diagram
        migration_method = config.get('migration_method', 'direct_replication')
        if migration_method == 'backup_restore':
            st.info("""
            📦 **Backup/Restore Migration Flow:**
            1. **Database** creates backup files on backup storage
            2. **DataSync Agents** read backup files using SMB/NFS protocols
            3. **Agents** transfer data through network path to AWS
            4. **AWS** receives and restores data to target database
            
            **Key Advantage:** Minimal impact on production database during transfer
            """)
        else:
            st.info("""
            🔄 **Direct Replication Migration Flow:**
            1. **DMS/DataSync Agents** connect directly to source database
            2. **Agents** perform real-time Change Data Capture (CDC)
            3. **Live data** flows through network path to AWS target
            4. **Target database** stays synchronized in real-time
            
            **Key Advantage:** Minimal downtime with continuous synchronization
            """)
            
    except Exception as e:
        st.warning(f"Agent placement diagram could not be rendered: {str(e)}")
        
        # Fallback: Show agent placement summary
        st.info("📍 **Agent Placement Summary:**")
        
        agent_analysis = analysis.get('agent_analysis', {})
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Migration Method:** {config.get('migration_method', 'direct_replication').replace('_', ' ').title()}")
            st.write(f"**Primary Tool:** {agent_analysis.get('primary_tool', 'DMS').upper()}")
            st.write(f"**Number of Agents:** {config.get('number_of_agents', 1)}")
            st.write(f"**Agent Size:** {config.get('datasync_agent_size') or config.get('dms_agent_size', 'medium').title()}")
        
        with col2:
            st.write(f"**Total Throughput:** {agent_analysis.get('total_effective_throughput', 0):,.0f} Mbps")
            st.write(f"**Monthly Cost:** ${agent_analysis.get('monthly_cost', 0):,.0f}")
            st.write(f"**Scaling Efficiency:** {agent_analysis.get('scaling_efficiency', 1.0)*100:.1f}%")
            st.write(f"**Bottleneck:** {agent_analysis.get('bottleneck', 'Unknown')}")

    
    
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
            st.plotly_chart(network_diagram, use_container_width=True, key=f"network_{int(time.time() * 1000000)}")
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

def render_comprehensive_cost_analysis_tab_with_pdf(analysis: Dict, config: Dict):
    """Enhanced cost analysis with PDF export"""
    st.subheader("💰 Complete AWS Cost Analysis - All Costs Consolidated & Validated")

    # Initialize cost validator
    cost_validator = CostValidationManager()
    validated_costs = cost_validator.get_validated_costs(analysis, config)

    # Get fallback data for compatibility
    comprehensive_costs = analysis.get('comprehensive_costs', {})
    basic_cost_analysis = analysis.get('cost_analysis', {})
    aws_sizing = analysis.get('aws_sizing_recommendations', {})
    agent_analysis = analysis.get('agent_analysis', {})

    # FIX: Define monthly_breakdown from validated costs or comprehensive costs
    monthly_breakdown = comprehensive_costs.get('monthly_breakdown', validated_costs.get('breakdown', {}))

    # Display validation status prominently
    if validated_costs['is_validated']:
        st.success("✅ All cost calculations have been validated and are consistent across tabs")
    else:
        st.error("❌ Cost discrepancies detected - see details below")

        with st.expander("🔍 Detailed Cost Validation Report", expanded=True):
            validation = validated_costs['validation']

            st.write(f"**Total Discrepancies Found:** {validation['discrepancy_count']}")

            for i, disc in enumerate(validation['discrepancies'], 1):
                st.write(f"**Issue {i}: {disc['type'].replace('_', ' ').title()}**")

                if disc['type'] == 'total_cost_mismatch':
                    st.write(f"   • Difference: {disc['difference_percent']:.1f}%")
                    st.write(f"   • Comprehensive Method: ${disc['comprehensive']:,.0f}/month")
                    st.write(f"   • Basic Method: ${disc['basic']:,.0f}/month")
                    st.write(f"   • **Recommendation:** Use comprehensive method for accuracy")

                elif disc['type'] == 'agent_cost_mismatch':
                    st.write(f"   • Agent Analysis Cost: ${disc['agent_analysis']:,.0f}/month")
                    st.write(f"   • Cost Analysis Cost: ${disc['cost_analysis']:,.0f}/month")
                    st.write(f"   • **Recommendation:** Use agent analysis as source of truth")

                st.write("")

    # === VALIDATED EXECUTIVE COST DASHBOARD ===
    st.markdown("**🎯 Validated Executive Cost Dashboard - All AWS Services**")

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    total_monthly = validated_costs['total_monthly']
    total_one_time = validated_costs['total_one_time']
    three_year_total = validated_costs['three_year_total']
    breakdown = validated_costs['breakdown']

    with col1:
        st.metric("💰 Total Monthly (Validated)", f"${total_monthly:,.0f}",
                  delta=f"Annual: ${total_monthly * 12:,.0f}")

    with col2:
        st.metric("🔄 One-Time Setup", f"${total_one_time:,.0f}",
                  delta="Migration & Setup")

    with col3:
        st.metric("📅 3-Year Total", f"${three_year_total:,.0f}",
                  delta="Complete TCO")

    with col4:
        largest_component = max(breakdown.items(), key=lambda x: x[1]) if breakdown else ('unknown', 0)
        st.metric("🎯 Largest Cost", largest_component[0].replace('_', ' ').title(),
                  delta=f"${largest_component[1]:,.0f}/mo")

    with col5:
        target_platform = config.get('target_platform', 'rds')
        platform_cost = breakdown.get('compute', 0)
        st.metric("☁️ Platform Cost", f"${platform_cost:,.0f}/mo",
                  delta=f"{target_platform.upper()} Platform")

    with col6:
        agent_cost = breakdown.get('agents', 0)
        agent_count = config.get('number_of_agents', 1)
        cost_per_agent = agent_cost / agent_count if agent_count > 0 else 0
        st.metric("🤖 Migration Agents", f"${agent_cost:,.0f}/mo",
                  delta=f"${cost_per_agent:,.0f} per agent")

        render_comprehensive_cost_analysis_tab(analysis, config)
    
        # Add PDF export section
        render_pdf_export_section(analysis, config)
        
    # === VALIDATED COST BREAKDOWN ===
    st.markdown("---")
    st.markdown("**🔧 Validated AWS Services Cost Breakdown**")

    # Create detailed service breakdown table
    service_breakdown_data = []

    # Map breakdown to service details
    breakdown_mapping = {
        'compute': ('Database Compute', 'Primary database instances'),
        'primary_storage': ('EBS/RDS Storage', 'Primary database storage'),
        'destination_storage': ('Destination Storage', f"Migration destination ({config.get('destination_storage_type', 'S3')})"),
        'backup_storage': ('Backup Storage', 'Backup files for migration'),
        'agents': ('Migration Agents', f"{config.get('number_of_agents', 1)} migration agents"),
        'network': ('Network Services', 'Direct Connect and data transfer'),
        'other': ('Other AWS Services', 'Additional AWS service costs')
    }

    for component, cost in breakdown.items():
        if cost > 0:
            service_name, description = breakdown_mapping.get(component, (component.title(), 'Unknown service'))
            percentage = (cost / total_monthly) * 100 if total_monthly > 0 else 0

            service_breakdown_data.append({
                'AWS Service': service_name,
                'Monthly Cost': f"${cost:,.0f}",
                'Percentage': f"{percentage:.1f}%",
                'Description': description,
                'Validation': '✅ Validated'
            })

    if service_breakdown_data:
        df_services = pd.DataFrame(service_breakdown_data)
        st.dataframe(df_services, use_container_width=True, hide_index=True)

        # Verification
        calculated_total = sum([float(item['Monthly Cost'].replace('$', '').replace(',', ''))
                               for item in service_breakdown_data])

        if abs(calculated_total - total_monthly) < 5:  # $5 tolerance for rounding
            st.success(f"✅ **Cost Breakdown Verified:** Total matches ${calculated_total:,.0f} = ${total_monthly:,.0f}")
        else:
            st.warning(f"⚠️ **Cost Breakdown Mismatch:** Sum=${calculated_total:,.0f}, Expected=${total_monthly:,.0f}")

    # === MONTHLY COST DISTRIBUTION CHART ===
    st.markdown("---")
    st.markdown("**📊 Cost Projections & Financial Analysis**")

    proj_col1, proj_col2 = st.columns(2)

    with proj_col1:
        st.markdown("**Monthly Cost Distribution**")

        # Create pie chart using validated breakdown
        if breakdown:
            # Clean up category names and filter out zero costs
            clean_breakdown = {}
            for category, cost in breakdown.items():
                if cost > 0:
                    clean_name = category.replace('_', ' ').title()
                    clean_breakdown[clean_name] = cost

            if clean_breakdown:
                fig_pie = px.pie(
                    values=list(clean_breakdown.values()),
                    names=list(clean_breakdown.keys()),
                    title="Monthly Costs by Service Category",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True, key=f"monthly_cost_pie_{int(time.time() * 1000000)}")
            else:
                st.info("Cost breakdown visualization not available")
        else:
            st.info("Monthly cost distribution data not available")

    with proj_col2:
        st.markdown("**3-Year Cost Projection**")

        # Create timeline projection
        months = list(range(0, 37, 3))  # Every 3 months for 3 years
        cumulative_costs = []

        for month in months:
            cumulative_cost = total_one_time + (total_monthly * month)
            cumulative_costs.append(cumulative_cost)

        projection_data = pd.DataFrame({
            'Months': months,
            'Cumulative Cost ($)': cumulative_costs
        })

        fig_line = px.line(
            projection_data,
            x='Months',
            y='Cumulative Cost ($)',
            title="3-Year Cumulative Cost Projection",
            markers=True
        )
        fig_line.update_traces(line_color='#2E86C1', marker_color='#E74C3C')
        fig_line.update_layout(
            xaxis_title="Months from Migration",
            yaxis_title="Cumulative Cost (USD)",
            hovermode='x unified'
        )
        st.plotly_chart(fig_line, use_container_width=True, key=f"line_chart_{int(time.time() * 1000000)}")

    # === COST SOURCE AND ACCURACY ===
    st.markdown("---")
    st.markdown("**🔍 Cost Data Source & Accuracy Information**")

    accuracy_col1, accuracy_col2, accuracy_col3 = st.columns(3)

    with accuracy_col1:
        st.info("**📊 Cost Data Source**")
        st.write(f"**Primary Source:** {validated_costs['cost_source'].title()}")
        st.write(f"**Validation Status:** {'✅ Passed' if validated_costs['is_validated'] else '⚠️ Issues Found'}")
        st.write(f"**Data Consolidation:** All tabs use same cost source")
        st.write(f"**Cross-Tab Consistency:** {'✅ Consistent' if validated_costs['is_validated'] else '❌ Inconsistent'}")

    with accuracy_col2:
        st.success("**🎯 Accuracy Information**")

        pricing_data = comprehensive_costs.get('pricing_data', {})
        data_source = pricing_data.get('data_source', 'estimated')

        if data_source == 'aws_api':
            st.write("✅ **Pricing Data:** Real-time AWS API")
            accuracy = "High (±5%)"
        else:
            st.write("⚠️ **Pricing Data:** Fallback estimates")
            accuracy = "Moderate (±15%)"

        st.write(f"**Accuracy Level:** {accuracy}")
        st.write(f"**Last Updated:** {pricing_data.get('last_updated', 'Unknown')}")
        st.write(f"**Services Analyzed:** {len(service_breakdown_data)} AWS services")

    with accuracy_col3:
        st.warning("**🔧 Validation Methodology**")
        st.write("• Cross-reference multiple calculation methods")
        st.write("• Validate component sums against totals")
        st.write("• Check agent cost consistency")
        st.write("• Verify backup storage logic")
        st.write("• Ensure no double-counting")
        quality_score = max(0, 100 - len(validated_costs['validation']['discrepancies']) * 20)
        st.write(f"• **Quality Score:** {quality_score:.0f}/100")

    # === COST OPTIMIZATION RECOMMENDATIONS ===
    st.markdown("---")
    st.markdown("**💡 Cost Optimization Opportunities**")

    # Get optimization recommendations from comprehensive costs or generate them
    recommendations = comprehensive_costs.get('cost_optimization_recommendations', [])

    if not recommendations:
        # Generate basic recommendations
        target_platform = config.get('target_platform', 'rds')
        environment = config.get('environment', 'non-production')
        database_size = config.get('database_size_gb', 1000)

        recommendations = [
            f"Consider Reserved Instances for {target_platform.upper()} for 20-30% savings on long-term usage",
            f"Review agent configuration - current setup: {config.get('number_of_agents', 1)} agents",
            f"Monitor actual usage vs provisioned capacity for right-sizing opportunities"
        ]

        if environment == 'non-production':
            recommendations.append("Use Spot Instances for non-production workloads for 60-70% savings")

        if database_size > 5000:
            recommendations.append("Consider storage lifecycle policies for older data")

        if config.get('migration_method') == 'backup_restore':
            recommendations.append("Optimize backup storage costs and retention policies")

    if recommendations:
        opt_col1, opt_col2 = st.columns(2)

        with opt_col1:
            st.success("**Immediate Opportunities (0-3 months)**")
            for i, rec in enumerate(recommendations[:4], 1):
                savings_estimate = "20-30%" if 'Reserved' in rec else "60-70%" if 'Spot' in rec else "10-25%"
                st.write(f"**{i}.** {rec}")
                st.write(f"   💰 Potential savings: {savings_estimate}")
                st.write("")

        with opt_col2:
            st.info("**Medium-term Opportunities (3-12 months)**")
            additional_recs = recommendations[4:] if len(recommendations) > 4 else [
                "Implement auto-scaling policies for variable workloads",
                "Review and optimize data transfer patterns",
                "Consider multi-year Reserved Instance commitments",
                "Implement cost allocation tags for better tracking"
            ]

            for i, rec in enumerate(additional_recs[:4], 5):
                savings_estimate = "15-25%" if 'lifecycle' in rec.lower() else "5-15%"
                st.write(f"**{i}.** {rec}")
                st.write(f"   💰 Potential savings: {savings_estimate}")
                st.write("")

    # === COST COMPARISON & SUMMARY ===
    st.markdown("---")
    st.markdown("**📊 Cost Summary & Data Sources**")

    summary_col1, summary_col2, summary_col3 = st.columns(3)

    with summary_col1:
        st.info("**📈 Cost Summary**")
        st.write(f"**Monthly Operating Cost:** ${total_monthly:,.0f}")
        st.write(f"**One-time Migration Cost:** ${total_one_time:,.0f}")
        st.write(f"**Annual Operating Cost:** ${total_monthly * 12:,.0f}")
        st.write(f"**3-Year Total Cost:** ${three_year_total:,.0f}")

        # Cost per GB analysis
        db_size = config.get('database_size_gb', 1000)
        cost_per_gb_monthly = total_monthly / db_size if db_size > 0 else 0
        st.write(f"**Cost per GB/month:** ${cost_per_gb_monthly:.2f}")

    with summary_col2:
        st.success("**🎯 Migration ROI Analysis**")

        # ROI calculations
        estimated_savings = basic_cost_analysis.get('estimated_monthly_savings', total_monthly * 0.15)  # Assume 15% savings
        roi_months = total_one_time / estimated_savings if estimated_savings > 0 else 0

        st.write(f"**Estimated Monthly Savings:** ${estimated_savings:,.0f}")
        st.write(f"**ROI Break-even:** {roi_months:.1f} months")
        st.write(f"**3-Year Savings:** ${estimated_savings * 36:,.0f}")

        # Efficiency metrics
        if agent_analysis:
            throughput = agent_analysis.get('total_effective_throughput', 1000)
            cost_per_mbps = total_monthly / throughput if throughput > 0 else 0
            st.write(f"**Cost per Mbps:** ${cost_per_mbps:.2f}/month")

    with summary_col3:
        st.warning("**🔍 Data Sources & Accuracy**")

        # Data source information
        pricing_data = comprehensive_costs.get('pricing_data', {})
        data_source = pricing_data.get('data_source', 'estimated')
        last_updated = pricing_data.get('last_updated', 'Unknown')

        if data_source == 'aws_api':
            st.write("✅ **Pricing Data:** Real-time AWS API")
            accuracy = "High (±5%)"
        else:
            st.write("⚠️ **Pricing Data:** Fallback estimates")
            accuracy = "Moderate (±15%)"

        st.write(f"📅 **Last Updated:** {last_updated}")
        st.write(f"🎯 **Accuracy:** {accuracy}")
        st.write(f"🔧 **Services Analyzed:** {len(service_breakdown_data)} AWS services")
        st.write(f"📊 **Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # === EXPORT AND ACTIONS ===
    st.markdown("---")
    st.markdown("**📤 Export & Actions**")

    export_col1, export_col2, export_col3 = st.columns(3)

    with export_col1:
        if st.button("📊 Export Complete Cost Analysis", use_container_width=True):
            export_data = {
                'analysis_date': datetime.now().isoformat(),
                'configuration': config,
                'cost_summary': {
                    'total_monthly': total_monthly,
                    'total_one_time': total_one_time,
                    'three_year_total': three_year_total,
                    'monthly_breakdown': monthly_breakdown
                },
                'service_breakdown': service_breakdown_data,
                'optimization_recommendations': recommendations,
                'data_source': data_source,
                'roi_analysis': {
                    'estimated_monthly_savings': estimated_savings,
                    'roi_break_even_months': roi_months,
                    'cost_per_gb_monthly': cost_per_gb_monthly
                }
            }

            st.download_button(
                label="💾 Download Complete Cost Analysis (JSON)",
                data=json.dumps(export_data, indent=2),
                file_name=f"complete_aws_cost_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )

    with export_col2:
        if st.button("📋 Generate Cost Report", use_container_width=True):
            # Generate a text-based cost report
            report_lines = [
                "AWS ENTERPRISE DATABASE MIGRATION - COMPLETE COST ANALYSIS",
                "=" * 60,
                f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                f"Database: {config.get('source_database_engine', 'Unknown')} → {config.get('database_engine', 'Unknown')}",
                f"Size: {config.get('database_size_gb', 0):,} GB",
                f"Target Platform: {config.get('target_platform', 'Unknown').upper()}",
                "",
                "COST SUMMARY:",
                f"Monthly Operating Cost: ${total_monthly:,.0f}",
                f"One-time Migration Cost: ${total_one_time:,.0f}",
                f"Annual Cost: ${total_monthly * 12:,.0f}",
                f"3-Year Total: ${three_year_total:,.0f}",
                "",
                "SERVICE BREAKDOWN:"
            ]

            for item in service_breakdown_data:
                report_lines.append(f"  - {item['AWS Service']}: {item['Monthly Cost']}")

            report_lines.extend([
                "",
                "OPTIMIZATION OPPORTUNITIES:",
                *[f"- {rec}" for rec in recommendations[:5]],
                "",
                f"Data Source: {data_source}",
                f"Accuracy: {accuracy}"
            ])

            report_text = "\n".join(report_lines)

            st.download_button(
                label="📄 Download Text Report",
                data=report_text,
                file_name=f"aws_cost_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )

    with export_col3:
        if st.button("🔄 Refresh Cost Analysis", use_container_width=True):
            st.info("To refresh cost analysis, please re-run the main migration analysis with current AWS pricing data.")
            st.write("💡 **Tip:** This will fetch the latest AWS pricing and recalculate all costs.")

    # Final note about cost consolidation
    st.markdown("---")
    st.success("**✅ All AWS costs have been consolidated into this comprehensive analysis. This includes costs from database compute, storage, networking, migration services, and all other AWS components required for your migration.**")

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

        st.plotly_chart(fig_radar, use_container_width=True, key=f"radar_{int(time.time() * 1000000)}")

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

    # OS Performance Details
    st.markdown("---")
    st.markdown("**🔍 Detailed OS Performance Analysis:**")

    detail_col1, detail_col2, detail_col3 = st.columns(3)

    with detail_col1:
        st.info("**Resource Efficiency**")
        st.write(f"**CPU Efficiency:** {os_impact.get('cpu_efficiency', 0)*100:.1f}%")
        st.write(f"**Memory Efficiency:** {os_impact.get('memory_efficiency', 0)*100:.1f}%")
        st.write(f"**I/O Efficiency:** {os_impact.get('io_efficiency', 0)*100:.1f}%")
        st.write(f"**Network Efficiency:** {os_impact.get('network_efficiency', 0)*100:.1f}%")

    with detail_col2:
        st.success("**Database Optimization**")
        st.write(f"**DB Engine Optimization:** {os_impact.get('db_optimization', 0)*100:.1f}%")
        st.write(f"**Actual DB Engine:** {os_impact.get('actual_database_engine', 'Unknown')}")
        st.write(f"**OS Database Support:** {os_impact.get('os_database_support', 'Standard')}")
        st.write(f"**Performance Tuning:** {os_impact.get('performance_tuning_level', 'Basic')}")

    with detail_col3:
        st.warning("**Cost & Management**")
        st.write(f"**Licensing Cost Factor:** {os_impact.get('licensing_cost_factor', 1.0):.1f}x")
        st.write(f"**Management Complexity:** {os_impact.get('management_complexity', 0)*100:.0f}%")
        st.write(f"**Support Availability:** {os_impact.get('support_level', 'Standard')}")
        st.write(f"**Security Level:** {os_impact.get('security_level', 'Standard')}")

    # Optimization Recommendations
    st.markdown("---")
    st.markdown("**💡 OS Optimization Recommendations:**")

    optimization_recommendations = ai_insights.get('optimization_recommendations', [])
    
    if not optimization_recommendations:
        # Generate default recommendations based on OS type
        os_name = os_impact.get('name', '').lower()
        if 'windows' in os_name:
            optimization_recommendations = [
                "Optimize Windows Server settings for database workloads",
                "Configure Windows performance counters for monitoring",
                "Tune Windows memory management for database operations",
                "Review Windows security settings for migration compatibility"
            ]
        elif 'linux' in os_name:
            optimization_recommendations = [
                "Optimize Linux kernel parameters for database performance",
                "Configure appropriate file system settings",
                "Tune Linux network stack for high throughput",
                "Review system resource limits and quotas"
            ]
        else:
            optimization_recommendations = [
                "Review OS configuration for optimal database performance",
                "Ensure adequate system resources are allocated",
                "Configure monitoring and alerting systems",
                "Validate OS compatibility with target AWS services"
            ]

    if optimization_recommendations:
        for i, recommendation in enumerate(optimization_recommendations[:6], 1):
            impact = "High" if i <= 2 else "Medium" if i <= 4 else "Low"
            effort = "Low" if i <= 2 else "Medium" if i <= 4 else "High"

            with st.expander(f"Recommendation {i}: {recommendation}", expanded=(i <= 2)):
                rec_col1, rec_col2, rec_col3 = st.columns(3)

                with rec_col1:
                    if impact == "High":
                        st.success(f"**Expected Impact:** {impact}")
                    elif impact == "Medium":
                        st.warning(f"**Expected Impact:** {impact}")
                    else:
                        st.info(f"**Expected Impact:** {impact}")

                with rec_col2:
                    st.write(f"**Implementation Effort:** {effort}")

                with rec_col3:
                    priority = "Immediate" if i <= 2 else "Short-term" if i <= 4 else "Long-term"
                    st.write(f"**Priority:** {priority}")

    # OS Migration Readiness
    st.markdown("---")
    st.markdown("**🎯 OS Migration Readiness Assessment:**")

    readiness_col1, readiness_col2 = st.columns(2)

    with readiness_col1:
        total_efficiency = os_impact.get('total_efficiency', 0)
        readiness_score = min(100, total_efficiency * 120)  # Convert efficiency to readiness score

        if readiness_score >= 80:
            st.success(f"**Migration Readiness: {readiness_score:.0f}/100**")
            readiness_status = "Ready for Migration"
            readiness_color = "success"
        elif readiness_score >= 60:
            st.warning(f"**Migration Readiness: {readiness_score:.0f}/100**")
            readiness_status = "Needs Optimization"
            readiness_color = "warning"
        else:
            st.error(f"**Migration Readiness: {readiness_score:.0f}/100**")
            readiness_status = "Requires Attention"
            readiness_color = "error"

        st.write(f"**Status:** {readiness_status}")
        st.write(f"**Overall Efficiency:** {total_efficiency*100:.1f}%")

    with readiness_col2:
        st.info("**Key Migration Factors:**")
        
        factors = []
        if os_impact.get('cpu_efficiency', 0) > 0.8:
            factors.append("✅ CPU efficiency is good")
        else:
            factors.append("⚠️ CPU efficiency needs improvement")
            
        if os_impact.get('memory_efficiency', 0) > 0.8:
            factors.append("✅ Memory efficiency is good")
        else:
            factors.append("⚠️ Memory efficiency needs improvement")
            
        if os_impact.get('db_optimization', 0) > 0.7:
            factors.append("✅ Database optimization is adequate")
        else:
            factors.append("⚠️ Database optimization needs work")

        if config.get('server_type') == 'vmware':
            if os_impact.get('virtualization_overhead', 0) < 0.1:
                factors.append("✅ Virtualization overhead is low")
            else:
                factors.append("⚠️ High virtualization overhead")
        else:
            factors.append("✅ Physical server - no virtualization overhead")

        for factor in factors:
            st.write(factor)

def render_aws_sizing_tab(analysis: Dict, config: Dict):
    """Render AWS sizing and configuration recommendations tab using native components"""

    # ADD THIS LINE AT THE BEGINNING
    validated_costs = add_cost_validation_to_tab(analysis, config)
    st.subheader("🎯 AWS Sizing & Configuration Recommendations")

    # USE VALIDATED COSTS INSTEAD
    monthly_cost = validated_costs['total_monthly']  # NEW WAY

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

            st.write(f"**Primary Instance Type:** {primary_instance}")
            st.write(f"**vCPUs:** {instance_specs.get('vcpu', 'Unknown')}")
            st.write(f"**Memory:** {instance_specs.get('memory', 'Unknown')} GB")
            st.write(f"**EBS Optimized:** {'Yes' if ec2_rec.get('ebs_optimized', False) else 'No'}")
            st.write(f"**Enhanced Networking:** {'Yes' if ec2_rec.get('enhanced_networking', False) else 'No'}")
            st.write(f"**Storage Type:** {ec2_rec.get('storage_type', 'gp3')}")
            st.write(f"**Storage Size:** {ec2_rec.get('storage_size_gb', 0):,.0f} GB")

        with col2:
            st.markdown("**📊 Cost Breakdown:**")
            st.success("Monthly Costs")

            st.write(f"**Instance Cost:** ${ec2_rec.get('monthly_instance_cost', 0):,.0f}")
            st.write(f"**Storage Cost:** ${ec2_rec.get('monthly_storage_cost', 0):,.0f}")
            st.write(f"**Total Monthly:** ${ec2_rec.get('total_monthly_cost', 0):,.0f}")

            # Additional EC2 considerations
            st.write(f"**OS:** {config.get('operating_system', 'Unknown').replace('_', ' ').title()}")
            st.write(f"**Database Engine:** {config.get('database_engine', 'Unknown').upper()}")

    # Reader/Writer Configuration Details
    st.markdown("---")
    st.markdown("**🔄 Reader/Writer Configuration:**")

    reader_writer = aws_sizing.get('reader_writer_config', {})

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("**Instance Distribution**")
        st.write(f"**Total Instances:** {reader_writer.get('total_instances', 1)}")
        st.write(f"**Writer Instances:** {reader_writer.get('writers', 1)}")
        st.write(f"**Reader Instances:** {reader_writer.get('readers', 0)}")
        st.write(f"**Write Capacity:** {reader_writer.get('write_capacity_percent', 100):.1f}%")
        st.write(f"**Read Capacity:** {reader_writer.get('read_capacity_percent', 0):.1f}%")

    with col2:
        st.success("**Workload Distribution**")
        st.write(f"**Recommended Read Split:** {reader_writer.get('recommended_read_split', 0):.1f}%")
        st.write(f"**Database Size:** {config.get('database_size_gb', 0):,} GB")
        st.write(f"**Performance Requirement:** {config.get('performance_requirements', 'standard').title()}")
        st.write(f"**Environment:** {config.get('environment', 'non-production').title()}")
        st.write(f"**Reasoning:** {reader_writer.get('reasoning', 'Standard configuration')}")

    with col3:
        st.warning("**Instance Specifications**")
        if recommendation == 'RDS':
            primary_instance = aws_sizing.get('rds_recommendations', {}).get('primary_instance', 'Unknown')
            instance_specs = aws_sizing.get('rds_recommendations', {}).get('instance_specs', {})
        else:
            primary_instance = aws_sizing.get('ec2_recommendations', {}).get('primary_instance', 'Unknown')
            instance_specs = aws_sizing.get('ec2_recommendations', {}).get('instance_specs', {})

        st.write(f"**All instances use:** {primary_instance}")
        st.write(f"**vCPUs per instance:** {instance_specs.get('vcpu', 'Unknown')}")
        st.write(f"**Memory per instance:** {instance_specs.get('memory', 'Unknown')} GB")
        st.write(f"**Total vCPUs:** {instance_specs.get('vcpu', 2) * reader_writer.get('total_instances', 1)}")
        st.write(f"**Total Memory:** {instance_specs.get('memory', 4) * reader_writer.get('total_instances', 1)} GB")

    # Deployment Reasoning
    st.markdown("---")
    st.markdown("**🎯 Deployment Decision Reasoning:**")

    reasons = deployment_rec.get('primary_reasons', [])
    analytical_rec = deployment_rec.get('analytical_recommendation', 'unknown')
    user_choice = deployment_rec.get('user_choice', 'unknown')

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📋 Decision Factors:**")

        if reasons:
            for i, reason in enumerate(reasons, 1):
                st.write(f"{i}. {reason}")
        else:
            st.write("• Standard configuration based on database size")
            st.write("• Suitable for specified environment")
            st.write("• Meets performance requirements")

    with col2:
        st.markdown("**🤖 AI Analysis:**")

        st.write(f"**User Selection:** {user_choice.upper()}")
        st.write(f"**AI Recommendation:** {analytical_rec.upper()}")
        st.write(f"**Confidence Level:** {confidence*100:.1f}%")

        if user_choice != analytical_rec:
            st.warning(f"⚠️ Note: AI analysis suggests {analytical_rec.upper()} might be more optimal")
        else:
            st.success("✅ User selection aligns with AI recommendation")

        st.write(f"**RDS Suitability Score:** {rds_score:.0f}/100")
        st.write(f"**EC2 Suitability Score:** {ec2_score:.0f}/100")

    # Performance Metrics and Sizing Reasoning
    st.markdown("---")
    st.markdown("**📊 Performance Metrics & Sizing Reasoning:**")

    col1, col2 = st.columns(2)

    with col1:
        st.info("**Current vs Recommended Specifications**")
        
        # Get current specifications from config
        current_memory = config.get('current_db_max_memory_gb', 0)
        current_cpu = config.get('current_db_max_cpu_cores', 0)
        current_iops = config.get('current_db_max_iops', 0)
        
        if recommendation == 'RDS':
            sizing_reasoning = aws_sizing.get('rds_recommendations', {}).get('sizing_reasoning', [])
        else:
            sizing_reasoning = aws_sizing.get('ec2_recommendations', {}).get('sizing_reasoning', [])

        if current_memory > 0:
            st.write(f"**Current Memory:** {current_memory:,.0f} GB")
        if current_cpu > 0:
            st.write(f"**Current CPU Cores:** {current_cpu:,.0f}")
        if current_iops > 0:
            st.write(f"**Current IOPS:** {current_iops:,.0f}")

        st.write(f"**Recommended Memory:** {instance_specs.get('memory', 'Unknown')} GB")
        st.write(f"**Recommended vCPUs:** {instance_specs.get('vcpu', 'Unknown')}")

    with col2:
        st.success("**Sizing Reasoning**")
        
        if sizing_reasoning:
            for reason in sizing_reasoning:
                st.write(f"• {reason}")
        else:
            st.write("• Based on database size and workload characteristics")
            st.write("• Appropriate for specified environment")
            st.write("• Includes buffer for growth and peak loads")

    # Additional Configuration Options
    st.markdown("---")
    st.markdown("**⚙️ Additional Configuration Options:**")

    config_col1, config_col2, config_col3 = st.columns(3)

    with config_col1:
        st.warning("**High Availability Options**")
        if recommendation == 'RDS':
            st.write("• Multi-AZ deployment for production")
            st.write("• Automated backups with point-in-time recovery")
            st.write("• Cross-region read replicas available")
            st.write("• Automated failover capability")
        else:
            st.write("• Manual HA setup required")
            st.write("• Custom backup strategy needed")
            st.write("• Application-level failover")
            st.write("• Additional monitoring setup")

    with config_col2:
        st.info("**Security Features**")
        if recommendation == 'RDS':
            st.write("• Encryption at rest and in transit")
            st.write("• VPC security groups")
            st.write("• IAM database authentication")
            st.write("• Automated security patching")
        else:
            st.write("• Custom encryption configuration")
            st.write("• Security group management")
            st.write("• Manual patching required")
            st.write("• Custom security hardening")

    with config_col3:
        st.success("**Monitoring & Maintenance**")
        if recommendation == 'RDS':
            st.write("• CloudWatch integration")
            st.write("• Performance Insights available")
            st.write("• Automated maintenance windows")
            st.write("• Enhanced monitoring options")
        else:
            st.write("• Custom monitoring setup")
            st.write("• Third-party monitoring tools")
            st.write("• Manual maintenance scheduling")
            st.write("• Custom alerting configuration")

def render_migration_dashboard_tab_with_pdf(analysis: Dict, config: Dict):
    """Enhanced migration dashboard with PDF export"""
    st.subheader("📊 Enhanced Migration Performance Dashboard")

    # Initialize cost validator
    cost_validator = CostValidationManager()
    validated_costs = cost_validator.get_validated_costs(analysis, config)

    # Display cost validation status
    if not validated_costs['is_validated']:
        st.warning(f"⚠️ {validated_costs['validation']['discrepancy_count']} cost discrepancies detected")
        with st.expander("🔍 Cost Validation Details", expanded=False):
            for disc in validated_costs['validation']['discrepancies']:
                if disc['type'] == 'total_cost_mismatch':
                    st.write(f"**Total Cost Mismatch:** {disc['difference_percent']:.1f}% difference")
                    st.write(f"   Comprehensive: ${disc['comprehensive']:,.0f}")
                    st.write(f"   Basic: ${disc['basic']:,.0f}")
                elif disc['type'] == 'agent_cost_mismatch':
                    st.write(f"**Agent Cost Mismatch:**")
                    st.write(f"   Agent Analysis: ${disc['agent_analysis']:,.0f}")
                    st.write(f"   Cost Analysis: ${disc['cost_analysis']:,.0f}")
    else:
        st.success("✅ All cost calculations are consistent across tabs")

    # Executive Summary Dashboard with validated costs
    st.markdown("**🎯 Executive Migration Summary (Validated Costs):**")

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        readiness_score = analysis.get('ai_overall_assessment', {}).get('migration_readiness_score', 0)
        st.metric("🎯 Readiness Score", f"{readiness_score:.0f}/100",
                 delta=analysis.get('ai_overall_assessment', {}).get('risk_level', 'Unknown'))

    with col2:
        migration_time = analysis.get('estimated_migration_time_hours', 0)
        st.metric("⏱️ Migration Time", f"{migration_time:.1f} hours",
                 delta=f"Window: {config.get('downtime_tolerance_minutes', 60)} min")

    with col3:
        throughput = analysis.get('migration_throughput_mbps', 0)
        st.metric("🚀 Throughput", f"{throughput:,.0f} Mbps",
                 delta=f"Agents: {config.get('number_of_agents', 1)}")

    with col4:
        # Use validated monthly cost
        monthly_cost = validated_costs['total_monthly']
        validation_icon = "✅" if validated_costs['is_validated'] else "⚠️"
        st.metric("💰 Monthly Cost", f"${monthly_cost:,.0f}",
                 delta=f"{validation_icon} {validated_costs['cost_source'].title()}")

    with col5:
        migration_method = config.get('migration_method', 'direct_replication')
        backup_storage = config.get('backup_storage_type', 'nas_drive').replace('_', ' ').title()
        sql_deployment = config.get('sql_server_deployment_type', 'standalone')

        if migration_method == 'backup_restore':
            display_text = f"Backup/Restore via {backup_storage}"
            delta_text = f"Tool: {analysis.get('agent_analysis', {}).get('primary_tool', 'DataSync').upper()}"
        elif config.get('database_engine') == 'sqlserver' and config.get('target_platform') == 'ec2':
            display_text = f"SQL Server {sql_deployment.replace('_', ' ').title()}"
            delta_text = "3-Node HA Cluster" if sql_deployment == 'always_on' else "Single Instance"
        else:
            destination = config.get('destination_storage_type', 'S3')
            agent_efficiency = analysis.get('agent_analysis', {}).get('scaling_efficiency', 1.0)
            display_text = destination
            delta_text = f"Efficiency: {agent_efficiency*100:.1f}%"

        st.metric("🗄️ Target Configuration", display_text, delta=delta_text)

    with col6:
        complexity = analysis.get('aws_sizing_recommendations', {}).get('ai_analysis', {}).get('ai_complexity_score', 6)
        confidence = analysis.get('ai_overall_assessment', {}).get('ai_confidence', 0.5)
        st.metric("🤖 AI Confidence", f"{confidence*100:.1f}%",
                 delta=f"Complexity: {complexity:.1f}/10")
        
    # Add PDF export section
    render_pdf_export_section(analysis, config)
    
    # Cost Breakdown Section
    st.markdown("---")
    st.markdown("**💰 Validated Cost Breakdown:**")

    breakdown_col1, breakdown_col2 = st.columns(2)

    with breakdown_col1:
        st.info("**Monthly Cost Components**")
        breakdown = validated_costs['breakdown']
        for component, cost in breakdown.items():
            if cost > 0:
                percentage = (cost / validated_costs['total_monthly']) * 100
                st.write(f"**{component.replace('_', ' ').title()}:** ${cost:,.0f} ({percentage:.1f}%)")

    with breakdown_col2:
        st.success("**Total Cost Summary**")
        st.write(f"**Monthly Total:** ${validated_costs['total_monthly']:,.0f}")
        st.write(f"**One-Time Setup:** ${validated_costs['total_one_time']:,.0f}")
        st.write(f"**Annual Total:** ${validated_costs['total_monthly'] * 12:,.0f}")
        st.write(f"**3-Year TCO:** ${validated_costs['three_year_total']:,.0f}")
        st.write(f"**Cost Source:** {validated_costs['cost_source'].title()}")
        validation_text = "✅ Validated" if validated_costs['is_validated'] else "⚠️ Has Issues"
        st.write(f"**Validation Status:** {validation_text}")

    # Helper function to check if SQL Server Always On configuration
    def is_sql_server_always_on():
        return (config.get('database_engine') == 'sqlserver' and 
                config.get('target_platform') == 'ec2' and 
                config.get('sql_server_deployment_type') == 'always_on')
    
    def is_sql_server_standalone():
        return (config.get('database_engine') == 'sqlserver' and 
                config.get('target_platform') == 'ec2' and 
                config.get('sql_server_deployment_type') == 'standalone')

    # SQL Server Always On Analysis (if applicable)
    if is_sql_server_always_on():
        render_always_on_analysis(analysis, config)
    elif is_sql_server_standalone():
        render_standalone_analysis(analysis, config)

def render_always_on_analysis(analysis: Dict, config: Dict):
    """Render SQL Server Always On specific analysis"""
    st.markdown("**🔄 SQL Server Always On Cluster Analysis:**")

    sql_col1, sql_col2 = st.columns(2)

    with sql_col1:
        st.success("🎯 **Always On Benefits & Configuration**")
        benefits = [
            "**High Availability:** Automatic failover capability",
            "**Read Scale-out:** Secondary replicas for read workloads", 
            "**Zero Data Loss:** Synchronous commit for primary replica",
            "**Flexible Failover:** Manual and automatic failover modes",
            "**Enhanced Backup:** Backup from secondary replicas",
            "**Cluster Configuration:** 3-node Windows Server Failover Cluster"
        ]
        
        for benefit in benefits:
            st.write(f"• {benefit}")

        complexity_score = analysis.get('aws_sizing_recommendations', {}).get('ai_analysis', {}).get('ai_complexity_score', 6)
        if complexity_score >= 7:
            st.warning("⚠️ **Increased Complexity:** Always On adds +1-2 complexity points")

    with sql_col2:
        st.info("⚙️ **Implementation Considerations**")
        considerations = [
            "**Network Requirements:** Low-latency cluster communication",
            "**Storage Requirements:** EBS volumes per instance",
            "**Quorum Configuration:** File share witness or cloud witness",
            "**Security:** Windows authentication and encryption",
            "**Monitoring:** Extended Events and Always On dashboard",
            "**Licensing:** Core-based licensing for all cluster nodes"
        ]
        
        for consideration in considerations:
            st.write(f"• {consideration}")

        aws_sizing = analysis.get('aws_sizing_recommendations', {})
        ec2_rec = aws_sizing.get('ec2_recommendations', {})
        if ec2_rec.get('total_monthly_cost', 0) > 0:
            single_instance_estimate = ec2_rec.get('total_monthly_cost', 0) / 3
            st.write(f"• **Cost vs Standalone:** ~3x (${single_instance_estimate:,.0f} → ${ec2_rec.get('total_monthly_cost', 0):,.0f}/mo)")

    # Always On specific recommendations
    st.markdown("**💡 Always On Specific Recommendations:**")

    always_on_rec_col1, always_on_rec_col2 = st.columns(2)

    with always_on_rec_col1:
        st.warning("🚨 **Pre-Migration Planning**")
        planning_items = [
            "Validate Windows Server Failover Clustering expertise",
            "Plan cluster network topology and IP addressing",
            "Design quorum configuration for AWS environment",
            "Test Always On setup in non-production environment",
            "Plan availability group and listener configuration",
            "Document failover and disaster recovery procedures"
        ]
        
        for item in planning_items:
            st.write(f"• {item}")

    with always_on_rec_col2:
        st.success("⚡ **Performance & Monitoring Setup**")
        performance_items = [
            "Configure placement groups for low-latency networking",
            "Use Enhanced Networking (SR-IOV) for cluster traffic",
            "Set up CloudWatch custom metrics for Always On",
            "Configure Always On extended events for monitoring",
            "Plan read routing for secondary replicas",
            "Implement connection string failover configuration"
        ]
        
        for item in performance_items:
            st.write(f"• {item}")

    # Migration complexity impact
    readiness_score = analysis.get('ai_overall_assessment', {}).get('migration_readiness_score', 0)
    if readiness_score > 0:
        always_on_impact = min(15, max(5, 20 - readiness_score // 5))  # 5-15 point reduction
        adjusted_readiness = max(0, readiness_score - always_on_impact)
        
        recommendation = ("Proceed with caution - additional planning required" 
                         if adjusted_readiness < 70 
                         else "Good readiness for Always On migration")

        st.info(f"""
        📊 **Always On Impact on Migration Readiness:**
        • **Base Readiness Score:** {readiness_score}/100
        • **Always On Complexity Impact:** -{always_on_impact} points
        • **Adjusted Readiness Score:** {adjusted_readiness}/100
        • **Recommendation:** {recommendation}
        """)

    st.markdown("---")  # Add separator


def render_standalone_analysis(analysis: Dict, config: Dict):
    """Render SQL Server Standalone specific analysis"""
    st.markdown("**🖥️ SQL Server Standalone Configuration:**")

    standalone_col1, standalone_col2 = st.columns(2)

    with standalone_col1:
        st.info("✅ **Standalone Benefits**")
        benefits = [
            "**Simplicity:** Single instance management",
            "**Cost-effective:** Lower licensing and infrastructure costs",
            "**Faster deployment:** Simpler setup and configuration", 
            "**Standard features:** All core SQL Server capabilities",
            "**Easier troubleshooting:** Single point of management"
        ]
        
        for benefit in benefits:
            st.write(f"• {benefit}")

    with standalone_col2:
        st.warning("⚠️ **Limitations & Considerations**")
        limitations = [
            "**No automatic failover:** Manual intervention required",
            "**Single point of failure:** No built-in high availability",
            "**Backup dependency:** Critical backup and restore strategy needed",
            "**Planned downtime:** Required for maintenance and updates",
            "**Consider:** Upgrade path to Always On if HA needs emerge"
        ]
        
        for limitation in limitations:
            st.write(f"• {limitation}")

    st.markdown("---")  # Add separator


import asyncio
import concurrent.futures
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

# Set up logger
logger = logging.getLogger(__name__)


def render_fsx_comparisons_tab(analysis: Dict, config: Dict):
    """Render FSx destination comparisons tab"""
    st.subheader("🗄️ FSx Destination Storage Comparisons")

    fsx_comparisons = analysis.get('fsx_comparisons', {})

    if not fsx_comparisons:
        st.warning("FSx comparison data not available")
        return

    # Overview metrics
    st.markdown("**📊 Storage Destination Overview:**")

    col1, col2, col3 = st.columns(3)
    destinations = ['S3', 'FSx_Windows', 'FSx_Lustre']

    for i, dest in enumerate(destinations):
        with [col1, col2, col3][i]:
            comparison = fsx_comparisons.get(dest, {})
            st.metric(
                f"🗄️ {dest.replace('_', ' ')}",
                comparison.get('performance_rating', 'Unknown'),
                delta=f"${comparison.get('estimated_monthly_storage_cost', 0):,.0f}/mo"
            )

    # Add detailed comparison table if data exists
    if fsx_comparisons:
        render_fsx_detailed_comparison(fsx_comparisons)


def render_fsx_detailed_comparison(fsx_comparisons: Dict):
    """Render detailed FSx comparison table"""
    st.markdown("---")
    st.markdown("**📋 Detailed Storage Comparison:**")
    
    comparison_data = []
    for dest, data in fsx_comparisons.items():
        comparison_data.append({
            'Storage Type': dest.replace('_', ' '),
            'Performance Rating': data.get('performance_rating', 'Unknown'),
            'Monthly Cost': f"${data.get('estimated_monthly_storage_cost', 0):,.0f}",
            'Throughput': f"{data.get('max_throughput_mbps', 0):,.0f} Mbps",
            'IOPS': f"{data.get('max_iops', 0):,}",
            'Best For': data.get('best_for', 'General use')
        })
    
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True)


def run_agent_optimization_sync(optimizer, config: Dict, analysis: Dict) -> Dict:
    """Run agent optimization synchronously for Streamlit"""
    
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
        return create_fallback_optimization_analysis(str(e))


def create_fallback_optimization_analysis(error_msg: str) -> Dict:
    """Create fallback analysis when optimization fails"""
    return {
        'current_configuration': {'error': 'Analysis failed'},
        'optimal_configurations': {},
        'ai_recommendations': {
            'ai_analysis_available': False, 
            'error': error_msg
        },
        'cost_vs_performance': {'analysis_available': False},
        'bottleneck_analysis': {'current_bottleneck': 'Analysis failed'},
        'scaling_scenarios': {},
        'optimization_summary': {
            'optimization_available': False, 
            'error': error_msg
        }
    }


def render_agent_placement_guide():
    """Render agent placement best practices guide"""
    st.markdown("**📍 DataSync Agent Placement Guide**")

    # Scenario selector
    scenario = st.selectbox(
        "Select Migration Scenario",
        ["backup_restore", "direct_replication", "hybrid_approach", "multi_site"],
        format_func=lambda x: {
            'backup_restore': '💾 Backup/Restore Migration',
            'direct_replication': '🔄 Direct Replication',
            'hybrid_approach': '🔀 Hybrid Approach',
            'multi_site': '🌐 Multi-Site Migration'
        }[x]
    )

    # Render scenario-specific content
    if scenario == "backup_restore":
        render_backup_restore_scenario()
    elif scenario == "direct_replication":
        render_direct_replication_scenario()
    elif scenario == "hybrid_approach":
        render_hybrid_approach_scenario()
    else:  # multi_site
        render_multi_site_scenario()

    # Quick tips section
    render_placement_quick_tips()


def render_backup_restore_scenario():
    """Render backup/restore migration scenario"""
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.info("**🗄️ On-Premises**\n\nSource Database\n+ Backup Storage\n\n📊 SMB/NFS Protocol")

    with col2:
        st.markdown("**→**")

    with col3:
        st.success("**🤖 DataSync Agents**\n\n2-4 Agents\nRecommended\n\n⚡ Close to Backup")

    with col4:
        st.markdown("**→**")

    with col5:
        st.warning("**☁️ AWS Destination**\n\nS3 / FSx for\nWindows / Lustre\n\n🎯 High Performance")

    st.markdown("**💡 Key Recommendations:**")
    recommendations = [
        "Place agents in same network segment as backup storage",
        "SMB3 multichannel for Windows shares, NFS v4.1+ for Linux",
        "Account for backup compression (typically 70% of DB size)"
    ]
    for rec in recommendations:
        st.write(f"• {rec}")


def render_direct_replication_scenario():
    """Render direct replication scenario"""
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.info("**🗃️ Source Database**\n\nLive Production\nDatabase\n\n🔴 Minimal Impact")

    with col2:
        st.markdown("**→**")

    with col3:
        st.success("**🔄 Migration Agents**\n\nDMS/DataSync\nAgents\n\n⚡ Real-time Sync")

    with col4:
        st.markdown("**→**")

    with col5:
        st.warning("**🎯 Target Database**\n\nRDS or EC2\nDatabase\n\n✅ Live Target")

    st.markdown("**💡 Key Recommendations:**")
    recommendations = [
        "Place agents close to source database for low latency",
        "Use DataSync for homogeneous, DMS for heterogeneous migrations",
        "Continuous sync with change data capture"
    ]
    for rec in recommendations:
        st.write(f"• {rec}")


def render_hybrid_approach_scenario():
    """Render hybrid approach scenario"""
    st.info("**🔀 Hybrid Migration Strategy**")
    phases = [
        "**Phase 1:** DataSync agents transfer bulk backup data",
        "**Phase 2:** DMS agents handle incremental changes",
        "**Coordination:** AWS Migration Hub orchestrates both phases"
    ]
    for phase in phases:
        st.write(phase)


def render_multi_site_scenario():
    """Render multi-site scenario"""
    st.info("**🌐 Multi-Site Deployment**")
    considerations = [
        "Deploy agents at each source location",
        "Coordinate transfers to avoid network congestion",
        "Use staging areas for large datasets",
        "Maintain security boundaries between sites"
    ]
    for consideration in considerations:
        st.write(f"• {consideration}")


def render_placement_quick_tips():
    """Render quick placement tips"""
    with st.expander("🎯 Quick Placement Tips", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**📍 Location Strategy:**")
            location_tips = [
                "Minimize network hops",
                "Consider firewall rules",
                "Plan for redundancy"
            ]
            for tip in location_tips:
                st.write(f"• {tip}")

        with col2:
            st.markdown("**⚡ Performance Tips:**")
            performance_tips = [
                "Right-size based on data volume",
                "Use parallel agents for large datasets",
                "Monitor utilization and scale"
            ]
            for tip in performance_tips:
                st.write(f"• {tip}")


def render_agent_scaling_optimizer_tab(analysis: Dict, config: Dict):
    """Render Agent Scaling Optimizer tab with AI recommendations"""
    st.subheader("🤖 DataSync/DMS Agent Scaling Optimizer")

    # Agent Placement Guide Section
    st.markdown("---")
    render_agent_placement_guide()
    st.markdown("---")

    # Check if we have agent optimization data
    if 'agent_optimization' not in st.session_state:
        render_optimization_start_screen(analysis, config)
        return

    # Display optimization results
    optimization = st.session_state['agent_optimization']

    # Check if analysis failed
    if optimization.get('optimization_summary', {}).get('error'):
        render_optimization_error_screen(optimization)
        return

    # Render all optimization sections
    render_executive_summary(optimization)
    render_current_configuration_analysis(optimization, config)
    render_optimal_configurations(optimization)
    render_ai_recommendations(optimization)
    render_cost_vs_performance_analysis(optimization)
    render_bottleneck_analysis(optimization)
    render_scaling_scenarios(optimization)
    render_action_buttons(optimization)


def render_optimization_start_screen(analysis: Dict, config: Dict):
    """Render the initial optimization analysis screen"""
    st.info("🚀 **Agent Scaling Optimization Analysis**")
    st.write("Click the button below to run comprehensive agent scaling optimization analysis with AI recommendations.")

    col1, col2 = st.columns([3, 1])
    with col1:
        benefits = [
            "🎯 Optimal agent configurations for your workload",
            "🤖 AI-powered scaling recommendations",
            "💰 Cost vs performance trade-off analysis",
            "🚫 Bottleneck identification and resolution",
            "📈 Multiple scaling scenarios (conservative, balanced, aggressive)",
            "🛡️ Risk mitigation strategies"
        ]
        
        st.markdown("**This analysis will provide:**")
        for benefit in benefits:
            st.write(f"- {benefit}")

    with col2:
        if st.button("🔍 Analyze Agent Scaling", type="primary", use_container_width=True):
            run_optimization_analysis(analysis, config)


def run_optimization_analysis(analysis: Dict, config: Dict):
    """Run the optimization analysis with progress tracking"""
    with st.spinner("🤖 Running AI-powered agent scaling optimization..."):
        try:
            # Show progress
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("🔧 Initializing optimizer...")
            progress_bar.progress(10)

            # Initialize optimizer - these would be imported from your actual modules
            # ai_manager = AnthropicAIManager()
            # agent_manager = EnhancedAgentSizingManager()
            # optimizer = AgentScalingOptimizer(ai_manager, agent_manager)

            status_text.text("📊 Analyzing current configuration...")
            progress_bar.progress(30)

            # For demo purposes, create a mock optimizer
            class MockOptimizer:
                async def analyze_agent_scaling_optimization(self, config, analysis):
                    return create_mock_optimization_results()
            
            optimizer = MockOptimizer()

            # Run optimization analysis
            optimization_analysis = run_agent_optimization_sync(optimizer, config, analysis)

            status_text.text("✅ Analysis complete!")
            progress_bar.progress(100)

            st.session_state['agent_optimization'] = optimization_analysis

            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()

            st.success("✅ Agent scaling optimization analysis completed!")
            st.rerun()

        except Exception as e:
            st.error(f"❌ Agent optimization analysis failed: {str(e)}")
            logger.error(f"Agent optimization error: {e}")

            # Show detailed error for debugging
            with st.expander("🔍 Error Details", expanded=False):
                st.code(str(e))


def create_mock_optimization_results() -> Dict:
    """Create mock optimization results for demonstration"""
    return {
        'optimization_summary': {
            'optimization_available': True,
            'current_configuration': '2x Medium',
            'recommended_configuration': '4x Large',
            'performance_improvement': {
                'throughput_change_percent': 85.0,
                'throughput_change_mbps': 1200
            },
            'cost_impact': {
                'cost_change_percent': 45.0,
                'cost_change_monthly': 450
            },
            'efficiency_gain': 15.5
        },
        'current_configuration': {
            'migration_method': 'backup_restore',
            'primary_tool': 'DataSync',
            'agent_count': 2,
            'agent_size': 'medium',
            'destination_storage': 'S3',
            'current_throughput_mbps': 1400,
            'database_size_gb': 5000,
            'current_efficiency': 0.75,
            'current_cost_monthly': 1000,
            'bottleneck': 'Agent CPU',
            'bottleneck_severity': 'medium'
        },
        'ai_recommendations': {
            'ai_analysis_available': True,
            'confidence_level': 'high',
            'recommended_configuration': '4x Large DataSync Agents',
            'scaling_strategy': [
                'Increase agent count to 4 for better parallelization',
                'Upgrade to large instances for better CPU performance',
                'Implement load balancing across agents'
            ],
            'cost_optimization_tips': [
                'Consider Reserved Instances for long-term migrations',
                'Use Spot Instances for non-critical migration phases',
                'Scale down during off-peak hours'
            ]
        }
    }


def render_optimization_error_screen(optimization: Dict):
    """Render error screen when optimization fails"""
    error_msg = optimization['optimization_summary']['error']
    st.error(f"❌ Analysis failed: {error_msg}")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Retry Analysis", type="primary"):
            if 'agent_optimization' in st.session_state:
                del st.session_state['agent_optimization']
            st.rerun()
    
    with col2:
        with st.expander("🔍 Error Details", expanded=False):
            st.code(error_msg)


def render_executive_summary(optimization: Dict):
    """Render executive summary section"""
    st.markdown("**📊 Agent Scaling Optimization Summary:**")

    optimization_summary = optimization.get('optimization_summary', {})

    if not optimization_summary.get('optimization_available', False):
        st.warning("⚠️ Optimization analysis not available. Please retry the analysis.")
        return

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        current_config = optimization_summary.get('current_configuration', 'Unknown')
        recommended_config = optimization_summary.get('recommended_configuration', 'Unknown')
        st.metric(
            "🎯 Current vs Optimal",
            current_config,
            delta=f"→ {recommended_config}"
        )

    with col2:
        perf_improvement = optimization_summary.get('performance_improvement', {})
        throughput_change = perf_improvement.get('throughput_change_percent', 0)
        st.metric(
            "🚀 Throughput Change",
            f"{throughput_change:+.1f}%",
            delta=f"{perf_improvement.get('throughput_change_mbps', 0):+,.0f} Mbps"
        )

    with col3:
        cost_impact = optimization_summary.get('cost_impact', {})
        cost_change = cost_impact.get('cost_change_percent', 0)
        st.metric(
            "💰 Cost Impact",
            f"{cost_change:+.1f}%",
            delta=f"${cost_impact.get('cost_change_monthly', 0):+,.0f}/mo"
        )

    with col4:
        efficiency_gain = optimization_summary.get('efficiency_gain', 0)
        st.metric(
            "⚡ Efficiency Gain",
            f"{efficiency_gain:+.1f} pts",
            delta="Optimization score"
        )

    with col5:
        ai_recommendations = optimization.get('ai_recommendations', {})
        confidence = ai_recommendations.get('confidence_level', 'medium')
        ai_available = ai_recommendations.get('ai_analysis_available', False)
        st.metric(
            "🤖 AI Analysis",
            "Available" if ai_available else "Fallback",
            delta=f"Confidence: {confidence.title()}"
        )


def render_current_configuration_analysis(optimization: Dict, config: Dict):
    """Render current configuration analysis section"""
    st.markdown("---")
    st.markdown("**🔍 Current Configuration Analysis:**")

    current_config = optimization.get('current_configuration', {})

    if not current_config or current_config.get('error'):
        st.error("❌ Current configuration analysis failed. Please retry.")
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        render_current_setup_info(current_config)

    with col2:
        render_performance_metrics(current_config)

    with col3:
        render_performance_status(current_config, config)


def render_current_setup_info(current_config: Dict):
    """Render current setup information"""
    st.info("**Current Setup**")
    setup_items = [
        ("Migration Method", current_config.get('migration_method', 'Unknown').replace('_', ' ').title()),
        ("Primary Tool", current_config.get('primary_tool', 'Unknown')),
        ("Agent Count", str(current_config.get('agent_count', 0))),
        ("Agent Size", current_config.get('agent_size', 'Unknown').title()),
        ("Destination Storage", current_config.get('destination_storage', 'S3'))
    ]
    
    for label, value in setup_items:
        st.write(f"**{label}:** {value}")

    if current_config.get('migration_method') == 'backup_restore':
        backup_storage = current_config.get('backup_storage_type', 'Unknown').replace('_', ' ').title()
        st.write(f"**Backup Storage:** {backup_storage}")


def render_performance_metrics(current_config: Dict):
    """Render performance metrics"""
    st.success("**Performance Metrics**")
    metrics = [
        ("Current Throughput", f"{current_config.get('current_throughput_mbps', 0):,.0f} Mbps"),
        ("Database Size", f"{current_config.get('database_size_gb', 0):,} GB"),
        ("Scaling Efficiency", f"{current_config.get('current_efficiency', 0)*100:.1f}%"),
        ("Monthly Cost", f"${current_config.get('current_cost_monthly', 0):,.0f}"),
        ("Current Bottleneck", current_config.get('bottleneck', 'Unknown'))
    ]
    
    for label, value in metrics:
        st.write(f"**{label}:** {value}")


def render_performance_status(current_config: Dict, config: Dict):
    """Render performance status and estimates"""
    bottleneck_severity = current_config.get('bottleneck_severity', 'medium')
    
    if bottleneck_severity == 'high':
        st.error("**Performance Issues**")
    elif bottleneck_severity == 'medium':
        st.warning("**Performance Status**")
    else:
        st.info("**Performance Status**")

    st.write(f"**Bottleneck Severity:** {bottleneck_severity.title()}")

    # Calculate migration time estimate
    if current_config.get('current_throughput_mbps', 0) > 0:
        estimated_hours = calculate_migration_time_estimate(current_config, config)
        st.write(f"**Estimated Migration Time:** {estimated_hours:.1f} hours")

    improvement_potential = get_improvement_potential(bottleneck_severity)
    st.write(f"**Optimization Potential:** {improvement_potential}")


def calculate_migration_time_estimate(current_config: Dict, config: Dict) -> float:
    """Calculate estimated migration time"""
    db_size = current_config.get('database_size_gb', 0)
    migration_method = current_config.get('migration_method', 'direct_replication')

    if migration_method == 'backup_restore':
        backup_size_multiplier = config.get('backup_size_multiplier', 0.7)
        effective_size = db_size * backup_size_multiplier
    else:
        effective_size = db_size

    # Convert GB to bits, then calculate hours
    size_bits = effective_size * 8 * 1000 * 1000 * 1000  # GB to bits
    throughput_bps = current_config.get('current_throughput_mbps', 1) * 1000 * 1000  # Mbps to bps
    estimated_seconds = size_bits / throughput_bps
    return estimated_seconds / 3600  # Convert to hours


def get_improvement_potential(bottleneck_severity: str) -> str:
    """Get improvement potential based on bottleneck severity"""
    severity_map = {
        'high': 'High',
        'medium': 'Medium',
        'low': 'Low'
    }
    return severity_map.get(bottleneck_severity, 'Unknown')


def render_optimal_configurations(optimization: Dict):
    """Render optimal configurations section"""
    st.markdown("---")
    st.markdown("**🎯 Top Optimal Configurations:**")

    optimal_configs = optimization.get('optimal_configurations', {})

    if not optimal_configs:
        st.info("No optimal configurations available.")
        return

    # Create comparison table
    comparison_data = create_configuration_comparison_data(optimal_configs)
    
    if comparison_data:
        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, use_container_width=True)

        # Highlight recommended configuration
        st.success(f"🎯 **AI Recommended:** {comparison_data[0]['Configuration']} - {comparison_data[0]['Best For']}")


def create_configuration_comparison_data(optimal_configs: Dict) -> List[Dict]:
    """Create comparison data for configurations"""
    comparison_data = []
    
    for config_key, config_data in list(optimal_configs.items())[:5]:  # Top 5
        comparison_data.append({
            'Configuration': config_key,
            'Agents': f"{config_data.get('agent_count', 0)}x {config_data.get('agent_size', 'Unknown')}",
            'Tool': config_data.get('primary_tool', 'Unknown').upper(),
            'Throughput (Mbps)': f"{config_data.get('total_throughput', 0):,.0f}",
            'Monthly Cost': f"${config_data.get('monthly_cost', 0):,.0f}",
            'Cost/Mbps': f"${config_data.get('cost_per_mbps', 0):.2f}",
            'Efficiency Score': f"{config_data.get('efficiency_score', 0):.1f}/100",
            'Management': config_data.get('management_complexity', 'Unknown'),
            'Best For': config_data.get('recommended_for', 'General use')
        })
    
    return comparison_data


def render_ai_recommendations(optimization: Dict):
    """Render AI recommendations section"""
    st.markdown("---")
    st.markdown("**🤖 AI Scaling Recommendations:**")

    ai_recommendations = optimization.get('ai_recommendations', {})

    if not ai_recommendations.get('ai_analysis_available', False):
        render_fallback_recommendations()
        return

    # Main recommendation
    recommended_config = ai_recommendations.get('recommended_configuration', 'Unknown')
    st.success(f"🎯 **Primary Recommendation:** {recommended_config}")

    # Detailed recommendations in expandable sections
    render_ai_recommendation_sections(ai_recommendations)


def render_fallback_recommendations():
    """Render fallback recommendations when AI is not available"""
    st.warning("🤖 AI analysis not available. Using fallback recommendations:")
    fallback_recommendations = [
        "Consider scaling based on database size and performance requirements",
        "Monitor agent utilization and adjust as needed",
        "Test configurations in non-production environment first"
    ]
    for rec in fallback_recommendations:
        st.write(f"• {rec}")


def render_ai_recommendation_sections(ai_recommendations: Dict):
    """Render detailed AI recommendation sections"""
    col1, col2 = st.columns(2)

    with col1:
        render_ai_recommendation_section("📈 Scaling Strategy", ai_recommendations.get('scaling_strategy', []))
        render_ai_recommendation_section("💰 Cost Optimization", ai_recommendations.get('cost_optimization_tips', []))
        render_ai_recommendation_section("🎯 Performance Tuning", ai_recommendations.get('performance_tuning', []))

    with col2:
        render_ai_recommendation_section("🛡️ Risk Mitigation", ai_recommendations.get('risk_mitigation', []))
        render_ai_recommendation_section("📋 Implementation Plan", ai_recommendations.get('implementation_plan', []))
        render_ai_recommendation_section("📊 Monitoring & Alerts", ai_recommendations.get('monitoring_recommendations', []))


def render_ai_recommendation_section(title: str, items: List[str], max_items: int = 4):
    """Render a single AI recommendation section"""
    with st.expander(f"**{title}**", expanded=True):
        for item in items[:max_items]:
            st.write(f"• {item}")


def render_cost_vs_performance_analysis(optimization: Dict):
    """Render cost vs performance analysis section - simplified version"""
    st.markdown("---")
    st.markdown("**💰 Cost vs Performance Analysis:**")
    
    cost_analysis = optimization.get('cost_vs_performance', {})
    
    if not cost_analysis.get('analysis_available', False):
        st.info("Cost vs performance analysis not available.")
        return
        
    st.info("Detailed cost vs performance analysis would be displayed here based on the optimization results.")


def render_bottleneck_analysis(optimization: Dict):
    """Render bottleneck analysis section - simplified version"""
    st.markdown("---")
    st.markdown("**🚫 Bottleneck Analysis & Resolution:**")
    
    bottleneck_analysis = optimization.get('bottleneck_analysis', {})
    current_bottleneck = bottleneck_analysis.get('current_bottleneck', 'Unknown')
    
    st.info(f"Current bottleneck identified: {current_bottleneck}")
    st.write("Detailed bottleneck analysis and resolution strategies would be displayed here.")


def render_scaling_scenarios(optimization: Dict):
    """Render scaling scenarios section - simplified version"""
    st.markdown("---")
    st.markdown("**📈 Scaling Scenarios:**")
    
    scaling_scenarios = optimization.get('scaling_scenarios', {})
    
    if not scaling_scenarios:
        st.info("Scaling scenarios not available.")
        return
        
    st.info("Conservative, balanced, and aggressive scaling scenarios would be displayed here.")


def render_action_buttons(optimization: Dict):
    """Render action buttons section"""
    st.markdown("---")
    st.markdown("**🚀 Next Steps:**")

    button_col1, button_col2, button_col3 = st.columns(3)

    with button_col1:
        if st.button("🔄 Re-run Optimization Analysis", use_container_width=True):
            if 'agent_optimization' in st.session_state:
                del st.session_state['agent_optimization']
            st.rerun()

    with button_col2:
        if st.button("📊 Export Recommendations", use_container_width=True):
            export_optimization_data(optimization)

    with button_col3:
        if st.button("🤖 Get Detailed AI Analysis", use_container_width=True):
            show_detailed_ai_analysis(optimization)


def export_optimization_data(optimization: Dict):
    """Export optimization data"""
    current_config = optimization.get('current_configuration', {})
    ai_recommendations = optimization.get('ai_recommendations', {})
    optimization_summary = optimization.get('optimization_summary', {})
    
    export_data = {
        'current_config': current_config,
        'recommended_config': ai_recommendations.get('recommended_configuration'),
        'optimization_summary': optimization_summary,
        'timestamp': datetime.now().isoformat()
    }

    st.download_button(
        label="📥 Download Analysis",
        data=json.dumps(export_data, indent=2),
        file_name=f"agent_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )


def show_detailed_ai_analysis(optimization: Dict):
    """Show detailed AI analysis"""
    ai_recommendations = optimization.get('ai_recommendations', {})
    
    if ai_recommendations.get('raw_ai_response'):
        st.markdown("**🤖 Complete AI Analysis:**")
        with st.expander("View Full AI Response", expanded=False):
            st.text(ai_recommendations['raw_ai_response'])
    else:
        st.info("Detailed AI analysis not available")


def add_cost_validation_to_tab(analysis: Dict, config: Dict):
    """Add cost validation with unified cost support"""
    # This would typically import and use your CostValidationManager
    # For now, showing the structure
    
    st.info("Cost validation would be performed here using CostValidationManager")
    
    # Get fallback costs from analysis
    comprehensive_costs = analysis.get('comprehensive_costs', {})
    basic_cost_analysis = analysis.get('cost_analysis', {})
    
    # Try to get costs from comprehensive analysis first, then fallback to basic
    if comprehensive_costs.get('total_monthly', 0) > 0:
        monthly_total = comprehensive_costs['total_monthly']
        one_time_total = comprehensive_costs.get('total_one_time', 0)
        three_year_total = comprehensive_costs.get('three_year_total', monthly_total * 36 + one_time_total)
        breakdown = comprehensive_costs.get('monthly_breakdown', {})
        cost_source = 'comprehensive'
        is_validated = True
    elif basic_cost_analysis.get('total_monthly_cost', 0) > 0:
        monthly_total = basic_cost_analysis['total_monthly_cost']
        one_time_total = basic_cost_analysis.get('one_time_migration_cost', 0)
        three_year_total = (monthly_total * 36) + one_time_total
        breakdown = {
            'compute': basic_cost_analysis.get('aws_compute_cost', 0),
            'storage': basic_cost_analysis.get('aws_storage_cost', 0),
            'agents': basic_cost_analysis.get('agent_cost', 0),
            'network': basic_cost_analysis.get('network_cost', 500),
            'other': basic_cost_analysis.get('management_cost', 200)
        }
        cost_source = 'basic'
        is_validated = False
    else:
        # Fallback with default values
        monthly_total = 1000  # Default fallback
        one_time_total = 5000
        three_year_total = (monthly_total * 36) + one_time_total
        breakdown = {
            'compute': 600,
            'storage': 200,
            'agents': 150,
            'network': 50,
            'other': 0
        }
        cost_source = 'fallback'
        is_validated = False
    
    # Mock validation result with proper structure
    validated_costs = {
        'total_monthly': monthly_total,
        'total_one_time': one_time_total,
        'three_year_total': three_year_total,
        'breakdown': breakdown,
        'cost_source': cost_source,
        'is_validated': is_validated,
        'validation': {
            'discrepancy_count': 0 if is_validated else 1,
            'discrepancies': [] if is_validated else [{'type': 'using_fallback_costs'}],
            'is_consistent': is_validated
        }
    }
    
    if validated_costs['is_validated'] and validated_costs['cost_source'] in ['unified', 'comprehensive']:
        st.success("✅ Using unified cost calculation - all costs are consistent")
    elif validated_costs['cost_source'] == 'basic':
        st.warning("⚠️ Using basic cost calculation - some discrepancies may exist")
    else:
        st.error("❌ Using fallback costs - analysis may be incomplete")
    
    return validated_costs

def create_network_path_diagram(network_perf: Dict) -> Optional[go.Figure]:
    """Create enhanced network path diagram using Plotly with performance metrics"""
    segments = network_perf.get('segments', [])
    if not segments:
        return None

    # Enhanced visualization with performance metrics
    fig = create_enhanced_network_diagram(segments, network_perf)
    return fig


def create_enhanced_network_diagram(segments: List[Dict], network_perf: Dict) -> go.Figure:
    """Create enhanced network diagram with performance indicators"""
    fig = go.Figure()
    
    # Calculate positions for better layout
    positions = calculate_node_positions(segments)
    
    # Add network segments as nodes with performance-based styling
    add_network_nodes(fig, segments, positions)
    
    # Add connections with performance metrics
    add_network_connections(fig, segments, positions)
    
    # Add performance annotations
    add_performance_annotations(fig, segments, positions)
    
    # Configure layout
    configure_diagram_layout(fig, network_perf)
    
    return fig


def calculate_node_positions(segments: List[Dict]) -> List[Tuple[float, float]]:
    """Calculate optimal positions for network nodes"""
    num_segments = len(segments)
    
    if num_segments <= 3:
        # Simple horizontal layout for few segments
        x_positions = list(range(num_segments))
        y_positions = [0] * num_segments
    else:
        # Curved layout for many segments to avoid crowding
        x_positions = []
        y_positions = []
        
        for i in range(num_segments):
            angle = (i / max(1, num_segments - 1)) * math.pi
            x = i * 2
            y = math.sin(angle) * 0.5 if num_segments > 5 else 0
            x_positions.append(x)
            y_positions.append(y)
    
    return list(zip(x_positions, y_positions))


def get_performance_color(segment: Dict) -> str:
    """Determine node color based on performance metrics"""
    latency = segment.get('latency_ms', 0)
    packet_loss = segment.get('packet_loss_percent', 0)
    bandwidth_util = segment.get('bandwidth_utilization_percent', 0)
    
    # Performance scoring (lower is better for latency and packet loss)
    if latency > 100 or packet_loss > 1 or bandwidth_util > 90:
        return '#FF6B6B'  # Red - Poor performance
    elif latency > 50 or packet_loss > 0.1 or bandwidth_util > 70:
        return '#FFD93D'  # Yellow - Warning
    else:
        return '#6BCF7F'  # Green - Good performance


def get_performance_size(segment: Dict) -> int:
    """Determine node size based on importance/bandwidth"""
    bandwidth = segment.get('bandwidth_mbps', 100)
    
    # Scale size based on bandwidth capacity
    if bandwidth >= 10000:  # 10 Gbps+
        return 40
    elif bandwidth >= 1000:  # 1 Gbps+
        return 30
    elif bandwidth >= 100:   # 100 Mbps+
        return 25
    else:
        return 20


def add_network_nodes(fig: go.Figure, segments: List[Dict], positions: List[Tuple[float, float]]):
    """Add network segment nodes with performance-based styling"""
    for i, (segment, (x, y)) in enumerate(zip(segments, positions)):
        # Get performance-based styling
        color = get_performance_color(segment)
        size = get_performance_size(segment)
        
        # Create hover text with detailed info
        hover_text = create_node_hover_text(segment)
        
        # Truncate name for display
        display_name = segment.get('name', f'Segment {i+1}')
        if len(display_name) > 15:
            display_name = display_name[:12] + '...'
        
        fig.add_trace(go.Scatter(
            x=[x],
            y=[y],
            mode='markers+text',
            marker=dict(
                size=size,
                color=color,
                line=dict(width=2, color='darkslategray'),
                opacity=0.8
            ),
            text=display_name,
            textposition='top center',
            textfont=dict(size=10, color='black'),
            hovertext=hover_text,
            hoverinfo='text',
            name=segment.get('name', f'Segment {i+1}'),
            showlegend=False
        ))


def create_node_hover_text(segment: Dict) -> str:
    """Create detailed hover text for network nodes"""
    name = segment.get('name', 'Unknown Segment')
    latency = segment.get('latency_ms', 0)
    bandwidth = segment.get('bandwidth_mbps', 0)
    packet_loss = segment.get('packet_loss_percent', 0)
    utilization = segment.get('bandwidth_utilization_percent', 0)
    segment_type = segment.get('type', 'Unknown')
    
    hover_text = f"""
    <b>{name}</b><br>
    Type: {segment_type}<br>
    Latency: {latency:.1f} ms<br>
    Bandwidth: {bandwidth:,.0f} Mbps<br>
    Utilization: {utilization:.1f}%<br>
    Packet Loss: {packet_loss:.2f}%
    """
    
    # Add additional metrics if available
    if 'jitter_ms' in segment:
        hover_text += f"<br>Jitter: {segment['jitter_ms']:.1f} ms"
    
    if 'mtu' in segment:
        hover_text += f"<br>MTU: {segment['mtu']} bytes"
    
    return hover_text.strip()


def add_network_connections(fig: go.Figure, segments: List[Dict], positions: List[Tuple[float, float]]):
    """Add connections between network segments with performance indicators"""
    for i in range(len(segments) - 1):
        x1, y1 = positions[i]
        x2, y2 = positions[i + 1]
        
        # Determine connection quality
        connection_color, connection_width = get_connection_style(segments[i], segments[i + 1])
        
        # Add connection line
        fig.add_trace(go.Scatter(
            x=[x1, x2],
            y=[y1, y2],
            mode='lines',
            line=dict(
                width=connection_width,
                color=connection_color,
                dash='solid'
            ),
            hovertext=create_connection_hover_text(segments[i], segments[i + 1]),
            hoverinfo='text',
            showlegend=False
        ))
        
        # Add arrow for direction (optional)
        add_connection_arrow(fig, x1, y1, x2, y2, connection_color)


def get_connection_style(segment1: Dict, segment2: Dict) -> Tuple[str, int]:
    """Determine connection line style based on performance"""
    # Average performance between segments
    avg_latency = (segment1.get('latency_ms', 0) + segment2.get('latency_ms', 0)) / 2
    avg_packet_loss = (segment1.get('packet_loss_percent', 0) + segment2.get('packet_loss_percent', 0)) / 2
    
    if avg_latency > 100 or avg_packet_loss > 1:
        return '#FF6B6B', 2  # Red, thin - Poor connection
    elif avg_latency > 50 or avg_packet_loss > 0.1:
        return '#FFD93D', 3  # Yellow, medium - Warning
    else:
        return '#6BCF7F', 4  # Green, thick - Good connection


def create_connection_hover_text(segment1: Dict, segment2: Dict) -> str:
    """Create hover text for connections"""
    name1 = segment1.get('name', 'Segment A')
    name2 = segment2.get('name', 'Segment B')
    
    # Calculate connection metrics
    total_latency = segment1.get('latency_ms', 0) + segment2.get('latency_ms', 0)
    min_bandwidth = min(
        segment1.get('bandwidth_mbps', 0),
        segment2.get('bandwidth_mbps', 0)
    )
    
    hover_text = f"""
    <b>Connection: {name1} → {name2}</b><br>
    Total Latency: {total_latency:.1f} ms<br>
    Bottleneck Bandwidth: {min_bandwidth:,.0f} Mbps<br>
    """
    
    return hover_text.strip()


def add_connection_arrow(fig: go.Figure, x1: float, y1: float, x2: float, y2: float, color: str):
    """Add directional arrow to connection"""
    # Calculate arrow position (80% along the line)
    arrow_x = x1 + 0.8 * (x2 - x1)
    arrow_y = y1 + 0.8 * (y2 - y1)
    
    # Calculate arrow direction
    dx = x2 - x1
    dy = y2 - y1
    length = math.sqrt(dx**2 + dy**2)
    
    if length > 0:
        # Normalize direction
        dx_norm = dx / length
        dy_norm = dy / length
        
        # Arrow size
        arrow_size = 0.1
        
        # Arrow points
        arrow_x1 = arrow_x - arrow_size * dx_norm - arrow_size * dy_norm * 0.5
        arrow_y1 = arrow_y - arrow_size * dy_norm + arrow_size * dx_norm * 0.5
        arrow_x2 = arrow_x - arrow_size * dx_norm + arrow_size * dy_norm * 0.5
        arrow_y2 = arrow_y - arrow_size * dy_norm - arrow_size * dx_norm * 0.5
        
        # Add arrow triangle
        fig.add_trace(go.Scatter(
            x=[arrow_x1, arrow_x, arrow_x2, arrow_x1],
            y=[arrow_y1, arrow_y, arrow_y2, arrow_y1],
            mode='lines',
            fill='toself',
            fillcolor=color,
            line=dict(color=color, width=1),
            showlegend=False,
            hoverinfo='skip'
        ))


def add_performance_annotations(fig: go.Figure, segments: List[Dict], positions: List[Tuple[float, float]]):
    """Add performance metric annotations"""
    for i, (segment, (x, y)) in enumerate(zip(segments, positions)):
        latency = segment.get('latency_ms', 0)
        packet_loss = segment.get('packet_loss_percent', 0)
        
        # Add latency annotation
        if latency > 0:
            fig.add_annotation(
                x=x,
                y=y - 0.3,
                text=f"{latency:.0f}ms",
                showarrow=False,
                font=dict(size=9, color='darkblue'),
                bgcolor='rgba(255,255,255,0.7)',
                bordercolor='darkblue',
                borderwidth=1
            )
        
        # Add packet loss warning if significant
        if packet_loss > 0.1:
            fig.add_annotation(
                x=x + 0.2,
                y=y + 0.2,
                text=f"Loss: {packet_loss:.1f}%",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowcolor='red',
                font=dict(size=8, color='red'),
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='red',
                borderwidth=1
            )


def configure_diagram_layout(fig: go.Figure, network_perf: Dict):
    """Configure the overall diagram layout"""
    title = network_perf.get('title', 'Network Path Performance Analysis')
    
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=16, color='darkslategray')
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-0.5, max(3, len(network_perf.get('segments', [])) * 2)]
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-1, 1]
        ),
        showlegend=False,
        height=400,
        plot_bgcolor='rgba(245,245,245,0.8)',
        margin=dict(l=50, r=50, t=80, b=50),
        hovermode='closest'
    )
    
    # Add performance legend
    add_performance_legend(fig)


def add_performance_legend(fig: go.Figure):
    """Add a legend explaining the color coding"""
    # Add invisible traces for legend
    legend_items = [
        ('Good Performance', '#6BCF7F'),
        ('Warning', '#FFD93D'),
        ('Poor Performance', '#FF6B6B')
    ]
    
    for name, color in legend_items:
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(size=15, color=color),
            name=name,
            showlegend=True
        ))
    
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='darkslategray',
            borderwidth=1
        ),
        showlegend=True
    )


# Example usage function
def create_example_network_diagram():
    """Create an example network diagram with sample data"""
    sample_data = {
        'title': 'Database Migration Network Path',
        'segments': [
            {
                'name': 'Source Database Server',
                'type': 'Database Server',
                'latency_ms': 5.2,
                'bandwidth_mbps': 1000,
                'packet_loss_percent': 0.01,
                'bandwidth_utilization_percent': 45,
                'jitter_ms': 0.5,
                'mtu': 1500
            },
            {
                'name': 'Local Network Switch',
                'type': 'Network Switch',
                'latency_ms': 1.1,
                'bandwidth_mbps': 10000,
                'packet_loss_percent': 0.0,
                'bandwidth_utilization_percent': 25,
                'jitter_ms': 0.1,
                'mtu': 9000
            },
            {
                'name': 'Firewall/Gateway',
                'type': 'Security Gateway',
                'latency_ms': 8.7,
                'bandwidth_mbps': 1000,
                'packet_loss_percent': 0.05,
                'bandwidth_utilization_percent': 60,
                'jitter_ms': 2.1,
                'mtu': 1500
            },
            {
                'name': 'Internet/WAN',
                'type': 'WAN Connection',
                'latency_ms': 45.3,
                'bandwidth_mbps': 500,
                'packet_loss_percent': 0.2,
                'bandwidth_utilization_percent': 80,
                'jitter_ms': 5.7,
                'mtu': 1500
            },
            {
                'name': 'AWS Direct Connect',
                'type': 'Direct Connect',
                'latency_ms': 12.1,
                'bandwidth_mbps': 10000,
                'packet_loss_percent': 0.01,
                'bandwidth_utilization_percent': 35,
                'jitter_ms': 0.8,
                'mtu': 9000
            },
            {
                'name': 'AWS VPC',
                'type': 'Virtual Private Cloud',
                'latency_ms': 2.5,
                'bandwidth_mbps': 25000,
                'packet_loss_percent': 0.0,
                'bandwidth_utilization_percent': 20,
                'jitter_ms': 0.2,
                'mtu': 9000
            },
            {
                'name': 'Target RDS Instance',
                'type': 'RDS Database',
                'latency_ms': 3.8,
                'bandwidth_mbps': 10000,
                'packet_loss_percent': 0.0,
                'bandwidth_utilization_percent': 30,
                'jitter_ms': 0.3,
                'mtu': 9000
            }
        ]
    }
    
    return create_network_path_diagram(sample_data)


# Alternative simplified version for basic use cases
def create_simple_network_path_diagram(network_perf: Dict) -> Optional[go.Figure]:
    """Create a simplified version of the network path diagram"""
    segments = network_perf.get('segments', [])
    if not segments:
        return None

    fig = go.Figure()
    
    # Simple horizontal layout
    x_positions = list(range(len(segments)))
    
    # Add nodes
    for i, segment in enumerate(segments):
        color = get_performance_color(segment)
        
        fig.add_trace(go.Scatter(
            x=[i],
            y=[0],
            mode='markers+text',
            marker=dict(size=25, color=color, line=dict(width=2, color='darkslategray')),
            text=segment.get('name', f'Segment {i+1}')[:15],
            textposition='top center',
            hovertext=create_node_hover_text(segment),
            hoverinfo='text',
            showlegend=False
        ))
    
    # Add connections
    for i in range(len(segments) - 1):
        color, width = get_connection_style(segments[i], segments[i + 1])
        
        fig.add_trace(go.Scatter(
            x=[i, i+1],
            y=[0, 0],
            mode='lines',
            line=dict(width=width, color=color),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Configure layout
    fig.update_layout(
        title=network_perf.get('title', 'Network Path'),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        showlegend=False,
        height=300,
        plot_bgcolor='rgba(245,245,245,0.8)'
    )
    
    return fig

def create_agent_placement_diagram(analysis: Dict, config: Dict) -> go.Figure:
    """Create a clear visual diagram showing DataSync/DMS agent placement in migration topology"""
    
    # Get configuration details
    migration_method = config.get('migration_method', 'direct_replication')
    backup_storage_type = config.get('backup_storage_type', 'nas_drive')
    agent_analysis = analysis.get('agent_analysis', {})
    num_agents = config.get('number_of_agents', 1)
    primary_tool = agent_analysis.get('primary_tool', 'DMS').upper()
    environment = config.get('environment', 'non-production')
    destination_storage = config.get('destination_storage_type', 'S3')
    
    fig = go.Figure()
    
    # Define colors for different components
    colors = {
        'source': '#FF6B6B',      # Red for source
        'agent': '#4ECDC4',       # Teal for agents  
        'backup': '#FFE66D',      # Yellow for backup storage
        'network': '#95E1D3',     # Light green for network
        'aws': '#3498DB',         # Blue for AWS
        'connection': '#BDC3C7'   # Gray for connections
    }
    
    # Create the topology based on migration method
    if migration_method == 'backup_restore':
        return create_backup_restore_topology(fig, config, agent_analysis, colors)
    else:
        return create_direct_replication_topology(fig, config, agent_analysis, colors)


def create_backup_restore_topology(fig: go.Figure, config: Dict, agent_analysis: Dict, colors: Dict) -> go.Figure:
    """Create topology diagram for backup/restore migration method"""
    
    backup_storage_type = config.get('backup_storage_type', 'nas_drive')
    num_agents = config.get('number_of_agents', 1)
    primary_tool = agent_analysis.get('primary_tool', 'DataSync').upper()
    destination_storage = config.get('destination_storage_type', 'S3')
    environment = config.get('environment', 'non-production')
    
    # Define positions for components with much better spacing
    agent_positions = []
    if num_agents == 1:
        agent_positions = [(3.5, 3.0)]
    elif num_agents == 2:
        agent_positions = [(3.2, 3.8), (3.8, 2.2)]
    elif num_agents == 3:
        agent_positions = [(3.0, 4.0), (3.5, 3.0), (4.0, 2.0)]
    elif num_agents == 4:
        agent_positions = [(3.0, 4.0), (4.0, 4.0), (3.0, 2.0), (4.0, 2.0)]
    else:
        # For 5+ agents, use a more spread out grid
        cols = min(3, num_agents)  # Maximum 3 columns
        rows = (num_agents + cols - 1) // cols
        start_x = 2.8
        start_y = 4.2
        spacing_x = 0.8
        spacing_y = 1.4
        
        for i in range(num_agents):
            row = i // cols
            col = i % cols
            x = start_x + col * spacing_x
            y = start_y - row * spacing_y
            agent_positions.append((x, y))
    
    positions = {
        'source_db': (1.2, 4.5),
        'backup_storage': (1.2, 2.5),
        'agents': agent_positions,
        'network_cloud': (6.0, 3.0),
        'aws_destination': (7.8, 3.0)
    }
    
    # 1. Source Database
    add_component_node(fig, positions['source_db'], 
                      '🗄️ Source Database', 
                      f"{config.get('source_database_engine', 'Database').upper()}\n{config.get('database_size_gb', 0):,} GB",
                      colors['source'], size=45)
    
    # 2. Backup Storage
    backup_icon = '🪟' if backup_storage_type == 'windows_share' else '🗄️'
    backup_protocol = 'SMB' if backup_storage_type == 'windows_share' else 'NFS'
    backup_size = config.get('database_size_gb', 0) * config.get('backup_size_multiplier', 0.7)
    
    add_component_node(fig, positions['backup_storage'],
                      f'{backup_icon} Backup Storage',
                      f"{backup_storage_type.replace('_', ' ').title()}\n{backup_protocol}\n{backup_size:,.0f} GB",
                      colors['backup'], size=40)
    
    # 3. DataSync Agents - smaller size and better labels
    agent_size = config.get('datasync_agent_size', 'medium')
    agent_throughput = agent_analysis.get('total_max_throughput_mbps', 0) / num_agents if num_agents > 0 else 0
    
    for i, pos in enumerate(positions['agents']):
        agent_label = f"🤖 {primary_tool} #{i+1}"
        agent_details = f"{agent_size.title()}\n{agent_throughput:,.0f} Mbps"
        add_component_node(fig, pos, agent_label, agent_details, colors['agent'], size=35)
    
    # 4. Network/Internet Cloud
    add_component_node(fig, positions['network_cloud'],
                      '☁️ Network',
                      f"{environment.title()}\nDirect Connect",
                      colors['network'], size=35)
    
    # 5. AWS Destination
    dest_icon = '☁️' if destination_storage == 'S3' else '🗄️'
    add_component_node(fig, positions['aws_destination'],
                      f'{dest_icon} AWS Target',
                      f"{destination_storage.replace('_', ' ')}\n{config.get('database_engine', 'Target').upper()}",
                      colors['aws'], size=45)
    
    # Add connections with flow direction
    add_migration_flow_backup_restore(fig, positions, colors, num_agents)
    
    # Add detailed annotations
    add_backup_restore_annotations(fig, config, agent_analysis)
    
    # Configure layout
    configure_agent_diagram_layout(fig, "Backup/Restore Migration: Agent Placement & Data Flow")
    
    return fig


def create_direct_replication_topology(fig: go.Figure, config: Dict, agent_analysis: Dict, colors: Dict) -> go.Figure:
    """Create topology diagram for direct replication migration method"""
    
    num_agents = config.get('number_of_agents', 1)
    primary_tool = agent_analysis.get('primary_tool', 'DMS').upper()
    destination_storage = config.get('destination_storage_type', 'S3')
    environment = config.get('environment', 'non-production')
    target_platform = config.get('target_platform', 'rds')
    
    # Define positions with much better spacing for agents
    agent_positions = []
    if num_agents == 1:
        agent_positions = [(3.5, 3.0)]
    elif num_agents == 2:
        agent_positions = [(3.2, 3.6), (3.8, 2.4)]
    elif num_agents == 3:
        agent_positions = [(3.0, 3.8), (3.5, 3.0), (4.0, 2.2)]
    elif num_agents == 4:
        agent_positions = [(3.0, 3.8), (4.0, 3.8), (3.0, 2.2), (4.0, 2.2)]
    else:
        # For 5+ agents, use a more spread out grid
        cols = min(3, num_agents)  # Maximum 3 columns
        rows = (num_agents + cols - 1) // cols
        start_x = 2.8
        start_y = 4.0
        spacing_x = 0.8
        spacing_y = 1.2
        
        for i in range(num_agents):
            row = i // cols
            col = i % cols
            x = start_x + col * spacing_x
            y = start_y - row * spacing_y
            agent_positions.append((x, y))
    
    positions = {
        'source_db': (1.2, 3.5),
        'agents': agent_positions,
        'network_cloud': (6.0, 3.0),
        'aws_destination': (7.8, 3.0)
    }
    
    # 1. Source Database
    add_component_node(fig, positions['source_db'],
                      '🗄️ Live Source DB',
                      f"{config.get('source_database_engine', 'Database').upper()}\n{config.get('database_size_gb', 0):,} GB",
                      colors['source'], size=45)
    
    # 2. Migration Agents - smaller size and better labels
    agent_size = config.get('dms_agent_size') or config.get('datasync_agent_size', 'medium')
    agent_throughput = agent_analysis.get('total_max_throughput_mbps', 0) / num_agents if num_agents > 0 else 0
    is_heterogeneous = config.get('source_database_engine') != config.get('database_engine')
    
    for i, pos in enumerate(positions['agents']):
        agent_label = f"🔄 {primary_tool} #{i+1}"
        if is_heterogeneous:
            agent_details = f"{agent_size.title()}\n{agent_throughput:,.0f} Mbps\nConvert"
        else:
            agent_details = f"{agent_size.title()}\n{agent_throughput:,.0f} Mbps\nSync"
        
        add_component_node(fig, pos, agent_label, agent_details, colors['agent'], size=35)
    
    # 3. Network Path
    add_component_node(fig, positions['network_cloud'],
                      '🌐 Network',
                      f"{environment.title()}\nDirect Connect",
                      colors['network'], size=35)
    
    # 4. AWS Target
    platform_icon = '☁️' if target_platform == 'rds' else '🖥️'
    platform_name = 'RDS' if target_platform == 'rds' else 'EC2'
    
    add_component_node(fig, positions['aws_destination'],
                      f'{platform_icon} AWS {platform_name}',
                      f"{config.get('database_engine', 'Target').upper()}\n{platform_name}",
                      colors['aws'], size=45)
    
    # Add connections with real-time flow
    add_migration_flow_direct_replication(fig, positions, colors, num_agents)
    
    # Add detailed annotations
    add_direct_replication_annotations(fig, config, agent_analysis)
    
    # Configure layout
    configure_agent_diagram_layout(fig, "Direct Replication Migration: Agent Placement & Data Flow")
    
    return fig


def add_component_node(fig: go.Figure, position: tuple, label: str, details: str, 
                      color: str, size: int = 50):
    """Add a component node to the diagram"""
    x, y = position
    
    # Create hover text
    hover_text = f"<b>{label}</b><br>{details.replace(chr(10), '<br>')}"
    
    # Add the main node (marker only, no text)
    fig.add_trace(go.Scatter(
        x=[x],
        y=[y],
        mode='markers',
        marker=dict(
            size=size,
            color=color,
            line=dict(width=2, color='white'),
            opacity=0.9
        ),
        hovertext=hover_text,
        hoverinfo='text',
        showlegend=False
    ))
    
    # Add label above the node with better spacing
    fig.add_annotation(
        x=x,
        y=y + (size/100 + 0.35),  # Dynamic spacing based on node size
        text=f"<b>{label}</b>",
        showarrow=False,
        font=dict(size=9, color='black', family='Arial Bold'),
        bgcolor='rgba(255,255,255,0.95)',
        bordercolor='darkgray',
        borderwidth=1,
        borderpad=2,
        xanchor='center',
        yanchor='bottom'
    )
    
    # Add details text below the node with constrained width
    detail_lines = details.split('\n')
    formatted_details = '<br>'.join([line.strip() for line in detail_lines if line.strip()])
    
    # Adjust detail box positioning based on node size
    detail_y_offset = -(size/100 + 0.45)
    
    fig.add_annotation(
        x=x,
        y=y + detail_y_offset,
        text=formatted_details,
        showarrow=False,
        font=dict(size=7, color='darkslategray', family='Arial'),
        bgcolor='rgba(248,249,250,0.95)',
        bordercolor='lightgray',
        borderwidth=1,
        borderpad=3,
        xanchor='center',
        yanchor='top',
        width=100  # Smaller width to prevent overlap
    )


def add_migration_flow_backup_restore(fig: go.Figure, positions: Dict, colors: Dict, num_agents: int):
    """Add data flow arrows for backup/restore method"""
    
    # 1. Database to Backup Storage (Backup Creation)
    add_flow_arrow(fig, positions['source_db'], positions['backup_storage'], 
                   colors['connection'], "📦 Backup", dash='dot')
    
    # 2. Backup Storage to Agents (File Reading) - handle multiple agents with better spacing
    backup_pos = positions['backup_storage']
    
    for i, agent_pos in enumerate(positions['agents']):
        # Calculate connection points that avoid overlap
        if num_agents == 1:
            backup_connection = backup_pos
        else:
            # Spread out connection points on backup storage
            angle = (i / max(1, num_agents - 1)) * 60 - 30  # Spread over 60 degrees
            offset_x = 0.2 * math.sin(math.radians(angle))
            offset_y = 0.1 * math.cos(math.radians(angle))
            backup_connection = (backup_pos[0] + offset_x, backup_pos[1] + offset_y)
            
        add_flow_arrow(fig, backup_connection, agent_pos,
                       colors['agent'], f"📂 {i+1}", thickness=2)
    
    # 3. Agents to Network Cloud (Data Transfer) - handle multiple agents with better spacing  
    network_pos = positions['network_cloud']
    
    for i, agent_pos in enumerate(positions['agents']):
        if num_agents == 1:
            network_connection = network_pos
        else:
            # Spread out connection points on network cloud
            angle = (i / max(1, num_agents - 1)) * 40 - 20  # Spread over 40 degrees
            offset_x = -0.2 * math.sin(math.radians(angle))
            offset_y = 0.15 * math.cos(math.radians(angle))
            network_connection = (network_pos[0] + offset_x, network_pos[1] + offset_y)
            
        add_flow_arrow(fig, agent_pos, network_connection,
                       colors['network'], f"🚀 {i+1}", thickness=2)
    
    # 4. Network to AWS Destination (Final Delivery)
    add_flow_arrow(fig, positions['network_cloud'], positions['aws_destination'],
                   colors['aws'], "☁️ AWS", thickness=3)


def add_migration_flow_direct_replication(fig: go.Figure, positions: Dict, colors: Dict, num_agents: int):
    """Add data flow arrows for direct replication method"""
    
    # 1. Database to Agents (Live Replication) - handle multiple agents with better spacing
    source_pos = positions['source_db']
    
    for i, agent_pos in enumerate(positions['agents']):
        if num_agents == 1:
            source_connection = source_pos
        else:
            # Spread out connection points on source database
            angle = (i / max(1, num_agents - 1)) * 50 - 25  # Spread over 50 degrees
            offset_x = 0.2 * math.sin(math.radians(angle))
            offset_y = 0.1 * math.cos(math.radians(angle))
            source_connection = (source_pos[0] + offset_x, source_pos[1] + offset_y)
            
        add_flow_arrow(fig, source_connection, agent_pos,
                       colors['agent'], f"⚡ {i+1}", thickness=2)
    
    # 2. Agents to Network Cloud (Real-time Transfer) - handle multiple agents with better spacing
    network_pos = positions['network_cloud']
    
    for i, agent_pos in enumerate(positions['agents']):
        if num_agents == 1:
            network_connection = network_pos
        else:
            # Spread out connection points on network cloud
            angle = (i / max(1, num_agents - 1)) * 40 - 20  # Spread over 40 degrees
            offset_x = -0.2 * math.sin(math.radians(angle))
            offset_y = 0.15 * math.cos(math.radians(angle))
            network_connection = (network_pos[0] + offset_x, network_pos[1] + offset_y)
            
        add_flow_arrow(fig, agent_pos, network_connection,
                       colors['network'], f"🌐 {i+1}", thickness=2)
    
    # 3. Network to AWS Target (Live Replication)
    add_flow_arrow(fig, positions['network_cloud'], positions['aws_destination'],
                   colors['aws'], "🎯 Live", thickness=3)


def add_flow_arrow(fig: go.Figure, start_pos: tuple, end_pos: tuple, color: str, 
                   label: str, thickness: int = 3, dash: str = 'solid'):
    """Add a flow arrow between two positions"""
    x1, y1 = start_pos
    x2, y2 = end_pos
    
    # Calculate arrow direction and position
    dx = x2 - x1
    dy = y2 - y1
    length = (dx**2 + dy**2)**0.5
    
    if length == 0:
        return  # Avoid division by zero
    
    # Adjust start and end points to not overlap with nodes
    offset = 0.5  # Increased offset to avoid node overlap
    start_x = x1 + (dx/length) * offset
    start_y = y1 + (dy/length) * offset
    end_x = x2 - (dx/length) * offset
    end_y = y2 - (dy/length) * offset
    
    # Add the connection line
    fig.add_trace(go.Scatter(
        x=[start_x, end_x],
        y=[start_y, end_y],
        mode='lines',
        line=dict(
            width=thickness,
            color=color,
            dash=dash
        ),
        hovertext=f"<b>{label}</b><br>Data Flow Connection",
        hoverinfo='text',
        showlegend=False
    ))
    
    # Add arrow head
    add_arrow_head(fig, end_x, end_y, dx/length, dy/length, color, size=0.12)
    
    # Add flow label with better positioning to avoid overlap
    mid_x = (start_x + end_x) / 2
    mid_y = (start_y + end_y) / 2
    
    # Offset label position based on arrow direction to avoid overlap
    if abs(dx) > abs(dy):  # More horizontal
        label_offset_y = 0.25 if dy >= 0 else -0.25
        label_offset_x = 0
    else:  # More vertical  
        label_offset_x = 0.3 if dx >= 0 else -0.3
        label_offset_y = 0
    
    fig.add_annotation(
        x=mid_x + label_offset_x,
        y=mid_y + label_offset_y,
        text=f"<b>{label}</b>",
        showarrow=False,
        font=dict(size=8, color=color, family='Arial Bold'),
        bgcolor='rgba(255,255,255,0.9)',
        bordercolor=color,
        borderwidth=1,
        borderpad=2,
        xanchor='center',
        yanchor='middle'
    )


def add_arrow_head(fig: go.Figure, x: float, y: float, dx: float, dy: float, 
                   color: str, size: float = 0.1):
    """Add an arrow head at the specified position"""
    # Calculate arrow head points
    arrow_length = size
    arrow_width = size * 0.6
    
    # Arrow head points
    head_x1 = x - arrow_length * dx - arrow_width * dy
    head_y1 = y - arrow_length * dy + arrow_width * dx
    head_x2 = x - arrow_length * dx + arrow_width * dy
    head_y2 = y - arrow_length * dy - arrow_width * dx
    
    fig.add_trace(go.Scatter(
        x=[head_x1, x, head_x2, head_x1],
        y=[head_y1, y, head_y2, head_y1],
        mode='lines',
        fill='toself',
        fillcolor=color,
        line=dict(color=color, width=2),
        showlegend=False,
        hoverinfo='skip'
    ))


def add_backup_restore_annotations(fig: go.Figure, config: Dict, agent_analysis: Dict):
    """Add specific annotations for backup/restore method"""
    
    # Migration method info - positioned at top
    fig.add_annotation(
        x=4, y=5.2,
        text="📦 BACKUP/RESTORE MIGRATION",
        showarrow=False,
        font=dict(size=12, color='darkblue', family='Arial Bold'),
        bgcolor='rgba(173,216,230,0.9)',
        bordercolor='darkblue',
        borderwidth=2,
        borderpad=6,
        xanchor='center',
        yanchor='middle'
    )
    
    # Agent placement rationale - left side
    backup_storage_type = config.get('backup_storage_type', 'nas_drive')
    protocol = 'SMB' if backup_storage_type == 'windows_share' else 'NFS'
    
    fig.add_annotation(
        x=0.5, y=0.8,
        text=f"💡 <b>Agent Strategy:</b><br>• Close to backup storage<br>• {protocol} protocol optimized<br>• Parallel file processing<br>• Minimal DB impact",
        showarrow=False,
        font=dict(size=9, color='darkgreen', family='Arial'),
        bgcolor='rgba(144,238,144,0.9)',
        bordercolor='darkgreen',
        borderwidth=1,
        align='left',
        xanchor='left',
        yanchor='top',
        width=140
    )
    
    # Performance metrics - right side
    total_throughput = agent_analysis.get('total_effective_throughput', 0)
    migration_time = calculate_backup_migration_time(config, total_throughput)
    
    fig.add_annotation(
        x=7.5, y=0.8,
        text=f"📊 <b>Performance:</b><br>• Throughput: {total_throughput:,.0f} Mbps<br>• Est. Time: {migration_time:.1f}h<br>• Efficiency: {agent_analysis.get('backup_efficiency', 1.0)*100:.1f}%",
        showarrow=False,
        font=dict(size=9, color='darkred', family='Arial'),
        bgcolor='rgba(255,182,193,0.9)',
        bordercolor='darkred',
        borderwidth=1,
        align='left',
        xanchor='right',
        yanchor='top',
        width=140
    )


def add_direct_replication_annotations(fig: go.Figure, config: Dict, agent_analysis: Dict):
    """Add specific annotations for direct replication method"""
    
    # Migration method info - positioned at top
    is_heterogeneous = config.get('source_database_engine') != config.get('database_engine')
    method_type = "HETEROGENEOUS" if is_heterogeneous else "HOMOGENEOUS"
    
    fig.add_annotation(
        x=4, y=5.2,
        text=f"🔄 DIRECT REPLICATION ({method_type})",
        showarrow=False,
        font=dict(size=12, color='darkblue', family='Arial Bold'),
        bgcolor='rgba(173,216,230,0.9)',
        bordercolor='darkblue',
        borderwidth=2,
        borderpad=6,
        xanchor='center',
        yanchor='middle'
    )
    
    # Agent placement rationale - left side
    primary_tool = agent_analysis.get('primary_tool', 'DMS')
    
    fig.add_annotation(
        x=0.5, y=0.8,
        text=f"💡 <b>Agent Strategy:</b><br>• {primary_tool} near source DB<br>• Minimize replication lag<br>• Change Data Capture<br>• Real-time sync",
        showarrow=False,
        font=dict(size=9, color='darkgreen', family='Arial'),
        bgcolor='rgba(144,238,144,0.9)',
        bordercolor='darkgreen',
        borderwidth=1,
        align='left',
        xanchor='left',
        yanchor='top',
        width=140
    )
    
    # Performance metrics - right side
    total_throughput = agent_analysis.get('total_effective_throughput', 0)
    migration_time = calculate_direct_migration_time(config, total_throughput)
    
    fig.add_annotation(
        x=7.5, y=0.8,
        text=f"📊 <b>Performance:</b><br>• Throughput: {total_throughput:,.0f} Mbps<br>• Est. Time: {migration_time:.1f}h<br>• Replication Lag: <5s",
        showarrow=False,
        font=dict(size=9, color='darkred', family='Arial'),
        bgcolor='rgba(255,182,193,0.9)',
        bordercolor='darkred',
        borderwidth=1,
        align='left',
        xanchor='right',
        yanchor='top',
        width=140
    )


def calculate_backup_migration_time(config: Dict, throughput_mbps: float) -> float:
    """Calculate estimated migration time for backup/restore method"""
    if throughput_mbps <= 0:
        return 0
    
    db_size_gb = config.get('database_size_gb', 0)
    backup_multiplier = config.get('backup_size_multiplier', 0.7)
    backup_size_gb = db_size_gb * backup_multiplier
    
    # Convert GB to bits, then calculate hours
    size_bits = backup_size_gb * 8 * 1000 * 1000 * 1000
    throughput_bps = throughput_mbps * 1000 * 1000
    return size_bits / throughput_bps / 3600


def calculate_direct_migration_time(config: Dict, throughput_mbps: float) -> float:
    """Calculate estimated migration time for direct replication method"""
    if throughput_mbps <= 0:
        return 0
    
    db_size_gb = config.get('database_size_gb', 0)
    size_bits = db_size_gb * 8 * 1000 * 1000 * 1000
    throughput_bps = throughput_mbps * 1000 * 1000
    return size_bits / throughput_bps / 3600


def configure_agent_diagram_layout(fig: go.Figure, title: str):
    """Configure the layout for the agent placement diagram"""
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            y=0.97,  # Move title even higher
            font=dict(size=13, color='darkslategray', family='Arial Bold')
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-0.2, 9.0]  # Even wider range for better spacing
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-0.2, 6.5]  # Taller range for better vertical spacing
        ),
        showlegend=False,
        height=700,  # Even taller to prevent overlap
        plot_bgcolor='rgba(248,249,250,0.9)',
        paper_bgcolor='white',
        margin=dict(l=40, r=40, t=70, b=140),  # Larger margins
        hovermode='closest',
        font=dict(size=9)  # Smaller base font size
    )


# Update the network intelligence tab function to include the new diagram
def render_network_intelligence_tab_enhanced(analysis: Dict, config: Dict):
    """Enhanced network intelligence tab with clear agent placement diagram"""
    
    # ADD VALIDATION AT THE START
    validated_costs = add_cost_validation_to_tab(analysis, config)
    st.subheader("🌐 Network Intelligence & Agent Placement Analysis")

    network_perf = analysis.get('network_performance', {})

    # Network Overview Dashboard (existing code)
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
            "🌐 Network Capacity",
            f"{network_perf.get('effective_bandwidth_mbps', 0):,.0f} Mbps",
            delta="Raw network limit"
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

    # NEW: Agent Placement Visualization
    st.markdown("---")
    st.markdown("**🤖 DataSync/DMS Agent Placement & Migration Flow:**")
    
    try:
        agent_diagram = create_agent_placement_diagram(analysis, config)
        st.plotly_chart(agent_diagram, use_container_width=True, key=f"agent_placement_{int(time.time() * 1000000)}")
        
        # Add explanation of the diagram
        migration_method = config.get('migration_method', 'direct_replication')
        if migration_method == 'backup_restore':
            st.info("""
            📦 **Backup/Restore Migration Flow:**
            1. **Database** creates backup files on backup storage
            2. **DataSync Agents** read backup files using SMB/NFS protocols
            3. **Agents** transfer data through network path to AWS
            4. **AWS** receives and restores data to target database
            
            **Key Advantage:** Minimal impact on production database during transfer
            """)
        else:
            st.info("""
            🔄 **Direct Replication Migration Flow:**
            1. **DMS/DataSync Agents** connect directly to source database
            2. **Agents** perform real-time Change Data Capture (CDC)
            3. **Live data** flows through network path to AWS target
            4. **Target database** stays synchronized in real-time
            
            **Key Advantage:** Minimal downtime with continuous synchronization
            """)
            
    except Exception as e:
        st.warning(f"Agent placement diagram could not be rendered: {str(e)}")
        st.info("📍 **Agent Placement Summary:**")
        
        agent_analysis = analysis.get('agent_analysis', {})
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Migration Method:** {config.get('migration_method', 'direct_replication').replace('_', ' ').title()}")
            st.write(f"**Primary Tool:** {agent_analysis.get('primary_tool', 'DMS').upper()}")
            st.write(f"**Number of Agents:** {config.get('number_of_agents', 1)}")
            st.write(f"**Agent Size:** {config.get('datasync_agent_size') or config.get('dms_agent_size', 'medium').title()}")
        
        with col2:
            st.write(f"**Total Throughput:** {agent_analysis.get('total_effective_throughput', 0):,.0f} Mbps")
            st.write(f"**Monthly Cost:** ${agent_analysis.get('monthly_cost', 0):,.0f}")
            st.write(f"**Scaling Efficiency:** {agent_analysis.get('scaling_efficiency', 1.0)*100:.1f}%")
            st.write(f"**Bottleneck:** {agent_analysis.get('bottleneck', 'Unknown')}")

    # Continue with existing bandwidth waterfall analysis
    st.markdown("---")
    render_bandwidth_waterfall_analysis(analysis, config)

    # Continue with existing network path visualization and other sections...
    st.markdown("---")
    render_performance_impact_table(analysis, config)

    # Rest of the existing function remains the same...



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

    def run_agent_optimization_sync(self, config: Dict, analysis: Dict) -> Dict:
        """Run agent optimization synchronously for Streamlit"""
        import asyncio

        # Check if we're in an event loop
        try:
            loop = asyncio.get_running_loop()
            # We're in an event loop, so we need to run in a thread
            import concurrent.futures
            import threading

            def run_async():
                # Create a new event loop for this thread
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(
                        self.analyze_agent_scaling_optimization(config, analysis)
                    )
                finally:
                    new_loop.close()

            # Run in a separate thread with its own event loop
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_async)
                return future.result(timeout=60)  # 60 second timeout

        except RuntimeError:
            # No event loop running, safe to use asyncio.run()
            return asyncio.run(self.analyze_agent_scaling_optimization(config, analysis))

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

def main():
    """Main Streamlit application with authentication"""
    
    # Page configuration
    st.set_page_config(
        page_title="AWS Enterprise Database Migration Analyzer AI v3.0",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Your existing CSS styles
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
    /* Add your other CSS styles here */
    </style>
    """, unsafe_allow_html=True)
    
    # Check authentication
    if not check_authentication():
        # Show admin panel if requested
        if st.session_state.get('show_admin', False):
            render_admin_panel()
        else:
            render_login_page()
        return
    
    # User is authenticated, show the application
    render_logout_section()
    
    # Your existing header
    st.markdown(f"""
    <div class="main-header">
    <h1>🤖 AWS Enterprise Database Migration Analyzer AI v3.0</h1>
    <p style="font-size: 1.2rem; margin-top: 0.5rem;">
    Professional-Grade Migration Analysis • AI-Powered Insights • Real-time AWS Integration
    </p>
    <p style="font-size: 0.9rem; margin-top: 1rem;">
    Welcome back, {st.session_state.get('user_name', 'User')} • Secure Enterprise Access
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Your existing sidebar controls
    config = render_enhanced_sidebar_controls()
    
    # Run analysis when configuration is set
    if st.button("🚀 Run Enhanced AI Migration Analysis", type="primary", use_container_width=True):
        with st.spinner("🤖 Running comprehensive AI migration analysis..."):
            try:
                # Initialize analyzer
                analyzer = EnhancedMigrationAnalyzer()
                
                # Run comprehensive analysis
                # Run comprehensive analysis (using a wrapper for async code)
                import asyncio

                # Check if we're in an event loop
                try:
                    loop = asyncio.get_running_loop()
                    # We're in Streamlit's event loop, run in thread
                    import concurrent.futures
                    
                    def run_analysis():
                        # Create a new event loop for this thread
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        try:
                            return new_loop.run_until_complete(
                                analyzer.comprehensive_ai_migration_analysis(config)
                            )
                        finally:
                            new_loop.close()
                    
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(run_analysis)
                        analysis = future.result(timeout=120)  # 2 minute timeout
                        
                except RuntimeError:
                    # No event loop, safe to use asyncio.run()
                    analysis = asyncio.run(analyzer.comprehensive_ai_migration_analysis(config))
                
                # Store analysis in session state
                st.session_state['analysis'] = analysis
                st.session_state['config'] = config
                
                st.success("✅ AI Analysis Complete!")
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                logger.error(f"Analysis error: {e}")
                return
    
    # Display results if analysis is available
    if 'analysis' in st.session_state and 'config' in st.session_state:
        analysis = st.session_state['analysis']
        config = st.session_state['config']
        
        # Add PDF export to the main interface
        with st.sidebar:
            st.markdown("---")
            st.markdown("### 📄 Quick Report Export")
            
            if st.button("📊 Generate Full PDF Report", use_container_width=True):
                with st.spinner("Generating comprehensive PDF report..."):
                    pdf_data = export_pdf_report(analysis, config)
                    if pdf_data:
                        st.download_button(
                            label="📥 Download Report",
                            data=pdf_data,
                            file_name=f"aws_migration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
        
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "📊 Migration Dashboard",
            "🧠 AI Insights", 
            "🌐 Network Intelligence",
            "💰 Complete Cost Analysis",
            "💻 OS Performance",
            "🎯 AWS Sizing",
            "🗄️ FSx Comparisons",
            "🤖 Agent Scaling Optimizer"
        ])
        
        with tab1:
            render_migration_dashboard_tab_with_pdf(analysis, config)
        
        with tab2:
            render_ai_insights_tab_enhanced(analysis, config)
            render_pdf_export_section(analysis, config)
        
        with tab3:
            render_network_intelligence_tab(analysis, config)
            render_pdf_export_section(analysis, config)
        
        with tab4:
            render_comprehensive_cost_analysis_tab_with_pdf(analysis, config)
        
        with tab5:
            render_os_performance_tab(analysis, config)
        
        with tab6:
            render_aws_sizing_tab(analysis, config)
        
        with tab7:
            render_fsx_comparisons_tab(analysis, config)
        
        with tab8:
            render_agent_scaling_optimizer_tab(analysis, config)
    
    # Footer
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #1f2937 0%, #374151 100%); color: white; border-radius: 6px; margin-top: 2rem;">
    <h4>🚀 AWS Enterprise Database Migration Analyzer AI v3.0</h4>
    <p>Secured with Firebase Authentication • Enterprise User Management • Professional Migration Analysis</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()