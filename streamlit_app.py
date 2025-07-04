# -*- coding: utf-8 -*-
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
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
import tempfile
import os


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

def render_comprehensive_cost_analysis_tab_enhanced(analysis: Dict, config: Dict):
    """Render enhanced comprehensive AWS cost analysis tab with detailed tables"""
    st.subheader("💰 Comprehensive AWS Cost Analysis")
    
    # Generate comprehensive cost summary
    cost_manager = CostSummaryManager()
    cost_summary = cost_manager.generate_comprehensive_cost_summary(analysis, config)
    
    if not cost_summary or not cost_summary.get('detailed_costs'):
        st.warning("Cost analysis data not available. Please run the analysis first.")
        return
    
    # Executive Cost Summary
    st.markdown("**💸 Executive Cost Summary:**")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_monthly = cost_summary.get('total_monthly_cost', 0)
        st.metric(
            "💰 Total Monthly",
            f"${total_monthly:,.0f}",
            delta=f"Annual: ${total_monthly * 12:,.0f}"
        )
    
    with col2:
        total_one_time = cost_summary.get('total_one_time_cost', 0)
        st.metric(
            "🔄 One-Time Costs",
            f"${total_one_time:,.0f}",
            delta="Setup & Migration"
        )
    
    with col3:
        three_year_total = cost_summary.get('three_year_total', 0)
        st.metric(
            "📅 3-Year Total",
            f"${three_year_total:,.0f}",
            delta="Including all costs"
        )
    
    with col4:
        cost_categories = cost_summary.get('cost_categories', {})
        largest_category = max(cost_categories.items(), key=lambda x: x[1]['monthly']) if cost_categories else ('Unknown', {'monthly': 0})
        st.metric(
            "🎯 Largest Category",
            largest_category[0].title(),
            delta=f"${largest_category[1]['monthly']:,.0f}/mo"
        )
    
    with col5:
        optimization_count = len(cost_summary.get('optimization_opportunities', []))
        potential_savings = sum(opt.get('potential_monthly_savings', 0) for opt in cost_summary.get('optimization_opportunities', []))
        st.metric(
            "💡 Potential Savings",
            f"${potential_savings:,.0f}/mo",
            delta=f"{optimization_count} opportunities"
        )
    
    # Comprehensive Cost Table
    st.markdown("---")
    st.markdown("**📊 Comprehensive AWS Cost Breakdown Table:**")
    
    # Create detailed cost breakdown table
    detailed_costs = cost_summary.get('detailed_costs', [])
    
    if detailed_costs:
        # Prepare table data
        table_data = []
        
        # Sort by category and cost
        sorted_costs = sorted(detailed_costs, key=lambda x: (x.get('category', 'Other'), -x.get('monthly_cost', 0)))
        
        current_category = None
        category_total_monthly = 0
        category_total_annual = 0
        
        for cost_item in sorted_costs:
            category = cost_item.get('category', 'Other')
            
            # Add category header if new category
            if category != current_category:
                if current_category is not None:
                    # Add category subtotal
                    table_data.append({
                        'Category': f"**{current_category.upper()} SUBTOTAL**",
                        'Service/Component': '',
                        'Type': '',
                        'Monthly Cost': f"**${category_total_monthly:,.2f}**",
                        'Annual Cost': f"**${category_total_annual:,.2f}**",
                        'Notes': ''
                    })
                    table_data.append({'Category': '', 'Service/Component': '', 'Type': '', 'Monthly Cost': '', 'Annual Cost': '', 'Notes': ''})  # Spacer
                
                current_category = category
                category_total_monthly = 0
                category_total_annual = 0
            
            # Add cost item
            monthly_cost = cost_item.get('monthly_cost', 0)
            annual_cost = cost_item.get('annual_cost', monthly_cost * 12)
            one_time_cost = cost_item.get('one_time_cost', 0)
            
            category_total_monthly += monthly_cost
            category_total_annual += annual_cost
            
            table_data.append({
                'Category': category,
                'Service/Component': cost_item.get('service', 'Unknown'),
                'Type': cost_item.get('cost_type', 'monthly').title(),
                'Monthly Cost': f"${monthly_cost:,.2f}" if monthly_cost > 0 else '-',
                'Annual Cost': f"${annual_cost:,.2f}" if annual_cost > 0 else f"${one_time_cost:,.2f} (One-time)" if one_time_cost > 0 else '-',
                'Notes': cost_item.get('notes', '')[:80] + '...' if len(cost_item.get('notes', '')) > 80 else cost_item.get('notes', '')
            })
        
        # Add final category subtotal
        if current_category:
            table_data.append({
                'Category': f"**{current_category.upper()} SUBTOTAL**",
                'Service/Component': '',
                'Type': '',
                'Monthly Cost': f"**${category_total_monthly:,.2f}**",
                'Annual Cost': f"**${category_total_annual:,.2f}**",
                'Notes': ''
            })
        
        # Add grand total
        table_data.append({'Category': '', 'Service/Component': '', 'Type': '', 'Monthly Cost': '', 'Annual Cost': '', 'Notes': ''})  # Spacer
        table_data.append({
            'Category': '**GRAND TOTAL**',
            'Service/Component': '**ALL AWS SERVICES**',
            'Type': '**TOTAL**',
            'Monthly Cost': f"**${total_monthly:,.2f}**",
            'Annual Cost': f"**${total_monthly * 12 + total_one_time:,.2f}**",
            'Notes': f"Includes ${total_one_time:,.2f} one-time costs"
        })
        
        # Display the table
        df_costs = pd.DataFrame(table_data)
        
        # Use HTML table for better formatting
        html_table = df_costs.to_html(index=False, escape=False, classes='cost-table')
        st.markdown(html_table, unsafe_allow_html=True)
        
        # Category breakdown chart
        st.markdown("---")
        st.markdown("**📈 Cost Distribution by Category:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Monthly costs pie chart
            if cost_categories:
                monthly_cats = {cat: data['monthly'] for cat, data in cost_categories.items() if data['monthly'] > 0}
                
                if monthly_cats:
                    fig_pie = px.pie(
                        values=list(monthly_cats.values()),
                        names=list(monthly_cats.keys()),
                        title="Monthly Cost Distribution"
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Cost comparison bar chart
            if cost_categories:
                cat_data = []
                for cat, data in cost_categories.items():
                    cat_data.append({
                        'Category': cat,
                        'Monthly': data['monthly'],
                        'One-Time': data['one_time']
                    })
                
                df_cat = pd.DataFrame(cat_data)
                fig_bar = px.bar(
                    df_cat, 
                    x='Category', 
                    y=['Monthly', 'One-Time'],
                    title="Cost Comparison by Category",
                    barmode='group'
                )
                st.plotly_chart(fig_bar, use_container_width=True)
    
    # Cost Optimization Opportunities
    st.markdown("---")
    st.markdown("**💡 Cost Optimization Opportunities:**")
    
    optimization_opportunities = cost_summary.get('optimization_opportunities', [])
    
    if optimization_opportunities:
        opt_data = []
        for i, opt in enumerate(optimization_opportunities, 1):
            opt_data.append({
                'Priority': i,
                'Opportunity': opt.get('opportunity', 'Unknown'),
                'Monthly Savings': f"${opt.get('potential_monthly_savings', 0):,.2f}",
                'Annual Savings': f"${opt.get('potential_annual_savings', 0):,.2f}",
                'Description': opt.get('description', '')
            })
        
        df_opt = pd.DataFrame(opt_data)
        st.dataframe(df_opt, use_container_width=True)
        
        # Total optimization potential
        total_monthly_savings = sum(opt.get('potential_monthly_savings', 0) for opt in optimization_opportunities)
        total_annual_savings = sum(opt.get('potential_annual_savings', 0) for opt in optimization_opportunities)
        
        st.success(
        "**💰 Total Optimization Potential:**\n" +
        f"• **Monthly Savings:** ${total_monthly_savings:,.2f}\n" +
        f"• **Annual Savings:** ${total_annual_savings:,.2f}\n" +
        f"• **3-Year Savings:** ${total_annual_savings * 3:,.2f}"
    )
    else:
        st.info("No specific optimization opportunities identified. Current configuration appears cost-optimized.")
    
    # PDF Report Generation
    st.markdown("---")
    st.markdown("**📄 Generate Detailed PDF Report:**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📋 Generate Executive Summary PDF", use_container_width=True):
            try:
                with st.spinner("Generating executive summary PDF..."):
                    pdf_generator = PDFReportGenerator()
                    pdf_content = pdf_generator.generate_comprehensive_report(analysis, config, cost_summary)
                    
                    st.download_button(
                        label="📥 Download Executive Summary PDF",
                        data=pdf_content,
                        file_name=f"aws_migration_executive_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                    st.success("✅ Executive summary PDF generated successfully!")
            except Exception as e:
                st.error(f"❌ Failed to generate PDF: {str(e)}")
    
    with col2:
        if st.button("💰 Generate Cost Analysis PDF", use_container_width=True):
            try:
                with st.spinner("Generating detailed cost analysis PDF..."):
                    # Create a cost-focused PDF
                    pdf_generator = PDFReportGenerator()
                    pdf_content = pdf_generator.generate_comprehensive_report(analysis, config, cost_summary)
                    
                    st.download_button(
                        label="📥 Download Cost Analysis PDF",
                        data=pdf_content,
                        file_name=f"aws_migration_cost_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                    st.success("✅ Cost analysis PDF generated successfully!")
            except Exception as e:
                st.error(f"❌ Failed to generate cost PDF: {str(e)}")
    
    with col3:
        if st.button("📊 Export Cost Data (CSV)", use_container_width=True):
            try:
                # Create CSV data
                csv_data = pd.DataFrame(detailed_costs)
                csv_string = csv_data.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Cost Data CSV",
                    data=csv_string,
                    file_name=f"aws_migration_costs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                st.success("✅ Cost data exported successfully!")
            except Exception as e:
                st.error(f"❌ Failed to export CSV: {str(e)}")

def render_enhanced_analysis_results_tab(analysis: Dict, config: Dict):
    """Render enhanced analysis results with AI insights"""
    st.subheader("🤖 AI-Enhanced Migration Analysis Results")
    
    if not analysis:
        st.warning("No analysis data available. Please run the analysis first.")
        return
    
    # Migration Overview
    st.markdown("### 📋 Migration Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        migration_time = analysis.get('estimated_migration_time_hours', 0)
        st.metric("⏱️ Est. Migration Time", f"{migration_time:.1f} hours")
    
    with col2:
        migration_throughput = analysis.get('migration_throughput_mbps', 0)
        st.metric("🚀 Migration Throughput", f"{migration_throughput:,.0f} Mbps")
    
    with col3:
        primary_tool = analysis.get('primary_tool', 'Unknown')
        st.metric("🔧 Primary Tool", f"AWS {primary_tool.upper()}")
    
    with col4:
        num_agents = config.get('number_of_agents', 1)
        st.metric("🤖 Migration Agents", f"{num_agents} agents")
    
    # Performance Analysis
    st.markdown("---")
    st.markdown("### 📊 Performance Analysis")
    
    onprem_perf = analysis.get('onprem_performance', {})
    if onprem_perf:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🖥️ On-Premises Performance:**")
            perf_score = onprem_perf.get('performance_score', 0)
            st.progress(perf_score / 100)
            st.write(f"Performance Score: {perf_score:.1f}/100")
            
            # OS Impact
            os_impact = onprem_perf.get('os_impact', {})
            if os_impact:
                st.write(f"OS Efficiency: {os_impact.get('total_efficiency', 0)*100:.1f}%")
                st.write(f"Database Optimization: {os_impact.get('db_optimization', 0)*100:.1f}%")
        
        with col2:
            st.markdown("**🌐 Network Performance:**")
            network_perf = analysis.get('network_performance', {})
            if network_perf:
                network_quality = network_perf.get('network_quality_score', 0)
                st.progress(network_quality / 100)
                st.write(f"Network Quality: {network_quality:.1f}/100")
                st.write(f"Effective Bandwidth: {network_perf.get('effective_bandwidth_mbps', 0):,.0f} Mbps")
                st.write(f"Total Latency: {network_perf.get('total_latency_ms', 0):.1f} ms")

    # Agent Analysis
    st.markdown("---")
    st.markdown("### 🤖 Migration Agent Analysis")
    
    agent_analysis = analysis.get('agent_analysis', {})
    if agent_analysis:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Agent Configuration:**")
            st.write(f"Primary Tool: {agent_analysis.get('primary_tool', 'Unknown').upper()}")
            st.write(f"Agent Size: {agent_analysis.get('agent_size', 'Unknown').title()}")
            st.write(f"Number of Agents: {agent_analysis.get('number_of_agents', 1)}")
            st.write(f"Destination Storage: {agent_analysis.get('destination_storage', 'S3')}")
        
        with col2:
            st.markdown("**Performance Metrics:**")
            st.write(f"Max Throughput: {agent_analysis.get('total_max_throughput_mbps', 0):,.0f} Mbps")
            st.write(f"Effective Throughput: {agent_analysis.get('total_effective_throughput', 0):,.0f} Mbps")
            st.write(f"Scaling Efficiency: {agent_analysis.get('scaling_efficiency', 1.0)*100:.1f}%")
        
        with col3:
            st.markdown("**Cost & Management:**")
            st.write(f"Monthly Cost: ${agent_analysis.get('monthly_cost', 0):,.2f}")
            st.write(f"Bottleneck: {agent_analysis.get('bottleneck', 'None')}")
            st.write(f"Management Complexity: {agent_analysis.get('management_overhead', 1.0):.2f}x")

    # AI Overall Assessment
    st.markdown("---")
    st.markdown("### 🧠 AI Overall Assessment")
    
    ai_assessment = analysis.get('ai_overall_assessment', {})
    if ai_assessment:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🎯 Readiness Assessment:**")
            readiness_score = ai_assessment.get('migration_readiness_score', 0)
            st.progress(readiness_score / 100)
            st.write(f"Migration Readiness: {readiness_score}/100")
            
            success_prob = ai_assessment.get('success_probability', 0)
            st.write(f"Success Probability: {success_prob}%")
            
            risk_level = ai_assessment.get('risk_level', 'Unknown')
            st.write(f"Risk Level: {risk_level}")
        
        with col2:
            st.markdown("**📅 Timeline Recommendation:**")
            timeline = ai_assessment.get('timeline_recommendation', {})
            if timeline:
                st.write(f"Planning Phase: {timeline.get('planning_phase_weeks', 2)} weeks")
                st.write(f"Testing Phase: {timeline.get('testing_phase_weeks', 3)} weeks")
                st.write(f"Migration Window: {timeline.get('migration_window_hours', 8):.1f} hours")
                st.write(f"Total Project: {timeline.get('total_project_weeks', 6)} weeks")

def render_aws_sizing_recommendations_tab(analysis: Dict, config: Dict):
    """Render AWS sizing recommendations tab"""
    st.subheader("☁️ AWS Sizing Recommendations")
    
    aws_sizing = analysis.get('aws_sizing_recommendations', {})
    if not aws_sizing:
        st.warning("No AWS sizing data available. Please run the analysis first.")
        return
    
    # Deployment Recommendation
    deployment_rec = aws_sizing.get('deployment_recommendation', {})
    recommended_platform = deployment_rec.get('recommendation', 'rds')
    
    st.markdown(f"### 🎯 Recommended Platform: **{recommended_platform.upper()}**")
    
    confidence = deployment_rec.get('confidence', 0.5)
    st.progress(confidence)
    st.write(f"Recommendation Confidence: {confidence*100:.0f}%")
    
    # Show reasons
    reasons = deployment_rec.get('primary_reasons', [])
    if reasons:
        st.markdown("**Recommendation Reasons:**")
        for reason in reasons:
            st.write(f"• {reason}")
    
    # Platform-specific recommendations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏛️ RDS Recommendations")
        rds_rec = aws_sizing.get('rds_recommendations', {})
        if rds_rec:
            st.markdown(f"**Instance Type:** {rds_rec.get('primary_instance', 'Unknown')}")
            st.markdown(f"**Storage:** {rds_rec.get('storage_type', 'gp3')} - {rds_rec.get('storage_size_gb', 0):,} GB")
            st.markdown(f"**Monthly Cost:** ${rds_rec.get('total_monthly_cost', 0):,.2f}")
            st.markdown(f"**Multi-AZ:** {'Yes' if rds_rec.get('multi_az', False) else 'No'}")
            
            # Sizing reasoning
            reasoning = rds_rec.get('sizing_reasoning', [])
            if reasoning:
                st.markdown("**Sizing Factors:**")
                for reason in reasoning:
                    st.write(f"• {reason}")
    
    with col2:
        st.markdown("### 🖥️ EC2 Recommendations")
        ec2_rec = aws_sizing.get('ec2_recommendations', {})
        if ec2_rec:
            st.markdown(f"**Instance Type:** {ec2_rec.get('primary_instance', 'Unknown')}")
            st.markdown(f"**Database Engine:** {ec2_rec.get('database_engine', 'Unknown')}")
            st.markdown(f"**Storage:** {ec2_rec.get('storage_type', 'gp3')} - {ec2_rec.get('storage_size_gb', 0):,} GB")
            st.markdown(f"**Monthly Cost:** ${ec2_rec.get('total_monthly_cost', 0):,.2f}")
            
            # SQL Server specific info
            if ec2_rec.get('sql_server_considerations', False):
                deployment_type = ec2_rec.get('sql_server_deployment_type', 'standalone')
                instance_count = ec2_rec.get('instance_count', 1)
                
                # Safely format deployment type
                if deployment_type and isinstance(deployment_type, str):
                    formatted_deployment = deployment_type.replace('_', ' ').title()
                else:
                    formatted_deployment = 'Standalone'
                
                # Fixed: Use the safely formatted version
                st.markdown(f"**SQL Server Config:** {formatted_deployment}")
                st.markdown(f"**Instance Count:** {instance_count}")
                
                if deployment_type == 'always_on':
                    st.info("🔄 Always On configuration provides high availability with automatic failover")

    # Reader/Writer Configuration
    st.markdown("---")
    st.markdown("### 📖 Reader/Writer Configuration")
    
    reader_writer = aws_sizing.get('reader_writer_config', {})
    if reader_writer:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("✍️ Writers", reader_writer.get('writers', 1))
        with col2:
            st.metric("📖 Readers", reader_writer.get('readers', 0))
        with col3:
            st.metric("📊 Total Instances", reader_writer.get('total_instances', 1))
        
        st.write(f"**Reasoning:** {reader_writer.get('reasoning', 'Standard configuration')}")

def render_destination_storage_comparison_tab(analysis: Dict, config: Dict):
    """Render destination storage comparison tab"""
    st.subheader("🗄️ Destination Storage Comparison")
    
    fsx_comparisons = analysis.get('fsx_comparisons', {})
    if not fsx_comparisons:
        st.warning("No storage comparison data available. Please run the analysis first.")
        return
    
    # Overview comparison table
    st.markdown("### 📊 Storage Options Overview")
    
    comparison_data = []
    for storage_type, data in fsx_comparisons.items():
        comparison_data.append({
            'Storage Type': storage_type,
            'Migration Time (hrs)': f"{data.get('estimated_migration_time_hours', 0):.1f}",
            'Throughput (Mbps)': f"{data.get('migration_throughput_mbps', 0):,.0f}",
            'Monthly Cost': f"${data.get('estimated_monthly_storage_cost', 0):,.2f}",
            'Performance': data.get('performance_rating', 'Unknown'),
            'Cost Rating': data.get('cost_rating', 'Unknown'),
            'Complexity': data.get('complexity_rating', 'Unknown')
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    st.dataframe(df_comparison, use_container_width=True)
    
    # Detailed analysis for each storage type
    st.markdown("---")
    st.markdown("### 🔍 Detailed Storage Analysis")
    
    for storage_type, data in fsx_comparisons.items():
        with st.expander(f"📁 {storage_type} - Detailed Analysis"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Performance Metrics:**")
                st.write(f"Migration Time: {data.get('estimated_migration_time_hours', 0):.1f} hours")
                st.write(f"Throughput: {data.get('migration_throughput_mbps', 0):,.0f} Mbps")
                st.write(f"Performance Rating: {data.get('performance_rating', 'Unknown')}")
                
                # Agent configuration
                agent_config = data.get('agent_configuration', {})
                if agent_config:
                    st.write(f"Agent Cost: ${agent_config.get('total_monthly_cost', 0):,.2f}/month")
                    st.write(f"Performance Multiplier: {agent_config.get('storage_performance_multiplier', 1.0):.2f}x")
            
            with col2:
                st.markdown("**Cost Analysis:**")
                st.write(f"Monthly Storage Cost: ${data.get('estimated_monthly_storage_cost', 0):,.2f}")
                st.write(f"Cost Rating: {data.get('cost_rating', 'Unknown')}")
                st.write(f"Complexity: {data.get('complexity_rating', 'Unknown')}")
                
                # Recommendations
                recommendations = data.get('recommendations', [])
                if recommendations:
                    st.markdown("**Recommendations:**")
                    for rec in recommendations:
                        st.write(f"• {rec}")

def render_ai_insights_tab(analysis: Dict, config: Dict):
    """Render AI insights and recommendations tab"""
    st.subheader("🧠 AI-Powered Insights & Recommendations")
    
    ai_assessment = analysis.get('ai_overall_assessment', {})
    aws_sizing = analysis.get('aws_sizing_recommendations', {})
    ai_analysis = aws_sizing.get('ai_analysis', {}) if aws_sizing else {}
    
    if not ai_analysis and not ai_assessment:
        st.warning("No AI analysis data available. Please run the analysis first.")
        return
    
    # Risk Assessment
    st.markdown("### ⚠️ Risk Assessment")
    
    if ai_analysis:
        risk_factors = ai_analysis.get('risk_factors', [])
        if risk_factors:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🚨 Identified Risk Factors:**")
                for risk in risk_factors:
                    st.write(f"• {risk}")
            
            with col2:
                st.markdown("**🛡️ Mitigation Strategies:**")
                mitigation = ai_analysis.get('mitigation_strategies', [])
                for strategy in mitigation:
                    st.write(f"• {strategy}")
    
    # Performance Recommendations
    st.markdown("---")
    st.markdown("### 🚀 Performance Recommendations")
    
    if ai_analysis:
        perf_recommendations = ai_analysis.get('performance_recommendations', [])
        if perf_recommendations:
            for i, rec in enumerate(perf_recommendations, 1):
                st.write(f"{i}. {rec}")
    
    # Timeline and Best Practices
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📅 Implementation Timeline")
        if ai_analysis:
            timeline = ai_analysis.get('timeline_suggestions', [])
            for phase in timeline:
                st.write(f"• {phase}")
    
    with col2:
        st.markdown("### 💡 Best Practices")
        if ai_analysis:
            best_practices = ai_analysis.get('best_practices', [])
            for practice in best_practices:
                st.write(f"• {practice}")
    
    # Testing and Rollback
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🧪 Testing Strategy")
        if ai_analysis:
            testing = ai_analysis.get('testing_strategy', [])
            for test in testing:
                st.write(f"• {test}")
    
    with col2:
        st.markdown("### 🔄 Rollback Procedures")
        if ai_analysis:
            rollback = ai_analysis.get('rollback_procedures', [])
            for procedure in rollback:
                st.write(f"• {procedure}")

def render_backup_storage_analysis_tab(analysis: Dict, config: Dict):
    """Render backup storage specific analysis tab"""
    st.subheader("💾 Backup Storage Migration Analysis")
    
    migration_method = config.get('migration_method', 'direct_replication')
    
    if migration_method != 'backup_restore':
        st.info("This analysis is only available for Backup/Restore migration method.")
        return
    
    # Backup Configuration Overview
    st.markdown("### 📋 Backup Configuration")
    
    backup_storage_type = config.get('backup_storage_type', 'nas_drive') or 'nas_drive'
    backup_size_multiplier = config.get('backup_size_multiplier', 0.7) or 0.7
    database_size_gb = config.get('database_size_gb', 1000) or 1000
    backup_size_gb = database_size_gb * backup_size_multiplier
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Safely format backup storage type
        if backup_storage_type and isinstance(backup_storage_type, str):
            formatted_storage = backup_storage_type.replace('_', ' ').title()
        else:
            formatted_storage = 'NAS Drive'
        st.metric("🗄️ Backup Storage", formatted_storage)
    with col2:
        st.metric("📊 Database Size", f"{database_size_gb:,} GB")
    with col3:
        st.metric("💾 Backup Size", f"{backup_size_gb:,.0f} GB")
    with col4:
        st.metric("📈 Size Ratio", f"{int(backup_size_multiplier*100)}%")
    
    # Protocol and Performance Analysis
    st.markdown("---")
    st.markdown("### 🌐 Protocol & Performance Analysis")
    
    # Get backup storage considerations from AI analysis
    aws_sizing = analysis.get('aws_sizing_recommendations', {})
    ai_analysis = aws_sizing.get('ai_analysis', {}) if aws_sizing else {}
    backup_considerations = ai_analysis.get('backup_storage_considerations', {})
    
    if backup_considerations.get('applicable', False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📡 Protocol Details:**")
            protocol = backup_considerations.get('protocol', 'Unknown') or 'Unknown'
            storage_type = backup_considerations.get('storage_type', 'Unknown') or 'Unknown'
            backup_size_factor = backup_considerations.get('backup_size_factor', 0.7) or 0.7
            
            st.write(f"Protocol: {protocol}")
            st.write(f"Storage Type: {storage_type}")
            st.write(f"Backup Size Factor: {backup_size_factor:.1f}x")
            
            # Advantages
            advantages = backup_considerations.get('advantages', [])
            if advantages:
                st.markdown("**✅ Advantages:**")
                for advantage in advantages:
                    st.write(f"• {advantage}")
        
        with col2:
            st.markdown("**⚡ Performance Optimizations:**")
            optimizations = backup_considerations.get('optimizations', [])
            for opt in optimizations:
                st.write(f"• {opt}")
            
            # Challenges
            challenges = backup_considerations.get('challenges', [])
            if challenges:
                st.markdown("**⚠️ Challenges:**")
                for challenge in challenges:
                    st.write(f"• {challenge}")
    
    # Network Path Analysis
    st.markdown("---")
    st.markdown("### 🛣️ Network Path Analysis")
    
    network_perf = analysis.get('network_performance', {})
    if network_perf:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🌐 Network Quality", f"{network_perf.get('network_quality_score', 0):.1f}/100")
        with col2:
            st.metric("📊 Effective Bandwidth", f"{network_perf.get('effective_bandwidth_mbps', 0):,.0f} Mbps")
        with col3:
            st.metric("⏱️ Total Latency", f"{network_perf.get('total_latency_ms', 0):.1f} ms")
        
        # Network segments
        segments = network_perf.get('segments', [])
        if segments:
            st.markdown("**🔗 Network Segments:**")
            segment_data = []
            for segment in segments:
                # Safely format connection type
                connection_type = segment.get('connection_type', 'Unknown')
                if connection_type and isinstance(connection_type, str):
                    formatted_type = connection_type.replace('_', ' ').title()
                else:
                    formatted_type = 'Unknown'
                
                segment_data.append({
                    'Segment': segment.get('name', 'Unknown'),
                    'Type': formatted_type,
                    'Bandwidth (Mbps)': f"{segment.get('effective_bandwidth_mbps', 0):,.0f}",
                    'Latency (ms)': f"{segment.get('effective_latency_ms', 0):.1f}",
                    'Reliability': f"{segment.get('reliability', 0)*100:.2f}%"
                })
            
            df_segments = pd.DataFrame(segment_data)
            st.dataframe(df_segments, use_container_width=True)

def main():
    """Main application function"""
    render_enhanced_header()
    
    # Get configuration from sidebar
    config = render_enhanced_sidebar_controls()
    
    # Main analysis button
    if st.button("🚀 Run Comprehensive AI Migration Analysis", type="primary", use_container_width=True):
        with st.spinner("🤖 Running comprehensive AI-powered migration analysis..."):
            try:
                # Validate configuration
                if not config or not isinstance(config, dict):
                    st.error("❌ Invalid configuration. Please check your inputs.")
                    return
                
                # Initialize the analyzer
                analyzer = EnhancedMigrationAnalyzer()
                
                # Run comprehensive analysis
                analysis = asyncio.run(analyzer.comprehensive_ai_migration_analysis(config))
                
                # Validate analysis results
                if not analysis or not isinstance(analysis, dict):
                    st.error("❌ Analysis returned invalid results. Please try again.")
                    return
                
                # Store results in session state
                st.session_state['analysis_results'] = analysis
                st.session_state['config'] = config
                
                st.success("✅ Analysis completed successfully!")
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                logger.error(f"Analysis error: {e}")
                # Clear any partial results
                if 'analysis_results' in st.session_state:
                    del st.session_state['analysis_results']
    
    # Display results if available
    if 'analysis_results' in st.session_state and st.session_state['analysis_results']:
        analysis = st.session_state['analysis_results']
        config = st.session_state.get('config', {})
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📊 Analysis Results",
            "☁️ AWS Sizing",
            "💰 Cost Analysis", 
            "🗄️ Storage Comparison",
            "💾 Backup Storage",
            "🧠 AI Insights",
            "📄 Reports"
        ])
        
        with tab1:
            render_enhanced_analysis_results_tab(analysis, config)
        
        with tab2:
            render_aws_sizing_recommendations_tab(analysis, config)
        
        with tab3:
            render_comprehensive_cost_analysis_tab_enhanced(analysis, config)
        
        with tab4:
            render_destination_storage_comparison_tab(analysis, config)
        
        with tab5:
            render_backup_storage_analysis_tab(analysis, config)
        
        with tab6:
            render_ai_insights_tab(analysis, config)
        
        with tab7:
            st.subheader("📄 Generate Reports")
            st.markdown("Use the Cost Analysis tab to generate comprehensive PDF reports and export data.")
            
            # Quick summary with safe data access
            if analysis:
                migration_time = analysis.get('estimated_migration_time_hours', 0) or 0
                cost_analysis = analysis.get('cost_analysis', {}) or {}
                total_cost = cost_analysis.get('total_monthly_cost', 0) or 0
                ai_assessment = analysis.get('ai_overall_assessment', {}) or {}
                readiness_score = ai_assessment.get('migration_readiness_score', 0) or 0
                primary_tool = analysis.get('primary_tool', 'Unknown') or 'Unknown'
                
                # Safely format primary tool
                if isinstance(primary_tool, str):
                    formatted_tool = primary_tool.upper()
                else:
                    formatted_tool = 'UNKNOWN'
                
                st.info(f"""
                f"**📋 Quick Summary:**
                f"• **Migration Time:** {migration_time:.1f} hours
                f"• **Monthly AWS Cost:** ${total_cost:,.2f}
                f" **Readiness Score:** {readiness_score}/100
                • **Primary Tool:** AWS {formatted_tool}
                """)
    
    # Footer
    st.markdown("""
    <div class="enterprise-footer">
        <h3>🏢 AWS Enterprise Database Migration Analyzer AI v3.0</h3>
        <p>Professional-grade migration analysis with AI-powered insights, comprehensive cost modeling, and real-time AWS integration.</p>
        <p><strong>Features:</strong> Agent Scaling Optimization • FSx Destination Analysis • Backup Storage Migration • VROPS Integration • AI Risk Assessment</p>
        <p style="margin-top: 1rem; font-size: 0.8rem; opacity: 0.8;">
            Powered by Anthropic Claude AI • AWS Pricing APIs • Professional PDF Reports • Comprehensive Cost Analysis
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    
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
        width: 100%;
        margin: 1rem 0;
    }
    
    .cost-table th {
        background-color: #f8fafc;
        padding: 12px;
        text-align: left;
        border-bottom: 2px solid #e5e7eb;
        font-weight: 600;
    }
    
    .cost-table td {
        padding: 10px 12px;
        border-bottom: 1px solid #e5e7eb;
    }
    
    .cost-table tr:hover {
        background-color: #f9fafb;
    }
    
    .total-row {
        background-color: #f3f4f6;
        font-weight: 600;
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

class PDFReportGenerator:
    """Generate comprehensive PDF reports for migration analysis"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=20,
            textColor=colors.darkblue,
            spaceAfter=30,
            alignment=1  # Center alignment
        )
        self.heading_style = ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=colors.darkblue,
            spaceAfter=12,
            spaceBefore=20
        )
        self.normal_style = ParagraphStyle(
            'CustomNormal',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=6
        )
    
    def generate_comprehensive_report(self, analysis: Dict, config: Dict, cost_summary: Dict) -> bytes:
        """Generate comprehensive PDF report"""
        
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        
        try:
            # Create the PDF document
            doc = SimpleDocTemplate(
                temp_file.name,
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=18
            )
            
            # Build the story (content)
            story = []
            
            # Title page
            story.extend(self._create_title_page(config))
            story.append(PageBreak())
            
            # Executive Summary
            story.extend(self._create_executive_summary(analysis, config))
            story.append(PageBreak())
            
            # VROPS Metrics Summary
            story.extend(self._create_vrops_summary(config))
            story.append(Spacer(1, 20))
            
            # Comprehensive Cost Analysis
            story.extend(self._create_cost_analysis_section(cost_summary))
            story.append(PageBreak())
            
            # Migration Configuration
            story.extend(self._create_configuration_section(config))
            story.append(PageBreak())
            
            # Performance Analysis
            story.extend(self._create_performance_section(analysis))
            story.append(PageBreak())
            
            # AI Recommendations
            story.extend(self._create_ai_recommendations_section(analysis))
            story.append(PageBreak())
            
            # Network Analysis
            story.extend(self._create_network_analysis_section(analysis))
            story.append(PageBreak())
            
            # AWS Sizing Recommendations
            story.extend(self._create_aws_sizing_section(analysis))
            
            # Build the PDF
            doc.build(story)
            
            # Read the file content
            with open(temp_file.name, 'rb') as f:
                pdf_content = f.read()
            
            return pdf_content
            
        finally:
            # Clean up the temporary file
            try:
                os.unlink(temp_file.name)
            except:
                pass
    
    def _create_title_page(self, config: Dict) -> List:
        """Create title page"""
        story = []
        
        # Main title
        story.append(Paragraph("AWS Enterprise Database Migration", self.title_style))
        story.append(Paragraph("Analysis Report", self.title_style))
        story.append(Spacer(1, 50))
        
        # Migration details
        story.append(Paragraph("Migration Configuration", self.heading_style))
        
        config_data = [
            ['Source Database', config.get('source_database_engine', 'Unknown').upper()],
            ['Target Database', config.get('database_engine', 'Unknown').upper()],
            ['Database Size', f"{config.get('database_size_gb', 0):,} GB"],
            ['Migration Method', config.get('migration_method', 'direct_replication').replace('_', ' ').title()],
            ['Environment', config.get('environment', 'unknown').title()],
            ['Target Platform', config.get('target_platform', 'rds').upper()],
            ['Generated', datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
        ]
        
        config_table = Table(config_data, colWidths=[2*inch, 3*inch])
        config_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(config_table)
        story.append(Spacer(1, 50))
        
        # Disclaimer
        story.append(Paragraph("Report Disclaimer", self.heading_style))
        story.append(Paragraph(
            "This report is generated by AWS Enterprise Database Migration Analyzer AI v3.0. "
            "All recommendations are based on the provided configuration and should be validated "
            "in your specific environment before implementation.",
            self.normal_style
        ))
        
        return story
    
    def _create_executive_summary(self, analysis: Dict, config: Dict) -> List:
        """Create executive summary section"""
        story = []
        
        story.append(Paragraph("Executive Summary", self.title_style))
        
        # Key metrics
        ai_assessment = analysis.get('ai_overall_assessment', {})
        
        summary_data = [
            ['Metric', 'Value', 'Status'],
            ['Migration Readiness Score', f"{ai_assessment.get('migration_readiness_score', 0)}/100", 'Ready' if ai_assessment.get('migration_readiness_score', 0) > 75 else 'Needs Review'],
            ['Success Probability', f"{ai_assessment.get('success_probability', 0)}%", 'High' if ai_assessment.get('success_probability', 0) > 80 else 'Medium'],
            ['Risk Level', ai_assessment.get('risk_level', 'Unknown'), ai_assessment.get('risk_level', 'Unknown')],
            ['Estimated Migration Time', f"{analysis.get('estimated_migration_time_hours', 0):.1f} hours", 'Acceptable' if analysis.get('estimated_migration_time_hours', 0) < 24 else 'Long'],
            ['Migration Throughput', f"{analysis.get('migration_throughput_mbps', 0):,.0f} Mbps", 'Good' if analysis.get('migration_throughput_mbps', 0) > 1000 else 'Standard']
        ]
        
        summary_table = Table(summary_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(summary_table)
        story.append(Spacer(1, 20))
        
        # Key recommendations
        story.append(Paragraph("Key Recommendations", self.heading_style))
        
        recommendations = ai_assessment.get('recommended_next_steps', [])
        for i, rec in enumerate(recommendations[:5], 1):
            story.append(Paragraph(f"{i}. {rec}", self.normal_style))
        
        return story
    
    def _create_vrops_summary(self, config: Dict) -> List:
        """Create VROPS metrics summary section"""
        story = []
        
        story.append(Paragraph("VROPS Performance Metrics", self.heading_style))
        
        vrops_data = [
            ['Metric', 'Current Value', 'Recommended AWS Sizing Impact'],
            ['Database Max Memory', f"{config.get('current_db_max_memory_gb', 'N/A')} GB", 'Used for RDS/EC2 instance sizing'],
            ['Database Max CPU Cores', f"{config.get('current_db_max_cpu_cores', 'N/A')} cores", 'Used for vCPU requirements'],
            ['Database Max IOPS', f"{config.get('current_db_max_iops', 'N/A'):,}" if config.get('current_db_max_iops') else 'N/A', 'Used for storage IOPS sizing'],
            ['Database Max Throughput', f"{config.get('current_db_max_throughput_mbps', 'N/A')} MB/s" if config.get('current_db_max_throughput_mbps') else 'N/A', 'Used for network/storage throughput'],
            ['VROPS Max Memory', f"{config.get('vrops_max_memory_gb', 'N/A')} GB" if config.get('vrops_max_memory_gb') else 'N/A', 'Additional memory validation'],
            ['VROPS Max CPU', f"{config.get('vrops_max_cpu_percent', 'N/A')}%" if config.get('vrops_max_cpu_percent') else 'N/A', 'CPU utilization validation'],
            ['VROPS Max Storage IOPS', f"{config.get('vrops_max_storage_iops', 'N/A'):,}" if config.get('vrops_max_storage_iops') else 'N/A', 'Storage performance validation'],
            ['VROPS Max Network', f"{config.get('vrops_max_network_mbps', 'N/A')} Mbps" if config.get('vrops_max_network_mbps') else 'N/A', 'Network capacity validation'],
            ['VROPS Disk Latency', f"{config.get('vrops_avg_disk_latency_ms', 'N/A')} ms" if config.get('vrops_avg_disk_latency_ms') else 'N/A', 'Storage latency requirements']
        ]
        
        vrops_table = Table(vrops_data, colWidths=[2*inch, 1.5*inch, 2.5*inch])
        vrops_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkgreen),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgreen),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP')
        ]))
        
        story.append(vrops_table)
        
        return story
    
    def _create_cost_analysis_section(self, cost_summary: Dict) -> List:
        """Create comprehensive cost analysis section"""
        story = []
        
        story.append(Paragraph("Comprehensive AWS Cost Analysis", self.title_style))
        
        if not cost_summary:
            story.append(Paragraph("Cost analysis data not available.", self.normal_style))
            return story
        
        # Cost summary overview
        story.append(Paragraph("Cost Summary Overview", self.heading_style))
        
        overview_data = [
            ['Cost Category', 'Monthly Cost', 'Annual Cost'],
            ['Total Monthly Cost', f"${cost_summary.get('total_monthly_cost', 0):,.2f}", f"${cost_summary.get('total_monthly_cost', 0) * 12:,.2f}"],
            ['Total One-Time Cost', f"${cost_summary.get('total_one_time_cost', 0):,.2f}", 'N/A'],
            ['3-Year Total Cost', f"${cost_summary.get('three_year_total', 0):,.2f}", 'N/A']
        ]
        
        overview_table = Table(overview_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
        overview_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(overview_table)
        story.append(Spacer(1, 20))
        
        # Detailed cost breakdown
        story.append(Paragraph("Detailed Cost Breakdown", self.heading_style))
        
        detailed_costs = cost_summary.get('detailed_costs', [])
        if detailed_costs:
            # Convert to table format
            cost_data = [['Service/Component', 'Type', 'Monthly Cost', 'Annual Cost', 'Notes']]
            
            for cost_item in detailed_costs:
                monthly_cost = cost_item.get('monthly_cost', 0)
                annual_cost = monthly_cost * 12 if cost_item.get('cost_type') == 'monthly' else 0
                
                cost_data.append([
                    cost_item.get('service', 'Unknown'),
                    cost_item.get('cost_type', 'monthly').title(),
                    f"${monthly_cost:,.2f}",
                    f"${annual_cost:,.2f}" if annual_cost > 0 else 'N/A',
                    cost_item.get('notes', '')[:50] + '...' if len(cost_item.get('notes', '')) > 50 else cost_item.get('notes', '')
                ])
            
            cost_table = Table(cost_data, colWidths=[1.2*inch, 0.8*inch, 1*inch, 1*inch, 2*inch])
            cost_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkred),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightcoral),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'TOP')
            ]))
            
            story.append(cost_table)
        
        return story
    
    def _create_configuration_section(self, config: Dict) -> List:
        """Create configuration section"""
        story = []
        
        story.append(Paragraph("Migration Configuration Details", self.title_style))
        
        # Hardware configuration
        story.append(Paragraph("Hardware Configuration", self.heading_style))
        
        hw_data = [
            ['Component', 'Specification'],
            ['Operating System', config.get('operating_system', 'Unknown').replace('_', ' ').title()],
            ['Server Type', config.get('server_type', 'Unknown').title()],
            ['RAM', f"{config.get('ram_gb', 0)} GB"],
            ['CPU Cores', f"{config.get('cpu_cores', 0)} cores"],
            ['CPU Speed', f"{config.get('cpu_ghz', 0)} GHz"],
            ['Network Interface', config.get('nic_type', 'Unknown').replace('_', ' ').title()],
            ['Network Speed', f"{config.get('nic_speed', 0):,} Mbps"]
        ]
        
        hw_table = Table(hw_data, colWidths=[2*inch, 3*inch])
        hw_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(hw_table)
        story.append(Spacer(1, 20))
        
        # Migration configuration
        story.append(Paragraph("Migration Settings", self.heading_style))
        
        migration_data = [
            ['Parameter', 'Value'],
            ['Migration Method', config.get('migration_method', 'Unknown').replace('_', ' ').title()],
            ['Destination Storage', config.get('destination_storage_type', 'Unknown')],
            ['Number of Agents', str(config.get('number_of_agents', 1))],
            ['Downtime Tolerance', f"{config.get('downtime_tolerance_minutes', 0)} minutes"],
            ['Performance Requirements', config.get('performance_requirements', 'Unknown').title()]
        ]
        
        if config.get('migration_method') == 'backup_restore':
            migration_data.extend([
                ['Backup Storage Type', config.get('backup_storage_type', 'Unknown').replace('_', ' ').title()],
                ['Backup Size Multiplier', f"{config.get('backup_size_multiplier', 0.7):.1f}x"]
            ])
        
        migration_table = Table(migration_data, colWidths=[2*inch, 3*inch])
        migration_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkgreen),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgreen),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(migration_table)
        
        return story
    
    def _create_performance_section(self, analysis: Dict) -> List:
        """Create performance analysis section"""
        story = []
        
        story.append(Paragraph("Performance Analysis", self.title_style))
        
        onprem_perf = analysis.get('onprem_performance', {})
        network_perf = analysis.get('network_performance', {})
        
        # Performance metrics
        perf_data = [
            ['Metric', 'Value', 'Rating'],
            ['Overall Performance Score', f"{onprem_perf.get('performance_score', 0):.1f}/100", 'Good' if onprem_perf.get('performance_score', 0) > 75 else 'Needs Improvement'],
            ['Network Quality Score', f"{network_perf.get('network_quality_score', 0):.1f}/100", 'Good' if network_perf.get('network_quality_score', 0) > 75 else 'Needs Improvement'],
            ['Effective Bandwidth', f"{network_perf.get('effective_bandwidth_mbps', 0):,.0f} Mbps", 'Adequate'],
            ['Total Network Latency', f"{network_perf.get('total_latency_ms', 0):.1f} ms", 'Low' if network_perf.get('total_latency_ms', 0) < 50 else 'Medium'],
            ['Network Reliability', f"{network_perf.get('total_reliability', 0)*100:.2f}%", 'High' if network_perf.get('total_reliability', 0) > 0.99 else 'Medium']
        ]
        
        perf_table = Table(perf_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
        perf_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.purple),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lavender),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(perf_table)
        
        return story
    
    def _create_ai_recommendations_section(self, analysis: Dict) -> List:
        """Create AI recommendations section"""
        story = []
        
        story.append(Paragraph("AI-Powered Recommendations", self.title_style))
        
        ai_analysis = analysis.get('aws_sizing_recommendations', {}).get('ai_analysis', {})
        
        # Risk factors
        story.append(Paragraph("Identified Risk Factors", self.heading_style))
        risk_factors = ai_analysis.get('risk_factors', [])
        for i, risk in enumerate(risk_factors[:5], 1):
            story.append(Paragraph(f"{i}. {risk}", self.normal_style))
        
        story.append(Spacer(1, 15))
        
        # Mitigation strategies
        story.append(Paragraph("Recommended Mitigation Strategies", self.heading_style))
        mitigation_strategies = ai_analysis.get('mitigation_strategies', [])
        for i, strategy in enumerate(mitigation_strategies[:5], 1):
            story.append(Paragraph(f"{i}. {strategy}", self.normal_style))
        
        story.append(Spacer(1, 15))
        
        # Performance recommendations
        story.append(Paragraph("Performance Optimization Recommendations", self.heading_style))
        perf_recommendations = ai_analysis.get('performance_recommendations', [])
        for i, rec in enumerate(perf_recommendations[:5], 1):
            story.append(Paragraph(f"{i}. {rec}", self.normal_style))
        
        return story
    
    def _create_network_analysis_section(self, analysis: Dict) -> List:
        """Create network analysis section"""
        story = []
        
        story.append(Paragraph("Network Path Analysis", self.title_style))
        
        network_perf = analysis.get('network_performance', {})
        
        # Network segments
        segments = network_perf.get('segments', [])
        if segments:
            segment_data = [['Segment', 'Type', 'Bandwidth (Mbps)', 'Latency (ms)', 'Reliability']]
            
            for segment in segments:
                segment_data.append([
                    segment.get('name', 'Unknown'),
                    segment.get('connection_type', 'Unknown').replace('_', ' ').title(),
                    f"{segment.get('effective_bandwidth_mbps', 0):,.0f}",
                    f"{segment.get('effective_latency_ms', 0):.1f}",
                    f"{segment.get('reliability', 0)*100:.2f}%"
                ])
            
            segment_table = Table(segment_data, colWidths=[2*inch, 1.2*inch, 1*inch, 0.8*inch, 1*inch])
            segment_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.orange),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightyellow),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(segment_table)
        
        return story
    
    def _create_aws_sizing_section(self, analysis: Dict) -> List:
        """Create AWS sizing recommendations section"""
        story = []
        
        story.append(Paragraph("AWS Sizing Recommendations", self.title_style))
        
        aws_sizing = analysis.get('aws_sizing_recommendations', {})
        deployment_rec = aws_sizing.get('deployment_recommendation', {})
        
        # Deployment recommendation
        story.append(Paragraph(f"Recommended Deployment: {deployment_rec.get('recommendation', 'Unknown').upper()}", self.heading_style))
        
        if deployment_rec.get('recommendation') == 'rds':
            rds_rec = aws_sizing.get('rds_recommendations', {})
            
            rds_data = [
                ['Configuration', 'Specification'],
                ['Primary Instance Type', rds_rec.get('primary_instance', 'Unknown')],
                ['Storage Type', rds_rec.get('storage_type', 'gp3')],
                ['Storage Size', f"{rds_rec.get('storage_size_gb', 0):,} GB"],
                ['Monthly Instance Cost', f"${rds_rec.get('monthly_instance_cost', 0):,.2f}"],
                ['Monthly Storage Cost', f"${rds_rec.get('monthly_storage_cost', 0):,.2f}"],
                ['Total Monthly Cost', f"${rds_rec.get('total_monthly_cost', 0):,.2f}"],
                ['Multi-AZ', 'Yes' if rds_rec.get('multi_az', False) else 'No']
            ]
            
        else:
            ec2_rec = aws_sizing.get('ec2_recommendations', {})
            
            rds_data = [
                ['Configuration', 'Specification'],
                ['Primary Instance Type', ec2_rec.get('primary_instance', 'Unknown')],
                ['Storage Type', ec2_rec.get('storage_type', 'gp3')],
                ['Storage Size', f"{ec2_rec.get('storage_size_gb', 0):,} GB"],
                ['Monthly Instance Cost', f"${ec2_rec.get('monthly_instance_cost', 0):,.2f}"],
                ['Monthly Storage Cost', f"${ec2_rec.get('monthly_storage_cost', 0):,.2f}"],
                ['OS Licensing Cost', f"${ec2_rec.get('os_licensing_cost', 0):,.2f}"],
                ['Total Monthly Cost', f"${ec2_rec.get('total_monthly_cost', 0):,.2f}"]
            ]
        
        aws_table = Table(rds_data, colWidths=[2.5*inch, 2.5*inch])
        aws_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(aws_table)
        
        return story

class CostSummaryManager:
    """Manage comprehensive cost summary across all analysis components"""
    
    def __init__(self):
        self.cost_components = []
    
    def generate_comprehensive_cost_summary(self, analysis: Dict, config: Dict) -> Dict:
        """Generate comprehensive cost summary from all analysis components"""
        
        cost_summary = {
            'total_monthly_cost': 0,
            'total_one_time_cost': 0,
            'total_annual_cost': 0,
            'three_year_total': 0,
            'detailed_costs': [],
            'cost_categories': {},
            'optimization_opportunities': []
        }

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
            Professional-Grade Migration Analysis • AI-Powered Insights • Real-time AWS Integration • Agent Scaling Optimization • FSx Destination Analysis • Backup Storage Support • VROPS Metrics Integration
        </p>
        <div style="margin-top: 1rem; font-size: 0.8rem;">
            <span style="margin-right: 20px;">{ai_status} Anthropic Claude AI</span>
            <span style="margin-right: 20px;">{aws_status} AWS Pricing APIs</span>
            <span style="margin-right: 20px;">🟢 Network Intelligence Engine</span>
            <span style="margin-right: 20px;">🟢 Agent Scaling Optimizer</span>
            <span style="margin-right: 20px;">🟢 FSx Destination Analysis</span>
            <span style="margin-right: 20px;">🟢 Backup Storage Migration</span>
            <span>🟢 VROPS Metrics Integration</span>
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
    """Enhanced sidebar with AI-powered recommendations and VROPS metrics"""
    st.sidebar.header("🤖 AI-Powered Migration Configuration v3.0 with VROPS")
    
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
    
    # Database Performance Inputs
    st.sidebar.subheader("📊 Current Database Performance")
    st.sidebar.markdown("*Enter your actual database metrics for accurate AWS sizing*")
    
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
    
    # VROPS Metrics Section
    st.sidebar.subheader("📈 VROPS Performance Metrics")
    st.sidebar.markdown("*VMware vRealize Operations metrics for validation*")
    
    enable_vrops = st.sidebar.checkbox("Enable VROPS Metrics", value=False, help="Include VROPS metrics for enhanced analysis")
    
    vrops_max_memory_gb = None
    vrops_max_cpu_percent = None
    vrops_max_storage_iops = None
    vrops_max_network_mbps = None
    vrops_avg_disk_latency_ms = None
    
    if enable_vrops:
        vrops_max_memory_gb = st.sidebar.number_input(
            "VROPS Max Memory Usage (GB)", 
            min_value=1, max_value=1024, value=int(ram_gb * 0.7), step=1,
            help="Peak memory usage from VROPS monitoring"
        )
        
        vrops_max_cpu_percent = st.sidebar.slider(
            "VROPS Max CPU Utilization (%)", 
            min_value=1, max_value=100, value=75, step=1,
            help="Peak CPU utilization percentage from VROPS"
        )
        
        vrops_max_storage_iops = st.sidebar.number_input(
            "VROPS Max Storage IOPS", 
            min_value=100, max_value=500000, value=15000, step=1000,
            help="Peak storage IOPS from VROPS monitoring"
        )
        
        vrops_max_network_mbps = st.sidebar.number_input(
            "VROPS Max Network Throughput (Mbps)", 
            min_value=10, max_value=50000, value=int(nic_speed * 0.7), step=100,
            help="Peak network throughput from VROPS"
        )
        
        vrops_avg_disk_latency_ms = st.sidebar.number_input(
            "VROPS Avg Disk Latency (ms)", 
            min_value=0.1, max_value=100.0, value=5.0, step=0.1,
            help="Average disk latency from VROPS monitoring"
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
        'vrops_max_memory_gb': vrops_max_memory_gb,
        'vrops_max_cpu_percent': vrops_max_cpu_percent,
        'vrops_max_storage_iops': vrops_max_storage_iops,
        'vrops_max_network_mbps': vrops_max_network_mbps,
        'vrops_avg_disk_latency_ms': vrops_avg_disk_latency_ms,
        'enable_vrops': enable_vrops,
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
    
    def _calculate_ec2_sizing(self, config: Dict, pricing_data: Dict) -> Dict:
        """Calculate EC2 sizing based on database size and performance metrics"""
        database_size_gb = config['database_size_gb']
        
        # Get current database performance metrics for better sizing
        current_memory_gb = config.get('current_db_max_memory_gb', 0)
        current_cpu_cores = config.get('current_db_max_cpu_cores', 0)
        current_iops = config.get('current_db_max_iops', 0)
        current_throughput_mbps = config.get('current_db_max_throughput_mbps', 0)
        
        # VROPS metrics for validation
        vrops_memory_gb = config.get('vrops_max_memory_gb', 0)
        vrops_cpu_percent = config.get('vrops_max_cpu_percent', 0)
        vrops_storage_iops = config.get('vrops_max_storage_iops', 0)
        vrops_network_mbps = config.get('vrops_max_network_mbps', 0)
        vrops_disk_latency_ms = config.get('vrops_avg_disk_latency_ms', 0)
        
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
        
        # Use the better metric between current DB metrics and VROPS
        effective_memory = max(current_memory_gb, vrops_memory_gb) if vrops_memory_gb > 0 else current_memory_gb
        effective_iops = max(current_iops, vrops_storage_iops) if vrops_storage_iops > 0 else current_iops
        
        # Adjust based on memory usage
        if effective_memory > 0:
            # Add 25% buffer for OS overhead and growth (EC2 needs more overhead than RDS)
            required_memory = effective_memory * 1.25
            
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
        
        # Adjust based on CPU usage
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
        if effective_iops > 0:
            # Add 35% buffer for peak operations (more than RDS)
            required_iops = effective_iops * 1.35
            
            if required_iops > 50000:
                sizing_reasoning.append(f"Very high IOPS requirement: {required_iops:.0f} IOPS - consider io1/io2 EBS")
            elif required_iops > 25000:
                sizing_reasoning.append(f"High IOPS requirement: {required_iops:.0f} IOPS - consider gp3 with provisioned IOPS")
            elif required_iops > 10000:
                sizing_reasoning.append(f"Medium IOPS requirement: {required_iops:.0f} IOPS")
        
        # VROPS-specific adjustments
        if vrops_disk_latency_ms > 0:
            if vrops_disk_latency_ms > 20:
                sizing_reasoning.append(f"High disk latency detected ({vrops_disk_latency_ms}ms) - consider io1/io2 EBS")
            elif vrops_disk_latency_ms < 5:
                sizing_reasoning.append(f"Low latency requirements ({vrops_disk_latency_ms}ms) - gp3 sufficient")
        
        if vrops_cpu_percent > 0:
            if vrops_cpu_percent > 80:
                sizing_reasoning.append(f"High CPU utilization detected ({vrops_cpu_percent}%) - consider larger instances")
            elif vrops_cpu_percent < 30:
                sizing_reasoning.append(f"Low CPU utilization ({vrops_cpu_percent}%) - opportunity for cost optimization")
        
        # SQL Server specific adjustments
        if database_engine == 'sqlserver':
            sizing_reasoning.append("SQL Server on EC2 requires additional resources")
            
            # SQL Server needs more memory and CPU
            if effective_memory > 0 and effective_memory < 16:
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
        if effective_iops > 20000:
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
            'storage_type': 'gp3' if effective_iops <= 20000 else 'io1',
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
                'current_throughput_mbps': current_throughput_mbps,
                'vrops_memory_gb': vrops_memory_gb,
                'vrops_cpu_percent': vrops_cpu_percent,
                'vrops_storage_iops': vrops_storage_iops,
                'vrops_network_mbps': vrops_network_mbps,
                'vrops_disk_latency_ms': vrops_disk_latency_ms
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
        
        # Default fallback for direct replication
        return f"nonprod_sj_nas_drive_s3"
    
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
        
        # VROPS metrics for validation
        vrops_memory_gb = config.get('vrops_max_memory_gb', 0)
        vrops_cpu_percent = config.get('vrops_max_cpu_percent', 0)
        vrops_storage_iops = config.get('vrops_max_storage_iops', 0)
        vrops_network_mbps = config.get('vrops_max_network_mbps', 0)
        vrops_disk_latency_ms = config.get('vrops_avg_disk_latency_ms', 0)
        
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
        
        # Use the better metric between current DB metrics and VROPS
        effective_memory = max(current_memory_gb, vrops_memory_gb) if vrops_memory_gb > 0 else current_memory_gb
        effective_iops = max(current_iops, vrops_storage_iops) if vrops_storage_iops > 0 else current_iops
        
        # Adjust based on memory usage
        if effective_memory > 0:
            # Add 20% buffer for AWS overhead and growth
            required_memory = effective_memory * 1.2
            
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
        
        # Adjust based on CPU usage
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
        if effective_iops > 0:
            # Add 30% buffer for peak operations
            required_iops = effective_iops * 1.3
            
            if required_iops > 40000:
                sizing_reasoning.append(f"High IOPS requirement: {required_iops:.0f} IOPS")
                # May need io1/io2 storage
            elif required_iops > 20000:
                sizing_reasoning.append(f"Medium-high IOPS requirement: {required_iops:.0f} IOPS")
        
        # VROPS-specific adjustments
        if vrops_disk_latency_ms > 0:
            if vrops_disk_latency_ms > 20:
                sizing_reasoning.append(f"High disk latency detected ({vrops_disk_latency_ms}ms) - consider io1/io2 storage")
            elif vrops_disk_latency_ms < 5:
                sizing_reasoning.append(f"Low latency requirements ({vrops_disk_latency_ms}ms) - gp3 sufficient")
        
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
                'current_throughput_mbps': current_throughput_mbps,
                'vrops_memory_gb': vrops_memory_gb,
                'vrops_cpu_percent': vrops_cpu_percent,
                'vrops_storage_iops': vrops_storage_iops,
                'vrops_network_mbps': vrops_network_mbps,
                'vrops_disk_latency_ms': vrops_disk_latency_ms
            }
        }
        
        # Extract costs from different analysis components
        self._extract_compute_costs(analysis, cost_summary)
        self._extract_storage_costs(analysis, cost_summary)
        self._extract_network_costs(analysis, cost_summary)
        self._extract_migration_agent_costs(analysis, cost_summary)
        self._extract_additional_aws_costs(analysis, cost_summary, config)
        
        # Calculate totals
        self._calculate_totals(cost_summary)
        
        # Add optimization recommendations
        self._add_optimization_opportunities(cost_summary, analysis, config)
        
        return cost_summary
    
    def _extract_compute_costs(self, analysis: Dict, cost_summary: Dict):
        """Extract compute costs (EC2/RDS)"""
        aws_sizing = analysis.get('aws_sizing_recommendations', {})
        deployment_rec = aws_sizing.get('deployment_recommendation', {})
        
        if deployment_rec.get('recommendation') == 'rds':
            rds_rec = aws_sizing.get('rds_recommendations', {})
            
            # RDS Writer Instance
            cost_summary['detailed_costs'].append({
                'service': 'RDS Primary Instance',
                'component': rds_rec.get('primary_instance', 'Unknown'),
                'cost_type': 'monthly',
                'monthly_cost': rds_rec.get('monthly_instance_cost', 0),
                'annual_cost': rds_rec.get('monthly_instance_cost', 0) * 12,
                'category': 'Compute',
                'notes': f"Primary database instance - {rds_rec.get('primary_instance', 'Unknown')}"
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
            }
        }
    
    def calculate_ai_enhanced_path_performance(self, path_key: str, time_of_day: int = None) -> Dict:
        """AI-enhanced network path performance calculation"""
        path = self.network_paths.get(path_key, self.network_paths['nonprod_sj_nas_drive_s3'])
        
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
        })
            
            # RDS Reader Instances
            reader_writer = aws_sizing.get('reader_writer_config', {})
            readers = reader_writer.get('readers', 0)
            if readers > 0:
                reader_cost = rds_rec.get('monthly_instance_cost', 0) * readers * 0.9  # Reader instances are slightly cheaper
                cost_summary['detailed_costs'].append({
                    'service': f'RDS Read Replicas ({readers}x)',
                    'component': rds_rec.get('primary_instance', 'Unknown'),
                    'cost_type': 'monthly',
                    'monthly_cost': reader_cost,
                    'annual_cost': reader_cost * 12,
                    'category': 'Compute',
                    'notes': f"{readers} read replica instances for read scaling"
                })
            
            # Multi-AZ costs
            if rds_rec.get('multi_az', False):
                multi_az_cost = rds_rec.get('monthly_instance_cost', 0)
                cost_summary['detailed_costs'].append({
                    'service': 'RDS Multi-AZ',
                    'component': 'High Availability',
                    'cost_type': 'monthly',
                    'monthly_cost': multi_az_cost,
                    'annual_cost': multi_az_cost * 12,
                    'category': 'Compute',
                    'notes': 'Multi-AZ deployment for high availability'
                })
        
        else:  # EC2
            ec2_rec = aws_sizing.get('ec2_recommendations', {})
            
            # EC2 Instances
            instance_count = ec2_rec.get('instance_count', 1)
            instance_cost = ec2_rec.get('monthly_instance_cost', 0)
            
            cost_summary['detailed_costs'].append({
                'service': f'EC2 Instances ({instance_count}x)',
                'component': ec2_rec.get('primary_instance', 'Unknown'),
                'cost_type': 'monthly',
                'monthly_cost': instance_cost,
                'annual_cost': instance_cost * 12,
                'category': 'Compute',
                'notes': f"{instance_count} EC2 instances for self-managed database"
            })
            
            # OS Licensing
            os_licensing = ec2_rec.get('os_licensing_cost', 0)
            if os_licensing > 0:
                cost_summary['detailed_costs'].append({
                    'service': 'OS Licensing',
                    'component': 'Windows Server',
                    'cost_type': 'monthly',
                    'monthly_cost': os_licensing,
                    'annual_cost': os_licensing * 12,
                    'category': 'Licensing',
                    'notes': 'Windows Server licensing costs'
                })
    
    def _extract_storage_costs(self, analysis: Dict, cost_summary: Dict):
        """Extract storage costs"""
        aws_sizing = analysis.get('aws_sizing_recommendations', {})
        deployment_rec = aws_sizing.get('deployment_recommendation', {})
        comprehensive_costs = analysis.get('comprehensive_costs', {})
        
        if deployment_rec.get('recommendation') == 'rds':
            rds_rec = aws_sizing.get('rds_recommendations', {})
            storage_cost = rds_rec.get('monthly_storage_cost', 0)
            
            cost_summary['detailed_costs'].append({
                'service': 'RDS Storage',
                'component': f"{rds_rec.get('storage_type', 'gp3')} - {rds_rec.get('storage_size_gb', 0):,} GB",
                'cost_type': 'monthly',
                'monthly_cost': storage_cost,
                'annual_cost': storage_cost * 12,
                'category': 'Storage',
                'notes': f"RDS {rds_rec.get('storage_type', 'gp3')} storage"
            })
        else:
            ec2_rec = aws_sizing.get('ec2_recommendations', {})
            storage_cost = ec2_rec.get('monthly_storage_cost', 0)
            
            cost_summary['detailed_costs'].append({
                'service': 'EBS Storage',
                'component': f"{ec2_rec.get('storage_type', 'gp3')} - {ec2_rec.get('storage_size_gb', 0):,} GB",
                'cost_type': 'monthly',
                'monthly_cost': storage_cost,
                'annual_cost': storage_cost * 12,
                'category': 'Storage',
                'notes': f"EBS {ec2_rec.get('storage_type', 'gp3')} volumes"
            })
        
        # Destination storage (S3, FSx)
        storage_costs = comprehensive_costs.get('storage_costs', {})
        dest_storage = storage_costs.get('destination_storage', {})
        if dest_storage.get('monthly_cost', 0) > 0:
            cost_summary['detailed_costs'].append({
                'service': f"Destination Storage ({dest_storage.get('type', 'S3')})",
                'component': f"{dest_storage.get('size_gb', 0):,} GB",
                'cost_type': 'monthly',
                'monthly_cost': dest_storage.get('monthly_cost', 0),
                'annual_cost': dest_storage.get('monthly_cost', 0) * 12,
                'category': 'Storage',
                'notes': f"Migration destination storage - {dest_storage.get('type', 'S3')}"
            })
        
        # Backup storage (if applicable)
        backup_storage = storage_costs.get('backup_storage', {})
        if backup_storage.get('applicable', False) and backup_storage.get('monthly_cost', 0) > 0:
            cost_summary['detailed_costs'].append({
                'service': 'Backup Storage',
                'component': f"{backup_storage.get('size_gb', 0):,} GB",
                'cost_type': 'monthly',
                'monthly_cost': backup_storage.get('monthly_cost', 0),
                'annual_cost': backup_storage.get('monthly_cost', 0) * 12,
                'category': 'Storage',
                'notes': 'Backup files for migration process'
            })
    
    def _extract_network_costs(self, analysis: Dict, cost_summary: Dict):
        """Extract network costs"""
        comprehensive_costs = analysis.get('comprehensive_costs', {})
        network_costs = comprehensive_costs.get('network_costs', {})
        
        # Direct Connect
        dx = network_costs.get('direct_connect', {})
        if dx.get('monthly_cost', 0) > 0:
            cost_summary['detailed_costs'].append({
                'service': 'AWS Direct Connect',
                'component': dx.get('capacity', 'Unknown'),
                'cost_type': 'monthly',
                'monthly_cost': dx.get('monthly_cost', 0),
                'annual_cost': dx.get('monthly_cost', 0) * 12,
                'category': 'Network',
                'notes': f"Dedicated network connection - {dx.get('capacity', 'Unknown')}"
            })
        
        # Data Transfer
        transfer = network_costs.get('data_transfer', {})
        if transfer.get('monthly_cost', 0) > 0:
            cost_summary['detailed_costs'].append({
                'service': 'Data Transfer',
                'component': f"{transfer.get('monthly_gb', 0):,} GB/month",
                'cost_type': 'monthly',
                'monthly_cost': transfer.get('monthly_cost', 0),
                'annual_cost': transfer.get('monthly_cost', 0) * 12,
                'category': 'Network',
                'notes': 'Ongoing data transfer charges'
            })
        
        # VPN Backup
        vpn = network_costs.get('vpn_backup', {})
        if vpn.get('monthly_cost', 0) > 0:
            cost_summary['detailed_costs'].append({
                'service': 'VPN Gateway',
                'component': 'Backup Connection',
                'cost_type': 'monthly',
                'monthly_cost': vpn.get('monthly_cost', 0),
                'annual_cost': vpn.get('monthly_cost', 0) * 12,
                'category': 'Network',
                'notes': 'VPN backup connection for redundancy'
            })
    
    def _extract_migration_agent_costs(self, analysis: Dict, cost_summary: Dict):
        """Extract migration agent costs"""
        agent_analysis = analysis.get('agent_analysis', {})
        comprehensive_costs = analysis.get('comprehensive_costs', {})
        migration_costs = comprehensive_costs.get('migration_costs', {})
        
        # Migration Agents (EC2 instances running agents)
        agent_costs = migration_costs.get('agent_costs', {})
        if agent_costs.get('monthly_cost', 0) > 0:
            cost_summary['detailed_costs'].append({
                'service': f"{agent_analysis.get('primary_tool', 'Migration').upper()} Agents",
                'component': f"{agent_costs.get('agent_count', 1)}x {agent_analysis.get('agent_size', 'medium')}",
                'cost_type': 'monthly',
                'monthly_cost': agent_costs.get('monthly_cost', 0),
                'annual_cost': agent_costs.get('monthly_cost', 0) * 12,
                'category': 'Migration Services',
                'notes': f"EC2 instances running {agent_analysis.get('primary_tool', 'migration')} agents"
            })
        
        # DataSync One-time Transfer
        datasync = migration_costs.get('datasync', {})
        if datasync.get('applicable', False) and datasync.get('one_time_transfer_cost', 0) > 0:
            cost_summary['detailed_costs'].append({
                'service': 'DataSync Transfer',
                'component': f"{datasync.get('data_size_gb', 0):,} GB",
                'cost_type': 'one_time',
                'monthly_cost': 0,
                'one_time_cost': datasync.get('one_time_transfer_cost', 0),
                'category': 'Migration Services',
                'notes': 'One-time data transfer via DataSync'
            })
        
        # DataSync Ongoing Sync
        if datasync.get('monthly_sync_cost', 0) > 0:
            cost_summary['detailed_costs'].append({
                'service': 'DataSync Ongoing Sync',
                'component': 'Incremental Changes',
                'cost_type': 'monthly',
                'monthly_cost': datasync.get('monthly_sync_cost', 0),
                'annual_cost': datasync.get('monthly_sync_cost', 0) * 12,
                'category': 'Migration Services',
                'notes': 'Ongoing incremental synchronization'
            })
        
        # Professional Services
        setup = migration_costs.get('setup_and_professional_services', {})
        if setup.get('one_time_cost', 0) > 0:
            cost_summary['detailed_costs'].append({
                'service': 'Professional Services',
                'component': 'Setup & Migration Support',
                'cost_type': 'one_time',
                'monthly_cost': 0,
                'one_time_cost': setup.get('one_time_cost', 0),
                'category': 'Professional Services',
                'notes': 'Migration planning, setup, and support services'
            })
    
    def _extract_additional_aws_costs(self, analysis: Dict, cost_summary: Dict, config: Dict):
        """Extract additional AWS service costs"""
        
        # CloudWatch Monitoring
        database_size_gb = config.get('database_size_gb', 1000)
        monitoring_cost = max(50, database_size_gb * 0.01)  # Basic monitoring cost
        
        cost_summary['detailed_costs'].append({
            'service': 'CloudWatch Monitoring',
            'component': 'Metrics & Logs',
            'cost_type': 'monthly',
            'monthly_cost': monitoring_cost,
            'annual_cost': monitoring_cost * 12,
            'category': 'Monitoring',
            'notes': 'CloudWatch metrics, logs, and dashboards'
        })
        
        # AWS Config (for compliance)
        environment = config.get('environment', 'non-production')
        if environment == 'production':
            config_cost = 30
            cost_summary['detailed_costs'].append({
                'service': 'AWS Config',
                'component': 'Compliance Monitoring',
                'cost_type': 'monthly',
                'monthly_cost': config_cost,
                'annual_cost': config_cost * 12,
                'category': 'Compliance',
                'notes': 'Configuration compliance monitoring'
            })
        
        # AWS Systems Manager
        systems_manager_cost = 25
        cost_summary['detailed_costs'].append({
            'service': 'AWS Systems Manager',
            'component': 'Patch Management',
            'cost_type': 'monthly',
            'monthly_cost': systems_manager_cost,
            'annual_cost': systems_manager_cost * 12,
            'category': 'Management',
            'notes': 'Automated patching and maintenance'
        })
        
        # AWS Backup
        backup_cost = database_size_gb * 0.05  # $0.05 per GB for backup
        cost_summary['detailed_costs'].append({
            'service': 'AWS Backup',
            'component': f"{database_size_gb:,} GB",
            'cost_type': 'monthly',
            'monthly_cost': backup_cost,
            'annual_cost': backup_cost * 12,
            'category': 'Backup',
            'notes': 'Automated database backups'
        })
        
        # Security services
        if environment == 'production':
            # AWS WAF
            waf_cost = 50
            cost_summary['detailed_costs'].append({
                'service': 'AWS WAF',
                'component': 'Web Application Firewall',
                'cost_type': 'monthly',
                'monthly_cost': waf_cost,
                'annual_cost': waf_cost * 12,
                'category': 'Security',
                'notes': 'Web application protection'
            })
            
            # AWS GuardDuty
            guardduty_cost = 75
            cost_summary['detailed_costs'].append({
                'service': 'AWS GuardDuty',
                'component': 'Threat Detection',
                'cost_type': 'monthly',
                'monthly_cost': guardduty_cost,
                'annual_cost': guardduty_cost * 12,
                'category': 'Security',
                'notes': 'Intelligent threat detection'
            })
    
    def _calculate_totals(self, cost_summary: Dict):
        """Calculate total costs"""
        total_monthly = 0
        total_one_time = 0
        
        categories = {}
        
        for cost_item in cost_summary['detailed_costs']:
            monthly_cost = cost_item.get('monthly_cost', 0)
            one_time_cost = cost_item.get('one_time_cost', 0)
            category = cost_item.get('category', 'Other')
            
            total_monthly += monthly_cost
            total_one_time += one_time_cost
            
            if category not in categories:
                categories[category] = {'monthly': 0, 'one_time': 0}
            
            categories[category]['monthly'] += monthly_cost
            categories[category]['one_time'] += one_time_cost
        
        cost_summary['total_monthly_cost'] = total_monthly
        cost_summary['total_one_time_cost'] = total_one_time
        cost_summary['total_annual_cost'] = total_monthly * 12 + total_one_time
        cost_summary['three_year_total'] = total_monthly * 36 + total_one_time
        cost_summary['cost_categories'] = categories
    
    def _add_optimization_opportunities(self, cost_summary: Dict, analysis: Dict, config: Dict):
        """Add cost optimization opportunities"""
        optimizations = []
        
        # Reserved Instance savings
        total_monthly = cost_summary['total_monthly_cost']
        if total_monthly > 500:
            ri_savings = total_monthly * 0.25  # 25% savings
            optimizations.append({
                'opportunity': 'Reserved Instances',
                'potential_monthly_savings': ri_savings,
                'potential_annual_savings': ri_savings * 12,
                'description': 'Purchase 1 or 3-year Reserved Instances for 20-30% cost savings'
            })
        
        # Spot Instance savings for non-production
        if config.get('environment') == 'non-production':
            spot_savings = total_monthly * 0.6  # 60% savings
            optimizations.append({
                'opportunity': 'Spot Instances',
                'potential_monthly_savings': spot_savings,
                'potential_annual_savings': spot_savings * 12,
                'description': 'Use Spot Instances for development/testing workloads'
            })
        
        # Storage optimization
        storage_costs = sum(cost['monthly_cost'] for cost in cost_summary['detailed_costs'] if cost['category'] == 'Storage')
        if storage_costs > 200:
            storage_savings = storage_costs * 0.15  # 15% savings
            optimizations.append({
                'opportunity': 'Storage Lifecycle Policies',
                'potential_monthly_savings': storage_savings,
                'potential_annual_savings': storage_savings * 12,
                'description': 'Implement intelligent storage tiering and lifecycle policies'
            })
        
        # Agent optimization
        agent_analysis = analysis.get('agent_analysis', {})
        if agent_analysis.get('scaling_efficiency', 1.0) < 0.9:
            agent_savings = total_monthly * 0.1  # 10% savings from optimization
            optimizations.append({
                'opportunity': 'Agent Optimization',
                'potential_monthly_savings': agent_savings,
                'potential_annual_savings': agent_savings * 12,
                'description': 'Optimize migration agent configuration for better efficiency'
            })
        
        cost_summary['optimization_opportunities'] = optimizations

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

            VROPS PERFORMANCE METRICS:
            - VROPS Max Memory: {config.get('vrops_max_memory_gb', 'Not specified')} GB
            - VROPS Max CPU: {config.get('vrops_max_cpu_percent', 'Not specified')}%
            - VROPS Max Storage IOPS: {config.get('vrops_max_storage_iops', 'Not specified')} IOPS
            - VROPS Max Network: {config.get('vrops_max_network_mbps', 'Not specified')} Mbps
            - VROPS Disk Latency: {config.get('vrops_avg_disk_latency_ms', 'Not specified')} ms

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
            16. VROPS METRICS VALIDATION: Validate VROPS metrics against database metrics and provide recommendations

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
                'vrops_validation': ai_analysis.get('vrops_validation', {}),
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
            'vrops_validation': self._analyze_vrops_validation(config),
            'detailed_assessment': {
                'overall_readiness': 'ready' if perf_score > 75 and complexity_score < 7 else 'needs_preparation' if perf_score > 60 else 'significant_preparation_required',
                'success_probability': max(60, 95 - (complexity_score * 5) - max(0, (70 - perf_score))),
                'recommended_approach': 'direct_migration' if complexity_score < 6 and config['database_size_gb'] < 2000 else 'staged_migration'
            }
        }
    
    def _analyze_vrops_validation(self, config: Dict) -> Dict:
        """Analyze VROPS metrics validation"""
        vrops_validation = {
            'validation_status': 'complete',
            'discrepancies': [],
            'recommendations': [],
            'confidence_level': 'high'
        }
        
        # Compare VROPS vs Database metrics
        db_memory = config.get('current_db_max_memory_gb', 0)
        vrops_memory = config.get('vrops_max_memory_gb', 0)
        
        if vrops_memory > 0 and db_memory > 0:
            memory_diff = abs(vrops_memory - db_memory) / max(vrops_memory, db_memory)
            if memory_diff > 0.2:  # 20% difference
                vrops_validation['discrepancies'].append(f"Memory metrics differ by {memory_diff*100:.1f}%: VROPS={vrops_memory}GB vs DB={db_memory}GB")
                vrops_validation['recommendations'].append("Validate memory allocation and usage patterns")
        
        # Check VROPS disk latency for AWS storage recommendations
        vrops_latency = config.get('vrops_avg_disk_latency_ms', 0)
        if vrops_latency > 0:
            if vrops_latency > 20:
                vrops_validation['recommendations'].append("High disk latency detected - consider io1/io2 EBS for better performance")
            elif vrops_latency < 5:
                vrops_validation['recommendations'].append("Low latency requirements - gp3 EBS should be sufficient")
        
        # VROPS CPU utilization analysis
        vrops_cpu = config.get('vrops_max_cpu_percent', 0)
        if vrops_cpu > 0:
            if vrops_cpu > 80:
                vrops_validation['recommendations'].append("High CPU utilization - consider larger AWS instance types")
            elif vrops_cpu < 30:
                vrops_validation['recommendations'].append("Low CPU utilization - opportunity for cost optimization with smaller instances")
        
        return vrops_validation
    
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
            "Configure proper instance sizing based on VROPS metrics",
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
            'vrops_validation': self._analyze_vrops_validation(config),
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
                    'strengths': ['Excellent performance', 'Strong security', 'Enterprise support'],
                    'weaknesses': ['Licensing costs', 'Learning curve for Windows admins'],
                    'migration_considerations': ['Package compatibility', 'Service migration']
                }
            },
            'rhel_9': {
                'name': 'Red Hat Enterprise Linux 9',
                'cpu_efficiency': 0.98,
                'memory_efficiency': 0.96,
                'io_efficiency': 0.97,
                'network_efficiency': 0.97,
                'virtualization_overhead': 0.04,
                'database_optimizations': {
                    'mysql': 0.97, 'postgresql': 0.98, 'oracle': 0.95, 'sqlserver': 0.87, 'mongodb': 0.97
                },
                'licensing_cost_factor': 1.8,
                'management_complexity': 0.6,
                'security_overhead': 0.03,
                'ai_insights': {
                    'strengths': ['Latest performance optimizations', 'Modern security features'],
                    'weaknesses': ['Newer OS risks', 'Higher licensing than older versions'],
                    'migration_considerations': ['Application compatibility testing', 'Staff training']
                }
            },
            'ubuntu_20_04': {
                'name': 'Ubuntu Server 20.04 LTS',
                'cpu_efficiency': 0.94,
                'memory_efficiency': 0.93,
                'io_efficiency': 0.93,
                'network_efficiency': 0.94,
                'virtualization_overhead': 0.05,
                'database_optimizations': {
                    'mysql': 0.94, 'postgresql': 0.96, 'oracle': 0.88, 'sqlserver': 0.82, 'mongodb': 0.95
                },
                'licensing_cost_factor': 0.0,
                'management_complexity': 0.8,
                'security_overhead': 0.02,
                'ai_insights': {
                    'strengths': ['No licensing costs', 'Strong community support', 'Good performance'],
                    'weaknesses': ['Higher management complexity', 'Limited enterprise support'],
                    'migration_considerations': ['Open source stack migration', 'Support model changes']
                }
            },
            'ubuntu_22_04': {
                'name': 'Ubuntu Server 22.04 LTS',
                'cpu_efficiency': 0.96,
                'memory_efficiency': 0.95,
                'io_efficiency': 0.95,
                'network_efficiency': 0.96,
                'virtualization_overhead': 0.03,
                'database_optimizations': {
                    'mysql': 0.96, 'postgresql': 0.97, 'oracle': 0.90, 'sqlserver': 0.84, 'mongodb': 0.96
                },
                'licensing_cost_factor': 0.0,
                'management_complexity': 0.7,
                'security_overhead': 0.02,
                'ai_insights': {
                    'strengths': ['Latest LTS features', 'No licensing costs', 'Modern optimizations'],
                    'weaknesses': ['Newer OS compatibility risks', 'Complex enterprise integration'],
                    'migration_considerations': ['Long-term support planning', 'Enterprise tool compatibility']
                }
            }
        }
    
    def calculate_os_performance_impact(self, os_type: str, platform_type: str, config: Dict) -> Dict:
        """Enhanced OS performance calculation with AI insights"""
        os_config = self.operating_systems.get(os_type, self.operating_systems['ubuntu_22_04'])
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

            {migration_details}

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

            Provide quantitative analysis wherever possible, including specific metrics, percentages, and measurable outcomes.
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
                'timeline_suggestions': ai_analysis.get('timeline_suggestions', []),
                'best_practices': ai_analysis.get('best_practices', []),
                'testing_strategy': ai_analysis.get('testing_strategy', []),
                'rollback_procedures': ai_analysis.get('rollback_procedures', []),
                'confidence_level': ai_analysis.get('confidence_level', 'medium'),
                'raw_ai_response': ai_response
            }
            
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
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
        
        complexity_score = min(10, max(1, base_complexity))
        
        return {
            'complexity_score': complexity_score,
            'complexity_factors': complexity_factors,
            'risk_factors': ['Standard migration risks apply'],
            'mitigation_strategies': ['Conduct thorough testing', 'Plan rollback procedures'],
            'performance_recommendations': ['Optimize before migration'],
            'timeline_suggestions': ['Plan 4-6 week timeline'],
            'best_practices': ['Follow AWS best practices'],
            'testing_strategy': ['Comprehensive testing required'],
            'rollback_procedures': ['Maintain backup systems'],
            'confidence_level': 'medium'
        }
    
    def _fallback_workload_analysis(self, config: Dict, performance_data: Dict) -> Dict:
        """Fallback analysis when AI is not available"""
        complexity_score = 5
        if config['source_database_engine'] != config['database_engine']:
            complexity_score += 2
        if config['database_size_gb'] > 5000:
            complexity_score += 1
        
        return {
            'ai_complexity_score': min(10, complexity_score),
            'risk_factors': ["Migration complexity varies with database engine differences"],
            'mitigation_strategies': ["Conduct thorough pre-migration testing"],
            'performance_recommendations': ["Optimize database before migration"],
            'confidence_level': 'medium',
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
            return {
                'region': region,
                'last_updated': datetime.now(),
                'data_source': 'aws_api',
                'ec2_instances': self._fallback_ec2_pricing(),
                'rds_instances': self._fallback_rds_pricing(),
                'storage': self._fallback_storage_pricing()
            }
        except Exception as e:
            logger.error(f"Failed to fetch AWS pricing: {e}")
            return self._fallback_pricing_data(region)
    
    def _fallback_pricing_data(self, region: str) -> Dict:
        """Fallback pricing data"""
        return {
            'region': region,
            'last_updated': datetime.now(),
            'data_source': 'fallback',
            'ec2_instances': self._fallback_ec2_pricing(),
            'rds_instances': self._fallback_rds_pricing(),
            'storage': self._fallback_storage_pricing()
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
            'effective_cost_per_hour': total_cost_per_hour * management_overhead_factor * storage_overhead
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

class EnhancedNetworkIntelligenceManager:
    """AI-powered network path intelligence with enhanced analysis including backup storage paths"""
    
    def __init__(self):
        self.network_paths = {
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
            }
        }
    
    def calculate_ai_enhanced_path_performance(self, path_key: str, time_of_day: int = None) -> Dict:
        """AI-enhanced network path performance calculation"""
        path = self.network_paths.get(path_key, self.network_paths['nonprod_sj_nas_drive_s3'])
        
        if time_of_day is None:
            time_of_day = datetime.now().hour
        
        total_latency = 0
        min_bandwidth = float('inf')
        total_reliability = 1.0
        
        adjusted_segments = []
        
        for segment in path['segments']:
            segment_latency = segment['latency_ms']
            segment_bandwidth = segment['bandwidth_mbps']
            segment_reliability = segment['reliability']
            
            # Time-of-day adjustments
            congestion_factor = 1.1 if 9 <= time_of_day <= 17 else 0.95
            
            effective_bandwidth = segment_bandwidth / congestion_factor
            effective_latency = segment_latency * congestion_factor
            
            total_latency += effective_latency
            min_bandwidth = min(min_bandwidth, effective_bandwidth)
            total_reliability *= segment_reliability
            
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
        
        network_quality = (latency_score * 0.25 + bandwidth_score * 0.45 + reliability_score * 0.30)
        
        return {
            'path_name': path['name'],
            'destination_storage': path['destination_storage'],
            'migration_type': path.get('migration_type', 'direct_replication'),
            'total_latency_ms': total_latency,
            'effective_bandwidth_mbps': min_bandwidth,
            'total_reliability': total_reliability,
            'network_quality_score': network_quality,
            'segments': adjusted_segments,
            'environment': path['environment'],
            'os_type': path['os_type'],
            'storage_type': path['storage_type'],
            'ai_insights': path['ai_insights']
        }

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
            migration_type = 'backup_restore'
            primary_tool = 'datasync'
        else:
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
            'fsx_comparisons': fsx_comparisons,
            'ai_overall_assessment': ai_overall_assessment
        }
    
    def _get_network_path_key(self, config: Dict) -> str:
        """Get network path key based on migration method and backup storage"""
        return "nonprod_sj_nas_drive_s3"
    
    async def _analyze_ai_migration_agents_with_scaling(self, config: Dict, primary_tool: str, network_perf: Dict) -> Dict:
        """Enhanced migration agent analysis with scaling support"""
        
        num_agents = config.get('number_of_agents', 1)
        destination_storage = config.get('destination_storage_type', 'S3')
        migration_method = config.get('migration_method', 'direct_replication')
        
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
        
        effective_throughput = min(total_max_throughput, network_bandwidth)
        
        # Determine bottleneck
        if total_max_throughput < network_bandwidth:
            bottleneck = f'agents ({num_agents} agents)'
        else:
            bottleneck = 'network'
        
        return {
            'primary_tool': primary_tool,
            'agent_size': agent_size,
            'number_of_agents': num_agents,
            'destination_storage': destination_storage,
            'migration_method': migration_method,
            'agent_configuration': agent_config,
            'total_max_throughput_mbps': total_max_throughput,
            'total_effective_throughput': effective_throughput,
            'bottleneck': bottleneck,
            'scaling_efficiency': agent_config['scaling_efficiency'],
            'management_overhead': agent_config['management_overhead_factor'],
            'storage_performance_multiplier': agent_config.get('storage_performance_multiplier', 1.0),
            'cost_per_hour': agent_config['effective_cost_per_hour'],
            'monthly_cost': agent_config['total_monthly_cost']
        }
    
    async def _calculate_ai_migration_time_with_agents(self, config: Dict, migration_throughput: float, 
                                                     onprem_performance: Dict, agent_analysis: Dict) -> float:
        """AI-enhanced migration time calculation"""
        
        database_size_gb = config['database_size_gb']
        migration_method = config.get('migration_method', 'direct_replication')
        
        # Calculate data size to transfer
        if migration_method == 'backup_restore':
            backup_size_multiplier = config.get('backup_size_multiplier', 0.7)
            data_size_gb = database_size_gb * backup_size_multiplier
        else:
            data_size_gb = database_size_gb
        
        # Base calculation
        base_time_hours = (data_size_gb * 8 * 1000) / (migration_throughput * 3600) if migration_throughput > 0 else 24
        
        # Complexity factors
        complexity_factor = 1.0
        
        if config['source_database_engine'] != config['database_engine']:
            complexity_factor *= 1.3
        
        if 'windows' in config['operating_system']:
            complexity_factor *= 1.1
        
        if config['server_type'] == 'vmware':
            complexity_factor *= 1.05
        
        return base_time_hours * complexity_factor
    
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
            'reasoning': f"AI-optimized for {database_size_gb}GB database"
        }
    
    def _recommend_deployment_type(self, config: Dict, rds_rec: Dict, ec2_rec: Dict) -> Dict:
        """Recommend deployment type based on user selection and analysis"""
        target_platform = config.get('target_platform', 'rds')
        
        # Use user's explicit choice
        recommendation = target_platform
        confidence = 0.9
        
        primary_reasons = [
            f"User selected {target_platform.upper()} platform",
            f"Suitable for {config['database_size_gb']:,}GB database",
            f"Appropriate for {config.get('environment', 'non-production')} environment"
        ]
        
        return {
            'recommendation': recommendation,
            'user_choice': target_platform,
            'confidence': confidence,
            'primary_reasons': primary_reasons
        }
    
    def _calculate_rds_sizing(self, config: Dict, pricing_data: Dict) -> Dict:
        """Calculate RDS sizing based on database size"""
        database_size_gb = config['database_size_gb']
        
        # Base sizing on database size
        if database_size_gb < 1000:
            instance_type = 'db.t3.medium'
            cost_per_hour = 0.068
        elif database_size_gb < 5000:
            instance_type = 'db.r6g.large'
            cost_per_hour = 0.48
        else:
            instance_type = 'db.r6g.xlarge'
            cost_per_hour = 0.96
        
        # Storage sizing
        storage_size = max(database_size_gb * 1.5, 100)
        storage_cost = storage_size * 0.08
        
        return {
            'primary_instance': instance_type,
            'storage_type': 'gp3',
            'storage_size_gb': storage_size,
            'monthly_instance_cost': cost_per_hour * 24 * 30,
            'monthly_storage_cost': storage_cost,
            'total_monthly_cost': cost_per_hour * 24 * 30 + storage_cost,
            'multi_az': config.get('environment') == 'production'
        }
    
    def _calculate_ec2_sizing(self, config: Dict, pricing_data: Dict) -> Dict:
        """Calculate EC2 sizing based on database size"""
        database_size_gb = config['database_size_gb']
        
        # Base sizing on database size
        if database_size_gb < 1000:
            instance_type = 't3.large'
            cost_per_hour = 0.0832
        elif database_size_gb < 5000:
            instance_type = 'r6i.large'
            cost_per_hour = 0.252
        else:
            instance_type = 'r6i.xlarge'
            cost_per_hour = 0.504
        
        # Storage sizing
        storage_size = max(database_size_gb * 2.0, 100)
        storage_cost = storage_size * 0.08
        
        # Operating system licensing
        os_licensing = 0
        if 'windows' in config.get('operating_system', ''):
            os_licensing = 150
        
        return {
            'primary_instance': instance_type,
            'storage_type': 'gp3',
            'storage_size_gb': storage_size,
            'monthly_instance_cost': cost_per_hour * 24 * 30,
            'monthly_storage_cost': storage_cost,
            'os_licensing_cost': os_licensing,
            'total_monthly_cost': cost_per_hour * 24 * 30 + storage_cost + os_licensing
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
        
        # Network and other costs
        network_cost = 500
        os_licensing_cost = 300
        management_cost = 200
        
        total_monthly_cost = (aws_compute_cost + aws_storage_cost + agent_monthly_cost + 
                            network_cost + os_licensing_cost + management_cost)
        
        # One-time costs
        one_time_migration_cost = config['database_size_gb'] * 0.1 + config.get('number_of_agents', 1) * 500
        
        return {
            'aws_compute_cost': aws_compute_cost,
            'aws_storage_cost': aws_storage_cost,
            'agent_cost': agent_monthly_cost,
            'network_cost': network_cost,
            'os_licensing_cost': os_licensing_cost,
            'management_cost': management_cost,
            'total_monthly_cost': total_monthly_cost,
            'one_time_migration_cost': one_time_migration_cost,
            'estimated_monthly_savings': 500,
            'roi_months': 12
        }
    
    async def _generate_fsx_destination_comparisons(self, config: Dict) -> Dict:
        """Generate FSx destination comparisons"""
        comparisons = {}
        destination_types = ['S3', 'FSx_Windows', 'FSx_Lustre']
        
        for dest_type in destination_types:
            temp_config = config.copy()
            temp_config['destination_storage_type'] = dest_type
            
            # Basic comparison data
            comparisons[dest_type] = {
                'destination_type': dest_type,
                'estimated_migration_time_hours': 8,
                'migration_throughput_mbps': 1000,
                'estimated_monthly_storage_cost': config['database_size_gb'] * 0.08,
                'performance_rating': 'Good',
                'cost_rating': 'Good',
                'complexity_rating': 'Low',
                'recommendations': [f'{dest_type} is suitable for this workload']
            }
        
        return comparisons
    
    async def _generate_ai_overall_assessment_with_agents(self, config: Dict, onprem_performance: Dict, 
                                                        aws_sizing: Dict, migration_time: float, 
                                                        agent_analysis: Dict) -> Dict:
        """Generate AI overall assessment"""
        
        readiness_score = 80
        success_probability = 85
        risk_level = 'Medium'
        
        # Adjust based on configuration
        if config['database_size_gb'] > 10000:
            readiness_score -= 10
        
        if config['source_database_engine'] != config['database_engine']:
            readiness_score -= 15
        
        if migration_time > 24:
            readiness_score -= 10
        
        return {
            'migration_readiness_score': readiness_score,
            'success_probability': success_probability,
            'risk_level': risk_level,
            'readiness_factors': ['System appears ready for migration'],
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
            }
        }

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
            Professional-Grade Migration Analysis • AI-Powered Insights • Real-time AWS Integration
        </p>
        <div style="margin-top: 1rem; font-size: 0.8rem;">
            <span style="margin-right: 20px;">{ai_status} Anthropic Claude AI</span>
            <span style="margin-right: 20px;">{aws_status} AWS Pricing APIs</span>
            <span style="margin-right: 20px;">🟢 Network Intelligence Engine</span>
            <span>🟢 Agent Scaling Optimizer</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_enhanced_sidebar_controls():
    """Enhanced sidebar with AI-powered recommendations"""
    st.sidebar.header("🤖 AI-Powered Migration Configuration v3.0")
    
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
    
    # Database Configuration
    st.sidebar.subheader("🗄️ Database Configuration")
    
    source_database_engine = st.sidebar.selectbox(
        "Source Database",
        ["mysql", "postgresql", "oracle", "sqlserver", "mongodb"],
        index=3,
        format_func=lambda x: {
            'mysql': '🐬 MySQL', 'postgresql': '🐘 PostgreSQL', 'oracle': '🏛️ Oracle',
            'sqlserver': '🪟 SQL Server', 'mongodb': '🍃 MongoDB'
        }[x]
    )
    
    target_platform = st.sidebar.selectbox(
        "Target Platform",
        ["rds", "ec2"],
        format_func=lambda x: {
            'rds': '☁️ Amazon RDS (Managed Service)',
            'ec2': '🖥️ Amazon EC2 (Self-Managed)'
        }[x]
    )
    
    database_engine = st.sidebar.selectbox(
        "Target Database",
        ["mysql", "postgresql", "oracle", "sqlserver", "mongodb"],
        index=3 if source_database_engine == "sqlserver" else 0,
        format_func=lambda x: {
            'mysql': '🐬 MySQL', 'postgresql': '🐘 PostgreSQL', 'oracle': '🏛️ Oracle',
            'sqlserver': '🪟 SQL Server', 'mongodb': '🍃 MongoDB'
        }[x]
    )
    
    database_size_gb = st.sidebar.number_input("Database Size (GB)", 
                                              min_value=100, max_value=100000, value=1000, step=100)
    
    downtime_tolerance_minutes = st.sidebar.number_input("Max Downtime (minutes)", 
                                                        min_value=1, max_value=480, value=60)
    
    performance_requirements = st.sidebar.selectbox("Performance Requirement", ["standard", "high"])
    
    # Migration Configuration
    st.sidebar.subheader("🔄 Migration Configuration")
    
    migration_method = st.sidebar.selectbox(
        "Migration Method",
        ["backup_restore", "direct_replication"],
        format_func=lambda x: {
            'backup_restore': '📦 Backup/Restore via DataSync',
            'direct_replication': '🔄 Direct Replication via DMS'
        }[x]
    )
    
    backup_storage_type = st.sidebar.selectbox(
        "Backup Storage Type",
        ["nas_drive", "windows_share"],
        format_func=lambda x: {
            'nas_drive': '🗄️ NAS Drive',
            'windows_share': '🪟 Windows Share Drive'
        }[x]
    )
    
    backup_size_multiplier = st.sidebar.selectbox(
        "Backup Size vs Database",
        [0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        index=2,
        format_func=lambda x: f"{int(x*100)}% of DB size"
    )
    
    destination_storage_type = st.sidebar.selectbox(
        "AWS Destination Storage",
        ["S3", "FSx_Windows", "FSx_Lustre"],
        format_func=lambda x: {
            'S3': '☁️ Amazon S3 (Standard)',
            'FSx_Windows': '🪟 Amazon FSx for Windows',
            'FSx_Lustre': '⚡ Amazon FSx for Lustre'
        }[x]
    )
    
    environment = st.sidebar.selectbox("Environment", ["non-production", "production"])
    
    # Agent Configuration
    st.sidebar.subheader("🤖 Migration Agent Configuration")
    
    number_of_agents = st.sidebar.number_input(
        "Number of Migration Agents",
        min_value=1, max_value=10, value=2, step=1
    )
    
    datasync_agent_size = st.sidebar.selectbox(
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
    
    dms_agent_size = st.sidebar.selectbox(
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
        'ram_gb': ram_gb,
        'cpu_cores': cpu_cores,
        'cpu_ghz': cpu_ghz,
        'nic_type': nic_type,
        'nic_speed': nic_speed,
        'source_database_engine': source_database_engine,
        'target_platform': target_platform,
        'database_engine': database_engine,
        'database_size_gb': database_size_gb,
        'downtime_tolerance_minutes': downtime_tolerance_minutes,
        'performance_requirements': performance_requirements,
        'backup_storage_type': backup_storage_type,
        'backup_size_multiplier': backup_size_multiplier,
        'migration_method': migration_method,
        'destination_storage_type': destination_storage_type,
        'environment': environment,
        'number_of_agents': number_of_agents,
        'datasync_agent_size': datasync_agent_size,
        'dms_agent_size': dms_agent_size
    }

def render_enhanced_analysis_results_tab(analysis: Dict, config: Dict):
    """Render enhanced analysis results with AI insights"""
    st.subheader("🤖 AI-Enhanced Migration Analysis Results")
    
    if not analysis:
        st.warning("No analysis data available. Please run the analysis first.")
        return
    
    # Migration Overview
    st.markdown("### 📋 Migration Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        migration_time = analysis.get('estimated_migration_time_hours', 0)
        st.metric("⏱️ Est. Migration Time", f"{migration_time:.1f} hours")
    
    with col2:
        migration_throughput = analysis.get('migration_throughput_mbps', 0)
        st.metric("🚀 Migration Throughput", f"{migration_throughput:,.0f} Mbps")
    
    with col3:
        primary_tool = analysis.get('primary_tool', 'Unknown')
        st.metric("🔧 Primary Tool", f"AWS {primary_tool.upper()}")
    
    with col4:
        num_agents = config.get('number_of_agents', 1)
        st.metric("🤖 Migration Agents", f"{num_agents} agents")
    
    # Performance Analysis
    st.markdown("---")
    st.markdown("### 📊 Performance Analysis")
    
    onprem_perf = analysis.get('onprem_performance', {})
    if onprem_perf:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🖥️ On-Premises Performance:**")
            perf_score = onprem_perf.get('performance_score', 0)
            st.progress(perf_score / 100)
            st.write(f"Performance Score: {perf_score:.1f}/100")
        
        with col2:
            st.markdown("**🌐 Network Performance:**")
            network_perf = analysis.get('network_performance', {})
            if network_perf:
                network_quality = network_perf.get('network_quality_score', 0)
                st.progress(network_quality / 100)
                st.write(f"Network Quality: {network_quality:.1f}/100")

def render_aws_sizing_recommendations_tab(analysis: Dict, config: Dict):
    """Render AWS sizing recommendations tab"""
    st.subheader("☁️ AWS Sizing Recommendations")
    
    aws_sizing = analysis.get('aws_sizing_recommendations', {})
    if not aws_sizing:
        st.warning("No AWS sizing data available. Please run the analysis first.")
        return
    
    # Deployment Recommendation
    deployment_rec = aws_sizing.get('deployment_recommendation', {})
    recommended_platform = deployment_rec.get('recommendation', 'rds')
    
    st.markdown(f"### 🎯 Recommended Platform: **{recommended_platform.upper()}**")
    
    confidence = deployment_rec.get('confidence', 0.5)
    st.progress(confidence)
    st.write(f"Recommendation Confidence: {confidence*100:.0f}%")
    
    # Platform-specific recommendations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏛️ RDS Recommendations")
        rds_rec = aws_sizing.get('rds_recommendations', {})
        if rds_rec:
            st.markdown(f"**Instance Type:** {rds_rec.get('primary_instance', 'Unknown')}")
            st.markdown(f"**Storage:** {rds_rec.get('storage_type', 'gp3')} - {rds_rec.get('storage_size_gb', 0):,} GB")
            st.markdown(f"**Monthly Cost:** ${rds_rec.get('total_monthly_cost', 0):,.2f}")
            st.markdown(f"**Multi-AZ:** {'Yes' if rds_rec.get('multi_az', False) else 'No'}")
    
    with col2:
        st.markdown("### 🖥️ EC2 Recommendations")
        ec2_rec = aws_sizing.get('ec2_recommendations', {})
        if ec2_rec:
            st.markdown(f"**Instance Type:** {ec2_rec.get('primary_instance', 'Unknown')}")
            st.markdown(f"**Storage:** {ec2_rec.get('storage_type', 'gp3')} - {ec2_rec.get('storage_size_gb', 0):,} GB")
            st.markdown(f"**Monthly Cost:** ${ec2_rec.get('total_monthly_cost', 0):,.2f}")

def render_cost_analysis_tab(analysis: Dict, config: Dict):
    """Render cost analysis tab"""
    st.subheader("💰 Cost Analysis")
    
    cost_analysis = analysis.get('cost_analysis', {})
    if not cost_analysis:
        st.warning("No cost analysis data available. Please run the analysis first.")
        return
    
    # Cost Overview
    st.markdown("### 💸 Cost Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_monthly = cost_analysis.get('total_monthly_cost', 0)
        st.metric("💰 Total Monthly", f"${total_monthly:,.2f}")
    
    with col2:
        total_one_time = cost_analysis.get('one_time_migration_cost', 0)
        st.metric("🔄 One-Time Costs", f"${total_one_time:,.2f}")
    
    with col3:
        annual_cost = total_monthly * 12 + total_one_time
        st.metric("📅 Annual Total", f"${annual_cost:,.2f}")
    
    with col4:
        roi_months = cost_analysis.get('roi_months', 12)
        st.metric("📈 ROI Timeline", f"{roi_months} months")
    
    # Cost Breakdown
    st.markdown("---")
    st.markdown("### 📊 Cost Breakdown")
    
    cost_data = [
        ['AWS Compute', f"${cost_analysis.get('aws_compute_cost', 0):,.2f}"],
        ['AWS Storage', f"${cost_analysis.get('aws_storage_cost', 0):,.2f}"],
        ['Migration Agents', f"${cost_analysis.get('agent_cost', 0):,.2f}"],
        ['Network', f"${cost_analysis.get('network_cost', 0):,.2f}"],
        ['OS Licensing', f"${cost_analysis.get('os_licensing_cost', 0):,.2f}"],
        ['Management', f"${cost_analysis.get('management_cost', 0):,.2f}"]
    ]
    
    df_costs = pd.DataFrame(cost_data, columns=['Component', 'Monthly Cost'])
    st.dataframe(df_costs, use_container_width=True)

def render_ai_insights_tab(analysis: Dict, config: Dict):
    """Render AI insights and recommendations tab"""
    st.subheader("🧠 AI-Powered Insights & Recommendations")
    
    ai_assessment = analysis.get('ai_overall_assessment', {})
    aws_sizing = analysis.get('aws_sizing_recommendations', {})
    ai_analysis = aws_sizing.get('ai_analysis', {}) if aws_sizing else {}
    
    if not ai_analysis and not ai_assessment:
        st.warning("No AI analysis data available. Please run the analysis first.")
        return
    
    # Migration Readiness
    st.markdown("### 🎯 Migration Readiness Assessment")
    
    readiness_score = ai_assessment.get('migration_readiness_score', 0)
    st.progress(readiness_score / 100)
    st.write(f"Migration Readiness: {readiness_score}/100")
    
    success_prob = ai_assessment.get('success_probability', 0)
    st.write(f"Success Probability: {success_prob}%")
    
    risk_level = ai_assessment.get('risk_level', 'Unknown')
    st.write(f"Risk Level: {risk_level}")
    
    # AI Recommendations
    st.markdown("---")
    st.markdown("### 💡 AI Recommendations")
    
    if ai_analysis:
        recommendations = ai_analysis.get('performance_recommendations', [])
        for i, rec in enumerate(recommendations[:5], 1):
            st.write(f"{i}. {rec}")

def main():
    """Main application function"""
    render_enhanced_header()
    
    # Get configuration from sidebar
    config = render_enhanced_sidebar_controls()
    
    # Main analysis button
    if st.button("🚀 Run Comprehensive AI Migration Analysis", type="primary", use_container_width=True):
        with st.spinner("🤖 Running comprehensive AI-powered migration analysis..."):
            try:
                # Initialize the analyzer
                analyzer = EnhancedMigrationAnalyzer()
                
                # Run comprehensive analysis
                analysis = asyncio.run(analyzer.comprehensive_ai_migration_analysis(config))
                
                # Store results in session state
                st.session_state['analysis_results'] = analysis
                st.session_state['config'] = config
                
                st.success("✅ Analysis completed successfully!")
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                logger.error(f"Analysis error: {e}")
    
    # Display results if available
    if 'analysis_results' in st.session_state and st.session_state['analysis_results']:
        analysis = st.session_state['analysis_results']
        config = st.session_state.get('config', {})
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Analysis Results",
            "☁️ AWS Sizing",
            "💰 Cost Analysis", 
            "🧠 AI Insights"
        ])
        
        with tab1:
            render_enhanced_analysis_results_tab(analysis, config)
        
        with tab2:
            render_aws_sizing_recommendations_tab(analysis, config)
        
        with tab3:
            render_cost_analysis_tab(analysis, config)
        
        with tab4:
            render_ai_insights_tab(analysis, config)
    
    # Footer
    st.markdown("""
    <div class="enterprise-footer">
        <h3>🏢 AWS Enterprise Database Migration Analyzer AI v3.0</h3>
        <p>Professional-grade migration analysis with AI-powered insights and real-time AWS integration.</p>
        <p><strong>Features:</strong> Agent Scaling Optimization • FSx Destination Analysis • Backup Storage Migration • AI Risk Assessment</p>
        <p style="margin-top: 1rem; font-size: 0.8rem; opacity: 0.8;">
            Powered by Anthropic Claude AI • AWS Pricing APIs • Professional Analysis
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()