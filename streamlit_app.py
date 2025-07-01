import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import math
from typing import Dict, List, Tuple

# Page configuration
st.set_page_config(
    page_title="AWS Migration Network Pattern Analyzer",
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
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .pattern-card {
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
        padding: 1.5rem;
        border-radius: 6px;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border-left: 3px solid #f59e0b;
        border: 1px solid #d97706;
    }
    
    .network-path-card {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        padding: 1.5rem;
        border-radius: 6px;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border-left: 3px solid #22c55e;
        border: 1px solid #e5e7eb;
    }
    
    .recommendation-card {
        background: linear-gradient(135deg, #fefdf8 0%, #fefce8 100%);
        padding: 1.5rem;
        border-radius: 6px;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border-left: 3px solid #f59e0b;
        border: 1px solid #e5e7eb;
    }
    
    .waterfall-card {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        padding: 1.5rem;
        border-radius: 6px;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border-left: 3px solid #0ea5e9;
        border: 1px solid #e5e7eb;
    }
    
    .service-card {
        background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%);
        padding: 1.5rem;
        border-radius: 6px;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border-left: 3px solid #8b5cf6;
        border: 1px solid #e5e7eb;
    }
</style>
""", unsafe_allow_html=True)

class NetworkPatternAnalyzer:
    """Enhanced class for analyzing network migration patterns with comprehensive service support"""
    
    def __init__(self):
        # Base Network Connectivity Patterns
        self.network_patterns = {
            # San Jose Patterns
            'sj_nonprod_vpc_endpoint': {
                'name': 'San Jose Non-Prod → AWS VPC Endpoint',
                'source': 'San Jose',
                'environment': 'non-production',
                'pattern_type': 'vpc_endpoint',
                'segments': [
                    {'name': 'San Jose Internal LAN', 'bandwidth_mbps': 10000, 'latency_ms': 1, 'reliability': 0.999},
                    {'name': 'VPC Endpoint Connection', 'bandwidth_mbps': 2000, 'latency_ms': 8, 'reliability': 0.998}
                ],
                'total_bandwidth_mbps': 2000,
                'total_latency_ms': 9,
                'cost_factor': 1.5,
                'security_level': 'high',
                'protocol_overhead': 0.05,
                'network_congestion_factor': 0.9,
                'vpc_endpoint_limitations': {
                    'ipv4_only': True,
                    'no_shared_vpc': True,
                    'no_dedicated_tenancy': True,
                    'requires_4_network_interfaces': True,
                    'additional_security_groups': True,
                    'privatelink_routing_overhead': 0.03
                }
            },
            'sj_nonprod_direct_connect': {
                'name': 'San Jose Non-Prod → AWS Direct Connect',
                'source': 'San Jose',
                'environment': 'non-production',
                'pattern_type': 'direct_connect',
                'segments': [
                    {'name': 'San Jose Internal LAN', 'bandwidth_mbps': 10000, 'latency_ms': 1, 'reliability': 0.999},
                    {'name': 'Direct Connect (DX)', 'bandwidth_mbps': 2000, 'latency_ms': 12, 'reliability': 0.998}
                ],
                'total_bandwidth_mbps': 2000,
                'total_latency_ms': 13,
                'cost_factor': 2.0,
                'security_level': 'high',
                'protocol_overhead': 0.03,
                'network_congestion_factor': 0.95
            },
            'sj_prod_direct_connect': {
                'name': 'San Jose Production → AWS Direct Connect',
                'source': 'San Jose',
                'environment': 'production',
                'pattern_type': 'direct_connect',
                'segments': [
                    {'name': 'San Jose Internal LAN', 'bandwidth_mbps': 10000, 'latency_ms': 1, 'reliability': 0.999},
                    {'name': 'Production Direct Connect', 'bandwidth_mbps': 10000, 'latency_ms': 8, 'reliability': 0.9999}
                ],
                'total_bandwidth_mbps': 10000,
                'total_latency_ms': 9,
                'cost_factor': 3.5,
                'security_level': 'very_high',
                'protocol_overhead': 0.02,
                'network_congestion_factor': 0.98
            },
            # San Antonio Patterns
            'sa_prod_via_sj': {
                'name': 'San Antonio Production → San Jose → AWS',
                'source': 'San Antonio',
                'environment': 'production',
                'pattern_type': 'multi_hop',
                'segments': [
                    {'name': 'San Antonio Internal LAN', 'bandwidth_mbps': 10000, 'latency_ms': 1, 'reliability': 0.999},
                    {'name': 'SA to SJ Private Line', 'bandwidth_mbps': 10000, 'latency_ms': 12, 'reliability': 0.9995},
                    {'name': 'SJ to AWS Direct Connect', 'bandwidth_mbps': 10000, 'latency_ms': 8, 'reliability': 0.9999}
                ],
                'total_bandwidth_mbps': 10000,
                'total_latency_ms': 21,
                'cost_factor': 4.0,
                'security_level': 'very_high',
                'protocol_overhead': 0.04,
                'network_congestion_factor': 0.92
            }
        }
        
        # Service-Specific Configurations
        self.migration_services = {
            'datasync': {
                'name': 'AWS DataSync',
                'use_case': 'File and object data transfer',
                'protocols': ['NFS', 'SMB', 'HDFS', 'S3'],
                'vpc_endpoint_compatible': True,
                'encryption_in_transit': True,
                'encryption_at_rest': True,
                'bandwidth_efficiency': 0.92,
                'latency_sensitivity': 'medium',
                'sizes': {
                    'small': {
                        'vcpu': 2, 'memory_gb': 4, 'throughput_mbps': 250, 'cost_per_hour': 0.042,
                        'vpc_endpoint_throughput_reduction': 0.1,
                        'optimal_file_size_mb': '1-100',
                        'concurrent_transfers': 8
                    },
                    'medium': {
                        'vcpu': 2, 'memory_gb': 8, 'throughput_mbps': 500, 'cost_per_hour': 0.085,
                        'vpc_endpoint_throughput_reduction': 0.08,
                        'optimal_file_size_mb': '100-1000',
                        'concurrent_transfers': 16
                    },
                    'large': {
                        'vcpu': 4, 'memory_gb': 16, 'throughput_mbps': 1000, 'cost_per_hour': 0.17,
                        'vpc_endpoint_throughput_reduction': 0.05,
                        'optimal_file_size_mb': '1000+',
                        'concurrent_transfers': 32
                    },
                    'xlarge': {
                        'vcpu': 8, 'memory_gb': 32, 'throughput_mbps': 2000, 'cost_per_hour': 0.34,
                        'vpc_endpoint_throughput_reduction': 0.03,
                        'optimal_file_size_mb': '1000+',
                        'concurrent_transfers': 64
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
                'bandwidth_efficiency': 0.88,
                'latency_sensitivity': 'high',
                'requires_endpoints': True,
                'supports_cdc': True,
                'sizes': {
                    'small': {
                        'vcpu': 2, 'memory_gb': 4, 'throughput_mbps': 200, 'cost_per_hour': 0.042,
                        'max_connections': 50,
                        'optimal_table_size_gb': '1-10'
                    },
                    'medium': {
                        'vcpu': 2, 'memory_gb': 8, 'throughput_mbps': 400, 'cost_per_hour': 0.085,
                        'max_connections': 100,
                        'optimal_table_size_gb': '10-100'
                    },
                    'large': {
                        'vcpu': 4, 'memory_gb': 16, 'throughput_mbps': 800, 'cost_per_hour': 0.17,
                        'max_connections': 200,
                        'optimal_table_size_gb': '100-500'
                    },
                    'xlarge': {
                        'vcpu': 8, 'memory_gb': 32, 'throughput_mbps': 1500, 'cost_per_hour': 0.34,
                        'max_connections': 400,
                        'optimal_table_size_gb': '500+'
                    },
                    'xxlarge': {
                        'vcpu': 16, 'memory_gb': 64, 'throughput_mbps': 2500, 'cost_per_hour': 0.68,
                        'max_connections': 800,
                        'optimal_table_size_gb': '1000+'
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
                'bandwidth_efficiency': 0.95,
                'latency_sensitivity': 'low',
                'requires_active_directory': True,
                'supports_deduplication': True,
                'supports_compression': True,
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
                    },
                    'xlarge': {
                        'storage_gb': 65536, 'throughput_mbps': 2048, 'cost_per_hour': 1.60,
                        'iops': 24576, 'max_concurrent_users': 2000
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
                'bandwidth_efficiency': 0.98,
                'latency_sensitivity': 'very_low',
                'supports_s3_integration': True,
                'supports_scratch_persistent': True,
                'sizes': {
                    'small': {
                        'storage_gb': 1200, 'throughput_mbps': 240, 'cost_per_hour': 0.15,
                        'iops': 'unlimited', 'max_concurrent_clients': 100
                    },
                    'medium': {
                        'storage_gb': 2400, 'throughput_mbps': 480, 'cost_per_hour': 0.30,
                        'iops': 'unlimited', 'max_concurrent_clients': 200
                    },
                    'large': {
                        'storage_gb': 7200, 'throughput_mbps': 1440, 'cost_per_hour': 0.90,
                        'iops': 'unlimited', 'max_concurrent_clients': 500
                    },
                    'xlarge': {
                        'storage_gb': 50400, 'throughput_mbps': 10080, 'cost_per_hour': 6.30,
                        'iops': 'unlimited', 'max_concurrent_clients': 2000
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
                'bandwidth_efficiency': 0.85,
                'latency_sensitivity': 'medium',
                'supports_caching': True,
                'types': ['File Gateway', 'Volume Gateway', 'Tape Gateway'],
                'sizes': {
                    'small': {
                        'vcpu': 4, 'memory_gb': 16, 'throughput_mbps': 125, 'cost_per_hour': 0.05,
                        'cache_gb': 150, 'max_volumes': 32
                    },
                    'medium': {
                        'vcpu': 8, 'memory_gb': 32, 'throughput_mbps': 250, 'cost_per_hour': 0.10,
                        'cache_gb': 300, 'max_volumes': 64
                    },
                    'large': {
                        'vcpu': 16, 'memory_gb': 64, 'throughput_mbps': 500, 'cost_per_hour': 0.20,
                        'cache_gb': 600, 'max_volumes': 128
                    },
                    'xlarge': {
                        'vcpu': 32, 'memory_gb': 128, 'throughput_mbps': 1000, 'cost_per_hour': 0.40,
                        'cache_gb': 1200, 'max_volumes': 256
                    }
                }
            }
        }
    
    def determine_optimal_pattern(self, source_location: str, environment: str, migration_service: str) -> str:
        """Determine the optimal network pattern based on requirements and service"""
        if source_location == 'San Jose':
            if environment == 'production':
                return 'sj_prod_direct_connect'
            else:
                # VPC endpoint works well for most services except FSx
                if migration_service in ['fsx_windows', 'fsx_lustre']:
                    return 'sj_nonprod_direct_connect'
                else:
                    return 'sj_nonprod_vpc_endpoint'
        elif source_location == 'San Antonio':
            return 'sa_prod_via_sj'
        return 'sj_nonprod_direct_connect'
    
    def assess_service_compatibility(self, pattern_key: str, migration_service: str, service_size: str) -> Dict:
        """Assess compatibility and limitations for the selected service and network pattern"""
        pattern = self.network_patterns[pattern_key]
        service = self.migration_services[migration_service]
        service_spec = service['sizes'][service_size]
        
        compatibility_assessment = {
            'service_name': service['name'],
            'is_vpc_endpoint': pattern['pattern_type'] == 'vpc_endpoint',
            'vpc_endpoint_compatible': service.get('vpc_endpoint_compatible', False),
            'warnings': [],
            'performance_impacts': [],
            'requirements': [],
            'recommendations': []
        }
        
        # VPC Endpoint Compatibility Assessment
        if compatibility_assessment['is_vpc_endpoint']:
            if not compatibility_assessment['vpc_endpoint_compatible']:
                compatibility_assessment['warnings'].append(
                    f"{service['name']} does not support VPC Endpoints. Direct Connect recommended."
                )
            else:
                vpc_limitations = pattern.get('vpc_endpoint_limitations', {})
                
                if vpc_limitations.get('ipv4_only'):
                    compatibility_assessment['warnings'].append(
                        "VPC Endpoints only support IPv4 - IPv6 configurations not supported"
                    )
                
                if vpc_limitations.get('requires_4_network_interfaces') and migration_service == 'datasync':
                    compatibility_assessment['requirements'].append(
                        f"{service['name']} creates 4 network interfaces in your VPC"
                    )
                
                # Service-specific VPC endpoint impacts
                if migration_service == 'datasync':
                    vpc_reduction = service_spec.get('vpc_endpoint_throughput_reduction', 0.05)
                    compatibility_assessment['performance_impacts'].append(
                        f"Expected {vpc_reduction*100:.1f}% throughput reduction through VPC endpoint"
                    )
                elif migration_service == 'dms':
                    compatibility_assessment['performance_impacts'].append(
                        "DMS through VPC endpoint may increase replication lag"
                    )
                elif migration_service == 'storage_gateway':
                    compatibility_assessment['performance_impacts'].append(
                        "Storage Gateway caching efficiency may be reduced through VPC endpoint"
                    )
        
        # Service-specific requirements and limitations
        if migration_service == 'fsx_windows':
            compatibility_assessment['requirements'].append(
                "Requires Active Directory integration for authentication"
            )
            compatibility_assessment['requirements'].append(
                "SMB protocol requires ports 445, 135, and dynamic RPC ports"
            )
        elif migration_service == 'fsx_lustre':
            compatibility_assessment['requirements'].append(
                "Requires Lustre client installation on compute instances"
            )
            compatibility_assessment['requirements'].append(
                "High-performance networking recommended (SR-IOV, enhanced networking)"
            )
        elif migration_service == 'dms':
            compatibility_assessment['requirements'].append(
                "Requires source and target endpoint configuration"
            )
            compatibility_assessment['requirements'].append(
                "Source database requires binary logging (MySQL) or equivalent"
            )
        elif migration_service == 'storage_gateway':
            compatibility_assessment['requirements'].append(
                "Requires local cache storage allocation"
            )
            compatibility_assessment['requirements'].append(
                "Time synchronization between on-premises and AWS critical"
            )
        
        # Performance recommendations
        if service.get('latency_sensitivity') == 'very_low':
            if pattern['total_latency_ms'] > 10:
                compatibility_assessment['recommendations'].append(
                    f"Consider dedicated Direct Connect for {service['name']} - current latency {pattern['total_latency_ms']}ms may impact performance"
                )
        elif service.get('latency_sensitivity') == 'high':
            if pattern['total_latency_ms'] > 20:
                compatibility_assessment['recommendations'].append(
                    f"High latency detected ({pattern['total_latency_ms']}ms) - may impact {service['name']} performance"
                )
        
        return compatibility_assessment
    
    def calculate_service_throughput(self, pattern_key: str, migration_service: str, service_size: str, num_instances: int) -> Dict:
        """Calculate effective service throughput considering service-specific characteristics"""
        pattern = self.network_patterns[pattern_key]
        service = self.migration_services[migration_service]
        service_spec = service['sizes'][service_size]
        
        # Get base service capacity
        if 'throughput_mbps' in service_spec:
            base_service_throughput = service_spec['throughput_mbps'] * num_instances
        else:
            # For services without explicit throughput (like some FSx configs)
            base_service_throughput = 1000 * num_instances  # Default estimate
        
        # Apply service-specific efficiency
        bandwidth_efficiency = service.get('bandwidth_efficiency', 0.9)
        efficient_throughput = base_service_throughput * bandwidth_efficiency
        
        # Apply VPC endpoint reduction for compatible services
        if (pattern['pattern_type'] == 'vpc_endpoint' and 
            service.get('vpc_endpoint_compatible') and 
            migration_service == 'datasync'):
            vpc_reduction = service_spec.get('vpc_endpoint_throughput_reduction', 0.05)
            vpc_limitations = pattern.get('vpc_endpoint_limitations', {})
            privatelink_overhead = vpc_limitations.get('privatelink_routing_overhead', 0.03)
            
            vpc_adjusted_throughput = efficient_throughput * (1 - vpc_reduction - privatelink_overhead)
        else:
            vpc_adjusted_throughput = efficient_throughput
        
        # Apply scaling efficiency
        if num_instances == 1:
            scaling_efficiency = 1.0
        elif num_instances <= 3:
            scaling_efficiency = 0.95
        elif num_instances <= 5:
            scaling_efficiency = 0.90
        else:
            scaling_efficiency = 0.85
        
        effective_service_throughput = vpc_adjusted_throughput * scaling_efficiency
        network_bandwidth = pattern['total_bandwidth_mbps']
        effective_throughput = min(effective_service_throughput, network_bandwidth)
        
        # Calculate utilization
        network_utilization = (effective_throughput / network_bandwidth) * 100
        service_utilization = (effective_throughput / effective_service_throughput) * 100 if effective_service_throughput > 0 else 0
        
        return {
            'effective_throughput_mbps': effective_throughput,
            'network_bandwidth_mbps': network_bandwidth,
            'service_throughput_mbps': effective_service_throughput,
            'vpc_adjusted_throughput_mbps': vpc_adjusted_throughput,
            'base_service_throughput_mbps': base_service_throughput,
            'network_utilization_percent': network_utilization,
            'service_utilization_percent': service_utilization,
            'bottleneck': 'network' if effective_throughput == network_bandwidth else 'service',
            'scaling_efficiency': scaling_efficiency,
            'bandwidth_efficiency': bandwidth_efficiency,
            'latency_ms': pattern['total_latency_ms'],
            'vpc_impact_percent': ((efficient_throughput - vpc_adjusted_throughput) / efficient_throughput * 100) if efficient_throughput > 0 else 0,
            'service_efficiency_percent': bandwidth_efficiency * 100
        }
    
    def calculate_enhanced_bandwidth_waterfall(self, pattern_key: str, migration_service: str, service_size: str, num_instances: int) -> Dict:
        """Enhanced bandwidth waterfall analysis with service-specific considerations"""
        pattern = self.network_patterns[pattern_key]
        service = self.migration_services[migration_service]
        service_spec = service['sizes'][service_size]
        
        # Step 1: Theoretical Maximum
        theoretical_max = max([segment['bandwidth_mbps'] for segment in pattern['segments']])
        
        # Step 2: Network Path Limitation
        network_limitation = min([segment['bandwidth_mbps'] for segment in pattern['segments']])
        network_reduction = theoretical_max - network_limitation
        
        # Step 3: Service Capacity Limitation
        if 'throughput_mbps' in service_spec:
            total_service_capacity = service_spec['throughput_mbps'] * num_instances
        else:
            total_service_capacity = 1000 * num_instances
        
        service_limited_bandwidth = min(network_limitation, total_service_capacity)
        service_reduction = network_limitation - service_limited_bandwidth
        
        # Step 4: Service Bandwidth Efficiency
        bandwidth_efficiency = service.get('bandwidth_efficiency', 0.9)
        efficiency_adjusted_bandwidth = service_limited_bandwidth * bandwidth_efficiency
        efficiency_reduction = service_limited_bandwidth - efficiency_adjusted_bandwidth
        
        # Step 5: VPC Endpoint Specific Reductions
        vpc_endpoint_adjusted_bandwidth = efficiency_adjusted_bandwidth
        vpc_endpoint_reduction = 0
        
        if (pattern['pattern_type'] == 'vpc_endpoint' and 
            service.get('vpc_endpoint_compatible') and 
            migration_service == 'datasync'):
            vpc_throughput_reduction = service_spec.get('vpc_endpoint_throughput_reduction', 0.05)
            vpc_endpoint_adjusted_bandwidth *= (1 - vpc_throughput_reduction)
            
            vpc_limitations = pattern.get('vpc_endpoint_limitations', {})
            privatelink_overhead = vpc_limitations.get('privatelink_routing_overhead', 0.03)
            vpc_endpoint_adjusted_bandwidth *= (1 - privatelink_overhead)
            
            vpc_endpoint_reduction = efficiency_adjusted_bandwidth - vpc_endpoint_adjusted_bandwidth
        
        # Step 6: Scaling Efficiency Impact
        if num_instances == 1:
            scaling_efficiency = 1.0
        elif num_instances <= 3:
            scaling_efficiency = 0.95
        elif num_instances <= 5:
            scaling_efficiency = 0.90
        else:
            scaling_efficiency = 0.85
        
        scaling_adjusted_bandwidth = vpc_endpoint_adjusted_bandwidth * scaling_efficiency
        scaling_reduction = vpc_endpoint_adjusted_bandwidth - scaling_adjusted_bandwidth
        
        # Step 7: Protocol Overhead
        protocol_overhead = pattern.get('protocol_overhead', 0.03)
        protocol_adjusted_bandwidth = scaling_adjusted_bandwidth * (1 - protocol_overhead)
        protocol_reduction = scaling_adjusted_bandwidth - protocol_adjusted_bandwidth
        
        # Step 8: Network Congestion
        congestion_factor = pattern.get('network_congestion_factor', 0.95)
        final_effective_bandwidth = protocol_adjusted_bandwidth * congestion_factor
        congestion_reduction = protocol_adjusted_bandwidth - final_effective_bandwidth
        
        # Step 9: Service-specific Quality of Service adjustments
        if pattern['environment'] == 'production':
            qos_factor = 0.98
        else:
            qos_factor = 0.95
        
        # Additional QoS adjustment for latency-sensitive services
        if service.get('latency_sensitivity') == 'very_low':
            qos_factor *= 0.98
        elif service.get('latency_sensitivity') == 'high':
            qos_factor *= 0.99
        
        qos_adjusted_bandwidth = final_effective_bandwidth * qos_factor
        qos_reduction = final_effective_bandwidth - qos_adjusted_bandwidth
        
        # Build enhanced waterfall steps
        steps = [
            {'name': 'Theoretical Maximum', 'value': theoretical_max, 'cumulative': theoretical_max, 'type': 'positive'},
            {'name': 'Network Path Limit', 'value': -network_reduction, 'cumulative': network_limitation, 'type': 'negative'},
            {'name': f'{service["name"]} Capacity', 'value': -service_reduction, 'cumulative': service_limited_bandwidth, 'type': 'negative'},
            {'name': f'Service Efficiency ({bandwidth_efficiency*100:.0f}%)', 'value': -efficiency_reduction, 'cumulative': efficiency_adjusted_bandwidth, 'type': 'negative'}
        ]
        
        # Add VPC endpoint step if applicable
        if vpc_endpoint_reduction > 0:
            steps.append({'name': 'VPC Endpoint Overhead', 'value': -vpc_endpoint_reduction, 'cumulative': vpc_endpoint_adjusted_bandwidth, 'type': 'negative'})
        
        steps.extend([
            {'name': 'Scaling Efficiency', 'value': -scaling_reduction, 'cumulative': scaling_adjusted_bandwidth, 'type': 'negative'},
            {'name': 'Protocol Overhead', 'value': -protocol_reduction, 'cumulative': protocol_adjusted_bandwidth, 'type': 'negative'},
            {'name': 'Network Congestion', 'value': -congestion_reduction, 'cumulative': final_effective_bandwidth, 'type': 'negative'},
            {'name': 'Service QoS', 'value': -qos_reduction, 'cumulative': qos_adjusted_bandwidth, 'type': 'negative'},
            {'name': 'Final Effective', 'value': qos_adjusted_bandwidth, 'cumulative': qos_adjusted_bandwidth, 'type': 'total'}
        ])
        
        return {
            'steps': steps,
            'summary': {
                'theoretical_max_mbps': theoretical_max,
                'network_limited_mbps': network_limitation,
                'service_limited_mbps': service_limited_bandwidth,
                'efficiency_adjusted_mbps': efficiency_adjusted_bandwidth,
                'vpc_endpoint_adjusted_mbps': vpc_endpoint_adjusted_bandwidth,
                'final_effective_mbps': qos_adjusted_bandwidth,
                'total_reduction_mbps': theoretical_max - qos_adjusted_bandwidth,
                'service_efficiency_reduction_mbps': efficiency_reduction,
                'vpc_endpoint_reduction_mbps': vpc_endpoint_reduction,
                'efficiency_percentage': (qos_adjusted_bandwidth / theoretical_max) * 100,
                'primary_bottleneck': self._identify_enhanced_bottleneck(network_reduction, service_reduction, efficiency_reduction, vpc_endpoint_reduction, scaling_reduction, protocol_reduction, congestion_reduction, qos_reduction),
                'scaling_efficiency': scaling_efficiency,
                'bandwidth_efficiency': bandwidth_efficiency,
                'service_name': service['name'],
                'protocol_overhead_pct': protocol_overhead * 100,
                'congestion_impact_pct': (1 - congestion_factor) * 100,
                'qos_overhead_pct': (1 - qos_factor) * 100,
                'vpc_endpoint_impact_pct': (vpc_endpoint_reduction / theoretical_max) * 100 if vpc_endpoint_reduction > 0 else 0
            }
        }
    
    def _identify_enhanced_bottleneck(self, network_red, service_red, efficiency_red, vpc_red, scaling_red, protocol_red, congestion_red, qos_red) -> str:
        """Identify the primary bottleneck in the enhanced bandwidth waterfall"""
        reductions = {
            'Network Path': network_red,
            'Service Capacity': service_red,
            'Service Efficiency': efficiency_red,
            'VPC Endpoint': vpc_red,
            'Scaling Inefficiency': scaling_red,
            'Protocol Overhead': protocol_red,
            'Network Congestion': congestion_red,
            'Service QoS': qos_red
        }
        return max(reductions.items(), key=lambda x: x[1])[0]
    
    def estimate_migration_time(self, data_size_gb: int, effective_throughput_mbps: int, migration_service: str) -> Dict:
        """Enhanced migration time estimation with service-specific considerations"""
        data_size_gbits = data_size_gb * 8
        service = self.migration_services[migration_service]
        
        if effective_throughput_mbps > 0:
            migration_time_hours = data_size_gbits / (effective_throughput_mbps / 1000) / 3600
        else:
            migration_time_hours = float('inf')
        
        # Service-specific overhead calculations
        if migration_service == 'datasync':
            setup_overhead_hours = 1
            validation_overhead_hours = data_size_gb / 2000  # Faster validation
        elif migration_service == 'dms':
            setup_overhead_hours = 3  # More complex setup
            validation_overhead_hours = data_size_gb / 500   # More thorough validation
        elif migration_service in ['fsx_windows', 'fsx_lustre']:
            setup_overhead_hours = 2
            validation_overhead_hours = data_size_gb / 1000
        elif migration_service == 'storage_gateway':
            setup_overhead_hours = 2
            validation_overhead_hours = data_size_gb / 800   # Cache warming time
        else:
            setup_overhead_hours = 2
            validation_overhead_hours = data_size_gb / 1000
        
        total_migration_time = migration_time_hours + setup_overhead_hours + validation_overhead_hours
        
        # Service-specific recommendations for window sizing
        if migration_service == 'dms' and service.get('supports_cdc'):
            recommended_window_multiplier = 1.1  # CDC reduces downtime
        elif migration_service in ['fsx_windows', 'fsx_lustre']:
            recommended_window_multiplier = 1.3  # File system consistency checks
        else:
            recommended_window_multiplier = 1.2
        
        return {
            'data_transfer_hours': migration_time_hours,
            'setup_hours': setup_overhead_hours,
            'validation_hours': validation_overhead_hours,
            'total_hours': total_migration_time,
            'total_days': total_migration_time / 24,
            'recommended_window_hours': math.ceil(total_migration_time * recommended_window_multiplier),
            'service_name': service['name'],
            'supports_incremental': migration_service in ['dms', 'storage_gateway']
        }
    
    def generate_enhanced_ai_recommendation(self, config: Dict, analysis_results: Dict) -> Dict:
        """Enhanced AI-powered recommendations with service-specific insights"""
        data_size = config['data_size_gb']
        migration_service = config['migration_service']
        migration_time = analysis_results['migration_time']
        throughput_analysis = analysis_results['throughput_analysis']
        service_compatibility = analysis_results.get('service_compatibility', {})
        
        service = self.migration_services[migration_service]
        recommendations = []
        priority_score = 0
        
        # Service-specific compatibility recommendations
        if len(service_compatibility.get('warnings', [])) > 0:
            recommendations.append({
                'type': 'service_compatibility',
                'priority': 'high',
                'description': f'{service["name"]} has {len(service_compatibility["warnings"])} compatibility warnings with the selected network pattern.',
                'impact': 'Configuration changes required to ensure proper operation'
            })
            priority_score += 20
        
        # VPC Endpoint specific recommendations for different services
        if service_compatibility.get('is_vpc_endpoint'):
            if not service_compatibility.get('vpc_endpoint_compatible'):
                recommendations.append({
                    'type': 'vpc_endpoint_incompatible',
                    'priority': 'critical',
                    'description': f'{service["name"]} does not support VPC Endpoints. Direct Connect is required.',
                    'impact': 'Migration cannot proceed with current network configuration'
                })
                priority_score += 30
            else:
                vpc_impact = throughput_analysis.get('vpc_impact_percent', 0)
                if vpc_impact > 5:
                    recommendations.append({
                        'type': 'vpc_endpoint_performance',
                        'priority': 'medium',
                        'description': f'VPC Endpoint reduces {service["name"]} throughput by {vpc_impact:.1f}%. Consider Direct Connect for better performance.',
                        'impact': 'Moderate performance improvement with Direct Connect'
                    })
                    priority_score += 10
        
        # Service-specific performance recommendations
        if throughput_analysis['bottleneck'] == 'network':
            recommendations.append({
                'type': 'network_optimization',
                'priority': 'high',
                'description': f'Network bandwidth is limiting {service["name"]} performance. Consider upgrading connectivity or optimizing transfer patterns.',
                'impact': 'Significant throughput improvement possible'
            })
            priority_score += 20
        elif throughput_analysis['bottleneck'] == 'service':
            recommendations.append({
                'type': 'service_optimization',
                'priority': 'medium',
                'description': f'{service["name"]} capacity is the bottleneck. Consider scaling to more instances or larger sizes.',
                'impact': 'Moderate throughput improvement'
            })
            priority_score += 10
        
        # Service-specific latency recommendations
        latency_sensitivity = service.get('latency_sensitivity', 'medium')
        current_latency = throughput_analysis['latency_ms']
        
        if latency_sensitivity == 'very_low' and current_latency > 10:
            recommendations.append({
                'type': 'latency_optimization',
                'priority': 'high',
                'description': f'{service["name"]} is latency-sensitive. Current {current_latency}ms latency may impact performance. Consider dedicated Direct Connect.',
                'impact': 'Significant performance improvement for latency-sensitive workloads'
            })
            priority_score += 15
        elif latency_sensitivity == 'high' and current_latency > 20:
            recommendations.append({
                'type': 'latency_warning',
                'priority': 'medium',
                'description': f'High latency ({current_latency}ms) detected for {service["name"]}. Monitor performance closely.',
                'impact': 'Potential performance degradation'
            })
            priority_score += 8
        
        # Migration time recommendations
        if migration_time['total_hours'] > 72:
            if migration_service == 'dms' and service.get('supports_cdc'):
                recommendations.append({
                    'type': 'cdc_strategy',
                    'priority': 'medium',
                    'description': 'Long migration time detected. Leverage DMS Change Data Capture for minimal downtime migration.',
                    'impact': 'Significant downtime reduction'
                })
                priority_score += 12
            elif migration_service == 'storage_gateway':
                recommendations.append({
                    'type': 'incremental_strategy',
                    'priority': 'medium',
                    'description': 'Consider Storage Gateway caching strategy for gradual data migration.',
                    'impact': 'Reduced initial migration impact'
                })
                priority_score += 10
            else:
                recommendations.append({
                    'type': 'timeline_optimization',
                    'priority': 'high',
                    'description': f'Migration time exceeds 3 days for {service["name"]}. Consider parallel processing or staged migration.',
                    'impact': 'Significant time reduction possible'
                })
                priority_score += 15
        
        # Large data recommendations
        if data_size > 10000:
            if migration_service in ['fsx_windows', 'fsx_lustre']:
                recommendations.append({
                    'type': 'fsx_large_data',
                    'priority': 'high',
                    'description': f'Large dataset detected for {service["name"]}. Consider pre-populating from S3 or using backup/restore strategy.',
                    'impact': 'Faster initial data population'
                })
                priority_score += 18
            elif migration_service == 'datasync':
                recommendations.append({
                    'type': 'datasync_large_data',
                    'priority': 'medium',
                    'description': 'Large dataset for DataSync. Enable bandwidth throttling and schedule transfers during off-peak hours.',
                    'impact': 'Reduced network impact during migration'
                })
                priority_score += 12
        
        # Service-specific scaling recommendations
        service_efficiency = throughput_analysis.get('service_efficiency_percent', 90)
        if service_efficiency < 85:
            recommendations.append({
                'type': 'service_efficiency',
                'priority': 'medium',
                'description': f'{service["name"]} efficiency is {service_efficiency:.1f}%. Review configuration and optimization settings.',
                'impact': 'Better resource utilization'
            })
            priority_score += 8
        
        # Service-specific security recommendations
        if service.get('requires_active_directory') and migration_service == 'fsx_windows':
            recommendations.append({
                'type': 'security_integration',
                'priority': 'medium',
                'description': 'FSx for Windows requires Active Directory integration. Ensure proper domain trust and security groups.',
                'impact': 'Secure and seamless authentication'
            })
            priority_score += 10
        
        return {
            'recommendations': recommendations,
            'overall_priority_score': priority_score,
            'migration_complexity': 'high' if priority_score > 50 else 'medium' if priority_score > 25 else 'low',
            'confidence_level': 'high' if len(recommendations) <= 3 else 'medium',
            'service_name': service['name'],
            'service_type': migration_service
        }

def render_header():
    """Render enhanced application header"""
    st.markdown("""
    <div class="main-header">
        <h1>🌐 AWS Migration Network Pattern Analyzer</h1>
        <p style="font-size: 1.2rem; margin-top: 0.5rem;">
            Comprehensive Service Analysis • Network Pattern Optimization • VPC Endpoint Assessment • Performance Modeling
        </p>
        <p style="font-size: 0.9rem; margin-top: 0.5rem; opacity: 0.9;">
            DataSync • DMS • FSx Windows • FSx Lustre • Storage Gateway • VPC Endpoint • Direct Connect
        </p>
    </div>
    """, unsafe_allow_html=True)

def render_enhanced_sidebar_controls():
    """Render enhanced sidebar configuration controls with service selection"""
    st.sidebar.header("🔧 Migration Configuration")
    
    st.sidebar.subheader("📍 Source Location")
    source_location = st.sidebar.selectbox(
        "Data Center Location",
        ["San Jose", "San Antonio"],
        help="Select the source data center location"
    )
    
    environment = st.sidebar.selectbox(
        "Environment Type", 
        ["non-production", "production"],
        help="Production environments have dedicated high-bandwidth connections"
    )
    
    st.sidebar.subheader("🚀 Migration Service")
    migration_service = st.sidebar.selectbox(
        "AWS Migration Service",
        ["datasync", "dms", "fsx_windows", "fsx_lustre", "storage_gateway"],
        format_func=lambda x: {
            'datasync': '📁 AWS DataSync (File/Object Transfer)',
            'dms': '🗄️ AWS DMS (Database Migration)',
            'fsx_windows': '🪟 FSx for Windows (File Shares)',
            'fsx_lustre': '⚡ FSx for Lustre (HPC/ML)',
            'storage_gateway': '🔗 Storage Gateway (Hybrid)'
        }[x],
        help="Select the AWS service for your migration"
    )
    
    # Display service information
    analyzer = NetworkPatternAnalyzer()
    service_info = analyzer.migration_services[migration_service]
    
    st.sidebar.info(f"""
    **Service:** {service_info['name']}
    **Use Case:** {service_info['use_case']}
    **Protocols:** {', '.join(service_info['protocols'])}
    **VPC Endpoint Compatible:** {'✅' if service_info['vpc_endpoint_compatible'] else '❌'}
    """)
    
    st.sidebar.subheader("💾 Data Configuration")
    if migration_service == 'dms':
        source_database = st.sidebar.selectbox(
            "Source Database",
            ["MySQL", "PostgreSQL", "Oracle", "SQL Server", "MongoDB"],
            help="Source database engine"
        )
        
        target_database = st.sidebar.selectbox(
            "Target Database", 
            ["MySQL", "PostgreSQL", "Oracle", "SQL Server", "MongoDB"],
            help="Target database engine on AWS"
        )
        
        data_size_gb = st.sidebar.number_input(
            "Database Size (GB)",
            min_value=100,
            max_value=100000,
            value=1000,
            step=100,
            help="Total database size to migrate"
        )
        
        is_homogeneous = source_database == target_database
        migration_type = "Homogeneous" if is_homogeneous else "Heterogeneous"
        st.sidebar.info(f"**Migration Type:** {migration_type}")
        
    elif migration_service in ['fsx_windows', 'fsx_lustre']:
        data_size_gb = st.sidebar.number_input(
            "File System Size (GB)",
            min_value=100,
            max_value=100000,
            value=2000,
            step=100,
            help="Total file system size"
        )
        source_database = target_database = None
        is_homogeneous = True
        migration_type = "File System Migration"
        
    elif migration_service == 'storage_gateway':
        gateway_type = st.sidebar.selectbox(
            "Storage Gateway Type",
            ["File Gateway", "Volume Gateway", "Tape Gateway"],
            help="Type of Storage Gateway deployment"
        )
        
        data_size_gb = st.sidebar.number_input(
            "Storage Size (GB)",
            min_value=100,
            max_value=100000,
            value=5000,
            step=100,
            help="Total storage to migrate"
        )
        source_database = target_database = None
        is_homogeneous = True
        migration_type = f"Hybrid Storage - {gateway_type}"
        
    else:  # datasync
        data_size_gb = st.sidebar.number_input(
            "Data Size (GB)",
            min_value=100,
            max_value=100000,
            value=1000,
            step=100,
            help="Total data size to transfer"
        )
        source_database = target_database = None
        is_homogeneous = True
        migration_type = "File/Object Transfer"
    
    st.sidebar.success(f"✅ {migration_type}")
    
    st.sidebar.subheader("⚙️ Service Configuration")
    
    # Service size selection
    service_sizes = list(analyzer.migration_services[migration_service]['sizes'].keys())
    service_size = st.sidebar.selectbox(
        f"{service_info['name']} Size",
        service_sizes,
        index=1 if len(service_sizes) > 1 else 0,
        format_func=lambda x: f"{x.title()} - {analyzer.migration_services[migration_service]['sizes'][x].get('throughput_mbps', 'Variable')} {'Mbps' if 'throughput_mbps' in analyzer.migration_services[migration_service]['sizes'][x] else ''}",
        help=f"{service_info['name']} instance configuration"
    )
    
    # Number of instances
    if migration_service in ['fsx_windows', 'fsx_lustre']:
        num_instances = 1
        st.sidebar.info("FSx services are managed - single instance per file system")
    else:
        num_instances = st.sidebar.number_input(
            "Number of Instances",
            min_value=1,
            max_value=8,
            value=2,
            help="Number of parallel service instances"
        )
    
    st.sidebar.subheader("⏱️ Migration Requirements")
    max_downtime_hours = st.sidebar.number_input(
        "Maximum Downtime (hours)",
        min_value=1,
        max_value=168,
        value=24,
        help="Maximum acceptable downtime window"
    )
    
    return {
        'source_location': source_location,
        'environment': environment,
        'migration_service': migration_service,
        'source_database': source_database,
        'target_database': target_database,
        'data_size_gb': data_size_gb,
        'is_homogeneous': is_homogeneous,
        'migration_type': migration_type,
        'service_size': service_size,
        'num_instances': num_instances,
        'max_downtime_hours': max_downtime_hours
    }

def render_service_compatibility_analysis(config: Dict, analyzer: NetworkPatternAnalyzer):
    """Render service-specific compatibility analysis"""
    pattern_key = analyzer.determine_optimal_pattern(
        config['source_location'], 
        config['environment'], 
        config['migration_service']
    )
    
    service_compatibility = analyzer.assess_service_compatibility(
        pattern_key, config['migration_service'], config['service_size']
    )
    
    service_info = analyzer.migration_services[config['migration_service']]
    
    st.markdown(f"**🔍 {service_info['name']} Compatibility Analysis:**")
    
    # Service overview
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Service Type", config['migration_service'].upper())
    with col2:
        vpc_compatible = "✅ Yes" if service_compatibility['vpc_endpoint_compatible'] else "❌ No"
        st.metric("VPC Endpoint Support", vpc_compatible)
    with col3:
        latency_req = service_info.get('latency_sensitivity', 'medium').title()
        st.metric("Latency Sensitivity", latency_req)
    
    # Warnings
    if service_compatibility['warnings']:
        st.markdown("**🚨 Compatibility Warnings:**")
        for warning in service_compatibility['warnings']:
            st.warning(f"• {warning}")
    
    # Requirements
    if service_compatibility['requirements']:
        st.markdown("**📋 Service Requirements:**")
        for requirement in service_compatibility['requirements']:
            st.info(f"• {requirement}")
    
    # Performance Impacts
    if service_compatibility['performance_impacts']:
        st.markdown("**📉 Performance Considerations:**")
        for impact in service_compatibility['performance_impacts']:
            st.warning(f"• {impact}")
    
    # Recommendations
    if service_compatibility['recommendations']:
        st.markdown("**💡 Service-Specific Recommendations:**")
        for recommendation in service_compatibility['recommendations']:
            st.success(f"• {recommendation}")
    
    # Summary card
    compatibility_status = "⚠️ Compatibility Issues" if service_compatibility['warnings'] else "✅ Compatible"
    
    st.markdown(f"""
    <div class="service-card">
        <h4>📊 Service Compatibility Summary</h4>
        <p><strong>Service:</strong> {service_info['name']}</p>
        <p><strong>Network Pattern:</strong> {pattern_key.replace('_', ' ').title()}</p>
        <p><strong>Status:</strong> {compatibility_status}</p>
        <p><strong>Warnings:</strong> {len(service_compatibility['warnings'])}</p>
        <p><strong>Requirements:</strong> {len(service_compatibility['requirements'])}</p>
        <p><strong>Use Case:</strong> {service_info['use_case']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    return service_compatibility

def create_enhanced_network_diagram(pattern_key: str, migration_service: str, analyzer: NetworkPatternAnalyzer):
    """Create enhanced network path diagram with service-specific annotations"""
    pattern = analyzer.network_patterns[pattern_key]
    service_info = analyzer.migration_services[migration_service]
    
    fig = go.Figure()
    
    num_segments = len(pattern['segments'])
    x_positions = [i * 200 for i in range(num_segments + 1)]
    y_position = 50
    
    for i, segment in enumerate(pattern['segments']):
        line_width = max(3, min(12, segment['bandwidth_mbps'] / 200))
        reliability = segment['reliability']
        
        # Service-specific coloring
        if pattern['pattern_type'] == 'vpc_endpoint':
            if service_info['vpc_endpoint_compatible']:
                line_color = '#f59e0b'  # Orange for compatible VPC Endpoint
            else:
                line_color = '#ef4444'  # Red for incompatible
        elif reliability > 0.999:
            line_color = '#22c55e'
        elif reliability > 0.995:
            line_color = '#f59e0b'
        else:
            line_color = '#ef4444'
        
        fig.add_trace(go.Scatter(
            x=[x_positions[i], x_positions[i+1]],
            y=[y_position, y_position],
            mode='lines+markers',
            line=dict(width=line_width, color=line_color),
            marker=dict(size=12, color='#1e40af'),
            name=segment['name'],
            hovertemplate=f"""
            <b>{segment['name']}</b><br>
            Bandwidth: {segment['bandwidth_mbps']:,} Mbps<br>
            Latency: {segment['latency_ms']:.1f} ms<br>
            Reliability: {segment['reliability']*100:.3f}%<br>
            <extra></extra>
            """
        ))
        
        mid_x = (x_positions[i] + x_positions[i+1]) / 2
        
        # Enhanced annotation with service compatibility
        annotation_text = f"{segment['bandwidth_mbps']:,} Mbps<br>{segment['latency_ms']:.1f} ms"
        if pattern['pattern_type'] == 'vpc_endpoint':
            if service_info['vpc_endpoint_compatible']:
                annotation_text += f"<br>✅ {service_info['name']}"
            else:
                annotation_text += f"<br>❌ {service_info['name']}"
        
        fig.add_annotation(
            x=mid_x,
            y=y_position + 15,
            text=annotation_text,
            showarrow=False,
            font=dict(size=10),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='#e5e7eb',
            borderwidth=1
        )
    
    # Source and destination with service info
    fig.add_trace(go.Scatter(
        x=[x_positions[0]],
        y=[y_position],
        mode='markers+text',
        marker=dict(size=25, color='#059669', symbol='square'),
        text=[f"{pattern['source']}<br>{migration_service.upper()}"],
        textposition='bottom center',
        name='Source',
        hovertemplate=f"<b>Source: {pattern['source']}</b><br>Service: {service_info['name']}<extra></extra>"
    ))
    
    fig.add_trace(go.Scatter(
        x=[x_positions[-1]],
        y=[y_position], 
        mode='markers+text',
        marker=dict(size=25, color='#dc2626', symbol='square'),
        text=['AWS West-2'],
        textposition='bottom center',
        name='Destination',
        hovertemplate="<b>Destination: AWS West-2</b><extra></extra>"
    ))
    
    # Enhanced title with service compatibility
    compatibility_status = "✅" if service_info['vpc_endpoint_compatible'] or pattern['pattern_type'] != 'vpc_endpoint' else "❌"
    title = f"Network Path: {pattern['name']} | {service_info['name']} {compatibility_status}"
    
    fig.update_layout(
        title=title,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        showlegend=False,
        height=300,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

def create_enhanced_bandwidth_waterfall_chart(waterfall_data: Dict):
    """Create enhanced bandwidth waterfall chart with service-specific considerations"""
    steps = waterfall_data['steps']
    service_name = waterfall_data['summary']['service_name']
    
    fig = go.Figure()
    
    for i, step in enumerate(steps):
        if step['type'] == 'positive':
            fig.add_trace(go.Bar(
                x=[step['name']],
                y=[step['value']],
                marker_color='#22c55e',
                name=step['name'],
                text=[f"{step['value']:.0f} Mbps"],
                textposition='outside',
                hovertemplate=f"<b>{step['name']}</b><br>Bandwidth: {step['value']:.0f} Mbps<extra></extra>"
            ))
        elif step['type'] == 'total':
            fig.add_trace(go.Bar(
                x=[step['name']],
                y=[step['value']],
                marker_color='#1e40af',
                name=step['name'],
                text=[f"{step['value']:.0f} Mbps"],
                textposition='outside',
                hovertemplate=f"<b>{step['name']}</b><br>Final Bandwidth: {step['value']:.0f} Mbps<extra></extra>"
            ))
        else:
            # Service-specific coloring
            if 'Service Efficiency' in step['name']:
                color = '#8b5cf6'  # Purple for service efficiency
            elif 'VPC Endpoint' in step['name']:
                color = '#f59e0b'  # Orange for VPC endpoint
            elif service_name in step['name']:
                color = '#06b6d4'  # Cyan for service-specific
            else:
                color = '#ef4444'  # Red for other reductions
                
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
        title=f"Enhanced Bandwidth Waterfall Analysis: {service_name} Performance Impact",
        xaxis_title="Migration Pipeline Stages",
        yaxis_title="Bandwidth (Mbps)",
        showlegend=False,
        height=500,
        xaxis=dict(tickangle=45),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
        template="plotly_white"
    )
    
    return fig

def render_enhanced_network_analysis_tab(config: Dict, analyzer: NetworkPatternAnalyzer):
    """Render enhanced network analysis tab with comprehensive service considerations"""
    st.subheader("🌐 Enhanced Network Path Analysis")
    
    pattern_key = analyzer.determine_optimal_pattern(
        config['source_location'], 
        config['environment'], 
        config['migration_service']
    )
    
    pattern = analyzer.network_patterns[pattern_key]
    service_info = analyzer.migration_services[config['migration_service']]
    
    # Service compatibility assessment
    service_compatibility = render_service_compatibility_analysis(config, analyzer)
    
    throughput_analysis = analyzer.calculate_service_throughput(
        pattern_key,
        config['migration_service'],
        config['service_size'], 
        config['num_instances']
    )
    
    # Enhanced overview metrics
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric(
            "🚀 Migration Service",
            config['migration_service'].upper(),
            delta=service_info['name']
        )
    
    with col2:
        st.metric(
            "🌐 Network Pattern",
            pattern['pattern_type'].replace('_', ' ').title(),
            delta=f"From {config['source_location']}"
        )
    
    with col3:
        st.metric(
            "📡 Network Bandwidth", 
            f"{pattern['total_bandwidth_mbps']:,} Mbps",
            delta=f"Latency: {pattern['total_latency_ms']}ms"
        )
    
    with col4:
        service_efficiency = throughput_analysis.get('service_efficiency_percent', 90)
        st.metric(
            "⚙️ Service Efficiency",
            f"{service_efficiency:.1f}%",
            delta=f"{throughput_analysis['bandwidth_efficiency']*100:.0f}% base"
        )
    
    with col5:
        vpc_impact = throughput_analysis.get('vpc_impact_percent', 0)
        delta_text = f"VPC Impact: -{vpc_impact:.1f}%" if vpc_impact > 0 else f"Utilization: {throughput_analysis['network_utilization_percent']:.1f}%"
        st.metric(
            "⚡ Effective Throughput",
            f"{throughput_analysis['effective_throughput_mbps']:,.0f} Mbps",
            delta=delta_text
        )
    
    with col6:
        bottleneck = throughput_analysis['bottleneck']
        st.metric(
            "🎯 Bottleneck",
            bottleneck.title(),
            delta=f"Scaling: {throughput_analysis['scaling_efficiency']*100:.1f}%"
        )
    
    # Enhanced network path visualization
    st.markdown("**🗺️ Enhanced Network Path Visualization:**")
    network_diagram = create_enhanced_network_diagram(pattern_key, config['migration_service'], analyzer)
    st.plotly_chart(network_diagram, use_container_width=True)
    
    return {
        'pattern_key': pattern_key,
        'pattern': pattern,
        'throughput_analysis': throughput_analysis,
        'service_compatibility': service_compatibility,
        'service_info': service_info
    }

def render_enhanced_bandwidth_waterfall_tab(config: Dict, analyzer: NetworkPatternAnalyzer):
    """Render enhanced bandwidth waterfall analysis tab with service-specific considerations"""
    st.subheader("💧 Enhanced Bandwidth Waterfall Analysis")
    
    pattern_key = analyzer.determine_optimal_pattern(
        config['source_location'], 
        config['environment'], 
        config['migration_service']
    )
    
    waterfall_data = analyzer.calculate_enhanced_bandwidth_waterfall(
        pattern_key,
        config['migration_service'],
        config['service_size'], 
        config['num_instances']
    )
    
    service_info = analyzer.migration_services[config['migration_service']]
    
    # Enhanced summary metrics
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric(
            "🏁 Theoretical Max",
            f"{waterfall_data['summary']['theoretical_max_mbps']:,.0f} Mbps",
            delta="Starting point"
        )
    
    with col2:
        st.metric(
            "🚧 Network Limited",
            f"{waterfall_data['summary']['network_limited_mbps']:,.0f} Mbps",
            delta=f"-{waterfall_data['summary']['theoretical_max_mbps'] - waterfall_data['summary']['network_limited_mbps']:,.0f}"
        )
    
    with col3:
        st.metric(
            f"⚙️ {service_info['name']} Limited",
            f"{waterfall_data['summary']['service_limited_mbps']:,.0f} Mbps",
            delta=f"-{waterfall_data['summary']['network_limited_mbps'] - waterfall_data['summary']['service_limited_mbps']:,.0f}"
        )
    
    with col4:
        service_efficiency = waterfall_data['summary']['bandwidth_efficiency'] * 100
        st.metric(
            "📊 Service Efficiency",
            f"{service_efficiency:.0f}%",
            delta=f"-{waterfall_data['summary']['service_efficiency_reduction_mbps']:,.0f} Mbps"
        )
    
    with col5:
        vpc_impact = waterfall_data['summary'].get('vpc_endpoint_impact_pct', 0)
        if vpc_impact > 0:
            st.metric(
                "⚠️ VPC Endpoint Impact",
                f"{vpc_impact:.1f}%",
                delta=f"-{waterfall_data['summary']['vpc_endpoint_reduction_mbps']:,.0f} Mbps"
            )
        else:
            st.metric(
                "✅ No VPC Impact",
                "0%",
                delta="Direct Connect"
            )
    
    with col6:
        st.metric(
            "🎯 Final Effective",
            f"{waterfall_data['summary']['final_effective_mbps']:,.0f} Mbps",
            delta=f"{waterfall_data['summary']['efficiency_percentage']:.1f}% efficient"
        )
    
    # Enhanced waterfall chart
    st.markdown(f"**📊 Enhanced Bandwidth Degradation Analysis for {service_info['name']}:**")
    waterfall_chart = create_enhanced_bandwidth_waterfall_chart(waterfall_data)
    st.plotly_chart(waterfall_chart, use_container_width=True)
    
    # Enhanced breakdown with service-specific details
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔍 Detailed Impact Analysis:**")
        
        impact_data = []
        for step in waterfall_data['steps'][1:-1]:
            if step['value'] < 0:
                impact_data.append({
                    'Factor': step['name'],
                    'Reduction (Mbps)': f"{abs(step['value']):.0f}",
                    'Remaining (Mbps)': f"{step['cumulative']:.0f}",
                    'Impact (%)': f"{(abs(step['value']) / waterfall_data['summary']['theoretical_max_mbps']) * 100:.1f}%"
                })
        
        df_impact = pd.DataFrame(impact_data)
        st.dataframe(df_impact, use_container_width=True)
        
        # Enhanced insights with service-specific information
        insights_text = f"""
        <div class="waterfall-card">
            <h4>Enhanced Performance Insights</h4>
            <p><strong>Service:</strong> {service_info['name']}</p>
            <p><strong>Overall Efficiency:</strong> {waterfall_data['summary']['efficiency_percentage']:.1f}% of theoretical maximum</p>
            <p><strong>Total Bandwidth Loss:</strong> {waterfall_data['summary']['total_reduction_mbps']:,.0f} Mbps</p>
            <p><strong>Primary Bottleneck:</strong> {waterfall_data['summary']['primary_bottleneck']}</p>
            <p><strong>Service Bandwidth Efficiency:</strong> {waterfall_data['summary']['bandwidth_efficiency']*100:.1f}%</p>
            <p><strong>Protocol Overhead:</strong> {waterfall_data['summary']['protocol_overhead_pct']:.1f}%</p>
        """
        
        # Add service-specific insights
        if waterfall_data['summary'].get('vpc_endpoint_impact_pct', 0) > 0:
            insights_text += f"""
            <p><strong>VPC Endpoint Impact:</strong> {waterfall_data['summary']['vpc_endpoint_impact_pct']:.1f}% throughput reduction</p>
            <p><strong>Service Recommendation:</strong> Consider Direct Connect for {service_info['name']} optimal performance</p>
            """
        
        insights_text += "</div>"
        
        st.markdown(insights_text, unsafe_allow_html=True)
    
    with col2:
        st.markdown("**📈 Service Performance Breakdown:**")
        
        # Enhanced pie chart with service-specific breakdown
        labels = ['Effective Bandwidth']
        values = [waterfall_data['summary']['final_effective_mbps']]
        colors = ['#22c55e', '#8b5cf6', '#f59e0b', '#ef4444', '#06b6d4', '#f97316', '#84cc16']
        
        for i, step in enumerate(waterfall_data['steps'][1:-1]):
            if step['value'] < 0:
                labels.append(step['name'] + ' Loss')
                values.append(abs(step['value']))
        
        fig_pie = px.pie(
            values=values,
            names=labels,
            title=f"{service_info['name']} Bandwidth Allocation",
            color_discrete_sequence=colors
        )
        
        fig_pie.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Bandwidth: %{value:.0f} Mbps<br>Percentage: %{percent}<extra></extra>'
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Service-specific performance tips
        st.markdown(f"""
        <div class="service-card">
            <h4>📝 {service_info['name']} Performance Tips</h4>
            <p><strong>Protocols:</strong> {', '.join(service_info['protocols'])}</p>
            <p><strong>Latency Sensitivity:</strong> {service_info.get('latency_sensitivity', 'medium').title()}</p>
            <p><strong>Encryption:</strong> {'✅' if service_info.get('encryption_in_transit') else '❌'} In-Transit, {'✅' if service_info.get('encryption_at_rest') else '❌'} At-Rest</p>
            <p><strong>Use Case:</strong> {service_info['use_case']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    return waterfall_data

def render_enhanced_migration_timing_tab(config: Dict, network_analysis: Dict, analyzer: NetworkPatternAnalyzer):
    """Render enhanced migration timing analysis tab with service-specific considerations"""
    st.subheader("⏱️ Enhanced Migration Time Analysis")
    
    throughput_analysis = network_analysis['throughput_analysis']
    service_info = network_analysis['service_info']
    
    migration_time = analyzer.estimate_migration_time(
        config['data_size_gb'],
        throughput_analysis['effective_throughput_mbps'],
        config['migration_service']
    )
    
    # Enhanced time analysis metrics
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric(
            "📊 Data Size",
            f"{config['data_size_gb']:,} GB",
            delta=f"{config['data_size_gb'] * 8:,} Gbits"
        )
    
    with col2:
        st.metric(
            f"🚀 {service_info['name']} Speed",
            f"{throughput_analysis['effective_throughput_mbps']:,.0f} Mbps",
            delta=f"{throughput_analysis['effective_throughput_mbps']/8:.0f} MB/s"
        )
    
    with col3:
        st.metric(
            "⚙️ Setup Time",
            f"{migration_time['setup_hours']:.1f} hours",
            delta=f"{service_info['name']} specific"
        )
    
    with col4:
        st.metric(
            "🔄 Data Transfer Time",
            f"{migration_time['data_transfer_hours']:.1f} hours",
            delta=f"{migration_time['data_transfer_hours']/24:.1f} days"
        )
    
    with col5:
        st.metric(
            "✅ Validation Time",
            f"{migration_time['validation_hours']:.1f} hours",
            delta="Quality assurance"
        )
    
    with col6:
        meets_requirement = migration_time['total_hours'] <= config['max_downtime_hours']
        delta_text = "✅ Meets requirement" if meets_requirement else "❌ Exceeds limit"
        st.metric(
            "🎯 Total vs Limit",
            f"{migration_time['total_hours']:.1f}h / {config['max_downtime_hours']}h",
            delta=delta_text
        )
    
    # Service-specific timing insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📅 Migration Timeline Breakdown:**")
        
        timeline_data = [
            {"Phase": "Setup & Configuration", "Hours": migration_time['setup_hours'], "Percentage": (migration_time['setup_hours'] / migration_time['total_hours']) * 100},
            {"Phase": "Data Transfer", "Hours": migration_time['data_transfer_hours'], "Percentage": (migration_time['data_transfer_hours'] / migration_time['total_hours']) * 100},
            {"Phase": "Validation & Testing", "Hours": migration_time['validation_hours'], "Percentage": (migration_time['validation_hours'] / migration_time['total_hours']) * 100}
        ]
        
        df_timeline = pd.DataFrame(timeline_data)
        df_timeline['Hours'] = df_timeline['Hours'].round(1)
        df_timeline['Percentage'] = df_timeline['Percentage'].round(1)
        st.dataframe(df_timeline, use_container_width=True)
        
        # Service-specific features
        if migration_time.get('supports_incremental'):
            st.success(f"✅ {service_info['name']} supports incremental migration - downtime can be minimized")
        
        # Enhanced timeline visualization
        fig_timeline = px.bar(
            df_timeline,
            x='Phase',
            y='Hours',
            title=f"{service_info['name']} Migration Timeline",
            color='Phase',
            text='Hours'
        )
        fig_timeline.update_traces(texttemplate='%{text:.1f}h', textposition='outside')
        fig_timeline.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_timeline, use_container_width=True)
    
    with col2:
        st.markdown("**📊 Service Performance Analysis:**")
        
        # Performance metrics specific to the service
        if config['migration_service'] == 'datasync':
            service_spec = analyzer.migration_services['datasync']['sizes'][config['service_size']]
            perf_metrics = [
                {"Metric": "Concurrent Transfers", "Value": service_spec.get('concurrent_transfers', 'N/A')},
                {"Metric": "Optimal File Size", "Value": service_spec.get('optimal_file_size_mb', 'Variable')},
                {"Metric": "Protocol Efficiency", "Value": f"{throughput_analysis['bandwidth_efficiency']*100:.1f}%"},
                {"Metric": "VPC Endpoint Impact", "Value": f"{throughput_analysis.get('vpc_impact_percent', 0):.1f}%"}
            ]
        elif config['migration_service'] == 'dms':
            service_spec = analyzer.migration_services['dms']['sizes'][config['service_size']]
            perf_metrics = [
                {"Metric": "Max Connections", "Value": service_spec.get('max_connections', 'N/A')},
                {"Metric": "Optimal Table Size", "Value": service_spec.get('optimal_table_size_gb', 'Variable')},
                {"Metric": "CDC Support", "Value": "✅ Yes" if analyzer.migration_services['dms'].get('supports_cdc') else "❌ No"},
                {"Metric": "Replication Lag", "Value": "Low" if throughput_analysis['latency_ms'] < 15 else "Medium"}
            ]
        elif config['migration_service'] in ['fsx_windows', 'fsx_lustre']:
            service_spec = analyzer.migration_services[config['migration_service']]['sizes'][config['service_size']]
            perf_metrics = [
                {"Metric": "Storage Capacity", "Value": f"{service_spec.get('storage_gb', 'Variable'):,} GB"},
                {"Metric": "IOPS", "Value": service_spec.get('iops', 'Variable')},
                {"Metric": "Max Concurrent Users", "Value": service_spec.get('max_concurrent_users', service_spec.get('max_concurrent_clients', 'N/A'))},
                {"Metric": "Protocol Efficiency", "Value": f"{throughput_analysis['bandwidth_efficiency']*100:.1f}%"}
            ]
        elif config['migration_service'] == 'storage_gateway':
            service_spec = analyzer.migration_services['storage_gateway']['sizes'][config['service_size']]
            perf_metrics = [
                {"Metric": "Cache Size", "Value": f"{service_spec.get('cache_gb', 'Variable')} GB"},
                {"Metric": "Max Volumes", "Value": service_spec.get('max_volumes', 'N/A')},
                {"Metric": "Caching Support", "Value": "✅ Yes" if analyzer.migration_services['storage_gateway'].get('supports_caching') else "❌ No"},
                {"Metric": "Protocol Efficiency", "Value": f"{throughput_analysis['bandwidth_efficiency']*100:.1f}%"}
            ]
        
        df_perf = pd.DataFrame(perf_metrics)
        st.dataframe(df_perf, use_container_width=True)
        
        # Service-specific recommendations
        st.markdown(f"""
        <div class="service-card">
            <h4>⚡ {service_info['name']} Optimization Tips</h4>
            <p><strong>Latency Sensitivity:</strong> {service_info.get('latency_sensitivity', 'medium').title()}</p>
            <p><strong>Current Latency:</strong> {throughput_analysis['latency_ms']:.1f}ms</p>
            <p><strong>Bandwidth Efficiency:</strong> {throughput_analysis['bandwidth_efficiency']*100:.1f}%</p>
            <p><strong>Recommended Window:</strong> {migration_time['recommended_window_hours']:.1f} hours</p>
        </div>
        """, unsafe_allow_html=True)
    
    return migration_time

def render_enhanced_ai_recommendations_tab(config: Dict, network_analysis: Dict, migration_time: Dict, analyzer: NetworkPatternAnalyzer):
    """Render enhanced AI recommendations tab with comprehensive service-specific insights"""
    st.subheader("🤖 Enhanced AI-Powered Migration Recommendations")
    
    analysis_results = {
        'throughput_analysis': network_analysis['throughput_analysis'],
        'migration_time': migration_time,
        'pattern': network_analysis['pattern'],
        'service_compatibility': network_analysis.get('service_compatibility', {}),
        'service_info': network_analysis['service_info']
    }
    
    ai_recommendations = analyzer.generate_enhanced_ai_recommendation(config, analysis_results)
    
    # Enhanced AI Analysis Overview
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "🎯 Migration Complexity",
            ai_recommendations['migration_complexity'].title(),
            delta=f"Priority Score: {ai_recommendations['overall_priority_score']}"
        )
    
    with col2:
        st.metric(
            "🤖 AI Confidence",
            ai_recommendations['confidence_level'].title(),
            delta=f"{len(ai_recommendations['recommendations'])} recommendations"
        )
    
    with col3:
        st.metric(
            "🚀 Service Type",
            ai_recommendations['service_type'].upper(),
            delta=ai_recommendations['service_name']
        )
    
    with col4:
        bottleneck = analysis_results['throughput_analysis']['bottleneck']
        st.metric(
            "🔍 Primary Bottleneck",
            bottleneck.title(),
            delta="Optimization target"
        )
    
    with col5:
        optimal_instances = min(8, max(1, int(config['max_downtime_hours'] / migration_time['total_hours'] * config['num_instances'])))
        st.metric(
            "⚙️ Optimal Instances",
            f"{optimal_instances}",
            delta=f"Current: {config['num_instances']}"
        )
    
    # Enhanced AI Recommendations with service-specific categorization
    st.markdown("**💡 Enhanced AI-Generated Recommendations:**")
    
    # Group recommendations by type for better organization
    critical_recs = [rec for rec in ai_recommendations['recommendations'] if rec['priority'] == 'critical']
    high_recs = [rec for rec in ai_recommendations['recommendations'] if rec['priority'] == 'high']
    medium_recs = [rec for rec in ai_recommendations['recommendations'] if rec['priority'] == 'medium']
    low_recs = [rec for rec in ai_recommendations['recommendations'] if rec['priority'] == 'low']
    
    # Critical recommendations
    if critical_recs:
        st.markdown("### 🚨 Critical Issues")
        for i, rec in enumerate(critical_recs, 1):
            with st.expander(f"🔴 {rec['type'].replace('_', ' ').title()}", expanded=True):
                st.error(f"**Description:** {rec['description']}")
                st.write(f"**Expected Impact:** {rec['impact']}")
                st.write(f"**Priority Level:** {rec['priority'].title()}")
    
    # High priority recommendations
    if high_recs:
        st.markdown("### ⚠️ High Priority")
        for i, rec in enumerate(high_recs, 1):
            with st.expander(f"🟠 {rec['type'].replace('_', ' ').title()}", expanded=True):
                st.warning(f"**Description:** {rec['description']}")
                st.write(f"**Expected Impact:** {rec['impact']}")
                st.write(f"**Priority Level:** {rec['priority'].title()}")
    
    # Medium priority recommendations
    if medium_recs:
        st.markdown("### 📋 Medium Priority")
        for i, rec in enumerate(medium_recs, 1):
            with st.expander(f"🟡 {rec['type'].replace('_', ' ').title()}", expanded=False):
                st.info(f"**Description:** {rec['description']}")
                st.write(f"**Expected Impact:** {rec['impact']}")
                st.write(f"**Priority Level:** {rec['priority'].title()}")
    
    # Low priority recommendations
    if low_recs:
        st.markdown("### ✅ Low Priority")
        for i, rec in enumerate(low_recs, 1):
            with st.expander(f"🟢 {rec['type'].replace('_', ' ').title()}", expanded=False):
                st.success(f"**Description:** {rec['description']}")
                st.write(f"**Expected Impact:** {rec['impact']}")
                st.write(f"**Priority Level:** {rec['priority'].title()}")
    
    # Enhanced summary with service-specific insights
    service_info = analysis_results['service_info']
    
    st.markdown("### 📊 Migration Strategy Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="recommendation-card">
            <h4>🎯 Recommended Migration Strategy</h4>
            <p><strong>Service:</strong> {service_info['name']}</p>
            <p><strong>Complexity:</strong> {ai_recommendations['migration_complexity'].title()}</p>
            <p><strong>Confidence:</strong> {ai_recommendations['confidence_level'].title()}</p>
            <p><strong>Timeline:</strong> {migration_time['total_hours']:.1f} hours ({migration_time['total_days']:.1f} days)</p>
            <p><strong>Recommended Window:</strong> {migration_time['recommended_window_hours']:.1f} hours</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="service-card">
            <h4>⚙️ Service-Specific Considerations</h4>
            <p><strong>Use Case:</strong> {service_info['use_case']}</p>
            <p><strong>Protocols:</strong> {', '.join(service_info['protocols'])}</p>
            <p><strong>VPC Endpoint Compatible:</strong> {'✅' if service_info['vpc_endpoint_compatible'] else '❌'}</p>
            <p><strong>Encryption:</strong> In-Transit: {'✅' if service_info.get('encryption_in_transit') else '❌'}, At-Rest: {'✅' if service_info.get('encryption_at_rest') else '❌'}</p>
            <p><strong>Latency Sensitivity:</strong> {service_info.get('latency_sensitivity', 'medium').title()}</p>
        </div>
        """, unsafe_allow_html=True)
    
    return ai_recommendations

# Main application
def main():
    render_header()
    
    # Initialize analyzer
    analyzer = NetworkPatternAnalyzer()
    
    # Enhanced sidebar configuration
    config = render_enhanced_sidebar_controls()
    
    # Enhanced main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌐 Network Analysis", 
        "💧 Bandwidth Waterfall",
        "⏱️ Migration Timing", 
        "🤖 AI Recommendations"
    ])
    
    with tab1:
        network_analysis = render_enhanced_network_analysis_tab(config, analyzer)
    
    with tab2:
        waterfall_data = render_enhanced_bandwidth_waterfall_tab(config, analyzer)
    
    with tab3:
        migration_time = render_enhanced_migration_timing_tab(config, network_analysis, analyzer)
    
    with tab4:
        render_enhanced_ai_recommendations_tab(config, network_analysis, migration_time, analyzer)

if __name__ == "__main__":
    main()