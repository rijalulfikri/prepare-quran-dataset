#!/usr/bin/env python3
"""
Vast.ai CLI Management Tool
Provides command-line interface for managing Vast.ai machines
"""
import argparse
from vast_api import (
    get_instances,
    search_offers,
    format_instance_for_display,
    display_offers
)


def display_instances(instances):
    """Display instances in a formatted table
    
    Args:
        instances (list): List of instance dictionaries from Vast.ai API
    """
    if not instances:
        print("No running instances found.")
        return
    
    # Header
    header = f"{'ID':<10} {'Status':<12} {'GPU':<20} {'Num':<4} {'$/Hr':<10} {'Duration':<12} {'Machine':<10} {'Label':<20}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    
    # Format and display each instance
    for inst in instances:
        row = format_instance_for_display(inst)
        print(f"{row['id']:<10} {row['status']:<12} {row['gpu']:<20} {row['num_gpus']:<4} "
              f"{row['price']:<10} {row['duration']:<12} {row['machine']:<10} {row['label']:<20}")

def show_instances():
    """Main function to fetch and display currently running instances"""
    print("Fetching running instances...")
    
    instances = get_instances()
    if instances is None:
        return
    
    display_instances(instances)


def show_offers(query_list=None, order="score-", limit=None):
    """Main function to search and display available offers
    
    Args:
        query_list (list): List of query filters
        order (str): Order by field
        limit (int): Limit number of results
    """
    print("Searching for available machines...")
    
    offers = search_offers(query_list=query_list, order=order, limit=limit)
    if offers is None:
        return
    
    display_offers(offers)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="Manage Vast.ai machines")
    parser.add_argument(
        "--mode", 
        choices=["search", "instances"], 
        default="search", 
        help="Mode: 'search' for available offers, 'instances' for running instances (default: search)"
    )
    parser.add_argument(
        "query", 
        nargs="*", 
        help="Additional query filters for search mode (e.g. 'gpu_name=A100' 'geolocation=US')"
    )
    parser.add_argument(
        "--order", 
        default="dph", 
        help="Order by field for search mode (default: score-)"
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=None, 
        help="Limit number of results"
    )
    
    args_cli = parser.parse_args()

    if args_cli.mode == "instances":
        show_instances()
    else:
        gpu, max_price = "RTX_3090", 0.3
        query_list = [
            # f"gpu_name={gpu}",
            # f"dph<{max_price}",
            "verified=true",
            "rented=false",
            # Filters from the vast.ai URL
            "cuda_vers>=12.2",
            "disk_space>=30.056106996174615",  # instanceDiskSizeMin
            "duration>=30.000000000001",     # instanceDurationMin (in days)
            "rentable=true",                   # isOfferAvailable
            "cpu_cores_effective>=8",  # machineCpuCoresMin
            "cpu_ram>=8",      # machineCpuRamMin (in gigabytes)
            "num_gpus>=1",                     # offerGpuNumMin
            # "dlperf>=10",                     # dlperfMin
            "inet_down>=1217.7480857627863",   # machineMegabitDownloadMin (Mbps)
            "inet_up>=1260.691879265195",      # machineMegabitUploadMin (Mbps)
            "reliability>=0.9978015440369574", # machineReliabilityMin (0-1 scale)
            "disk_bw>=5404.704402525769",      # machineDiskBandwidthMin (MB/s)
            "min_bid>=0.219",                     
            # "verified=true",
            # "type=interruptible"


        ]
        show_offers(
            # query_list=list(args_cli.query), 
            query_list=query_list,
            order=args_cli.order, 
            limit=args_cli.limit
        )


if __name__ == "__main__":
    main()
