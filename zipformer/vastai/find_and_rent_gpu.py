#!/usr/bin/env python3
"""
Auto Rent Vast.ai GPU Instance
Automatically finds and rents the cheapest GPU matching your criteria
"""
import time
from vast_api import (
    get_api_key, 
    get_template_id_by_name,
    rent_instance,
    wait_for_startup,
    search_offers,
    # format_offer_for_display,
    display_offers
)

def find_cheapest_offer(gpu, max_price, offer_type="bid"):
    """
    Find the cheapest GPU offer
    
    Args:
        gpu (str): GPU model to search for
        max_price (float): Maximum price per hour
        offer_type (str): Pricing type ("bid" for interruptible, "on-demand", or "reserved")
        
    Returns:
        dict: Offer data, or None if not found
    """
    print(f"[*] Searching for cheapest {offer_type} offer for {gpu} < ${max_price}...")
    
    # Build query list
    #https://cloud.vast.ai/create/?instanceDiskSizeMin=10.056106996174615&instanceDurationMin=3600.000000000001&rentalType=interruptible&isAvxSupported=false&isHostSecure=false&isMachineIpStatic=false&isOfferAvailable=true&isOfferCompatible=true&isOfferVerified=true&isQueryInverted=false&machineCpuCoresMin=8.224910613248532&machineCpuRamMin=7912.950406552351&machineDiskBandwidthMin=5404.704402525769&machineMegabitDownloadMin=1217.7480857627863&machineMegabitUploadMin=1260.691879265195&machineReliabilityMin=0.11307956328284252&offerGpuNumMin=1&sorts=priceInstanceHourly-asc%2CofferTflops-asc
    #https://cloud.vast.ai/create/?instanceDiskSizeMin=10.056106996174615&instanceDurationMin=3600.000000000001&rentalType=interruptible&isAvxSupported=false&isHostSecure=false&isMachineIpStatic=false&isOfferAvailable=true&isOfferCompatible=true&isOfferVerified=true&isQueryInverted=false&machineCpuCoresMin=8.224910613248532&machineCpuRamMin=7912.950406552351&machineDiskBandwidthMin=5404.704402525769&machineMegabitDownloadMin=1217.7480857627863&machineMegabitUploadMin=1260.691879265195&machineReliabilityMin=0.9978015440369574&offerGpuNumMin=1&sorts=priceInstanceHourly-asc%2CofferTflops-asc
    # https://cloud.vast.ai/create/?instanceDiskSizeMin=30.060000000000052&instanceDurationMin=2602163.171511475&rentalType=interruptible&isAvxSupported=false&isHostSecure=false&isMachineIpStatic=false&isOfferAvailable=true&isOfferCompatible=true&isOfferVerified=true&isQueryInverted=false&machineCpuCoresMin=7.889861635946874&machineCpuRamMin=8192&machineDiskBandwidthMin=5404.704402525769&machineMegabitDownloadMin=1217.7480857627863&machineMegabitUploadMin=1260.691879265195&machineReliabilityMin=0.9972052147249632&offerGpuNumMin=1&priceInstanceTerabyteDownloadMax=0.17099191298781136&priceInstanceTerabyteUploadMax=0.13584185781575728&sorts=priceInstanceHourly-asc%2CofferTflops-asc
    query_list = [
        *([] if not gpu else [f"gpu_name={gpu}"]),
        "rented=false",
        "verified=true",
        f"dph<{max_price}",
        # Filters from the vast.ai URL
        "disk_space>=10.056106996174615",  # instanceDiskSizeMin
        "duration>=31",     # instanceDurationMin (in days)
        "rentable=true",                   # isOfferAvailable
        "cpu_cores_effective>=8",  # machineCpuCoresMin
        "cpu_ram>=8",      # machineCpuRamMin (in gigabytes)
        "inet_down>=1217.7480857627863",   # machineMegabitDownloadMin (Mbps)
        "inet_up>=1260.691879265195",      # machineMegabitUploadMin (Mbps)
        "reliability>=0.11307956328284252", # machineReliabilityMin (0-1 scale)
        "num_gpus>=1",                     # offerGpuNumMin
        "disk_bw>=5404.704402525769",      # machineDiskBandwidthMin (MB/s)
        "inet_down_cost<=0.17099191298781136", # priceInstanceTerabyteDownloadMax
        "inet_up_cost<=0.13584185781575728", # priceInstanceTerabyteUploadMax
    ]
    
    # search_args = VastSearchOffersArgs(query_list, order="dph+")
    # vast.ARGS = search_args
    
    try:
        offers = search_offers(query_list=query_list, order="dph+", offer_type=offer_type)
        
        if offers and len(offers) > 0:
            cheapest = offers[0]
            
            if offers is None:
                return
            
            display_offers(offers)


            return cheapest
        else:
            print("    -> No matching offers found.")
            return None
    except Exception as e:
        print(f"    -> Error searching for offers: {e}")
        import traceback
        traceback.print_exc()
        return None

# --- KONFIGURASI ---
GPU_MODEL = None#"RTX_3090"
MAX_PRICE = 0.321        # DPH (Dollar per hour)
DISK_SPACE = 150         
IMAGE_TAG = "K2FSA"    # Nama Docker image yang dicari
OFFER_TYPE = "on-demand"     # "bid" (interruptible) atau "on-demand"
# IMAGE_TAG = "pytorch/pytorch"
# -------------------
#https://cloud.vast.ai/create/?instanceDiskSizeMin=10.056106996174615&instanceDurationMin=3600.000000000001&rentalType=interruptible&isAvxSupported=false&isHostSecure=false&isMachineIpStatic=false&isOfferAvailable=true&isOfferCompatible=true&isOfferVerified=true&isQueryInverted=false&machineCpuCoresMin=8.224910613248532&machineCpuRamMin=7912.950406552351&machineDiskBandwidthMin=5404.704402525769&machineMegabitDownloadMin=1217.7480857627863&machineMegabitUploadMin=1260.691879265195&machineReliabilityMin=0.11307956328284252&offerGpuNumMin=1&sorts=priceInstanceHourly-asc%2CofferTflops-asc

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    try:
        # # Validasi API key tersedia
        # print("[*] Memeriksa VAST_API_KEY...")
        # get_api_key()
        # print("    -> API key ditemukan!")
        
        # 1. Cari/Buat Template ID dulu
        templ_id = get_template_id_by_name(IMAGE_TAG)
        
        if templ_id:
            # 2. Cari GPU Murah
            offer = find_cheapest_offer(GPU_MODEL, MAX_PRICE, offer_type=OFFER_TYPE)
            
            # if offer:
            #     # 3. Sewa menggunakan ID Offer DAN ID Template
            #     instance_id = rent_instance(offer['id'], templ_id, DISK_SPACE)
                
            #     if instance_id:
            #         # 4. Tunggu hingga running
            #         wait_for_startup(instance_id)
        else:
            print("[!] Script berhenti karena masalah Template.")
    
    except ValueError as e:
        print(f"\n[ERROR] {e}")
        print("\nCara set environment variable:")
        print("  export VAST_API_KEY='your-api-key-here'")
    except KeyboardInterrupt:
        print("\n\n[!] Script dihentikan oleh user.")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()