#!/usr/bin/env python3
"""
Vast.ai API Library
Provides core functionality for interacting with Vast.ai API.

Environment Variables:
    VAST_API_KEY  - Vast.ai API key (required for all operations; falls back to
                    the key file written by ``vastai set api-key``)
    VAST_LABEL    - Instance label filter used by get_instance_ids_by_label()
"""
import json
import os
import time
import traceback

import requests
import vast
from vast import APIKEY_FILE  # also re-exported for callers that need it

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_VAST_BASE_URL = "https://console.vast.ai"
_DEFAULT_RETRY = 3


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------


def get_env_variable(var_name: str, *, required: bool = True, default=None) -> str:
    """Get an environment variable value with optional validation.

    Args:
        var_name: Name of the environment variable to retrieve.
        required: If ``True``, raise :exc:`ValueError` when the variable is
            absent or empty.  If ``False``, return *default* instead.
        default: Fallback value returned when *required* is ``False`` and the
            variable is not set.

    Returns:
        The stripped string value of the variable, or *default*.

    Raises:
        ValueError: When *required* is ``True`` and the variable is absent or
            empty.

    Examples:
        >>> api_key = get_env_variable('VAST_API_KEY')
        >>> hf_token = get_env_variable('HF_TOKEN', required=False, default='')
    """
    value = os.getenv(var_name)

    if value and value.strip():
        return value.strip()

    if required:
        raise ValueError(
            f"Environment variable '{var_name}' is required but not set or empty. "
            f"Please set it using: export {var_name}='your-value'"
        )

    return default


def get_api_key() -> str:
    """Return the Vast.ai API key.

    Reads the ``VAST_API_KEY`` environment variable, falling back to the key
    file written by ``vastai set api-key`` (:data:`vast.APIKEY_FILE`).

    Raises:
        ValueError: If neither source provides a key.
    """
    api_key = get_env_variable("VAST_API_KEY", required=False)
    if api_key:
        return api_key

    if os.path.isfile(APIKEY_FILE):
        with open(APIKEY_FILE, encoding="utf-8") as f:
            api_key = f.read().strip()
        if api_key:
            return api_key

    raise ValueError(
        f"Vast.ai API key not found in VAST_API_KEY or {APIKEY_FILE}. "
        "Set it using: export VAST_API_KEY='your-value' or vastai set api-key 'your-value'"
    )


# ---------------------------------------------------------------------------
# Argparse-namespace mocks (passed directly to vast.*() entry-points)
# ---------------------------------------------------------------------------


class _VastBaseArgs:
    """Shared defaults for all Vast argparse-namespace mocks."""

    def __init__(self) -> None:
        self.raw = True
        self.explain = False
        self.retry = _DEFAULT_RETRY
        self.curl = False
        self.url = _VAST_BASE_URL
        self.api_key = get_api_key()


class VastSearchOffersArgs(_VastBaseArgs):
    """Mock argparse.Namespace for ``vast.search__offers``."""

    def __init__(
        self,
        query_list=None,
        order: str = "score-",
        limit=None,
        offer_type: str = "on-demand",
    ) -> None:
        super().__init__()
        self.query = query_list if query_list else []
        self.no_default = False
        self.type = offer_type
        self.order = order
        self.limit = limit
        self.storage = 5.0
        self.disable_bundling = False
        self.new = False


class VastInstanceArgs(_VastBaseArgs):
    """Mock argparse.Namespace for ``vast.show__instances``."""

    def __init__(self) -> None:
        super().__init__()
        self.quiet = False


class VastDestroyArgs(_VastBaseArgs):
    """Mock argparse.Namespace for ``vast.destroy__instance``."""

    def __init__(self, instance_id: int) -> None:
        super().__init__()
        self.id = instance_id


class VastSearchTemplateArgs(_VastBaseArgs):
    """Mock argparse.Namespace for ``vast.search__templates``."""

    def __init__(self, name_query: str) -> None:
        super().__init__()
        self.raw = False
        self.query = [f"name={name_query}"]


class VastCreateTemplateArgs(_VastBaseArgs):
    """Mock argparse.Namespace for ``vast.create__template``."""

    def __init__(self, image_name: str, template_name: str) -> None:
        super().__init__()
        self.raw = False
        self.image = image_name
        self.name = template_name


class VastCreateInstanceArgs(_VastBaseArgs):
    """Mock argparse.Namespace for ``vast.create__instance`` (template-based)."""

    def __init__(self, offer_id: int, template_id, disk_space: float) -> None:
        super().__init__()
        self.id = offer_id
        self.image = str(template_id)
        self.disk = disk_space
        self.onstart = "bash"
        self.label = None
        self.env = []
        self.args = []
        self.jupyter = False
        self.direct = False
        self.force = False


class CreateInstanceArgs(_VastBaseArgs):
    """Mock argparse.Namespace for ``vast.create__instance`` (direct Docker image).

    Unlike :class:`VastCreateInstanceArgs`, which uses a template ID, this
    class accepts a Docker image reference directly (e.g.
    ``pytorch/pytorch:2.7.1-...``) together with ``--onstart-cmd``, ``--ssh``,
    ``--direct``, and ``--env`` parameters.
    """

    def __init__(
        self,
        offer_id: int,
        image: str,
        env: str = "",
        onstart_cmd: str = None,
        disk: float = 20.0,
        ssh: bool = True,
        direct: bool = True,
        label: str = None,
    ) -> None:
        super().__init__()

        # Required
        self.id = offer_id
        self.image = image
        self.disk = disk

        # Env variables & port mapping (raw string processed by vast's parse_env())
        self.env = env

        # Startup script as a string
        self.onstart_cmd = onstart_cmd
        self.onstart = None        # File path (unused; using onstart_cmd instead)
        self.entrypoint = None

        # Launch type
        self.ssh = ssh
        self.direct = direct
        self.jupyter = False
        self.jupyter_lab = False
        self.jupyter_dir = None

        # Optional
        self.label = label
        self.login = None          # Docker login for private registries
        self.extra = None
        self.args = None
        self.force = False
        self.cancel_unavail = False
        self.bid_price = None      # None = on-demand
        self.python_utf8 = False
        self.lang_utf8 = False
        self.user = None
        self.template_hash = None  # None = use image directly (no template)

        # Volume (unused)
        self.create_volume = None
        self.link_volume = None
        self.volume_size = None
        self.mount_path = None
        self.volume_label = None


# ---------------------------------------------------------------------------
# Core API functions
# ---------------------------------------------------------------------------


def create_instance(
    offer_id: int,
    image: str,
    env: str = "",
    onstart_cmd: str = None,
    disk: float = 20.0,
    ssh: bool = True,
    direct: bool = True,
    label: str = None,
):
    """Create a Vast.ai instance from a Docker image (without a template).

    Args:
        offer_id:    Offer ID from ``vastai search offers``.
        image:       Docker image to use (e.g. ``pytorch/pytorch:2.7.1-...``).
        env:         Env-variable / port-mapping string (Docker CLI format).
                     Example: ``'-e DATA_DIRECTORY=/workspace/ -e JUPYTER_DIR=/'``
        onstart_cmd: Startup command executed when the container first starts.
        disk:        Local disk size in GB (default: 20).
        ssh:         ``True`` to enable SSH access (default: ``True``).
        direct:      ``True`` for faster direct SSH/Jupyter connections (default: ``True``).
        label:       Optional label for the instance.

    Returns:
        int | None: Contract/instance ID on success, ``None`` on failure.
    """
    print(f"[*] Creating instance from offer ID: {offer_id}")
    print(f"    Image    : {image}")
    print(f"    Disk     : {disk} GB")
    print(f"    SSH      : {ssh}")
    print(f"    Direct   : {direct}")
    print(f"    Env      : {env}")
    if onstart_cmd:
        preview = onstart_cmd[:80] + ("..." if len(onstart_cmd) > 80 else "")
        print(f"    Onstart  : {preview}")

    args = CreateInstanceArgs(
        offer_id=offer_id,
        image=image,
        env=env,
        onstart_cmd=onstart_cmd,
        disk=disk,
        ssh=ssh,
        direct=direct,
        label=label,
    )
    vast.ARGS = args

    try:
        result = vast.create__instance(args)

        if result is None:
            print("    -> Failed: no response from API.")
            return None

        # vast.create__instance with raw=True returns a requests.Response
        if hasattr(result, "json"):
            data = result.json()
        elif isinstance(result, dict):
            data = result
        else:
            print(f"    -> Unexpected response type: {result}")
            return None

        if data.get("success"):
            instance_id = data.get("new_contract")
            print(f"[+] Instance created successfully! Contract ID: {instance_id}")
            return instance_id

        error_msg = data.get("msg", "Unknown error")
        print(f"    -> Failed to create instance. Error: {error_msg}")
        return None

    except Exception as e:
        print(f"    -> Exception while creating instance: {e}")
        traceback.print_exc()
        return None


def _direct_ssh_port(instance: dict):
    """Return the host port mapped to the container's port 22, or ``None``."""
    for mapping in (instance.get("ports") or {}).get("22/tcp") or []:
        if mapping.get("HostPort"):
            return int(mapping["HostPort"])
    return None


def ssh_endpoint(instance: dict, prefer_direct: bool = True):
    """Return ``(ip, port)`` for SSH into an instance dict from ``show instances``.

    Direct SSH goes to the machine's ``public_ipaddr`` on the host port mapped
    to the container's port 22; proxy SSH goes through ``ssh_host``
    (``sshN.vast.ai``) on ``ssh_port``. The preferred kind is used when its
    info is available, otherwise the other one.

    Returns:
        tuple: ``(ip, port, is_direct)``, or ``(None, None, False)`` if neither
        is available yet.
    """
    direct = ((instance.get("public_ipaddr") or "").strip(), _direct_ssh_port(instance), True)
    proxy = (instance.get("ssh_host"), instance.get("ssh_port"), False)
    for ip, port, is_direct in ([direct, proxy] if prefer_direct else [proxy, direct]):
        if ip and port:
            return ip, port, is_direct
    return None, None, False


def create_instance_with_ssh_info(
    offer_id: int,
    image: str,
    env: str = "",
    onstart_cmd: str = None,
    disk: float = 20.0,
    ssh: bool = True,
    direct: bool = True,
    label: str = None,
    poll_interval: int = 10,
    timeout: int = 600,
    prefer_direct: bool = True,
):
    """Create a Vast.ai instance and wait until running, then return SSH info.

    This function is identical to :func:`create_instance` in its creation
    parameters, but after the instance is created it polls until the status
    becomes ``'running'`` and returns the SSH connection details instead of
    just the contract ID.

    Args:
        offer_id:      Offer ID from ``vastai search offers``.
        image:         Docker image to use (e.g. ``pytorch/pytorch:...``).
        env:           Env-variable / port-mapping string (Docker CLI format).
        onstart_cmd:   Startup command executed when the container first starts.
        disk:          Local disk size in GB (default: 20).
        ssh:           ``True`` to enable SSH access (default: ``True``).
        direct:        ``True`` for faster direct SSH/Jupyter connections (default: ``True``).
        label:         Optional label for the instance.
        poll_interval: Status polling interval in seconds (default: 10).
        timeout:       Maximum time to wait in seconds (default: 600).
        prefer_direct: ``True`` to keep polling until the machine's public IP
                       and mapped SSH port are available, instead of returning
                       the ``sshN.vast.ai`` proxy as soon as the instance runs
                       (see :func:`ssh_endpoint`). The proxy is returned only if
                       the direct endpoint never appears before the timeout.

    Returns:
        dict | None: Dictionary with ``instance_id`` (int), ``ip`` (str),
        ``ssh_port`` (int), and ``direct`` (bool) when the instance is running,
        or ``None`` on failure or timeout.

    Example::

        info = create_instance_with_ssh_info(
            offer_id=12345678,
            image="pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime",
        )
        if info:
            print(f"ID      : {info['instance_id']}")
            print(f"Host    : {info['ip']}")
            print(f"Port    : {info['ssh_port']}")
            # ssh -p {info['ssh_port']} root@{info['ip']}
    """
    instance_id = create_instance(
        offer_id=offer_id,
        image=image,
        env=env,
        onstart_cmd=onstart_cmd,
        disk=disk,
        ssh=ssh,
        direct=direct,
        label=label,
    )

    if instance_id is None:
        return None

    print(f"[*] Waiting for instance {instance_id} to be running (timeout: {timeout}s)...")

    show_args = VastInstanceArgs()
    vast.ARGS = show_args

    elapsed = 0
    proxy_info = None  # cadangan jika endpoint direct tidak muncul sampai timeout
    while elapsed < timeout:
        try:
            instances = vast.show__instances(show_args)

            if instances:
                target = next(
                    (inst for inst in instances if inst.get("id") == instance_id),
                    None,
                )

                if target:
                    status = target.get("actual_status", "unknown")
                    print(f"    -> Status: {status}...", end="\r")

                    if status == "running":
                        ip, port, is_direct = ssh_endpoint(target, prefer_direct)

                        if not ip or not port:
                            print(
                                f"\n    -> Instance running but SSH info not yet available "
                                f"(ip={ip}, port={port}). Waiting..."
                            )
                        else:
                            info = {
                                "instance_id": instance_id,
                                "ip": ip,
                                "ssh_port": port,
                                "direct": is_direct,
                                "machine_id": target.get("machine_id"),
                                "dlperf": target.get("dlperf"),
                            }
                            if is_direct or not prefer_direct:
                                print(f"\n[+] Instance RUNNING!")
                                return info
                            if proxy_info is None:
                                print(
                                    f"\n    -> Instance running, only proxy SSH ({ip}:{port}) so far. "
                                    "Waiting for direct IP/port..."
                                )
                            proxy_info = info

        except Exception as e:
            print(f"\n    -> Error polling status: {e}")
            traceback.print_exc()

        time.sleep(poll_interval)
        elapsed += poll_interval

    if proxy_info:
        print(f"\n[!] Timeout ({timeout}s) waiting for direct SSH; using proxy "
              f"{proxy_info['ip']}:{proxy_info['ssh_port']}.")
        return proxy_info
    print(f"\n[!] Timeout ({timeout}s) reached. Instance {instance_id} is not yet running.")
    return None


def get_instances():
    """Fetch all running instances from the Vast.ai API.

    Returns:
        list | None: List of instance dicts, or ``None`` on error.
    """
    instance_args = VastInstanceArgs()
    vast.ARGS = instance_args

    try:
        return vast.show__instances(instance_args)
    except Exception as e:
        print(f"Error fetching instances: {e}")
        traceback.print_exc()
        return None


def get_first_started_instance(
    target_label: str = None, target_ip: str = None, verbose: bool = True, prefer_direct: bool = True
):
    """Return SSH info for the first running or loading instance.

    If target_label is provided, only return an instance matching the label.
    If target_ip is provided, only return an instance reachable at that IP,
    either as ``ssh_host`` (SSH proxy) or ``public_ipaddr`` (direct SSH).
    Otherwise the endpoint is chosen by :func:`ssh_endpoint` (``prefer_direct``).

    Returns:
        dict | None: Dictionary with ``instance_id``, ``ip``, ``ssh_port``, and
        ``direct`` (``False`` for the proxy), or ``None`` if no running
        instances are found.
    """
    filters = []
    if target_label:
        filters.append(f"label '{target_label}'")
    if target_ip:
        filters.append(f"IP {target_ip}")
    filter_desc = f" with {' and '.join(filters)}" if filters else ""
    if verbose:
        print(f"[*] Retrieving the first running or loading instance{filter_desc}...")

    instances = get_instances()
    if not instances:
        return None

    for target in list(instances):
        status = target.get("actual_status")
        if status not in ["running", "loading"]:
            continue

        if target_label and target.get("label") != target_label:
            continue

        ssh_host = target.get("ssh_host")
        public_ip = (target.get("public_ipaddr") or "").strip()
        if not target_ip:
            ip, port, is_direct = ssh_endpoint(target, prefer_direct)
        elif target_ip == ssh_host:
            ip, port, is_direct = ssh_host, target.get("ssh_port"), False
        elif target_ip == public_ip:
            # Direct SSH: port 22 container di-map ke port lain di host
            ip, port, is_direct = public_ip, _direct_ssh_port(target), True
        else:
            continue
        if ip and port:
            return {
                "instance_id": target.get("id"),
                "ip": ip,
                "ssh_port": port,
                "direct": is_direct,
                "machine_id": target.get("machine_id"),
                "dlperf": target.get("dlperf"),
            }
    
    if verbose:
        print(f"    -> No running or loading instances found{filter_desc} and valid SSH info.")
    return None


def get_instance_ids_by_label():
    """Return instance IDs whose label matches the ``VAST_LABEL`` env var.

    Returns:
        list[int]: IDs of matching instances (empty list on error or no match).

    Raises:
        ValueError: If ``VAST_LABEL`` is not set or empty.
    """
    target_label = get_env_variable("VAST_LABEL", required=True)
    print(f"Filtering instances by label: {target_label}")

    instances = get_instances()
    if instances is None:
        return []

    filtered_ids = []
    for instance in instances:
        instance_label = instance.get("label") or ""
        if instance_label == target_label:
            instance_id = instance.get("id")
            if instance_id is not None:
                filtered_ids.append(instance_id)

    return filtered_ids


def destroy_instance(instance_id: int) -> bool:
    """Destroy a Vast.ai instance.

    Args:
        instance_id: ID of the instance to destroy.

    Returns:
        ``True`` on success, ``False`` on error.
    """
    destroy_args = VastDestroyArgs(instance_id)
    vast.ARGS = destroy_args

    try:
        vast.destroy__instance(destroy_args)
        print(f"Instance {instance_id} destroyed successfully.")
        return True
    except Exception as e:
        print(f"Error destroying instance {instance_id}: {e}")
        traceback.print_exc()
        return False


def search_offers(
    query_list=None,
    order: str = "score-",
    limit=None,
    offer_type: str = "on-demand",
):
    """Search for available offers on Vast.ai.

    Args:
        query_list: List of query filters
            (e.g. ``['gpu_name=A100', 'geolocation=US']``).
        order:      Sort key (default: ``"score-"``).
        limit:      Maximum number of results to return (default: ``None``).
        offer_type: Pricing type: ``"on-demand"``, ``"bid"`` (interruptible),
            or ``"reserved"`` (default: ``"on-demand"``).

    Returns:
        list | None: List of offer dicts, or ``None`` on error.
    """
    vast_args = VastSearchOffersArgs(
        query_list=query_list, order=order, limit=limit, offer_type=offer_type
    )
    vast.ARGS = vast_args

    try:
        return vast.search__offers(vast_args)
    except Exception as e:
        print(f"Error searching offers: {e}")
        traceback.print_exc()
        return None


def rent_instance(offer_id: int, template_id, disk_space: float = 20) -> int:
    """Rent an instance using a template ID.

    Args:
        offer_id:    Offer ID to rent.
        template_id: Docker image template ID.
        disk_space:  Disk space in GB (default: 20).

    Returns:
        int | None: Instance/contract ID, or ``None`` on failure.
    """
    print(f"[*] Renting instance offer {offer_id} with template ID {template_id}...")

    create_args = VastCreateInstanceArgs(offer_id, template_id, disk_space)
    vast.ARGS = create_args

    try:
        result = vast.create__instance(create_args)

        if result and isinstance(result, dict):
            if result.get("success"):
                new_id = result.get("new_contract")
                print(f"    -> SUCCESS! Contract ID: {new_id}")
                return new_id
            error_msg = result.get("msg", "Unknown Error")
            print(f"    -> Failed to rent. Error: {error_msg}")
            return None

        print("    -> Failed to rent. Invalid response.")
        return None

    except Exception as e:
        print(f"    -> Error creating instance: {e}")
        traceback.print_exc()
        return None


def wait_for_startup(instance_id: int, timeout: int = 600, poll_interval: int = 10) -> bool:
    """Poll until the instance reaches ``'running'`` status.

    Args:
        instance_id:    Instance ID to wait for.
        timeout:        Maximum seconds to wait before giving up (default: 600).
        poll_interval:  Seconds between each status check (default: 10).

    Returns:
        ``True`` when the instance is running, ``False`` on timeout.
    """
    print(f"[*] Waiting for instance {instance_id} to start (timeout: {timeout}s)...")

    show_args = VastInstanceArgs()
    vast.ARGS = show_args

    elapsed = 0
    while elapsed < timeout:
        try:
            instances = vast.show__instances(show_args)

            if instances:
                my_instance = next(
                    (item for item in instances if item["id"] == instance_id), None
                )

                if my_instance:
                    status = my_instance.get("actual_status")
                    print(f"    -> Status: {status}...", end="\r")

                    if status == "running":
                        print(f"\n[+] Instance RUNNING!")
                        print(
                            f"    SSH: ssh -p {my_instance['ssh_port']} root@{my_instance['ssh_host']}"
                        )
                        print("    (Use Terminal/Putty with the details above)")
                        return True

        except Exception as e:
            print(f"\n    -> Error checking status: {e}")

        time.sleep(poll_interval)
        elapsed += poll_interval

    print(f"\n[!] Timeout ({timeout}s) reached. Instance {instance_id} is not yet running.")
    return False


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def format_instance_for_display(instance: dict) -> dict:
    """Convert a raw instance dict into a display-ready row dict.

    Args:
        instance: Instance data from the Vast.ai API.

    Returns:
        Dict with keys: ``id``, ``status``, ``gpu``, ``num_gpus``, ``price``,
        ``duration``, ``machine``, ``label``.
    """
    instance_id = str(instance.get("id", "-"))
    status = str(instance.get("actual_status", "-"))
    gpu_model = str(instance.get("gpu_name", "-"))
    num_gpus = str(instance.get("num_gpus", "-"))

    price_val = instance.get("dph_total") or 0
    price = f"${price_val:.4f}"

    duration = "-"
    start_time = instance.get("start_date")
    if start_time:
        elapsed = time.time() - start_time
        hours = int(elapsed / 3600)
        minutes = int((elapsed % 3600) / 60)
        duration = f"{hours}h {minutes}m"

    machine_id = str(instance.get("machine_id", "-"))
    label = instance.get("label") or "-"

    return {
        "id": instance_id,
        "status": status,
        "gpu": gpu_model,
        "num_gpus": num_gpus,
        "price": price,
        "duration": duration,
        "machine": machine_id,
        "label": label,
    }


def format_offer_for_display(offer: dict) -> dict:
    """Convert a raw offer dict into a display-ready row dict.

    Args:
        offer: Offer data from the Vast.ai API.

    Returns:
        Dict with keys: ``id``, ``gpu``, ``num_gpus``, ``ram``, ``dlperf``,
        ``price``, ``reliability``, ``geolocation``, ``machine_id``,
        ``host_id``.
    """
    offer_id = str(offer.get("id", "-"))
    gpu_model = str(offer.get("gpu_name", "-"))
    num_gpus = str(offer.get("num_gpus", "-"))

    ram_mb = offer.get("gpu_ram") or 0
    ram_gb = f"{int(ram_mb / 1024)}GB"

    dlperf_val = offer.get("dlperf") or 0
    dlperf = f"{dlperf_val:.1f}"

    price_val = offer.get("dph_total") or 0
    price = f"${price_val:.4f}"

    reliability_val = offer.get("reliability2") or 0
    reliability = f"{reliability_val * 100:.1f}%"

    geolocation = offer.get("geolocation") or "Unknown"
    machine_id = str(offer.get("machine_id", "-"))
    host_id = str(offer.get("host_id", "-"))

    return {
        "id": offer_id,
        "gpu": gpu_model,
        "num_gpus": num_gpus,
        "ram": ram_gb,
        "dlperf": dlperf,
        "price": price,
        "reliability": reliability,
        "geolocation": geolocation,
        "machine_id": machine_id,
        "host_id": host_id,
    }


def display_offers(offers: list) -> None:
    """Print offers in a formatted table.

    Args:
        offers: List of offer dicts from the Vast.ai API.
    """
    if not offers:
        print("No offers found.")
        return

    header = (
        f"{'No':<4} {'Offer ID':<12} {'Machine':<11} {'Host':<10} "
        f"{'GPU':<20} {'Num':<4} {'RAM':<8} {'DLPerf':<8} "
        f"{'Reliab':<8} {'Region':<20} {'$/Hr':<10}"
    )
    separator = "-" * len(header)
    print(separator)
    print(header)
    print(separator)

    for idx, offer in enumerate(offers, start=1):
        row = format_offer_for_display(offer)
        print(
            f"{idx:>4} {row['id']:<12} {row['machine_id']:<11} {row['host_id']:<10} "
            f"{row['gpu']:<20} {row['num_gpus']:<4} {row['ram']:<8} "
            f"{row['dlperf']:<8} {row['reliability']:<8} {row['geolocation']:<20} "
            f"{row['price']:<10}"
        )


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------


def get_first_instance_id() -> int:
    """Return the first instance ID matching the ``VAST_LABEL`` env var.

    Returns:
        int: The first matching instance ID.

    Raises:
        ValueError: If no matching instances are found.
    """
    print("Getting first instance ID...")
    instance_ids = get_instance_ids_by_label()

    if not instance_ids:
        raise ValueError("No instances found matching the VAST_LABEL environment variable.")

    return instance_ids[0]


def _parse_template_query(query_string: str, fields: set) -> dict:
    """Parse a simple ``field=value`` query string for template searches.

    Args:
        query_string: A string like ``"name=K2FSA"``.
        fields:       Set of recognised field names.

    Returns:
        Parsed query dict (empty if the field is not in *fields*).
    """
    query = {}
    if "=" in query_string:
        field, value = query_string.split("=", 1)
        field = field.strip()
        value = value.strip()
        if field in fields:
            query[field] = {"eq": value}
    return query


def search_template_by_name(template_name: str) -> list:
    """Search for Vast.ai machine templates by name via the REST API.

    The ``vast.search__templates()`` entry-point only prints results and does
    not return data, so this function calls the REST API directly.

    Args:
        template_name: Template name to search for (e.g. the value of
            ``IMAGE_TAG``).

    Returns:
        list: Matching template dicts, or ``[]`` if none are found.
    """
    print(f"[*] Searching for template with name: '{template_name}'...")

    _TEMPLATE_FIELDS = {
        "creator_id", "created_at", "count_created", "default_tag",
        "docker_login_repo", "id", "image", "jup_direct", "hash_id",
        "private", "name", "recent_create_date", "recommended_disk_space",
        "recommended", "ssh_direct", "tag", "use_ssh",
    }

    try:
        api_key = get_api_key()
        query = _parse_template_query(f"name={template_name}", _TEMPLATE_FIELDS)

        params = {
            "select_cols": json.dumps(["*"]),
            "select_filters": json.dumps(query),
            "api_key": api_key,
        }
        url = f"{_VAST_BASE_URL}/api/v0/template/"
        response = requests.get(url, params=params, headers={"Accept": "application/json"})
        response.raise_for_status()

        templates = response.json().get("templates", [])

        if templates:
            print(f"    -> Found {len(templates)} template(s):")
            for idx, template in enumerate(templates, 1):
                t_id = template.get("id")
                t_name = template.get("name", "N/A")
                t_image = template.get("image", "N/A")
                t_tag = template.get("tag", "N/A")
                print(f"       {idx}. ID: {t_id} | Name: {t_name} | Image: {t_image}:{t_tag}")
        else:
            print("    -> No templates found.")

        return templates

    except Exception as e:
        print(f"    -> Error searching for template: {e}")
        traceback.print_exc()
        return []


def get_template_id_by_name(template_name: str):
    """Return the ID of the first template matching *template_name*.

    Args:
        template_name: Template name to search for.

    Returns:
        int | None: Template ID if found, otherwise ``None``.
    """
    templates = search_template_by_name(template_name)

    if templates:
        template_id = templates[0].get("id")
        print(f"[+] Using Template ID: {template_id}")
        return template_id

    return None


def get_offer_id_by_machine_id(machine_id, offers=None, num_gpus=None):
    """Return the offer ID for a given machine ID, optionally filtered by GPU count.

    Searches through *offers* (or fetches fresh results when *offers* is
    ``None``) and returns the offer ID of the first matching entry.

    Args:
        machine_id (int | str): The machine ID to look up.
        offers (list | None): Pre-fetched offer list from :func:`search_offers`.
            When ``None`` (default), a fresh list is fetched automatically.
        num_gpus (int | None): When provided, only offers with exactly this
            many GPUs are considered.

    Returns:
        int | None: The offer ID on match, otherwise ``None``.

    Example::

        # Any offer on machine 14165
        offer_id = get_offer_id_by_machine_id(14165)

        # Only the 2-GPU offer on machine 42402
        offer_id = get_offer_id_by_machine_id(42402, num_gpus=2)
    """
    # Normalise to string for comparison (API may return int or str)
    target = str(machine_id)

    if offers is None:
        print(f"[*] Fetching offers to look up machine ID {target}...")
        offers = search_offers(query_list=[f"machine_id={target}", f"num_gpus={num_gpus}"])

    if not offers:
        print(f"[!] No offers available to search for machine ID {target}.")
        return None

    for offer in offers:
        if str(offer.get("machine_id", "")) != target:
            continue

        if num_gpus is not None and int(offer.get("num_gpus", 0)) != int(num_gpus):
            continue

        offer_id = offer.get("id")
        found_gpus = offer.get("num_gpus")
        print(f"[+] Machine {target} (num_gpus={found_gpus}) → Offer ID: {offer_id}")
        return offer_id

    gpu_info = f", num_gpus={num_gpus}" if num_gpus is not None else ""
    print(f"[!] Machine ID {target}{gpu_info} not found in offers.")
    return None


# Default query filters for a reliable transcription workload.
#
# CATATAN SATUAN:
#   inet_down  : Mbps (Megabit/s)  → 1000 = 1 Gbps
#   disk_bw    : MB/s              → 5000 = 5 GB/s
#   cpu_ram    : GB
#   duration   : jam (hours)       → 168 = 7 hari
#   dph_total  : USD/jam
#
# Field yang TIDAK dikenali Vast.ai API (jangan masukkan ke query list):
#   reliability2  → gunakan filter client-side di _get_best_offer_id_impl()
#   type          → sudah default on-demand di VastSearchOffersArgs
#
# Kriteria:
# - Host reliability > 98%    : [client-side] reliability2 >= 0.98
# - AVX support               : has_avx=true
# - CUDA >= 12.8              : cuda_max_good >= 12.8
# - Max duration > 7 days     : duration >= 168 (jam)
# - Price < $0.40/hr          : dph_total < 0.4
# - CPU cores >= 8            : cpu_cores_effective >= 8
# - CPU RAM > 8 GB            : cpu_ram > 8
# - Disk bandwidth > 5 GB/s   : disk_bw >= 5000 (MB/s)
# - Internet DL > 1 Gbps      : inet_down >= 1000 (Mbps)
# - DL performance > 100      : dlperf > 100
DEFAULT_OFFER_QUERY: list[str] = [
    # 'machine_id=16571',
    # 'machine_id=51839',
    "reliability>=0.9882",
    "has_avx=true",
    # "gpu_name=RTX_5090",
    "cuda_max_good>=12.8",
    "dph_total<=0.5",
    "cpu_cores_effective>=8",
    "cpu_ram>=8",
    "disk_bw>=5000",
    "inet_down>=800",
    "dlperf>=100",
    # "dlperf>=70.23",
    # "dlperf>44",
    f"inet_down_cost<={6.62/1000}",
    # "type=on-demand",
     # "duration>=168",
]

# Threshold client-side untuk field yang tidak didukung Vast.ai query API.
# GPU filter menggunakan OR — tidak bisa diekspresikan dalam satu query string.
_CLIENT_SIDE_MIN_RELIABILITY = 0.98
_CLIENT_SIDE_ALLOWED_GPUS: list[str] | None = [
    "RTX_5090",
    "RTX_4090",
    "RTX_3090",
]


def get_best_offer_id(
    query_list: list[str] | None = None,
    order: str = "dph_total+",
    num_gpus: int | None = None,
) -> int | None:
    """Search Vast.ai for the best available offer matching the given filters.

    Returns the offer ID of the top-ranked result.  When no *query_list* is
    provided the :data:`DEFAULT_OFFER_QUERY` filters are used, which select
    machines suitable for a typical transcription workload.

    Args:
        query_list: List of filter strings accepted by ``vastai search offers``
            (e.g. ``['reliability2>=0.98', 'dph_total<0.4']``).
            Defaults to :data:`DEFAULT_OFFER_QUERY` when ``None``.
        order:      Sort column (default: ``"dph_total+"`` — lowest price first).
        num_gpus:   When provided, only offers with exactly this many GPUs are
            considered (applied as a client-side filter after the API call).

    Returns:
        int | None: The offer ID of the best matching offer, or ``None`` if no
        suitable offer is found.

    Example::

        # Use built-in defaults
        offer_id = get_best_offer_id()

        # Custom filters
        offer_id = get_best_offer_id(
            query_list=["reliability2>=0.99", "dph_total<0.3", "num_gpus=1"],
        )
    """
    return _get_best_offer_id_impl(query_list=query_list, order=order, num_gpus=num_gpus)


def get_best_offer(
    query_list: list[str] | None = None,
    order: str = "dph_total+",
    num_gpus: int | None = None,
) -> dict | None:
    """Search Vast.ai for the best available offer and return the full offer dict.

    Like :func:`get_best_offer_id` but returns the raw offer dictionary so
    callers can inspect any field (GPU model, DL perf, price, reliability …).

    Args:
        query_list: Filter strings (defaults to :data:`DEFAULT_OFFER_QUERY`).
        order:      Sort column (default: ``"dph_total+"`` — cheapest first).
        num_gpus:   Client-side GPU count filter.

    Returns:
        dict | None: The first matching offer dict, or ``None`` if none found.
    """
    return _get_best_offer_id_impl(
        query_list=query_list, order=order, num_gpus=num_gpus, return_full=True
    )


def _get_best_offer_id_impl(
    query_list: list[str] | None,
    order: str,
    num_gpus: int | None,
    return_full: bool = False,
):
    """Shared implementation for :func:`get_best_offer_id` and :func:`get_best_offer`."""
    effective_query = query_list if query_list is not None else DEFAULT_OFFER_QUERY

    print("[*] Searching for best offer on Vast.ai with filters:")
    for q in effective_query:
        print(f"    {q}")
    # Tampilkan semua filter client-side yang aktif
    print(f"    reliability>={_CLIENT_SIDE_MIN_RELIABILITY}  [client-side]")
    if _CLIENT_SIDE_ALLOWED_GPUS:
        gpu_list = " | ".join(_CLIENT_SIDE_ALLOWED_GPUS)
        print(f"    gpu_name in [{gpu_list}]  [client-side]")
    if num_gpus is not None:
        print(f"    num_gpus={num_gpus}  [client-side]")

    offers = search_offers(query_list=effective_query, order=order)

    if not offers:
        print("[!] No offers found matching the given criteria.")
        return None

    for offer in offers:
        # Client-side: filter reliability (tidak didukung Vast.ai query API)
        if (offer.get("reliability") or 0) < _CLIENT_SIDE_MIN_RELIABILITY:
            continue

        # Client-side: filter GPU — OR condition tidak bisa dipakai di query API
        if _CLIENT_SIDE_ALLOWED_GPUS:
            gpu = (offer.get("gpu_name") or "").replace(" ", "_")
            if not any(allowed in gpu for allowed in _CLIENT_SIDE_ALLOWED_GPUS):
                continue

        if num_gpus is not None and int(offer.get("num_gpus", 0)) != int(num_gpus):
            continue

        offer_id = offer.get("id")
        gpu_name = offer.get("gpu_name", "unknown")
        n_gpus = offer.get("num_gpus", "?")
        price = offer.get("dph_total", 0)
        reliability = offer.get("reliability", 0) * 100
        dlperf = offer.get("dlperf", 0) or 0
        machine_id = offer.get("machine_id", "?")

        print(
            f"[+] Best offer found → ID: {offer_id} | "
            f"Machine: {machine_id} | GPU: {n_gpus}× {gpu_name} | "
            f"${price:.4f}/hr | DLPerf: {dlperf:.1f} | Reliability: {reliability:.1f}%"
        )
        return offer if return_full else offer_id

    gpu_filter_info = ""
    if _CLIENT_SIDE_ALLOWED_GPUS:
        gpu_filter_info += f" gpu_name in {_CLIENT_SIDE_ALLOWED_GPUS}"
    if num_gpus is not None:
        gpu_filter_info += f" num_gpus={num_gpus}"
    print(f"[!] No offers found{gpu_filter_info} after client-side filtering.")
    return None

    