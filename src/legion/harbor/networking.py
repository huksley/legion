"""Owned container networks whose subnets do not overlap host or VPN routes."""

import asyncio
import ipaddress
import json
import os
import random
import sys
from contextlib import suppress

POOL = ipaddress.ip_network("172.16.0.0/12")


async def command(*args):
    process = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )

    async def communicate():
        try:
            return await asyncio.wait_for(process.communicate(), timeout=30)
        except TimeoutError:
            with suppress(ProcessLookupError):
                process.kill()
            await process.wait()
            raise RuntimeError(f"Networking command timed out: {args[0]}") from None

    communication = asyncio.create_task(communicate())
    try:
        stdout, stderr = await asyncio.shield(communication)
    except asyncio.CancelledError:
        # Let a Docker resource mutation finish before its owner cleans up.
        await communication
        raise
    return process.returncode, stdout.decode(), stderr.decode()


def candidate_subnets(routes):
    occupied = [
        ipaddress.ip_network(route["dst"], strict=False)
        for route in routes
        if route.get("dst") not in (None, "default", "0.0.0.0/0")
    ]
    occupied = [network for network in occupied if network.overlaps(POOL)]
    candidates = list(POOL.subnets(new_prefix=24))
    random.SystemRandom().shuffle(candidates)
    return [n for n in candidates if not any(n.overlaps(r) for r in occupied)]


def darwin_routes(table):
    """`netstat -rn -f inet` destinations as `ip -j route` entries ({"dst": "a.b.c.d/n"}):
    macOS writes a network with its trailing zero octets left out ("10/8", "172.16",
    "192.168.1") and a host without a prefix."""
    routes = []
    for line in table.splitlines():
        fields = line.split()
        if not fields:
            continue
        destination, _, prefix = fields[0].partition("/")
        octets = destination.split(".")
        if not 1 <= len(octets) <= 4 or not all(o.isdigit() for o in octets):
            continue
        address = ".".join(octets + ["0"] * (4 - len(octets)))
        bits = prefix or str(8 * len(octets))
        with suppress(ValueError):
            routes.append({"dst": str(ipaddress.ip_network(f"{address}/{bits}", strict=False))})
    return routes


class TrialNetworks:
    def __init__(self, run_id, engine="docker"):
        if engine not in ("docker", "podman"):
            raise ValueError("Unsupported container engine")
        self.engine = engine
        self.run_id = run_id
        self.names = []
        self.candidates = []
        self.allocations = {}

    async def start(self):
        remote = os.environ.get(
            "DOCKER_HOST" if self.engine == "docker" else "CONTAINER_HOST", "unix://"
        ).startswith(("tcp:", "ssh:"))
        if remote or sys.platform not in ("linux", "darwin"):
            raise ValueError("This release requires a local Linux container engine")
        if sys.platform == "darwin":
            # Docker Desktop / podman machine: the engine's networks live in its Linux VM
            # behind NAT; the Mac's own routes (a VPN's) are still kept clear of them.
            code, stdout, stderr = await command("netstat", "-rn", "-f", "inet")
            if code:
                raise RuntimeError(f"Cannot inspect host/VPN routes: {stderr.strip()}")
            self.candidates = candidate_subnets(darwin_routes(stdout))
            return
        code, stdout, stderr = await command(
            "ip", "-j", "-4", "route", "show", "table", "all"
        )
        if code:
            raise RuntimeError(f"Cannot inspect host/VPN routes: {stderr.strip()}")
        self.candidates = candidate_subnets(json.loads(stdout))

    async def create(self, participant, overlay):
        name = f"legion-{self.run_id}-{participant}"
        self.names.append(name)
        while self.candidates:
            subnet = str(self.candidates.pop())
            code, _, stderr = await command(
                self.engine,
                "network",
                "create",
                "--label",
                f"io.legion.run={self.run_id}",
                "--subnet",
                subnet,
                name,
            )
            if not code:
                self.allocations[participant] = {"name": name, "subnet": subnet}
                # Other task-authored networks remain under Harbor's control.
                overlay.write_text(
                    json.dumps(
                        {"networks": {"default": {"external": True, "name": name}}}
                    )
                )
                return overlay
            if "overlap" not in stderr.lower():
                raise RuntimeError(f"Cannot create trial network: {stderr.strip()}")
        raise RuntimeError(
            "No unused /24 Docker subnet remains outside host/VPN routes in 172.16.0.0/12"
        )

    async def stop(self):
        errors = []
        for name in self.names:
            code, stdout, stderr = await command(
                self.engine, "network", "inspect", name
            )
            if code:
                if (
                    "not found" not in stderr.lower()
                    and "no such network" not in stderr.lower()
                ):
                    errors.append(stderr.strip() or f"Cannot inspect network {name}")
                continue
            network = json.loads(stdout)[0]
            if (network.get("Labels") or network.get("labels") or {}).get(
                "io.legion.run"
            ) != self.run_id:
                errors.append(f"Network ownership mismatch: {name}")
                continue
            code, _, stderr = await command(self.engine, "network", "rm", name)
            if code:
                errors.append(stderr.strip())
        if errors:
            raise RuntimeError("; ".join(errors))
