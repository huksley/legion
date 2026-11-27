import ipaddress
import json

import pytest

from legion.harbor import networking


def test_subnets_exclude_vpn_routes_from_all_tables():
    routes = [
        {"dst": "default", "gateway": "192.168.1.1"},
        {"dst": "172.17.0.0/16", "dev": "docker0"},
        {"dst": "172.18.0.0/16", "dev": "tailscale0", "table": 52},
        {"dst": "172.20.0.0/17", "dev": "tailscale0", "table": 52},
        {"dst": "172.30.1.9", "type": "local"},
    ]
    candidates = networking.candidate_subnets(routes)
    assert candidates
    assert all(n.prefixlen == 24 for n in candidates)
    for route in routes[1:]:
        occupied = ipaddress.ip_network(route["dst"])
        assert all(not n.overlaps(occupied) for n in candidates)


@pytest.mark.parametrize("engine", ["docker", "podman"])
async def test_allocator_handles_concurrent_reservation_and_owns_cleanup(
    tmp_path, monkeypatch, engine
):
    calls = []
    creates = 0

    async def command(*args):
        nonlocal creates
        calls.append(args)
        if args[0] == "ip":
            return 0, "[]", ""
        if args[2] == "create":
            creates += 1
            return (
                (1, "", "Pool overlaps with other one on this address space")
                if creates == 1
                else (0, "network-id", "")
            )
        if args[2] == "inspect":
            return (
                0,
                json.dumps(
                    [
                        {
                            ("labels" if engine == "podman" else "Labels"): {
                                "io.legion.run": "test-run"
                            }
                        }
                    ]
                ),
                "",
            )
        return 0, "", ""

    monkeypatch.setattr(networking, "command", command)
    networks = networking.TrialNetworks("test-run", engine=engine)
    await networks.start()
    path = await networks.create("agent-001", tmp_path / "overlay.json")
    assert creates == 2
    assert json.loads(path.read_text())["networks"]["default"] == {
        "external": True,
        "name": "legion-test-run-agent-001",
    }
    await networks.stop()
    assert calls[-1] == (engine, "network", "rm", "legion-test-run-agent-001")


@pytest.mark.parametrize("failure", ["ownership", "daemon"])
async def test_cleanup_reports_failures_and_never_deletes_foreign_networks(
    monkeypatch, failure
):
    calls = []

    async def command(*args):
        calls.append(args)
        if failure == "daemon":
            return 1, "", "Cannot connect to the Docker daemon"
        return 0, json.dumps([{"Labels": {"io.legion.run": "another-run"}}]), ""

    monkeypatch.setattr(networking, "command", command)
    networks = networking.TrialNetworks("test-run")
    networks.names = ["a-network"]
    with pytest.raises(RuntimeError):
        await networks.stop()
    assert all(call[2] == "inspect" for call in calls)


async def test_podman_ignores_unrelated_docker_endpoint(monkeypatch):
    monkeypatch.setenv("DOCKER_HOST", "ssh://unrelated-host")
    monkeypatch.delenv("CONTAINER_HOST", raising=False)

    async def command(*args):
        return 0, "[]", ""

    monkeypatch.setattr(networking, "command", command)
    await networking.TrialNetworks("local", engine="podman").start()
    monkeypatch.setenv("CONTAINER_HOST", "ssh://remote-host")
    with pytest.raises(ValueError, match="local Linux"):
        await networking.TrialNetworks("remote", engine="podman").start()


NETSTAT = """Routing tables

Internet:
Destination        Gateway            Flags               Netif Expire
default            192.168.1.1        UGScg                 en0
10/8               utun3              UGSc                utun3
127                127.0.0.1          UCS                   lo0
172.16             utun3              UGSc                utun3
172.20.4/22        utun3              UGSc                utun3
192.168.1          link#14            UCS                   en0
192.168.1.1/32     link#14            UCS                   en0
"""


async def test_macos_keeps_container_networks_clear_of_its_routes(monkeypatch):
    """Docker Desktop: the Mac's routes come from `netstat` (networks written without their
    trailing zero octets), and no candidate overlaps them."""
    assert networking.darwin_routes(NETSTAT) == [
        {"dst": "10.0.0.0/8"}, {"dst": "127.0.0.0/8"}, {"dst": "172.16.0.0/16"},
        {"dst": "172.20.4.0/22"}, {"dst": "192.168.1.0/24"}, {"dst": "192.168.1.1/32"}]
    calls = []

    async def command(*args):
        calls.append(args)
        return 0, NETSTAT, ""

    monkeypatch.setattr(networking, "command", command)
    monkeypatch.setattr(networking.sys, "platform", "darwin")
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    trial_networks = networking.TrialNetworks("mac")
    await trial_networks.start()
    assert calls == [("netstat", "-rn", "-f", "inet")]
    for busy in ("172.16.0.0/16", "172.20.4.0/22"):
        assert not any(n.overlaps(ipaddress.ip_network(busy))
                       for n in trial_networks.candidates)
    monkeypatch.setattr(networking.sys, "platform", "win32")
    with pytest.raises(ValueError, match="local Linux"):
        await networking.TrialNetworks("windows").start()
