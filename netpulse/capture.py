"""
Packet capture and flow aggregation module.

Captures raw packets at layer 3/4 and aggregates them into 5-tuple flows.
Uses a rolling ring buffer to hold the last 5 minutes of packets in memory.
"""

import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from scapy.all import sniff, IP, TCP, UDP, ICMP  # type: ignore
import numpy as np
import pandas as pd


@dataclass
class PacketRecord:
    """Represents a captured packet with essential metadata."""
    timestamp: float
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    bytes: int
    flags: Dict[str, int] = field(default_factory=dict)
    direction: str = "forward"  # or "backward"


@dataclass
class FlowRecord:
    """Represents an aggregated network flow."""
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    start_time: float
    end_time: float
    total_bytes_forward: int = 0
    total_bytes_backward: int = 0
    packet_count_forward: int = 0
    packet_count_backward: int = 0
    syn_count: int = 0
    ack_count: int = 0
    fin_count: int = 0
    rst_count: int = 0
    forward_sizes: List[int] = field(default_factory=list)
    backward_sizes: List[int] = field(default_factory=list)
    forward_inter_arrival: List[float] = field(default_factory=list)
    backward_inter_arrival: List[float] = field(default_factory=list)
    
    @property
    def duration_ms(self) -> float:
        """Flow duration in milliseconds."""
        return (self.end_time - self.start_time) * 1000
    
    @property
    def five_tuple(self) -> Tuple:
        """Return the 5-tuple identifier for this flow."""
        return (self.src_ip, self.dst_ip, self.src_port, self.dst_port, self.protocol)


class RingBuffer:
    """Thread-safe ring buffer for packet storage."""
    
    def __init__(self, duration_minutes: int = 5):
        self.duration_seconds = duration_minutes * 60
        self.packets: List[PacketRecord] = []
        self.lock = threading.Lock()
    
    def add(self, packet: PacketRecord) -> None:
        """Add a packet to the buffer."""
        with self.lock:
            self.packets.append(packet)
            self._cleanup(packet.timestamp)
    
    def _cleanup(self, current_time: float) -> None:
        """Remove packets older than the buffer duration."""
        cutoff = current_time - self.duration_seconds
        self.packets = [p for p in self.packets if p.timestamp >= cutoff]
    
    def flush(self) -> List[PacketRecord]:
        """Flush and return all packets from the buffer."""
        with self.lock:
            packets = self.packets.copy()
            self.packets = []
            return packets
    
    def get_packets(self) -> List[PacketRecord]:
        """Get a copy of current packets without flushing."""
        with self.lock:
            return self.packets.copy()


class PacketCapture:
    """Main packet capture and flow aggregation class."""
    
    def __init__(self, interface: Optional[str] = None, buffer_duration: int = 5):
        self.interface = interface
        self.buffer = RingBuffer(duration_minutes=buffer_duration)
        self.running = False
        self.capture_thread: Optional[threading.Thread] = None
        self.flows: Dict[Tuple, FlowRecord] = {}
        self.lock = threading.Lock()
    
    def _packet_callback(self, packet) -> None:
        """Callback function for packet capture."""
        try:
            if not packet.haslayer(IP):
                return
            
            ip_layer = packet[IP]
            src_ip = ip_layer.src
            dst_ip = ip_layer.dst
            
            # Determine protocol and ports
            protocol = "OTHER"
            src_port = 0
            dst_port = 0
            flags = {}
            bytes_count = len(packet)
            
            if packet.haslayer(TCP):
                protocol = "TCP"
                tcp_layer = packet[TCP]
                src_port = tcp_layer.sport
                dst_port = tcp_layer.dport
                flags = {
                    "S": int(tcp_layer.flags.S),
                    "A": int(tcp_layer.flags.A),
                    "F": int(tcp_layer.flags.F),
                    "R": int(tcp_layer.flags.R),
                }
            elif packet.haslayer(UDP):
                protocol = "UDP"
                udp_layer = packet[UDP]
                src_port = udp_layer.sport
                dst_port = udp_layer.dport
            elif packet.haslayer(ICMP):
                protocol = "ICMP"
            
            # Create packet record
            record = PacketRecord(
                timestamp=time.time(),
                src_ip=src_ip,
                dst_ip=dst_ip,
                src_port=src_port,
                dst_port=dst_port,
                protocol=protocol,
                bytes=bytes_count,
                flags=flags,
            )
            
            self.buffer.add(record)
            
        except Exception as e:
            # Silently ignore malformed packets
            pass
    
    def start_capture(self) -> None:
        """Start packet capture in a background thread."""
        if self.running:
            return
        
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
    
    def _capture_loop(self) -> None:
        """Main capture loop using scapy."""
        try:
            sniff(
                iface=self.interface,
                prn=self._packet_callback,
                store=False,
                stop_filter=lambda x: not self.running
            )
        except Exception as e:
            print(f"Capture error: {e}")
            self.running = False
    
    def stop_capture(self) -> None:
        """Stop packet capture."""
        self.running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
    
    def aggregate_flows(self) -> pd.DataFrame:
        """Aggregate buffered packets into flow records."""
        packets = self.buffer.flush()
        
        if not packets:
            return pd.DataFrame()
        
        # Group packets by 5-tuple
        flow_data: Dict[Tuple, List[PacketRecord]] = defaultdict(list)
        for pkt in packets:
            key = (pkt.src_ip, pkt.dst_ip, pkt.src_port, pkt.dst_port, pkt.protocol)
            flow_data[key].append(pkt)
        
        # Convert to flow records
        flow_records = []
        for key, pkts in flow_data.items():
            pkts_sorted = sorted(pkts, key=lambda p: p.timestamp)
            
            flow = FlowRecord(
                src_ip=key[0],
                dst_ip=key[1],
                src_port=key[2],
                dst_port=key[3],
                protocol=key[4],
                start_time=pkts_sorted[0].timestamp,
                end_time=pkts_sorted[-1].timestamp,
            )
            
            # Aggregate statistics
            prev_time = None
            for pkt in pkts_sorted:
                # Determine direction (simplified: first packet defines forward)
                is_forward = (pkt.src_ip == flow.src_ip)
                
                if is_forward:
                    flow.total_bytes_forward += pkt.bytes
                    flow.packet_count_forward += 1
                    flow.forward_sizes.append(pkt.bytes)
                else:
                    flow.total_bytes_backward += pkt.bytes
                    flow.packet_count_backward += 1
                    flow.backward_sizes.append(pkt.bytes)
                
                # Inter-arrival times
                if prev_time is not None:
                    iat = pkt.timestamp - prev_time
                    if is_forward:
                        flow.forward_inter_arrival.append(iat)
                    else:
                        flow.backward_inter_arrival.append(iat)
                prev_time = pkt.timestamp
                
                # TCP flags
                flow.syn_count += pkt.flags.get("S", 0)
                flow.ack_count += pkt.flags.get("A", 0)
                flow.fin_count += pkt.flags.get("F", 0)
                flow.rst_count += pkt.flags.get("R", 0)
            
            flow_records.append(flow)
        
        # Convert to DataFrame
        return self._flows_to_dataframe(flow_records)
    
    def _flows_to_dataframe(self, flows: List[FlowRecord]) -> pd.DataFrame:
        """Convert flow records to a pandas DataFrame."""
        if not flows:
            return pd.DataFrame()
        
        data = []
        for flow in flows:
            row = {
                "src_ip": flow.src_ip,
                "dst_ip": flow.dst_ip,
                "src_port": flow.src_port,
                "dst_port": flow.dst_port,
                "protocol": flow.protocol,
                "duration_ms": flow.duration_ms,
                "total_bytes_forward": flow.total_bytes_forward,
                "total_bytes_backward": flow.total_bytes_backward,
                "packet_count_forward": flow.packet_count_forward,
                "packet_count_backward": flow.packet_count_backward,
                "syn_count": flow.syn_count,
                "ack_count": flow.ack_count,
                "fin_count": flow.fin_count,
                "rst_count": flow.rst_count,
                "forward_sizes": flow.forward_sizes,
                "backward_sizes": flow.backward_sizes,
                "forward_inter_arrival": flow.forward_inter_arrival,
                "backward_inter_arrival": flow.backward_inter_arrival,
                "start_time": flow.start_time,
                "end_time": flow.end_time,
            }
            data.append(row)
        
        return pd.DataFrame(data)
    
    def get_stats(self) -> Dict:
        """Get current capture statistics."""
        packets = self.buffer.get_packets()
        
        if not packets:
            return {
                "packets_in_buffer": 0,
                "flows_active": 0,
                "bytes_per_second": 0,
                "packets_per_second": 0,
            }
        
        # Calculate rates
        now = time.time()
        recent_packets = [p for p in packets if now - p.timestamp < 1.0]
        
        return {
            "packets_in_buffer": len(packets),
            "flows_active": len(set(
                (p.src_ip, p.dst_ip, p.src_port, p.dst_port, p.protocol)
                for p in packets
            )),
            "bytes_per_second": sum(p.bytes for p in recent_packets),
            "packets_per_second": len(recent_packets),
        }
