#!/usr/bin/env python3
"""
preprocess.py

Script di preprocessing 5G-NIDD:
- Elabora i file PCAPNG nelle cartelle data/BS1 e data/BS2.
- Raggruppa i pacchetti in flussi, utilizzando:
    • una finestra temporale (time_window)
    • un numero massimo di pacchetti per flusso (flow_length)
- Per ogni flusso vengono calcolate feature aggregate:
    1. Durata del flusso (differenza tra il timestamp massimo e quello minimo)
    2. Numero totale di pacchetti
    3. Byte totali (somma delle lunghezze dei pacchetti)
    4. Dimensione media dei pacchetti
    5. Deviazione standard della dimensione dei pacchetti
    6. Intervallo medio tra pacchetti
    7. Deviazione standard degli intervalli
    8. TTL medio
    9. Proporzione di pacchetti TCP
   10. Proporzione di pacchetti UDP
   11. Proporzione di pacchetti GTP
- Viene assegnato un label al flusso: se gli IP (sorgente e destinazione) corrispondono a quelli
  definiti in mapping (attaccante e vittima) il flusso riceve il label specifico, altrimenti 0.
- Infine i dati aggregati vengono suddivisi in training (70%), test (15%) e validazione (15%)
  e salvati in file .npz come vettori compressi:
      • Per BS1: train_1, test_1, validation_1 e rispettivi label ytrain_1, xtest_1, xval_1.
      • Per BS2: train_2, test_2, validation_2 e rispettivi label ytrain_2, xtest_2, xval_2.
      
Utilizzo:
    python preprocess.py --flow_length 50 --time_window 10
"""

import os
import pyshark
import socket
import argparse
import glob
import numpy as np
from collections import OrderedDict
from sklearn.model_selection import train_test_split

# Mappatura per BS1: [ip_attacker, ip_victim, label]
mapping_bs1 = {
    "Goldeneye_BS1.pcapng": ["10.155.15.3", "10.41.150.68", 1],
    "ICMPflood_BS1.pcapng": ["10.155.15.8", "10.41.150.68", 2],
    "Slowloris_BS1.pcapng": ["10.155.15.0", "10.41.150.68", 3],
    "SSH_BS1.pcapng": ["10.155.15.7", "10.41.150.68", 4],
    "SYNflood_BS1.pcapng": ["10.155.15.4", "10.41.150.68", 5],
    "SYNScan_BS1.pcapng": ["10.155.15.1", "10.41.150.68", 6],
    "TCPConnect_BS1.pcapng": ["10.155.15.1", "10.41.150.68", 7],
    "Torshammer_BS1.pcapng": ["10.155.15.4", "10.41.150.68", 8],
    "UDPflood_BS1.pcapng": ["10.155.15.4", "10.41.150.68", 9],
    "UDPScan_BS1.pcapng": ["10.155.15.9", "10.41.150.68", 10]
}

# Mappatura per BS2 (gli IP possono variare in base al dataset)
mapping_bs2 = {
    "Goldeneye_BS2.pcapng": ["10.155.15.3", "10.41.150.68", 1],
    "ICMPflood_BS2.pcapng": ["10.155.15.8", "10.41.150.68", 2],
    "Slowloris_BS2.pcapng": ["10.155.15.0", "10.41.150.68", 3],
    "SSH_BS2.pcapng": ["10.155.15.7", "10.41.150.68", 4],
    "SYNflood_BS2.pcapng": ["10.155.15.4", "10.41.150.68", 5],
    "SYNScan_BS2.pcapng": ["10.155.15.1", "10.41.150.68", 6],
    "TCPConnect_BS2.pcapng": ["10.155.15.1", "10.41.150.68", 7],
    "Torshammer_BS2.pcapng": ["10.155.15.4", "10.41.150.68", 8],
    "UDPflood_BS2.pcapng": ["10.155.15.4", "10.41.150.68", 9],
    "UDPScan_BS2.pcapng": ["10.155.15.9", "10.41.150.68", 10]
}

def initialize_flow(packet, timestamp):
    """
    Inizializza un nuovo flusso a partire dal primo pacchetto.
    Estrae il 5-tuple (src_ip, src_port, dst_ip, dst_port, protocol) e inizializza le strutture per aggregare le feature.
    """
    try:
        src_ip = str(packet.ip.src)
        dst_ip = str(packet.ip.dst)
        protocol = int(packet.ip.proto)
    except AttributeError:
        return None

    src_port = 0
    dst_port = 0
    if hasattr(packet, 'tcp'):
        try:
            src_port = int(packet.tcp.srcport)
            dst_port = int(packet.tcp.dstport)
        except:
            pass
    elif hasattr(packet, 'udp'):
        try:
            src_port = int(packet.udp.srcport)
            dst_port = int(packet.udp.dstport)
        except:
            pass

    flow_key = (src_ip, src_port, dst_ip, dst_port, protocol)
    flow = {
        "timestamps": [timestamp],
        "packet_lengths": [],
        "ip_ttl": [],
        "total_bytes": 0,
        "protocol_counts": {"TCP": 0, "UDP": 0, "GTP": 0}
    }
    try:
        pkt_len = int(packet.ip.len)
    except:
        pkt_len = 0
    flow["packet_lengths"].append(pkt_len)
    flow["total_bytes"] += pkt_len
    try:
        ttl = int(packet.ip.ttl)
    except:
        ttl = 0
    flow["ip_ttl"].append(ttl)
    if hasattr(packet, 'tcp'):
        flow["protocol_counts"]["TCP"] += 1
    elif hasattr(packet, 'udp'):
        flow["protocol_counts"]["UDP"] += 1
    if hasattr(packet, 'gtp'):
        flow["protocol_counts"]["GTP"] += 1

    return flow_key, flow

def update_flow(flow, packet, timestamp):
    """
    Aggiorna un flusso esistente con un nuovo pacchetto.
    """
    flow["timestamps"].append(timestamp)
    try:
        pkt_len = int(packet.ip.len)
    except:
        pkt_len = 0
    flow["packet_lengths"].append(pkt_len)
    flow["total_bytes"] += pkt_len
    try:
        ttl = int(packet.ip.ttl)
    except:
        ttl = 0
    flow["ip_ttl"].append(ttl)
    if hasattr(packet, 'tcp'):
        flow["protocol_counts"]["TCP"] += 1
    elif hasattr(packet, 'udp'):
        flow["protocol_counts"]["UDP"] += 1
    if hasattr(packet, 'gtp'):
        flow["protocol_counts"]["GTP"] += 1
    return flow

def aggregate_flow_features(flow):
    """
    Calcola le feature aggregate per un flusso:
      1. Durata del flusso
      2. Numero totale di pacchetti
      3. Byte totali
      4. Dimensione media dei pacchetti
      5. Deviazione standard della dimensione dei pacchetti
      6. Intervallo medio tra pacchetti
      7. Deviazione standard degli intervalli
      8. TTL medio
      9. Proporzione di pacchetti TCP
     10. Proporzione di pacchetti UDP
     11. Proporzione di pacchetti GTP
    """
    timestamps = np.array(flow["timestamps"])
    packet_lengths = np.array(flow["packet_lengths"])
    
    duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0
    total_packets = len(timestamps)
    total_bytes = flow["total_bytes"]
    avg_pkt_len = np.mean(packet_lengths) if total_packets > 0 else 0
    std_pkt_len = np.std(packet_lengths) if total_packets > 0 else 0
    
    if total_packets > 1:
        inter_arrivals = np.diff(timestamps)
        avg_interarrival = np.mean(inter_arrivals)
        std_interarrival = np.std(inter_arrivals)
    else:
        avg_interarrival = 0
        std_interarrival = 0
    
    avg_ttl = np.mean(flow["ip_ttl"]) if len(flow["ip_ttl"]) > 0 else 0
    
    tcp_count = flow["protocol_counts"]["TCP"]
    udp_count = flow["protocol_counts"]["UDP"]
    gtp_count = flow["protocol_counts"]["GTP"]
    prop_tcp = tcp_count / total_packets if total_packets > 0 else 0
    prop_udp = udp_count / total_packets if total_packets > 0 else 0
    prop_gtp = gtp_count / total_packets if total_packets > 0 else 0
    
    feature_vector = [
        duration,
        total_packets,
        total_bytes,
        avg_pkt_len,
        std_pkt_len,
        avg_interarrival,
        std_interarrival,
        avg_ttl,
        prop_tcp,
        prop_udp,
        prop_gtp
    ]
    return feature_vector

def process_pcap_file(pcap_file, mapping, max_flow_len, time_window):
    """
    Elabora un file PCAPNG raggruppando i pacchetti in flussi.
    """
    flows_dict = OrderedDict()  # chiave: (src, srcport, dst, dstport, protocol) -> { window_start: flow }
    cap = pyshark.FileCapture(pcap_file, keep_packets=False)
    
    for i, pkt in enumerate(cap):
        try:
            timestamp = float(pkt.sniff_timestamp)
        except:
            continue
        
        added = False
        # Verifica se il pacchetto rientra in un flusso esistente nella finestra temporale
        for key in flows_dict:
            for window_start, flow in flows_dict[key].items():
                if timestamp <= window_start + time_window and len(flow["timestamps"]) < max_flow_len:
                    try:
                        pkt_src = str(pkt.ip.src)
                        pkt_dst = str(pkt.ip.dst)
                        pkt_proto = int(pkt.ip.proto)
                    except:
                        continue
                    if key[0] == pkt_src and key[2] == pkt_dst and key[4] == pkt_proto:
                        flows_dict[key][window_start] = update_flow(flow, pkt, timestamp)
                        added = True
                        break
            if added:
                break
        
        if not added:
            res = initialize_flow(pkt, timestamp)
            if res is None:
                continue
            key, new_flow = res
            if key not in flows_dict:
                flows_dict[key] = {}
            flows_dict[key][timestamp] = new_flow
        
        if i % 1000 == 0:
            print(f"{pcap_file} - processati {i} pacchetti")
    cap.close()
    
    # Aggrega le feature per ciascun flusso e assegna il label in base alla mapping
    aggregated_flows = []
    filename =  os.path.basename(pcap_file)
    if filename in mapping:
        ip_attacker, ip_victim, file_label = mapping[filename]
    else:
        ip_attacker, ip_victim, file_label = None, None, 0
    
    for key, windows in flows_dict.items():
        for window_start, flow in windows.items():
            features = aggregate_flow_features(flow)
            src_ip, _, dst_ip, _, _ = key
            # Se il flusso coinvolge gli IP attaccante e vittima (in qualsiasi ordine) assegna il label dell'attacco, altrimenti 0
            flow_label = file_label if (ip_attacker is not None and ((src_ip == ip_attacker and dst_ip == ip_victim) or (src_ip == ip_victim and dst_ip == ip_attacker))) else 0
            aggregated_flows.append((key, features, flow_label))
    
    print(f"Totale flussi aggregati in {pcap_file}: {len(aggregated_flows)}")
    return aggregated_flows

def process_folder(folder_path, mapping, max_flow_len, time_window):
    """
    Elabora tutti i file PCAPNG in una cartella e aggrega i flussi.
    """
    all_flows = []
    pcap_files = glob.glob(os.path.join(folder_path, "*.pcapng"))
    for pcap_file in pcap_files:
        print(f"Elaborazione file: {pcap_file}")
        flows = process_pcap_file(pcap_file, mapping, max_flow_len, time_window)
        all_flows.extend(flows)
    return all_flows

def split_and_save(flows, save_path, prefix):
    """
    Separa le feature e i label dai flussi aggregati, esegue lo split in training (70%), test (15%) e validazione (15%)
    e salva i vettori compressi in un file .npz.
    """
    X = []
    y = []
    for _, features, label in flows:
        X.append(features)
        y.append(label)
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=int)
    
    # Split: 70% training, 15% test, 15% validazione
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, shuffle=True)
    
    np.savez_compressed(save_path,
                        **{f"train_{prefix}": X_train,
                           f"test_{prefix}": X_test,
                           f"validation_{prefix}": X_val,
                           f"ytrain_{prefix}": y_train,
                           f"xtest_{prefix}": y_test,
                           f"xval_{prefix}": y_val})
    print(f"Vettori compressi salvati in {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Preprocessing avanzato per 5G-NIDD")
    parser.add_argument("--flow_length", type=int, default=50, help="Numero massimo di pacchetti per flusso")
    parser.add_argument("--time_window", type=int, default=10, help="Finestra temporale (in secondi) per raggruppare i pacchetti")
    args = parser.parse_args()
    
    # Definisce il percorso: lo script è in src, i dati in ../data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    data_dir = os.path.join(project_root, "data")
    
    # Elaborazione BS1
    bs1_folder = os.path.join(data_dir, "BS1")
    print("Elaborazione dei file BS1...")
    flows_bs1 = process_folder(bs1_folder, mapping_bs1, args.flow_length, args.time_window)
    bs1_save_path = os.path.join(data_dir, "BS1_flows.npz")
    split_and_save(flows_bs1, bs1_save_path, prefix="1")
    
    # Elaborazione BS2
    bs2_folder = os.path.join(data_dir, "BS2")
    print("Elaborazione dei file BS2...")
    flows_bs2 = process_folder(bs2_folder, mapping_bs2, args.flow_length, args.time_window)
    bs2_save_path = os.path.join(data_dir, "BS2_flows.npz")
    split_and_save(flows_bs2, bs2_save_path, prefix="2")
    
if __name__=="__main__":
    main()
