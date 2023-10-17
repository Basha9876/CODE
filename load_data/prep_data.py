import numpy as np
import pandas as pd


def prepare_data(data):
    data['protocol_type'].replace(to_replace=['tcp', 'udp', 'icmp'],value=[0,1,2],inplace=True)
    data['service'].replace(to_replace=['ftp_data', 'other', 'private', 'http', 'remote_job', 'name',
            'netbios_ns', 'eco_i', 'mtp', 'telnet', 'finger', 'domain_u',
            'supdup', 'uucp_path', 'Z39_50', 'smtp', 'csnet_ns', 'uucp',
            'netbios_dgm', 'urp_i', 'auth', 'domain', 'ftp', 'bgp', 'ldap',
            'ecr_i', 'gopher', 'vmnet', 'systat', 'http_443', 'efs', 'whois',
            'imap4', 'iso_tsap', 'echo', 'klogin', 'link', 'sunrpc', 'login',
            'kshell', 'sql_net', 'time', 'hostnames', 'exec', 'ntp_u',
            'discard', 'nntp', 'courier', 'ctf', 'ssh', 'daytime', 'shell',
            'netstat', 'pop_3', 'nnsp', 'IRC', 'pop_2', 'printer', 'tim_i',
            'pm_dump', 'red_i', 'netbios_ssn', 'rje', 'X11', 'urh_i',
            'http_8001', 'aol', 'http_2784', 'tftp_u', 'harvest'],value=list(range(70)),inplace=True)
    data['flag'].replace(to_replace=['SF', 'S0', 'REJ', 'RSTR', 'SH', 'RSTO', 'S1', 'RSTOS0', 'S3',
            'S2', 'OTH'],value=list(range(11)),inplace=True)
    data['label'].replace(to_replace=['normal', 'neptune', 'warezclient', 'ipsweep', 'portsweep',
            'teardrop', 'nmap', 'satan', 'smurf', 'pod', 'back',
            'guess_passwd', 'ftp_write', 'multihop', 'rootkit',
            'buffer_overflow', 'imap', 'warezmaster', 'phf', 'land',
            'loadmodule', 'spy', 'perl'],value=list(range(23)),inplace=True)

    label = data['label']
    data = data.drop('label',axis=1)
    
    
    return data,label