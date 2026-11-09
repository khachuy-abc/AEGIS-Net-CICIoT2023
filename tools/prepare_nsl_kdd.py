"""
Turns the raw NSL-KDD text files (no header) into one CSV with a header, so main.py can read it like any other dataset.
Attack names are grouped into the 4 classic families (DoS, Probe, R2L, U2R); "normal" becomes Benign.
The last column of the raw files ("difficulty") is NOT a network feature and is removed.

    python tools/prepare_nsl_kdd.py --train KDDTrain+.txt --out nsl_kdd/nsl_kdd.csv
Download: https://github.com/defcom17/NSL_KDD
"""
import argparse
import os

import pandas as pd

COLS = ("duration protocol_type service flag src_bytes dst_bytes land wrong_fragment urgent hot num_failed_logins logged_in num_compromised "
        "root_shell su_attempted num_root num_file_creations num_shells num_access_files num_outbound_cmds is_host_login is_guest_login count "
        "srv_count serror_rate srv_serror_rate rerror_rate srv_rerror_rate same_srv_rate diff_srv_rate srv_diff_host_rate dst_host_count "
        "dst_host_srv_count dst_host_same_srv_rate dst_host_diff_srv_rate dst_host_same_src_port_rate dst_host_srv_diff_host_rate "
        "dst_host_serror_rate dst_host_srv_serror_rate dst_host_rerror_rate dst_host_srv_rerror_rate").split() + ["label", "difficulty"]
FAMILY = {}
for fam, names in dict(
        DoS="back land neptune pod smurf teardrop apache2 udpstorm processtable worm mailbomb",
        Probe="satan ipsweep nmap portsweep mscan saint",
        R2L="guess_passwd ftp_write imap phf multihop warezmaster warezclient spy xlock xsnoop snmpguess snmpgetattack httptunnel sendmail named",
        U2R="buffer_overflow loadmodule rootkit perl sqlattack xterm ps").items():
    FAMILY.update({n: fam for n in names.split()})

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    df = pd.read_csv(a.train, header=None, names=COLS).drop(columns=["difficulty"])
    df["label"] = df["label"].map(lambda v: "normal" if v == "normal" else FAMILY.get(v, v))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    df.to_csv(a.out, index=False)
    print(df["label"].value_counts().to_string())
