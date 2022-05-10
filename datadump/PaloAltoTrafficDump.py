#!/usr/bin/python3

"""
This script will dump data from a Palo alto PA-200 appliance.
The interaction will be with the ElasticSearch engine, for info
refer to https://www.elastic.co/partners/palo-alto-networks/
This script will interact  with Elastic API such as:
https://192.168.111.10:9200/"+indexName+"/_search?size=9999
With a given username and password.
In order to use it you will have to setup passwords and private
data using keyring as it follows:
    
keyring.set_password("palo_alto_dump", "username", "APIUsername")
keyring.set_password("palo_alto_dump", "password", "APIPassword")
keyring.set_password("palo_alto_dump", "indexname", "elasticindexname")

The script will produce a dumped file in .csv format.
"""

import requests
import json
from requests.packages import urllib3
import urllib.request
import urllib.parse
import pandas as pd
import datetime
import time
from datetime import datetime, timedelta
import sys
import getopt
import keyring

urllib3.disable_warnings()
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def GetAlerts(start_time, end_time):
    """
    This method will recieve data from the Elastic API.
    """
    indexName = keyring.get_password("palo_alto_dump", "indexname")
    username = keyring.get_password("palo_alto_dump", "username")
    password = keyring.get_password("palo_alto_dump", "password")
    eventUrl = "https://192.168.111.10:9200/"+indexName+"/_search?size=9999"
    headersGet = {'Accept': 'application/json',
                  'Content-type': 'application/json'}
    queryGet = json.dumps(
        {"query": {"range": {"@timestamp": {"gte": start_time, "lt": end_time}}}})
    responseGet = requests.post(eventUrl, headers=headersGet, auth=(
        username, password), verify=False, data=queryGet)

    return responseGet.json()


def get_chunk(start_time, end_time):
    """
    The puropouse of this method is to execute the previous GetAlterts in a 
    more structured way, doing basic field selection and sorting at the end.
    The data will be dumped in chunks of minutes_back minutes.
    """
    response = GetAlerts(start_time, end_time)
    i = 0
    columns_names = ["timestamp", "src_ip", "dst_ip", "src_port",
                     "dst_port", "transport", "application", "bytes_sent", "bytes_recieved"]
    df = pd.DataFrame(columns=columns_names)
    row_list = []
    for y in response['hits']['hits']:
        i = i + 1
        # The script will skip incomplete or mistaken records.
        error_flag = False
        try:
            timestamp = y["_source"]["@timestamp"]
            src_ip = y["_source"]["source"]["ip"]
            dst_ip = y["_source"]["destination"]["ip"]

            src_port = y["_source"]["source"]["port"]
            dst_port = y["_source"]["destination"]["port"]

            bytes_src = y["_source"]["source"]["bytes"]
            bytes_dst = y["_source"]["destination"]["bytes"]

            transport = y["_source"]["network"]["transport"]

            application = y["_source"]["network"]["application"]

            packets_src = y["_source"]["source"]["packets"]
            packets_dst = y["_source"]["destination"]["packets"]

            duration = y["_source"]["event"]["duration"]

        except:
            error_flag = True
        if not error_flag:
            row = {"timestamp": timestamp,
                   "src_ip": src_ip,
                   "dst_ip": dst_ip,
                   "src_port": src_port,
                   "dst_port": dst_port,
                   "bytes_src": bytes_src,
                   "bytes_dst": bytes_dst,
                   "transport": transport,
                   "application": application,

                   "packets_src": packets_src,
                   "packets_dst": packets_dst,
                   "duration": duration

                   }
            row_list.append(row)
    df = pd.DataFrame(row_list)
    return df


if __name__ == "__main__":
    #Fields that should be set:#
    # Num. of previous days to dump (Edit this field)
    num_days = 20
    # Time step of the field (Edit this field)
    minutes_back = 10
    # Script name
    filename = "1.04to20.02.csv"
    # Start datetime
    start_ts = "2022-04-01T01:00:00.000+01:00"

    # Dump script:
    n_blocks = num_days*(1440/minutes_back)
    start_ts_datetime = datetime.fromisoformat(start_ts)
    end_ts_datetime = start_ts_datetime - timedelta(minutes=minutes_back)
    df = get_chunk(end_ts_datetime.isoformat(), start_ts_datetime.isoformat())
    print("--------------------------------------")
    print("            Firewall Dump ")
    print("--------------------------------------")
    print("Dumping ", num_days, "time_step, ", minutes_back, "m")
    print("Total blocks: ", int(n_blocks))
    print("--------------------------------------")

    salva = 7
    contatore = 0
    start_time = time.time()  # Time taken for benchmark

    for i in range(1, int(n_blocks)):
        start_time2 = time.time()
        contatore = contatore + 1
        start_ts_datetime = end_ts_datetime
        end_ts_datetime = start_ts_datetime - timedelta(minutes=minutes_back)
        df1 = get_chunk(end_ts_datetime.isoformat(),
                        start_ts_datetime.isoformat())
        df = pd.concat([df, df1], sort=False)
        print("Blocco", i, " di ", int(n_blocks), " righe ", len(
            df), " tempo ", (time.time() - start_time2), "s")

    print("Sorting and saving")
    df = df.sort_values(by='timestamp', ascending=True)
    df.to_csv(filename, index=False)

    print("-----------------------------")
    print("Process terminated, dataset len: ", len(df))
    print("time: ", (time.time() - start_time), "s")
    print("-----------------------------")
    ended = input()
