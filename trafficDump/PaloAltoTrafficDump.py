
#!/usr/bin/python3

import requests
import json
from requests.auth import HTTPBasicAuth
from requests.packages import urllib3
from logging import StreamHandler
import urllib.request
import urllib.parse
import pandas as pd 
import datetime
import time
from datetime import datetime, timedelta
import sys, getopt

urllib3.disable_warnings()
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)



def GetAlerts(start_time,end_time):
    indexName = "filebeat-paloaltofw-*"
    username = "alessandro-mini"
    password = "xdLbk:!x7H2Ks4N"
    eventUrl = "https://192.168.111.10:9200/"+indexName+"/_search?size=9999"
    headersGet = {'Accept': 'application/json', 'Content-type': 'application/json'}
    queryGet = json.dumps({"query": {"range": {"@timestamp": {"gte": start_time,"lt": end_time}}}})
    responseGet = requests.post(eventUrl, headers=headersGet, auth=(username,password), verify=False, data=queryGet)
    
    return responseGet.json()

#datetime_object = datetime.fromisoformat(timestamp)


def get_chunk(start_time,end_time):
    response = GetAlerts(start_time,end_time)
    i = 0
    columns_names= ["timestamp","src_ip","dst_ip","src_port","dst_port","transport","application","bytes_sent","bytes_recieved"]
    df = pd.DataFrame(columns = columns_names)
    row_list = []
    for y in response['hits']['hits']:
        i = i + 1 
        error_flag = False  #Saltare record incompleti.
        try:
            timestamp = y["_source"]["@timestamp"]
            src_ip = y["_source"]["client"]["ip"]
            dst_ip = y["_source"]["server"]["ip"]
            src_port = y["_source"]["client"]["port"]
            dst_port = y["_source"]["server"]["port"]
            packets = y["_source"]["client"]["packets"]
            transport = y["_source"]["network"]["transport"]
            application = y["_source"]["network"]["application"]
            bytes_sent = y["_source"]["client"]["packets"]
            bytes_recieved = y["_source"]["server"]["packets"]    
        except:
            #print("[",i,"]  Record Incompleto")
            error_flag = True
        if not error_flag:
            row = {"timestamp":timestamp,
                   "src_ip":src_ip,
                   "dst_ip":dst_ip,
                   "src_port":src_port,
                   "dst_port":dst_port,
                   "transport":transport,
                   "application":application,
                   "bytes_sent":bytes_sent,
                   "bytes_recieved":bytes_recieved,
                   "packets":packets,
                   }
            row_list.append(row)
            #print("[",i,"]",src_ip, "-> ", dst_ip, " : ", dst_port)
    df = pd.DataFrame(row_list)   
    #df = df.sort_values(by='timestamp', ascending=True)
    #df = df.reset_index(drop = True)
    return df




#Script per dumpare n giorni di traffico a slot di minute_back
num_days=45
minutes_back = 10
n_blocks= num_days*(1440/minutes_back)
filename = "1.04to20.02.csv"
start_ts = "2022-04-01T01:00:00.000+01:00"
start_ts_datetime = datetime.fromisoformat(start_ts)
end_ts_datetime = start_ts_datetime - timedelta(minutes=minutes_back)
df = get_chunk(end_ts_datetime.isoformat(),start_ts_datetime.isoformat())
print("--------------------------------------")
print("            Firewall Dump ")
print("--------------------------------------")
print("Dumping ", num_days, "time_step, ",minutes_back,"m")
print("Total blocks: ", int(n_blocks))
print("--------------------------------------")

#288 sono i blocchi di 5 minuti in 1 giorno

#288 sono i blocchi di 5 minuti in 1 giorno
#Ciclo inizia da 1 perchè prima iterazione salta.
salva = 7
contatore = 0
start_time = time.time() #Benchmark



for i in range(1,int(n_blocks)):    
    start_time2 = time.time()
    contatore = contatore + 1 
    start_ts_datetime = end_ts_datetime
    end_ts_datetime = start_ts_datetime - timedelta(minutes=minutes_back)
    df1 = get_chunk(end_ts_datetime.isoformat(),start_ts_datetime.isoformat())    
    df = pd.concat([df,df1], sort=False)
    print("Blocco" , i , " di ",int(n_blocks), " righe "  , len(df), " tempo ",(time.time() - start_time2) , "s")

print("Sorting and saving")
df = df.sort_values(by='timestamp', ascending=True)
df.to_csv(filename,index= False)


print("-----------------------------")
print("Process terminated, dataset len: ", len(df))
print("time: ", (time.time() - start_time) , "s")
print("-----------------------------")

