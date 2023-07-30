import requests
import json
import pandas as pd
import numpy as np
from pandas import ExcelWriter
from ratelimit import *
import http.client, urllib.request, urllib.parse, urllib.error, base64, json

"""
headers = {   # Request headers 
    'Ocp-Apim-Subscription-Key': 'e21a5038378c450799372c7ecb0238b6' # Don't forget to put your keys in xxxx
}

"""

#1 วิ เรียกได้ 5 ครั้ง

class RateLimiter:
    def __init__(self, headers):
        headers = headers
        return
    @rate_limited(1500, 300)
    def call_get_api(self, url):
        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            print('Cannot call API: {}'.format(response.status_code))
        return response

limiter = RateLimiter(headers)


def AMC_requests ():
    req = requests.get(f'https://api.sec.or.th/FundFactsheet/fund/amc', headers = headers)
    df_AMC = pd.read_json(req.content)
    return df_AMC

def fund_requests (amc) :
    all_funds = pd.DataFrame(columns=['proj_id', 'proj_abbr_name','proj_name_en', 'proj_name_th','unique_id'])
    
    for unique_id in amc.unique_id:
        req = requests.get(f'https://api.sec.or.th/FundFactsheet/fund/amc/{unique_id}', headers = headers)
        projects = pd.read_json(req.content)
        all_funds = all_funds.append(projects[['proj_id', 'proj_abbr_name','proj_name_en', 'proj_name_th','unique_id']])
    return all_funds