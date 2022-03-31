# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 15:06:27 2022

@author: Alessandro Mini
"""

from ipwhois import IPWhois
obj = IPWhois('170.72.100.196')
print("Owner : ", obj.lookup_whois()["nets"][-1]["name"])