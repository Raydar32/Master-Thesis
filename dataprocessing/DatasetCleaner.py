# -*- coding: utf-8 -*-
"""
This script is a datacleaner for Ergon s.r.l, it implements the
DataCleaner interface.
"""

import pandas as pd
from dataprocessing.DataCleanerInterface import DataCleaner
from datetime import timedelta


class ErgonDataCleaner(DataCleaner):

    def setMinimumHoursToBeClustered(self, hoursMinimum):
        """
        This methods sets a minimum (k in thesis) of hours of traffic
        that a user must produce to be clustered.
        """
        self.hoursMinimum = hoursMinimum

    def cleanDataset(self):
        """
        This method cleans the dataset and it is an implementation of 
        the abstract method in the interface.
        """
        application_exclusions = ["ms-product-activation",
                                  "ms-update",
                                  "incomplete",
                                  "ms-ds-smbv3",
                                  "ntp-base",
                                  "icmp",
                                  "zabbix",
                                  "informix",
                                  "insufficient-data",
                                  "dhcp",
                                  "ms-local-security-management",
                                  "collectd",
                                  "ms-update-optimization-p2p",
                                  "ntp",
                                  "ping"]

        # Start processing the dataset.
        self.vprint("processing ", self.datasetPath.name)

        self.vprint("Removing duplicate data ", len(
            self.df)-len(self.df.drop_duplicates()))
        self.df = self.df.drop_duplicates()
        self.vprint("Removing junk data ...")
        self.df = self.df[~ self.df.src_port.eq(0)]
        self.df = self.df[~ self.df.dst_port.eq(0)]
        self.df = self.df[~ self.df.bytes_src.eq(0)]
        self.df = self.df[~ self.df.bytes_dst.eq(0)]

        self.vprint("Applying protocols exclusion rules ...")
        self.df = self.df[~self.df['application'].isin(application_exclusions)]

        self.vprint("Removing night traffic from 18:00 to 8:00")
        self.df["timestamp"] = pd.to_datetime(self.df["timestamp"], utc=True)
        self.df = self.df.set_index('timestamp')
        night = self.df.between_time('18:00', '8:00')
        self.df = self.df[~self.df.index.isin(night.index)]
        self.vprint("Removing non-business days")
        notBusinessDay = self.df[self.df.index.dayofweek >= 5]
        self.df = self.df[~self.df.index.isin(notBusinessDay.index)]
        self.df = self.df.reset_index()

        k = self.hoursMinimum

        self.vprint("Identifying sources with more then ",
                    k, " hours of traffic ")
        self.df["timestamp"] = pd.to_datetime(self.df["timestamp"], utc=True)

        sufficient = []
        insufficient = []
        for ip in self.df["src_ip"].unique():

            delta = self.df.loc[self.df["src_ip"] == ip]["timestamp"].iat[-1] - \
                self.df.loc[self.df["src_ip"] == ip]["timestamp"].iat[0]

            if delta > timedelta(hours=k):
                sufficient.append(ip)

            else:
                insufficient.append(ip)

        # At the end the dataset is sorted.
        self.df = self.df[~self.df['src_ip'].isin(insufficient)]
        self.df = self.df.sort_values(by='timestamp', ascending=True)
