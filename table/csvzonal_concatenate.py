#!/usr/bin/python
"""
NAME
    csvzonal_concat.py
    Concatenate csv files by column, and use the date on filename as column name
DESCRIPTION
    Input data for this script will use csv result from dbf2csv translation
    Some notes:
          (i)  All files required to follow the naming convention "_yyyymmdd.ext"
         (ii)  ArcGIS Zonal Statistics as Table is use to generate DBF file based on a zone
               Usually the result will have column: ID zone, ZONE_CODE, COUNT, AREA, MIN, MAX, MEAN
        (iii)  By utilizing dbf2csv tool (https://github.com/akadan47/dbf2csv), all DBFs are
               converted to csv using this command: dbf2csv . output/ -d $'\t' -q all
         (iv)  Then finally concatenate all the CSVs and you will get the output inside 
               results folder: area.csv, count.csv, max.csv, mean.csv, min.csv and zone_code.csv
               and the column will be the date information for each file: ID, YYYYMMDD
          (v)  Some adjustment are required: date on filename, ID/Zone name and type (string or integer)
USAGE
    python csvzonal_concat.py <csv_dir>
CONTACT
    Benny Istanto
    Climate Geographer
    GOST, The World Bank
LICENSE
    This script is in the public domain, free from copyrights or restrictions.
VERSION
    $Id$
TODO
    xx
"""
# usage: python csvzonal_concat.py <csv_dir>

import csv
import os
import re
import sys
from collections import defaultdict

csvdir = sys.argv[1]
cols = None
ymds = set()
fids = set()

#col => ymd => fid => value
result = defaultdict(lambda: defaultdict(dict))

for fname in os.listdir(csvdir):
    # Change value inside the curlybracket {}, if your filename has yyyymmdd then the value is 8
    m = re.match(r'.*_(\d{8}).csv', fname)
    if not m:
        continue
    print('Processing', fname)
    ymd = int(m.group(1))
    ymds.add(ymd)
    with open(csvdir + '/' + fname) as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    if cols is None:
        # Change text inside apostrophe '', this is the column used as a ZONE in Zonal Statistics
        # and will use as a KEYFIELD to concatenate
        cols = [c for c in rows[0].keys() if c != 'Value']
    for r in rows:
        # Check the value in the ZONE column, 
        # then adjust text 'str' below into 'int' for integer/number, or 'str' for string/text
        fid = int(r['Value']) # Dont forget to change text inside apostrophe ''
        fids.add(fid)
        for c in cols:
            result[c][ymd][fid] = r[c]

symds = list(sorted(ymds))
sfids = list(sorted(fids))
os.mkdir(csvdir + '/results')
for c in cols:
    fpath = f'{csvdir}/results/{c.lower()}.csv'
    with open(fpath, 'w') as f:
        w = csv.writer(f)
        w.writerow(['Value'] + symds) # Dont forget to change text inside apostrophe ''
        for fid in sfids:
            w.writerow([fid] + [
                result[c][ymd].get(fid)
                for ymd in symds
            ])
