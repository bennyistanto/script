#!/usr/bin/env python
# Import system modules
import optparse, sys, os, traceback, errno
import ast
import re
import json
import ConfigParser
import re
import logging
import arcpy
from arcpy.sa import *


#function
class VampireDefaults:

    def __init__(self):
        # set up logging
        self.logger = logging.getLogger('Vampire')
        logging.basicConfig(filename='vampire.log', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            level=logging.DEBUG, filemode='w')


        # load default values from .ini file
        self.config = ExtParser()
        cur_dir = os.path.join(os.getcwd(), 'vampire.ini')
        ini_files = ['vampire.ini',
                     os.path.join(os.getcwd(), 'vampire.ini'),
                     cur_dir]
        dataset = self.config.read(ini_files)
        if len(dataset) == 0:
            msg = "Failed to open/find vampire.ini in {0}, {1} and {2}".format(ini_files[0], ini_files[1], ini_files[2])
            raise ValueError, msg
        self.countries = dict(self.config.items('country'))
        self.countries = dict((k.title(), v) for k, v in self.countries.iteritems())
        self.country_codes_l = []
        self.country_codes = {}

        for c in self.countries:
            cc = ast.literal_eval(self.countries[c].replace("\n", ""))
            if 'chirps_boundary_file' in ast.literal_eval(self.countries[c].replace("\n", "")):
                _chirps_boundary_file = ast.literal_eval(self.countries[c].replace("\n", ""))['chirps_boundary_file']
                p = re.match(r'.*\$\{(?P<param>.*)\}.*', _chirps_boundary_file)
                if p:
                    # has a reference
                    _chirps_boundary_file = _chirps_boundary_file.replace('${'+p.group('param')+'}',
                                                            self.config.get('CHIRPS', p.group('param')))
                    cc['chirps_boundary_file'] = _chirps_boundary_file
            if 'modis_boundary_file' in ast.literal_eval(self.countries[c].replace("\n", "")):
                _modis_boundary_file = ast.literal_eval(self.countries[c].replace("\n", ""))['modis_boundary_file']
                p = re.match(r'.*\$\{(?P<param>.*)\}.*', _modis_boundary_file)
                if p:
                    # has a reference
                    _modis_boundary_file = _modis_boundary_file.replace('${'+p.group('param')+'}',
                                                            self.config.get('MODIS', p.group('param')))
                    cc['modis_boundary_file'] = _modis_boundary_file
            self.countries[c] = cc
            self.country_codes[cc['abbreviation']] = c
            self.country_codes_l.append(cc['abbreviation'])
        return

    def get(self, section, item):
        return self.config.get(section, item)

    def get_home_country(self):
        return self.config.get('vampire', 'home_country')

    def get_country(self, country=None):
        if not country:
            return self.countries
        return self.countries[country]

    def get_country_code(self, country=None):
        if country is None:
            return self.country_codes
        if country in self.countries:
            return self.countries[country]['abbreviation']
        return

    def get_country_name(self, country_code):
        for c in self.countries:
            if country_code.lower() == self.countries[c]['abbreviation'].lower():
                return c
        return None

    def print_defaults(self):
        print self.config._sections

#function
class ExtParser(ConfigParser.SafeConfigParser):
     #implementing extended interpolation
     def __init__(self, *args, **kwargs):
         self.cur_depth = 0
         ConfigParser.SafeConfigParser.__init__(self, *args, **kwargs)


     def get(self, section, option, raw=False, vars=None):
         r_opt = ConfigParser.SafeConfigParser.get(self, section, option, raw=True, vars=vars)
         if raw:
             return r_opt

         ret = r_opt
         re_oldintp = r'%\((\w*)\)s'
         re_newintp = r'\$\{(\w*):(\w*)\}'

         m_new = re.findall(re_newintp, r_opt)
         if m_new:
             for f_section, f_option in m_new:
                 self.cur_depth = self.cur_depth + 1
                 if self.cur_depth < ConfigParser.MAX_INTERPOLATION_DEPTH:
                     sub = self.get(f_section, f_option, vars=vars)
                     ret = ret.replace('${{{0}:{1}}}'.format(f_section, f_option), sub)
                 else:
                     raise ConfigParser.InterpolationDepthError, (option, section, r_opt)



         m_old = re.findall(re_oldintp, r_opt)
         if m_old:
             for l_option in m_old:
                 self.cur_depth = self.cur_depth + 1
                 if self.cur_depth < ConfigParser.MAX_INTERPOLATION_DEPTH:
                     sub = self.get(section, l_option, vars=vars)
                     ret = ret.replace('%({0})s'.format(l_option), sub)
                 else:
                     raise ConfigParser.InterpolationDepthError, (option, section, r_opt)

         self.cur_depth = self.cur_depth - 1
         return ret


vp = VampireDefaults()
nmonthly = ['01','02','03','04', '05', '06', '07', '08', '09', '10', '11', '12']
ndekad = ['1','2','3']

data_folder = "Z:\\Temp\\DryWetSeason\\Month12_movingby_Dekad"
lta_folder = "Z:\\Temp\\DryWetSeason\\Statistics_Month12_movingby_Dekad"
dekad_pattern = vp.get('CHIRPS', 'global_dekad_pattern') # global_monthly_pattern, global_seasonal_pattern, global_dekad_pattern


Moregex_dekad = re.compile(dekad_pattern)

def dekadLT():
    dictionary = {}
    for i in nmonthly:
        for j in ndekad:
            index = i+j
            content = []
            for file_dekad in os.listdir(data_folder):
                if file_dekad.endswith(".tif") or file_dekad.endswith(".tiff"):
                    Moresult_dekad = Moregex_dekad.match(file_dekad)
                    Dmonth = Moresult_dekad.group('month')
                    Ddekad = Moresult_dekad.group('dekad')
                    if Ddekad == j and Dmonth == i:
                       content.append(os.path.join(data_folder, file_dekad))
            dictionary[index] = content

    for k in nmonthly:
        for l in ndekad:
            index = k + l
            listoffile = dictionary[index]
            ext = ".tif"
    
            newfilename_dekad_avg = 'chirps-v2.0.1981-2019.{0}.{1}.dekad.39yrs.avg{2}'.format(k, l, ext)
            newfilename_dekad_std = 'chirps-v2.0.1981-2019.{0}.{1}.dekad.39yrs.std{2}'.format(k, l, ext)
            newfilename_dekad_max = 'chirps-v2.0.1981-2019.{0}.{1}.dekad.39yrs.max{2}'.format(k, l, ext)
            newfilename_dekad_min = 'chirps-v2.0.1981-2019.{0}.{1}.dekad.39yrs.min{2}'.format(k, l, ext)
            print(newfilename_dekad_avg)

            if arcpy.Exists(os.path.join(lta_folder, newfilename_dekad_avg)):
                print(newfilename_dekad_avg + " exists")
            else:
                arcpy.CheckOutExtension("spatial")
                outCellStatistics_avg = CellStatistics(listoffile, "MEAN", "DATA")
                outCellStatistics_avg.save(os.path.join(lta_folder, newfilename_dekad_avg))
                arcpy.CheckInExtension("spatial")

            if arcpy.Exists(os.path.join(lta_folder, newfilename_dekad_std)):
                print(newfilename_dekad_std + " exists")
            else:
                arcpy.CheckOutExtension("spatial")
                outCellStatistics_std = CellStatistics(listoffile, "STD", "DATA")
                outCellStatistics_std.save(os.path.join(lta_folder, newfilename_dekad_std))
                arcpy.CheckInExtension("spatial")

            if arcpy.Exists(os.path.join(lta_folder, newfilename_dekad_max)):
                print(newfilename_dekad_max + " exists")
            else:
                arcpy.CheckOutExtension("spatial")
                outCellStatistics_max = CellStatistics(listoffile, "MAXIMUM", "DATA")
                outCellStatistics_max.save(os.path.join(lta_folder, newfilename_dekad_max))
                arcpy.CheckInExtension("spatial")

            if arcpy.Exists(os.path.join(lta_folder, newfilename_dekad_min)):
                print(newfilename_dekad_min + " exists")
            else:
                arcpy.CheckOutExtension("spatial")
                outCellStatistics_min = CellStatistics(listoffile, "MINIMUM", "DATA")
                outCellStatistics_min.save(os.path.join(lta_folder, newfilename_dekad_min))
                arcpy.CheckInExtension("spatial")

dekadLT()
