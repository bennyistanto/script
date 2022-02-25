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


# function
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



os.chdir("Z:\\Temp\\DryWetSeason")

vp = VampireDefaults()
base_product_name = vp.get('CHIRPS','base_data_name')

def dekadRainfallAnomaly():
    average_pattern_dekad = vp.get('CHIRPS_Longterm_Average','global_lta_dekad_pattern')
    dekad_pattern = vp.get('CHIRPS', 'global_dekad_pattern')
    AVGdir_dekad = 'Z:\\Temp\\DryWetSeason\\Statistics_Month24_movingby_Dekad'
    dekaddir = 'Z:\\Temp\\DryWetSeason\\Month24_movingby_Dekad'
    ResultDir_dekad = 'Z:\\Temp\\DryWetSeason\\Output'
    AVGregex_dekad = re.compile(average_pattern_dekad)
    Moregex_dekad = re.compile(dekad_pattern)



    for Dfilename in os.listdir(dekaddir):
        if Dfilename.endswith(".tif") or Dfilename.endswith(".tiff"):
            #print(Dfilename)
            Moresult_dekad = Moregex_dekad.match(Dfilename)
            Dmonth = Moresult_dekad.group('month')
            Ddekad = Moresult_dekad.group('dekad')
            for ADfilename in os.listdir(AVGdir_dekad):
                if ADfilename.endswith(".tif"):
                    if AVGregex_dekad.match(ADfilename):
                        AVGresult_dekad = AVGregex_dekad.match(ADfilename)
                        SDmonth = AVGresult_dekad.group('month')
                        SDdekad = AVGresult_dekad.group('dekad')
                        if SDmonth == Dmonth and SDdekad == Ddekad:
                            #print(Dfilename+" match with "+ADfilename)
                            AVGFile_dekad = os.path.join(AVGdir_dekad, ADfilename)
                            MoFile_dekad = os.path.join(dekaddir, Dfilename)
                            month = SDmonth
                            year = Moresult_dekad.group('year')
                            ext = ".tif"
                            newfilename_dekad = '{0}.{1}.{2}.{3}.month24_ratio_anom{4}'.format(base_product_name, year, month, Ddekad, ext)
                            
                            print(newfilename_dekad)
                            if arcpy.Exists(os.path.join(ResultDir_dekad, newfilename_dekad)):
                                 print(newfilename_dekad + " is already exist")
                            else:
                                arcpy.CheckOutExtension("spatial")
                                newRaster_dekad = Int(100 * Raster(MoFile_dekad) / Raster(AVGFile_dekad))
                                newRaster_dekad.save(os.path.join(ResultDir_dekad, newfilename_dekad))
                                arcpy.CheckInExtension("spatial")
                        continue
                else:
                    continue
            continue
        else:
            continue

dekadRainfallAnomaly()
