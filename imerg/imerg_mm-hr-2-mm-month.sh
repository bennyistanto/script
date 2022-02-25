#!/bin/bash
# author: https://github.com/andyprasetya
# Convert IMERG Daily data (mm/hr) to monthly (mm/month)
# -- Start script: change directory ke target directory, misal: 'imerg'
cd imerg
# -- Set shell option ke 'nullglob' - optional, bisa di-comment/disabled
shopt -s nullglob
# Mulai loop ke seluruh file dalam target directory.
# Biasanya menggunakan 'i', tapi ini pakai 'file' untuk memperjelas penulisan variable
for file in *.nc4;
do
  # Untuk mengolah string, digunakan fungsi 'cut'
  # Variable PREFIX, yaitu 'imerg_'
  PREFIX=`echo "$file" | cut -c1-6`
  # Variable YEAR, dari imerg_YYYYMMDD.nc4
  YEAR=`echo "$file" | cut -c7-10 | bc`
  # Variable MM, dari imerg_YYYYMMDD.nc4
  # Ada yang khusus, jika ada prefix '0', seperti 01,02,..09, maka bilangannya jadi OCTAL.
  # Bilangan ini harus diconvert dulu ke base-10 dengan 'bc' karena akan dikalikan dengan 720, 744 dan 696
  MMMONTH=`echo "$file" | cut -c11-12 | bc`
  # Variable DD, dari imerg_YYYYMMDD.nc4
  DDDATE=`echo "$file" | cut -c13-14`
  # Variable EXTENSION, yaitu 'nc4'
  EXTENSION="${file##*.}"
  # Block: Identifikasi leap year modulus 4
  if [ $((YEAR % 4)) == 0 ]; then
    if [ $MMMONTH == 2 ]; then 
      # perlakuan jika MM nya '02'
      MMMONTH=`echo "$((MMMONTH * 696))"`
    elif [ $MMMONTH == 1 -o $MMMONTH == 3 -o $MMMONTH == 5 -o $MMMONTH == 7 -o $MMMONTH == 8 -o $MMMONTH == 10 -o $MMMONTH == 12 ]; then
      # perlakuan jika hasil perkalian < 1000, tambah prefix '0'
      if [ $((MMMONTH)) == 1 ]; then
        # tambah prefix '0'
        MMMONTH=`echo 0"$((MMMONTH * 744))"`
      else
        # tanpa prefix
        MMMONTH=`echo "$((MMMONTH * 744))"`
      fi
    else
      MMMONTH=`echo "$((MMMONTH * 720))"`
    fi
  else
    if [ $MMMONTH == 1 -o $MMMONTH == 3 -o $MMMONTH == 5 -o $MMMONTH == 7 -o $MMMONTH == 8 -o $MMMONTH == 10 -o $MMMONTH == 12 ]; then
      if [ $((MMMONTH)) == 1 ]; then
        # tambah prefix '0'
        MMMONTH=`echo 0"$((MMMONTH * 744))"`
      else
        # tanpa prefix
        MMMONTH=`echo "$((MMMONTH * 744))"`
      fi
    else
      MMMONTH=`echo "$((MMMONTH * 720))"`
    fi
  fi
  # Block operasi file - contoh: di-list saja dari filename original ke filename baru
  echo ${file} $PREFIX$YEAR$MMMONTH$DDDATE.$EXTENSION
  # Contoh: jika akan scan + copy + rename ke directory lain yang sejajar dengan 'imerg'
  # -- cp ${file} ../renamedfiles/$PREFIX$YEAR$MMMONTH$DDDATE.$EXTENSION
done
# command untuk check hasil operasi:
# ls -l ../renamedfiles
