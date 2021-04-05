#! /bin/bash

WGET=`which wget`
if [ "$WGET" = '' -o ! -x "$WGET" ]; then
  echo "This script requires the wget command"
  exit 1
fi 
CONVERT=`which convert`
if [ "$CONVERT" = '' -o ! -x "$CONVERT" ]; then
  echo "This script requires the ImageMagick convert command"
  exit 1
fi

#
# Get some standard images
#
if [ ! -d Std ]; then
    mkdir Std
    cd Std
    wget -O lena.tif http://sipi.usc.edu/database/misc/4.2.04.tiff
    wget -O mandrill.tif http://sipi.usc.edu/database/misc/4.2.03.tiff
    wget -O airplane.tif http://sipi.usc.edu/database/misc/4.2.05.tiff
    wget -O bridge.grey.tif http://sipi.usc.edu/database/misc/5.2.10.tiff
    wget -O peppers.tif http://sipi.usc.edu/database/misc/4.2.07.tiff
    wget -O boats.tif http://sipi.usc.edu/database/misc/boat.512.tiff
    wget -O man.grey.tif http://sipi.usc.edu/database/misc/5.3.01.tiff
    wget -O house.tif http://sipi.usc.edu/database/misc/4.1.05.tiff
    for file in *.tif; do
	basefile=`basename $file .tif`
	pngfile=$basefile.png
	echo "Converting $file to $pngfile"
	convert $file $pngfile
	/bin/rm -f $file
    done
    wget -O boats.bmp http://www.hlevkin.com/TestImages/BoatsColor.bmp
    wget -O barbara.bmp http://www.hlevkin.com/TestImages/barbara.bmp
    wget -O goldhill.bmp http://www.hlevkin.com/TestImages/goldhill.bmp
    wget -O monarch.bmp http://www.hlevkin.com/TestImages/monarch.bmp
    wget -O kiel.grey.bmp http://www.hlevkin.com/TestImages/kiel.bmp
    for file in *.bmp; do
	basefile=`basename $file .bmp`
	pngfile=$basefile.png
	echo "Converting $file to $pngfile"
	convert $file $pngfile
	/bin/rm -f $file
    done
    wget -O barbara.grey.png https://web.archive.org/web/20070209141039/http://decsai.ugr.es/~javier/denoise/barbara.png
    wget -O lena.grey.png https://web.archive.org/web/20070328214632/http://decsai.ugr.es/~javier/denoise/lena.png
    cd ..
fi

#
# Get Kodak test images
#
if [ ! -d Kodak ]; then
    mkdir Kodak
    cd Kodak
    n=1
    while [ $n -le 24 ]; do
      f=`printf "kodim%02d.png" $n`
      if [ ! -f "$f" ]; then
        wget http://r0k.us/graphics/kodak/kodak/$f
      fi
      n=`expr $n + 1`
    done
    cd ..
fi

exit 0
