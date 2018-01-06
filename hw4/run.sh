#!/bin/bash

#wget --no-check-certificate "https://drive.google.com/uc?export=download&id=12c1_2X0WcMXx18Vw0Aefx7Fv1riP5YEv" -O my_dqn_model.data-00000-of-00001
#wget --no-check-certificate "https://drive.google.com/uc?export=download&id=IAmTheFileWithIDAsXXXXXXXXXXXXXXXXXX" -O OutPutFile
#https://drive.google.com/open?id=12c1_2X0WcMXx18Vw0Aefx7Fv1riP5YEv#知道連結可以檢視
#https://drive.google.com/file/d/12c1_2X0WcMXx18Vw0Aefx7Fv1riP5YEv/view?usp=sharing#公開在網路上

#model : model.data-00000-of-00001
#https://drive.google.com/file/d/1Yd1tWGSuD3G90RWSRJbY1BUBBUhDYA1J/view?usp=sharing
#wget --no-check-certificate "https://drive.google.com/uc?export=download&id=1Yd1tWGSuD3G90RWSRJbY1BUBBUhDYA1J" -O model.data-00000-of-00001
if [ ! -f model.data-00000-of-00001 ]; then
    curl ftp://140.112.107.150/r06725053/model.data-00000-of-00001 -o model.data-00000-of-00001
fi
echo "model.data-00000-of-00001 ok!!!!"

#load : skipthoughts
#https://drive.google.com/drive/folders/11cq1IP15RJHLMDoB6eqJqY-ImfLgxZB-?usp=sharing
#wget --no-check-certificate "https://drive.google.com/uc?export=download&id=11cq1IP15RJHLMDoB6eqJqY-ImfLgxZB-" -O skipthoughts
if [ ! -f Data/skipthoughts/bi_skip.npz ]; then
    curl ftp://140.112.107.150/r06725053/skipthoughts/bi_skip.npz -o skipthoughts/bi_skip.npz
fi
echo "bi_skip.npz ok!!!!"

if [ ! -f Data/skipthoughts/btable.npy ]; then
    curl ftp://140.112.107.150/r06725053/skipthoughts/btable.npy -o skipthoughts/btable.npy
fi
echo "btable.npy ok!!!!"

if [ ! -f Data/skipthoughts/uni_skip.npz ]; then
    curl ftp://140.112.107.150/r06725053/skipthoughts/uni_skip.npz -o Data/skipthoughts/uni_skip.npz
fi
echo "uni_skip.npz ok!!!!"

if [ ! -f Data/skipthoughts/utable.npy ]; then
    curl ftp://140.112.107.150/r06725053/skipthoughts/utable.npy -o skipthoughts/utable.npy
fi
echo "utable.npy ok!!!!"

python3 generate.py $1
