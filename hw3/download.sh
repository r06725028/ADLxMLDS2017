#wget --no-check-certificate https://googledrive.com/host/ID -O /本地端的檔案儲存路徑與檔名
#wget --no-check-certificate "https://googledrive.com/host/12c1_2X0WcMXx18Vw0Aefx7Fv1riP5YEv" -O my_dqn_model.data-00000-of-00001
wget --no-check-certificate "https://drive.google.com/uc?export=download&id=12c1_2X0WcMXx18Vw0Aefx7Fv1riP5YEv" -O my_dqn_model.data-00000-of-00001
#-O my_dqn_model.data-00000-of-00001
#python3 test.py --test_dqn
python3 test.py --test_pg --test_dqn

#wget --no-check-certificate "https://drive.google.com/uc?export=download&id=IAmTheFileWithIDAsXXXXXXXXXXXXXXXXXX" -O OutPutFile


#https://drive.google.com/open?id=12c1_2X0WcMXx18Vw0Aefx7Fv1riP5YEv#知道連結可以檢視

#https://drive.google.com/file/d/12c1_2X0WcMXx18Vw0Aefx7Fv1riP5YEv/view?usp=sharing#公開在網路上


