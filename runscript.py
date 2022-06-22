import os
import sys

exp = sys.argv[1]
fromR = int(sys.argv[2])
toR = int(sys.argv[3])
program = sys.argv[4]

currentpath = os.getcwd()
os.system('mkdir '+exp)
os.chdir(currentpath+'/'+exp)
currentpath = os.getcwd()

for k in range(fromR,toR):
    print(k)
    os.system('mkdir '+str(k))
    os.chdir(currentpath+'/'+str(k))
    os.system('time ../../'+program)
    os.chdir(currentpath)
