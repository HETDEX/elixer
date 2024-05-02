from __future__ import print_function
import sys


#very little error control
#basically, consume a *_fib.txt file and spit out id_string, amp, relative fiber number, CCD X, CCD Y
args = list(map(str.lower,sys.argv)) #python3 map is no longer a list, so need to cast here

fn = args[1]


def parse_panacea_fits_name(name):
    d = {}
    if name is not None:
        toks = name.split("_")  # multi_fits_basename = "multi_" + self.specid + "_" + self.ifu_slot_id + "_" + self.ifu_id + "_"
        if len(toks) == 6:
            try:
                d['specid'] = toks[1].zfill(3)
                d['ifuslot'] = toks[2].zfill(3)
                d['ifuid'] = toks[3].zfill(3)
                d['amp'] = toks[4][0:2]
                d['side'] = toks[4][0]
            except:
                pass
        return d

#AMP  = ["LU","LL","RL","RU"] #in order from bottom to top
#AMP_OFFSET = {"LU":1,"LL":113,"RL":225,"RU":337}
AMP_OFFSET = {"LU":0,"LL":112,"RL":224,"RU":336}

def relative_fibnum(fnum_448,amp):
    #input is the fiber number as 1-448
    #amp is LU, LL, RU, RL
    #output is the fiber number 1-112 for the amp
    #return fnum_448 - AMP_OFFSET[amp] + 1
    return AMP_OFFSET[amp] + 112 - fnum_448

def parse_line(line):
    if line[0] == '#':
        return None

    specid = []
    ifuslot = []
    ifuid =  []
    amp = []
    side = []

    date = []
    obsid = []
    expid = []

    fnum448 =[]
    fnum112=[]
    ccd_x = []
    ccd_y = []

    toks = line.split()

    i = 17 # first fiber id
    while i < len(toks):
        name = toks[i] #17
        d = parse_panacea_fits_name(name)

        if len(d) < 1:
            break

        specid.append(d['specid'])
        ifuslot.append(d['ifuslot'])
        ifuid.append(d['ifuid'])
        amp.append(d['amp'])
        side.append(d['side'])

        i += 1 #18
        date.append(toks[i])

        i += 1 #19
        obsid.append(toks[i])

        i +=1 #20
        expid.append(toks[i])

        i += 1 #21
        fnum448.append(toks[i])
        fnum112.append(str(relative_fibnum(int(toks[i]),d['amp'])))

        i += 4 #25
        ccd_x.append(toks[i])

        i += 1 #26
        ccd_y.append(toks[i])

        i += 1 #27 (next  entry)


    i = 0
    for i in range(len(specid)):
        print(date[i],obsid[i].zfill(7),expid[i].zfill(2),specid[i],ifuslot[i],ifuid[i],side[i],amp[i],fnum448[i].zfill(3),
              fnum112[i].zfill(3),ccd_x[i].zfill(4),ccd_y[i].zfill(4))


with open(fn,'r') as f:
    print("#    ", "        ", "    ", "spc", "slt", "ifu", " ", "  ", "fib", "fib", "CCD",  " CCD ")
    print("#date", "   obsid", "  ex", "id ", "id ", "id ", "s", "am", "ccd", "amp", " X ",  "  Y")
    while True:
        line = f.readline()
        if not line:
            break
        parse_line(line)
