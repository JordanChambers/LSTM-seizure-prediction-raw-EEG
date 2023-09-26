import os, h5py, sys, gc, scipy.io
#os.environ["CUDA_VISIBLE_DEVICES"]="3"
import numpy as np
import tensorflow as tf
#tf.config.threading.set_inter_op_parallelism_threads(4)

class jlrn:
    pat = sys.argv[1]
    jeleclist = range(16)
    bflist = list()
    cflist = list()
    jnumb = 8
    jsz = 4000
    #jntms = [[15,75], [1440,1000000]]
    jntms = [[0,4], [4,100000]]
    #jntms = [[15,75], [75,1400], [1440,1000000]]

def jget_file_list():
    jlrn.fdict = dict()
    for pos in range(len(jlrn.jntms)):
        jlrn.fdict[pos] = dict()
        jlrn.fdict[pos]['all'] = list()
        jlrn.fdict[pos]['tm'] = dict()

    f = open('./filelists/jcpfl05_080_%s_checked01.txt' % (sys.argv[1]), 'r')
    for line in f.readlines():
        info = line.split()
        tm = int(info[1])
        jpos = -1
        for pos in range(len(jlrn.jntms)):
            if tm >= jlrn.jntms[pos][0]:
                if tm < jlrn.jntms[pos][1]:
                    jpos = pos
                    break
        if jpos >= 0:
            jlrn.fdict[pos]['all'].append(info[0])
            jlrn.fdict[pos]['tm'][info[0]] = tm
    tot = 0
    for pos in jlrn.fdict.keys():
        tot += len(jlrn.fdict[pos]['all'])
        print(pos, len(jlrn.fdict[pos]['all']))
    print('tot', tot)
    return

def jget_cflist():
    cdict = {}
    for i in range(len(jlrn.jntms)):
        cdict[i] = len(jlrn.fdict[i]['all'])
    print(cdict)
    hval = 0
    for i in cdict.keys():
        if cdict[i] > hval:
            hval = cdict[i]
    cld = dict()
    cfd = dict()
    cvlist = [0,4000,8000,12000,16000,19000,2000,6000,10000,14000,18000,1000,5000,9000,13000,17000,3000,7000,11000,15000,18000]
    for i in cdict.keys():
        cld[i] = list()
        cfd[i] = hval//cdict[i] + (hval % cdict[i] > 0)
        print(i, hval, cdict[i], cfd[i])
    for pos in range(len(jlrn.jntms)):
        for fn in jlrn.fdict[pos]['all']:
            for i in range(cfd[pos]):
                if i < len(cvlist):
                    cld[pos].append([fn, cvlist[i]])
                else:
                    cld[pos].append([fn, np.random.randint(0,19000)])
    for i in cdict.keys():
        np.random.shuffle(cld[i])
        print(i, len(cld[i]))
    jlrn.cflist = list()
    for i in range(hval):
        for j in cld.keys():
            jlrn.cflist.append(cld[j][i])
    print(len(jlrn.cflist))
    return

def jget_normdict_titanx(fn):
    f = open('./normlist/jnormlistALL_%s.txt' % (sys.argv[1]), 'r')
    l30 = []
    # 2300220101226082611882.mat
    # ../data23002/230022010092713181184.mat
    fnd = fn[29:44]
    for line in f.readlines():
        info = line.split()
        if len(info) < 1:
            continue
        if info[0] == fnd:
            break
        l30.append(info)
        if len(l30) > 30:
            del l30[0]
    if len(l30) != 30:
        print("Warning: incorrect length of l30: %d" % (len(l30)))
    tdict = dict()
    for jelec in range(16):
        tdict[jelec] = [0,0]
    for jpos in range(len(l30)):
        for jelec in range(16):
            tdict[jelec][0] += float(l30[jpos][jelec*2 + 1])
            tdict[jelec][1] += int(l30[jpos][jelec*2 + 2])
    ndict = dict()
    for jelec in range(16):
        if tdict[jelec][1] == 0:
            ndict[jelec] = 1
        else:
            ndict[jelec] = tdict[jelec][0]/tdict[jelec][1]
    #print(fn, fnd)
    #print(l30[0][0], l30[-1][0])
    #print(ndict)
    return ndict

def jget_data():
    #print("Getting data...")
    if len(jlrn.cflist) == 0:
        jget_cflist()
    jlrn.jnumb = 32#len(jlrn.cflist)
    if len(jlrn.cflist) < jlrn.jnumb:
        jlrn.jnumb = len(jlrn.cflist)
    jx1data = np.zeros((jlrn.jnumb,jlrn.jsz,16))
    jydata = np.zeros((jlrn.jnumb,len(jlrn.jntms)))
    ycnt = 0
    for fnd in jlrn.cflist[:jlrn.jnumb]:
        fn = fnd[0]
        #print(fn)
        # fn '../data23002/2300220101120011410520.mat'
        #tod = ((int(fn[26:28])*60) + int(fn[28:30]))/1440
        #ctod = np.min([np.abs(tod - 0.125), np.abs(tod - 1.125)])
        # fn /media/NVdata/Patient_23_002/Data_2011_07_08/Hour_15/UTC_15_38_00.mat
        tod = ((int(fn[-12:-10])*60) + int(fn[-9:-7]))/1440
        ctod = np.min([np.abs(tod - 0.125), np.abs(tod - 1.125)])
        nfn = fn[:-16]
        tmin = fn[-9]
        jmin = int(fn[-8])
        nfn = nfn.replace('/media/', '/data/gpfs/projects/punim0264/test/')
        ffn = '%sCUTC_%s_%s0_00.mat' % (nfn, fn[-12:-10], tmin)

        jnormd = jget_normdict_titanx(fn)
        
        cjoff = fnd[1]
        fan = scipy.io.loadmat(ffn)
        dmin = int(fan['Data'].shape[0]/10)
        for jelec in jlrn.jeleclist:
            jdata = fan['Data'][jmin*dmin:(jmin+1)*dmin,jelec]
            if np.count_nonzero(np.isnan(jdata)) > 0:
                jdata[np.isnan(jdata)] = jdata[~np.isnan(jdata)].mean()
            jdata = jdata - jdata.mean()
            jdata = jdata/jnormd[jelec]
            jx1data[ycnt,:,jelec] = jdata[cjoff:cjoff+jlrn.jsz]
        for pos in jlrn.fdict.keys():
            if fn in jlrn.fdict[pos]['all']:
                jydata[ycnt,pos] = 1
                break
        ycnt += 1
        #fan.close()
    jlrn.cflist = jlrn.cflist[jlrn.jnumb:]
    #print("...got data (%d)" % (len(jlrn.cflist)))
    return jx1data, jydata

def jrun():
    jget_file_list()
    #jget_cflist()
    #killnow()

    jinput = tf.keras.layers.Input((jlrn.jsz, 16))
    jlstm1 = tf.keras.layers.LSTM(256, input_shape=jinput.shape, activation='sigmoid', recurrent_activation='sigmoid', recurrent_dropout=0.25, return_sequences=True)(jinput)
    jave1 = tf.keras.layers.MaxPooling1D(pool_size=5)(jlstm1)
    jlstm2 = tf.keras.layers.LSTM(128, input_shape=jlstm1.shape, activation='sigmoid', recurrent_activation='sigmoid', recurrent_dropout=0.25, return_sequences=True)(jave1)
    jave2 = tf.keras.layers.MaxPooling1D(pool_size=2)(jlstm2)
    jlstm3 = tf.keras.layers.LSTM(32, input_shape=jlstm2.shape, activation='sigmoid', recurrent_activation='sigmoid', recurrent_dropout=0.25, return_sequences=True)(jave2)
    jave3 = tf.keras.layers.MaxPooling1D(pool_size=5)(jlstm3)
    jlstm4 = tf.keras.layers.LSTM(32, input_shape=jlstm3.shape, activation='sigmoid', recurrent_activation='sigmoid', recurrent_dropout=0.25, return_sequences=False)(jave3)
    joutput = tf.keras.layers.Dense(len(jlrn.jntms), activation='sigmoid')(jlstm4)
    jlrn.model = tf.keras.models.Model(inputs=jinput, outputs=joutput)
    jopt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    jlrn.model.compile(loss='mse', optimizer=jopt, metrics=['accuracy'])
    #print(jlrn.model.summary())
    #killnow()
    jsep = 0
    if sys.argv[2] != '0':
        jlrn.model = tf.keras.models.load_model('./models/j22lstm04_%s_%s.h5' % (sys.argv[1], sys.argv[2]))
        jsep = int(sys.argv[2])+1
    
    for jepoch in range(jsep,101):
        jget_cflist()
        tot = [0,0,0]
        while len(jlrn.cflist) > 0:
            jx1data, jydata = jget_data()
            hist = jlrn.model.fit(jx1data, jydata, verbose=False)
            #print(hist.history)
            tot[0] += 1
            tot[1] += hist.history['loss'][0]
            tot[2] += hist.history['accuracy'][0]
            #print(tot, tot[1]/tot[0], tot[2]/tot[0])
            if jepoch < 1:
                print(jepoch, len(jlrn.cflist), tot[1]/tot[0], tot[2]/tot[0])
                sys.stdout.flush()
            tf.keras.backend.clear_session()
            gc.collect()
        print(jepoch, len(jlrn.cflist), tot[1]/tot[0], tot[2]/tot[0])
        sys.stdout.flush()
        print("Saving", './models/j22lstm04_%s_%d.h5' % (sys.argv[1], jepoch))
        jlrn.model.save('./models/j22lstm04_%s_%d.h5' % (sys.argv[1], jepoch))
    return

jrun()

