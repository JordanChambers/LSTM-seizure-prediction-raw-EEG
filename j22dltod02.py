import os, h5py, sys, gc, scipy.io, calendar, time
#os.environ["CUDA_VISIBLE_DEVICES"]="3"
import numpy as np
import tensorflow as tf
#tf.config.threading.set_inter_op_parallelism_threads(4)

class jlrn:
    pat = sys.argv[1]
    epoch = sys.argv[2]
    jeleclist = range(16)
    jsize = 34
    #jntms = [[15,75], [1440,1000000]]
    #jntms = [[0,15], [15,75], [75,1400], [1440,100000]]
    jntms = [[0,5], [5,65], [65,480], [480,1440], [1440,100000]]
    #jntms = [[15,75], [75,1400], [1440,1000000]]
    #jntms = [[0,15], [15,75], [75,1400], [1440,4600]]

def jget_data_train():
    fin = open('./inputs/j22lstm02a5_%s_%s_080.txt' % (jlrn.pat, jlrn.epoch), 'r')
    cnd = dict()
    cld = dict()
    for i in range(len(jlrn.jntms)):
        cnd[i] = 0
        cld[i] = list()
    for line in fin.readlines():
        info = line.split()
        if len(info) != 28:
            continue
        tos = float(info[1])
        jans = -1
        for i in range(len(jlrn.jntms)):
            if tos >= jlrn.jntms[i][0]:
                if tos <= jlrn.jntms[i][1]:
                    jans = i
                    break
        if jans >= 0:
            cnd[jans] += 1
            yr = int(line[34:38])
            mn = int(line[39:41])
            dy = int(line[42:44])
            hr = int(line[50:52])
            mi = int(line[60:62])
            mtm = calendar.timegm((yr, mn, dy, hr, mi, 0, 0, 0, 0))
            jlist = list()
            wday = time.localtime(time.mktime((yr, mn, dy, 0, 0, 0, 0, 0, 0))).tm_wday
            jlist.append(5*((hr*60)+mi)/1440)
            jmin = min([abs(jlist[-1]-0), abs(5-jlist[-1])])
            jlist.append(2*jmin)
            jlist.append(5*((wday*1440)+(hr*60)+mi)/(7*1440))
            jmin = min([abs(jlist[-1]-0), abs(5-jlist[-1])])
            jlist.append(2*jmin)
            jlist.append(5*(((dy-1)*1440)+(hr*60)+mi)/(31*1440))
            jmin = min([abs(jlist[-1]-0), abs(5-jlist[-1])])
            jlist.append(2*jmin)
            jlist.append(5*(((mn-1)*31*1440)+((dy-1)*1440)+(hr*60)+mi)/(12*31*1440))
            jmin = min([abs(jlist[-1]-0), abs(5-jlist[-1])])
            jlist.append(2*jmin)
            jlist.append(np.log(int(info[2])))
            for i in range(25):
                jlist.append(5*float(info[3+i]))
            cld[jans].append(jlist)
    print('found:', cnd)
    jmax = np.max(list(cnd.values()))
    csize = len(jlrn.jntms)
    jlrn.xdata = np.zeros((csize*jmax, jlrn.jsize))
    jlrn.ydata = np.zeros((csize*jmax, len(jlrn.jntms)))
    cpos = 0
    for csc in range(csize):
        jcnt = 0
        jpos = 0
        jlen = len(cld[csc])
        while jcnt < jmax:
            if jcnt <= jlen:
                #print(jcnt, jpos, jlen)
                jlrn.xdata[cpos,:] = cld[csc][jpos]
            else:
                a = np.random.rand(jlrn.jsize)
                b = 0.95 + (a/10)
                jlrn.xdata[cpos,:] = b*cld[csc][jpos]
            jlrn.ydata[cpos,csc] = 1
            cpos += 1
            jcnt += 1
            jpos += 1
            if jpos >= jlen:
                jpos = 0
    print("Shuffling data...")
    p = np.random.permutation(jlrn.xdata.shape[0])
    jlrn.xdata = jlrn.xdata[p]
    jlrn.ydata = jlrn.ydata[p]
    print("...shuffle done")
    print("Data size:", jlrn.xdata.shape[0])
    return

def jget_data_test():
    fin = open('./inputs/j22lstm02a5_%s_%s_80100.txt' % (jlrn.pat, jlrn.epoch), 'r')
    cnd = dict()
    cld = dict()
    for i in range(len(jlrn.jntms)):
        cnd[i] = 0
    lall = list()
    lans = list()
    for line in fin.readlines():
        info = line.split()
        if len(info) != 28:
            continue
        tos = float(info[1])
        jans = -1
        for i in range(len(jlrn.jntms)):
            if tos >= jlrn.jntms[i][0]:
                if tos <= jlrn.jntms[i][1]:
                    jans = i
                    break
        if jans >= 0:
            cnd[jans] += 1
            yr = int(line[34:38])
            mn = int(line[39:41])
            dy = int(line[42:44])
            hr = int(line[50:52])
            mi = int(line[60:62])
            mtm = calendar.timegm((yr, mn, dy, hr, mi, 0, 0, 0, 0))
            jlist = list()
            wday = time.localtime(time.mktime((yr, mn, dy, 0, 0, 0, 0, 0, 0))).tm_wday
            jlist.append(5*((hr*60)+mi)/1440)
            jmin = min([abs(jlist[-1]-0), abs(5-jlist[-1])])
            jlist.append(2*jmin)
            jlist.append(5*((wday*1440)+(hr*60)+mi)/(7*1440))
            jmin = min([abs(jlist[-1]-0), abs(5-jlist[-1])])
            jlist.append(2*jmin)
            jlist.append(5*(((dy-1)*1440)+(hr*60)+mi)/(31*1440))
            jmin = min([abs(jlist[-1]-0), abs(5-jlist[-1])])
            jlist.append(2*jmin)
            jlist.append(5*(((mn-1)*31*1440)+((dy-1)*1440)+(hr*60)+mi)/(12*31*1440))
            jmin = min([abs(jlist[-1]-0), abs(5-jlist[-1])])
            jlist.append(2*jmin)
            jlist.append(np.log(int(info[2])))
            for i in range(25):
                jlist.append(5*float(info[3+i]))
            lall.append(jlist)
            lans.append(jans)
    print('found:', cnd)
    jmax = np.max(list(cnd.values()))
    csize = len(jlrn.jntms)
    jlrn.xdata = np.zeros((len(lall), jlrn.jsize))
    jlrn.ydata = np.zeros((len(lall), len(jlrn.jntms)))
    for jpos in range(len(lall)):
        jlrn.xdata[jpos,:] = lall[jpos]
        jlrn.ydata[jpos,lans[jpos]] = 1
    print("Data size:", jlrn.xdata.shape[0])
    return

def jrun():
    jget_data_train()

    #Create the model
    jinput = tf.keras.layers.Input((jlrn.jsize))
    jdense1 = tf.keras.layers.Dense(10*len(jlrn.jntms), activation='sigmoid')(jinput)
    jdrop1 = tf.keras.layers.Dropout(0.25)(jdense1)
    joutput = tf.keras.layers.Dense(len(jlrn.jntms), activation='sigmoid')(jdrop1)
    jlrn.model = tf.keras.models.Model(inputs=jinput, outputs=joutput)
    jopt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    jlrn.model.compile(loss='mse', optimizer=jopt, metrics=['accuracy'])
    #print(jlrn.model.summary())

    #Fit the model
    jlrn.model.fit(jlrn.xdata, jlrn.ydata, epochs=20, verbose=1)
    #jlrn.model.fit(jlrn.xdata[:,0:9], jlrn.ydata, epochs=5, verbose=0)
    print("Saving", './cmodels/j22dltod02_%s_%s.h5' % (jlrn.pat, jlrn.epoch))
    jlrn.model.save('./cmodels/j22dltod02_%s_%s.h5' % (jlrn.pat, jlrn.epoch))

    jget_data_test()
    pdict = dict()
    adict = dict()
    for i in range(len(jlrn.jntms)):
        pdict[i] = dict()
        adict[i] = dict()
        for j in range(len(jlrn.jntms)):
            pdict[i][j] = 0
            adict[i][j] = 0
    preds = jlrn.model.predict(jlrn.xdata)
    for pos in range(preds.shape[0]):
        p = -1
        cmax = -1
        for jpos in range(len(jlrn.jntms)):
            if preds[pos,jpos] > cmax:
                cmax = preds[pos,jpos]
                p = jpos
        a = -1
        for jpos in range(len(jlrn.jntms)):
            if jlrn.ydata[pos,jpos] > 0.9:
                a = jpos
                break
        pdict[p][a] += 1
        adict[a][p] += 1
    print('pdict', pdict)
    print('adict', adict)
    jlist = list()
    for pos in adict.keys():
        tot = 0
        for jpos in adict[pos].keys():
            tot += adict[pos][jpos]
        jstr = '%d: ' % (pos)
        for jpos in adict[pos].keys():
            jstr += '%g (%d) ' % (adict[pos][jpos]/tot, jpos)
        jlist.append(adict[pos][pos]/tot)
        print(jstr)
    for pos in pdict.keys():
        tih = 0
        for jpos in pdict[pos].keys():
            tih += pdict[pos][jpos]
        print('selectivity %d: %g' % (pos, tih/preds.shape[0]))
    print('mean', np.mean(jlist))
    return

jrun()

