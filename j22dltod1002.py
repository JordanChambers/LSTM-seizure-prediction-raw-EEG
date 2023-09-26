import os, h5py, sys, gc, scipy.io, calendar, time, math
#os.environ["CUDA_VISIBLE_DEVICES"]="3"
import numpy as np
import tensorflow as tf
#tf.config.threading.set_inter_op_parallelism_threads(4)
from scipy.stats import norm, sem

class jlrn:
    pat = sys.argv[1]
    epoch = sys.argv[2]
    jeleclist = range(16)
    jsize = 34
    #jntms = [[15,75], [1440,1000000]]
    #jntms = [[0,15], [15,75], [75,1400], [1440,100000]]
    jntms = [[40,80], [80,100000]]
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

def auc(sz, inter, plot=False):
    ''' Calculates the Area under the curve of the receiver operating characteristic curve

    :param sz: array, forecasts for seizure samples
    :param inter: array, forecasts for interictal samples
    :param plot: boolean, turn on to plot ROC curve
    :return: auc, area under the curve
    '''

    if sz.size==0 or inter.size==0:
        return 0., 0, 0

    sz[sz<10**(-40)] = 10**(-40)
    inter[inter < 10 ** (-40)] = 10 ** (-40)
    # Initialise graph and fpr, tpr arrays.
    minimum = min(sz.min(), inter.min())  # smallest forecast (to get minimum of x-axis)
    # print('Max', max(sz.max(), inter.max()))
    # print('Min', minimum)
    min_exp = int(np.log10(minimum))-1  # smallest forecast in log scale
    steps_per_decade = 20  # resolution of the AUC calculation. Here decade refers to order of magnitude
    vals = -min_exp*steps_per_decade+1
    fpr = np.empty(vals)
    tpr = np.empty(vals)

    # Slowly increase threshold, determining fpr and tpr at each iteration
    for i, threshold_log in enumerate(np.arange(min_exp, .1/steps_per_decade,1./steps_per_decade)):  # 2nd argument is exclusive limit (makes inclusive limit = 0 ie 10**0=1. second argument is step size
        threshold = 10**threshold_log
        fpr[i] = float(inter[inter>threshold].size) / inter.size
        tpr[i] = float(sz[sz>threshold].size) / sz.size

    # AUC calculated as area under curve with the curve extrapolated between points using trapezoids
    auc = np.trapz(np.flip(tpr,0), np.flip(fpr, 0)) # Flip  reverses the order to make auc positive

    return auc, fpr, tpr


def auc_se(a, m, n):
    ''' Calculates the standard error of an AUC value

    :param a: area under the ROC curve
    :param m: int, number of sz samples
    :param n: int, number of interictal samples
    :return:
    '''

    q1 = a/(2-a)                                                    # intermediate step
    q2 = (2*a**2) / (1+a)                                           # intermediate step
    a_var = (a*(1-a) + (m-1)*(q1-a**2) + (n-1)*(q2-a**2))/(m*n)     # variance of AUC
    a_se = math.sqrt(a_var)                                         # standard error of AUC

    return a_se


def auc_hanleyci(sz, inter, alpha=.05, bonferroni=1, plot=False):
    ''' Calculates AUC with confidence interval using the Hanley Method

    :param sz: array, forecasts for seizure samples
    :param inter: array, forecasts for interictal samples
    :param alpha: p-value significance threshold, default= 0.05
    :param bonferroni: int, number of AUCs calculated. Used to make the bonferroni adjustment for multiple tests
    :param plot: boolean, turn on to plot ROC curve
    :return:
    '''

    # ------- Calculating AUC --------
    a, _, _ = auc(sz, inter, plot)

    # -------- Calculating CI --------
    m = sz.size  # Sz samples
    n = inter.size  # Non sz samples
    alpha_adjusted = alpha / bonferroni  # alpha with Bonfferoni adjustment
    z = norm.ppf(1 - alpha_adjusted / 2)  # z-score of CI edge
    a_se = auc_se(a, m, n)  # standard error of AUC
    ci_low = a - z * a_se  # confidence interval minimum
    ci_hi = a + z * a_se  # confidence interval maximum

    # Set limits of CI to limits of AUC
    if ci_low < 0:
        ci_low = 0
    if ci_hi > 1:
        ci_hi = 1

    ci = [ci_low, ci_hi]  # Confidence interval

    return a, ci_low, ci_hi

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
    jlrn.model.fit(jlrn.xdata, jlrn.ydata, epochs=20, verbose=0)
    #jlrn.model.fit(jlrn.xdata[:,0:9], jlrn.ydata, epochs=5, verbose=0)
    print("Saving", './cmodels/j22dltod1002_%s_%s.h5' % (jlrn.pat, jlrn.epoch))
    jlrn.model.save('./cmodels/j22dltod1002_%s_%s.h5' % (jlrn.pat, jlrn.epoch))

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
    alist = np.zeros((preds.shape[0],))
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
        alist[pos] = 0.5 + (0.5*(preds[pos][0] - preds[pos][1]))
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
    m = tf.keras.metrics.AUC()
    m.update_state(jlrn.ydata[:,0], alist)
    print(jlrn.pat, jlrn.epoch, 'AUC', m.result().numpy())
    sz = alist[jlrn.ydata[:,0]==1]
    inter = alist[jlrn.ydata[:,0]==0]
    print(len(sz), len(inter))
    a, ci1, ci2 = auc_hanleyci(sz, inter)
    print(jlrn.pat, jlrn.epoch, 'auc hanleyci 1002', a, ci1, ci2)
    # tsum = np.sum(jlrn.ydata[:,0])
    # fout = open('./sstables/j22dltod08_%s_%s.h5' % (jlrn.pat, jlrn.epoch), 'w')
    # for thres in range(10000,-1,-1):
    #     jsen = np.sum(jlrn.ydata[alist>=thres/10000,0])/tsum
    #     jsel = np.sum(alist>=thres/10000)/alist.shape[0]
    #     #cval = np.sum(jlrn.ydata[alist>=thres/100,0])
    #     print(jlrn.pat, jlrn.epoch, 'thres', thres, jsel, jsen)
    #     fout.write('%s %s thres %d %g %g\n' % (jlrn.pat, jlrn.epoch, thres, jsel, jsen))
    return

jrun()

