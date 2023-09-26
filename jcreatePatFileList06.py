import h5py, time, os, calendar
import numpy as np

class jnv:
    pat_list = ['23_002', '23_003', '23_004', '23_005', '23_006', '23_007', '24_001', '24_002', '24_004', '24_005', '25_001', '25_002', '25_003', '25_004', '25_005']
    pat_index = {'23_002':0, '23_003':1, '23_004':2, '23_005':3, '23_006':4, '23_007':5, '24_001':6, '24_002':7, '24_004':8, '24_005':9, '25_001':10, '25_002':11, '25_003':12, '25_004':13, '25_005':14}
    convtm = 1000000

def create_patient_times():
    jnv.patient_times = dict()
    # Recording start in seconds since epoch (1/1/1970)
    f = h5py.File('/media/NVdata/Annotations/record_start.mat', 'r')
    for pat in jnv.pat_list:
        jnv.patient_times[pat] = dict()
        jnv.patient_times[pat]['record_start'] = f['record_start'][jnv.pat_index[pat], 0]
    f.close()
    # Recording length in seconds
    f = h5py.File('/media/NVdata/Annotations/record_lengths.mat', 'r')
    for pat in jnv.pat_list:
        jnv.patient_times[pat]['record_lengths'] = f['record_lengths'][0, jnv.pat_index[pat]]
    f.close()
    print('patient_times', jnv.patient_times)
    return

def get_datafiles_patient(pat):
    ## pat 23_007 has seizures of type 4. Not sure who added this, so using old annots file which only contains seizure types 1, 2, 3
    if pat in ['23_007']:
        filename = '/media/NVdata/Annotations/%s_Annots_old.mat' % (pat)
    else:
        filename = '/media/NVdata/Annotations/%s_Annots.mat' % (pat)
    fan = h5py.File(filename, 'r')
    sztimes = np.copy(fan['SzTimes'])
    sztype = np.copy(fan['SzType'])
    fan.close()
    #exclude type 3 seizures
    sztimes = sztimes[sztype != 3]
    ctm = jnv.patient_times[pat]['record_start'] - 60 # Substract 60 so I can easily take 1 minute steps
    etm = ctm + jnv.patient_times[pat]['record_lengths'] + 120 # Add 120 because substract 60 on previous line and add another 60 for rounding down to the nearest minute
    f100d = jnv.patient_times[pat]['record_start'] + (100*24*60*60) # used to exlcude the first 100 days
    # Check basic info if required
    print(pat, 'start:', time.gmtime(jnv.patient_times[pat]['record_start']))
    print(pat, 'f100d:', time.gmtime(f100d))
    print(pat, 'end:', time.gmtime(jnv.patient_times[pat]['record_start'] + jnv.patient_times[pat]['record_lengths']))
    print(pat, 'number of seizures:', len(sztimes))
    for spos in range(sztimes.shape[0]):
        print(pat, 'sztime', spos, time.gmtime(jnv.patient_times[pat]['record_start'] + (sztimes[spos]/jnv.convtm)))
    while ctm < etm:
        ctm += 60
        if ctm < f100d:
            #print("Skipping first 100 days")
            continue
        # calculate the time to next seizure and time since last seizure
        psz = -1
        pszv = 1e10
        nsz = -1
        nszv = 1e10
        for spos in range(sztimes.shape[0]):
            stm = jnv.patient_times[pat]['record_start'] + (sztimes[spos]/jnv.convtm)
            if stm <= ctm:
                if ctm - stm < pszv:
                    pszv = ctm - stm
                    psz = spos
            if stm > ctm:
                if stm - ctm < nszv:
                    nszv = stm - ctm
                    nsz = spos
        # exclude if last seizure was within 4 hours
        if psz >= 0:
            if pszv < (4*60*60):
                continue
        # exlcude if next seizure is too far away, e.g. 100 minutes
        #if nsz >= 0:
        #    if nszv > (100*60):
        #        continue
        # create filename, try to open file, check for data dropouts (exclude if missing more than 400 data points on any electrode)
        ctmf = time.gmtime(ctm)
        filename = '/media/NVdata/Patient_%s/' % (pat) + time.strftime("Data_%Y_%m_%d/Hour_%H/UTC_%H_%M_00.mat", ctmf)
        jtake = 1
        if filename not in goodfilelist:
            try:
                cfan = h5py.File(filename, 'r')
                for jelec in range(16):
                    cvn = np.count_nonzero(np.isnan(cfan['Data'][jelec,:]))
                    if cvn > 400:
                        jtake = 0
                        continue
                cfan.close()
            except:
                jtake = 0
        if jtake == 1:
            if psz == -1:
                pszv = -1
            if nsz == -1:
                nszv = -1
            #print(filename, nsz, nszv, psz, pszv)
            print(filename, int(nszv/60), int(pszv/60))
        #else:
        #    print(filename, 'fail')
    return

def jrun():
    create_patient_times()
    for pat in jnv.pat_list:
        get_datafiles_patient(pat)
    return

jrun()
