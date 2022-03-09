import copy
import datetime
import glob
import matplotlib
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
from os.path import exists
import pandas as pd
import shutil
import time
from mpl_toolkits.axisartist.axislines import Subplot
from pathlib import Path
from shutil import copyfile
import sys
from subprocess import DEVNULL
from subprocess import call
from tqdm import tqdm
import sys

NUM_PROCESSORS = 50

if __name__ == "__main__":
    if len(sys.argv) < 2:
        user_arg = input('Name of file with user defined inputs (including the extension)?\n \n')
    else:
        user_arg = sys.argv[1]
    copyfile(user_arg, 'fit_inputs.py')


from fit_inputs import *
K_ORDER = list(PROTON_ORBITS)
K_ORDER.extend(NEUTRON_ORBITS)


BG_HAM_FANS = [AI_SPE, AI_STRONG, AI_COUL, AI_ISOV, AI_ISOT]
INITIAL_HAM_FANS = [INITIAL_SPE, INITIAL_STRONG, INITIAL_COUL, INITIAL_ISOV, INITIAL_ISOT]

DATA_TYPE_INCLUSION = [fit_to_energies, fit_to_MED, fit_to_TED, fit_to_delMED]


if __name__ == "__main__":
    if AI_COUL == '':
        if INITIAL_COUL == '':
            INCLUDE_COULOMB = False
        else:
            input('mismatch in coulomb inclusion, check user defined inputs')
    else:
        if INITIAL_COUL == '':
            input('mismatch in coulomb inclusion, check user defined inputs')
        INCLUDE_COULOMB = True

    if AI_ISOV == '':
        if INITIAL_ISOV == '':
            INCLUDE_ISOVECTOR = False
        else:
            input('mismatch in isovector inclusion, check user defined inputs')
    else:
        if INITIAL_ISOV == '':
            input('mismatch in isovector inclusion, check user defined inputs')
        INCLUDE_ISOVECTOR = True

    if AI_ISOT == '':
        if INITIAL_ISOT == '':
            INCLUDE_ISOTENSOR = False
        else:
            input('mismatch in isovector inclusion, check user defined inputs')
    else:
        if INITIAL_ISOT == '':
            input('mismatch in isovector inclusion, check user defined inputs')
        INCLUDE_ISOTENSOR = True







pd.options.mode.chained_assignment = None  # default='warn'

# dictionary of orbit labels used in NushellX
# k ; n l 2j
orbitList = {
    "1": "1 0 1",
    "2": "1 1 3",
    "3": "1 1 1",
    "4": "1 2 5",
    "5": "1 2 3",
    "6": "2 0 1",
    "7": "1 3 7",
    "8": "1 3 5",
    "9": "2 1 3",
    "10": "2 1 1",
    "11": "1 4 9",
    "12": "1 4 7",
    "13": "2 2 5",
    "14": "2 2 3",
    "15": "3 0 1",
    "16": "1 5 11",
    "17": "1 5 9",
    "18": "2 3 7",
    "19": "2 3 5",
    "20": "3 1 3",
    "21": "3 1 1",
    "22": "1 6 13",
    "23": "1 6 11",
    "24": "2 4 9",
    "25": "2 4 7",
    "26": "3 2 5",
    "27": "3 2 3",
    "28": "4 0 1",
    "29": "1 7 15",
}

'''
The first few definitions are simply and unlikely to need to be edited. 
'''

def make_model_space():
    s = '{} {}\n'.format(A_CORE, Z_CORE)
    for o in PROTON_ORBITS:
        x = orbitList[str(o)]
        x = x.split()
        s += '-1 {} {} {}\n'.format(int(x[0])-1, x[1], x[2])
    for o in NEUTRON_ORBITS:
        x = orbitList[str(o)]
        x = x.split()
        s += '1 {} {} {}\n'.format(int(x[0])-1, x[1], x[2])
    s += '\n\n'
    with open('groups/{}/{}.mod'.format(MODEL_SPACE_NAME,MODEL_SPACE_NAME), 'w') as f:
        f.write(s)
    return


def orbit_spin(k):
    x = orbitList[str(k)]
    x = x.split()
    x = int(x[2])
    return x


def swap_element_label(z, fixlength=False):
    namelist = ['', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
                'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br',
                'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd','Ag', 'Cd', 'In', 'Sn']
    if z in namelist:
        x = namelist.index(z)
    else:
        x = namelist[int(z)]
        if fixlength:
            x = str.lower(x)
            if len(namelist[int(z)]) == 1:
                x = '{}_'.format(str.lower(x))

    return x


def cycle_lbl(x):
    #print(x)
    l = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
         'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D',
         'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
         'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    if x > 9:
        return l[x - 10]
    else:
        return str(x)


def os_path(path):
    return str(Path(path))


def make_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def rms_from_lists(x, y):
    if len(x) != len(y):
        input('rms deviation error: lists not of same size')
    else:
        rms = 0.
        for i in range(len(x)):
            rms += (float(x[i]) - float(y[i])) ** 2.
        rms = (rms / len(x)) ** .5
        return rms


def write_csv_file(fn, list):
    with open(fn, 'w') as f:
        for i in range(0, len(list)):
            f.write(str(list[i]))
            if i + 1 != len(list):
                f.write(',')


'''
The following class (and some functions) control the Hamiltonian and writing/manipulating it.
'''


def group_number(row):
    with open('fit_labels.dat', 'r') as f:
        for line in f.readlines():
            if line.split()[1] == row['group']:
                return int(line.split()[0])
    return 0


def group_tbme(row):
    o = [row['k1'], row['k2'], row['k3'], row['k4']]
    # o = [abs(x) for x in o]   #  JORDAN  Need new way of connecting orbits between nn and pn
    o = [K_ORDER[x - 1] for x in o]
    o = [max(o[0], o[1]), min(o[0], o[1]), max(o[2], o[3]), min(o[2], o[3])]
    x = 'unorganized'
    if o[0] < o[2]:
        o = [o[2], o[3], o[0], o[1]]
    if o[0] == o[2] and o[1] < o[3]:
        o = [o[2], o[3], o[0], o[1]]

    if o[0] == o[2] and o[1] == o[3]:
        diagonal = True
    else:
        diagonal = False
    J, T = row['j'], row['t']

    # added to remove specific orbits, with absolute k-order references

    low_orbits = True
    for exclude in EXCLUDE_ORBIT_LIST:
        if exclude in o:
            low_orbits = False

    # Here is where you can change the groupings and what qualifies as its own group
    # if a group is labeled as 'other' it is not fit. Any other name will create a new
    # group and fit it.



    if STBME_CHOICE == 1:
        x = 'strong'
    if STBME_CHOICE == 2:
        x = '{}'.format(T)
    if STBME_CHOICE == 3:
        x = '{}'.format(row['type'])
    if STBME_CHOICE == 4:
        x = '{:02}{:02}-{:02}{:02}-{}{}'.format(o[0], o[1], o[2], o[3], J, T)
    if STBME_CHOICE == 5:
        if diagonal:
            x = '{:02}{:02}-{:02}{:02}-{}{}'.format(o[0], o[1], o[2], o[3], J, T)
        else:
            x = 'other'
    if STBME_CHOICE == 6:
        x = '{:02}{:02}-{:02}{:02}-{}{}'.format(row['k1'], row['k2'], row['k3'], row['k4'], J, T)
    if STBME_CHOICE == 7:
        if int(T) > 0 and diagonal:
            x = '{:02}{:02}-{:02}{:02}-{}{}'.format(o[0], o[1], o[2], o[3], J, T)
        else:
            x = 'other'
    if STBME_CHOICE == 8:
        if diagonal and low_orbits:
            x = '{:02}{:02}-{:02}{:02}-{}{}'.format(o[0], o[1], o[2], o[3], J, T)
        else:
            x = 'other'
    if STBME_CHOICE == 9:
        if not Path('custom.dat').is_file():
            input('Can not find custom.dat file with group names')

        tbpf = CUSTOM_GROUPS.copy()

        tbpf = tbpf[(tbpf['k1'] == row['k1']) & (tbpf['k2'] == row['k2']) & (tbpf['k3'] == row['k3']) & (
                    tbpf['k4'] == row['k4']) & (tbpf['j'] == J) & (tbpf['t'] == T)]

        if len(tbpf) == 0:
            input('custom grouping function broken, TBME not found')
        else:
            if tbpf['cgn'].tolist()[0] != '0':
                x = tbpf['cgn'].tolist()[0]
            else:
                x = 'other'

    if x == 'unorganized':
        input('grouping function broken, be sure to select a defined STBME_CHOICE')

    return x



def type_tbme(row):
    x = 'error'
    orbits = [row['k1'], row['k2'], row['k3'], row['k4']]

    cutoff = len(PROTON_ORBITS) + .5
    if max(orbits) > cutoff and min(orbits) > cutoff:
        x = 'nn'
    if max(orbits) > cutoff and min(orbits) < cutoff:
        x = 'pn'
    if max(orbits) < cutoff and min(orbits) < cutoff:
        x = 'pp'
    return x


def read_tbme(name):
    tbpf = pd.DataFrame(np.genfromtxt(name, skip_header=0, dtype='str', usecols=(0, 1, 2, 3, 4, 5, 6)),
                        columns=['k1', 'k2', 'k3', 'k4', 'j', 't', 'v'])
    tbpf = tbpf.astype({'k1': int, 'k2': int, 'k3': int, 'k4': int, 'j': int, 't': int, 'v': float})
    return tbpf


def read_spe(name):
    spf = pd.DataFrame(np.genfromtxt(name, skip_header=0, dtype=str), columns=['type', 'k', 'v'])
    spf = spf.astype({'type': str, 'k': int, 'v': float})
    return spf


def update_hamiltonian(old_ham, bg_ham, mults):
    iter_rms, count = 0., 0
    bg_rms, bg_count = 0., 0
    new_ham = copy.deepcopy(old_ham)
    g_num = 0

    for index, row in new_ham.spe.iterrows():
        new_spe = row['v'] * mults[g_num]
        iter_rms += (row['v'] - new_spe) ** 2.
        bg_rms += (bg_ham.spe.iloc[index]['v'] - new_spe) ** 2.
        #print(bg_ham.spe.iloc[index]['v'] - new_spe)
        new_ham.spe.at[index, 'v'] = new_spe
        g_num += 1
        count += 1

    for index, row in new_ham.strong.iterrows():
        group_name = row['group_num']
        if group_name > 0:

            new_tbme = row['v'] * mults[group_name - 1]
            iter_rms += (row['v'] - new_tbme) ** 2.
            bg_rms += (bg_ham.strong.iloc[index]['v'] - new_tbme) ** 2.
            #print((bg_ham.strong.iloc[index]['v'] - new_tbme) ** 2.)
            #input()
            count += 1
            new_ham.strong.at[index, 'v'] = new_tbme

    if INCLUDE_ISOVECTOR:
        new_ham.a_iv = old_ham.a_iv * mults[-3]

    if INCLUDE_ISOTENSOR:
        new_ham.a_it = old_ham.a_it * mults[-2]

    if INCLUDE_COULOMB:
        for index, row in new_ham.coul.iterrows():
            new_tbme = row['v'] * mults[-1]
            iter_rms += (row['v'] - new_tbme) ** 2.
            bg_rms += (bg_ham.coul.iloc[index]['v'] - new_tbme) ** 2.
            count += 1
            new_ham.coul.at[index, 'v'] = new_tbme

    new_ham.make_files(A_MIN, A_MAX, 'groups/')  # readd this now akm now

    iter_rms = (iter_rms / count) ** 0.5
    bg_rms = (bg_rms / count) ** 0.5



    return new_ham, bg_rms, iter_rms


def order_tbme(df, include_v = True):
    j_k_list = [orbit_spin(x) for x in K_ORDER]
    for index, row in df.iterrows():
        k1, k2, k3, k4, J, T = int(row['k1']), int(row['k2']), int(row['k3']), int(row['k4']), int(row['j']), int(row['t'])
        j1, j2, j3, j4 = j_k_list[k1-1], j_k_list[k2-1], j_k_list[k3-1], j_k_list[k4-1]
        sign = 1
        if k1 > k2:
            k1, k2 = k2, k1
            sign *= -1 * (-1) ** ((j1 + j2) / 2 - J + 1 - T)

        if k3 > k4:
            k3, k4 = k4, k3
            sign *= -1 * (-1) ** ((j3 + j4) / 2 - J + 1 - T)
        if k1 > k3:
            k1, k2, k3, k4 = k3, k4, k1, k2
        if k1 == k3:
            if k2 > k4:
                k1, k2, k3, k4 = k3, k4, k1, k2

        df.at[index, 'k1'] = k1
        df.at[index, 'k2'] = k2
        df.at[index, 'k3'] = k3
        df.at[index, 'k4'] = k4
        if include_v:
            df.at[index, 'v'] = sign * row['v']

    df = df.drop_duplicates()

    return df

if Path('custom.dat').is_file():
    CUSTOM_GROUPS = pd.DataFrame(np.genfromtxt('custom.dat', skip_header=0, dtype='str'),
                        columns=['k1', 'k2', 'k3', 'k4', 'j', 't', 'cgn'])
    CUSTOM_GROUPS = CUSTOM_GROUPS.astype({'k1': int, 'k2': int, 'k3': int, 'k4': int, 'j': int, 't': int, 'cgn': str})
    # print(tbpf)
    CUSTOM_GROUPS = order_tbme(CUSTOM_GROUPS, False)

class Hamiltonian:
    def __init__(self, y, group_folder='groups'):
        if isinstance(y, str):
            z = []
            with open(y, 'r') as f:
                for line in f:
                    z.append(line.strip())
            y = z


        spe_file, strong_file = y[0], y[1]
        include_tick = 1

        self.strong_mass = A_CORE+2
        self.strong_expo = 0.3
        self.coul_mass = A_CORE+2
        self.coul_expo = (1. / 6.)
        self.group_folder = group_folder

        self.spe = read_spe(spe_file)
        self.strong = read_tbme(strong_file)
        if INCLUDE_COULOMB:
            coul_file = y[1+include_tick]
            include_tick+=1
            self.coul = read_tbme(coul_file)
        else:
            self.coul = pd.DataFrame()


        self.strong = order_tbme(self.strong)

        self.coul = order_tbme(self.coul)

        self.strong['type'] = self.strong.apply(lambda x: type_tbme(x), axis=1)
        self.strong['group'] = self.strong.apply(lambda x: group_tbme(x), axis=1)

        if INCLUDE_ISOVECTOR:
            iv = y[1 + include_tick]
            include_tick += 1
            self.a_iv = float(iv)
            self.ivtb = self.make_isovector(self.a_iv)
        else:
            self.a_iv = 0.
            self.ivtb = self.make_isovector(self.a_iv)
        if INCLUDE_ISOTENSOR:
            it = y[1 + include_tick]
            include_tick += 1
            self.a_it = float(it)
            self.ittb = self.make_isotensor(self.a_it)
        else:
            self.a_it = 0.
            self.ittb = self.make_isotensor(self.a_it)

        self.grouped = self.strong.groupby(['group'])
        self.group_count = self.make_files(A_MIN, A_MAX, group_folder)
        self.normalize_scale(A_CORE+2) # rescale back to A_CORE + 2 to keep consistency
        self.strong['group_num'] = self.strong.apply(lambda x: group_number(x), axis=1)

    def strength_list(self):
        spe = list(self.spe['v'].tolist())


        folder = '{}/{}/{}'.format(self.group_folder, MODEL_SPACE_NAME, A_CORE+2)
        for i in range(len(spe)+1, self.group_count+1):
            with open('{}/g{}.int'.format(folder, i), 'r') as f:
                lines = f.readlines()
                x = lines[2].split()[6]
                spe.append(float(x))

        if INCLUDE_ISOVECTOR:
            spe[-3] = self.a_iv

        if INCLUDE_ISOTENSOR:
            spe[-2] = self.a_it

        return spe


    def comparison_to_bg(self, bg_ham, name):
        #a = bg_ham.strong.copy()
        #b = self.strong.copy()
        if INCLUDE_COULOMB or INCLUDE_ISOVECTOR or INCLUDE_ISOTENSOR:
            a = bg_ham.full_tbme()
            b = self.full_tbme()
        else:
            a = bg_ham.strong.copy()
            b = self.strong.copy()

        #print(a)
        #print(b)
        #input()
        c = a.merge(b, left_on=['k1', 'k2', 'k3', 'k4', 'j', 't'],
                    right_on=['k1', 'k2', 'k3', 'k4', 'j', 't'])

        spe = bg_ham.full_spe()

        spe_count = len(spe)
        spe_line = '! {}\n! {}  '.format('initial', 0) + ('{:>10.4f}' * spe_count) + '\n'
        s = spe_line.format(*spe)
        spe = self.full_spe()
        spe_line = '! {}\n  {}  '.format('fitted', 0) + ('{:>10.4f}' * spe_count) + '\n'
        sx = '!{:>2}{:>3}{:>3}{:>3}{:>4}{:>3}{:>13}{:>13}{:>13}\n'
        sxy = '{:>3}{:>3}{:>3}{:>3}{:>4}{:>3}{:>13.6f}{:>13.6f}{:>13.6f}\n'
        s += spe_line.format(*spe)
        s += sx.format('k1', 'k2', 'k3', 'k4', 'J', 'T', 'fitted', 'initial', 'delta')

        for index, row in c.iterrows():
            out_list = [int(row[x]) for x in ['k1', 'k2', 'k3', 'k4', 'j', 't']]
            #if row['group_num_x'] == 0:
            #    s += '{:>3}{:>3}{:>3}{:>3}{:>4}{:>3}{:>13.6f}{:>13}{:>13}\n'.format(*out_list, row['v_y'], '', '')
            #else:
            #    s += sxy.format(*out_list, row['v_y'], row['v_x'], row['v_y'] - row['v_x'])
            s += sxy.format(*out_list, row['v_y'], row['v_x'], row['v_y'] - row['v_x'])

        with open(name,'w') as f:
            f.write(s)

        return

    def make_files(self, a_min, a_max, group_folder):
        gn = 0
        for a in range(a_min, a_max + 1):
            folder = '{}/{}/{}/'.format(group_folder, MODEL_SPACE_NAME, a)
            make_folder(folder)
            self.int_at_mass(a, folder)
            gn = self.write_groups(folder)
        self.normalize_scale(A_CORE+2)
        return gn

    def full_tbme(self):

        full_tb = pd.merge(self.coul, self.strong, on=['k1', 'k2', 'k3', 'k4', 'j', 't'], how='outer',
                           suffixes=('c', 's'))

        if INCLUDE_COULOMB:
            if not INCLUDE_ISOTENSOR:
                if not INCLUDE_ISOVECTOR:
                    full_tb = full_tb.fillna(0)
                    full_tb['v'] = full_tb.apply(lambda x: float(x['vc']) + float(x['vs']),axis=1)
                    full_tb = full_tb[['k1', 'k2', 'k3', 'k4', 'j', 't', 'v']]
                    full_tb['kl1'] = full_tb.apply(lambda x: K_ORDER[int(x['k1']) - 1], axis=1)
                    full_tb['kl2'] = full_tb.apply(lambda x: K_ORDER[int(x['k2']) - 1], axis=1)
                    full_tb['kl3'] = full_tb.apply(lambda x: K_ORDER[int(x['k3']) - 1], axis=1)
                    full_tb['kl4'] = full_tb.apply(lambda x: K_ORDER[int(x['k4']) - 1], axis=1)
                    full_tb = full_tb.sort_values(by=['t', 'j', 'kl1', 'kl2', 'kl3', 'kl4'])
                    return full_tb
        full_tb = pd.merge(full_tb, self.ivtb, on=['k1', 'k2', 'k3', 'k4', 'j', 't'], how='outer',
                           suffixes=(None, 'iv'))
        full_tb = pd.merge(full_tb, self.ittb, on=['k1', 'k2', 'k3', 'k4', 'j', 't'], how='outer',
                           suffixes=(None, 'it'))
        full_tb = full_tb.fillna(0)
        #print(full_tb.head(20))
        #input()
        full_tb['v'] = full_tb.apply(lambda x: float(x['vc']) + float(x['vs']) + float(x['v']) + float(x['vit']),
                                     axis=1)
        #if withGroupNum: was considering having the option to include group_num to use in comparison_to_bg() but realized
        # that the coulomb and strong tbme are all added together so the group number thing wasn't really as important
        # it was just to only print out the initial and difference to the bg when something was actually fit.
        #    full_tb = full_tb[['k1', 'k2', 'k3', 'k4', 'j', 't', 'v', 'group_num']]
        #else:
        full_tb = full_tb[['k1', 'k2', 'k3', 'k4', 'j', 't', 'v']]
        #full_tb['kl1'] = full_tb.apply(lambda x: K_DICT[str(int(x['k1']))], axis=1)
        #full_tb['kl2'] = full_tb.apply(lambda x: K_DICT[str(int(x['k2']))], axis=1)
        #full_tb['kl3'] = full_tb.apply(lambda x: K_DICT[str(int(x['k3']))], axis=1)
        #full_tb['kl4'] = full_tb.apply(lambda x: K_DICT[str(int(x['k4']))], axis=1)
        full_tb['kl1'] = full_tb.apply(lambda x: K_ORDER[int(x['k1'])-1], axis=1)
        full_tb['kl2'] = full_tb.apply(lambda x: K_ORDER[int(x['k2'])-1], axis=1)
        full_tb['kl3'] = full_tb.apply(lambda x: K_ORDER[int(x['k3'])-1], axis=1)
        full_tb['kl4'] = full_tb.apply(lambda x: K_ORDER[int(x['k4'])-1], axis=1)
        full_tb = full_tb.sort_values(by=['t', 'j', 'kl1', 'kl2', 'kl3', 'kl4'])
        return full_tb

    def full_spe(self):
        x = self.spe.copy()
        orbits = len(K_ORDER)
        spe = np.zeros(orbits)
        # If you wanted to add mass dependence to the SPE, this is where you'd need to start.
        # Would need to separate strong and coulomb again, and only list strong under neutron orbits in spe input
        for index, row in x.iterrows():
            if row['type'] == 'c':
                spe[row['k'] - 1] += float(row['v'])
            if row['type'] == 's':
                spe[row['k'] - 1] += float(row['v'])
                #spe[row['k'] - 1 + int(orbits / 2)] += float(row['v'])
        return spe

    def normalize_scale(self, a):
        coul_scale = (float(self.coul_mass) / float(a)) ** self.coul_expo
        strong_scale = (float(self.strong_mass) / float(a)) ** self.strong_expo

        if INCLUDE_COULOMB:
            self.coul.loc[:, 'v'] = self.coul['v'].apply(lambda x: x * coul_scale)
            self.coul = order_tbme(self.coul)
        self.strong.loc[:, 'v'] = self.strong['v'].apply(lambda x: x * strong_scale)

        self.coul_mass = int(a)
        self.strong_mass = int(a)


        self.strong = order_tbme(self.strong)
        if INCLUDE_ISOTENSOR:
            self.ittb = self.make_isotensor(self.a_it)
        if INCLUDE_ISOVECTOR:
            self.ivtb = self.make_isovector(self.a_iv)


    def int_at_mass(self, a, folder):
        self.normalize_scale(a)
        if INCLUDE_COULOMB or INCLUDE_ISOVECTOR or INCLUDE_ISOTENSOR:
            self.full_tb = self.full_tbme()
        else:
            self.full_tb = self.strong
        self.full_sp = self.full_spe()
        s = self.int_file(self.full_sp, self.full_tb, 'hamil.A{:02d}.int'.format(a))
        with open('{}/hamil.int'.format(folder), 'w') as f:
            f.write(s)

    def store(self, folder, assume_current=True):
        with open('{}/all.spe'.format(folder), 'w') as f:
            for index, row in self.spe.iterrows():
                s = '{:<3}{:<3}{:>10.5f}\n'.format(row['type'], row['k'], row['v'])
                f.write(s)
        with open('{}/strong.tbme'.format(folder), 'w') as f:
            s = self.int_file(None, self.strong, 'strong', replace_k=False)
            f.write(s)

        self.int_at_mass(self.strong_mass, folder)

        if INCLUDE_COULOMB:
            with open('{}/coul.tbme'.format(folder), 'w') as f:
                s = self.int_file(None, self.coul, 'coul', replace_k=False)
                f.write(s)
        if assume_current:
            with open('current.iter', 'w') as f:
                xlist = [os_path('{}/all.spe'.format(folder)), os_path('{}/strong.tbme'.format(folder)),
                         os_path('{}/coul.tbme'.format(folder)), self.a_iv, self.a_it]
                s = '{}\n{}\n{}\n{:<10.6f}\n{:<10.6f}\n'.format(*xlist)
                f.write(s)

        with open('{}/current.iter'.format(folder), 'w') as f:
            xlist = [os_path('{}/all.spe'.format(folder)), os_path('{}/strong.tbme'.format(folder)),
                     os_path('{}/coul.tbme'.format(folder)), self.a_iv, self.a_it]
            s = '{}\n{}\n{}\n{:<10.6f}\n{:<10.6f}\n'.format(*xlist)
            f.write(s)

    def make_isotensor(self, alpha):
        x = self.strong
        y = x[(x['type'] == 'pn') & (x['t'] == 1)]
        y.loc[:, 'v'] = y['v'].apply(lambda i: i * alpha)
        return y

    def make_isovector(self, alpha):
        x = self.strong
        y = x[(x['type'] != 'pn') & (x['t'] == 1)]
        y.loc[:, 'v'] = y['v'].apply(lambda i: alpha * i)
        z = []
        for index, row in y.iterrows():
            if row['type'] == 'nn':
                z.append(row['v'])
            if row['type'] == 'pp':
                z.append(-row['v'])

        y.loc[:, 'v'] = z
        return y

    def int_file(self, spe, tbme, name, replace_k=True):
        if tbme is None:
            nmat = 1
        else:
            nmat = len(tbme.index)
        nmat=0
        if spe is not None:
            spe_count = len(spe)
            spe_line = '! {}\n  {}  '.format(name, nmat) + ('{:>10.4f}' * spe_count) + '\n'
            s = spe_line.format(*spe)
        else:
            s = ''
        if tbme is None:
            s += "{:>3}{:>3}{:>3}{:>3}{:>4}{:>3}{:>13.6f}\n".format(1, 1, 1, 1, 0, 1, 0.0)
        else:
            for index, row in tbme.iterrows():
                out_list = [int(row[x]) for x in ['k1', 'k2', 'k3', 'k4', 'j', 't']]
                #if replace_k:
                #    for i in range(4):
                #        out_list[i] = K_DICT[str(out_list[i])]
                s += '{:>3}{:>3}{:>3}{:>3}{:>4}{:>3}{:>13.6f}\n'.format(*out_list, row['v'])
        return s

    def write_groups(self, folder):
        label_file = open('fit_labels.dat', 'w')
        spe = self.spe['v'].tolist()
        spe_count = len(spe)
        if not INCLUDE_COULOMB:
            spe_count *= 2
        g_num = 0

        for index, row in self.spe.iterrows():
            x = [0.0] * spe_count
            if row['type'] == 'c':
                x[row['k'] - 1] = row['v']
            if row['type'] == 's':
                x[row['k'] - 1] = row['v']
                #x[row['k'] - 1 + int(spe_count / 2)] = row['v']
            g_num += 1
            s = self.int_file(x, None, g_num)
            spin = orbit_spin(K_ORDER[g_num-1])
            if int(row['k']) <= len(PROTON_ORBITS):
                spe_name = 'spe p{}'.format(spin)
            else:
                spe_name = 'spe n{}'.format(spin)
            #spe_name = 'spe-{}'.format(spin)
            with open('{}/g{}.int'.format(folder, g_num), 'w') as f:
                f.write(s)
            label_file.write('{:>4}{:>20}{:>4}\n'.format(g_num, spe_name, 1))

        for key, item in self.grouped:
            chunk = self.grouped.get_group(key)
            if key != 'other':
                g_num += 1
                with open('{}/g{}.int'.format(folder, g_num), 'w') as f:
                    z = [0.] * spe_count
                    s = self.int_file(z, chunk, g_num)
                    f.write(s)
                label_file.write('{:>4}{:>20}{:>4}\n'.format(g_num, key, len(chunk)))

        xlist = []
        keylist = []
        if INCLUDE_ISOVECTOR:
            xlist.append(self.ivtb)
            keylist.append('isovector')
        if INCLUDE_ISOTENSOR:
            xlist.append(self.ittb)
            keylist.append('isotensor')
        if INCLUDE_COULOMB:
            xlist.append(self.coul)
            keylist.append('coulomb')

        for i in range(len(xlist)):
            x = xlist[i]
            key = keylist[i]
            g_num += 1
            with open('{}/g{}.int'.format(folder, g_num), 'w') as f:
                z = [0.] * spe_count
                s = self.int_file(z, x, g_num)
                f.write(s)
            label_file.write('{:>4}{:>20}{:>4}\n'.format(g_num, key, len(x)))

        label_file.close()

        with open('{}/parts.nux'.format(folder), 'w') as g:
            for i in range(g_num):  #THIS IS REMOVED TEMPORARILY TO INCREASE SPEED
                if i+1 not in DO_NOT_FIT or CALCULATE_ALL:
                    g.write("g" + str(i + 1) + '\n')
        with open('{}/finch.nux'.format(folder), 'w') as g:
            g.write('0')
        #input('groups made')
        return g_num


'''
This section covers running NushellX for a given set of data with a given Hamiltonian (handled above) and extracting 
the important data for the fit
'''

def fix_width_dataframe(file, name=''):
    '''
    quickly created function to ensure that a new data set is set in columns with correct spacing/widths
    so that setup_dataframe(file, ham) doesn't break
    '''
    with open(file, 'r') as f:
        if name != '':
            new_name = file[:-4:] + '-fixed.dat'
        else:
            new_name = 'fwf-levels.dat'
        with open(new_name, 'w') as g:
            lines = f.readlines()
            for line in lines:
                spl = line.split()
                if len(spl) == 0:
                    continue
                x = ['']*10
                for i in range(0,9):
                    if i + 1 == len(x):
                        x[i] = ' '.join(spl[i:])
                    else:
                        if i == 7 or i == 8:
                            x[i] = float(spl[i])
                        else:
                            x[i] = spl[i]
                s = '{:>3}{:>3}{:>5}{:>3}{:>6}{:>4}{:>4}{:>12.4f}{:>10.4f}{:>50}\n'
                g.write(s.format(*x))

#fix_width_dataframe('sdpf-levels')
#input()


def setup_dataframe(file, ham):
    fix_width_dataframe(file)
    pf = pd.read_fwf('fwf-levels.dat',
                     names=['Element', 'A', '2Tz', '2T', '2J', 'P', 'jnum', 'BE', 'error', 'comment'],
                     widths=[3, 3, 5, 3, 6, 4, 4, 12, 10, 50],
                     header=None)


    convert_dic = {'A': int, '2T': int, '2J': int, 'BE': float, 'error': float}

    pf = pf.astype(convert_dic)

    # added to speedup sd fit, need to remove later

    #pf = pf[pf['2T'] == 1]
    #pf = pf[pf['jnum'] == 1]
    #pf = pf[pf['2J'] < 6]

    #print(pf)
    #input()

    pf['Z'] = pf.apply(lambda x: (x['A'] - x['2Tz']) / 2, axis=1).astype(int)
    pf['N'] = pf.apply(lambda x: (x['A'] + x['2Tz']) / 2, axis=1).astype(int)


    pf['Zval'] = pf.apply(lambda x: x['Z'] - Z_CORE, axis=1).astype(int)
    pf['Aval'] = pf.apply(lambda x: x['A'] - A_CORE, axis=1).astype(int)

    pf['Energy'] = pf['BE'] * (pf['BE'] < 0.0)
    pf['Energy'].replace(0, np.nan, inplace=True)
    pf['Energy'].fillna(method='ffill', inplace=True)
    pf['Energy'] = pf['Energy'] + pf['BE'] * (pf['BE'] > 0.0)

    pf['Nucleus'] = pf.apply(lambda x: '{}{}'.format(swap_element_label(x.Z), x.A), axis=1)
    pf['NucJ'] = pf.apply(lambda x: '{}-{}-{}'.format(x.Nucleus, x['2J'], x['P']), axis=1).astype(str)
    pf['NucJ-lev'] = pf.apply(lambda x: '{}-{}-{}-{}'.format(x.Nucleus, x['2J'], x['P'], x['jnum']), axis=1).astype(str)
    #print(pf.head(20))
    #input()
    pf['NuName'] = pf.apply(
        lambda x: 'xy{}{}{}{}'.format(cycle_lbl(x['Zval']), cycle_lbl(x['Aval']), x['P'], cycle_lbl(x['2J'])),
        axis=1).astype(str)

    pf['lptName'] = pf.apply(lambda x: '{}{}y.lpt'.format(swap_element_label(x.Z, fixlength=True), x.A), axis=1).astype(
        str)
    # pf = pf.drop(['comment'], axis=1)

    return pf


def generate_ans(fit_loc, a, z, j, p, jnum, min_error):
    j = str(round(float(j) / 2., 1))
    s = "lpe,    {}\n{}\nn\nhamil\n{:>3}\n{:>3}\n{:>4},{:>4}, 1.0\n{:>3}{:>3}\nst\n"
    filename = os.path.join(fit_loc, "mop.ans")
    with open(filename, 'w') as f:
        # Readd this if you want to save time AND all excited levels in data have well known ground states.
        #if min_error > 2.:
        #    f.write(s.format(jnum+1, MODEL_SPACE_NAME, z, a, j, j, p, 0))
        #else:
        #    f.write(s.format(jnum+1, MODEL_SPACE_NAME, z, a, j, j, p, 2))
        f.write(s.format(jnum+1, MODEL_SPACE_NAME, z, a, j, j, p, 2))

    for filename in glob.glob(os.path.join('groups/{}/{}/'.format(MODEL_SPACE_NAME, a), '*.*')):
        shutil.copy(filename, fit_loc)
    shutil.copy(os.path.join('groups/{}/'.format(MODEL_SPACE_NAME), '{}.mod'.format(MODEL_SPACE_NAME)), fit_loc)


def single_calc(x):
    df_nuc, group_count, run_name = x
    nuc = df_nuc['NucJ'].iloc[0]

    wd = os.path.join('calculations/{}/'.format(MODEL_SPACE_NAME), nuc)
    make_folder(wd)

    generate_ans(wd, df_nuc.A.iloc[0], df_nuc.Z.iloc[0], df_nuc['2J'].iloc[0], df_nuc['P'].iloc[0],
                 df_nuc['jnum'].max(), df_nuc['error'].min())

    nuc_name = '{}{}'.format(swap_element_label(df_nuc.Z.iloc[0], fixlength=True), df_nuc.A.iloc[0])


    call('shell mop.ans', shell=True, cwd=wd, stdout=DEVNULL, stderr=DEVNULL)

    call('mop.bat', shell=True, cwd=wd, stdout=DEVNULL, stderr=DEVNULL)

    add_trs_files('{}/{}'.format(wd, nuc_name))
    for index, row in df_nuc.iterrows():
        k = row['jnum']
        nuc_ovl = []
        lpt_name = os.path.join(wd, row['lptName'])
        #if not exists(lpt_name):
        #    continue
        lpt = read_lpt(lpt_name)
        #if row['error'] < 2.:  readd this if changed at line 588
        for i in range(0, group_count):
            if i+1 not in DO_NOT_FIT or CALCULATE_ALL:
                ovl_name = os_path('{}/{}.xg{}'.format(wd, row['NuName'], i + 1))
                ovl = np.loadtxt(ovl_name, skiprows=1)
                ovl = np.atleast_2d(ovl)
                nuc_ovl.append(str(round(float(ovl[k - 1, 1]), 4)))
        nuc_ovl.insert(0, float(lpt[lpt['numj'] == k]['energy']) + ZERO_BODY_TERM)
        write_csv_file(os_path('{}/ovl/{}-{}.xfit'.format(run_name, nuc, k)), nuc_ovl)
        copyfile(os.path.join(wd, row['lptName']), os_path('{}/ovl/{}.lpt'.format(run_name, nuc)))
        #need to copy occ file as well


def run_multiproc(df, r_name, group_count):
    nuc_list = df['NucJ'].unique().tolist()
    #print(nuc_list)
    #input()
    chunks = []
    for nuc in nuc_list:
        #print(df[df['NucJ'] == nuc])
        #input()
        chunks.append([df[df['NucJ'] == nuc], group_count, r_name])
    #print(chunks)
    make_folder('{}/ovl'.format(r_name))
    p = multiprocessing.Pool(NUM_PROCESSORS)

    for _ in tqdm(p.imap_unordered(single_calc, chunks), total=len(chunks), ncols=50):
        pass
    p.close()
    p.join()


# NEED TO GENERALIZE LINESKIPS/ HAVE A GLOBAL VARIABLE
def read_lpt(location, lineskips=6, corner=False):
    with open(location, 'r') as f:
        int_contents = f.readlines()
        if corner:
            for line in int_contents:
                spl = line.split()
                if len(spl) != 0:
                    if spl[0] == '1':
                        be = float(spl[-1]) - 127.619
                        return be
        # noinspection PyTypeChecker
        y = np.genfromtxt(int_contents, delimiter=[5, 5, 9, 6, 5, 9, 15, 13], skip_header=lineskips,
                          skip_footer=0,
                          dtype=str,
                          usecols=(0, 1, 2, 3, 4, 6, 7))
        y = np.atleast_2d(y)

        levels = pd.DataFrame(y, columns=['num', 'numj', 'excitation', 'J', 'T', 'filename', 'energy'])
        convert_dict = {'num': int,
                        'numj': int,
                        'excitation': float,
                        'J': str,
                        'T': str,
                        'filename': str,
                        'energy': float
                        }
        levels = levels.astype(convert_dict)

    return levels


'''
The meat of the fitting is done here, everything above is for handling the NushellX code and physics. Beneath here is 
pure math and results storing
'''


def avg_group_ratios(a, b):
    # spe
    sp_ratio = [a.spe['v'].values.tolist()[i] / b.spe['v'].values.tolist()[i] for i in
                range(len(a.spe['v'].values.tolist()))]

    # tbme
    v_ratio = [a.strong['v'].values.tolist()[i] / b.strong['v'].values.tolist()[i] for i in
               range(len(a.strong.values.tolist()))]
    v_groups = a.strong['group_num']


    x = pd.DataFrame({'ratio': pd.Series(v_ratio, dtype=float), 'num': pd.Series(v_groups, dtype=int)})
    ratio_means = x.groupby('num').mean().reset_index().sort_values(by='num')['ratio'].values.tolist()


    #print(v_groups.min())
    #input()
    if v_groups.min() == 0:
        ratio_means = ratio_means[1:]
    # ratio_means = np.atleast_1d(ratio_means)
    # remaining groups
    iso_ratio = []
    c_ratio = []
    if INCLUDE_COULOMB:
        c_ratios = [a.coul['v'].values.tolist()[i] / b.coul['v'].values.tolist()[i] for i in
                    range(len(a.coul['v'].values.tolist()))]
        c_ratio = [sum(c_ratios) / float(len(c_ratios))]

    if INCLUDE_ISOVECTOR:
        iso_ratio.append(a.a_iv / b.a_iv)

    if INCLUDE_ISOTENSOR:
        iso_ratio.append(a.a_it / b.a_it)
    z = sp_ratio + ratio_means + iso_ratio + c_ratio
    #print(z)
    #input()

    return z


def setup_b_list(df):
    df = df[df['error'] < 2.]
    p_df = df[df['2Tz'] < 0]
    n_df = df[df['2Tz'] > 0]
    df = p_df.merge(n_df, on=['A', '2T', '2J', 'jnum'], how='inner')
    df['E'] = df['Energy_x'] - df['Energy_y']
    df['BE'] = df['BE_x'] - df['BE_y']
    df['BE'] = df['BE'] / df['2T']
    df['error'] = (df['error_x'] + df['error_y'])
    df = df[['A', '2T', '2J', 'jnum', 'E', 'BE', 'error', 'NucJ_x', 'NucJ_y', 'Nucleus_x']]
    return df


def setup_c_list(df):
    df = df[df['error'] < 2.]
    df = df[df['2T'] == 2]
    df1 = df[df['2Tz'] == 2]
    df2 = df[df['2Tz'] == 0]
    df3 = df[df['2Tz'] == -2]
    start = 10
    startj = 10
    count = 0
    x = []
    for index, row in df2.iterrows():
        if row['A'] != start:
            x.append(1)
            count = 1
            start = row['A']
        else:
            count += 1
            x.append(count)
    df2['num'] = x
    start = 10
    startj = 10
    count = 0
    x = []
    df2.sort_values(by=['A', '2J'], inplace=True)
    for index, row in df2.iterrows():
        if row['A'] != start or row['2J'] != startj:
            x.append(1)
            count = 1
            start = row['A']
            startj = row['2J']
        else:
            count += 1
            x.append(count)
    df2['jnum'] = x

    df1 = df1.merge(df2, on=['A', '2T', '2J', 'jnum'], how='inner')

    df = df1.merge(df3, on=['A', '2T', '2J', 'jnum'], how='inner')

    df['BE'] = (df['Energy_x'] + df['Energy'] - 2 * df['Energy_y']) / (df['2T'])
    df['error'] = (df['error_x'] + 2 * df['error_y'] + df['error']) / df['2T']
    df.reset_index(inplace=True)
    df = df[['A', '2T', '2J', 'BE', 'error', 'NucJ-lev_x', 'NucJ-lev', 'NucJ-lev_y', 'num']]

    df = df.sort_values(['num','A'])

    return df


def setup_b_diff_list(df):
    df5 = df[(df['A'] < 28) & (df['jnum'] == 1) & (df['2J'] == 5) & (df['2T'] == 1)]
    df1 = df[(df['A'] > 28) & (df['A'] < 32) & (df['jnum'] == 1) & (df['2J'] == 1) & (df['2T'] == 1)]
    df3 = df[(df['A'] > 32) & (df['jnum'] == 1) & (df['2J'] == 3) & (df['2T'] == 1)]
    df = pd.concat([df5, df1, df3], ignore_index=True)

    df_1 = df.copy()
    df_2 = df.copy()
    df_1['A'] += 2
    df_2 = df_2.merge(df_1, how='inner', on=['A', '2T', 'jnum'])
    df_2['BE'] = df_2['E_x'] - df_2['E_y']
    df_2['BE'] = df_2['E_x'] - df_2['E_y']
    df_2['error'] = (df_2['error_x'] ** 2. + df_2['error_y'] ** 2.) ** .5

    return df_2


# COMBINE THESE TWO INTO ONE
def subset_groups(group_list, hold, no_energy=False):
    group_list = list(group_list)
    if len(group_list) > 1:
        for i in sorted(hold, reverse=True):
            if no_energy:
                del (group_list[i - 1])
            else:
                #print(group_list,i)
                del (group_list[i])
    return group_list


def superset_groups(group_list, hold, dimension, no_energy=False):
    x = [1] * dimension
    j = 0
    for i in range(dimension):
        if (i + 1) not in hold:
            #print(i, group_list[j])
            x[i] = group_list[j]
            j += 1
    #print('x')
    #print(x)
    return x


def perform_fit(level_data, folder, cur_ham, bg_ham, varied, dimension, held_groups):
    med_data = setup_b_list(level_data)

    med_diff_data = setup_b_diff_list(med_data)

    c_data = setup_c_list(level_data)

    dim = dimension - len(held_groups)

    fg = open('contributions.dat', 'w')

    first_list = subset_groups(cur_ham.strength_list(), held_groups, no_energy=True)

    first_list_bg = subset_groups(bg_ham.strength_list(), held_groups, no_energy=True)

    def setup_fit(df, store_output=True, iterate_fit=True, multipliers=None, fit_type='energy'):
        if store_output:
            output_file = open(os_path('{}/output-{}.dat'.format(folder, fit_type)), 'w')

        e_matrix = np.zeros((dim, dim), dtype=float)
        e_vector = np.zeros(dim, dtype=float)
        last_nucjx = 'boop'
        rms, count = 0., 0
        chi = 0.
        for index, row in df.iterrows():
            e_energy, e_error = row['BE'], row['error']

            def read_xfit(loc, nuc, k):
                if k == 'x':
                    return np.genfromtxt(os_path('{}/ovl/{}.xfit'.format(folder, nuc)),
                                         delimiter=',', dtype=float)
                else:
                    return np.genfromtxt(os_path('{}/ovl/{}-{}.xfit'.format(folder, nuc, k)),
                                         delimiter=',', dtype=float)

            if fit_type == 'energy':
                sigma_squared = (e_error ** 2 + .140 ** 2)
                ovl_list = read_xfit(folder, row['NucJ'], row['jnum'])
                ovl_list = np.atleast_1d(ovl_list)
                if e_energy < 0.0:
                    gs_ovl_list = ovl_list.copy()
                else:
                    ovl_list -= gs_ovl_list[:len(ovl_list)]

                out_label = row['NucJ'] + '-' + str(row['jnum'])

            def b_ovl_list(nuc_1, nuc_2, num):
                ovl_list_x = read_xfit(folder, nuc_1, num)
                ovl_list_y = read_xfit(folder, nuc_2, num)

                ovl_list = ovl_list_x - ovl_list_y
                ovl_list = np.atleast_1d(ovl_list)
                ovl_list = ovl_list / row['2T']
                return ovl_list

            if fit_type == 'med':
                sigma_squared = (e_error ** 2 + .070 ** 2)
                ovl_list = b_ovl_list(row['NucJ_x'], row['NucJ_y'], row['jnum'])
                if row['Nucleus_x'] == last_nucjx:
                    ovl_list -= gsb_ovl_list
                else:
                    last_nucjx = row['Nucleus_x']
                    gsb_ovl_list = ovl_list.copy()
                out_label = 'MED {} {:>12}{:>6}'.format(row['A'], row['NucJ_x'], row['jnum'])

            if fit_type == 'med_diff':
                sigma_squared = (e_error ** 2 + .050 ** 2)
                ovl_list_1 = b_ovl_list(row['NucJ_x_x'], row['NucJ_y_x'], row['jnum'])
                ovl_list_2 = b_ovl_list(row['NucJ_x_y'], row['NucJ_y_y'], row['jnum'])
                ovl_list = ovl_list_1 - ovl_list_2
                out_label = 'delMED {} {:>12}{:>6}'.format(row['A'], row['NucJ_x_x'], row['jnum'])

            if fit_type == 'c':
                sigma_squared = (e_error ** 2 + .015 ** 2)
                ovl_list_1 = read_xfit(folder, row['NucJ-lev_x'], 'x')
                ovl_list_2 = read_xfit(folder, row['NucJ-lev_y'], 'x')
                ovl_list_3 = read_xfit(folder, row['NucJ-lev'], 'x')

                ovl_list = ovl_list_1 + ovl_list_3 - 2 * ovl_list_2
                ovl_list = np.atleast_1d(ovl_list)

                ovl_list = ovl_list / float(row['2T'])

                out_label = 'c {} {:>12}{:>6}'.format(row['A'], row['NucJ-lev'], row['num'])

            if CALCULATE_ALL:
                ovl_list = subset_groups(ovl_list, held_groups)
            t_energy = ovl_list[0]

            energy_diff = t_energy - e_energy

            s = '{:>12.4f}' * len(ovl_list) + '\n'
            s = out_label + '{:>12.4f}{:>12.4f}'.format(row['BE'], row['error']) + s

            #if fit_type != 'energy':
            fg.write(s.format(*ovl_list))

            if store_output:
                out_string = "{:<30}{:>10.4f}{:>10.4f}{:>10.4f}{:>10.4f}\n"
                out_line = out_string.format(out_label, e_energy, e_error, t_energy, energy_diff)
                output_file.write(out_line)

            if e_error < 2.00 and iterate_fit:  # remove non model space shell levels from fit
                fit_list = [float(i) for i in ovl_list[1:]]
                fit_energy = np.sum(fit_list) - energy_diff
                if multipliers is not None:
                    fit_list = [fit_list[j] * multipliers[j] for j in range(len(fit_list))]

                fit_list = [fit_list[j] / first_list[j] for j in range(len(fit_list))]
                for k in range(0, dim):
                    e_vector[k] += ((fit_energy * fit_list[k]) / sigma_squared)
                    for j in range(0, dim):
                        e_matrix[k, j] += ((fit_list[k] * fit_list[j]) / sigma_squared)
                chi += ((energy_diff) ** 2. / sigma_squared)
                if e_error < 0.25:
                    rms += energy_diff ** 2.
                    count += 1

        rms = (rms / count) ** .5
        #print('the {} chi-squared ratio is {:>15.4f}{:>15.4f}'.format(fit_type, chi/(count), (chi/(count))**.5))
        out_line = '\nrms deviation: {:>12.4f}\n{}'.format(rms, count)
        if store_output:
            output_file.write(out_line)
            output_file.close()
        return rms, e_matrix, e_vector


    fit_matrix = np.zeros((dim, dim), dtype=float)
    fit_vector = np.zeros(dim, dtype=float)
    data_types = [level_data, med_data, c_data, med_diff_data]
    data_labels = ['energy', 'med', 'c', 'med_diff']
    rms_outputs = []
    s = ''
    for k in range(len(DATA_TYPE_INCLUSION)):
        if len(data_types[k]) > 0:
            a, b, c = setup_fit(data_types[k], fit_type=data_labels[k])
            if DATA_TYPE_INCLUSION[k]:
                #a, b, c = setup_fit(data_types[k],fit_type=data_labels[k])
                fit_matrix += b
                fit_vector += c
                rms_outputs.append(a)
                s += '{:>8} rms: {:>8.3f} '.format(data_labels[k], a)
    fg.close()
    console_output_str = s



    # THE ALGORITHM
    At, D, A = np.linalg.svd(fit_matrix)
    np.savetxt('{}/svd.dat'.format(folder), D, delimiter=',')
    np.savetxt('{}/vectors.dat'.format(folder), At, delimiter=',')

    di = 1. / D
    c = np.dot(A, fit_vector)
    yCurrent = c * di
    avg_ratios = avg_group_ratios(bg_ham, cur_ham)
    avg_ratios = subset_groups(avg_ratios, held_groups, no_energy=True)
    #yNew = np.dot(A, avg_ratios)
    yNew = np.dot(A, first_list_bg)
    #print(first_list)
    #print(first_list_bg)

    new_multipliers = []
    new_covariance = []
    for n in range(0, dim + 1):
        # calculate all multipliers for each vlc
        y = yNew.copy()
        for i in range(0, n):
            y[i] = yCurrent[i]
        new_mults = np.dot(At, y)
        new_mults = [ new_mults[j] / first_list[j] for j in range(len(first_list))] #added to change fitting procedure
        new_multipliers.append(new_mults)

        # Store the resulting Hamiltonians in subfolder
        if STORE_EVERY_VLC:
            x = superset_groups(new_mults, held_groups, dimension)
            vlc_ham, vlc_bg_rms, vlc_var_rms = update_hamiltonian(cur_ham, bg_ham, x)
            vlc_folder = '{}/ivlc/ivlc-{}'.format(folder, n)
            make_folder(vlc_folder)
            vlc_ham.store(vlc_folder, assume_current=False)
            vlc_ham.comparison_to_bg(bg_ham, '{}\\hamil-comparison.int'.format(vlc_folder))

        #Caclulate covariances
        D_new = D.copy()
        for i in range(0, n):
            D_new[i] = 1. / D_new[i]
        for i in range(n, len(D_new)):
            D_new[i] = 0.
        D_new = np.diag(D_new)
        covariance_vlc = np.dot(At.T, np.dot(D_new, At))
        new_covariance.append(covariance_vlc)

    with open(os_path('{}/output-{}.dat'.format(folder, 'mults')), 'w') as f:
        for i in range(len(new_multipliers)):
            x = []
            for j in range(len(new_covariance[i])):
                x.append(new_covariance[i][j, j] ** 0.5)
            s = '{:>12.4f}' * len(new_multipliers[i]) + '\t'
            f.write(s.format(*new_multipliers[i]))
            s = '{:>12.4f}' * len(new_multipliers[i]) + '\n'
            f.write(s.format(*x))

    # SUGGESTED CHANGE
    '''
    The iteration to the next varied linear combination (while backward) is really just repeating the last iteration of
    the previous vlc run. So to save an iteration per vlc on the way backward, code in a way to retrieve all multipliers
    easily and jump directly to next vlc here.
    '''

    x = superset_groups(new_multipliers[varied], held_groups, dimension)
    #print(x)
    #input('waiting')

    mult_rms = rms_from_lists(new_multipliers[varied], np.ones(len(new_multipliers[varied])))
    new_ham, bg_rms, variable_rms = update_hamiltonian(cur_ham, bg_ham, x)

    new_ham.store(folder)
    new_ham.comparison_to_bg(bg_ham, '{}\\hamil-comparison.int'.format(folder))

    #print("\titeration rms: {:>6.3f} MeV".format(variable_rms))
    #print("\tbg var rms: {:>6.3f} MeV".format(bg_rms))

    #print(console_output_str)

    return rms_outputs, bg_rms, mult_rms


def tbme_plot(r_name, bg_ham, curr_ham, vlc_num):
    # I want a TBME scatter plot for fitted TBME
    a = bg_ham.full_tbme().copy()
    #a = a[a['group'] != 'other']
    b = curr_ham.full_tbme().copy()
    #b = b[b['group'] != 'other']
    c = a.merge(b, left_on=['k1', 'k2', 'k3', 'k4', 'j', 't'],
                right_on=['k1', 'k2', 'k3', 'k4', 'j', 't'])

    fig = plt.figure(figsize=(6, 6))
    matplotlib.rcParams.update({'font.size': 12})
    ax = Subplot(fig, 111)
    fig.add_subplot(ax)
    ax.set_xlabel('TBME (MeV) starting')
    ax.set_ylabel('TBME (MeV) with {} vlc'.format(vlc_num))
    ax.set_aspect('equal')
    x, y = c['v_x'].tolist(), c['v_y'].tolist()
    minf = int(min([min(x), min(y)])) - 1
    maxf = int(max([max(x), max(y)])) + 1
    plt.xticks(np.arange(minf, maxf, 1.))
    plt.yticks(np.arange(minf, maxf, 1.))
    plt.plot([minf, maxf], [minf, maxf], ls='--', color='black')
    ax.scatter(x, y)
    ax.axhline(0., ls='-', color='black')
    ax.axvline(0., ls='-', color='black')
    plt_name = r_name + '\\tbme-vlc-{}.png'.format(vlc_num)
    plt.savefig(plt_name)
    plt.close(fig)

    return

def add_trs_files(nucleus):
    with open(nucleus + "0a.trs", 'w') as f:
        f.write('1')
    with open(nucleus + "1a.trs", 'w') as f:
        f.write('1')
    with open(nucleus + "0b.trs", 'w') as f:
        f.write('1')
    with open(nucleus + "1b.trs", 'w') as f:
        f.write('1')


def yes_no(answer):
    yes = set(['yes', 'y', 'ye', ''])
    no = set(['no', 'n'])
    while True:
        choice = raw_input(answer).lower()
        if choice in yes:
            return True
        elif choice in no:
            return False
        else:
            print("Please respond with \'yes\' or \'no\'")


def is_non_zero_file(fpath):
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0

if __name__ == "__main__":
    fit_folders = ['calculations', 'groups', 'runs']
    for f in fit_folders:
        make_folder('{}/'.format(f))





    bg_ham = Hamiltonian(BG_HAM_FANS, group_folder='groups-ai')
    initial_ham = Hamiltonian(INITIAL_HAM_FANS)

    make_model_space()


    num_groups = initial_ham.group_count


    initial_ham.store('groups/{}/'.format(MODEL_SPACE_NAME))
    fit_data = setup_dataframe(DATA_FILE_NAME, initial_ham)

    if NEW_RUN:
        now = datetime.datetime.now()
        run_name = "runs/{}/{}-{}-{}_{}-{}".format(MODEL_SPACE_NAME, now.year, now.month, now.day, now.hour, now.minute)
        make_folder(run_name)
        with open('run_name.dat', 'w') as f:
            f.write(run_name)
    else:
        with open('run_name.dat', 'r') as f:
            run_name = f.readlines()[0].strip()

    copyfile(user_arg, '{}/{}'.format(run_name, user_arg))
    held_groups = []
                         

    # if not fitting on c coeff or delMED than do not fit isotensor/isovector
    if DATA_TYPE_INCLUSION[2] == False and INCLUDE_ISOTENSOR:
        held_groups.append(num_groups-1)
    if DATA_TYPE_INCLUSION[3] == False and INCLUDE_ISOVECTOR:
        held_groups.append(num_groups - 2)

    if len(DO_NOT_FIT) > 0:
        held_groups = DO_NOT_FIT

    # held_groups = []

    if MAX_GROUPS_CHOICE == 0:
        max_groups = num_groups - len(held_groups)
    else:
        max_groups = MAX_GROUPS_CHOICE

    print('Maximum Varied Linear Combinations: {}\n\n'.format(max_groups))

    iteration_number = 1
    with open(run_name + '/rms.dat', 'w') as f:
        f.write("{:>5}{:>5}{:>5}{:>10}{:>10}\n".format("i", 'vlc', 'v_i', 'var rms', 'e rms'))
    energy_rms_list = []
    variable_rms_list = []

    open('stop.dat', 'w').close()


    for direction in ['forward', 'backward']:
        if VLC_CHOICES == []:
            if direction == 'forward':
                varied_lc_list = [i for i in range(5, max_groups, 5)] # for a full run
                varied_iter_max = FORWARD_ITER_MAX
            if direction == 'backward':
                varied_lc_list = [max_groups - i for i in range(0, max_groups + 1)] # for a full run
                varied_iter_max = BACKWARD_ITER_MAX
        else:
            varied_lc_list = VLC_CHOICES
            varied_iter_max = BACKWARD_ITER_MAX
        for varied_lcs in varied_lc_list:
            if varied_lcs > max_groups:
                input('Error: number of varied LCs is greater than number of groups')
                sys.exit()
            variable_rms, vlc_iteration = 100.0, 1
            rms_cutoff = 0.01 #in fractional percent of multipliers for parameters (0.01 is one percent)
            while variable_rms > rms_cutoff and vlc_iteration <= varied_iter_max:
                if is_non_zero_file('stop.dat'):
                    answer = yes_no('File stop.dat is not empty. Did you want to stop this run? (y/n)\n\n')
                    if answer:
                        sys.exit(0)

                s = 'n = {:<3d} vlc = {:<3d} i = {:<3d} initiated at {}'
                s = s.format(iteration_number, varied_lcs, vlc_iteration, time.strftime("%H:%M", time.localtime()))
                print(s)
                #input('stop and check')
                if direction == 'backward' and vlc_iteration == 1 and varied_lcs != varied_lc_list[0]:
                    last_fans = os_path('{}/ivlc/ivlc-{}/current.iter'.format(iteration_name, varied_lcs))
                    if exists(last_fans):
                        shutil.copy(last_fans, 'current.iter')



                iteration_name = os_path('{}/{}/{}/'.format(run_name, direction, iteration_number))
                make_folder(iteration_name)

                current_ham = Hamiltonian('current.iter')
                current_ham.make_files(A_MIN, A_MAX, 'groups/')

                if iteration_number == 1:
                    input('\nAll files generated, next is multiprocessing. This may take awhile.\n\nPress Enter to continue...')

                if NEW_RUN or iteration_number > NUM_SKIP_RUNS:
                    run_multiproc(fit_data, iteration_name, num_groups)

                energy_rms, bg_rms, variable_rms = perform_fit(fit_data, iteration_name, current_ham,
                                                               bg_ham, varied_lcs, num_groups, held_groups)


                
                data_labels = ['energy', 'med', 'c', 'med_diff']
                s = ''
                for k in range(len(DATA_TYPE_INCLUSION)):
                    if DATA_TYPE_INCLUSION[k]:
                        s += '{:>8} rms: {:>8.3f} '.format(data_labels[k], energy_rms[k])


                s += '  bg rms: {:>8.3f}  mult rms: {:>8.3f}'.format(bg_rms, variable_rms)
                print(s)

                print_list = [iteration_number, varied_lcs, vlc_iteration, bg_rms]
                print_list.extend(energy_rms)

                with open(run_name + '/rms.dat', 'a') as f:
                    s3 = "{:>5}{:>5}{:>5}" + '{:>10.4f}'*(len(energy_rms)+1) + '\n'
                    f.write(s3.format(*print_list))



                if PAUSE_BETWEEN and iteration_number <= NUM_SKIP_RUNS:
                    input("Waiting to continue...")

                if direction == 'backward':
                    if variable_rms <= rms_cutoff or vlc_iteration >= varied_iter_max:
                        vlc_name = os_path('{}/vlc-{}/'.format(run_name, varied_lcs))
                        make_folder(vlc_name)
                        current_ham.store(vlc_name)
                        current_ham.int_at_mass(A_CORE+2, vlc_name)
                        current_ham.comparison_to_bg(bg_ham, '{}\\hamil-comparison.int'.format(vlc_name))
                        energy_rms_list.append(energy_rms[0])
                        variable_rms_list.append(bg_rms)
                        tbme_plot(vlc_name, bg_ham, current_ham, varied_lcs)
                        for filename in glob.glob(os.path.join(iteration_name, 'output*.*')):
                            shutil.copy(filename, vlc_name)

                        print('\n\n\n')

                iteration_number += 1
                vlc_iteration += 1

