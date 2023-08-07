import csv
import os 
import glob
import numpy as np

import matplotlib.pyplot as plt
from hpe_from_imu.configuration import Config
from hpe_from_imu.configuration import Constants as C

conf = Config(C.config_path)
paths = conf["paths"]

def print_csv_to_tikz_bottom_up(experiments=[], side=''):
    print('\rReading CSV files')

    files = get_experiment_files('joints_angle_results', experiments)
    files.extend(get_experiment_files('joints_position_results', experiments))

    for file in files:
        fields, rows = [], []  
        try:
            print(file.split('/')[-3], file.split('/')[-1], '\n')
            with open(file, 'r') as f:
                reader = csv.reader(f)
                fields = next(reader)
                for row in reader:
                    rows.append(row)
        except:
            continue

        order_anlge = [3,4,0,1,2,5,6,8,9,11,12,13,14,7,10]
        order_anlge_l = [3,0,2,5,6,8,11,13,7,10]
        order_anlge_r = [4,1,2,5,6,9,12,14,7,10]
        order_pos = [6,7,3,4,0,1,2,5,9,10,12,13,14,15,16,17,18,19,8,11]
        order_pos_l = [6,3,0,2,5,9,12,14,16,18,8,11]
        order_pos_r = [7,4,1,2,5,10,13,15,17,19,8,11]

        if len(fields)==15:
            if (side == 'left' or side == 'l'):
                order = order_anlge_l
            elif (side == 'right' or side == 'r'):
                order = order_anlge_r              
            else:
                order = order_anlge
        
            if (side =='mean' or side == 'm'):
                values_l = np.array([float(rows[0][i]) for i in order_anlge_l])
                values_r = np.array([float(rows[0][i]) for i in order_anlge_r])
                values = np.add(values_l,values_r)/2
                values = ["%.2f" % x for x in values]
                header = [fields[i] for i in order_anlge_l]
                for i, head in enumerate(header):
                    header[i]  = head.replace('L-','')
            else:
                values = [rows[0][i] for i in order]
                header = [fields[i] for i in order]

        elif len(fields)==20:
            if (side == 'left' or side == 'l'):
                order = order_pos_l
            elif (side == 'right' or side == 'r'):
                order = order_pos_r
            else:
                order = order_pos
            
            if (side =='mean' or side == 'm'):
                values_l = np.array([float(rows[0][i]) for i in order_pos_l])
                values_r = np.array([float(rows[0][i]) for i in order_pos_r])
                values = np.add(values_l,values_r)/2
                values = ["%.2f" % x for x in values]
                header = [fields[i] for i in order_pos_l]
                for i, head in enumerate(header):
                    header[i]  = head.replace('L-','')
            else:
                header = [fields[i] for i in order] 
                values = [rows[0][i] for i in order]
        else:
            header = fields
            values = rows[0]            

        print('(' + ')(' .join(field[0] + ',' + field[1] for field in zip(header, values))+')\n')

def print_csv_to_tikz(experiments=[]):
    print('\rReading CSV files')
    files = get_experiment_files('joints_results', experiments)
    for file in files:
        fields, rows = [], []  
        try:
            print(file.split('/')[-3], file.split('/')[-1], '\n')
            with open(file, 'r') as f:
                reader = csv.reader(f)
                fields = next(reader)
                for row in reader:
                    rows.append(row)

            for field in fields:
                field = field.replace('Should.', 'Shoulder')
            print("Angle Error")
            print('(' + ')(' .join(field[0] + ',' + field[1] for field in zip(fields, rows[0]))+')')
            print("Joint Error")
            print('(' + ')(' .join(field[0] + ',' + field[1] for field in zip(fields, rows[1]))+')\n')
        except:
            continue

def plot_csv(experiments=[]):  
    files = get_experiment_files('offline_results', experiments)
    for i, file in enumerate(files):
        fields, rows = [], []  
        try:
            with open(file, 'r') as f:
                reader = csv.reader(f)
                fields = next(reader)
                for row in reader:
                    rows.append(row)
        except:
            continue
        print(file.split('/')[-3], file.split('/')[-1])
        print(fields)
        print([round(float(item),2) for item in rows[0]])
        print([round(float(item),2) for item in rows[1]], '\n')

def plot_power_draw(experiments=[]):
    files = get_experiment_files('power_draw', experiments)
    for i, file in enumerate(files):
        
        fields, rows = [], []  
        try:
            with open(file, 'r') as f:
                reader = csv.reader(f)
                fields = next(reader)
                for row in reader:
                    rows.append(row)
        except:
            continue
        print(file.split('/')[-3], file.split('/')[-1])
        print(fields)
        for row in rows:
            print(row[0], [round(float(item),2) for item in row[1:-1]], row[-1])
        print('\n')


def get_experiment_files(file_key:str, experiments=[]):
    path = os.path.join(paths['workspace_dir'], 'experiments')
    if experiments:
        files=[]
        for exp in experiments:
            try:
                files.extend(glob.glob(os.path.join(path, ('*'+exp+'*/*/*'+file_key+'.csv'))))
            except:
                print(exp, 'was not found')
    else:  
        files = glob.glob(os.path.join(path, '*/*/*'+file_key+'.csv'))
    return sorted(files)

def delete_experiment_files(file_key:str, experiments=[]):
    files = get_experiment_files(file_key, experiments)
    for file in files:
        os.remove(file)
        print(f'Deleted file: {file}')

if __name__ == '__main__':
    #print_csv_to_tikz(['REF', 'AWO', 'LKF'])
    #print_csv_to_tikz()
    #print_csv_to_tikz_bottom_up()
    #print_csv_to_tikz_bottom_up(['REF', 'spine3', 'collar', 'shirt'],side='m')
    #print_csv_to_tikz_bottom_up(['REF', 'AWO', 'LKF'],side='m')
    #print_csv_to_tikz_bottom_up(['REF'],side='r')
    #plot_csv(['REF', 'AWO', 'LKF'])
    #plot_csv(['spine3', 'collar', 'shirt'])
    plot_power_draw()
    #delete_experiment_files('power_draw')