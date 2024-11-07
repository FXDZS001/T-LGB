import numpy as np
import pandas as pd
import os
from util import *
from scipy.interpolate import interp1d

# 创建了以日期文件夹下00和12的文件夹
# ├── 20180101
# │   ├── 00
# │   └── 12
# ├── 20180102
# │   ├── 00
# │   └── 12
# ├── ...
# └── 20210331
#     ├── 00
#     └── 12
def DirectoryMake():

    '''
    创建日期目录文件
    :return:
    '''

    start_date = '2018-01-01'
    end_date = '2021-03-31'
    start_date = time.strptime(start_date, '%Y-%m-%d') # 使用 time.strptime() 将字符串日期转换为 time.struct_time 对象。
    end_date = time.strptime(end_date, '%Y-%m-%d')

    all_dates = pd.date_range(datetime.datetime(start_date.tm_year, start_date.tm_mon, start_date.tm_mday),
                              datetime.datetime(end_date.tm_year, end_date.tm_mon, end_date.tm_mday), freq='D')# 生成一个包含一系列日期的 pandas.DatetimeIndex 对象
    all_dates = np.array(all_dates.values)

    root_path = r'D:\atmosphere\code\haiwu\ec\dayby12hous'

    for dates in all_dates:
        path = root_path + GenerateDates(dates) + '00'
        if not os.path.exists(path):
            os.mkdir(path)
        path = root_path + GenerateDates(dates) + '12'
        if not os.path.exists(path):
            os.mkdir(path)
        print(GenerateDates(dates))
# D:\atmosphere\code\MeiLan\class
# ├── sc_sample1
# ├── sc_sample2
# ├── sc_sample3
# ├── ...
# └── sc_sample21
def listMake():
    root_path = r'D:\atmosphere\code\MeiLan\class'
    for i in range(1, 22):

        path = root_path + '\\sc_sample' + str(i)
        if not os.path.exists(path):
            os.mkdir(path)
        print(path)
#  对ec数据的提取，但这个是低空的，取点计算。高空的取面，10*10
def readec():
    start_time = '2019-04-25'
    end_time = '2019-05-31'
    root_path = r'D:\ecthin_local'
    dates = pd.date_range(start_time, end_time, freq='D')
    dates = np.array(dates.values)
    # elements = [['SST'], ['T2m'] ,['Ws2m'], ['D2m'] ,['U10m'] ,['V10m'],['U100m'], ['V100'] ,['SLP'] , ['VIS'],
    #             ['Surf.P'], ['TotalPrec.'],
    #             ['GH_500hPa'],['GH_700hPa'],['GH_850hPa'],['GH_925hPa'], ['GH_1000hPa'],
    #             ['U_500hPa'],['U_700hPa'],['U_850hPa'],['U_925hPa'],['U_1000hPa'],
    #             ['V_500hPa'], ['V_700hPa'], ['V_850hPa'],['V_925hPa'],['V_1000hPa'],
    #             ['Temp._500hPa'], ['Temp._700hPa'],['Temp._850hPa'], ['Temp._925hPa'], ['Temp._1000hPa'],
    #             ['RH_500hPa'],['RH_700hPa'],['RH_850hPa'],['RH_925hPa'],['RH_1000hPa'],
    #             ['Q_500hPa'],['Q_700hPa'],['Q_850hPa'], ['Q_925hPa'],['Q_1000hPa'],
    #             ['VV_500hPa'],['VV_700hPa'],['VV_850hPa'],['VV_925hP'], ['VV_1000hPa']]

    elements = ['SST', 'T2m', 'D2m', 'U10m', 'V10m', 'U100m', 'V100',
                'SLP', 'VIS', 'Surf.P', 'TotalPrec.',
                'GH_500hPa', 'GH_700hPa', 'GH_850hPa', 'GH_925hPa', 'GH_1000hPa',
                'U_500hPa', 'U_700hPa', 'U_850hPa', 'U_925hPa', 'U_1000hPa',
                'V_500hPa', 'V_700hPa', 'V_850hPa', 'V_925hPa', 'V_1000hPa',
                'Temp._500hPa', 'Temp._700hPa','Temp._850hPa', 'Temp._925hPa', 'Temp._1000hPa',
                'RH_500hPa','RH_700hPa','RH_850hPa','RH_925hPa','RH_1000hPa',
                'Q_500hPa','Q_700hPa','Q_850hPa', 'Q_925hPa','Q_1000hPa',
                'VV_500hPa','VV_700hPa','VV_850hPa','VV_925hPa', 'VV_1000hPa']

    ele1 = ['SST', 'T2m', 'D2m', 'U10m', 'V10m', 'U100m', 'V100', 'SLP', 'VIS','Surf.P', 'TotalPrec.']
    ele2 = ['GH_500hPa', 'GH_700hPa', 'GH_850hPa', 'GH_925hPa', 'GH_1000hPa',
                'U_500hPa', 'U_700hPa', 'U_850hPa', 'U_925hPa', 'U_1000hPa',
                'V_500hPa', 'V_700hPa', 'V_850hPa', 'V_925hPa', 'V_1000hPa',
                'Temp._500hPa', 'Temp._700hPa','Temp._850hPa', 'Temp._925hPa', 'Temp._1000hPa',
                'RH_500hPa','RH_700hPa','RH_850hPa','RH_925hPa','RH_1000hPa',
                'Q_500hPa','Q_700hPa','Q_850hPa', 'Q_925hPa','Q_1000hPa',
                'VV_500hPa','VV_700hPa','VV_850hPa','VV_925hPa', 'VV_1000hPa']

    hour = list(range(70*3))[::3]  # 创建3day,每隔3h的小时名
    hours = []
    for i in range(70):
        hours.append(str(hour[i]).zfill(3))  # 在数据前补充0

    for i in dates:
        date = GenerateDates(i)
        print(date)
        for j in hours[:25]:
            ec3_00 = ec3_12 = np.zeros([3, 46])
            for k in range(len(elements)):
                # EC_00
                file_path = os.path.join(root_path, date + '00', elements[k] +'_fcst_' + j + '.npy')
                try:
                    ch = np.load(file_path)
                except:
                    print('missing date: {}_00_{}'.format(date, j))
                    with open('./ec/missing.txt', "a") as file:
                        file.write(date + '00_' + j + "\n")
                    file.close()
                    break

                if elements[k] in ele1:
                    ec3_00[:, k] = [ch[41][39], ch[41][39], ch[42][38]] # 通过分辨率和经纬度计算站点的数据位置，并提取出来
                elif elements[k] in ele2:
                    ec3_00[:, k] = [ch[20][19], ch[20][19], ch[20][19]]

                # EC_12
                file_path = os.path.join(root_path, date + '12', elements[k] +'_fcst_' + j + '.npy')
                try:
                    ch = np.load(file_path)
                except:
                    print('missing date: {}_12_{}'.format(date, j))
                    with open('./ec/missing.txt', "a") as file:
                        file.write(date + '12_' + j + "\n")
                    file.close()
                    break

                if elements[k] in ele1:
                    ec3_12[:, k] = [ch[41][39], ch[41][39], ch[42][38]]
                elif elements[k] in ele2:
                    ec3_12[:, k] = [ch[20][19], ch[20][19], ch[20][19]]

            np.save(os.path.join('./ec/dayby12hours', date + '00', date + j), ec3_00)
            np.save(os.path.join('./ec/dayby12hours', date + '12', date + j), ec3_12)

# 数据进行缺失数据的填充处理。具体来说，它会针对一个给定的日期和时次，尝试读取数据文件。
# 如果文件不存在，则利用前一个和后一个时次的数据进行平均，以填充缺失数据。H控制取00还是12进行填充
def MissECFill(miss_date, H):
    '''
    EC数据缺失，单个填充
    :param miss_date: 缺失日期
    :param H: 00/12
    :return:
    '''

    start_time = miss_date
    end_time = miss_date
    root_path = r'D:\ecthin_local'
    dates = pd.date_range(start_time, end_time, freq='D')
    dates = np.array(dates.values)
    # elements = [['SST'], ['T2m'] ,['Ws2m'], ['D2m'] ,['U10m'] ,['V10m'],['U100m'], ['V100'] ,['SLP'] , ['VIS'],
    #             ['Surf.P'], ['TotalPrec.'],
    #             ['GH_500hPa'],['GH_700hPa'],['GH_850hPa'],['GH_925hPa'], ['GH_1000hPa'],
    #             ['U_500hPa'],['U_700hPa'],['U_850hPa'],['U_925hPa'],['U_1000hPa'],
    #             ['V_500hPa'], ['V_700hPa'], ['V_850hPa'],['V_925hPa'],['V_1000hPa'],
    #             ['Temp._500hPa'], ['Temp._700hPa'],['Temp._850hPa'], ['Temp._925hPa'], ['Temp._1000hPa'],
    #             ['RH_500hPa'],['RH_700hPa'],['RH_850hPa'],['RH_925hPa'],['RH_1000hPa'],
    #             ['Q_500hPa'],['Q_700hPa'],['Q_850hPa'], ['Q_925hPa'],['Q_1000hPa'],
    #             ['VV_500hPa'],['VV_700hPa'],['VV_850hPa'],['VV_925hP'], ['VV_1000hPa']]

    elements = ['SST', 'T2m', 'D2m', 'U10m', 'V10m', 'U100m', 'V100',
                'SLP', 'VIS', 'Surf.P', 'TotalPrec.',
                'GH_500hPa', 'GH_700hPa', 'GH_850hPa', 'GH_925hPa', 'GH_1000hPa',
                'U_500hPa', 'U_700hPa', 'U_850hPa', 'U_925hPa', 'U_1000hPa',
                'V_500hPa', 'V_700hPa', 'V_850hPa', 'V_925hPa', 'V_1000hPa',
                'Temp._500hPa', 'Temp._700hPa','Temp._850hPa', 'Temp._925hPa', 'Temp._1000hPa',
                'RH_500hPa','RH_700hPa','RH_850hPa','RH_925hPa','RH_1000hPa',
                'Q_500hPa','Q_700hPa','Q_850hPa', 'Q_925hPa','Q_1000hPa',
                'VV_500hPa','VV_700hPa','VV_850hPa','VV_925hPa', 'VV_1000hPa']

    ele1 = ['SST', 'T2m', 'D2m', 'U10m', 'V10m', 'U100m', 'V100', 'SLP', 'VIS','Surf.P', 'TotalPrec.']
    ele2 = ['GH_500hPa', 'GH_700hPa', 'GH_850hPa', 'GH_925hPa', 'GH_1000hPa',
                'U_500hPa', 'U_700hPa', 'U_850hPa', 'U_925hPa', 'U_1000hPa',
                'V_500hPa', 'V_700hPa', 'V_850hPa', 'V_925hPa', 'V_1000hPa',
                'Temp._500hPa', 'Temp._700hPa','Temp._850hPa', 'Temp._925hPa', 'Temp._1000hPa',
                'RH_500hPa','RH_700hPa','RH_850hPa','RH_925hPa','RH_1000hPa',
                'Q_500hPa','Q_700hPa','Q_850hPa', 'Q_925hPa','Q_1000hPa',
                'VV_500hPa','VV_700hPa','VV_850hPa','VV_925hPa', 'VV_1000hPa']

    hour = list(range(70*3))[::3]  # 创建3day,每隔3h的小时名
    hours = []
    for i in range(70):
        hours.append(str(hour[i]).zfill(3))  # 在数据前补充0

    for i in dates:
        date = GenerateDates(i)
        print(date)
        for j in range(25):
            ec3_00 = ec3_12 = np.zeros([3, 46])
            for k in range(len(elements)):

                # EC_00
                if H == '00':
                    file_path = os.path.join(root_path, date + '00', elements[k] +'_fcst_' + hours[j] + '.npy')
                    try:
                        ch = np.load(file_path)
                    except:
                        ch0 = np.load(os.path.join(root_path, date + '00', elements[k] + '_fcst_' + hours[j - 1] + '.npy'))
                        ch1 = np.load(os.path.join(root_path, date + '00', elements[k] + '_fcst_' + hours[j + 1] + '.npy'))
                        ch = (ch0 + ch1) / 2

                    if elements[k] in ele1:
                        ec3_00[:, k] = [ch[41][39], ch[41][39], ch[42][38]]
                    elif elements[k] in ele2:
                        ec3_00[:, k] = [ch[20][19], ch[20][19], ch[20][19]]

                # EC_12
                if H == '12':
                    file_path = os.path.join(root_path, date + '12', elements[k] +'_fcst_' + hours[j] + '.npy')
                    try:
                        ch = np.load(file_path)
                    except:
                        ch0 = np.load(os.path.join(root_path, date + '12', elements[k] + '_fcst_' + hours[j - 1] + '.npy'))
                        ch1 = np.load(os.path.join(root_path, date + '12', elements[k] + '_fcst_' + hours[j + 1] + '.npy'))
                        ch = (ch0 + ch1) / 2
                        # ch0 = np.load(os.path.join(root_path, date + '12', elements[k] + '_fcst_' + hours[j - 2] + '.npy'))
                        # ch1 = np.load(os.path.join(root_path, date + '12', elements[k] + '_fcst_' + hours[j - 1] + '.npy'))
                        # ch = ch1 * 2 - ch0

                    if elements[k] in ele1:
                        ec3_12[:, k] = [ch[41][39], ch[41][39], ch[42][38]]
                    elif elements[k] in ele2:
                        ec3_12[:, k] = [ch[20][19], ch[20][19], ch[20][19]]

            if H == '00':
                np.save(os.path.join('./ec/dayby12hours', date + '00', date + hours[j]), ec3_00)
            if H == '12':
                np.save(os.path.join('./ec/dayby12hours', date + '12', date + hours[j]), ec3_12)


def genarateDay(date):
    pass



# 填充缺失值
def FillSc(rs, T2):
    rs_time = rs[:, 0]
    for i in range(len(rs) - 1):
        time_now = time.mktime(rs_time[i].timetuple())  # 获取当前时刻时间戳
        time_next = time.mktime(rs_time[i + 1].timetuple())  # 获取下一时刻时间戳
        if time_next - time_now == 7200:  # 一小时时间戳差为3600
            print('Missing 1 Hour : ' + str(rs_time[i]))
            v = []
            v.append(datetime.datetime.fromtimestamp(time_now + 3600))
            v[1: len(rs[0])] = (rs[i, 1:len(rs[0])] + rs[i + 1, 1:len(rs[0])]) / 2
            rs = np.insert(rs, i + 1, values=v, axis=0)
            #break
        elif time_next - time_now == 10800:
            print('Missing 2 Hour : ' + str(rs_time[i]))
            for j in range(2):
                v = []
                v.append(datetime.datetime.fromtimestamp(time_now + 3600 * (j + 1)))
                v[1: len(rs[0])] = rs[i, 1:len(rs[0])] + (j + 1) * (rs[i + 1 + j, 1:len(rs[0])] - rs[i, 1:len(rs[0])]) / 3
                rs = np.insert(rs,i + j + 1,values=v,axis=0)
            #break
            # v1 = []
            # v2 = []
            # v1.append(datetime.fromtimestamp(time_now + 3600))
            # v1[1: len(rs[0])] = (rs[i, 1:len(rs[0])] + rs[i + 1, 1:len(rs[0])]) / 3
            # v2.append(datetime.fromtimestamp(time_now + 7200))
            # v2[1: len(rs[0])] = 2 * (rs[i, 1:len(rs[0])] + rs[i + 1, 1:len(rs[0])]) / 3
            # v = []
            # v.append(v1)
            # v.append(v2)
            # rs = np.insert(rs, i + 1, values=v, axis=0)
        elif time_next - time_now == 14400:
            print('Missing 3 Hour : ' + str(rs_time[i]))
            for j in range(3):
                v = []
                v.append(datetime.datetime.fromtimestamp(time_now + 3600 * (j + 1)))
                v[1: len(rs[0])] = rs[i, 1:len(rs[0])] + (j + 1) * (rs[i + 1 + j, 1:len(rs[0])] - rs[i, 1:len(rs[0])]) / 4
                rs = np.insert(rs, i + j + 1, values=v, axis=0)
            #break
            # v1 = []
            # v2 = []
            # v3 = []
            # v1.append(datetime.fromtimestamp(time_now + 3600))
            # v1[1: len(rs[0])] = (rs[i, 1:len(rs[0])] + rs[i + 1, 1:len(rs[0])]) / 4
            # v2.append(datetime.fromtimestamp(time_now + 7200))
            # v2[1: len(rs[0])] = 2 * (rs[i, 1:len(rs[0])] + rs[i + 1, 1:len(rs[0])]) / 4
            # v3.append(datetime.fromtimestamp(time_now + 10800))
            # v3[1: len(rs[0])] = 3 * (rs[i, 1:len(rs[0])] + rs[i + 1, 1:len(rs[0])]) / 4
            # v = []
            # v.append(v1)
            # v.append(v2)
            # v.append(v3)
            # rs = np.insert(rs, i + 1, values=v, axis=0)
        elif time_next - time_now >= 18000:
            n = (time_next - time_now) / 3600
            print('Missing ' + str(int(n)) + ' Hour : ' + str(rs_time[i]))
            for j in range(int(n - 1)):
                v = []
                v.append(datetime.datetime.fromtimestamp(time_now + 3600 * (j + 1)))
                v[1: len(rs[0])] = rs[i - (24 - j), 1:len(rs[0])]
                rs = np.insert(rs, i + j + 1, values=v, axis=0)
            #break
        elif i == len(rs) - 2:  # 检查是否已经到达了最后一行的前一行。
            T2 = True
    return rs, T2

# 将原始数据切割成多个样本，并为每个样本生成对应的标签
def sampleMake(start_time, end_time,year):

    dates = pd.date_range(start_time, end_time, freq='D')

    dates = np.array(dates.values)

    # root_path = r'D:\atmosphere\code\haiwu\ec\dayby12hours'
    save_path = r'D:\hainan\回归'
    rs = np.load(fr"D:\hainan\填补缺失值后的数据\{year}定安.npy", allow_pickle=True)
    sc_datas = rs[6:, :]

    hour = list(range(60 * 3))[::3]  # 创建3day,每隔3h的小时名
    hours = []
    for i in range(60):
        hours.append(str(hour[i]).zfill(3))  # 在数据前补充0

    sc_datas = np.array(sc_datas)
    for i in range(len(dates) - 3):  # 最后三天用来构建label
        print(dates[i])
        filename = GenerateDates(dates[i])
        # print(filename + ' processing')
        for j in range(4):
            sc00 = sc_datas[i * 24 + j * 3: i * 24 + j * 3 + 3, 1:]
            sc00 = sc00.flatten()
            sc12 = sc_datas[i * 24 + (j + 4) * 3: i * 24 + (j + 4) * 3 + 3, 1:]
            sc12 = sc12.flatten()
            # sc00 = sc00.reshape(sc00.shape[0], sc00.shape[1] * sc00.shape[2])
            # sc12 = sc12.reshape(sc12.shape[0], sc12.shape[1] * sc12.shape[2])
            # sc00 = sc00.reshape(1, -1)
            # sc12 = sc12.reshape(1, -1)

            for k in range(21):
                label00 = sc_datas[i * 24 + j * 3 + (k + 2) * 3 - 1, -1]

                label12 = sc_datas[i * 24 + (j + 4) * 3 + (k + 2) * 3 - 1, -1]
                np.save(os.path.join(save_path, 'sc', 'sample' + str(k + 1), filename + hours[j + 1][1:]), sc00)# 这样岂不是sample1-21都是相同的？用一个就行？是一样的但label不一样
                np.save(os.path.join(save_path, 'sc', 'sample' + str(k + 1), filename + hours[j + 5][1:]), sc12)
                np.save(os.path.join(save_path, 'label', 'sample' + str(k + 1), filename + hours[j+1][1:]), label00)
                np.save(os.path.join(save_path, 'label', 'sample' + str(k + 1), filename + hours[j+5][1:]), label12)
                pass

# 创建21个文件夹
def listMake():
    root_path = r'D:\hainan\回归\label'
    for i in range(1,22):

        path = root_path + '\\sample' + str(i)
        if not os.path.exists(path):
            os.mkdir(path)
        print(path)

# 合并成一个大型的 numpy 数组
def SampleCollect(start_time, end_time,n):
    '''
    collect single sample to make the entire samples
    :param start_time:
    :param end_time:
    :return:
    '''
    dates = pd.date_range(start_time, end_time, freq='D')
    dates = np.array(dates.values)
    root_path = r'D:\hainan\回归\label'

    hour = list(range(60*3))[::3]  # 创建3day,每隔3h的小时名
    hours = []
    for i in range(60):
        hours.append(str(hour[i]).zfill(3))  # 在数据前补充0

    for i in range(21):
        print('samples' + str(i + 1) + ' processing')
        samples = []
        for date in dates:
            for j in range(8):
                filename = GenerateDates(date) + hours[j+1][1:]
                ch0 = np.load(os.path.join(root_path, 'sample' + str(i+1), filename + '.npy'), allow_pickle=True)
                samples.append(ch0)

        np.save(os.path.join(root_path, 'sample', f'samples{n}' + str(i+1) + '.npy'), np.array(samples))
        # simple11和simple12.....不是一样的？对的，但后续合并是simple11和simple12....

# 依次加载三个年份（编号为 1、2、3）的样本数据，并将其拼接成一个新的数组，然后保存到指定的路径。
def SampleCollect1():
    '''
    三年数据合成
    :return:
    '''
    root_path = r'D:\hainan\回归\label\sample'
    path = 'D:\hainan\回归\label\sample\定安'
    for j in range(1, 22):
        rs1 = np.load(os.path.join(root_path, 'samples' + str(1) + str(j) + '.npy'), allow_pickle=True)
        rs2 = np.load(os.path.join(root_path, 'samples' + str(2) + str(j) + '.npy'), allow_pickle=True)
        rs3 = np.load(os.path.join(root_path, 'samples' + str(3) + str(j) + '.npy'), allow_pickle=True)
        rs4 = np.load(os.path.join(root_path, 'samples' + str(4) + str(j) + '.npy'), allow_pickle=True)
        samples = np.concatenate((rs1, rs2, rs3, rs4), axis=0)
        print(j)
        np.save(os.path.join(path, 'samples' + str(j) + '.npy'), np.array(samples))
        #print(samples)

# 对样本进行分类
def classification():
    '''
    对样本进行分类
    :return:
    '''
    root_path = r'D:\atmosphere\code\haiwu\samples\59754'
    for j in range(1, 22):
        sort1 = []
        sort2 = []
        sort3 = []
        sort4 = []
        print(j)
        rs = np.load(os.path.join(root_path, 'samples' + str(j) + '.npy'), allow_pickle=True)
        for k in range(len(rs)):
            VIS = rs[k, :]
            if VIS[-1] == 1:
                sort1.append(VIS)
            elif VIS[-1] == 2:
                sort2.append(VIS)
            elif VIS[-1] == 3:
                sort3.append(VIS)
            elif VIS[-1] == 4:
                sort4.append(VIS)

        np.save(os.path.join(root_path, 'class1', 'samples' + str(j) + '.npy'), sort1)

        np.save(os.path.join(root_path, 'class2', 'samples' + str(j) + '.npy'), sort2)

        np.save(os.path.join(root_path, 'class3', 'samples' + str(j) + '.npy'), sort3)

        np.save(os.path.join(root_path, 'class4', 'samples' + str(j) + '.npy'), sort4)
            # np.save(os.path.join(root_path, 'class' + str(i),  'samples' + str(j) + '.npy'), sort1)
            # np.save(os.path.join(root_path, 'class' + str(i),  'samples' + str(j) + '.npy'), sort2)
            # np.save(os.path.join(root_path, 'class' + str(i),  'samples' + str(j) + '.npy'), sort3)
            # np.save(os.path.join(root_path, 'class' + str(i),  'samples' + str(j) + '.npy'), sort4)



# 过滤1000m以上的
def filter():
    path = r'D:\hainan\回归'
    for i in range(1, 22):
        filtered_samples = []
        filtered_label = []
        rs2 = np.load(os.path.join(path, '三者合一', 'samples' + str(i) + '.npy'), allow_pickle=True)
        rs3 = np.load(os.path.join(path, 'label', 'sample', '5站总', 'samples' + str(i) + '.npy'), allow_pickle=True)

        # 遍历数据并判断
        for j in range(len(rs3)):
            if rs3[j] <= 1000:
                filtered_label.append(rs3[j])
                filtered_samples.append(rs2[j,:])
                # filtered_data = np.array(filtered_data)
        print(i)
        # 保存过滤后的数据
        np.save(os.path.join(path, '回归', 'samples', 'samples' + str(i) + '.npy'), np.array(filtered_samples))
        np.save(os.path.join(path, '回归', 'label', 'samples' + str(i) + '.npy'), np.array(filtered_label))

if __name__ == '__main__':
    # readec()
    # Readsc1()
    # DirectoryMake()
    # MissECFill('2020-05-20', '12')
    # rs = np.load(r"E:\label\白沙\低\label\baisa_label1.npy", allow_pickle=True)
    # print(rs)
    # print(rs)
    # listMake()
    # rs = pd.read_excel(r'C:\Users\admin\Desktop\美兰\2021\2021白沙.xlsx', sheet_name='Sheet1')

    # l = ['白沙','澄迈','定安','三亚']
    # for i in l:
    #     rs = np.load(fr'D:\hainan\实测原始数据\美兰\2022\毕业\2022{i}.npy', allow_pickle=True)
    #     T2 = True
    #     rs, T2 = FillSc(rs, T2)
    #     print(rs.shape)
    #     np.save(fr"D:\hainan\填补缺失值后的数据\2022{i}.npy", rs)

    # ec = np.load(r"D:\hainan\ViT_label\总站\低\label\label1.npy", allow_pickle=True)
    # print(ec)

    # '20210101': 20220531
    # sampleMake('20210101', '20210531', 2022)
    # SampleCollect('20210101', '20210528', 5)

    #SampleCollect1()

    # import itertools
    # l = {'20180101':20180512,   '20181201':20190423,    '20191201':20200531,    '20201201':20201231 }
    # p = [2018,2019,2020,2021,2022]
    # years_cycle = itertools.cycle(p)
    # sample_number_cycle = itertools.cycle(range(1, 5))
    # for j , k in l.items():
    #     year = next(years_cycle)
    #     sample_number = next(sample_number_cycle)
    #     print(j, k, year, sample_number)
    #     sampleMake(j, str(k), year)
    #     SampleCollect(j, str(k-3),sample_number)
    # SampleCollect1()

    # label1 = np.load(r"D:\MeiLan\class\ EC_ViT1.npy", allow_pickle=True)
    # label2 = np.load(r"D:\MeiLan\class\ EC_ViT1.npy", allow_pickle=True)
    # # ec2 = ec1.copy()
    # label = np.concatenate((label1, label2), axis=0)
    # print(label.shape)
    # np.save(r"D:\MeiLan\class\ec1", label)
    #classification()
    # ec2 = np.load(r"D:\atmosphere\code\haiwu\ec\dayby12hours\2019123012\20191230072.npy")
    # ec = (ec1 + ec2)/2
    # print(ec2)

    # np.save(r"D:\atmosphere\code\haiwu\ec\dayby12hours\2019122912\20191229072", ec)
    # ec = np.load(r"D:\MeiLan\ViT_data\TiV_data\TiVec_samples1.npy", allow_pickle=True)
    # print(ec.shape)
