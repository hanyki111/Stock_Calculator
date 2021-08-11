import pickle
import numpy as np

from datetime import datetime as dt
from datetime import timedelta as td
from dateutil.relativedelta import relativedelta

import os


### 긁어온 정보를 계산한다



### ai_recommand --> list, score 계산
def ai_recommand_calculate(self):




    pass


def split_date(object_dict, path, name, overwrite=False):
    # bond_dict의 경우 하위 dict가 3m, 6m ...이므로 각 3m, 6m...을 object_dict로 넣어야 함

    dates = list(object_dict.keys())

    yearmonth_list = []
    for date in dates:
        yearmonth = date[:7]
        if yearmonth not in yearmonth_list:
            yearmonth_list.append(yearmonth)

    yearmonth_dict = dict.fromkeys(yearmonth_list, [])
    for date in dates:
        templist = yearmonth_dict[date[:7]][:]
        templist.append(date)
        yearmonth_dict[date[:7]] = templist

    save_dict = dict.fromkeys(yearmonth_list, [])
    ### 개별 가격정보의 키를 이용하여 월별 데이터 저장
    for date in dates:
        year = date[:4]
        month = date[5:7]
        if not os.path.isdir("Files/" + path + "/" + year):
            os.makedirs("Files/" + path + "/" + year)
        if not os.path.isdir("Files/" + path + "/" + year + "/" + month):
            os.makedirs("Files/" + path + "/" + year + "/" + month)
        '''
        if np.ndim(object_dict[date]) == 1:
            temp = object_dict[date]
        else:
        '''
        temp = save_dict[year+'-'+month][:]
        temp.append(object_dict[date])
        save_dict[year+'-'+month] = temp

    for i in range(len(save_dict.keys())):

        key = list(save_dict.keys())[i]
        year = key[:4]
        month = key[5:7]
        if not os.path.isdir("Files/" + path + "/" + year):
            os.makedirs("Files/" + path + "/" + year)
        if not os.path.isdir("Files/" + path + "/" + year + "/" + month):
            os.makedirs("Files/" + path + "/" + year + "/" + month)

        dir_file = "Files/" + path + "/" + year + "/" + month + "/" + name

        ### 파일이 이미 존재하느냐
        if not overwrite:  # overwrite == False --> 덮어쓰지 않을 때
            if not os.path.isfile(dir_file):
                with open(dir_file, 'wb') as f:
                    pickle.dump(save_dict[key], f)
            elif i <= 1:
                with open(dir_file, 'wb') as f:
                    pickle.dump(save_dict[key], f)
            else:  # 덮어쓰기 --> False 상태, 파일 있고 i >=2 --> 반복문 제외
                break
        else:  # overwrite == True --> 덮어쓸 때
            with open(dir_file, 'wb') as f:
                pickle.dump(save_dict[key], f)

    pass  # split_date 의 끝


def months_loading(object_dict, path, object_filename, n_month=15, company_tf=False):

    date_list = []
    date = dt.today()

    for i in range(n_month):
        # for i in range(n_month):
        date_list.append(str(date - relativedelta(months=i))[:7])

    for dates in date_list:
        print("{} 가격 정보 로딩 중 ...".format(dates))
        year = dates[:4]
        month = dates[5:7]
        dir = "Files/" + path + "/" + year + "/" + month + "/"

        if company_tf:  # 회사 정보를 로딩하는 경우
            try:
                file_list = os.listdir(dir)
                for files in file_list:
                    name = files + "." if files == "JYP Ent" else files  # JYP 빌어먹을 얘들 회사이름 JYP Ent.임

                    with open(dir + name, 'rb') as f:
                        data = pickle.load(f)
                    for each_data in data:
                        object_dict[name][1][each_data[0]] = each_data

            except FileNotFoundError:
                print("{} - {} 항목 없음, 건너뜀".format(year, month))

        else:
            try:
                file = dir + object_filename
                with open(file, 'rb') as f:
                    data = pickle.load(f)
                if np.ndim(data) > 1:
                    for each_data in data:
                        object_dict[each_data[0]] = each_data
                else:  # kosis_dict의 경우
                    object_dict[data[0]] = data
            except FileNotFoundError:
                print("{} - {} 항목 없음, 건너뜀".format(year, month))
    # return object_dict
    pass


if __name__=="__main__":
    # company 로딩
    company_pickle = [None,None]
    with open("Files/code_list", 'rb') as f:
        company_pickle[0] = pickle.load(f)
    with open("Files/company_info", 'rb') as f:
        company_pickle[1] = pickle.load(f)

    months_loading(company_pickle[1], "Stock_Price", "", company_tf=True)

    # KOSPI, KOSDAQ, bond, vkospi, kosis 로딩
    try:
        KOSPI_dict = {}
        months_loading(KOSPI_dict, "Market", "KOSPI")
    except FileNotFoundError:
        KOSPI_dict = {}
    try:
        KOSDAQ_dict = {}
        months_loading(KOSDAQ_dict, "Market", "KOSDAQ")
    except FileNotFoundError:
        KOSDAQ_dict = {}
    try:
        bond_dict = {'3m' : {}, '6m' : {}, '9m' : {}, '1y' : {}, '1y6m' : {}, '2y' : {}, '3y' : {}, '5y' : {}}
        for key in bond_dict.keys():
            months_loading(bond_dict[key], "Market", "bond_"+key)
    except FileNotFoundError:
        bond_dict = {'3m' : {}, '6m' : {}, '9m' : {}, '1y' : {}, '1y6m' : {}, '2y' : {}, '3y' : {}, '5y' : {}}
    try:
        vkospi_dict = {}
        months_loading(vkospi_dict, "Market", "vkospi")
    except FileNotFoundError:
        vkospi_dict = {}
    try:
        kosis_dict = {}
        months_loading(kosis_dict, "Market", "kosis")
    except FileNotFoundError:
        kosis_dict = {}
