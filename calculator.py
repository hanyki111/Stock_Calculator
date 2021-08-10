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


### 1개 파일 company_dict의 정보 중 가격정보를 연-월 별로 쪼개서 저장 (현재는 파일이 너무 무거움)
### 나중에 연-월별로 쪼개서 저장한 가격정보를 불러와서 합치는 작업도 필요


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
                    company_name = files + "." if files == "JYP Ent" else files  # JYP 빌어먹을 얘들 회사이름 JYP Ent.임
                    with open(dir + company_name, 'rb') as f:
                        data = pickle.load(f)

                    object_dict[company_name][1][data[0][0]] = data[0]

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

    # company_dict 로딩
    with open("company_pickle", "rb") as f:
        company_dict = pickle.load(f)
    '''
    # 가격 쪼개기 --> 해당 부분 함수화 하여 다른 곳에서도 사용. 채권, 코스피닥 등
    for company_name in company_dict[1].keys():
        if company_name != "종목명":
            print(company_name)
            object_dict = company_dict[1][company_name]
            path = "Stock_Price"
            split_date(object_dict, path, company_name)
            
            items = company_dict[1][company_name]
            # 각 종목 1번 (가격정보) dates
            dates = list(items[1].keys())

            yearmonth_list = []
            for date in dates:
                yearmonth = date[:7]
                if yearmonth not in yearmonth_list:
                    yearmonth_list.append(yearmonth)

            # 각 개별 가격정보의 키를 저장
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
                if not os.path.isdir("Files/Stock_Price/" + year):
                    os.makedirs("Files/Stock_Price/" + year)
                if not os.path.isdir("Files/Stock_Price/" + year + "/" + month):
                    os.makedirs("Files/Stock_Price/" + year + "/" + month)

                temp = save_dict[year+'-'+month][:]
                temp.append(items[1][date])
                save_dict[year+'-'+month] = temp

            for key in save_dict.keys():
                year = key[:4]
                month = key[5:7]
                if not os.path.isdir("Files/Stock_Price/" + year):
                    os.makedirs("Files/Stock_Price/" + year)
                if not os.path.isdir("Files/Stock_Price/" + year + "/" + month):
                    os.makedirs("Files/Stock_Price/" + year + "/" + month)

                dir_file = "Files/Stock_Price/" + year + "/" + month + "/" + company_name
                with open(dir_file, 'wb') as f:
                    pickle.dump(save_dict[key], f)
            '''

    # 가격 제외 나머지 저장
    # 회사 코드 리스트
    with open("Files/code_list", 'wb') as f:
        pickle.dump(company_dict[0], f)

    # 회사 기본정보, 연간재무, 분기재무
    company_noprice_dict = {}
    for key in company_dict[1].keys():
        company_noprice_dict[key] = [
            company_dict[1][key][0],
            {},
            company_dict[1][key][2],
            company_dict[1][key][3]
        ]
    with open("Files/company_info", "wb") as f:
        pickle.dump(company_noprice_dict, f)


    ################ N개월 자료 로딩 ##################

    # 자료 로딩

    company_dict = [None,None]
    with open("Files/code_list", 'rb') as f:
        company_dict[0] = pickle.load(f)
    with open("Files/company_info", 'rb') as f:
        company_dict[1] = pickle.load(f)

    # 개별 회사 가격 정보 로딩
    n_month = 15

    date_list = []
    date = dt.today()

    for i in range(2, n_month):
    # for i in range(n_month):
        date_list.append(str(date - relativedelta(months=i))[:7])

    for dates in date_list:
        print("{} 가격 정보 로딩 중 ...".format(dates))
        year = dates[:4]
        month = dates[5:7]
        dir = "Files/Stock_Price/" + year + "/" + month + "/"
        file_list = os.listdir(dir)
        for files in file_list:
            company_name = files + "." if files == "JYP Ent" else files  # JYP 빌어먹을 얘들 회사이름 JYP Ent.임
            with open(dir + files, 'rb') as f:
                data = pickle.load(f)
            company_dict[1][company_name][1][data[0][0]] = data[0]


    # pickle 저장

    with open("company_pickle", "wb") as f:
        pickle.dump(company_dict, f)