# import ast
import asyncssh
import csv
import json
import numpy as np
import pandas as pd
import paramiko
import pickle
import random
import time

import datetime
from datetime import datetime as dt
from datetime import timedelta as td
from dateutil.relativedelta import relativedelta

from tensorflow import keras
from scipy import stats

import os


class loading:

    def split_date(self, object_dict, path, name, overwrite=False):
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


    def months_loading(self, object_dict, path, object_filename, n_month=15, company_tf=False):

        date_list = []
        date = dt.today()

        for i in range(n_month):
            # for i in range(n_month):
            date_list.append(str(date - relativedelta(months=i))[:7])

        for dates in date_list:
            # print("{} 가격 정보 로딩 중 ...".format(dates))
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
                    if type(data[0]) == list or type(data[0]) == np.ndarray:
                        for each_data in data:
                            object_dict[each_data[0]] = each_data
                    else:  # kosis_dict의 경우
                        object_dict[data[0]] = data
                except FileNotFoundError:
                    print("{} - {} 항목 없음, 건너뜀".format(year, month))
        # return object_dict
        pass  # months_loading의 끝


    def loading_data(self, n_month=15):
        # company 로딩
        company_pickle = [None,None]
        with open("Files/code_list", 'rb') as f:
            company_pickle[0] = pickle.load(f)
        with open("Files/company_info", 'rb') as f:
            company_pickle[1] = pickle.load(f)

        self.months_loading(company_pickle[1], "Stock_Price", "", n_month=n_month, company_tf=True)

        # KOSPI, KOSDAQ, bond, vkospi, kosis 로딩
        try:
            KOSPI_dict = {}
            self.months_loading(KOSPI_dict, "Market", "KOSPI", n_month=n_month)
        except FileNotFoundError:
            KOSPI_dict = {}
        try:
            KOSDAQ_dict = {}
            self.months_loading(KOSDAQ_dict, "Market", "KOSDAQ", n_month=n_month)
        except FileNotFoundError:
            KOSDAQ_dict = {}
        try:
            bond_dict = {'3m' : {}, '6m' : {}, '9m' : {}, '1y' : {}, '1y6m' : {}, '2y' : {}, '3y' : {}, '5y' : {}}
            for key in bond_dict.keys():
                self.months_loading(bond_dict[key], "Market", "bond_"+key, n_month=n_month)
        except FileNotFoundError:
            bond_dict = {'3m' : {}, '6m' : {}, '9m' : {}, '1y' : {}, '1y6m' : {}, '2y' : {}, '3y' : {}, '5y' : {}}
        try:
            vkospi_dict = {}
            self.months_loading(vkospi_dict, "Market", "vkospi", n_month=n_month)
        except FileNotFoundError:
            vkospi_dict = {}
        try:
            kosis_dict = {}
            self.months_loading(kosis_dict, "Market", "kosis", n_month=n_month)
        except FileNotFoundError:
            kosis_dict = {}

        return company_pickle, KOSPI_dict, KOSDAQ_dict, bond_dict, vkospi_dict, kosis_dict

### 긁어온 정보를 계산한다

class FGI:

    def cal_cumulative_dist(self, x, mean, std):
        norm_dist = stats.norm(loc = mean, scale = std)
        return (norm_dist.cdf(x=x))

    def cal_cumulative_dist_scorize(self, x, mean, std):
        norm_dist = stats.norm(loc = mean, scale = std)
        if x > mean:
            return (norm_dist.cdf(x=x) - norm_dist.cdf(x=mean)) * 200
        else:
            return (norm_dist.cdf(x=mean) - norm_dist.cdf(x=x)) * -200

    def ratio_52_highlow(self, KOSPI_dict, KOSDAQ_dict):  # + 탐욕, - 공포
        # KOSPI_dict, KOSDAQ_dict를 입력
        # ratio_data_list 가 나온다
        # ratio_data_list = [날짜, high, low, high-low, 분포점수]

        ratio_data_list = []

        for dates in KOSPI_dict.keys():
            date = dates
            # print(dates)
            # print(KOSPI_dict[dates])
            # print(KOSDAQ_dict[dates])
            # kospi_high = int(KOSPI_dict[dates][4])
            # kospi_low = int(KOSPI_dict[dates][5])
            # kosdaq_high = int(KOSDAQ_dict[dates][4])
            # kosdaq_low = int(KOSDAQ_dict[dates][5])


            try:
                kospi_high = int(KOSPI_dict[dates][4])
            except Exception as e:
                print(dates, e)
                kospi_high = 0
            try:
                kospi_low = int(KOSPI_dict[dates][5])
            except Exception as e:
                print(dates, e)
                kospi_low = 0
            try:
                kosdaq_high = int(KOSDAQ_dict[dates][4])
            except Exception as e:
                print(dates, e)
                kosdaq_high = 0
            try:
                kosdaq_low = int(KOSDAQ_dict[dates][5])
            except Exception as e:
                print(dates, e)
                kosdaq_low = 0

            high = kospi_high + kosdaq_high
            low = kospi_low + kosdaq_low

            ratio_data_list.append([date, high, low, high-low])

        ratio_data_index3_np = np.array(ratio_data_list)[:, 3].astype(int)

        ratio_mean = np.mean(ratio_data_index3_np)
        ratio_std = np.std(ratio_data_index3_np)

        ratio_data_dict = {}

        for i in range(len(ratio_data_list)):
            ratio_data_list[i].append(
                self.cal_cumulative_dist_scorize(ratio_data_list[i][3], ratio_mean, ratio_std)
            )

            ratio_data_dict[ratio_data_list[i][0]] = ratio_data_list[i]

        return ratio_data_dict

    def average125_kospi_estrangement(self, KOSPI_dict):  # + 탐욕, - 공포
        # KOSPI_dict의 125일 이평선과 당일 값의 이격도를 계산
        # [날짜, 당일 코스피, ]
        kospi_value = np.array(sorted(list(KOSPI_dict.values())), dtype=object)[::-1]
        while not len(kospi_value[0])==8:
            kospi_value = kospi_value[1:]
        for i in range(len(kospi_value)):
            kospi_value[i] = np.array(kospi_value[i], dtype=object)
        print(kospi_value[0])
        average125 = pd.DataFrame(kospi_value[:, 1].astype(float)).rolling(window=125).mean().dropna().to_numpy().reshape(-1)

        estrangement_average125_kospi_list = []
        for i in range(len(average125)):
            estrangement_average125_kospi_list.append([
                kospi_value[i][0],  # 날짜
                float(kospi_value[i][1]),  # 코스피 값
                float(average125[i]),  # 125일 이평선 값
                float(kospi_value[i][1]) - float(average125[i])  # 코스피와 125일 이평선 사이의 이격도
                                                    ])

        estrangement_index3_np = np.array(estrangement_average125_kospi_list)[:, 3].astype(float)

        estrangement_mean = np.mean(estrangement_index3_np)
        estrangement_std = np.std(estrangement_index3_np)

        estrangement_dict = {}

        for i in range(len(estrangement_average125_kospi_list)):

            estrangement_average125_kospi_list[i].append(
                self.cal_cumulative_dist_scorize(estrangement_average125_kospi_list[i][3],
                                estrangement_mean,
                                estrangement_std)
            )

            estrangement_dict[estrangement_average125_kospi_list[i][0]] = estrangement_average125_kospi_list[i]

        return estrangement_dict
        pass  # average125_kospi_estrangement 의 끝

    def safe_haven_demand(self, KOSPI_dict, bond_dict_2y):  # + 탐욕, - 공포
        # KOSPI_dict의 20일간 주식 수익률과 2년 만기 채권의 수익률을 계산
        # [날짜, 당일 코스피, ]
        # 20일간 수익률 계산을 위해 [::-1] 을 취함 - 과거 날짜가 앞 인덱스
        # 해당 혼란 방지를 위해 모든 아웃풋을 딕셔너리로 (key = date)
        temp = list(KOSPI_dict.values())
        temp.sort(reverse=True)
        kospi_value = np.array(temp)
        # 최신 날짜부터

        # 이 부분이 뭔가 이상함함
        stock_20d_profit = pd.DataFrame(kospi_value[:, 1].astype(float)).pct_change(periods=20).dropna().to_numpy()[::-1]

        stock_bond_list = []
        stock_bond_dict = {}

        for i in range(len(stock_20d_profit)):
            try:
                date = kospi_value[i][0]
                stock_bond_list.append([
                    kospi_value[i][0],  # 날짜
                    float(kospi_value[i][1]),  # kospi 값
                    float(stock_20d_profit[i]) * 100,  # kospi 20일 수익률 (%)
                    float(bond_dict_2y[date][1]),  # 국고채 수익률
                    float(stock_20d_profit[i]) * 100 - float(bond_dict_2y[date][1]),  # kospi 수익률 - 국고채 수익률
                ])
            except KeyError:
                print("{} Date Key Error 발생".format(date))

        stock_bond_index4_np = np.array(stock_bond_list)[:, 4].astype(float)

        stock_bond_mean = np.mean(stock_bond_index4_np)
        stock_bond_std = np.std(stock_bond_index4_np)

        for i in range(len(stock_bond_list)):

            stock_bond_list[i].append(
                self.cal_cumulative_dist_scorize(
                    stock_bond_list[i][4],
                    stock_bond_mean,
                    stock_bond_std
                )
            )

            date = stock_bond_list[i][0]
            stock_bond_dict[date] = stock_bond_list[i]

        return stock_bond_dict
        pass  # safe_haven_demand 의 끝

    def credit_vs_speculation(self, bond_dict_2y):  # + 탐욕, - 공포
        # - 점수 : 투기 이율 평균이 작다 --> 공포. + : 위험선호 = 탐욕

        credit_spec_list = []
        credit_spec_dict = {}

        for date in bond_dict_2y.keys():

            credit_average = np.mean(bond_dict_2y[date][2:9].astype(float))  # 1은 국고채,
            # AAA, AA+, AA, AA-, A+, A, A-
            spec_average = np.mean(bond_dict_2y[date][9:12].astype(float))
            # BBB+, BBB, BBB-
            credit_spec_list.append(
            [
                date,  # 날짜
                credit_average,  # 신용등급 채권 이율 평균
                spec_average,  # 투기등급 채권 이율 평균
                spec_average / credit_average
            ])

        credit_spec_index3_np = np.array(credit_spec_list)[:, 3].astype(float)
        credit_spec_mean = np.mean(credit_spec_index3_np)
        credit_spec_std = np.std(credit_spec_index3_np)

        for i in range(len(credit_spec_list)):
            date = credit_spec_list[i][0]

            credit_spec_list[i].append(
                self.cal_cumulative_dist_scorize(
                    credit_spec_list[i][3],
                    credit_spec_mean,
                    credit_spec_std
                )
            )
            credit_spec_dict[date] = credit_spec_list[i]

        return credit_spec_dict
        pass  # credit_vs_speculation의 끝

    def stock_tradevol_breath(self, KOSPI_dict, KOSDAQ_dict):  # + : 탐욕, - : 공포
        # KOSPI_dict, KOSDAQ_dict 투입
        # stock_breath_dict 리턴

        stock_breath_list = []
        stock_breath_dict = {}

        for dates in KOSPI_dict.keys():
            try:
                kospi_vol_p = KOSPI_dict[dates][6]
                kospi_vol_m = KOSPI_dict[dates][7]
                kosdaq_vol_p = KOSDAQ_dict[dates][6]
                kosdaq_vol_m = KOSDAQ_dict[dates][7]

                stock_breath_list.append(
                [
                    dates,  # 날짜
                    kospi_vol_p,  # 가격 증가한 종목 거래량
                    kosdaq_vol_p,  # 가격 증가한 종목 거래량
                    kospi_vol_p + kosdaq_vol_p,
                    kospi_vol_m,  # 가격 하락한 종목 거래량
                    kosdaq_vol_m,
                    kospi_vol_m + kosdaq_vol_m,
                    kospi_vol_p + kosdaq_vol_p + kospi_vol_m + kosdaq_vol_m

                ])

            except KeyError:
                pass

        stock_breath_index7_np = np.array(stock_breath_list)[:, 7].astype(float)
        stock_breath_mean = np.mean(stock_breath_index7_np)
        stock_breath_std = np.std(stock_breath_index7_np)

        for i in range(len(stock_breath_list)):
            stock_breath_list[i].append(
                self.cal_cumulative_dist_scorize(
                stock_breath_list[i][7],
                stock_breath_mean,
                stock_breath_std)
            )
            date = stock_breath_list[i][0]
            stock_breath_dict[date] = stock_breath_list[i]

        return stock_breath_dict
        pass  # kospi_tradevol_breath 의 끝

    def average50_vkospi_estrange(self, vkospi_dict):  # + : 탐욕, - : 공포
        # 원래라면 + 공포, - 탐욕 이므로 -값을 붙여서 탐욕과 공포를 바꾼다
        # vkospi_dict의 50일 이평선과 당일 값의 이격도를 계산
        # [날짜, 당일 v코스피, 125 평균, v코스피-평균, 점수]

        #sort 필요함
        temp2 = list(vkospi_dict.values())
        # numpy.ndarray 와 list 간 정렬문제로, 임시 리스트를 만들어 복사한다
        temp = []
        for i in temp2:
            if type(i) == np.ndarray:
                temp.append(i.tolist())
            else:
                temp.append(i)
        del temp2
        vkospi_value = np.array(temp, dtype=object)
        average50= pd.DataFrame(vkospi_value[:, 1].astype(float)).rolling(window=50).mean().dropna().to_numpy().reshape(-1)

        estrangement_vkospi_list = []
        for i in range(len(average50)):
            estrangement_vkospi_list.append([
                vkospi_value[i][0],  # 날짜
                float(vkospi_value[i][1]),  # v코스피 값
                float(average50[i]),  # 50일 이평선 값
                float(vkospi_value[i][1]) - float(average50[i])  # v코스피와 50일 이평선 사이의 이격도
            ])

        estrangement_index3_np = np.array(estrangement_vkospi_list)[:, 3].astype(float)
        estrangement_mean = np.mean(estrangement_index3_np)
        estrangement_std = np.std(estrangement_index3_np)

        estrangement_dict = {}

        for i in range(len(estrangement_vkospi_list)):

            estrangement_vkospi_list[i].append(
                -1 * self.cal_cumulative_dist_scorize(estrangement_vkospi_list[i][3],
                                estrangement_mean,
                                estrangement_std)
            )

            estrangement_dict[estrangement_vkospi_list[i][0]] = estrangement_vkospi_list[i]

        return estrangement_dict
        pass  # average_vkospi_estrangement 의 끝

    def fear_greed_index(self, KOSPI_dict, KOSDAQ_dict, bond_dict_2y, vkospi_dict):

        ratio_highlow_dict = self.ratio_52_highlow(KOSPI_dict, KOSDAQ_dict)
        kospi_estrangement_dict = self.average125_kospi_estrangement(KOSPI_dict)
        safe_haven_dict = self.safe_haven_demand(KOSPI_dict, bond_dict_2y)
        credit_spec_dict = self.credit_vs_speculation(bond_dict_2y)
        stock_tradevol_dict = self.stock_tradevol_breath(KOSPI_dict, KOSDAQ_dict)
        vkospi_estrangement_dict = self.average50_vkospi_estrange(vkospi_dict)

        dicts_list = [ratio_highlow_dict,
                    kospi_estrangement_dict,
                    safe_haven_dict,
                    credit_spec_dict,
                    stock_tradevol_dict,
                    vkospi_estrangement_dict]
        ### 시작날짜 조정

        keys_list = [
            list(ratio_highlow_dict.keys()),
            list(kospi_estrangement_dict.keys()),
            list(safe_haven_dict.keys()),
            list(credit_spec_dict.keys()),
            list(stock_tradevol_dict.keys()),
            list(vkospi_estrangement_dict.keys())
        ]
        key_min_list = []
        for keylist in keys_list:
            key_min_list.append(len(keylist))

        keyindex = key_min_list.index(np.min(np.array(key_min_list)))
        keys = keys_list[keyindex]

        keys.sort(reverse=True)

        fgi_dict = {}
        # fgi_list = []

        for dates in keys:
            try:
                temp_list = []
                temp_list.append(dates)
                for dicts in dicts_list:
                    temp_list.append(dicts[dates][-1])
                mean = np.mean(np.array(temp_list)[1:].astype(float))
                temp_list.append(mean)

                # fgi_list.append(temp_list)
                fgi_dict[dates] = temp_list
            except KeyError:

                pass

        return fgi_dict
        pass  # fear_greed_index 의 끝

class DNN(FGI):

    # 전체 시장의 mean, std를 계산하는 함수를 따로 두고
    # 해당 값을 이용해서 각 종목 / 전체 종목의 value를 두는 함수를 추가로 생성
    # 아래는 전체 시장의 mean, std를 계산하는 함수
    def market_index(self, kosis_dict, year, month, day=28):
        # day 는 다른 것과 모양을 맞추기 위해 그냥 넣음
        # kosis_dict[date] = 월, 선행종합지수. 선행종합지수 순환변동치, 동행지수, 동행지수 순환변동치, 후행종합지수
        setting_date = datetime.date(year=year, month=month, day=day)
        
        months_4 = setting_date - relativedelta(months=4)    
        year_4m = months_4.year
        month_4m = months_4.month
        
        date_list = list(kosis_dict.keys())
        date_list.sort(reverse=True)
        
        index = date_list.index(str(year_4m) + '-' + str(month_4m).zfill(2))

        average_index_456 = (float(kosis_dict[date_list[index]][2]) + 
                            float(kosis_dict[date_list[index+1]][2]) + 
                            float(kosis_dict[date_list[index+2]][2])) / 3
        average_index_567 = (float(kosis_dict[date_list[index+3]][2]) + 
                            float(kosis_dict[date_list[index+1]][2]) + 
                            float(kosis_dict[date_list[index+2]][2])) / 3
    
        present_idx_predict = average_index_456 * 0.58 + 41.93
        past_idx_predict = average_index_567 * 0.58 + 41.93
        
        return present_idx_predict, past_idx_predict
    
    # 이 함수는 위의 market_index를 대입하여 전체 company의 mean, std 점수를 낸다. 
    def company_market_beta(self, company_dict, present_market_idx, past_market_idx):
        
        temp_beta_list = []
        for keys in company_dict.keys():
            try:
                beta = company_dict[keys][0][21]
                temp_beta_list.append((float(present_market_idx) - float(past_market_idx)) * float(beta))
            except Exception as e:
                print(keys, e)
                pass
                
        temp_beta_np = np.array(temp_beta_list)
        temp_beta_mean = temp_beta_np.mean()
        temp_beta_std = temp_beta_np.std()
        
        return temp_beta_mean, temp_beta_std
        pass  # company_market_beta의 끝

    def per_pbr_debt_meanstd(self, company_dict, pbr_per_debt, year, month, day=28):
        # 1 / pbr, 1 / per인 점을 명심한다
        # debt 는 여기서는 그대로 나가지만 나중에 계산할 땐 1점에서 빼야한다는 점을 명심한다
        # company_dict == company_pickle[1]
        #  매출액, 영업이익, 당기순이익, 영업이익률, 순이익률, ROE, 부채비율, 당좌비율, 유보율, EPS, PER, BPS, PBR
        # mean, std가 나온다
        
        input_date = datetime.datetime(year=year, month=month, day=day)

        temp_list = []
        if pbr_per_debt == 'pbr':
            pbr_per_debt_index = 12
        elif pbr_per_debt == 'per':  # per
            pbr_per_debt_index = 10
        elif pbr_per_debt == 'debt':
            pbr_per_debt_index = 6
        

        for keys in company_dict.keys():
            if company_dict[keys][3] != {}:
                term_list = list(company_dict[keys][3].keys())
                term_list.sort(reverse=True)

                for i in range(len(term_list)):
                    term_date = datetime.datetime(year=int(term_list[i][:4]), month=int(term_list[i][5:7]), day=28)
                    if input_date - term_date >= datetime.timedelta(0):
                        #print(term_date)
                        list_index = i
                        break

                if float(company_dict[keys][3][term_list[list_index]][pbr_per_debt_index]) == -1:  # 데이터가 전부 -1일 경우 다음 분기 정보 사용
                    list_index += 1

                if float(company_dict[keys][3][term_list[list_index]][pbr_per_debt_index]) == -1:  # 2번째
                    print("Error {}종목".format(keys))

                try:
                    if pbr_per_debt == 'debt':  # debt : 그대로 넣는다
                        temp_list.append(float(company_dict[keys][3][term_list[list_index]][pbr_per_debt_index]))                    
                    else: # pbr, per
                        # 각 회사의 1 / PER 을 넣는다 --> 인덱스 10
                        temp_list.append(
                            1 / float(company_dict[keys][3][term_list[list_index]][pbr_per_debt_index])
                        )
                except Exception:  # PER 없는 회사
                    pass

        temp_np = np.array(temp_list)
        mean = np.mean(temp_np)
        std = np.std(temp_np)

        return mean, std
        pass  # def per_meanstd의 끝

    def company_price_variability_meanstd(self, company_dict, year=0, month=0, day=28):
        # (52주 최고 - 최저) / 52주 최저
        
        variability_list = []
        for keys in company_dict.keys():
            try:
                variability = (int(company_dict[keys][0][8]) - int(company_dict[keys][0][9])) / int(company_dict[keys][0][9])
                variability_list.append(variability)            
            except Exception as e:
                pass
            
        variability_np = np.array(variability_list)
        mean = np.mean(variability_np)
        std = np.std(variability_np)
        
        return mean, std
        pass  # company_price_variability_meanstd의 끝
    
    def company_trade_vol_meanstd(self, company_dict, year, month, day):
        # 5일 평균 거래량 * 가격
        # 특정 날짜의 전체 회사의 거래량 * 가격을 이용한 mean, std

        input_date = datetime.datetime(year=year, month=month, day=day)
        trade_vol_list = []
        
        
        for keys in company_dict.keys():
            if company_dict[keys][1] != {}:
                term_list = list(company_dict[keys][1].keys())
                term_list.sort(reverse=True)
                
                for i in range(len(term_list)):
                    term_date = datetime.datetime(year=int(term_list[i][:4]), month=int(term_list[i][5:7]), day=int(term_list[i][8:]))
                    if input_date - term_date >= datetime.timedelta(0):
                        #print(term_date)
                        list_index = i
                        break
                
                try:
                    trade_vol = []
                    for i in range(5):
                        trade_vol.append(company_dict[keys][1][term_list[list_index + i]][1] * 
                                        company_dict[keys][1][term_list[list_index + i]][2]) # 가격 * 거래량
                    mean_trade_vol = np.array(trade_vol, dtype=np.uint64).mean()
                    trade_vol_list.append(mean_trade_vol)
                except Exception as e:
                    print(e, keys)
                    pass  
                
        trade_vol_np = np.array(trade_vol_list)
        mean = np.mean(trade_vol_np)
        std = np.std(trade_vol_np)                

        return mean, std
        pass  # def company_trade_vol_meanstd의 끝
    
    def company_weight_meanstd(self, company_dict, year, month, day):
        
        input_date = datetime.datetime(year=year, month=month, day=day)
        weight_list = []
        
        
        for keys in company_dict.keys():
            if company_dict[keys][1] != {}:
                term_list = list(company_dict[keys][1].keys())
                term_list.sort(reverse=True)
                
                for i in range(len(term_list)):
                    term_date = datetime.datetime(year=int(term_list[i][:4]), month=int(term_list[i][5:7]), day=int(term_list[i][8:]))
                    if input_date - term_date >= datetime.timedelta(0):
                        #print(term_date)
                        list_index = i
                        break
                
                try:
                    # 가격 * 상장 주식 수
                    weight_list.append(
                        int(
                        int(company_dict[keys][1][term_list[list_index]][1]) *
                        int(company_dict[keys][0][3])
                        /10000))
                except Exception as e:
                    print(e, keys)
                    pass  
                
        weight_np = np.array(weight_list)
        mean = np.mean(weight_np)
        std = np.std(weight_np)                

        return mean, std
        pass  # def company_weightmeanstd의 끝
        
    # 이 함수는 개별 회사 단위로 들어가며 for keys in dict.keys() 를 해줘야 한다
    def company_price_multi(self, company_dict, company_name, year, month, day):
        #각 회사의 그 당시 가격이 52주 최저 대비 몇 배인가 --> 를 역수로 (1/그 수) 하여 리스트에 추가
        #52주 최저는 당시 52주 최저가 아니라 현재 기록되어 있는 바를 사용
        input_date = datetime.datetime(year=year, month=month, day=day)
        
        temp_list = []
        key = company_name   
        
        for dates in company_dict[key][1].keys():
            try:        
                temp_list.append(
                    # 52주 최저 / 해당 날짜 가격
                int(company_dict[key][0][9]) / int(company_dict[key][1][dates][1])
                )
            except Exception as e:
                print("")
                pass

        temp_np = np.array(temp_list)
        mean = np.mean(temp_np)
        std = np.std(temp_np)
        
        # 해당 날짜의 값 구하기
        term_list = list(company_dict[key][1].keys())
        for list_index in range(len(term_list)):
            if input_date - datetime.datetime.strptime(term_list[list_index], '%Y-%m-%d') >=  datetime.timedelta(0):
                break 
        
        try:
            company_price_multi = company_dict[key][0][9] / company_dict[key][1][term_list[list_index]][1]
            output = FGI.cal_cumulative_dist(
                company_price_multi,
                mean,
                std
                )
        except:
            output = 0
            
        return output
        pass  # company_price_multi 의 끝

    # 이 함수는 개별 회사 단위로 들어가며 for keys in dict.keys() 를 해줘야 한다
    def company_days_moving(self, company_dict, company_name, d20_d5_number, year, month, day):
        # 20일 이평선 +- std 어디에 위치해있는가? (볼린져 밴드)
        # 위의 것과는 다르게 이 함수는 개별 회사 단위로 들어가며 for keys in dict.keys() 를 해줘야 한다    
        
        key = company_name

        # sort 필요
        price_list = np.array(list(company_dict[key][1].values()))
        
        # 20d, 5d rolling mean을 시행하고, 없는 값들을 제외한다
        price_rolling_mean = np.array(pd.DataFrame(price_list[:, 1]).rolling(window=d20_d5_number).mean().dropna())
        # dtype을 object로 변환
        price_rolling_mean = price_rolling_mean.astype(object)
        
        # std에 대해서 동일 항목 수행
        price_rolling_std = np.array(pd.DataFrame(price_list[:, 1]).rolling(window=d20_d5_number).mean().dropna())
        price_rolling_std = price_rolling_std.astype(object)
        price_mean_dict = {}
        price_std_dict = {}
        for i in range(len(price_rolling_mean)):
            # price_list의 위에서부터 20개가 rolling_mean의 0번 값이었다
            # 날짜를 삽입
            price_mean_dict[price_list[i][0]] = price_rolling_mean[i][0]
            price_std_dict[price_list[i][0]] = price_rolling_std[i][0]
        
        str_date = str(year)+'-'+str(month).zfill(2)+'-'+str(day).zfill(2)
        
        # 날짜가 없으면 에러를 일으키는게 맞다
        if not str_date in price_mean_dict.keys():
            print("해당 날짜 자료 없음")
            raise IndexError
            
        price_now = int(company_dict[key][1][str_date][1])
        mean_now = price_mean_dict[str_date]
        std_now = price_std_dict[str_date]
        
        output = self.cal_cumulative_dist(price_now, mean_now, std_now)
        return output    
        pass  # company_days_moving의 끝
    

    # 전체 점수
    def entire_market_meanstd(self, kosis_dict, company_dict, year, month, day):
        
        present_idx, past_idx = self.market_index(kosis_dict, year, month, day)
        mean_marketbeta, std_marketbeta = self.company_market_beta(company_dict, present_idx, past_idx)
        mean_company_profit, std_company_profit = self.per_pbr_debt_meanstd(company_dict, 'per', year, month, day)
        mean_company_property, std_company_property = self.per_pbr_debt_meanstd(company_dict, 'pbr', year, month, day)
        mean_company_debt, std_company_debt = self.per_pbr_debt_meanstd(company_dict, 'debt', year, month, day)
        mean_company_variability, std_company_variability = self.company_price_variability_meanstd(company_dict)
        mean_company_tradevol, std_company_tradevol = self.company_trade_vol_meanstd(company_dict, year, month, day)
        mean_company_weight, std_company_weight = self.company_weight_meanstd(company_dict, year, month, day)
        
        output = [
        mean_marketbeta,
        std_marketbeta,
        mean_company_profit,
        std_company_profit,
        mean_company_property,
        std_company_property,
        mean_company_debt,
        std_company_debt,
        mean_company_variability,
        std_company_variability,
        mean_company_tradevol,
        std_company_tradevol,
        mean_company_weight,
        std_company_weight]
        
        return output
        pass  # entire_market_meanstd의 끝

    def company_score(self, company_dict, fear_greed_index_dict, kosis_dict, keys, year, month, day, mean_marketbeta, std_marketbeta, mean_company_profit,
        std_company_profit, mean_company_property, std_company_property, mean_company_debt, std_company_debt, mean_company_variability,
        std_company_variability, mean_company_tradevol, std_company_tradevol, mean_company_weight, std_company_weight):       
        # keys = company_name
        
        # year, month, day로 만드는 date (str)
        date = str(year) + '-' + str(month).zfill(2) + '-' + str(day).zfill(2)
        input_date = datetime.datetime(year=year, month=month, day=day)
        
        fear_greed_index = fear_greed_index_dict[date]

        # 52주 베타
        present_market_idx, past_market_idx = self.market_index(kosis_dict, year, month, day=28)
        company_marketbeta = float(company_dict[keys][0][21]) * (present_market_idx - past_market_idx)
        company_marketbeta_score = self.cal_cumulative_dist(company_marketbeta, mean_marketbeta, std_marketbeta)
        
        # 1/per (profit) ~ company_debt까지.
        # 해당 부분에서 사용할 
        term_list = list(company_dict[keys][3].keys())
        term_list.sort(reverse=True)

        for i in range(len(term_list)):
            term_date = datetime.datetime(year=int(term_list[i][:4]), month=int(term_list[i][5:7]), day=28)
            if input_date - term_date >= datetime.timedelta(0):
                #print(term_date)
                list_index = i
                break

        if float(company_dict[keys][3][term_list[list_index]][10]) == -1:  # 데이터가 전부 -1일 경우 다음 분기 정보 사용
            list_index += 1

        if float(company_dict[keys][3][term_list[list_index]][10]) == -1:  # 2번째
            print("{} 종목 Profit 로딩 Error".format(keys))    
        
        company_profit = 1 / float(company_dict[keys][3][term_list[list_index]][10])
        company_profit_score = self.cal_cumulative_dist(company_profit, mean_company_profit, std_company_profit)
        
        company_property = 1 / float(company_dict[keys][3][term_list[list_index]][12])
        company_property_score = self.cal_cumulative_dist(company_property, mean_company_property, std_company_property)
        
        company_debt = float(company_dict[keys][3][term_list[list_index]][6])
        company_debt_score = self.cal_cumulative_dist(company_debt,mean_company_debt, std_company_debt)
        
        # 임시값(?) company_weight
        term_list = list(company_dict[keys][1].keys())
        term_list.sort(reverse=True)

        for i in range(len(term_list)):
            term_date = datetime.datetime(year=int(term_list[i][:4]), month=int(term_list[i][5:7]), day=int(term_list[i][8:]))
            if input_date - term_date >= datetime.timedelta(0):
                #print(term_date)
                list_index = i
                break
        
        company_weight =int(company_dict[keys][0][2]) * int(company_dict[keys][0][3]) / 10000
        company_weight_score = self.cal_cumulative_dist(company_weight, mean_company_weight, std_company_weight)
            
        # (52 최고 - 최저) / 52최저 : price_variability_meanstd
        company_variability = (int(company_dict[keys][0][8]) - int(company_dict[keys][0][9])) / int(company_dict[keys][0][9])
        company_variability_score = self.cal_cumulative_dist(company_variability, mean_company_variability, std_company_variability)
        
        # 가격 * 거래량
        company_tradevol = int(company_dict[keys][1][date][1]) * int(company_dict[keys][1][date][2])
        company_tradevol_score = self.cal_cumulative_dist(company_tradevol, mean_company_tradevol, std_company_tradevol)
        
        # 5일, 20일 이평선
        company_days5_score = self.company_days_moving(company_dict, keys, 5, year, month, day)
        company_days20_score = self.company_days_moving(company_dict, keys, 20, year, month, day)
        
        # 회사 가격이 52주 최저 대비 몇배인가 : price_multi
        company_multi_score = self.company_price_multi(company_dict, keys, year, month, day)
        
        # 코스피 / 코스닥 여부
        company_kospi_kosdaq = 1 if company_dict[keys][0][20] == 'kospi' else 0
        
        output = [
            company_marketbeta_score,
            company_weight_score,
            company_profit_score,
            company_property_score,
            company_debt_score,
            company_multi_score,
            company_variability_score,
            company_tradevol_score,
            company_days5_score,
            company_days20_score,
            fear_greed_index[-1],
            company_kospi_kosdaq
        ]
        return output
        pass  # company_score의 끝


class DNN_Training(DNN):

    def company_score(self, company_dict, fear_greed_index_dict, kosis_dict, keys, year, month, day, mean_marketbeta, std_marketbeta, mean_company_profit,
    std_company_profit, mean_company_property, std_company_property, mean_company_debt, std_company_debt, mean_company_variability,
    std_company_variability, mean_company_tradevol, std_company_tradevol, mean_company_weight, std_company_weight):       
    # keys = company_name
    
        # year, month, day로 만드는 date (str)
        date = str(year) + '-' + str(month).zfill(2) + '-' + str(day).zfill(2)
        input_date = datetime.datetime(year=year, month=month, day=day)
        
        fear_greed_index = fear_greed_index_dict[date]
        
        # 52주 베타

        present_market_idx, past_market_idx = self.market_index(kosis_dict, year, month, day=28)
        company_marketbeta = float(company_dict[keys][0][21]) * (present_market_idx - past_market_idx)
        company_marketbeta_score = self.cal_cumulative_dist(company_marketbeta, mean_marketbeta, std_marketbeta)
        
        # 1/per (profit) ~ company_debt까지.
        # 해당 부분에서 사용할 
        term_list = list(company_dict[keys][3].keys())
        term_list.sort(reverse=True)

        for i in range(len(term_list)):
            term_date = datetime.datetime(year=int(term_list[i][:4]), month=int(term_list[i][5:7]), day=28)
            if input_date - term_date >= datetime.timedelta(0):
                #print(term_date)
                list_index = i
                break

        if float(company_dict[keys][3][term_list[list_index]][10]) == -1:  # 데이터가 전부 -1일 경우 다음 분기 정보 사용
            list_index += 1

        if float(company_dict[keys][3][term_list[list_index]][10]) == -1:  # 2번째
            print("{} 종목 Profit 로딩 Error".format(keys))    
        
        company_profit = 1 / float(company_dict[keys][3][term_list[list_index]][10])
        company_profit_score = self.cal_cumulative_dist(company_profit, mean_company_profit, std_company_profit)
        
        company_property = 1 / float(company_dict[keys][3][term_list[list_index]][12])
        company_property_score = self.cal_cumulative_dist(company_property, mean_company_property, std_company_property)
        
        company_debt = float(company_dict[keys][3][term_list[list_index]][6])
        company_debt_score = self.cal_cumulative_dist(company_debt,mean_company_debt, std_company_debt)
        
        # 임시값(?) company_weight
        term_list = list(company_dict[keys][1].keys())
        term_list.sort(reverse=True)

        for i in range(len(term_list)):
            term_date = datetime.datetime(year=int(term_list[i][:4]), month=int(term_list[i][5:7]), day=int(term_list[i][8:]))
            if input_date - term_date >= datetime.timedelta(0):
                #print(term_date)
                list_index = i
                break
        
        company_weight =int(
                        int(company_dict[keys][1][term_list[list_index]][1]) *
                        int(company_dict[keys][0][3])
                        /10000)
        company_weight_score = self.cal_cumulative_dist(company_weight, mean_company_weight, std_company_weight)
            
        # (52 최고 - 최저) / 52최저 : price_variability_meanstd
        company_variability = (int(company_dict[keys][0][8]) - int(company_dict[keys][0][9])) / int(company_dict[keys][0][9])
        company_variability_score = self.cal_cumulative_dist(company_variability, mean_company_variability, std_company_variability)
        
        # 가격 * 거래량
        company_tradevol = int(company_dict[keys][1][date][1]) * int(company_dict[keys][1][date][2])
        company_tradevol_score = self.cal_cumulative_dist(company_tradevol, mean_company_tradevol, std_company_tradevol)
        
        # 5일, 20일 이평선
        company_days5_score = self.company_days_moving(company_dict, keys, 5, year, month, day)
        company_days20_score = self.company_days_moving(company_dict, keys, 20, year, month, day)
        
        # 회사 가격이 52주 최저 대비 몇배인가 : price_multi
        company_multi_score = self.company_price_multi(company_dict, keys, year, month, day)
        
        # 코스피 / 코스닥 여부
        company_kospi_kosdaq = 1 if company_dict[keys][0][20] == 'kospi' else 0
        
        output = [
            company_marketbeta_score,
            company_weight_score,
            company_profit_score,
            company_property_score,
            company_debt_score,
            company_multi_score,
            company_variability_score,
            company_tradevol_score,
            company_days5_score,
            company_days20_score,
            fear_greed_index[-1],
            company_kospi_kosdaq
        ]
        return output
        pass  # company_score의 끝

    def company_training_data(self, company_dict, fear_greed_index_dict, kosis_dict, keys, mean_marketbeta, std_marketbeta, mean_company_profit,
    std_company_profit, mean_company_property, std_company_property, mean_company_debt, std_company_debt, mean_company_variability,
    std_company_variability, mean_company_tradevol, std_company_tradevol, mean_company_weight, std_company_weight):       
    # keys = company_name

        price_list = list(company_dict[keys][1].values())
        answer_list = []

        print("{} 종목 training data 추출 중".format(keys))
        
        for i in range(23, 261):
            try:
                date_search = datetime.datetime.strptime(price_list[i][0], '%Y-%m-%d')
                year = date_search.year
                month = str(date_search.month).zfill(2)
                day = str(date_search.day).zfill(2)


                # 가격 정보 답안
                #18, 19, 20, 21, 22일 뒤의 평균 가격을 리스트로 추가하기 위한 부분
                price_output_list = []
                for j in range(5):
                    price_output_list.append(int(price_list[i-18-j][2]))
                price_output_np = np.array(price_output_list)
                avg_price_future = np.mean(price_output_np)
                avg_price_fut_ratio = avg_price_future / int(price_list[i][2])

                if avg_price_fut_ratio >= 1.05:
                    answer = 0
                elif avg_price_fut_ratio < 1.05 and avg_price_fut_ratio >= 1.005:
                    answer = 1
                elif avg_price_fut_ratio < 1.005 and avg_price_fut_ratio >= 0.995:
                    answer = 2
                elif avg_price_fut_ratio < 0.995 and avg_price_fut_ratio >= 0.95:
                    answer = 3
                elif avg_price_fut_ratio < 0.95:
                    answer = 4

                output = self.company_score(company_dict, fear_greed_index_dict, kosis_dict, keys, year, int(month), int(day), mean_marketbeta, std_marketbeta, mean_company_profit,
                                        std_company_profit, mean_company_property, std_company_property, mean_company_debt, std_company_debt, mean_company_variability,
                                        std_company_variability, mean_company_tradevol, std_company_tradevol, mean_company_weight, std_company_weight)
                output.append(answer)
                answer_list.append(output)

            except Exception as e:
                # print(e, date_search, keys)
                pass
                
        return answer_list
        pass  # company_score의 끝

    def entire_dnn_training_data(self, company_dict, fear_greed_index_dict, kosis_dict, mean_marketbeta, std_marketbeta, mean_company_profit,
    std_company_profit, mean_company_property, std_company_property, mean_company_debt, std_company_debt, mean_company_variability,
    std_company_variability, mean_company_tradevol, std_company_tradevol, mean_company_weight, std_company_weight):
        
        test_list = []
        for keys in company_dict.keys():
            if keys != "종목명":
                test_list.append(self.company_training_data(company_dict, fear_greed_index_dict, kosis_dict, keys, mean_marketbeta, std_marketbeta, mean_company_profit,
                                std_company_profit, mean_company_property, std_company_property, mean_company_debt, std_company_debt, mean_company_variability,
                                std_company_variability, mean_company_tradevol, std_company_tradevol, mean_company_weight, std_company_weight))
        return test_list
        pass  # entire_dnn_training_data 의 끝

'''
class server_related():

    async def send_file(self, file, remote_path, host, port, username, password):
        # file = 파일경로 및 이름
        async with asyncssh.connect(host, port=port, username=username, password=password) as conn:
            # 2. company_list_pickle file send
            async with conn.start_sftp_client() as sftp:
                # put(localpaths, remotepath=None, *, preserve=False, recurse=False, follow_symlinks=False, block_size=16384, max_requests=128, progress_handler=None, error_handler=None)
                await sftp.put(file, remotepath=remote_path)

            print("Host : {} Uploading Files : {}".format(host, file))

    async def command_cmd(self, cmd, host, port, username, password):
        async with asyncssh.connect(host, port=port, username=username, password=password) as conn:
            result = await conn.run(cmd, check=True)
            print(result.stdout, end='')
            return result.stdout
'''

if __name__=="__main__":

    loading_cls = loading()
    company_pickle, KOSPI_dict, KOSDAQ_dict, bond_dict, vkospi_dict, kosis_dict = loading_cls.loading_data(n_month=15)
    fgi = FGI()
    fgi_dict = fgi.fear_greed_index(KOSPI_dict, KOSDAQ_dict, bond_dict['2y'], vkospi_dict)

    date = list(fgi_dict.keys())[0]
    year = int(date[:4])
    month = int(date[5:7])
    day = int(date[8:])
    #
    print(date)


    '''
    ###
    dnn = DNN()
    company_dict = company_pickle[1]
    market_meanstd = dnn.entire_market_meanstd(kosis_dict, company_dict, year, month, day)

    # 모델 로딩 후 평가 필요

    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(12,), name = 'Input'),
        keras.layers.Dense(100, activation='relu', name='Dense_1'),
        keras.layers.Dense(5, activation='softmax', name='predictions')])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.load_weights('price_predict_v4_weight')
    print("keras 모듈 임포트, 모델 로딩")

    time.sleep(3)

    entire_dict = {}
    recommand_dict = {}

    for keys in company_dict.keys():
        print(keys + " 종목 가격 예측")
        try:
            if keys != "종목명":
                test_list = dnn.company_score(company_dict, fgi_dict, kosis_dict, keys, year, month, day, *market_meanstd)
                test_list = np.array(test_list).reshape(-1, 12)
                prediction = model.predict(test_list)

                print(keys, prediction)

                with open('temp.csv', 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(test_list)
                    writer.writerow([keys, prediction])

                entire_dict[keys] = prediction
                if prediction.argmax() == 0 and prediction[0][1] >= prediction[0][2] \
                        and prediction[0][1] >= prediction[0][3] and prediction[0][1] > prediction[0][4]:
                    recommand_dict[keys] = prediction
                    pass
                # if prediction.argmax() == 0:
                #     recommand_dict[keys] = prediction
                #     pass

        except:
            print(keys + " 종목 가격 예측 실패")
            pass

    input()
    '''

    ### DNN 추천목록
    if not os.path.isfile('Files\\DNN\\Entire\\' + date) or True:

        dnn = DNN()
        company_dict = company_pickle[1]
        market_meanstd = dnn.entire_market_meanstd(kosis_dict, company_dict, year, month, day)

        # 모델 로딩 후 평가 필요

        model = keras.Sequential([
            keras.layers.InputLayer(input_shape=(12,), name = 'Input'),
            keras.layers.Dense(100, activation='relu', name='Dense_1'),
            keras.layers.Dense(5, activation='softmax', name='predictions')])

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
                      loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.load_weights('price_predict_dnn_v2_weight')
        print("keras 모듈 임포트, 모델 로딩")

        time.sleep(3)

        entire_dict = {}
        recommand_dict = {}

        try:
            print("민감도 설정, 0~1 사이의 소수 입력. 오류 발생 시 0.4로 설정")
            sensitivity = float(input("민감도 : "))
        except ValueError:
            sensitivity = 0.4

        for keys in company_dict.keys():
            print(keys + " 종목 가격 예측")
            try:
                if keys != "종목명":
                    test_list = dnn.company_score(company_dict, fgi_dict, kosis_dict, keys, year, month, day, *market_meanstd)
                    test_list = np.array(test_list).reshape(-1, 12)
                    prediction = model.predict(test_list)

                    print(keys, prediction)
                    entire_dict[keys] = prediction


                    if prediction[0].argmax() == 0 and prediction[0][0] >= sensitivity:
                        recommand_dict[keys] = prediction
                        pass
                    # if prediction.argmax() == 0:
                    #     recommand_dict[keys] = prediction
                    #     pass

            except:
                print(keys + " 종목 가격 예측 실패")
                pass

        recommand_list = list(recommand_dict.keys())

        with open('Files\\DNN\\Entire\\'+date, 'wb') as f:
            pickle.dump(entire_dict, f)

        with open('Files\\DNN\\Recommand\\'+date, 'wb') as f:
            pickle.dump(recommand_dict, f)

        with open('Files\\DNN\\Recommand_list\\'+date, 'wb') as f:
            pickle.dump(recommand_list, f)

        ### 서버로 파일 옮기기
        print("서버 파일 업데이트 시작 ...")
        server_dir = "/root/projects/Files/"
        server_host = '192.168.0.8'
        server_info = {'username': 'root', 'password': 'gksdlr77', 'port': 22}
        # ssh
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy)
        ssh.connect(server_host, **server_info)
        # sftp
        transport = paramiko.Transport((server_host, server_info['port']))
        transport.connect(None, server_info['username'], server_info['password'])
        sftp = paramiko.SFTPClient.from_transport(transport)

        # 날짜 폴더 생성
        print("날짜 폴더 생성")
        stdin, stdout, stderr = ssh.exec_command("mkdir " + server_dir + "AI/" + date)
        time.sleep(1)
        print(stdout.readlines())
        # input("날짜 폴더 생성 완료?")
        '''        
        try:
            stdin, stdout, stderr = ssh.exec_command("mkdir " + server_dir + "AI/" + date)
            print(stdout.readlines())
            # server.command_cmd("mkdir " + server_dir + "AI/" + date, **server_info)
            # os.makedirs(server_dir + "AI/" + date)
        except FileExistsError:
            pass
        '''

        # 지정된 경로에 recommand_list, recommand_dict 파일을 put 한다
        # recommand_dict 파일경로: 'Files\\DNN\\Recommand\\'+date
        # recommand_list 파일경로: 'Files\\DNN\\Recommand_list\\'+date
        sftp.put('Files\\DNN\\Recommand\\'+date, server_dir + "AI/" + date + "/score")
        sftp.put('Files\\DNN\\Recommand_list\\'+date, server_dir + "AI/" + date + "/list")

        # 이하 기존 windows 산하 서버 폴더에 저장했던 코드
        ''' 
        with open(server_dir + "AI/" + date + "/list", 'wb') as f:
            pickle.dump(recommand_list, f)
        with open(server_dir + "AI/" + date + "/score", 'wb') as f:
            pickle.dump(recommand_dict, f)
        '''
        ### filelist.json 수정하기

        # filelist.json을 get 한다
        sftp.get(server_dir + "filelist.json", "filelist.json")
        # get한 filelist.json을 수정한다
        with open("filelist.json", 'r') as f:
            json_filelist = json.load(f)

        # json_filelist['AI'] = [date] + json_filelist['AI'][:-1]
        if date not in json_filelist['AI']:
            if len(json_filelist['AI']) >=5:
                json_filelist['AI'] = [date] + json_filelist['AI'][:-1]
            else:
                json_filelist['AI'] += [date]

        with open("filelist.json", 'w') as f:
            json.dump(json_filelist, f)
        # 수정한 filelist.json을 서버에 put 한다
        sftp.put("filelist.json", server_dir + "filelist.json")
        # 현재 폴더의 filelist.json을 삭제한다
        os.remove("filelist.json")

        company_pickle, KOSPI_dict, KOSDAQ_dict, bond_dict, vkospi_dict, kosis_dict = loading_cls.loading_data(n_month=5)
        company_dict = company_pickle[1]

        # 개별 회사 데이터를 로컬 Files/Stocks/에 저장한다
        # 저장한 파일을 서버에 put한다
        for keys in company_dict.keys():
            temp_dict = {}
            temp_dict[keys] = company_dict[keys]
            with open("Files/Stocks/" + keys, 'wb') as f:
                pickle.dump(temp_dict, f)
            sftp.put("Files/Stocks/" + keys, server_dir + "/Stocks/" + keys)

    else:
        print(date + " 날짜 추천목록 있음. 분석 종료.")

    ### 저평가

    if not os.path.isfile('Files\\Undervalued\\Entire\\' + date):
        company_dict = company_pickle[1]

        for keys in company_dict.keys():

            if keys != "종목명":

                pass

            pass
    '''
    try:
        #if dict_company_list[keys][0][21] <= (max_volumerank * float(self.Line_MaxVoulmeRank.text()) / 100):
        if keys != "종목명":
            #print(keys)
            #print(dict_company_list[keys][0][0])
            if float(dict_company_list[keys][0][0][9]) * 1.05 >= dict_company_list[keys][1][1][2]: #52최저*1.05 > 최신 종가
                if dict_company_list[keys][0][0][19] == None and dict_company_list[keys][0][18] >= dict_company_list[keys][0][17]:
                    self.List_Lowest_Company.addItem(keys)
                    pass
                elif dict_company_list[keys][0][0][19] >= dict_company_list[keys][0][0][18] >= dict_company_list[keys][0][0][17]:
                    self.List_Lowest_Company.addItem(keys)
            	    pass

        except:
            print("종목 : " + keys + " - 계산 오류 발생, 해당 종목 건너뜀")
            pass'''


    ### 마법공식
        # PER, ROA(ROE)를 기준으로
        # ROE = (PBR / PER) * 100
        ### 전체 업종, 각 섹터별로 1/PER, ROE가 큰 순서대로 랭킹을 매긴다


    company_code_list = []
    # 한국거래소 - 주식 - 상장현황 - 상장회사검색
    with open('Company_Code.csv', 'r', encoding='UTF-8') as codefile:
        csvCode = csv.reader(codefile)
        for row in csvCode:
            company_code_list.append([row[1].zfill(6), row[2], row[3], row[4], row[5], row[6]])

    company_info_list = []
    # 리스트에 PER, PBR, ROE를 추가
    for i in range(1, len(company_code_list)):
        try:
            # print(key, company_dict[key][0][4], company_dict[key][0][6])
            key = company_code_list[i][1]
            per = float(company_dict[key][0][4])
            pbr = float(company_dict[key][0][6])
            roe = (pbr / per) * 100

            company_code_list[i].append(per)
            company_code_list[i].append(pbr)
            company_code_list[i].append(roe)
            # 정보가 있는 회사들만 추려서 저장
            company_info_list.append(company_code_list[i])
        except (TypeError, ValueError):
            print(key, company_dict[key][0][4], company_dict[key][0][6])
            pass

    # company_code_list에서 업종 리스트를 추려서 unique 값을 저장. '전체' 추가
    company_info_np = np.array(company_info_list)
    sector_list = list(set(company_info_np[:, 3]))
    sector_list.insert(0, '전체')

    # 딕셔너리 초기화 및 초기 모양 설정
    magic_formula_dict = {}
    for sectors in sector_list:
        magic_formula_dict[sectors] = []

    # 각 섹터별 종목 이름, ROE, 1/PER 추가
    for company in company_info_list:
        sector = company[3]
        company_name = company[1]
        roe = company[8]
        per_rev = 1 / company[6]
        to_add_list = [company_name, roe, per_rev]
        magic_formula_dict['전체'].append(to_add_list)
        magic_formula_dict[sector].append(to_add_list)

    # magic_formula_dict 내의 정보들을 pandas 객체로 바꾸고 rank 추가. rank 방법은 소수점을 없애기 위해 min 처리
    for sector in magic_formula_dict:
        magic_formula_dict[sector] = pd.DataFrame(magic_formula_dict[sector], columns = ['KEY', 'ROE', 'PER_REV'])
        magic_formula_dict[sector]['ROE_RANK'] = magic_formula_dict[sector]['ROE'].rank(method='min')
        magic_formula_dict[sector]['PER_REV_RANK'] = magic_formula_dict[sector]['PER_REV'].rank(method='min')
        magic_formula_dict[sector]['COMP_RANK'] = magic_formula_dict[sector]['ROE_RANK'] + \
                                                  magic_formula_dict[sector]['PER_REV_RANK']



tmp_pd = pd.DataFrame(magic_formula_dict['전체'])
tmp_pd['per_rev_rank'] = tmp_pd[2].rank(method='average')




        pass




'''
    # 다음은 모델 훈련 코드
    # 
    import ast
    import tensorflow as tf
    
    loading_cls = loading()
    # 훈련용
    company_pickle, KOSPI_dict, KOSDAQ_dict, bond_dict, vkospi_dict, kosis_dict = loading_cls.loading_data(n_month=33)
   
    fgi = FGI()
    fgi_dict = fgi.fear_greed_index(KOSPI_dict, KOSDAQ_dict, bond_dict['2y'], vkospi_dict)

    date = list(fgi_dict.keys())[0]
    year = int(date[:4])
    month = int(date[5:7])
    day = int(date[8:])
    #
    print(date)
    date = list(fgi_dict.keys())[0]
    year = int(date[:4])
    month = int(date[5:7])
    day = int(date[8:])
    dnn= DNN()
    company_dict = company_pickle[1]
    market_meanstd = dnn.entire_market_meanstd(kosis_dict, company_dict, year, month, day)
    
    dnn_training = DNN_Training()
    test_list = dnn_training.entire_dnn_training_data(company_dict, fgi_dict, kosis_dict, *market_meanstd)

    random.shuffle(test_list)
    entire_training_np = np.array(test_list)
    entire_training_pd = pd.DataFrame(entire_training_np)
    entire_training_pd.to_csv("training_data_3.csv", index=False)    
    
    entire_training_pd = pd.read_csv("training_data_3.csv")
    entire_training_np = np.array(entire_training_pd)
    entire_training_list = entire_training_np.tolist()
    
    
    temp = []
    for i in range(len(entire_training_list)):
        temp.append(ast.literal_eval(entire_training_list[i][0]))

    test_list = []
    for i in range(len(temp)):
        if temp[i] != []:
            for j in range(len(temp[i])):
                test_list.append(temp[i][j])      

    entire_training_np = np.array(test_list)

    del entire_training_pd
    del entire_training_list
    del test_list
    
    training_data = entire_training_np[:-30000, :12]
    training_label = entire_training_np[:-30000, -1]
    test_data = entire_training_np[-30000:, :12]
    test_label = entire_training_np[-30000:, -1]
    
    with tf.device('/device:GPU:0'):
        model = keras.Sequential([
            keras.layers.InputLayer(input_shape=(12,), name = 'Input'),
            keras.layers.Dense(100, activation = 'relu', name = 'Dense_1'),
            keras.layers.Dense(5, activation = 'softmax', name = 'predictions')])
            
        model.compile(optimizer = keras.optimizers.Adam(learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
        
        model.fit(training_data, training_label, batch_size = 1024, epochs=12800)
        model.fit(training_data, training_label, batch_size = 512, epochs=3200)
        
    model.evaluate(test_data, test_label, verbose = 2)
    
    model.save('price_predict_dnn_v2.h5')
    model.save_weights('price_predict_dnn_v2_weight')   
     
    model = keras.models.load_model('price_predict_dnn.h5')
'''