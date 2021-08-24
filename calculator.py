import pickle
import numpy as np
import pandas as pd

import datetime
from datetime import datetime as dt
from datetime import timedelta as td
from dateutil.relativedelta import relativedelta

from scipy import stats

import os


### 긁어온 정보를 계산한다

class FGI:
    def cal_cumulative_dist(self, x, mean, std):
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
            try:
                kospi_high = int(KOSPI_dict[dates][4])
            except Exception as e:
                print(e)
                kospi_high = 0
            try:
                kospi_low = int(KOSPI_dict[dates][5])
            except Exception as e:
                print(e)
                kospi_low = 0
            try:
                kosdaq_high = int(KOSDAQ_dict[dates][4])
            except Exception as e:
                print(e)
                kosdaq_high = 0
            try:
                kosdaq_low = int(KOSDAQ_dict[dates][5])
            except Exception as e:
                print(e)
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
                self.cal_cumulative_dist(ratio_data_list[i][3], ratio_mean, ratio_std)
            )
            
            ratio_data_dict[ratio_data_list[i][0]] = ratio_data_list[i]
            
        return ratio_data_dict
    
    def average125_kospi_estrangement(self, KOSPI_dict):  # + 탐욕, - 공포
        # KOSPI_dict의 125일 이평선과 당일 값의 이격도를 계산
        # [날짜, 당일 코스피, ]
        kospi_value = np.array(list(KOSPI_dict.values()))
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
                self.cal_cumulative_dist(estrangement_average125_kospi_list[i][3],
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
        
        kospi_value = np.array(list(KOSPI_dict.values()))
        # 최신 날짜부터
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
                self.cal_cumulative_dist(
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
                self.cal_cumulative_dist(
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
                self.cal_cumulative_dist(
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
        vkospi_value = np.array(list(vkospi_dict.values()))
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
                -1 * self.cal_cumulative_dist(estrangement_vkospi_list[i][3],
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
        fgi_list = []
        
        for dates in keys:
            try:
                temp_list = []
                temp_list.append(dates)
                for dicts in dicts_list:
                    temp_list.append(dicts[dates][-1])
                mean = np.mean(np.array(temp_list)[1:].astype(float))
                temp_list.append(mean)
                
                fgi_list.append(temp_list)
                fgi_dict[dates] = fgi_list
            except KeyError:
                
                pass
            
        return fgi_dict
        pass  # fear_greed_index 의 끝

class dnn:
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

    def beta52weeks_meanstd(self, present_idx, past_idx, company_dict):
        # 52주 베타의 std, mean만 계산한다
        # 이 계산은 한 번만 하면 됨. 굳이 이전처럼 매번 계산할 필요가 없음
        
        temp_list = []
        for keys in company_dict.keys():
            try:
                beta_52weeks = company_dict[keys][0][21]
                temp_list.append((present_idx - past_idx) * beta_52weeks)
            except Exception:
                pass
            
        temp_np = np.array(temp_list)
        mean_beta = np.mean(temp_np)
        std_beta = np.std(temp_np)
        
        return mean_beta, std_beta
        pass  # beta52weeks_meanstd 의 끝

    def per_meanstd(self, company_dict, year, month, day=28):
        
        # company_dict == company_pickle[1]
        #  매출액, 영업이익, 당기순이익, 영업이익률, 순이익률, ROE, 부채비율, 당좌비율, 유보율, EPS, PER, BPS, PBR
        
        input_date = datetime.date(year=year, month=month, day=day)
        
        temp_list = []
        
        for keys in company_dict.keys():
            if company_dict[keys][3] != {}:
                term_list = list(company_dict[keys][3].keys())
                term_list.sort(reverse=True)

                for i in range(len(term_list)):
                    term_date = datetime.date(year=int(term_list[i][:4]), month=int(term_list[i][5:8]), day=28)
                    if input_date - term_date >= datetime.timedelta(0):
                        #print(term_date)
                        list_index = i
                        break

                if float(company_dict[keys][3][term_list[list_index]][10]) == -1:  # 데이터가 전부 -1일 경우 다음 분기 정보 사용
                    list_index += 1

                if float(company_dict[keys][3][term_list[list_index]][10]) == -1:  # 2번째
                    print("Error")

                try:
                    # 각 회사의 1 / PER 을 넣는다 --> 인덱스 10
                    temp_list.append(
                        1 / float(company_dict[keys][3][term_list[list_index]][10])
                    )
                except Exception:  # PER 없는 회사
                    pass
        
        temp_np = np.array(temp_list)
        mean = np.mean(temp_np)
        std = np.std(temp_np)
        
        return mean, std
        pass  # def per_meanstd의 끝

    def week_52_beta_with_market_index(self):

        pass

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
                if type(data[0]) == list or type(data[0]) == np.ndarray:
                    for each_data in data:
                        object_dict[each_data[0]] = each_data
                else:  # kosis_dict의 경우
                    object_dict[data[0]] = data
            except FileNotFoundError:
                print("{} - {} 항목 없음, 건너뜀".format(year, month))
    # return object_dict
    pass  # months_loading의 끝


def loading_data():
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

    return company_pickle, KOSPI_dict, KOSDAQ_dict, bond_dict, vkospi_dict, kosis_dict


if __name__=="__main__":

    pass