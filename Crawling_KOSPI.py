# how to name 'variables' :
# name_name2_category
# ex. company_code__List = []

import csv
import datetime
import json
import os
import time

from datetime import datetime as dt
from dateutil.relativedelta import relativedelta
from io import BytesIO

import aiohttp
import asyncio
import asyncssh
import numpy as np
import pandas as pd
import pickle
import requests

from bs4 import BeautifulSoup



def company_list():
    company_code_List = []
    with open('Company_Code.csv', 'r', encoding='UTF-8') as codefile:
        csvCode = csv.reader(codefile)
        for row in csvCode:
            company_code_List.append([row[1].zfill(6), row[2]])

    return company_code_List


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
    pass

# 예상 data 구조 :
# company_dict = { company name : [info_list, price_data_dict, financial_data_year_dict, financial_data_quarter_dict]}

# company_list -> crawling machines 수대로 자름 -> 
# info_crawling, price_crawling, financial_crawling은 분산 크롤링 가능
# market_crawling, bond_crawling, vkospi_crawling, kosis_crawling은 분산 크롤링 불가
# info crawling에 재무정보, 가격정보 빼는 것 고려

async def async_get(url, headers, limits=10):
    connector_limit = aiohttp.TCPConnector(limit = limits)
    async with aiohttp.ClientSession(connector=connector_limit) as session:
        async with session.get(url, headers=headers) as res:
            html = await res.text()
    
    print("Crawling to {}".format(url))
    return html


async def info_crawling_async(company_code, company_name, limits=10):
    
    info_list = ["" for i in range(23)]
    info_list[0] = company_code
    info_list[1] = company_name

    url_0 = "https://finance.naver.com/item/main.nhn?code=" + company_code
    url_1 = "https://navercomp.wisereport.co.kr/v2/company/c1010001.aspx?cmp_cd=" + company_code
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.113 Safari/537.36'}
    htmls = await asyncio.gather(async_get(url_0, headers, limits=limits), async_get(url_1, headers, limits=limits))
    await asyncio.sleep(2)

    try:  # 페이지 로딩
        html = htmls[0]
        soup = BeautifulSoup(html, 'html.parser')
        info_list[2] = [item.get_text().strip() for item in soup.select \
            ("div.rate_info div.today span.blind")][0].replace(",", "")  # 가격 정보
        info_list[3] = [item.get_text().strip() for item in soup.select("div.first tr td")] \
            [2].replace(",", "").replace("\n", "").replace("1", "")

        if info_list[2] == "" or info_list[2].isdecimal() == False:
            info_list[2] = None
        if info_list[3] == "" or info_list[3].isdecimal() == False:
            info_list[3] = None

        try:
            info_list[4] = [item.get_text().strip().replace(",", "")
                            for item in soup.select("table.per_table #_per")][0]
            if info_list[4] == []:
                info_list[4] = [item.get_text().strip().replace(",", "")
                                for item in soup.select("table.per_table #_cns_per")][0]
            info_list[5] = [item.get_text().strip().replace(",", "")
                            for item in soup.select("table.per_table #_eps")][0]
            info_list[6] = [item.get_text().strip().replace(",", "")
                            for item in soup.select("table.per_table #_pbr")][0]
            info_list[7] = round(float(info_list[2]) / float(info_list[6].replace(",", "")))

        except Exception as e:
            print("PER PBR 크롤링 오류 - 종목 : " + company_name + " / Code " + company_code)
            print(e)
            info_list[4] = None
            info_list[5] = None
            info_list[6] = None
            info_list[7] = None

        price52minmax = [item.get_text().strip('\n').strip('\t').split() for item in soup.select("div.tab_con1 div table")]
        try:  # 52주 최고최저
            price52max = int(price52minmax[2].index("최고")) + 2
            price52min = int(price52minmax[2].index("최저")) + 2
            info_list[8] = int(price52minmax[2][price52max].replace(",", ""))
            info_list[9] = int(price52minmax[2][price52min].replace(",", ""))
            pass

        except ValueError:
            price52max = int(price52minmax[2].index("52주최고l최저")) + 1
            price52min = int(price52minmax[2].index("52주최고l최저")) + 3
            info_list[8] = int(price52minmax[2][price52max].replace(",", ""))
            info_list[9] = int(price52minmax[2][price52min].replace(",", ""))

        except IndexError:
            try:
                price52max = int(price52minmax[0].index("52주최고l최저")) + 1
                price52min = int(price52minmax[0].index("52주최고l최저")) + 3
                info_list[8] = int(price52minmax[0][price52max].replace(",", ""))
                info_list[9] = int(price52minmax[0][price52min].replace(",", ""))

            except Exception as e:
                print("52주 최고 최저가 크롤링 오류 - 종목 : " + company_name + " / Code " + company_code)
                print(e)
                info_list[8], info_list[9] = None
                pass

        try:  # 당기 순이익
            companyprofits = [item.get_text().strip('\n').strip('\t').split() for item in
                            soup.select("div.section div.sub_section table tbody")][2]
            companyprofits_1st = int(companyprofits.index("당기순이익")) + 1
            companyprofits_last = int(companyprofits.index("영업이익률")) - 1

            if companyprofits_last - companyprofits_1st == 9:
                for j in range(10):
                    temp = companyprofits[companyprofits_1st + j].replace(",", "").replace("-", "").replace("'", "")
                    info_list[10 + j] = float(temp) if temp != "" and temp.isnumeric() == True else None
            else:
                for j in range(3):
                    temp = companyprofits[companyprofits_1st + j].replace(",", "").replace("-", "").replace("'", "")
                    info_list[10 + j] = float(temp) if temp != "" and temp.isnumeric() == True else None
                info_list[13] = None
                for j in range(6):
                    temp = companyprofits[companyprofits_1st + j].replace(",", "").replace("-", "").replace("'", "")
                    info_list[14 + j] = float(temp) if temp != "" and temp.isnumeric() == True else None
                info_list[19] = None
        except ValueError:  # 당기 순이익 try
            companyprofits = [item.get_text().strip('\n').strip('\t').split() for item in
                            soup.select("div.section div.sub_section table tbody")][2]
            companyprofits_1st = int(companyprofits.index("당기순이익")) + 1
            companyprofits_last = int(companyprofits.index("영업이익률")) - 1

            if companyprofits_last - companyprofits_1st == 9:
                for j in range(10):
                    temp = companyprofits[companyprofits_1st + j].replace(",", "").replace("-", "").replace("'", "")
                    info_list[10 + j] = float(temp) if str(temp) != "" and temp.isnumeric() == True else None
            else:
                for j in range(3):
                    temp = companyprofits[companyprofits_1st + j].replace(",", "").replace("-", "").replace("'", "")
                    info_list[10 + j] = float(temp) if str(temp) != "" and temp.isnumeric() == True else None
                info_list[13] = None
                for j in range(6):
                    temp = companyprofits[companyprofits_1st + j].replace(",", "").replace("-", "").replace("'", "")
                    info_list[14 + j] = float(temp) if str(temp) != "" and temp.isnumeric() == True else None
                info_list[19] = None
        except IndexError:
            print("영업이익 오류 - 종목 : {} , 코드 : {}".format(company_name, company_code))
            for j in range(10):
                info_list[10 + j] = None

        try:  # KOSPI, KOSDAQ 여부
            kospidaqstr = str(soup.select("div.description img")[0])
            kospidaqstr_find_front = kospidaqstr.find("class") + 7  # "class=kospi" ' 검색에서 "class=를 제외하기 위함
            kospidaqstr_find_end = kospidaqstr.find("height") - 2  # "class=kospi" ' 검색에서 뒤의  '를 제외하기 위함
            market_type = kospidaqstr[kospidaqstr_find_front:kospidaqstr_find_end]
            info_list[20] = market_type
        except Exception as e:
            print(company_code + "종목 코스피/코스닥 여부 크롤링 실패, 확인 필요")
            print(e)
            market_type = 'Error'
            info_list[20] = market_type

    except Exception as e:  # 페이지 로드 오류
        print("페이지 로드 및 크롤링 오류 - 종목 : " + company_name + " / Code " + company_code)
        print(e)

    
    try:  # 52Beta 크롤링
        html2 = htmls[1]
        soup2 = BeautifulSoup(html2, 'html.parser')
        items = [item.get_text().strip().replace("\n", "").replace("\t", "").replace("\r", "") \
                for item in soup2.select("div.body tbody")][0].split("/")
        loc_52w_beta_str = items[8].find("베타")
        loc_52w_pub_str = items[8].find("발행")
        beta_52weeks = items[8][loc_52w_beta_str + 2: loc_52w_pub_str]
        info_list[21] = beta_52weeks

    except Exception as e:
        beta_52weeks = 0
        info_list[21] = beta_52weeks
        print("{} 종목 52 베타 크롤링 오류".format(company_name))
        print(e)

    info_list[22] = str(datetime.datetime.today())[:10].replace('-', '').replace(" ", "")

    return info_list


async def price_crawling_async(company_code, company_name, price_data_dict, update=False, limits=10):
    # price_data_dict : 해당 주식만의 가격 dict를 넣는다
    # 날짜, 종가, 거래량, 기관순매매량, 외인순매매량, 외인보유량, 외인보유율, 0, high/low, tradingvolume +-
    # tradingvol은 -1이 있을 가능성이 있고, 0이 있을 가능성이 적으므로 디폴트를 0으로 한다

    if company_name != "종목명" and len(company_code) == 6 and company_name != "000 8층":
        print("{} 종목 가격 정보 크롤링 중".format(company_name))
        url = "https://finance.naver.com/item/frgn.nhn?code=" + company_code + "&page="
        header = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.113 Safari/537.36'}
        page_num = 1
        page_num_switch = False

        while page_num_switch == False:

            print("{} 페이지 크롤링 중".format(page_num))

            html = await async_get(url + str(page_num), header, limits=limits)
            soup = BeautifulSoup(html, 'html.parser')
            items = [item.get_text().strip('\n').strip('\t').split()
                     for item in soup.select("div.section table tr")]
            end_num = len(items)
            if items != []:
                for i in range(2):
                    for j in range(0, end_num):
                        try:
                            if not items[j]:
                                del items[j]
                        except IndexError:
                            break;

            for i in range(9, len(items) - 1):  # 9번 ~ 29번
                if items[i] != []:  # 데이터 카공 : 컴마 제외 등
                    # 이 부분에 items의 총 길이에 맞게끔 빈 리스트 생성
                    del items[i][2], items[i][2]  # 전일비, 등락률 제거

                    while len(items[i]) < 10:
                        items[i].append(0)

                    for j in range(9):
                        if type(items[i][j]) == str:
                            items[i][j] = items[i][j].replace(",", "")  # 콤마 제거
                    items[i][0] = items[i][0].replace('.', '-')  # 날짜를 ####-##-## 형식으로
                    items[i][6] = items[i][6].replace("%", '')

                    # 데이터형식 변경
                    temp_list = np.zeros(len(items[i]), dtype=object)
                    temp_list[0] = items[i][0]
                    for j in range(1, 6):
                        temp_list[j] = int(items[i][j])
                    temp_list[6] = float(items[i][6])

                    items[i] = temp_list[:]


                    if items[i][0] not in price_data_dict.keys() or update:
                        price_data_dict[items[i][0]] = items[i]
                    else:
                        page_num_switch = True
                        break

                else:  # items[i] == [] :
                    page_num_switch = True
                    break;
            page_num += 1

        # Cal_52HighLow, TradingVol
        # TradingVol : 거래량. 전날 대비 가격이 올랐으면 +, 내렸으면 -
        try:  # Cal_52HighLow, TradingVol 계산을 여기서 처리
            price_data_values = np.array(list(price_data_dict.values()))
            for i in range(len(price_data_values)):
                if price_data_values[i][8] == 0 or price_data_values[i][8] == '0' or update == True:
                    try:
                        if np.min(np.array(price_data_values[i: i + 260][:, 2], dtype=int)) == int(price_data_values[i][2]):  # 52주 최소가 해당 값
                            price_data_values[i][8] = 'low'
                        elif np.max(np.array(price_data_values[i: i + 260][:, 2], dtype=int)) == int(price_data_values[i][2]):  # 52주 최대가 해당 값
                            price_data_values[i][8] = 'high'
                        else:
                            price_data_values[i][8] = '1'
                    except IndexError:
                        price_data_values[i][8] = '1'
                else:  # 이 이후의 값들은 이미 계산되어 있음. update==false임
                    break

                if price_data_values[i][9] == 0 or price_data_values[i][9] =='0' or update == True:
                    try:
                        if int(price_data_values[i][2]) >= int(price_data_values[i + 1][2]):
                            price_data_values[i][9] = int(price_data_values[i][2])
                        else:
                            price_data_values[i][9] = -1 * int(price_data_values[i][2])
                    except IndexError:
                        price_data_values[i][9] = 0
                else:  # 이 이후의 값들은 이미 계산되어 있음. update==false임
                    break

        except Exception as e:
            print("{} 종목 52 최고 - 최저, 거래량 계산 오류")
            print(e)
            pass

        # calculated data update
        for i in range(len(price_data_values)):
            date = price_data_values[i][0]
            if not update:
                if not np.any(price_data_dict[date] == price_data_values[i]) == False:
                    break
            price_data_dict[date] = price_data_values

    return price_data_dict


async def financial_crawling_async(company_code, company_name, financial_data_year_dict, financial_data_quarter_dict, limits=10):
    # financial_data_dict : 해당 주식만의 dictionary를 넣는다
    # 매출액, 영업이익, 당기순이익, 영업이익률, 순이익률, ROE, 부채비율, 당좌비율, 유보율, EPS, PER, BPS, PBR, 주당배당금, Expected Y/N 순

    url = "https://finance.naver.com/item/main.nhn?code="
    print("{} 종목 재무정보 크롤링".format(company_name))
    header = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.113 Safari/537.36'}
    htmls = await async_get(url + company_code, headers=header, limits=limits)
    soup = BeautifulSoup(htmls, 'html.parser')

    try:
        temp_table = soup.select("div.sub_section table tbody")
        temp_table_bottom = temp_table[2].text.replace('\t', ' ').replace('\n', ' ').replace('\xa0', '-1').replace(",",
                                                                                                                   "").split()

        temp_table_thead = soup.select("div.sub_section table thead")
        temp_table_head = temp_table_thead[2].text.replace('\t', ' ').replace('\n', ' ').replace('\xa0', '-1').replace(
            ",", "").split()

        for k in range(len(temp_table_head)):
            if 'IFRS연결' in temp_table_head:
                temp_table_head = temp_table_head[temp_table_head.index('분기') + 2:temp_table_head.index('IFRS연결')]
                break
            if 'IFRS별도' in temp_table_head:
                temp_table_head = temp_table_head[temp_table_head.index('분기') + 2:temp_table_head.index('IFRS별도')]
                break
        for k in range(len(temp_table_head)):
            if 'GAAP' in temp_table_head:
                temp_table_head = temp_table_head[:k]

        # 연도 index --> .12 삭제 필요
        for tp_year_quarter in range(len(temp_table_head)):
            if 'E' in temp_table_head[tp_year_quarter]:
                break
        for i in range(tp_year_quarter + 1):
            temp_table_head[i] = temp_table_head[i][0:temp_table_head[i].find(".")] + temp_table_head[i][
                                                                                      temp_table_head[i].find(".") + 3:]

        financial_info_table_temp = [
            temp_table_bottom[temp_table_bottom.index("매출액") + 1: temp_table_bottom.index('영업이익')],
            temp_table_bottom[temp_table_bottom.index("영업이익") + 1: temp_table_bottom.index('당기순이익')],
            temp_table_bottom[temp_table_bottom.index("당기순이익") + 1: temp_table_bottom.index('영업이익률')],
            temp_table_bottom[temp_table_bottom.index("영업이익률") + 1: temp_table_bottom.index('순이익률')],
            temp_table_bottom[temp_table_bottom.index("순이익률") + 1: temp_table_bottom.index('ROE(지배주주)')],
            temp_table_bottom[temp_table_bottom.index("ROE(지배주주)") + 1: temp_table_bottom.index('부채비율')],
            temp_table_bottom[temp_table_bottom.index("부채비율") + 1: temp_table_bottom.index('당좌비율')],
            temp_table_bottom[temp_table_bottom.index("당좌비율") + 1: temp_table_bottom.index('유보율')],
            temp_table_bottom[temp_table_bottom.index("유보율") + 1: temp_table_bottom.index('EPS(원)')],
            temp_table_bottom[temp_table_bottom.index("EPS(원)") + 1: temp_table_bottom.index('PER(배)')],
            temp_table_bottom[temp_table_bottom.index("PER(배)") + 1: temp_table_bottom.index('BPS(원)')],
            temp_table_bottom[temp_table_bottom.index("BPS(원)") + 1: temp_table_bottom.index('PBR(배)')],
            temp_table_bottom[temp_table_bottom.index("PBR(배)") + 1: temp_table_bottom.index('주당배당금(원)')]]

        financial_info_table_np = np.array(financial_info_table_temp).T

        for i in range(len(temp_table_head)):
            if '(E)' in temp_table_head[i]:
                np.append(financial_info_table_np[i], ["Y"], axis=0)
            else:
                np.append(financial_info_table_np[i], ["N"], axis=0)

        for i in range(len(temp_table_head)):
            if '(E)' in temp_table_head[i]:
                break

        for j in range(len(temp_table_head)):
            if j <= i:
                financial_data_year_dict[temp_table_head[j].replace("(E)", "").replace(".12", "")] = \
                    financial_info_table_np[j]
            else:  # j > i:
                financial_data_quarter_dict[temp_table_head[j].replace("(E)", "")] = financial_info_table_np[j]

    except Exception as e:
        print("{} 종목 재무정보 크롤링 실패".format(company_name))
        print(e)

    return financial_data_year_dict, financial_data_quarter_dict


async def company_crawling_async(company_list, company_dict, update=False, limits=10):

    loop = asyncio.get_event_loop()

    for companies in company_list:
        
        company_code = companies[0]
        company_name = companies[1]

        if company_name not in company_dict.keys():
            company_dict[company_name] = [[], {}, {}, {}]

        try:
            price_dict = company_dict[company_name][1]
        except Exception:
            price_dict = {}
        
        try:
            financial_year_dict = company_dict[company_name][2]
            financial_quarter_dict = company_dict[company_name][3]
        except Exception:
            financial_year_dict = {}
            financial_quarter_dict = {}

        info_crawl = loop.create_task(info_crawling_async(company_code, company_name, limits=limits))
        price_crawl = loop.create_task(price_crawling_async(company_code, company_name, price_dict, update=update, limits=limits))
        financial_crawl = loop.create_task(financial_crawling_async(company_code, company_name, financial_year_dict, financial_quarter_dict, limits=limits))

        company_dict[company_name][0] = await info_crawl
        company_dict[company_name][1] = await price_crawl
        company_dict[company_name][2], company_dict[company_name][3] = await financial_crawl

    return company_dict

# price_crawling 과 거의 같은 코드를 통해 크롤링
# kospi, kosdaq 코드를 하나로 합치고 이름을 market으로 변경
# kospi highlow, kosdaq highlow, tradingvol_plusminus를 함께 처리. 
# 기존 load -> collect -> updateDatabase에서 load와 update를 제거
# market_crawling으로 전부 처리 (price에서 cal52, tradingvol을 함께 처리했으니)

def bond_crawling(bond_dict, time_sleep = 1):
    # bond_dict의 구조 : 
    # bond_dict = {bond_3m_dict, bond_6m_dict, bond_9m_dict, bond_1y_dict, bond_1y6m_dict, bond_2y_dict, bond_3y_dict, bond_5y_dict}
    # bond_NX_dict = {'2021-05-21' : ...}

    '''
    초기화
    bond_dict = {'3m' : {}, '6m' : {}, '9m' : {}, '1y' : {}, '1y6m' : {}, '2y' : {}, '3y' : {}, '5y' : {}}
    '''

    url = "https://www.kisrating.com/ratingsStatistics/statics_spreadExcel.do"
    date_loop_switch = True
    date = datetime.datetime.now()
    date_ymd = str(date.year) + '.' + str(date.month).zfill(2) + '.' + str(date.day).zfill(2)

    while date_loop_switch:
        print("{} 날짜 채권 정보 크롤링 중".format(date_ymd.replace('.', '-')))
        header = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.113 Safari/537.36'}
        try:
            request = requests.post(url, data={'startDt': date_ymd}, headers=header)
        except (TimeoutError, requests.exceptions.ConnectionError):
            print("Request Error 발생, 크롤링 중단")
            date_loop_switch = False
            pass
        df_pd = pd.read_excel(BytesIO(request.content))
        df_np = np.array(df_pd)

        try:  # 자료가 있는 date 라면 try 성공
            df_date = df_np[14][0][-11:-1].replace('.', '-')

            try:
                # 키 값(date) 있음 --> 크롤링 중단
                print("{} 날짜 값 {} 존재, 크롤링 중단".format(str(df_date), bond_dict['3m'][df_date]))
                date_loop_switch = False
                break

            except KeyError:
                for i in range(len(list(bond_dict.keys()))):
                    bond_dict[list(bond_dict.keys())[i]][df_date] = np.concatenate(([df_date], df_np[:, i + 1][1:12]))


        except IndexError:  # 자료가 없는 date(일요일 등)는 넘김
            print("{} 날짜 채권 자료 없음".format(date_ymd))

        finally:
            date -= datetime.timedelta(days=1)
            date_ymd = str(date.year) + '.' + str(date.month).zfill(2) + '.' + str(date.day).zfill(2)

        time.sleep(time_sleep)

    return bond_dict


def vkospi_crawling(vkospi_dict):
    # 일자, 종가, 대비, 등락률, 시가, 고가, 저가

    try:
        # 날짜 정보가 존재할 경우
        vkospi_dictkey_list = list(vkospi_dict.keys())
        from_date = str(vkospi_dict[vkospi_dictkey_list[0]][0][0]).replace("/", "").replace("-", "")
        today = datetime.datetime.now()
        to_date = str(today.year) + str(today.month).zfill(2) + str(today.day).zfill(2)

    except IndexError:
        # 맨 처음에 크롤링할 경우 : 날짜 정보가 하나도 없음
        from_date = '20000601'
        today = datetime.datetime.now()
        to_date = str(today.year) + str(today.month).zfill(2) + str(today.day).zfill(2)

    url_1 = "http://data.krx.co.kr/comm/fileDn/GenerateOTP/generate.cmd"
    form_data_1 = {
        'strtDd': from_date,
        'endDd': to_date,
        'tboxidxCd_finder_drvetcidx0_2': '%EC%BD%94%EC%8A%A4%ED%94%BC+200+%EB%B3%80%EB%8F%99%EC%84%B1%EC%A7%80%EC%88%98',
        'codeNmidxCd_finder_drvetcidx0_2': '%EC%BD%94%EC%8A%A4%ED%94%BC+200+%EB%B3%80%EB%8F%99%EC%84%B1%EC%A7%80%EC%88%98',
        'param1idxCd_finder_drvetcidx0_2': '',
        'indTpCd': '1',
        'idxIndCd': '300',
        'idxCd': '1',
        'idxCd2': '300',
        'url': 'dbms/MDC/STAT/standard/MDCSTAT01201',
        'csvxls_isNo': 'false',
        'name': 'fileDown'
    }
    headers_1 = {
        'Accept': 'text/plain, */*; q=0.01',
        'Accept-Encoding': 'gzip, deflate',
        'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
        'Connection': 'keep-alive',
        'Content-Length': '403',
        'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
        'Cookie': '__smVisitorID=Vua__f9O8_P; JSESSIONID=0SX3nsD9iTFGg6aJGRYbGmou1b1yvz6ClDE8y1nJmgxEcTFtJHbStgfBnSH3x6RY.bWRjX2RvbWFpbi9tZGNvd2FwMi1tZGNhcHAwMQ==; finder_drvetcidx_finderCd=finder_drvetcidx; finder_drvetcidx_tbox=%EC%BD%94%EC%8A%A4%ED%94%BC%20200%20%EB%B3%80%EB%8F%99%EC%84%B1%EC%A7%80%EC%88%98; finder_drvetcidx_codeNm=%EC%BD%94%EC%8A%A4%ED%94%BC%20200%20%EB%B3%80%EB%8F%99%EC%84%B1%EC%A7%80%EC%88%98; finder_drvetcidx_codeVal=1; finder_drvetcidx_codeVal2=300',
        'DNT': '1',
        'Host': 'data.krx.co.kr',
        'Origin': 'http://data.krx.co.kr',
        'Referer': 'http://data.krx.co.kr/contents/MDC/MDI/mdiLoader/index.cmd?menuId=MDC0201010305',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.113 Safari/537.36',
        'X-Requested-With': 'XMLHttpRequest'
    }
    request_1 = requests.post(url_1, data=form_data_1, headers=headers_1)

    url_2 = "http://data.krx.co.kr/comm/fileDn/download_excel/download.cmd"
    form_data_2 = {
        'code': request_1.content,
    }
    headers_2 = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.113 Safari/537.36'}
    request_2 = requests.post(url_2, data=form_data_2, headers=headers_2)

    toread = BytesIO()
    toread.write(request_2.content)
    toread.seek(0)
    df_vkospi = pd.read_excel(toread)

    for i in range(len(df_vkospi)):
        date = df_vkospi['일자'][i].replace("/", "-")
        vkospi_dict[date] = np.ravel(np.array(df_vkospi.loc[[i]]))
        vkospi_dict[date][0] = vkospi_dict[date][0].replace("/", "-")

    return vkospi_dict


def kosis_crawling(kosis_dict):  # 경기동반지수 등 통계청 관련 크롤링
    # 월, 선행종합지수. 선행종합지수 순환변동치, 동행지수, 동행지수 순환변동치, 후행종합지수
    date = datetime.datetime.now()
    yearmonth = str(date.year) + str(date.month).zfill(2)
    delta = relativedelta(months=1)

    crawling_switch = True

    while crawling_switch:
        try:  # 데이터가 없다면 errMsg : "데이터가 존재하지 않습니다." 가 뜬다. Try / Except로 다음 월로 넘겨야 함
            print("{} 데이터 크롤링".format(yearmonth[:4] + '-' + yearmonth[4:]))

            url = "http://kosis.kr/openapi/statisticsData.do?method=getList&apiKey=NmIxMTdkYjkyNjgyYTQ0ZDY0ZGI5YTFhNGFlOWFkNTk=&format=json&jsonVD=Y&userStatsId=hanyki111/101/DT_1C8013/2/1/20200608160652&prdSe=" \
                  "M&startPrdDe=" + yearmonth + "&endPrdDe=" + yearmonth
            header = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.113 Safari/537.36'}
            request = requests.get(url, headers=header)
            json_data = json.load(BytesIO(request.content))
            result_list = [yearmonth[:4] + '-' + yearmonth[4:]]
            for i in range(len(json_data)):
                result_list.append(json_data[i]['DT'])

            if yearmonth[:4] + '-' + yearmonth[4:] in kosis_dict.keys() and \
                    list(kosis_dict[yearmonth[:4] + '-' + yearmonth[4:]]) == result_list:  # 데이터 있음
                break
            elif yearmonth[:4] == '2014':
                break
            else:
                kosis_dict[yearmonth[:4] + '-' + yearmonth[4:]] = result_list
                date -= delta
                yearmonth = str(date.year) + str(date.month).zfill(2)

        except TypeError:  # 현재 연월 Error
            date -= delta
            yearmonth = str(date.year) + str(date.month).zfill(2)

        except json.decoder.JSONDecodeError:
            date -= delta
            yearmonth = str(date.year) + str(date.month).zfill(2)

    return kosis_dict


async def dispersion_crawling_data(company_list, company_dict, update=None, limits=None):
    active_machines_dict = {}

    # nickname, address, id, password, port
    with open('active_machines.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for line in reader:
            active_machines_dict[line[0]] = {'host': line[1],
                                             'username': line[2], 'password': line[3], 'port': int(line[4])}

    machines = list(active_machines_dict.keys())
    print(machines)
    company_number_divided_len = round(len(company_list) / len(machines))
    company_divided_list = [0 for items in machines]

    # 리스트를 쪼개서 피클로 저장     
    # 각 machines가 행할 행동을 tasks에 지정

    loop = asyncio.get_event_loop()
    tasks = []
    for i in range(len(machines)):

        company_divided_list[i] = company_list[i * company_number_divided_len: (i + 1) * company_number_divided_len]
        company_divided_dict = {}

        # company_divided_pickle 
        for companies in company_divided_list[i]:
            try:
                company_name = companies[1]
                company_divided_dict[company_name] = company_dict[company_name]
            except KeyError:
                company_dict[company_name] = [[], {}, {}, {}]
                company_divided_dict[company_name] = company_dict[company_name]
                pass
        company_divided_pickle = [company_divided_list[i], company_divided_dict]

        with open("company_divided_pickle_" + str(i), 'wb') as f:
            pickle.dump(company_divided_pickle, f)

        tasks.append(command_machines("company_divided_pickle_" + str(i), **active_machines_dict[machines[i]], update=update, limits=limits))

    # 각 machines가 행동을 하도록 명령 (=크롤링)
    # for task in tasks:
    #     asyncio.gather(task)
    await asyncio.gather(*tasks)

    print("각 머신들에게 크롤링 명령 완료")

    # 각 machines 에서 크롤링 결과물 파일을 전송받음
    loop = asyncio.get_event_loop()
    tasks = []
    for i in range(len(machines)):
        tasks.append(loop.create_task(getfiles_machiens(**active_machines_dict[machines[i]])))

    for task in tasks:
        print(task)
        await task

    print("각 머신들에게 크롤링 파일 받기기 완료")
   # 전송받은 파일들을 하나로 합침

    file_list = os.listdir(os.getcwd())
    company_list = list()
    company_dict = {}
    for file in file_list:
        if file.startswith('company_divided_pickle'):
            with open(file, 'rb') as f:
                temp_dict = pickle.load(f)
            company_list = company_list + temp_dict[0]
            company_dict.update(temp_dict[1])
            if os.path.isfile(file):
                os.remove(file)

    company_pickle = [company_list, company_dict]
    with open('company_pickle', 'wb') as f:
        pickle.dump(company_pickle, f)

    return company_pickle


async def command_machines(crawling_kospi_dispersion_file, host, port, username, password, update=None, limits=None, update_py=True):
    # 1. asyncssh.connect to the machine
    async with asyncssh.connect(host, port=port, username=username, password=password) as conn:
        # 2. company_list_pickle file send
        async with conn.start_sftp_client() as sftp:
            # await sftp.put(#local path, remote path, file)
            if update_py:
                await sftp.put("Crawling_KOSPI_Dispersion.py")
            await sftp.put(crawling_kospi_dispersion_file)
            # 3. parallel command send
        print("Host : {} Command Crawling".format(host))
        await conn.run('python Crawling_KOSPI_Dispersion.py ' + str(update) + ' ' + str(limits), check=True)

async def getfiles_machiens(host, port, username, password):
    async with asyncssh.connect(host, port=port, username=username, password=password) as conn:
        print("Host : {} Get Files".format(host))
        file_text = await conn.run("ls -a", check=True)
        file_list = file_text.stdout.split('\n')
        for file in file_list:
            if file.startswith("company_divided_pickle"):
                filename = file
        try:
            async with conn.start_sftp_client() as sftp:
                await sftp.get(filename)
            await conn.run('rm ' + filename, check=True)
        except UnboundLocalError:
            print("company_divided_pickle 없음")
            pass


# KOSPI 크롤링
async def KOSPI_crawling(KOSPI_data_dict, update=False):

    # 날짜, 체결가, (전일비, 등락률)--> 이것들 제외, 거래량(천주), 거래대금(백만)
    # 4에 (자체계산) high, 5에 low, 6에 (자체계산) 가격 증가한 거래량, 7에 (자체계산) 가격 감소한 거래량
    # 4 ~ 7 자체계산은 KOSPI_KOSDAQ_calculate에서 계산

    # 기존거는 전일비,등락률이 없음
    # 전일비, 등락률을 빼자

    URL = "https://finance.naver.com/sise/sise_index_day.nhn?code=KOSPI&page="
    header = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.113 Safari/537.36'}
    page_num = 1
    page_num_switch = False
    print("KOSPI 정보 크롤링")

    while page_num_switch == False:
        print("KOSPI {} 페이지 크롤링".format(page_num))
        URL_text = URL + str(page_num)
        html = await async_get(URL_text, header, limits=1)
        soup = BeautifulSoup(html, 'html.parser')
        items = [item.get_text().strip('\n').strip('\t').split() for item in soup.select("div table tr")]
        end_num = len(items)
        # 빈칸 삭제 : 왠지 2번 해야 사라짐
        for j in range(2):
            for i in range(0, end_num, 1):
                try:
                    if not items[i]:
                        del items[i]

                except IndexError:
                    break;
        del items[4]  # 크롤링 시 왠지 모르게 4번에서 항목 중복이 발생

        for j in range(1, len(items) - 1, 1):  # 1번 ~ 7번

            # 데이터 가공 : 컴마 제외 등
            if items[j] != []:
                items[j][0] = items[j][0].replace(".", "-")  # 날짜 형식 2021-01-01 형식
                items[j][1] = items[j][1].replace(",", "")  # 종가 , 제거
                items[j][4] = items[j][4].replace(",", "")  # 거래량 , 제거
                items[j][5] = items[j][5].replace(",", "")  if len(items[j]) > 5 else items[j].append(0)# 거래대금 , 제거
            try:
                if items[j][0] not in KOSPI_data_dict.keys():
                    del items[j][2]  # 전일비 제거
                    del items[j][2]  # 등락률 제거
                    KOSPI_data_dict[items[j][0]] = items[j]
                elif update == False:  # 문법 수정이 뜨긴 하지만 내가 헷갈려서 그냥 이대로 씀
                    page_num_switch = True
                    break
                else:  # items[i] == [] :
                    page_num_switch = True
                    break;
            except IndexError:
                print("KOSPI IndexError 발생으로 해당 지점에서 크롤링 종료")
                page_num_switch = True
                break;
        page_num += 1

    return KOSPI_data_dict
    pass # async def KOSPI_crawling의 끝


# KOSDAQ 크롤링
async def KOSDAQ_crawling(KOSDAQ_data_dict, update=False):
    # 날짜, 체결가, 전일비, 등락률, 거래량(천주), 거래대금(백만)

    URL = "https://finance.naver.com/sise/sise_index_day.nhn?code=KOSDAQ&page="
    header = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.113 Safari/537.36'}
    page_num = 1
    page_num_switch = False
    print("KOSPI 정보 크롤링")

    while page_num_switch == False:
        print("KOSDAQ {} 페이지 크롤링".format(page_num))
        URL_text = URL + str(page_num)
        html = await async_get(URL_text, header, limits=1)
        soup = BeautifulSoup(html, 'html.parser')
        items = [item.get_text().strip('\n').strip('\t').split() for item in soup.select("div table tr")]
        end_num = len(items)
        # 빈칸 삭제 : 왠지 2번 해야 사라짐
        for j in range(2):
            for i in range(0, end_num, 1):
                try:
                    if not items[i]:
                        del items[i]

                except IndexError:
                    break;
        del items[4]  # 크롤링 시 왠지 모르게 4번에서 항목 중복이 발생

        for j in range(1, len(items) - 1, 1):  # 1번 ~ 7번

            # 데이터 가공 : 컴마 제외 등
            if items[j] != []:
                items[j][0] = items[j][0].replace(".", "-")  # 날짜 형식 2021-01-01 형식
                items[j][1] = items[j][1].replace(",", "")  # 종가 , 제거
                items[j][4] = items[j][4].replace(",", "")  # 거래량 , 제거
                items[j][5] = items[j][5].replace(",", "")  if len(items[j]) > 5 else items[j].append(0)# 거래대금 , 제거

            try:
                if items[j][0] not in KOSDAQ_data_dict.keys():
                    KOSDAQ_data_dict[items[j][0]] = items[j]
                elif update == False:  # 문법 수정이 뜨긴 하지만 내가 헷갈려서 그냥 이대로 씀
                    page_num_switch = True
                    break
                else:  # items[i] == [] :
                    page_num_switch = True
                    break;
            except IndexError:
                print("KOSDAQ IndexError 발생으로 해당 지점에서 크롤링 종료")
                page_num_switch = True
                break;
        page_num += 1

    return KOSDAQ_data_dict
    pass # async def KOSDAQ_crawling의 끝


# KOSPI, KOSDAQ, bond, vkospi, kosis 크롤링한다
async def market_crawling(KOSPI_data_dict, KOSDAQ_data_dict, bond_dict, vkospi_dict, kosis_dict, update=False):
    loop = asyncio.get_event_loop()

    kospi_crawl = loop.create_task(KOSPI_crawling(KOSPI_data_dict, update=update))
    kosdaq_crawl = loop.create_task(KOSDAQ_crawling(KOSDAQ_data_dict, update=update))

    KOSPI_data_dict_return = await kospi_crawl
    KOSDAQ_data_dict_return = await kosdaq_crawl

    bond_dict_return = bond_crawling(bond_dict)
    vkospi_dict_return = vkospi_crawling(vkospi_dict)
    kosis_dict_return = kosis_crawling(kosis_dict)

    return KOSPI_data_dict_return, KOSDAQ_data_dict_return, bond_dict_return, vkospi_dict_return, kosis_dict_return

    pass


# KOSPI, KOSDAQ 52 High&Low, Trading Volume 계산
def KOSPI_KOSDAQ_calculate(company_price_dict, KOSPI_data_dict, KOSDAQ_data_dict, indiv_company_update=False, force_update=False):

    dict_list = [KOSPI_data_dict, KOSDAQ_data_dict]

    if force_update:
        for date in KOSPI_data_dict.keys():
            for i in range(4, 8):
                try:
                    KOSPI_data_dict[date][i] = 0
                except IndexError:
                    pass
                try:
                    KOSDAQ_data_dict[date][i] = 0
                except IndexError:
                    pass

    for key in company_price_dict.keys():
        if key != '종목명':

            # print("{} 종목에서 52주 high, low, 거래량 계산하여 KOSPI, KOSDAQ dictionary update 중".format(key))

            market_type = company_price_dict[key][0][20]
            company_name = company_price_dict[key][0][1]

            high_value = 0
            low_value = 0

            price_data_dict = company_price_dict[key][1]

            if indiv_company_update:  # 개별 company_price_dict의 모든 가격정보 업데이트
                # 날짜, 종가, 거래량, 기관순매매량, 외인순매매량, 외인보유량, 외인보유율, 0, high/low, tradingvolume +-
                try:  # Cal_52HighLow, TradingVol 계산

                    price_data_values = np.array(list(price_data_dict.values()), dtype=object)
                    for i in range(len(price_data_values)):

                        # 데이터형식 변경
                        temp_list = np.zeros(len(price_data_values[i]), dtype=object)
                        temp_list[0] = price_data_values[i][0]
                        for j in range(1, 6):
                            temp_list[j] = int(price_data_values[i][j])
                        temp_list[6] = float(price_data_values[i][6])

                        price_data_values[i] = temp_list[:]

                        if indiv_company_update:
                            price_data_values[i][8] = 0
                            price_data_values[i][9] = 0

                        if price_data_values[i][8] == 0 or price_data_values[i][8] == '0':
                            try:
                                if np.min(np.array(price_data_values[i: i + 260][:, 2], dtype=int)) == int(price_data_values[i][2]):  # 52주 최소가 해당 값
                                    price_data_values[i][8] = 'low'
                                elif np.max(np.array(price_data_values[i: i + 260][:, 2], dtype=int)) == int(price_data_values[i][2]):  # 52주 최대가 해당 값
                                    price_data_values[i][8] = 'high'
                                else:
                                    price_data_values[i][8] = '1'  # 52주 최저, 최고 아님. 계산 완료
                            except IndexError:
                                price_data_values[i][8] = '1'
                        else:  # 이 이후의 값들은 이미 계산되어 있음. update==false임
                            break

                        if price_data_values[i][9] == 0 or price_data_values[i][9] == '0':
                            try:
                                if int(price_data_values[i][2]) >= int(price_data_values[i + 1][2]):
                                    price_data_values[i][9] = int(price_data_values[i][2])
                                else:
                                    price_data_values[i][9] = -1 * int(price_data_values[i][2])
                            except IndexError:
                                price_data_values[i][9] = 0
                        else:  # 이 이후의 값들은 이미 계산되어 있음. update==false임
                            break

                        # calculated data update
                        date = price_data_values[i][0]
                        price_data_dict[date] = price_data_values[i]
                        company_price_dict[key][1][date] = price_data_dict[date]

                except Exception as e:
                    print("{} 종목 52 최고 - 최저, 거래량 계산 오류".format(key))
                    print(e)

            for dates in price_data_dict.keys():
                if market_type == 'kospi':
                    dict_num = 0
                else:
                    dict_num = 1

                data_dict = dict_list[dict_num]

                if len(data_dict[dates]) < 8: # high/low, trdvol +, - 가 추가되지 않은 경우
                    for i in range(8 - len(data_dict[dates])):
                        data_dict[dates].append(0)

                if force_update or \
                    data_dict[dates][4] == 0 or data_dict[dates][5] == 0:  # high/low 추가 안 됨
                    if price_data_dict[dates][8] == 'high':
                        data_dict[dates][4] += 1
                    elif price_data_dict[dates][8] == 'low':
                        data_dict[dates][5] += 1

                if force_update or \
                    data_dict[dates][6] == 0 or data_dict[dates][7] == 0:
                    if price_data_dict[dates][9] >= 0:
                        data_dict[dates][6] += price_data_dict[dates][9]
                    else:
                        data_dict[dates][7] += price_data_dict[dates][9]

                print(data_dict[dates], dict_num)

    return KOSPI_data_dict, KOSDAQ_data_dict, company_price_dict
    pass  # KOSPI_KOSDAQ_calculate 의 끝


if __name__ == '__main__':

    # company 로딩
    print("START TIME : {}".format(str(datetime.datetime.now())))
    company_pickle = [None,None]
    with open("Files/code_list", 'rb') as f:
        company_pickle[0] = pickle.load(f)
    with open("Files/company_info", 'rb') as f:
        company_pickle[1] = pickle.load(f)

    months_loading(company_pickle[1], "Stock_Price", "", company_tf=True)

    # KOSPI, KOSDAQ 로딩 크롤링
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

    # 크롤링
    print("각 기계들에 크롤링 명령 시작")
    company_pickle = asyncio.run(dispersion_crawling_data(company_pickle[0], company_pickle[1], limits=20))
    # 회사 기본정보, 연간재무, 분기재무

    # overwrite_tf = True
    overwrite_tf = False

    company_noprice_dict = {}
    for key in company_pickle[1].keys():
        company_noprice_dict[key] = [
            company_pickle[1][key][0],
            {},
            company_pickle[1][key][2],
            company_pickle[1][key][3]
        ]
    with open("Files/company_info", "wb") as f:
        pickle.dump(company_noprice_dict, f)

    for company_name in company_pickle[1].keys():
        print(company_name + "종목 날짜 분리 중")
        if company_name != "종목명":
            object_dict = company_pickle[1][company_name][1]
            split_date(object_dict, "Stock_Price", company_name, overwrite=overwrite_tf)


    KOSPI_dict, KOSDAQ_dict, bond_dict, vkospi_dict, kosis_dict = asyncio.run(market_crawling(
        KOSPI_dict, KOSDAQ_dict, bond_dict, vkospi_dict, kosis_dict
    ))

    # 코스피, 코스닥 계산
    KOSPI_dict, KOSDAQ_dict, company_pickle[1] = KOSPI_KOSDAQ_calculate(company_pickle[1], KOSPI_dict, KOSDAQ_dict,
                                                                        indiv_company_update=True, force_update=True)
    # KOSPI_dict, KOSDAQ_dict, company_pickle[1] = KOSPI_KOSDAQ_calculate(company_pickle[1], KOSPI_dict, KOSDAQ_dict)

    overwrite_tf = False
    # overwrite_tf = True

    with open("Files/code_list", 'wb') as f:
        pickle.dump(company_pickle[0], f)

    print("KOSPI, KOSDAQ, kosis, vkospi, bond dictionary 저장")

    split_date(KOSPI_dict, "Market", "KOSPI", overwrite=overwrite_tf)
    split_date(KOSDAQ_dict, "Market", "KOSDAQ", overwrite=overwrite_tf)
    split_date(kosis_dict, "Market", "kosis", overwrite=overwrite_tf)
    split_date(vkospi_dict, "Market", "vkospi", overwrite=overwrite_tf)
    for keys in bond_dict.keys():
        split_date(bond_dict[keys], "Market", "bond_"+keys, overwrite=overwrite_tf)

    print("모든 크롤링 및 저장 작업 완료")
    print("END TIME : {}".format(str(datetime.datetime.now())))

'''
    with open("Files/KOSPI_dict", 'wb') as f:
        pickle.dump(KOSPI_dict, f)
    with open("Files/KOSDAQ_dict", 'wb') as f:
        pickle.dump(KOSDAQ_dict, f)
    with open("Files/bond_dict", 'wb') as f:
        pickle.dump(bond_dict, f)
    with open("Files/vkospi_dict", 'wb') as f:
        pickle.dump(vkospi_dict, f)
    with open("Files/kosis_dict", 'wb') as f:
        pickle.dump(kosis_dict, f)
    '''

'''
async def command_cmd(host, port, username, password):
    async with asyncssh.connect(host, port=port, username=username, password=password) as conn:
        file_list = await conn.run('ls -a', check=True)
        return file_list.stdout.split("\n")

async def command_cmd(cmd, host, port, username, password):
    async with asyncssh.connect(host, port=port, username=username, password=password) as conn:
        result = await conn.run(cmd, check=True)
        print(result.stdout, end='')

async def run_client():
    async with asyncssh.connect('localhost') as conn:
        async with conn.start_sftp_client() as sftp:
            await sftp.get('example.txt')

async def run_client(host, port, username, password):
    filename = input("put file name : ")
    async with asyncssh.connect(host, port=port, username=username, password=password) as conn:
        async with conn.start_sftp_client() as sftp:
            await sftp.put(filename)

task1 = asyncio.create_task(
    say_after(1, 'hello'))

task2 = asyncio.create_task(
    say_after(2, 'world'))

print(f"started at {time.strftime('%X')}")

# Wait until both tasks are completed (should take
# around 2 seconds.)
await task1
await task2

try:
    asyncio.get_event_loop().run_until_complete(run_client())
except (OSError, asyncssh.Error) as exc:
    sys.exit('SSH connection failed: ' + str(exc))


loop = asyncio.get_event_loop()
for i in range(1):
    task.append(loop.create_task(run_client(**active_machines_dict['redmi5'])))

for task in task:
    await task

'''
'''

    overwrite_tf = True
    KOSPI_dict = {}
    KOSDAQ_dict = {}

    KOSPI_dict = asyncio.run(KOSPI_crawling(KOSPI_dict, update=True))
    KOSDAQ_dict = asyncio.run(KOSDAQ_crawling(KOSDAQ_dict, update=True))

    split_date(KOSPI_dict, "Market", "KOSPI", overwrite=overwrite_tf)
    split_date(KOSDAQ_dict, "Market", "KOSDAQ", overwrite=overwrite_tf)



'''
'''
'''
'''
async def info_crawling_main(companylist, limits=10):

    loop = asyncio.get_event_loop()
    tasks = list()
    results = list()
    cycle = 0

    for companies in companylist:        
        tasks.append(loop.create_task(info_crawling_async(companies[0], companies[1], limits=limits)))
    
    for task in tasks:
        results.append(await task)
        
    return results
    
async def price_crawling_main(companylist, price_data_dict, update=False, limits=10):

    tasks = list()
    loop = asyncio.get_event_loop()
    results = list()
    for companies in companylist:
        tasks.append(loop.create_task(price_crawling_async(companies[0], companies[1], price_data_dict, update=update, limits=limits)))

    for task in tasks:
        results.append(await task)

    return results
'''
'''
def market_crawling(kospi_data_dict, kosdaq_data_dict, company_dict, update=False):
    # company_dict --> 각 회사별 전체 정보가 담긴 dict
    # high, low 가 0은 가능하고 -1은 가능하지 않으므로 디폴트를 -1로 지정한다
    # 날짜, 체결가, 거래량, 거래대금, high, low, tradevol +, tradevol -

    print("KOSPI 크롤링 중")
    url = "https://finance.naver.com/sise/sise_index_day.nhn?code=KOSPI&page="
    header = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.113 Safari/537.36'}
    page_num = 1
    page_num_switch = False

    while page_num_switch == False:

        page = requests.get(url + str(page_num), headers=header)
        html = page.text
        soup = BeautifulSoup(html, 'html.parser')
        items = [item.get_text().strip('\n').strip('\t').split()
                 for item in soup.select("div table tr")]
        end_num = len(items)
        if items != []:
            for i in range(2):
                for j in range(0, end_num):
                    try:
                        if not items[j]:
                            del items[j]
                    except IndexError:
                        break

        if items[4][0] == items[5][0]:
            del items[4]

        for i in range(1, len(items) - 1):  # 1번 ~ 7번
            if items[i] != []:  # 데이터 카공 : 컴마 제외 등
                # 이 부분에 items의 총 길이에 맞게끔 빈 리스트 생성
                while len(items[i]) < 8:
                    items[i].append(-1)

                items[i][0] = items[i][0].replace('.', '-')  # 날짜를 ####-##-## 형식으로
                items[i][1] = items[i][1].replace(',', '')  # 종가 ,
                items[i][4] = items[i][4].replace(',', '')  # 거리량 , 
                items[i][5] = items[i][5].replace(',', '') if len(items[i]) > 5 else items[i].append(0)  # 거래대금 0 제거

                del items[i][2], items[i][2]  # 전일비, 등락률 제거

                if items[i][0] not in kospi_data_dict.keys():
                    kospi_data_dict[items[i][0]] = items[i]
                elif update == False:
                    page_num_switch = True
                    break
            else:  # items[i] == [] :
                page_num_switch = True
                break;
        page_num += 1

    print("KOSDAQ 크롤링 중")
    url = "https://finance.naver.com/sise/sise_index_day.nhn?code=KOSDAQ&page="
    header = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.113 Safari/537.36'}
    page_num = 1
    page_num_switch = False

    while page_num_switch == False:

        page = requests.get(url + str(page_num), headers=header)
        html = page.text
        soup = BeautifulSoup(html, 'html.parser')
        items = [item.get_text().strip('\n').strip('\t').split()
                 for item in soup.select("div table tr")]
        end_num = len(items)
        if items != []:
            for i in range(2):
                for j in range(0, end_num):
                    try:
                        if not items[j]:
                            del items[j]
                    except IndexError:
                        break

        if items[4][0] == items[5][0]:
            del items[4]

        for i in range(1, len(items) - 1):  # 1번 ~ 7번
            if items[i] != []:  # 데이터 카공 : 컴마 제외 등
                # 이 부분에 items의 총 길이에 맞게끔 빈 리스트 생성
                while len(items[i]) < 10:
                    items[i].append(0)

                items[i][0] = items[i][0].replace('.', '-')  # 날짜를 ####-##-## 형식으로
                items[i][1] = items[i][1].replace(',', '')  # 종가 ,
                items[i][4] = items[i][4].replace(',', '')  # 거리량 , 
                items[i][5] = items[i][5].replace(',', '') if len(items[i]) > 5 else items[i].append(0)  # 거래대금 0 제거

                del items[i][2], items[i][2]  # 전일비, 등락률 제거

                if items[i][0] not in kospi_data_dict.keys():
                    kosdaq_data_dict[items[i][0]] = items[i]
                elif update == False:
                    page_num_switch = True
                    break
            else:  # items[i] == [] :
                page_num_switch = True
                break;
        page_num += 1


    return kospi_data_dict, kosdaq_data_dict
'''

'''
def info_crawling(company_code, company_name):
    # single company name -> return single company basic info
    # company_code_List --> crawling machines -> return pickle -> collecting pickles -> entire list

    url = "https://finance.naver.com/item/main.nhn?code="
    info_list = ["" for i in range(23)]
    # info_list 인덱스
    # 0 : 코드
    # 1 : 이름
    # 2 : 가격
    # 3 : 상장 주식 수
    # 4 : PER
    # 5 : EPS
    # 6 : PBR
    # 7 : BPS
    # 8 : 52최고
    # 9 : 52최저
    # 10 ~ 19 : 당기순이익
    # 20 : 코스피 / 코스닥
    # 21 : 52베타
    # 22 : 업데이트 날짜

    info_list[0] = company_code
    info_list[1] = company_name

    try:  # 페이지 로딩
        print("{} 종목 크롤링".format(company_name))
        page = requests.get(url + company_code)
        html = page.text
        soup = BeautifulSoup(html, 'html.parser')

        info_list[2] = [item.get_text().strip() for item in soup.select \
            ("div.rate_info div.today span.blind")][0].replace(",", "")  # 가격 정보
        info_list[3] = [item.get_text().strip() for item in soup.select("div.first tr td")] \
            [2].replace(",", "").replace("\n", "").replace("1", "")

        if info_list[2] == "" or info_list[2].isdecimal() == False:
            info_list[2] = None
        if info_list[3] == "" or info_list[3].isdecimal() == False:
            info_list[3] = None

            try:
                info_list[4] = [item.get_text().strip().replace(",", "")
                                for item in soup.select("table.per_table #_per")][0]
                if info_list[4] == []:
                    info_list[4] = [item.get_text().strip().replace(",", "")
                                    for item in soup.select("table.per_table #_cns_per")][0]
                info_list[5] = [item.get_text().strip().replace(",", "")
                                for item in soup.select("table.per_table #_eps")][0]
                info_list[6] = [item.get_text().strip().replace(",", "")
                                for item in soup.select("table.per_table #_pbr")][0]
                info_list[7] = round(float(info_list[2]) / float(info_list[6].replace(",", "")))

            except Exception as e:
                print("PER PBR 크롤링 오류 - 종목 : " + company_name + " / Code " + company_code)
                print(e)
                info_list[4] = None
                info_list[5] = None
                info_list[6] = None
                info_list[7] = None

            price52minmax = [item.get_text().strip('\n').strip('\t').split() for item in soup.select("div.tab_con1 div table")]
            try:  # 52주 최고최저
                price52max = int(price52minmax[2].index("최고")) + 2
                price52min = int(price52minmax[2].index("최저")) + 2
                info_list[8] = int(price52minmax[2][price52max].replace(",", ""))
                info_list[9] = int(price52minmax[2][price52min].replace(",", ""))
                pass

            except ValueError:
                price52max = int(price52minmax[2].index("52주최고l최저")) + 1
                price52min = int(price52minmax[2].index("52주최고l최저")) + 3
                info_list[8] = int(price52minmax[2][price52max].replace(",", ""))
                info_list[9] = int(price52minmax[2][price52min].replace(",", ""))

            except IndexError:
                try:
                    price52max = int(price52minmax[0].index("52주최고l최저")) + 1
                    price52min = int(price52minmax[0].index("52주최고l최저")) + 3
                    info_list[8] = int(price52minmax[0][price52max].replace(",", ""))
                    info_list[9] = int(price52minmax[0][price52min].replace(",", ""))

                except Exception as e:
                    print("52주 최고 최저가 크롤링 오류 - 종목 : " + company_name + " / Code " + company_code)
                    print(e)
                    info_list[8], info_list[9] = None
                    pass

            try:  # 당기 순이익
                companyprofits = [item.get_text().strip('\n').strip('\t').split() for item in
                                soup.select("div.section div.sub_section table tbody")][2]
                companyprofits_1st = int(companyprofits.index("당기순이익")) + 1
                companyprofits_last = int(companyprofits.index("영업이익률")) - 1

                if companyprofits_last - companyprofits_1st == 9:
                    for j in range(10):
                        temp = companyprofits[companyprofits_1st + j].replace(",", "").replace("-", "").replace("'", "")
                        info_list[10 + j] = float(temp) if temp != "" and temp.isnumeric() == True else None
                else:
                    for j in range(3):
                        temp = companyprofits[companyprofits_1st + j].replace(",", "").replace("-", "").replace("'", "")
                        info_list[10 + j] = float(temp) if temp != "" and temp.isnumeric() == True else None
                    info_list[13] = None
                    for j in range(6):
                        temp = companyprofits[companyprofits_1st + j].replace(",", "").replace("-", "").replace("'", "")
                        info_list[14 + j] = float(temp) if temp != "" and temp.isnumeric() == True else None
                    info_list[19] = None
            except ValueError:  # 당기 순이익 try
                companyprofits = [item.get_text().strip('\n').strip('\t').split() for item in
                                soup.select("div.section div.sub_section table tbody")][2]
                companyprofits_1st = int(companyprofits.index("당기순이익")) + 1
                companyprofits_last = int(companyprofits.index("영업이익률")) - 1

                if companyprofits_last - companyprofits_1st == 9:
                    for j in range(10):
                        temp = companyprofits[companyprofits_1st + j].replace(",", "").replace("-", "").replace("'", "")
                        info_list[10 + j] = float(temp) if str(temp) != "" and temp.isnumeric() == True else None
                else:
                    for j in range(3):
                        temp = companyprofits[companyprofits_1st + j].replace(",", "").replace("-", "").replace("'", "")
                        info_list[10 + j] = float(temp) if str(temp) != "" and temp.isnumeric() == True else None
                    info_list[13] = None
                    for j in range(6):
                        temp = companyprofits[companyprofits_1st + j].replace(",", "").replace("-", "").replace("'", "")
                        info_list[14 + j] = float(temp) if str(temp) != "" and temp.isnumeric() == True else None
                    info_list[19] = None
            except IndexError:
                print("영업이익 오류 - 종목 : {} , 코드 : {}".format(company_name, company_code))
                for j in range(10):
                    info_list[10 + j] = None

            try:  # KOSPI, KOSDAQ 여부
                kospidaqstr = str(soup.select("div.description img")[0])
                kospidaqstr_find_front = kospidaqstr.find("class") + 7  # "class=kospi" ' 검색에서 "class=를 제외하기 위함
                kospidaqstr_find_end = kospidaqstr.find("height") - 2  # "class=kospi" ' 검색에서 뒤의  '를 제외하기 위함
                market_type = kospidaqstr[kospidaqstr_find_front:kospidaqstr_find_end]
                info_list[20] = market_type
            except Exception as e:
                print(company_code + "종목 코스피/코스닥 여부 크롤링 실패, 확인 필요")
                print(e)
                market_type = 'Error'
                info_list[20] = market_type

    except Exception as e:  # 페이지 로드 오류
        print("페이지 로드 및 크롤링 오류 - 종목 : " + company_name + " / Code " + company_code)
        print(e)

    

    try:  # 52Beta 크롤링
        url = "https://navercomp.wisereport.co.kr/v2/company/c1010001.aspx?cmp_cd="
        page = requests.get(url + company_code)
        html = page.text
        soup = BeautifulSoup(html, 'html.parser')
        items = [item.get_text().strip().replace("\n", "").replace("\t", "").replace("\r", "") \
                 for item in soup.select("div.body tbody")][0].split("/")
        loc_52w_beta_str = items[8].find("베타")
        loc_52w_pub_str = items[8].find("발행")
        beta_52weeks = items[8][loc_52w_beta_str + 2: loc_52w_pub_str]
        info_list[21] = beta_52weeks
    except Exception as e:
        beta_52weeks = 0
        info_list[21] = beta_52weeks
        print("{} 종목 52 베타 크롤링 오류".format(company_name))
        print(e)

    info_list[22] = str(datetime.datetime.today())[:10].replace('-', '').replace(" ", "")

    return info_list


# price_crawling, cal_52highlow, cal_tradingvolme_plusminus
def price_crawling(company_code, company_name, price_data_dict, update=False):
    # price_data_dict : 해당 주식만의 가격 dict를 넣는다
    # 날짜, 종가, 거래량, 기관순매매량, 외인순매매량, 외인보유량, 외인보유율, 0, high/low, tradingvolume +-
    # tradingvol은 -1이 있을 가능성이 있고, 0이 있을 가능성이 적으므로 디폴트를 0으로 한다

    if company_name != "종목명" and len(company_code) == 6 and company_name != "000 8층":
        print("{} 종목 가격 정보 크롤링 중".format(company_name))
        url = "https://finance.naver.com/item/frgn.nhn?code=" + company_code + "&page="
        header = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.113 Safari/537.36'}
        page_num = 1
        page_num_switch = False

        while page_num_switch == False:

            print("{} 페이지 크롤링 중".format(page_num))

            page = requests.get(url + str(page_num), headers=header)
            html = page.text
            soup = BeautifulSoup(html, 'html.parser')
            items = [item.get_text().strip('\n').strip('\t').split()
                     for item in soup.select("div.section table tr")]
            end_num = len(items)
            if items != []:
                for i in range(2):
                    for j in range(0, end_num):
                        try:
                            if not items[j]:
                                del items[j]
                        except IndexError:
                            break;

            for i in range(9, len(items) - 1):  # 9번 ~ 29번
                if items[i] != []:  # 데이터 카공 : 컴마 제외 등
                    # 이 부분에 items의 총 길이에 맞게끔 빈 리스트 생성
                    del items[i][2], items[i][2]  # 전일비, 등락률 제거

                    while len(items[i]) < 10:

                        for j in range(9):
                            items[i][j] = items[i][j].replace(",", "") if type(items[i][j]) == str else items[i][j] # 콤마 제거
                        items[i][0] = items[i][0].replace('.', '-')  # 날짜를 ####-##-## 형식으로
                        items[i][6] = items[i][6].replace("%", '')

                    if items[i][0] not in price_data_dict.keys():
                        price_data_dict[items[i][0]] = items[i]
                    elif update == False:
                        page_num_switch = True
                        break
                else:  # items[i] == [] :
                    page_num_switch = True
                    break;
            page_num += 1

        # Cal_52HighLow, TradingVol
        # TradingVol : 거래량. 전날 대비 가격이 올랐으면 +, 내렸으면 -
        try:  # Cal_52HighLow, TradingVol 계산을 여기서 처리
            price_data_values = np.array(list(price_data_dict.values()))
            for i in range(len(price_data_values)):
                if price_data_values[i][8] == 0 or update == True:
                    try:
                        if np.min(price_data_values[i: i + 260][:, 2]) == price_data_values[i][2]:  # 52주 최소가 해당 값
                            price_data_values[i][8] = 'low'
                        elif np.max(price_data_values[i: i + 260][:, 2]) == price_data_values[i][2]:  # 52주 최대가 해당 값
                            price_data_values[i][8] = 'high'
                        else:
                            price_data_values[i][8] = '1'
                    except IndexError:
                        price_data_values[i][8] = '1'
                else:  # 이 이후의 값들은 이미 계산되어 있음. update==false임
                    break

                if price_data_values[i][9] == 0 or update == True:
                    try:
                        if price_data_values[i][2] >= price_data_values[i + 1][2]:
                            price_data_values[i][9] = price_data_values[i][2]
                        else:
                            price_data_values[i][9] = -1 * price_data_values[i][2]
                    except IndexError:
                        price_data_values[i][9] = 0
                else:  # 이 이후의 값들은 이미 계산되어 있음. update==false임
                    break

        except Exception as e:
            print("{} 종목 52 최고 - 최저, 거래량 계산 오류")
            print(e)
            pass

        # calculated data update
        for i in range(len(price_data_values)):
            date = price_data_values[i][0]
            if update == False and price_data_dict[date] == price_data_values[i]:
                break
            price_data_dict[date] = price_data_values

    return price_data_dict


def financial_crawling(company_code, company_name, financial_data_year_dict, financial_data_quarter_dict):
    # financial_data_dict : 해당 주식만의 dictionary를 넣는다
    # 매출액, 영업이익, 당기순이익, 영업이익률, 순이익률, ROE, 부채비율, 당좌비율, 유보율, EPS, PER, BPS, PBR, 주당배당금, Expected Y/N 순

    url = "https://finance.naver.com/item/main.nhn?code="
    print("{} 종목 재무정보 크롤링".format(company_name))
    header = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.113 Safari/537.36'}
    page = requests.get(url + company_code)
    html = page.text
    soup = BeautifulSoup(html, 'html.parser')

    try:
        temp_table = soup.select("div.sub_section table tbody")
        temp_table_bottom = temp_table[2].text.replace('\t', ' ').replace('\n', ' ').replace('\xa0', '-1').replace(",",
                                                                                                                   "").split()

        temp_table_thead = soup.select("div.sub_section table thead")
        temp_table_head = temp_table_thead[2].text.replace('\t', ' ').replace('\n', ' ').replace('\xa0', '-1').replace(
            ",", "").split()

        for k in range(len(temp_table_head)):
            if 'IFRS연결' in temp_table_head:
                temp_table_head = temp_table_head[temp_table_head.index('분기') + 2:temp_table_head.index('IFRS연결')]
                break
            if 'IFRS별도' in temp_table_head:
                temp_table_head = temp_table_head[temp_table_head.index('분기') + 2:temp_table_head.index('IFRS별도')]
                break
        for k in range(len(temp_table_head)):
            if 'GAAP' in temp_table_head:
                temp_table_head = temp_table_head[:k]

        # 연도 index --> .12 삭제 필요
        for tp_year_quarter in range(len(temp_table_head)):
            if 'E' in temp_table_head[tp_year_quarter]:
                break
        for i in range(tp_year_quarter + 1):
            temp_table_head[i] = temp_table_head[i][0:temp_table_head[i].find(".")] + temp_table_head[i][
                                                                                      temp_table_head[i].find(".") + 3:]

        financial_info_table_temp = [
            temp_table_bottom[temp_table_bottom.index("매출액") + 1: temp_table_bottom.index('영업이익')],
            temp_table_bottom[temp_table_bottom.index("영업이익") + 1: temp_table_bottom.index('당기순이익')],
            temp_table_bottom[temp_table_bottom.index("당기순이익") + 1: temp_table_bottom.index('영업이익률')],
            temp_table_bottom[temp_table_bottom.index("영업이익률") + 1: temp_table_bottom.index('순이익률')],
            temp_table_bottom[temp_table_bottom.index("순이익률") + 1: temp_table_bottom.index('ROE(지배주주)')],
            temp_table_bottom[temp_table_bottom.index("ROE(지배주주)") + 1: temp_table_bottom.index('부채비율')],
            temp_table_bottom[temp_table_bottom.index("부채비율") + 1: temp_table_bottom.index('당좌비율')],
            temp_table_bottom[temp_table_bottom.index("당좌비율") + 1: temp_table_bottom.index('유보율')],
            temp_table_bottom[temp_table_bottom.index("유보율") + 1: temp_table_bottom.index('EPS(원)')],
            temp_table_bottom[temp_table_bottom.index("EPS(원)") + 1: temp_table_bottom.index('PER(배)')],
            temp_table_bottom[temp_table_bottom.index("PER(배)") + 1: temp_table_bottom.index('BPS(원)')],
            temp_table_bottom[temp_table_bottom.index("BPS(원)") + 1: temp_table_bottom.index('PBR(배)')],
            temp_table_bottom[temp_table_bottom.index("PBR(배)") + 1: temp_table_bottom.index('주당배당금(원)')]]

        financial_info_table_np = np.array(financial_info_table_temp).T

        for i in range(len(temp_table_head)):
            if '(E)' in temp_table_head[i]:
                np.append(financial_info_table_np[i], ["Y"], axis=0)
            else:
                np.append(financial_info_table_np[i], ["N"], axis=0)

        for i in range(len(temp_table_head)):
            if '(E)' in temp_table_head[i]:
                break

        for j in range(len(temp_table_head)):
            if j <= i:
                financial_data_year_dict[temp_table_head[j].replace("(E)", "").replace(".12", "")] = \
                    financial_info_table_np[j]
            else:  # j > i:
                financial_data_quarter_dict[temp_table_head[j].replace("(E)", "")] = financial_info_table_np[j]

    except Exception as e:
        print("{} 종목 재무정보 크롤링 실패".format(company_name))
        print(e)

    return financial_data_year_dict, financial_data_quarter_dict
'''
