import aiohttp
import asyncio
import datetime
import pickle
from glob import glob

import sys

import numpy as np
import requests
from bs4 import BeautifulSoup


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

    except Exception as e:  # 페이지 로드 오류
        print("페이지 로드 및 크롤링 오류 - 종목 : " + company_name + " / Code " + company_code)
        print(e)

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
                        items[i].append(0)

                    for j in range(9):
                        items[i][j] = items[i][j].replace(",", "")  # 콤마 제거
                    items[i][0] = items[i][0].replace('.', '-')  # 날짜를 ####-##-## 형식으로
                    items[i][3] = items[i][3].replace("%", '')
                    items[i][8] = items[i][8].replace("%", '')

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


# price_crawling, cal_52highlow, cal_tradingvolme_plusminus
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
        last_crawling_date = ''

        while page_num_switch == False:

            print("{} 페이지 크롤링 중".format(page_num))

            html = await async_get(url + str(page_num), header, limits=limits)
            soup = BeautifulSoup(html, 'html.parser')
            items = [item.get_text().strip('\n').strip('\t').split()
                     for item in soup.select("div.section table tr")]
            end_num = len(items)

            if items != []:  # 빈 라인 삭제
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

                    if items[i][0] == last_crawling_date:
                        page_num_switch = True
                        break

                    if items[i][0] not in price_data_dict.keys() or update:
                        price_data_dict[items[i][0]] = items[i]
                    else:
                        page_num_switch = True
                        break

                else:  # items[i] == [] :
                    page_num_switch = True
                    break;

            if items[9] != []:
                last_crawling_date = items[9][0].replace('.', '-')
            else:
                page_num_switch = True
                break;
            page_num += 1

        # Cal_52HighLow, TradingVol
        # TradingVol : 거래량. 전날 대비 가격이 올랐으면 +, 내렸으면 -
        # try:  # Cal_52HighLow, TradingVol 계산을 여기서 처리
        #     price_data_values = np.array(list(price_data_dict.values()))
        #     for i in range(len(price_data_values)):
        #         if price_data_values[i][8] == 0 or price_data_values[i][8] == '0' or update == True:
        #             try:
        #                 if np.min(np.array(price_data_values[i: i + 260][:, 2], dtype=int)) == int(price_data_values[i][2]):  # 52주 최소가 해당 값
        #                     price_data_values[i][8] = 'low'
        #                 elif np.max(np.array(price_data_values[i: i + 260][:, 2], dtype=int)) == int(price_data_values[i][2]):  # 52주 최대가 해당 값
        #                     price_data_values[i][8] = 'high'
        #                 else:
        #                     price_data_values[i][8] = '1'
        #             except IndexError:
        #                 price_data_values[i][8] = '1'
        #         else:  # 이 이후의 값들은 이미 계산되어 있음. update==false임
        #             break
        #
        #         if price_data_values[i][9] == 0 or price_data_values[i][9] =='0' or update == True:
        #             try:
        #                 if int(price_data_values[i][2]) >= int(price_data_values[i + 1][2]):
        #                     price_data_values[i][9] = int(price_data_values[i][2])
        #                 else:
        #                     price_data_values[i][9] = -1 * int(price_data_values[i][2])
        #             except IndexError:
        #                 price_data_values[i][9] = 0
        #         else:  # 이 이후의 값들은 이미 계산되어 있음. update==false임
        #             break

        # except Exception as e:
        #     print("{} 종목 52 최고 - 최저, 거래량 계산 오류")
        #     print(e)
        #     pass

        # calculated data update
        # for i in range(len(price_data_values)):
        #     date = price_data_values[i][0]
        #     if not update:
        #         if not np.any(price_data_dict[date] == price_data_values[i]) == False:
        #             break
        #     price_data_dict[price_data_values[i][0]] = price_data_values[i]

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


async def company_crawling_async(company_list, company_dict, filename, start_num=0, update=False, limits=10):

    loop = asyncio.get_event_loop()

    # for companies in company_list:

    for i in range(start_num, len(company_list)):

        companies = company_list[i]
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

        # 중간중간 세이브
        company_pickle = [company_list, company_dict]
        with open(filename, 'wb') as f:
            pickle.dump(company_pickle, f)

        with open("temporarily_save.txt", 'w') as f:
            f.write(str(i))

    return company_dict


def pickle_load():  # 피클 로딩, 파일 삭제, return 리스트 및 딕셔너리
    filename = glob('company_divided_*')[0]  # 한 개만 나오도록 할 것
    with open(filename, 'rb') as f:
        company_divided_pickle = pickle.load(f)

    company_list = company_divided_pickle[0]
    company_dict = company_divided_pickle[1]
    '''
    if os.path.isfile(filename):
        os.remove(filename)
    '''
    return company_list, company_dict
    '''
        with open("News\\" + news_sec_dir + "\\" + date + ".txt", "rb") as f:
            article_index_old = pickle.load(f)
    '''

    pass


if __name__ == '__main__':

    # update=T/F, limits

    try:
        if sys.argv[1].replace('\n', '') == 'update':
            update = True
        else:
            update = False
    except Exception:
        update = False

    try:
        limits = int(sys.argv[2])
    except Exception:
        limits = 5

    # pc에서의 크롤링용
    # update = True

    # 중간 세이브

    start_time = datetime.datetime.now()

    try:
        with open("temporarily_save.txt", 'r') as f:
            start_num = int(f.read())
    except Exception as e:
        print(e)
        start_num = 0

    filename = glob('company_divided_*')[0]  # 한 개만 나오도록 할 것
    company_list, company_dict = pickle_load()    
    company_dict = asyncio.run(company_crawling_async(company_list, company_dict, filename, start_num=start_num, update=update, limits=limits))

    company_pickle = [company_list, company_dict]
    with open("company_pickle", 'wb') as f:
        pickle.dump(company_pickle, f)
    with open(filename, 'wb') as f:
        pickle.dump(company_pickle, f)

    end_time = datetime.datetime.now()

    # 중간 세이브 초기화
    with open("time_record.txt", 'w') as g:
        g.write("Start Time : " + str(start_time) + ", End Time : " + str(end_time))
    with open("temporarily_save.txt", 'w') as f:
        f.write("0")

    print("완료")

