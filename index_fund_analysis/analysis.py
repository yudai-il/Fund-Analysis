import pandas as pd
import numpy as np
import seaborn as sns
sns.set_style("whitegrid")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import date,datetime
plt.rcParams['font.sans-serif'] = ['SimHei']  # display Chinese Character
plt.rcParams['axes.unicode_minus'] = False  # display positive/negative mathematical symbols


net_profit = pd.read_pickle("index_fund/data/net_profit.pkl")
mkt = pd.read_pickle("index_fund/data/mkt.pkl").stack()
top_institutional_holding = pd.read_excel("index_fund/data/2015-2020HLD_Negshr.xlsx")


# 使用整体法对指数进行估值

def calc_index_pe(index):
    index_weights = pd.read_pickle("index_fund/data/%s_指数成分权重.pkl"%index)
    d = net_profit.reindex(index_weights.index).sum(axis=1)
    u = mkt.reindex(index_weights.index)
    d = d.groupby(d.index.get_level_values(0)).sum()
    u = u.groupby(u.index.get_level_values(0)).sum()
    return (u/d)


def calc_index_dividend(index):
    dividend = pd.read_pickle("index_fund/data/dividend_yield_ttm.pkl").stack().rename("dividend")
    index_weights = pd.read_pickle("index_fund/data/%s_指数成分权重.pkl"%index)
    _ = pd.concat([dividend.reindex(index_weights.index),index_weights.rename("weights")],axis=1)
    weighted_dividend = (_['dividend']*_['weights']).dropna()
    results = weighted_dividend.groupby(weighted_dividend.index.get_level_values(0)).sum()
    return results


def index_industry_allocation(index):
    index_weights = pd.read_pickle("index_fund/data/%s_指数成分权重.pkl"%index).rename("weights")
    industry_maps = pd.read_pickle("index_fund/data/shenwan_industry_level_1.pkl").reindex(index_weights.index)

    _ = pd.concat([index_weights,industry_maps],axis=1).dropna()
    results = (_.groupby([_.index.get_level_values(0),"index_name"])['weights'].sum()).unstack()

    return results


def calc_index_pb(index):

    index_weights = pd.read_pickle("index_fund/data/%s_指数成分权重.pkl"%index)
    u = mkt.reindex(index_weights.index)

    d = pd.read_pickle("index_fund/data/equity_parent_company.pkl")

    d = d.stack().reindex(index_weights.index)

    d = d.groupby(d.index.get_level_values(0)).sum()

    u = u.groupby(u.index.get_level_values(0)).sum()

    return u/d




index_lists = ['000905.XSHG',"000300.XSHG","000016.XSHG","399006.XSHE"]

index_names = ["中证500","沪深300","上证50","创业板指"]

il = ['000905',"000300","000016","399006"]


def index_valuation_():

    pe_ = {}
    for index in index_lists:
        print(index)
        index = index[:6]
        pe_[index] = calc_index_pe(index)

    pb_ = {}
    for index in index_lists:
        print(index)
        index = index[:6]
        pb_[index] = calc_index_pb(index)


    dividend_ = {}
    for index in index_lists:
        print(index)
        index = index[:6]
        dividend_[index] = calc_index_dividend(index)

    return pe_,pb_,dividend_


start_date = "2000-01-01"

def index_pe_analysis(pe_):
    # 指数PE 计算
    pe_1 = pd.concat(pe_,axis=1)
    pe_1 = pe_1.loc[pd.Timestamp(start_date).date():].dropna()
    perc_pe = (pe_1-pe_1.min())/(pe_1.max() - pe_1.min())

    _ = ((pe_1-pe_1.min())/(pe_1.max() - pe_1.min()))
    _ = _[il]
    _.index = pd.DatetimeIndex(_.index)
    _ = _.resample("Q").last()
    cmap = sns.cubehelix_palette(start = 1.5, rot = 3, gamma=0.8, as_cmap = True)
    _.index = _.index.astype(str)
    _.columns = index_names
    fig,ax = plt.subplots(figsize=(10,4))
    plt.subplots_adjust(bottom=0.3)
    sns.heatmap(_.T,cmap="RdBu",yticklabels=False)
    ""
    plt.yticks(np.arange(0,4.5,0.5),["","中证500","","沪深300","","上证50","","创业板指",""],rotation=0)

    return fig

def index_pb_analysis(pb_):
    # 指数PB 计算
    pb_1 = pd.concat(pb_,axis=1)
    pb_1 = pb_1.loc[pd.Timestamp(start_date).date():].dropna()
    perc_pb = (pb_1-pb_1.min())/(pb_1.max() - pb_1.min())
    _ = perc_pb
    _ = _[il]
    _.index = pd.DatetimeIndex(_.index)
    _ = _.resample("Q").last()
    cmap = sns.cubehelix_palette(start = 1.5, rot = 3, gamma=0.8, as_cmap = True)
    _.index = _.index.astype(str)
    _.columns = index_names
    fig,ax = plt.subplots(figsize=(10,4))
    plt.subplots_adjust(bottom=0.3)
    sns.heatmap(_.T,cmap="RdBu",yticklabels=False)
    ""
    plt.yticks(np.arange(0,4.5,0.5),["","中证500","","沪深300","","上证50","","创业板指",""],rotation=0)
    return fig

#  指数加权股息率

def index_weighted_dividend_ratio(dividend_):

    dividend_1 = pd.concat(dividend_,axis=1)
    dividend_1 = dividend_1[il]

    fig,ax = plt.subplots(figsize=(6,4))
    dividend_1.plot(ax=ax)
    plt.legend(index_names)
    return fig


# index_price = pd.read_pickle("index_fund/data/index_price_2011.pkl")
#
# fig,ax = plt.subplots(4,1,sharex=True,figsize=(8,6))
# for i,index in enumerate(index_lists):
#     index_price[index].plot(ax=ax[i])
#     ax[i].legend([index_names[i]],loc='upper left')
# plt.xticks(rotation=0)
# fig.tight_layout()


# 指数主要行业配置一览


def index_industry_allocation_plot():

    industry_alloc = {}
    for index in il:
        print(index)
        industry_alloc[index] = index_industry_allocation(index)

    industry_alloc = pd.concat(industry_alloc)

    for i,index in enumerate(il):

        _ = industry_alloc.loc[index]

        _.index = pd.DatetimeIndex(_.index)

        _ = _.resample("Q").apply(lambda x:x.iloc[-1])

        _ = _.sort_values(axis=1,by=_.index[-1],ascending=False)

        n_ = pd.concat([_.iloc[:,:5],_.iloc[:,5:].sum(axis=1).rename("其他")],axis=1).dropna()
        n_ = n_.T
        n_.sort_values(by=n_.columns[-1],inplace=True,ascending=False)

        plt.stackplot(n_.columns,n_)
        plt.legend(n_.index,loc='lower left')
    return industry_alloc


# 指数重仓股中的主要股东


def index_componnets_holders(index):
    index_weights = pd.read_pickle("index_fund/data/%s_指数成分权重.pkl"%index).rename("weights")
    top_holdings = index_weights.groupby(index_weights.index.get_level_values(0)).apply(lambda x:x.sort_values(ascending=False)[:10])
    top_holdings.index = top_holdings.index.droplevel(0)
    top_holdings = top_holdings.unstack()
    top_holdings.index = pd.DatetimeIndex(top_holdings.index)

    top_holdings.columns = top_holdings.columns.str[:6]
    top_holdings = top_holdings.resample("Q").apply(lambda x:x.iloc[-1])
    top_holdings.index = top_holdings.index.astype(str)
    top_holdings = top_holdings.unstack().dropna()

    _ = top_institutional_holding.iloc[2:].set_index(["Stkcd",'Reptdt'])['S0401a']

    _ = _.loc[top_holdings.index]

    _.dropna(inplace=True)

    _ = _.groupby(_.index.get_level_values(1)).value_counts()

    return _


def index_componnets_top_holder():

    inst_holders = {}
    for index in il:
        print(index)
        inst_holders[index] = index_componnets_holders(index)

    start_date = "2019-01-01"
    end_date = "2020-01-01"
    res = {}
    for index in il:
        _ = inst_holders[index].unstack().fillna(0)
        res[index] = _.loc[start_date:end_date].sum().sort_values(ascending=False)[:10].index

    return pd.concat(res)

# fund_holdings = pd.read_pickle("Data/Financial/Funds/fund_holdings.pkl")
#
# holdings = fund_holdings.loc['050009'].loc['2014-06-30']
# holdings = holdings[holdings.holdingsecType == "E"]
#
# asset_allocation = pd.read_pickle("Data/Financial/Funds/fund_asset_allocation.pkl")
#
# asset_allocation.loc['050009'].loc['2014-06-30']

"""
totalAsset           8.51213e+09
netAsset             8.46974e+09
equityMarketValue    7.57136e+09
bondMarketValue              NaN
cashMarketValue      9.39289e+08
otherMarketValue     1.48726e+06
publishDate           2014-07-18
"""
#
# fac1 = pd.read_pickle("index_fund/data/fac1.pkl")
# fac2 = pd.read_pickle("index_fund/data/fac2.pkl")
# fac3 = pd.read_pickle("index_fund/data/fac3.pkl")
# fac4 = pd.read_pickle("index_fund/data/fac4.pkl")
#
#
# fac = pd.concat([fac1.stack(),fac2.stack(),fac3.stack(),fac4.stack()],axis=1)


#
# industry_maps = pd.read_pickle("index_fund/data/shenwan_industry_level_1.pkl")
#
# _ = industry_maps['index_name'].unstack()
# _.columns = _.columns.str[:6]
# industry_maps = _.stack()
#
# cur_industry_maps_ = _.loc[pd.Timestamp("2014-06-30").date()].dropna()
#
# merged_data = pd.merge(holdings,cur_industry_maps_.rename("industry"),left_on="holdingTicker",right_index=True)
#
# merged_data['r'] = merged_data['marketValue']/ merged_data['marketValue'].sum()
#
#

# nav.loc['510330']


# ETF 分析


def price_plot(start_date,index_id,etf_id):


    index_price = pd.read_pickle("index_fund/data/index_price_2011.pkl")[index_id].loc[start_date:]

    nav = pd.read_csv("index_fund/data/%s.csv"%etf_id,encoding='gbk')
    nav = nav.set_index("endDate")['ACCUM_NAV'].loc[start_date:]
    nav.index = pd.DatetimeIndex(nav.index)

    fig,ax = plt.subplots(figsize=(8,4))
    index_price.plot(ax=ax)
    ax.legend(['沪深300'],loc='upper right')
    plt.xlabel("")
    plt.xticks(rotation=0)
    ax = ax.twinx()
    nav.plot(ax=ax,color='tomato')
    ax.grid(False)
    ax.legend(["华泰300ETF"],loc='upper left')

    fig.tight_layout()
    plt.show()


def returns_analysis_plot(price):
    from performance_analysis.plot import *

    plot_monthly_returns_heatmap(price)
    plot_annual_returns(price)

"2019-01-01"

from performance_analysis.stats import *


def tracking_error_plot(price,index_price):
    returns = price.pct_change()[1:]
    index_r = index_price.pct_change()[1:]
    # days = [22,44,66,126,252,252*2,252*3]

    # _res = {d:returns_analysis(returns.iloc[-d:]) for d in days}

    _ = pd.concat([returns.reindex(index_r.index).rename("r"),index_r.rename("index")],axis=1)[1:]

    tr = _.groupby(_.index.year).apply(lambda x:tracking_error(x['r'],x['index']))

    fig,ax = plt.subplots()
    ax.axvline(
        100*tr.values.mean(),
        color='tomato',
        linestyle='--',
        lw=4,
        alpha=0.7)
    (100*tr.sort_index(ascending=False)).plot(ax=ax, kind='barh', alpha=0.70)
    ax.axvline(0.002*100, color='black', linestyle='-', lw=3)
    ax.set_ylabel("")

    return fig


def etf_discount_(etf_id):

    nav0 = pd.read_csv("index_fund/data/%s.csv"%etf_id,encoding='gbk')
    nav0 = nav0.set_index("endDate")['NAV'].loc[start_date:]
    nav0.index = pd.DatetimeIndex(nav0.index)

    sec_price = pd.read_pickle("index_fund/data/%s_price.pkl"%etf_id)['close']


    discount_r = nav0/sec_price.reindex(nav0.index)-1

    fig,ax = plt.subplots(figsize=(8,4))
    discount_r.plot()
    plt.xticks(rotation=0)
    plt.xlabel("")
    plt.legend(["折价率"])
    return discount_r,fig




# alloc = pd.read_pickle("Data/Financial/Funds/fund_asset_allocation.pkl")
# fund_investment = pd.read_pickle("Data/Financial/Funds/fund_investor_structure.pkl")
#
# hds = pd.read_pickle("Data/Financial/Funds/fund_holdings.pkl")
#
#
# nav = pd.read_csv("index_fund/data/001015.csv",encoding='gbk')
#
#
# nav = nav.set_index("endDate")['ACCUM_NAV'].loc[start_date:]
# nav.index = pd.DatetimeIndex(nav.index)
#
# # r = nav.pct_change()
#
# (r.loc["2020-01-01":"2020-04-01"] - index_r.loc["2020-01-01":"2020-04-01"])
#
# (r.loc["2020-01-01":"2020-04-01"]+1).prod()-1
#
# (index_r.loc["2020-01-01":"2020-04-01"]+1).prod()-1
#


# fig,ax = plt.subplots(figsize=(8,4))
# index_price.reindex(nav.index).plot(ax=ax)
# ax.legend(['沪深300'],loc='upper right')
# plt.xlabel("")
# plt.xticks(rotation=0)
# ax = ax.twinx()
# nav.plot(ax=ax,color='tomato')
# ax.grid(False)
# ax.legend(["华夏300增强"],loc='upper left')
#
# fig.tight_layout()
# plt.show()


def assets_allocation_analysis(fund_id):

    alloc = pd.read_pickle("Data/Financial/Funds/fund_asset_allocation.pkl")

    alloc = alloc.loc[fund_id]

    r = alloc[['bondMarketValue','cashMarketValue','otherMarketValue','equityMarketValue']].div(alloc['totalAsset'],axis=0)

    r.columns = ["债券",'现金','其他',"权益"]
    r.fillna(0,inplace=True)

    r.index = pd.DatetimeIndex(r.index)

    fig,ax = plt.subplots(figsize=(8,4))
    plt.subplots_adjust(bottom=0.2)
    plt.stackplot(r.index,r.T,alpha=0.8,colors=['tomato','grey','orange','steelblue'])
    plt.legend(r.columns.tolist())
    plt.xticks(rotation=90)

    return fig


def holders_structure_(fund_id):

    fund_investment = pd.read_pickle("Data/Financial/Funds/fund_investor_structure.pkl")

    holdInfo = fund_investment.set_index("ticker").loc[fund_id].set_index("reportDate")[['instHoldRatio', 'indiHoldRatio']]

    holdInfo = holdInfo.astype(np.float)
    holdInfo.index = pd.DatetimeIndex(holdInfo.index)

    holdInfo = holdInfo.sort_index()

    fig,ax = plt.subplots(figsize=(8,4))
    plt.subplots_adjust(bottom=0.2)
    plt.stackplot(holdInfo.index,holdInfo.T,alpha=0.8,colors=['orange','steelblue'])
    plt.legend(["机构持有比率","个人持有比率"])
    plt.xticks(rotation=0)
    return fig


