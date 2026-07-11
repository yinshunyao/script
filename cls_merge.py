#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2026/05/27
# @Author  : ysy
# @Email   : xxx@qq.com 
# @Detail  : 
# @Software: PyCharm
cls_merge = \
{
    # 蝽类
    # 蝽
    "chun": {
        "daolvchun": 500,
    },
    # 红蝽
    "hongchun": {
        "tubeibanhongchun": 500,
        "xianbanhongchun": 500,
    },
    # 猎蝽
    "liechun": {
        "huangzuliechun": 500
    },
    # 叶蝉
    "yechan": {
        "daqingyechan": 500,  # 北京比赛需要报出
        "heiweiyechan": 500
    },
    # 蝗虫
    "huangchong": {
        "huangchong": 500,
        "huajinglvwenhuang": 500
    },
    # 蟋蟀
    "xishuai": {        # 北京比赛需要报出
        "xishuai": 500,
        "xiaoguantouxishuai": 500
    },
    # 鳃金龟
    "saijingui": {
        "anheisaijingui": 250,
        "daheisaijingui": 250,
        "malingshusaijingui": 500
    },
    # 花鳃金龟
    "xiaoyunbansaijingui": 500,
    # 丽金龟
    "huangheyilijingui": 1000,   # 北京比赛需要报出,
    # 绿丽金龟
    "lvlijingui": {
        "hongjiaolvlijingui": 500,
        "tonglvyilijingui": 500
    },
    # 犀金龟
    "xijingui": {
        "zhonghuaxiaobianxijingui": 1000
    },
    # 白色螟
    "ming_bai": {
        "bailajuanxuyeming": 200,
        "shuiming": 200,
        "youcaijiaoyeming": 200,
    },
    # ming2
    "doujiayeming": 1000,  # 北京比赛需要报出
    "huangyeming": 500,
    "guajuanming": 500,
    "maimuyeming": 1000,  # 北京比赛需要报出
    "qijiaoming": 500,
    "sibanjuanyeming": 1000,    # 北京比赛需要报出
    "tiancaibaidaiyeming": 500,
    # ming3
    "yumiming": 1000,   # 北京比赛需要报出
    "caodiming": 1000,  # 北京比赛需要报出
    "daozongjuanyeming": 500,
    "huangchizhuiyeyeming": 500,
    "huangyangjuanyeming": 1000,    # 北京比赛需要报出
    "miandajuanyeming": 500,
    "taozhuming": 1000,  # 北京比赛需要报出
    # minge
    "daming": 500,
    "huaming": {
        "erhuaming": 250,
        "sanhuaming": 250,
    },
    # 苔蛾，但像螟
    "taie": {
        "heidiantaie": 200,
        "huangtutaie": 200,
    },
    # 夜蛾
    "baibianqieyee": 500,
    "baitiaoyee": 1000,  # 北京比赛出现
    "bazidilaohu": 1000,  # 北京比赛出现
    "bowenyee": 500,
    "caoditanyee": 1000,
    "fayee": 1000,  # 北京比赛出现
    "ganlanyee": 1000,  # 北京比赛出现
    "huangyee": 500,
    "jinbanyee": 500,
    "jingwendiyee": 500,
    "kuanjingyee": 500,
    "maimianyee": 500,
    "maisuiyee": 500,
    "moyee": 500,
    "pingshaoyingyee": 500,
    "qiweiyee": 500,
    "sanchadilaohu": 500,
    "tiancaiyee": 1000,  # 北京比赛出现
    "xianweiyee": 500,
    "xiaodilaohu": 1000,  # 北京比赛出现
    "xiewenyee": 1000,  # 北京比赛出现
    "xiumuyee": 1000,  # 北京比赛出现
    "xuanyouyee": 1000,  # 北京比赛出现
    "yindingyee": 1000,  # 北京比赛出现
    "yinwenyee": 1000,  # 北京比赛出现
    "yiqieyee": 500,
    "zhongjinhuyee": 500,
    # yee2
    "dadilaohu": 500,
    "dongfangnianchong": 1000,  # 北京比赛出现
    "erdianweiyee": 1000,  # 北京比赛出现
    "huangdilaohu": 1000,  # 北京比赛出现
    "laoshinianchong": 1000,  # 北京比赛出现
    "mianlingchong": 1000,  # 北京比赛出现
    "yanqingchong": 500,
    # yee3
    "hongzonghuiyee": 500,
    "juanyee": 200,
    "xiaozaoqiaoe": 500,
    "yanyee": 500,
    # 其他
    "biancie": 500,
    "chimeidongyee": 500,
    "dingyee": 500,
    "danmainianchong": 500,
    "ganwenyee": 500,
    "liushangyee": 500,
    "meiguijinyee": 500,
    "muxuyee": 1000,  # 北京比赛出现
    "niaozuihuyee": 500,
    # 近似
    "pingzhangzhoue": 500,
    "zitiaochie": 500,
    # 灯蛾
    "badianhuidenge": 500,
    "chunlue": 500,
    "chenwudenge": 500,
    "fendiedenge": 500,
    "shanguangmeidenge": 500,
    "yibeilue": 500,
    # 有点像螟
    "nianmoyee": 500,
    "daodue": 500,
    # "天蛾",
    "ganshutiane": 500,
    "gouyuetiane": 500,
    "quewentiane": 500,
    # # 东方蝼蛄
    "dongfanglougu": 500,
    "caoling": 500,
    # # 菜蛾
    "caie": {"xiaocaie": 500},  # 小菜蛾
    "jifeng": {"yeesoujifeng": 500},
    "chie": {"youtongchihuo": 500},
    "shie": {"xiaoshie":500},
    # "dawen": 1000,
    "other": {
        "dawen": 1000,
        "wen": 500,
        "yinchichong": 500,
        "ying": 500,
        "yiwu": 500,
        "other": 1000,
    }
}