#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2026/05/01
# @Author  : ysy
# @Email   : xxx@qq.com 
# @Detail  : 样本个数备注来自 doc/参考资料/样本统计.md（big-cls-seg 各子目录条目数）。
# @Software: PyCharm
cls_map = {
  "cls": {
    # 776 = 215+251+310
    "bazidilaohu": [
      "bazidilaohu",
      "bazidilaohu-bei",
      "bazidilaohu-fu",
    ],
    "caodiming": {
      "caodiming-beijing": 190,  # 190
      "caodiming-net": 16,  # 16
    },
    "caoditanyee": 1,  # 184
    "dadilaohu": 0.8,  # 450
    "daming": 1,  # 139
    "daozongjuanyeming": 1,  # 621
    "dongfanglougu": 1,  # 140
    "dongfangnianchong": 1,  # 109
    "erdianweiyee": {
      "erdianweiyee-ba": 445,  # 445
      "erdianweiyee-beijing": 200,  # 200
    },
    "erhuaming": 1,  # 179
    "goujinzhenchong": 1, # 621
    "huangdilaohu": 1,  # 132
    "laoshinianchong": 1,  # 176
    "malingshusaijingui": 1,  # 377
    "mianlingchong": 1,  # 333
    "sanhuaming": 1,  # 72
    "taozhuming": 1,  # 172
    "tiancaibaidaiyeming": {
      "tiancaibaidaiyeming": 245,  # 245
      "tiancaibaidaiyeming-ba": 515,  # 515
    },
    "tiancaiyee": 1,  # 16
    # 983 = 163+820
    "xiaocaie": [
      "xiaocaie-bei",
      "xiaocaie-fu",
    ],
    # 374 = 131+243
    "xiaodilaohu": [
      "xiaodilaohu-bei",
      "xiaodilaohu-fu",
    ],
    "xiewenyee": 1,  # 756
    "yanqingchong": 1,  # 1173
    "yindingyee": {
      "yindingyee": 200,  # 200
      "yindingyee-ba": 167,  # 167
    },
    "yinwenyee": 1,  # 205
    "yumiming": 1,  # 968
    "daofeishi": [
      "daofeishi-small",
    ],  # daofeishi-small（统计表未单列目录）
    "ying": {
      "ying-small": 1,  # ying-small（统计表未单列目录）
    },
    "other_gui": {
      "anheisaijingui": 0.2,  # 189
      "daheisaijingui": 0.2,  # 76
      "hongjiaolvlijingui": 0.2,  # 42
      "huangheyilijingui": 0.2,  # 288
      "tonglvyilijingui": 0.2,  # 291
      "zhonghuaxiaobianxijingui": 0.2,  # 240
      "jiachong-small": 0.2,  # （统计表未单列目录）
      "daolvchun": 0.2,  # 54
      "xiaoyunbansaijingui": 0.2,  # 150
    },
    "other_chong": {
      "daqingyechan": 0.2,  # 75
      "huajinglvwenhuang": 0.2,  # 124
      "huangzuliechun": 0.2,  # 243
      "tubeibanhongchun": 0.2,  # 120
      "xianbanhongchun": 0.2,  # 30
      "xiaoguantouxishuai": 0.2,  # 458
    },
    "other_ming1": {
      "douyeming": 0.2,  # 288
      "guajuanming": 0.2,  # 315
      "qijiaoming": 0.2,  # 8
      "sibanjuanyeming": 0.2,  # 8
    },
    "other_ming2": {
      "shuiming": 0.5,  # 352
      "youcaijiaoyeming": 0.5,  # 958
    },
    "huangyeming": 1,  # 40
    "huangchizhuiyeyeming": 1,  # 304
    "maimuyeming": 1,  # 464
    "other_yee": {
      "baibianqieyee": 0.1,  # 352
      "baitiaoyee": 0.1,  # 374
      "fayee": 0.1,  # 136
      "ganlanyee": 0.1,  # 314
      "huangyee": 0.1,  # 300
      "jingwendiyee": 0.1,  # 332
      "maimianyee": 0.1,  # 572
      "maisuiyee": 0.1,  # 327
      "pingshaoyingyee": 0.1,  # 216
      "qiweiyee": 0.1,  # 1000
      "xianweiyee": 0.1,  # 86
      "xiumuyee": 0.1,  # 80
      "xuanqiyee": 0.1,  # 173
      "yiqieyee": 0.1,  # 349
    },
    "biancie": 1,  # 191
    "badianhuidenge": 1,  # 173
    "chunlue": 1,  # 30
    "daodue": 1,  # 85
    "fendiedenge": 1,  # 104
    "ganweidongyee": 1,  # 361
    "gouyuetiane": 1,  # 96
    "heidiantaie": 1,  # 301
    "huangtutaie": 1,  # 238
    "juanyee": {
      "juanyee-net": 1,  # 30
    },
    "kuanjingyee": 1,  # 132
    "meiguijinyee": 1,  # 74
    "moyee": 1,  # 309
    "niaozuihuyee": 1,  # 162
    "pingzhangzhoue": 1,  # 361
    "quewentiane": 1,  # 31
    "shanguangmeidenge": 1,  # 40
    "xiaoshie": 1,  # 154
    "yeesoujifeng": 1,  # 336
    "yibeilue": 1,  # 163
    "youtongchihuo": 1,  # 135
    "zhanmoyee": 1,  # 644
    "zhongjinhuyee": 1,  # 258
    "zitiaochie": 1,  # 116
    "other_small": {
      "yinchichong-small": 0.5,  # （统计表未单列目录）
      "wen-small": 0.5,  # （统计表未单列目录）
      "dawen": 0.5,  # 70
    },
  },
  "detect": {
    "bazidilaohu": {
      "bazidilaohu": 0.5,  # 215
      "bazidilaohu-bei": 0.3,  # 251
      "bazidilaohu-fu": 0.2,  # 310
    },
    "caodiming": {
      "caodiming-beijing": 0.8,  # 190
      "caodiming-net": 0.2,  # 16
    },
    "caoditanyee": 1,  # 184
    "dadilaohu": 1,  # 450
    "daming": 1,  # 139
    "daozongjuanyeming": 1,  # 621
    "dongfanglougu": 1,  # 140
    "dongfangnianchong": 1,  # 109
    "erdianweiyee": {
      "erdianweiyee-ba": 0.5,  # 445
      "erdianweiyee-beijing": 0.5,  # 200
    },
    "erhuaming": 1,  # 179
    "huangdilaohu": 1,  # 132
    "laoshinianchong": 1,  # 176
    "malingshusaijingui": 1,  # 377
    "mianlingchong": 1,  # 333
    "sanhuaming": 1,  # 72
    "taozhuming": 1,  # 172
    "tiancaibaidaiyeming": {
      "tiancaibaidaiyeming": 0.5,  # 245
      "tiancaibaidaiyeming-ba": 0.5,  # 515
    },
    "tiancaiyee": 1,  # 16
    # 983 = 163+820
    "xiaocaie": {
        "xiaocaie-bei": 0.7,
        "xiaocaie-fu": 0.3
    },
    # 374 = 131+243
    "xiaodilaohu": {
      "xiaodilaohu-bei": 0.7,
      "xiaodilaohu-fu": 0.3,
    },
    "xiewenyee": 1,  # 756
    "yanqingchong": 1,  # 1173
    "yindingyee": {
      "yindingyee": 0.5,  # 200
      "yindingyee-ba": 0.5,  # 167
    },
    "yinwenyee": 1,  # 205
    "yumiming": 1,  # 968
    "daofeishi": [
      "daofeishi-small",  # 拼接时不使用
    ],  # daofeishi-small（统计表未单列目录）
    "ying": {
      "ying-small": 1,  # ying-small（统计表未单列目录） # 拼接时不使用
    },
    "other_gui": {
      "anheisaijingui": 1,  # 189
      "daheisaijingui": 1,  # 76
      "hongjiaolvlijingui": 1,  # 42
      "huangheyilijingui": 1,  # 288
      "tonglvyilijingui": 1,  # 291
      "zhonghuaxiaobianxijingui": 1,  # 240
      "jiachong-small": 1,  # （统计表未单列目录）
      "daolvchun": 1,  # 54
      "xiaoyunbansaijingui": 1,  # 150
    },
    "other_chong": {
      "daqingyechan": 1,  # 75
      "huajinglvwenhuang": 1,  # 124
      "huangzuliechun": 1,  # 243
      "tubeibanhongchun": 1,  # 120
      "xianbanhongchun": 1,  # 30
      "xiaoguantouxishuai": 1,  # 458
    },
    "other_ming1": {
      "douyeming": 1,  # 288
      "guajuanming": 1,  # 315
      "qijiaoming": 1,  # 8
      "sibanjuanyeming": 1,  # 8
    },
    "other_ming2": {
      "shuiming": 1,  # 352
      "youcaijiaoyeming": 1,  # 958
    },
    "huangyeming": 1,  # 40
    "huangchizhuiyeyeming": 1,  # 304
    "maimuyeming": 1,  # 464
    "other_yee": {
      "baibianqieyee": 1,  # 352
      "baitiaoyee": 1,  # 374
      "fayee": 1,  # 136
      "ganlanyee": 1,  # 314
      "huangyee": 1,  # 300
      "jingwendiyee": 1,  # 332
      "maimianyee": 1,  # 572
      "maisuiyee": 1,  # 327
      "pingshaoyingyee": 1,  # 216
      "qiweiyee": 1,  # 1000
      "xianweiyee": 1,  # 86
      "xiumuyee": 1,  # 80
      "xuanqiyee": 1,  # 173
      "yiqieyee": 1,  # 349
    },
    "biancie": 1,  # 191
    "badianhuidenge": 1,  # 173
    "chunlue": 1,  # 30
    "daodue": 1,  # 85
    "fendiedenge": 1,  # 104
    "ganweidongyee": 1,  # 361
    "gouyuetiane": 1,  # 96
    "heidiantaie": 1,  # 301
    "huangtutaie": 1,  # 238
    "juanyee": {
      "juanyee-net": 1,  # 30
    },
    "kuanjingyee": 1,  # 132
    "meiguijinyee": 1,  # 74
    "moyee": 1,  # 309
    "niaozuihuyee": 1,  # 162
    "pingzhangzhoue": 1,  # 361
    "quewentiane": 1,  # 31
    "shanguangmeidenge": 1,  # 40
    "xiaoshie": 1,  # 154
    "yeesoujifeng": 1,  # 336
    "yibeilue": 1,  # 163
    "youtongchihuo": 1,  # 135
    "zhanmoyee": 1,  # 644
    "zhongjinhuyee": 1,  # 258
    "zitiaochie": 1,  # 116
    "dawen": 1,  # 70
    "other_small": {
      "yinchichong-small": 1,  # （统计表未单列目录）
      "wen-small": 1,  # （统计表未单列目录）
    },
  }
}
