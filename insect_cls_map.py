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
  # detect 轨：叶子目录名与 train/train_aug/batch_aug_classes.py CLASS_BATCH_ENTRIES 的 class_name 一致；
  # 按 class_name 首段「-」前前缀分组（与 augment 侧物种键习惯一致）；组内 / 标量权重均为 1（dict 内均分 expect_count）。
  "detect": {
    "anheisaijingui": {
      "anheisaijingui-ba": 1,
    },
    "badianhuidenge": {
      "badianhuidenge-ba": 1,
    },
    "baibianqieyee": {
      "baibianqieyee-ba": 1,
    },
    "bailajuanxuyeming": {
      "bailajuanxuyeming-ba": 1,
    },
    "baitiaoyee": {
      "baitiaoyee-ba": 1,
      "baitiaoyee-beijing": 1,
    },
    "bazidilaohu": {
      "bazidilaohu-ba-bei": 1,
      "bazidilaohu-ba-fu": 1,
      "bazidilaohu-beijing": 1,
    },
    "biancie-beijing": 1,
    "bowenyee-beijing": 1,
    "caodiming": {
      "caodiming-beijing": 1,
      "caodiming-net": 1,
    },
    "caoditanyee": {
      "caoditanyee-ba": 1,
      "caoditanyee-beijing": 1,
    },
    "caoling-beijing": 1,
    "chenwudenge": {
      "chenwudenge-ba": 1,
    },
    "chimeidongyee-beijing": 1,
    "chunlue": {
      "chunlue-ba": 1,
    },
    "dadilaohu": {
      "dadilaohu-ba": 1,
    },
    "daheisaijingui": {
      "daheisaijingui-ba": 1,
    },
    "daming": {
      "daming-ba": 1,
    },
    "danmainianchong-beijing": 1,
    "daodue": {
      "daodue-ba": 1,
      "daodue-ba-sc": 1,
    },
    "daofeishi-ba-sc": 1,
    "daolvchun": {
      "daolvchun-ba": 1,
    },
    "daozongjuanyeming": {
      "daozongjuanyeming-ba": 1,
    },
    "daqingyechan": {
      "daqingyechan-ba": 1,
      "daqingyechan-beijing": 1,
    },
    "dawen": {
      "dawen-ba": 1,
    },
    "dianguangyechan-ba-sc": 1,
    "dingyee-beijing": 1,
    "dongfanglougu": {
      "dongfanglougu-ba": 1,
      "dongfanglougu-beijing": 1,
    },
    "dongfangnianchong": {
      "dongfangnianchong-ba": 1,
      "dongfangnianchong-beijing": 1,
    },
    "doujiayeming-beijing": 1,
    "douyeming": {
      "douyeming-ba": 1,
    },
    "erdianweiyee": {
      "erdianweiyee-ba": 1,
      "erdianweiyee-beijing": 1,
    },
    "erhuaming": {
      "erhuaming-ba": 1,
      "erhuaming-ba-sc": 1,
    },
    "fayee": {
      "fayee-ba": 1,
    },
    "fendiedenge": {
      "fendiedenge-ba": 1,
    },
    "fuyou-ba-sc": 1,
    "ganlanyee": {
      "ganlanyee-ba": 1,
      "ganlanyee-beijing": 1,
    },
    "ganshutiane-beijing": 1,
    "ganweidongyee": {
      "ganweidongyee-ba": 1,
    },
    "ganwenyee-beijing": 1,
    "goujinzhenchong": {
      "goujinzhenchong-ba": 1,
    },
    "gouyuetiane": {
      "gouyuetiane-ba": 1,
    },
    "guajuanming": {
      "guajuanming-ba": 1,
      "guajuanming-beijing": 1,
    },
    "heidiantaie": {
      "heidiantaie-ba": 1,
    },
    "heiweiyechan": {
      "heiweiyechan-ba": 1,
    },
    "hongjiaolvlijingui": {
      "hongjiaolvlijingui-ba": 1,
    },
    "hongzonghuiyee-beijing": 1,
    "huajinglvwenhuang": {
      "huajinglvwenhuang-ba": 1,
    },
    "huangchizhuiyeyeming": {
      "huangchizhuiyeyeming-ba": 1,
      "huangchizhuiyeyeming-beijing": 1,
    },
    "huangchong-beijing": 1,
    "huangdilaohu": {
      "huangdilaohu-ba": 1,
      "huangdilaohu-beijing": 1,
    },
    "huangheyilijingui": {
      "huangheyilijingui-ba": 1,
    },
    "huangtutaie-beijing": 1,
    "huangyangjuanyeming-beijing": 1,
    "huangyee": {
      "huangyee-ba": 1,
    },
    "huangyeming": {
      "huangyeming-ba": 1,
    },
    "huangzuliechun": {
      "huangzuliechun-ba": 1,
    },
    "jiachong-ba-sc": 1,
    "jinbanyee-beijing": 1,
    "jingwendiyee": {
      "jingwendiyee-ba": 1,
    },
    "juanyee-net": 1,
    "kuanjingyee": {
      "kuanjingyee-ba": 1,
      "kuanjingyee-beijing": 1,
    },
    "laoshinianchong": {
      "laoshinianchong-ba": 1,
      "laoshinianchong-beijing": 1,
    },
    "liushangyee-beijing": 1,
    "maimianyee": {
      "maimianyee-ba": 1,
    },
    "maimuyeming": {
      "maimuyeming-ba": 1,
      "maimuyeming-beijing": 1,
    },
    "maisuiyee": {
      "maisuiyee-ba": 1,
    },
    "malingshusaijingui": {
      "malingshusaijingui-ba": 1,
    },
    "meiguijinyee": {
      "meiguijinyee-ba": 1,
    },
    "miandajuanyeming-beijing": 1,
    "mianlingchong": {
      "mianlingchong-ba": 1,
      "mianlingchong-beijing": 1,
    },
    "moyee": {
      "moyee-ba": 1,
    },
    "muxuyee-beijing": 1,
    "niaozuihuyee": {
      "niaozuihuyee-ba": 1,
    },
    "pingshaoyingyee": {
      "pingshaoyingyee-ba": 1,
    },
    "pingzhangzhoue": {
      "pingzhangzhoue-ba": 1,
    },
    "qijiaoming": {
      "qijiaoming-ba": 1,
    },
    "qiweiyee": {
      "qiweiyee-ba": 1,
    },
    "quewentiane": {
      "quewentiane-ba": 1,
    },
    "sanchadilaohu-beijing": 1,
    "sanhuaming": {
      "sanhuaming-ba": 1,
    },
    "shanguangmeidenge": {
      "shanguangmeidenge-ba": 1,
    },
    "shuiming": {
      "shuiming-ba": 1,
    },
    "sibanjuanyeming": {
      "sibanjuanyeming-ba": 1,
      "sibanjuanyeming-beijing": 1,
    },
    "taozhuming": {
      "taozhuming-ba": 1,
      "taozhuming-beijing": 1,
    },
    "tiancaibaidaiyeming": {
      "tiancaibaidaiyeming-ba": 1,
      "tiancaibaidaiyeming-beijing": 1,
    },
    "tiancaiyee": {
      "tiancaiyee-ba": 1,
      "tiancaiyee-beijing": 1,
    },
    "tonglvyilijingui": {
      "tonglvyilijingui-ba": 1,
    },
    "tubeibanhongchun": {
      "tubeibanhongchun-ba": 1,
    },
    "wen-ba-sc": 1,
    "xianbanhongchun": {
      "xianbanhongchun-ba": 1,
    },
    "xianweiyee": {
      "xianweiyee-ba": 1,
    },
    "xiaocaie": {
      "xiaocaie-ba-bei": 1,
      "xiaocaie-ba-fu": 1,
    },
    "xiaodilaohu": {
      "xiaodilaohu-ba-bei": 1,
      "xiaodilaohu-ba-fu": 1,
      "xiaodilaohu-beijing": 1,
    },
    "xiaoguantouxishuai": {
      "xiaoguantouxishuai-ba": 1,
    },
    "xiaoshie": {
      "xiaoshie-ba": 1,
    },
    "xiaoyunbansaijingui": {
      "xiaoyunbansaijingui-ba": 1,
    },
    "xiaozaoqiaoe": {
      "xiaozaoqiaoe-ba": 1,
    },
    "xiewenyee": {
      "xiewenyee-ba": 1,
      "xiewenyee-beijing": 1,
    },
    "xishuai-beijing": 1,
    "xiumuyee": {
      "xiumuyee-ba": 1,
      "xiumuyee-beijing": 1,
    },
    "xuanqiyee": {
      "xuanqiyee-ba": 1,
    },
    "xuanyouyee-beijing": 1,
    "yanqingchong": {
      "yanqingchong-ba": 1,
    },
    "yanyee-beijing": 1,
    "yeesoujifeng": {
      "yeesoujifeng-ba": 1,
    },
    "yibeilue": {
      "yibeilue-ba": 1,
    },
    "yinchichong-ba-sc": 1,
    "yindingyee": {
      "yindingyee-ba": 1,
      "yindingyee-beijing": 1,
    },
    "ying-ba-sc": 1,
    "yinwenyee": {
      "yinwenyee-ba": 1,
      "yinwenyee-beijing": 1,
    },
    "yiqieyee": {
      "yiqieyee-ba": 1,
    },
    "youcaijiaoyeming": {
      "youcaijiaoyeming-ba": 1,
    },
    "youtongchihuo": {
      "youtongchihuo-ba": 1,
      "youtongchihuo-ba-sc": 1,
    },
    "yumiming": {
      "yumiming-ba": 1,
      "yumiming-beijing": 1,
    },
    "zhanmoyee": {
      "zhanmoyee-ba": 1,
    },
    "zhonghuaxiaobianxijingui": {
      "zhonghuaxiaobianxijingui-ba": 1,
    },
    "zhongjinhuyee": {
      "zhongjinhuyee-ba": 1,
    },
    "zitiaochie": {
      "zitiaochie-ba": 1,
    },
  }
}
