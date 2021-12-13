import yfinance as yf
import pandas as pd
from matplotlib import pyplot as plt
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation, performance_metrics
from fbprophet.plot import plot_cross_validation_metric
from fbprophet.plot import add_changepoints_to_plot
from datetime import datetime
import numpy as np
from diskcache import FanoutCache
# from diskcache import Cache

import time
import math

class StockPredictor:
    def __init__(self, ticker, periods):
        self.ticker = ticker

        self.cache = FanoutCache(directory='./tmp', timeout=20, shards=4)
        # self.cache = FanoutCache(directory='./tmp', timeout=20, eviction_policy='none')
        # self.cache = Cache(directory='./tmp')
        self.cache.clear()
        self.cache_obj = []
        self.cache_expire = 60 * 60 * 12 # 12 hours
        self.prime_cache()

        self.stock_info = self.get_stock_info()
        self.stock_info['now'] = datetime.now()

        self.periods = self.restrict_max_periods(periods)

        self.forecaster()

    def prime_cache(self):
        self.cache_obj = [
            # {'ticker': 'AUDUSD=X', 'changepoint_prior_scale': 0.03}, #mape 0.1553
            # {'ticker': 'AUDUSD=X', 'changepoint_prior_scale': 0.03}, #mape 0.151914
            # {'ticker': 'AUDUSD=X', 'changepoint_prior_scale': 0.01}, #mape 0.151952
            # {'ticker': 'AUDUSD=X', 'changepoint_prior_scale': 0.02}, #mape 0.147149
            # {'ticker': 'AUDUSD=X', 'changepoint_prior_scale': 0.02}, #mape 0.142872
            # {'ticker': 'AUDUSD=X', 'changepoint_prior_scale': 0.02}, #mape 0.145305
            # {'ticker': 'AUDUSD=X', 'changepoint_prior_scale': 0.01}, #mape 0.138171
            # {'ticker': 'AUDUSD=X', 'changepoint_prior_scale': 0.01}, #mape 0.139665
            # {'ticker': 'AUDUSD=X', 'changepoint_prior_scale': 0.01}, #mape 0.142446 -3.05%
            # {'ticker': 'AUDUSD=X', 'changepoint_prior_scale': 0.01}, #mape 0.141402
            # {'ticker': 'AUDUSD=X', 'changepoint_prior_scale': 0.01}, #mape 0.147068 -1.46%
            # {'ticker': 'AUDUSD=X', 'changepoint_prior_scale': 0.01}, #mape 0.144783 365d -3.49% - 31/12/21 0.7021
            # {'ticker': 'AUDUSD=X', 'changepoint_prior_scale': 0.01}, #mape 0.152013 365d -5.2% - 31/12/21 0.7017

            # {'ticker': 'a200.ax', 'changepoint_prior_scale': 0.07},
            # {'ticker': 'a200.ax', 'changepoint_prior_scale': 0.03}, #mape 0.317347
            # {'ticker': 'a200.ax', 'changepoint_prior_scale': 0.42}, #mape 0.084283
            # {'ticker': 'a200.ax', 'changepoint_prior_scale': 0.09}, #mape 0.155679
            # {'ticker': 'a200.ax', 'changepoint_prior_scale': 0.07}, #mape 0.228445
            # {'ticker': 'a200.ax', 'changepoint_prior_scale': 0.5}, #mape 0.191195
            # {'ticker': 'a200.ax', 'changepoint_prior_scale': 0.26}, #mape 0.052278

            # {'ticker': 'acdc.ax', 'changepoint_prior_scale': 0.38},
            # {'ticker': 'acdc.ax', 'changepoint_prior_scale': 0.50},
            # {'ticker': 'acdc.ax', 'changepoint_prior_scale': 0.05},
            # {'ticker': 'acdc.ax', 'changepoint_prior_scale': 0.04}, #mape 0.2095
            # {'ticker': 'acdc.ax', 'changepoint_prior_scale': 0.50}, #mape 0.1579
            # {'ticker': 'acdc.ax', 'changepoint_prior_scale': 0.12}, #mape 0.18449
            # {'ticker': 'acdc.ax', 'changepoint_prior_scale': 0.42}, #mape 0.14471
            # {'ticker': 'acdc.ax', 'changepoint_prior_scale': 0.08}, #mape 0.094548
            # {'ticker': 'acdc.ax', 'changepoint_prior_scale': 0.05}, #mape 0.119586
            # {'ticker': 'acdc.ax', 'changepoint_prior_scale': 0.04}, #mape 0.128018
            # {'ticker': 'acdc.ax', 'changepoint_prior_scale': 0.01}, #mape 0.181006
            # {'ticker': 'acdc.ax', 'changepoint_prior_scale': 0.01}, #mape 0.187364
            # {'ticker': 'acdc.ax', 'changepoint_prior_scale': 0.45}, #mape 0.135106 28.99%
            # {'ticker': 'acdc.ax', 'changepoint_prior_scale': 0.25}, #mape 0.201508 18.96%
            # {'ticker': 'acdc.ax', 'changepoint_prior_scale': 0.29}, #mape 0.164247 21.73%
            # {'ticker': 'acdc.ax', 'changepoint_prior_scale': 0.24}, #mape 0.124192 18.85%
            # {'ticker': 'acdc.ax', 'changepoint_prior_scale': 0.08}, #mape 0.112776 25.08%
            # {'ticker': 'acdc.ax', 'changepoint_prior_scale': 0.12}, #mape 0.079953 365d 18.35% - 31/12/21 99.3612
            # {'ticker': 'acdc.ax', 'changepoint_prior_scale': 0.08}, #mape 0.065357 365d 32.58% - 31/12/21 105.7569
            # {'ticker': 'acdc.ax', 'changepoint_prior_scale': 0.31}, #mape 0.109954 365d 6.74% - 31/12/21 91.6252
            # {'ticker': 'acdc.ax', 'changepoint_prior_scale': 0.04}, #mape 0.150019 365d 27.93% - 31/12/21 104.4566
            # {'ticker': 'acdc.ax', 'changepoint_prior_scale': 0.04}, #mape 0.159290 365d 30.65% - 31/12/21 104.4742
            # {'ticker': 'acdc.ax', 'changepoint_prior_scale': 0.04}, #mape 0.196458 365d 17.6% - 31/12/21 102.717
            # {'ticker': 'acdc.ax', 'changepoint_prior_scale': 0.03}, #mape 0.224679 365d 12.24% - 31/12/21 102.7332
            # {'ticker': 'acdc.ax', 'changepoint_prior_scale': 0.03}, #mape 0.217776 365d 11.53% - 31/12/21 101.7529
            # {'ticker': 'acdc.ax', 'changepoint_prior_scale': 0.03}, #mape 0.207013 365d 10.17% - 31/12/21 100.4397
            {'ticker': 'acdc.ax', 'changepoint_prior_scale': 0.26}, #mape 0.177400 365d -1.82% - 31/12/21 95.327

            # {'ticker': 'asia.ax', 'changepoint_prior_scale': 0.21},
            # {'ticker': 'asia.ax', 'changepoint_prior_scale': 0.03},
            # {'ticker': 'asia.ax', 'changepoint_prior_scale': 0.49},
            # {'ticker': 'asia.ax', 'changepoint_prior_scale': 0.07}, #mape 0.1373
            # {'ticker': 'asia.ax', 'changepoint_prior_scale': 0.07}, #mape 0.1393
            # {'ticker': 'asia.ax', 'changepoint_prior_scale': 0.01}, #mape 0.17043
            # {'ticker': 'asia.ax', 'changepoint_prior_scale': 0.01}, #mape 0.162667
            # {'ticker': 'asia.ax', 'changepoint_prior_scale': 0.01}, #mape 0.164748
            # {'ticker': 'asia.ax', 'changepoint_prior_scale': 0.01}, #mape 0.183759
            # {'ticker': 'asia.ax', 'changepoint_prior_scale': 0.42}, #mape 0.138680
            # {'ticker': 'asia.ax', 'changepoint_prior_scale': 0.50}, #mape 0.138135
            # {'ticker': 'asia.ax', 'changepoint_prior_scale': 0.01}, #mape 0.231809 38.54%
            # {'ticker': 'asia.ax', 'changepoint_prior_scale': 0.01}, #mape 0.263879 21.12%
            # {'ticker': 'asia.ax', 'changepoint_prior_scale': 0.01}, #mape 0.279814 29.13%
            # {'ticker': 'asia.ax', 'changepoint_prior_scale': 0.01}, #mape 0.307085 10.33%
            # {'ticker': 'asia.ax', 'changepoint_prior_scale': 0.01}, #mape 0.341828 -3.18%
            # {'ticker': 'asia.ax', 'changepoint_prior_scale': 0.01}, #mape 0.338119 365d -16.75% - 31/12/21 9.9657
            # {'ticker': 'asia.ax', 'changepoint_prior_scale': 0.36}, #mape 0.239116 365d -56.57% - 31/12/21 8.2725
            # {'ticker': 'asia.ax', 'changepoint_prior_scale': 0.35}, #mape 0.092125 365d -47.44% - 31/12/21 8.8716
            # {'ticker': 'asia.ax', 'changepoint_prior_scale': 0.32}, #mape 0.083058 365d -30.16% - 31/12/21 9.6397
            # {'ticker': 'asia.ax', 'changepoint_prior_scale': 0.33}, #mape 0.167797 365d -34.01% - 31/12/21 10.0113
            # {'ticker': 'asia.ax', 'changepoint_prior_scale': 0.23}, #mape 0.181401 365d -32.53% - 31/12/21 9.7616
            # {'ticker': 'asia.ax', 'changepoint_prior_scale': 0.06}, #mape 0.278123 365d -34.84% - 31/12/21 9.8716
            # {'ticker': 'asia.ax', 'changepoint_prior_scale': 0.05}, #mape 0.300386 365d -30.97% - 31/12/21 9.9137
            # {'ticker': 'asia.ax', 'changepoint_prior_scale': 0.04}, #mape 0.315596 365d -29.99% - 31/12/21 9.8353
            {'ticker': 'asia.ax', 'changepoint_prior_scale': 0.04}, #mape 0.333477 365d -33.11% - 31/12/21 9.7691

            # {'ticker': 'csl.ax', 'changepoint_prior_scale': 0.01}, #mape 0.421860 365d 14.51% - 31/12/21 327.6463

            # {'ticker': 'cryp.ax', 'changepoint_prior_scale': 0.35}, #mape 0.042261 5d -3.81% - 23/11/21 10.5616
            # {'ticker': 'cryp.ax', 'changepoint_prior_scale': 0.23}, #mape 0.079717 6d -12.64% - 23/11/21 9.8692 =10.48 close
            # {'ticker': 'cryp.ax', 'changepoint_prior_scale': 0.35}, #mape 0.061727 7d 7.27% - 3/12/21 11.703
            # {'ticker': 'cryp.ax', 'changepoint_prior_scale': 0.03}, #mape 0.028697 9d 0.53% - 3/12/21 10.3007 =9.96 close
            {'ticker': 'cryp.ax', 'changepoint_prior_scale': 0.07}, #mape 0.064624 11d -13.6% - 21/12/21 7.439

            # {'ticker': 'dhhf.ax', 'changepoint_prior_scale': 0.02}, #mape 0.017303
            # {'ticker': 'dhhf.ax', 'changepoint_prior_scale': 0.02}, #mape 0.017396
            # {'ticker': 'dhhf.ax', 'changepoint_prior_scale': 0.02}, #mape 0.021721
            # {'ticker': 'dhhf.ax', 'changepoint_prior_scale': 0.02}, #mape 0.015739
            # {'ticker': 'dhhf.ax', 'changepoint_prior_scale': 0.02}, #mape 0.016436
            # {'ticker': 'dhhf.ax', 'changepoint_prior_scale': 0.02}, #mape 0.017751 21.23%
            # {'ticker': 'dhhf.ax', 'changepoint_prior_scale': 0.02}, #mape 0.025417 18.37%
            # {'ticker': 'dhhf.ax', 'changepoint_prior_scale': 0.02}, #mape 0.022027 20.74%
            # {'ticker': 'dhhf.ax', 'changepoint_prior_scale': 0.03}, #mape 0.017488 20.71%
            # {'ticker': 'dhhf.ax', 'changepoint_prior_scale': 0.03}, #mape 0.018066 23.51%
            # {'ticker': 'dhhf.ax', 'changepoint_prior_scale': 0.03}, #mape 0.016852 365d 23.84% - 31/12/21 32.6278
            # {'ticker': 'dhhf.ax', 'changepoint_prior_scale': 0.02}, #mape 0.017499 365d 21.74% - 31/12/21 32.342
            # {'ticker': 'dhhf.ax', 'changepoint_prior_scale': 0.02}, #mape 0.010894 365d 25.51% - 31/12/21 32.1858
            # {'ticker': 'dhhf.ax', 'changepoint_prior_scale': 0.02}, #mape 0.023498 365d 27.5% - 31/12/21 31.7624
            # {'ticker': 'dhhf.ax', 'changepoint_prior_scale': 0.03}, #mape 0.023412 365d 23.43% - 31/12/21 31.4628
            # {'ticker': 'dhhf.ax', 'changepoint_prior_scale': 0.02}, #mape 0.021211 365d 16.4% - 31/12/21 31.2116
            # {'ticker': 'dhhf.ax', 'changepoint_prior_scale': 0.02}, #mape 0.013419 365d 15.06% - 31/12/21 31.1454
            # {'ticker': 'dhhf.ax', 'changepoint_prior_scale': 0.02}, #mape 0.013098 365d 17.08% - 31/12/21 31.1156
            # {'ticker': 'dhhf.ax', 'changepoint_prior_scale': 0.01}, #mape 0.110467 365d 20.22% - 31/12/21 31.1694
            {'ticker': 'dhhf.ax', 'changepoint_prior_scale': 0.47}, #mape 0.099088 365d -17.66% - 31/12/21 29.4806

            # {'ticker': 'espo.ax', 'changepoint_prior_scale': 0.01},
            # {'ticker': 'espo.ax', 'changepoint_prior_scale': 0.02}, #mape 0.0622
            # {'ticker': 'espo.ax', 'changepoint_prior_scale': 0.02}, #mape 0.1206
            # {'ticker': 'espo.ax', 'changepoint_prior_scale': 0.01}, #mape 0.140703
            # {'ticker': 'espo.ax', 'changepoint_prior_scale': 0.01}, #mape 0.11134
            # {'ticker': 'espo.ax', 'changepoint_prior_scale': 0.03}, #mape 0.101008
            # {'ticker': 'espo.ax', 'changepoint_prior_scale': 0.01}, #mape 0.11194
            # {'ticker': 'espo.ax', 'changepoint_prior_scale': 0.03}, #mape 0.164366
            # {'ticker': 'espo.ax', 'changepoint_prior_scale': 0.01}, #mape 0.102609
            # {'ticker': 'espo.ax', 'changepoint_prior_scale': 0.01}, #mape 0.072559
            # {'ticker': 'espo.ax', 'changepoint_prior_scale': 0.08}, #mape 0.103001 62.9%
            # {'ticker': 'espo.ax', 'changepoint_prior_scale': 0.01}, #mape 0.065316 -14.62%
            # {'ticker': 'espo.ax', 'changepoint_prior_scale': 0.01}, #mape 0.069893 2.27%
            # {'ticker': 'espo.ax', 'changepoint_prior_scale': 0.01}, #mape 0.079136 0.18%
            # {'ticker': 'espo.ax', 'changepoint_prior_scale': 0.01}, #mape 0.082522 -5.05%
            # {'ticker': 'espo.ax', 'changepoint_prior_scale': 0.01}, #mape 0.075765 365d -14.13% - 23/11/21 10.9301
            # {'ticker': 'espo.ax', 'changepoint_prior_scale': 0.01}, #mape 0.062340 365d -10.74% - 23/11/21 11.0622
            # {'ticker': 'espo.ax', 'changepoint_prior_scale': 0.01}, #mape 0.034453 365d -2.9% - 23/11/21 11.0774
            # {'ticker': 'espo.ax', 'changepoint_prior_scale': 0.06}, #mape 0.047523 365d -13.75% - 23/11/21 10.7594
            # {'ticker': 'espo.ax', 'changepoint_prior_scale': 0.01}, #mape 0.037349 365d -12.89% - 23/11/21 10.9675
            # {'ticker': 'espo.ax', 'changepoint_prior_scale': 0.01}, #mape 0.034268 365d -19.78% - 23/11/21 11.0625
            # {'ticker': 'espo.ax', 'changepoint_prior_scale': 0.04}, #mape 0.071378 365d -17.85% - 23/11/21 11.5341 =12.34 close - 31/12/21 11.7213
            # {'ticker': 'espo.ax', 'changepoint_prior_scale': 0.29}, #mape 0.058615 365d 24.56% - 31/12/21 12.4421
            # {'ticker': 'espo.ax', 'changepoint_prior_scale': 0.07}, #mape 0.052311 365d 38.83% - 31/12/21 12.4618
            {'ticker': 'espo.ax', 'changepoint_prior_scale': 0.01}, #mape 0.048909 365d -1.68% - 31/12/21 11.6384

            # {'ticker': 'espo', 'changepoint_prior_scale': 0.34},
            # {'ticker': 'espo', 'changepoint_prior_scale': 0.17},
            # {'ticker': 'espo', 'changepoint_prior_scale': 0.173274},
            # {'ticker': 'espo', 'changepoint_prior_scale': 0.19208}, #mape 0.0661
            # {'ticker': 'espo', 'changepoint_prior_scale': 0.22}, #mape 0.1023
            # {'ticker': 'espo', 'changepoint_prior_scale': 0.18}, #mape 0.133571
            # {'ticker': 'espo', 'changepoint_prior_scale': 0.11}, #mape 0.160253
            # {'ticker': 'espo', 'changepoint_prior_scale': 0.01}, #mape 0.182150
            # {'ticker': 'espo', 'changepoint_prior_scale': 0.49}, #mape 0.162895
            # {'ticker': 'espo', 'changepoint_prior_scale': 0.01}, #mape 0.177959
            # {'ticker': 'espo', 'changepoint_prior_scale': 0.22}, #mape 0.100298
            # {'ticker': 'espo', 'changepoint_prior_scale': 0.28}, #mape 0.104785
            # {'ticker': 'espo', 'changepoint_prior_scale': 0.17}, #mape 0.165169 24.68%
            # {'ticker': 'espo', 'changepoint_prior_scale': 0.17}, #mape 0.180148 14.83%
            # {'ticker': 'espo', 'changepoint_prior_scale': 0.16}, #mape 0.203623  4.14%
            # {'ticker': 'espo', 'changepoint_prior_scale': 0.11}, #mape 0.219064 -5.97%
            # {'ticker': 'espo', 'changepoint_prior_scale': 0.48}, #mape 0.200261 -49.46%
            # {'ticker': 'espo', 'changepoint_prior_scale': 0.5}, #mape 0.126098 365d -49.46%
            # {'ticker': 'espo', 'changepoint_prior_scale': 0.34}, #mape 0.105264 365d -36.69% - 31/12/21 61.
            # {'ticker': 'espo', 'changepoint_prior_scale': 0.1}, #mape 0.122936 365d -19.16% - 31/12/21 62.9802
            # {'ticker': 'espo', 'changepoint_prior_scale': 0.45}, #mape 0.118047 365d 0.53% - 31/12/21 67.0827
            # {'ticker': 'espo', 'changepoint_prior_scale': 0.38}, #mape 0.138772 365d -3.47% - 31/12/21 68.8802
            # {'ticker': 'espo', 'changepoint_prior_scale': 0.25}, #mape 0.164090 365d 14.28% - 31/12/21 77.6607
            # {'ticker': 'espo', 'changepoint_prior_scale': 0.15}, #mape 0.181396 365d 23.81% - 31/12/21 85.4201
            # {'ticker': 'espo', 'changepoint_prior_scale': 0.11}, #mape 0.210955 365d 20.45% - 31/12/21 80.7406
            # {'ticker': 'espo', 'changepoint_prior_scale': 0.48}, #mape 0.212404 365d 13.65% - 31/12/21 77.578
            {'ticker': 'espo', 'changepoint_prior_scale': 0.5}, #mape 0.214806 365d 7.92% - 31/12/21 75.9053

            # {'ticker': 'ethi.ax', 'changepoint_prior_scale': 0.02},
            # {'ticker': 'ethi.ax', 'changepoint_prior_scale': 0.014904},

            # {'ticker': 'hack.ax', 'changepoint_prior_scale': 0.01},
            # {'ticker': 'hack.ax', 'changepoint_prior_scale': 0.02678}, #mape 0.0797
            # {'ticker': 'hack.ax', 'changepoint_prior_scale': 0.01}, #mape 0.056
            # {'ticker': 'hack.ax', 'changepoint_prior_scale': 0.01}, #mape 0.0485
            # {'ticker': 'hack.ax', 'changepoint_prior_scale': 0.01}, #mape 0.0490
            # {'ticker': 'hack.ax', 'changepoint_prior_scale': 0.01}, #mape 0.058214
            # {'ticker': 'hack.ax', 'changepoint_prior_scale': 0.04}, #mape 0.050385
            # {'ticker': 'hack.ax', 'changepoint_prior_scale': 0.03}, #mape 0.038263
            # {'ticker': 'hack.ax', 'changepoint_prior_scale': 0.02}, #mape 0.037973
            # {'ticker': 'hack.ax', 'changepoint_prior_scale': 0.02}, #mape 0.065794

            # {'ticker': 'hndq.ax', 'changepoint_prior_scale': 0.14},
            # {'ticker': 'hndq.ax', 'changepoint_prior_scale': 0.01},
            # {'ticker': 'hndq.ax', 'changepoint_prior_scale': 0.009593},
            # {'ticker': 'hndq.ax', 'changepoint_prior_scale': 0.26}, #mape 0.027785
            # {'ticker': 'hndq.ax', 'changepoint_prior_scale': 0.49}, #mape 0.0258
            # {'ticker': 'hndq.ax', 'changepoint_prior_scale': 0.06}, #mape 0.049505
            # {'ticker': 'hndq.ax', 'changepoint_prior_scale': 0.05}, #mape 0.05831
            # {'ticker': 'hndq.ax', 'changepoint_prior_scale': 0.01}, #mape 0.036996
            # {'ticker': 'hndq.ax', 'changepoint_prior_scale': 0.01}, #mape 0.047395
            # {'ticker': 'hndq.ax', 'changepoint_prior_scale': 0.01}, #mape 0.041121
            # {'ticker': 'hndq.ax', 'changepoint_prior_scale': 0.01}, #mape 0.040995
            # {'ticker': 'hndq.ax', 'changepoint_prior_scale': 0.02}, #mape 0.036756
            # {'ticker': 'hndq.ax', 'changepoint_prior_scale': 0.01}, #mape 0.034511 21.64%
            # {'ticker': 'hndq.ax', 'changepoint_prior_scale': 0.02}, #mape 0.024524 12.0%
            # {'ticker': 'hndq.ax', 'changepoint_prior_scale': 0.01}, #mape 0.028111 21.78%
            # {'ticker': 'hndq.ax', 'changepoint_prior_scale': 0.01}, #mape 0.026905 14.86%
            # {'ticker': 'hndq.ax', 'changepoint_prior_scale': 0.01}, #mape 0.028619 25.52%
            # {'ticker': 'hndq.ax', 'changepoint_prior_scale': 0.01}, #mape 0.035531 365d 15.41% - 06/12/21 35.5788
            # {'ticker': 'hndq.ax', 'changepoint_prior_scale': 0.01}, #mape 0.034879 365d 12.37% - 06/12/21 35.7833
            # {'ticker': 'hndq.ax', 'changepoint_prior_scale': 0.02}, #mape 0.013242 365d 28.1% - 06/12/21 36.0262
            # {'ticker': 'hndq.ax', 'changepoint_prior_scale': 0.01}, #mape 0.028534 365d 25.8% - 06/12/21 35.2247
            # {'ticker': 'hndq.ax', 'changepoint_prior_scale': 0.01}, #mape 0.018195 365d 20.21% - 06/12/21 35.1723
            # {'ticker': 'hndq.ax', 'changepoint_prior_scale': 0.01}, #mape 0.011203 365d 4.61% - 06/12/21 35.269
            # {'ticker': 'hndq.ax', 'changepoint_prior_scale': 0.10}, #mape 0.025131 365d 4.32% - 06/12/21 35.4263 - 31/12/21 35.8658
            # {'ticker': 'hndq.ax', 'changepoint_prior_scale': 0.09}, #mape 0.028406 365d 14.08% - 06/12/21 35.676 - 31/12/21 36.1587
            # {'ticker': 'hndq.ax', 'changepoint_prior_scale': 0.03}, #mape 0.027169 365d 21.28% - 06/12/21 35.6673 =34.88 - 31/12/21 36.1318
            {'ticker': 'hndq.ax', 'changepoint_prior_scale': 0.01}, #mape 0.020559 365d 17.77% - 31/12/21 36.0652

            # {'ticker': 'ivv.ax', 'changepoint_prior_scale': 0.01},
            # {'ticker': 'ivv.ax', 'changepoint_prior_scale': 0.004282},
            # {'ticker': 'ivv.ax', 'changepoint_prior_scale': 0.01}, #mape 0.1341
            # {'ticker': 'ivv.ax', 'changepoint_prior_scale': 0.01}, #mape 0.13373
            # {'ticker': 'ivv.ax', 'changepoint_prior_scale': 0.01}, #mape 0.137155
            # {'ticker': 'ivv.ax', 'changepoint_prior_scale': 0.01}, #mape 0.138201
            # {'ticker': 'ivv.ax', 'changepoint_prior_scale': 0.01}, #mape 0.138420
            # {'ticker': 'ivv.ax', 'changepoint_prior_scale': 0.01}, #mape 0.131091
            # {'ticker': 'ivv.ax', 'changepoint_prior_scale': 0.01}, #mape 0.131737

            # {'ticker': 'mnrs.ax', 'changepoint_prior_scale': 0.37},
            # {'ticker': 'mnrs.ax', 'changepoint_prior_scale': 0.18},
            # {'ticker': 'mnrs.ax', 'changepoint_prior_scale': 0.171128},
            # {'ticker': 'mnrs.ax', 'changepoint_prior_scale': 0.142070}, #mape 0.1113
            # {'ticker': 'mnrs.ax', 'changepoint_prior_scale': 0.13}, #mape 0.1162
            # {'ticker': 'mnrs.ax', 'changepoint_prior_scale': 0.13}, #mape 0.09327
            # {'ticker': 'mnrs.ax', 'changepoint_prior_scale': 0.10}, #mape 0.084046
            # {'ticker': 'mnrs.ax', 'changepoint_prior_scale': 0.03}, #mape 0.117335
            # {'ticker': 'mnrs.ax', 'changepoint_prior_scale': 0.06}, #mape 0.123390
            # {'ticker': 'mnrs.ax', 'changepoint_prior_scale': 0.06}, #mape 0.143267
            # {'ticker': 'mnrs.ax', 'changepoint_prior_scale': 0.02}, #mape 0.160701
            # {'ticker': 'mnrs.ax', 'changepoint_prior_scale': 0.02}, #mape 0.192525 5.79%
            # {'ticker': 'mnrs.ax', 'changepoint_prior_scale': 0.02}, #mape 0.209202 2.5%
            # {'ticker': 'mnrs.ax', 'changepoint_prior_scale': 0.02}, #mape 0.221340 -2.41%
            # {'ticker': 'mnrs.ax', 'changepoint_prior_scale': 0.02}, #mape 0.234285 -2.68%
            # {'ticker': 'mnrs.ax', 'changepoint_prior_scale': 0.02}, #mape 0.266588 -3.93%
            # {'ticker': 'mnrs.ax', 'changepoint_prior_scale': 0.02}, #mape 0.273668 365d -3.36%
            # {'ticker': 'mnrs.ax', 'changepoint_prior_scale': 0.02}, #mape 0.293806 365d -14.08% - 31/12/21 5.6153
            # {'ticker': 'mnrs.ax', 'changepoint_prior_scale': 0.01}, #mape 0.331432 365d -4.18% - 31/12/21 5.7841
            # {'ticker': 'mnrs.ax', 'changepoint_prior_scale': 0.5}, #mape 0.304101 365d -39.48% - 31/12/21 4.8427
            # {'ticker': 'mnrs.ax', 'changepoint_prior_scale': 0.48}, #mape 0.290367 365d -35.26% - 31/12/21 4.9888
            # {'ticker': 'mnrs.ax', 'changepoint_prior_scale': 0.25}, #mape 0.266190 365d -31.96% - 31/12/21 5.1579
            # {'ticker': 'mnrs.ax', 'changepoint_prior_scale': 0.35}, #mape 0.228771 365d -28.29% - 31/12/21 5.5703
            # {'ticker': 'mnrs.ax', 'changepoint_prior_scale': 0.42}, #mape 0.214750 365d -21.51% - 31/12/21 5.6916
            # {'ticker': 'mnrs.ax', 'changepoint_prior_scale': 0.31}, #mape 0.214996 365d -17.01% - 31/12/21 5.7785
            {'ticker': 'mnrs.ax', 'changepoint_prior_scale': 0.46}, #mape 0.204775 365d -14.18% - 31/12/21 5.844

            # {'ticker': 'moat.ax', 'changepoint_prior_scale': 0.01}, #mape 0.051501
            # {'ticker': 'moat.ax', 'changepoint_prior_scale': 0.01}, #mape 0.047211
            # {'ticker': 'moat.ax', 'changepoint_prior_scale': 0.01}, #mape 0.043347
            # {'ticker': 'moat.ax', 'changepoint_prior_scale': 0.01}, #mape 0.045689
            # {'ticker': 'moat.ax', 'changepoint_prior_scale': 0.01}, #mape 0.043959

            # {'ticker': 'ndq.ax', 'changepoint_prior_scale': 0.01},
            # {'ticker': 'ndq.ax', 'changepoint_prior_scale': 0.14},
            # {'ticker': 'ndq.ax', 'changepoint_prior_scale': 0.10},
            # {'ticker': 'ndq.ax', 'changepoint_prior_scale': 0.135506},
            # {'ticker': 'ndq.ax', 'changepoint_prior_scale': 0.11}, #mape 0.1422
            # {'ticker': 'ndq.ax', 'changepoint_prior_scale': 0.10}, #mape 0.1305
            # {'ticker': 'ndq.ax', 'changepoint_prior_scale': 0.06}, #mape 0.11099
            # {'ticker': 'ndq.ax', 'changepoint_prior_scale': 0.06}, #mape 0.101546
            # {'ticker': 'ndq.ax', 'changepoint_prior_scale': 0.05}, #mape 0.107253
            # {'ticker': 'ndq.ax', 'changepoint_prior_scale': 0.15}, #mape 0.098608
            # {'ticker': 'ndq.ax', 'changepoint_prior_scale': 0.5}, #mape 0.039191
            # {'ticker': 'ndq.ax', 'changepoint_prior_scale': 0.08}, #mape 0.025406
            # {'ticker': 'ndq.ax', 'changepoint_prior_scale': 0.06}, #mape 0.053262 16.24%
            # {'ticker': 'ndq.ax', 'changepoint_prior_scale': 0.04}, #mape 0.078174 15.01%
            # {'ticker': 'ndq.ax', 'changepoint_prior_scale': 0.04}, #mape 0.084660 17.09%
            # {'ticker': 'ndq.ax', 'changepoint_prior_scale': 0.04}, #mape 0.086998 15.28%
            # {'ticker': 'ndq.ax', 'changepoint_prior_scale': 0.02}, #mape 0.091984 16.55%
            # {'ticker': 'ndq.ax', 'changepoint_prior_scale': 0.02}, #mape 0.089002 365d 14.67% - 31/12/21 33.9305
            # {'ticker': 'ndq.ax', 'changepoint_prior_scale': 0.02}, #mape 0.097173 365d 15.86% - 31/12/21 33.9884
            # {'ticker': 'ndq.ax', 'changepoint_prior_scale': 0.01}, #mape 0.092349 365d 17.01% - 31/12/21 34.0976
            # {'ticker': 'ndq.ax', 'changepoint_prior_scale': 0.01}, #mape 0.084676 365d 21.29% - 31/12/21 34.0233
            # {'ticker': 'ndq.ax', 'changepoint_prior_scale': 0.01}, #mape 0.089959 365d 19.94% - 31/12/21 34.0086
            # {'ticker': 'ndq.ax', 'changepoint_prior_scale': 0.01}, #mape 0.095698 365d 12.73% - 31/12/21 34.0241
            # {'ticker': 'ndq.ax', 'changepoint_prior_scale': 0.01}, #mape 0.106942 365d 11.27% - 31/12/21 34.2094
            # {'ticker': 'ndq.ax', 'changepoint_prior_scale': 0.01}, #mape 0.108245 365d 12.22% - 31/12/21 34.3471
            # {'ticker': 'ndq.ax', 'changepoint_prior_scale': 0.01}, #mape 0.109281 365d 14.25% - 31/12/21 34.4403
            {'ticker': 'ndq.ax', 'changepoint_prior_scale': 0.01}, #mape 0.107946 365d 14.56% - 31/12/21 34.5026

            # {'ticker': 'rbtz.ax', 'changepoint_prior_scale': 0.01}, #mape 0.104636
            # {'ticker': 'rbtz.ax', 'changepoint_prior_scale': 0.01}, #mape 0.104374
            # {'ticker': 'rbtz.ax', 'changepoint_prior_scale': 0.01}, #mape 0.100508
            # {'ticker': 'rbtz.ax', 'changepoint_prior_scale': 0.01}, #mape 0.088163
            # {'ticker': 'rbtz.ax', 'changepoint_prior_scale': 0.01}, #mape 0.099713

            # {'ticker': 'spy.ax', 'changepoint_prior_scale': 0.01}, #mape 0.042770 365d 4.84% - 31/12/21 585.8236

            # {'ticker': 'tech.ax', 'changepoint_prior_scale': 0.50},

            # {'ticker': 'vas.ax', 'changepoint_prior_scale': 0.41}, #mape 0.145021 365d -3.89% - 31/12/21 89.6124

            # {'ticker': 'vap.ax', 'changepoint_prior_scale': 0.01}, #mape 0.184779
            # {'ticker': 'vap.ax', 'changepoint_prior_scale': 0.01}, #mape 0.177115

            # {'ticker': 'vdhg.ax', 'changepoint_prior_scale': 0.08} #mape 0.1893
            # {'ticker': 'vdhg.ax', 'changepoint_prior_scale': 0.07} #mape 0.20344
            # {'ticker': 'vdhg.ax', 'changepoint_prior_scale': 0.01}, #mape 0.218804
            # {'ticker': 'vdhg.ax', 'changepoint_prior_scale': 0.37}, #mape 0.239731
            # {'ticker': 'vdhg.ax', 'changepoint_prior_scale': 0.47}, #mape 0.103617
            # {'ticker': 'vdhg.ax', 'changepoint_prior_scale': 0.06}, #mape 0.122675
            # {'ticker': 'vdhg.ax', 'changepoint_prior_scale': 0.04}, #mape 0.146357

            # {'ticker': 'vts.ax', 'changepoint_prior_scale': 0.48} #mape 0.100424
            # {'ticker': 'vts.ax', 'changepoint_prior_scale': 0.46}, #mape 0.089874
            # {'ticker': 'vts.ax', 'changepoint_prior_scale': 0.50}, #mape 0.081716
            # {'ticker': 'vts.ax', 'changepoint_prior_scale': 0.50}, #mape 0.074273
            # {'ticker': 'vts.ax', 'changepoint_prior_scale': 0.46}, #mape 0.075568
            # {'ticker': 'vts.ax', 'changepoint_prior_scale': 0.01}, #mape 0.087665 365d -0.28% - 31/12/21 294.866

            #{'ticker': 'wxoz.ax', 'changepoint_prior_scale': 0.46}, #mape 0.044530 365d 2.75% - 31/12/21 42.2517

            # Crypto
            # {'ticker': 'ada-aud', 'changepoint_prior_scale': 0.5}, #mape 0.099397 365d 23.22% - 31/12/21 3.4321
            # {'ticker': 'ada-aud', 'changepoint_prior_scale': 0.5}, #mape 0.060881 365d 39.54% - 31/12/21 3.7317
            # {'ticker': 'ada-aud', 'changepoint_prior_scale': 0.06}, #mape 0.054723 365d 137.92% - 31/12/21 4.7248
            # {'ticker': 'ada-aud', 'changepoint_prior_scale': 0.06}, #mape 0.035837 365d 123.59% - 31/12/21 3.9581
            # {'ticker': 'ada-aud', 'changepoint_prior_scale': 0.49}, #mape 0.032304 365d -35.93% - 31/12/21 2.1864
            # {'ticker': 'ada-aud', 'changepoint_prior_scale': 0.07}, #mape 0.050877 365d 118.24% - 31/12/21 3.4132
            # {'ticker': 'ada-aud', 'changepoint_prior_scale': 0.06}, #mape 0.052293 365d 149.56% - 31/12/21 3.5006
            # {'ticker': 'ada-aud', 'changepoint_prior_scale': 0.16}, #mape 0.040169 365d % - 31/12/21
            # {'ticker': 'ada-aud', 'changepoint_prior_scale': 0.10}, #mape 0.514931 365d 91.49% - 31/12/21 2.0567
            {'ticker': 'ada-aud', 'changepoint_prior_scale': 0.09}, #mape 1.131180 365d 77.26% - 31/12/21 1.8952

            # {'ticker': 'avax-aud', 'changepoint_prior_scale': 0.22}, #mape 1.208027 365d 429.35% - 31/12/21 200.1994
            # {'ticker': 'avax-aud', 'changepoint_prior_scale': 0.01}, #mape 0.234387 365d 2.9% - 31/12/21 69.1424
            # {'ticker': 'avax-aud', 'changepoint_prior_scale': 0.01}, #mape 0.269232 365d -5.15% - 31/12/21 70.8753
            # {'ticker': 'avax-aud', 'changepoint_prior_scale': 0.01}, #mape 0.268861 365d -16.81% - 31/12/21 71.0216
            # {'ticker': 'avax-aud', 'changepoint_prior_scale': 0.01}, #mape 0.346033 365d -17.67% - 31/12/21 75.5291
            # {'ticker': 'avax-aud', 'changepoint_prior_scale': 0.01}, #mape 0.351856 365d -19.87% - 31/12/21 76.2725
            # {'ticker': 'avax-aud', 'changepoint_prior_scale': 0.01}, #mape 0.550412 365d % - 31/12/21
            # {'ticker': 'avax-aud', 'changepoint_prior_scale': 0.01}, #mape 0.604214 365d -32.24% - 31/12/21 98.201
            {'ticker': 'avax-aud', 'changepoint_prior_scale': 0.01}, #mape 0.504565 365d 109.34% - 31/12/21 115.776

            # {'ticker': 'btc-aud', 'changepoint_prior_scale': 0.47}, #mape 0.362955
            # {'ticker': 'btc-aud', 'changepoint_prior_scale': 0.47}, #mape 0.344679
            # {'ticker': 'btc-aud', 'changepoint_prior_scale': 0.01}, #mape 0.041872
            # {'ticker': 'btc-aud', 'changepoint_prior_scale': 0.01}, #mape 0.198685
            # {'ticker': 'btc-aud', 'changepoint_prior_scale': 0.01}, #mape 0.669145
            # {'ticker': 'btc-aud', 'changepoint_prior_scale': 0.01}, #mape 0.926041 169.96%
            # {'ticker': 'btc-aud', 'changepoint_prior_scale': 0.01}, #mape 1.020979 93.07%
            # {'ticker': 'btc-aud', 'changepoint_prior_scale': 0.01}, #mape 0.977486 74.24%
            # {'ticker': 'btc-aud', 'changepoint_prior_scale': 0.01}, #mape 0.970422 60.11%
            # {'ticker': 'btc-aud', 'changepoint_prior_scale': 0.01}, #mape 0.903822 365d 69.63% - 31/12/21 80,945.7489
            # {'ticker': 'btc-aud', 'changepoint_prior_scale': 0.01}, #mape 0.906683 365d 68.41% - 31/12/21 82,407.4949
            # {'ticker': 'btc-aud', 'changepoint_prior_scale': 0.42}, #mape 0.510305 365d -175.15%? - 31/12/21 47,033.0843
            # {'ticker': 'btc-aud', 'changepoint_prior_scale': 0.49}, #mape 0.506868 365d 150.7% - 31/12/21 129,048.0726
            # {'ticker': 'btc-aud', 'changepoint_prior_scale': 0.36}, #mape 0.660798 365d 223.35% - 31/12/21 139,983.0179
            # {'ticker': 'btc-aud', 'changepoint_prior_scale': 0.18}, #mape 0.026497 365d 136.87% - 31/12/21 118,246.7143
            # {'ticker': 'btc-aud', 'changepoint_prior_scale': 0.13}, #mape 0.025893 365d 135.63% - 31/12/21 116,405.1401
            # {'ticker': 'btc-aud', 'changepoint_prior_scale': 0.07}, #mape 0.037236 365d % - 31/12/21
            # {'ticker': 'btc-aud', 'changepoint_prior_scale': 0.08}, #mape 0.060787 365d 131.48% - 31/12/21 102,180.5875
            {'ticker': 'btc-aud', 'changepoint_prior_scale': 0.05}, #mape 0.077243 365d 124.42% - 31/12/21 93,474.845

            # {'ticker': 'doge-aud', 'changepoint_prior_scale': 0.49} #mape 0.986986
            # {'ticker': 'doge-aud', 'changepoint_prior_scale': 0.49} #mape 0.991088
            # {'ticker': 'doge-aud', 'changepoint_prior_scale': 0.44} #mape 0.981513
            # {'ticker': 'doge-aud', 'changepoint_prior_scale': 0.38} #mape 0.961726
            # {'ticker': 'doge-aud', 'changepoint_prior_scale': 0.47} #mape 0.924174
            # {'ticker': 'doge-aud', 'changepoint_prior_scale': 0.48} #mape 0.742291 449.02%
            # {'ticker': 'doge-aud', 'changepoint_prior_scale': 0.49} #mape 0.558224 291.19%
            # {'ticker': 'doge-aud', 'changepoint_prior_scale': 0.49} #mape 0.482191 167,71%
            # {'ticker': 'doge-aud', 'changepoint_prior_scale': 0.5} #mape 0.496846 33.99%
            # {'ticker': 'doge-aud', 'changepoint_prior_scale': 0.48}, #mape 0.544811 365d 21.36% - 31/12/21 0.4084
            # {'ticker': 'doge-aud', 'changepoint_prior_scale': 0.46}, #mape 0.494721 365d -54.47% - 31/12/21 0.3056
            # {'ticker': 'doge-aud', 'changepoint_prior_scale': 0.05}, #mape 0.358460 365d 150.92% - 31/12/21 0.3408
            # {'ticker': 'doge-aud', 'changepoint_prior_scale': 0.05}, #mape 0.327156 365d 41.78% - 31/12/21 0.3089
            # {'ticker': 'doge-aud', 'changepoint_prior_scale': 0.31}, #mape 0.036061 365d -56.64% - 31/12/21 0.2619
            # {'ticker': 'doge-aud', 'changepoint_prior_scale': 0.02}, #mape 0.049192 365d 135.13% - 31/12/21 0.4319
            # {'ticker': 'doge-aud', 'changepoint_prior_scale': 0.02}, #mape 0.068586 365d 149.41% - 31/12/21 0.4281
            # {'ticker': 'doge-aud', 'changepoint_prior_scale': 0.01}, #mape 0.389319 365d % - 31/12/21
            # {'ticker': 'doge-aud', 'changepoint_prior_scale': 0.01}, #mape 0.615227 365d 186.18% - 31/12/21 0.4028
            {'ticker': 'doge-aud', 'changepoint_prior_scale': 0.5}, #mape 0.450961 365d -160.75%? - 31/12/21 0.2097

            # {'ticker': 'dot1-aud', 'changepoint_prior_scale': 0.04}, #mape 0.491542 365d -273.6%? - 31/12/21 2.793
            # {'ticker': 'dot1-aud', 'changepoint_prior_scale': 0.03}, #mape 0.362170 365d -193.78%? - 31/12/21 15.3699
            # {'ticker': 'dot1-aud', 'changepoint_prior_scale': 0.02}, #mape 0.429097 365d -35.22% - 31/12/21 35.7784
            # {'ticker': 'dot1-aud', 'changepoint_prior_scale': 0.02}, #mape 0.078914 365d -17.18% - 31/12/21 48.3529
            # {'ticker': 'dot1-aud', 'changepoint_prior_scale': 0.01}, #mape 0.016673 365d 12.29% - 31/12/21 53.9206
            # {'ticker': 'dot1-aud', 'changepoint_prior_scale': 0.01}, #mape 0.107346 365d -12.87% - 31/12/21 54.5433
            # {'ticker': 'dot1-aud', 'changepoint_prior_scale': 0.01}, #mape 0.115900 365d -15.97% - 31/12/21 55.9931
            # {'ticker': 'dot1-aud', 'changepoint_prior_scale': 0.01}, #mape 0.232029 365d 7% - 31/12/21
            # {'ticker': 'dot1-aud', 'changepoint_prior_scale': 0.01}, #mape 0.123962 365d 108.2% - 31/12/21 56.1831
            {'ticker': 'dot1-aud', 'changepoint_prior_scale': 0.01}, #mape 0.207515 365d 169.54% - 31/12/21 55.021

            # {'ticker': 'eth-aud', 'changepoint_prior_scale': 0.48}, #mape 0.657902
            # {'ticker': 'eth-aud', 'changepoint_prior_scale': 0.49}, #mape 0.714721
            # {'ticker': 'eth-aud', 'changepoint_prior_scale': 0.50}, #mape 0.525303
            # {'ticker': 'eth-aud', 'changepoint_prior_scale': 0.48}, #mape 0.319017
            # {'ticker': 'eth-aud', 'changepoint_prior_scale': 0.05}, #mape 0.053857
            # {'ticker': 'eth-aud', 'changepoint_prior_scale': 0.01}, #mape 0.083224 188.2%
            # {'ticker': 'eth-aud', 'changepoint_prior_scale': 0.01}, #mape 0.166088 117.39%
            # {'ticker': 'eth-aud', 'changepoint_prior_scale': 0.01}, #mape 0.140434 78.28%
            # {'ticker': 'eth-aud', 'changepoint_prior_scale': 0.01}, #mape 0.049465 75.3%
            # {'ticker': 'eth-aud', 'changepoint_prior_scale': 0.01}, #mape 0.037415 365d 88.99% - 31/12/21 5,482.6559
            # {'ticker': 'eth-aud', 'changepoint_prior_scale': 0.01}, #mape 0.046395 365d 53.95% - 31/12/21 5,598.6058
            # {'ticker': 'eth-aud', 'changepoint_prior_scale': 0.01}, #mape 0.065372 365d 113.2% - 31/12/21 5,343.8595
            # {'ticker': 'eth-aud', 'changepoint_prior_scale': 0.13}, #mape 0.044688 365d 98.01% - 31/12/21 6,646.3888
            # {'ticker': 'eth-aud', 'changepoint_prior_scale': 0.01}, #mape 0.066063 365d 58.72% - 31/12/21 5,488.8566
            # {'ticker': 'eth-aud', 'changepoint_prior_scale': 0.01}, #mape 0.037121 365d 52.27% - 31/12/21 5,698.5311
            # {'ticker': 'eth-aud', 'changepoint_prior_scale': 0.01}, #mape 0.037399 365d 57.09% - 31/12/21 5,663.3119
            # {'ticker': 'eth-aud', 'changepoint_prior_scale': 0.01}, #mape 0.047643 365d % - 31/12/21
            # {'ticker': 'eth-aud', 'changepoint_prior_scale': 0.01}, #mape 0.182978 365d 84.49% - 31/12/21 5,986.1343
            {'ticker': 'eth-aud', 'changepoint_prior_scale': 0.25}, #mape 0.084006 365d 126.57% - 31/12/21 6,529.7996

            # {'ticker': 'ftm-aud', 'changepoint_prior_scale': 0.03}, #mape 0.983416 365d 2.46% - 31/12/21 3.4397
            # {'ticker': 'ftm-aud', 'changepoint_prior_scale': 0.03}, #mape 0.981094 365d 61.61% - 31/12/21 3.6361
            # {'ticker': 'ftm-aud', 'changepoint_prior_scale': 0.03}, #mape 0.980949 365d 65.45% - 31/12/21 3.6461
            # {'ticker': 'ftm-aud', 'changepoint_prior_scale': 0.03}, #mape 0.976409 365d % - 31/12/21
            # {'ticker': 'ftm-aud', 'changepoint_prior_scale': 0.03}, #mape 0.970866 365d 150.39% - 31/12/21 3.5935
            # {'ticker': 'ftm-aud', 'changepoint_prior_scale': 0.03}, #mape 0.973349 365d % - 31/12/21
            {'ticker': 'ftm-aud', 'changepoint_prior_scale': 0.03}, #mape 0.971620 365d 272.8% - 31/12/21 3.4313

            # {'ticker': 'ksm-aud', 'changepoint_prior_scale': 0.01}, #mape 1.249404 365d 31.8% - 31/12/21 556.4566
            # {'ticker': 'ksm-aud', 'changepoint_prior_scale': 0.01}, #mape 1.224055 365d 37.51% - 31/12/21 561.5833
            # {'ticker': 'ksm-aud', 'changepoint_prior_scale': 0.01}, #mape 1.294845 365d 89.69% - 31/12/21 577.5103
            # {'ticker': 'ksm-aud', 'changepoint_prior_scale': 0.01}, #mape 1.413083 365d 92.87% - 31/12/21 578.0697
            # {'ticker': 'ksm-aud', 'changepoint_prior_scale': 0.01}, #mape 1.455949 365d 71.62% - 31/12/21 571.192
            # {'ticker': 'ksm-aud', 'changepoint_prior_scale': 0.01}, #mape 1.354594 365d 38.3% - 31/12/21 566.9255
            # {'ticker': 'ksm-aud', 'changepoint_prior_scale': 0.01}, #mape 1.198022 365d 33.99% - 31/12/21 567.1289
            # {'ticker': 'ksm-aud', 'changepoint_prior_scale': 0.44}, #mape 0.749789 365d % - 31/12/21
            # {'ticker': 'ksm-aud', 'changepoint_prior_scale': 0.49}, #mape 0.804132 365d 222.92% - 31/12/21 669.1971
            # {'ticker': 'ksm-aud', 'changepoint_prior_scale': 0.5}, #mape 0.775505 365d % - 31/12/21
            {'ticker': 'ksm-aud', 'changepoint_prior_scale': 0.33}, #mape 0.100233 365d 74.57% - 31/12/21 412.1795

            # {'ticker': 'luna1-aud', 'changepoint_prior_scale': 0.05}, #mape 0.056163 365d 160.09% - 31/12/21 87.1617
            # {'ticker': 'luna1-aud', 'changepoint_prior_scale': 0.02}, #mape 0.044526 365d 108.2% - 31/12/21 67.0734
            # {'ticker': 'luna1-aud', 'changepoint_prior_scale': 0.04}, #mape 0.044522 365d 138.09% - 31/12/21 82.9233
            # {'ticker': 'luna1-aud', 'changepoint_prior_scale': 0.03}, #mape 0.136057 365d % - 31/12/21
            # {'ticker': 'luna1-aud', 'changepoint_prior_scale': 0.03}, #mape 0.146271 365d 258.83% - 31/12/21 82.1138
            {'ticker': 'luna1-aud', 'changepoint_prior_scale': 0.04}, #mape 0.529265 365d 174.09% - 31/12/21 91.3813

            # {'ticker': 'matic-aud', 'changepoint_prior_scale': 0.05}, #mape 0.672222 365d 96.86% - 31/12/21 2.6561
            # {'ticker': 'matic-aud', 'changepoint_prior_scale': 0.05}, #mape 0.690110 365d 115.05% - 31/12/21 2.656
            # {'ticker': 'matic-aud', 'changepoint_prior_scale': 0.02}, #mape 0.580596 365d % - 31/12/21
            # {'ticker': 'matic-aud', 'changepoint_prior_scale': 0.02}, #mape 0.559375 365d 87.96% - 31/12/21 2.4588
            {'ticker': 'matic-aud', 'changepoint_prior_scale': 0.10}, #mape 0.565214 365d 98.82% - 31/12/21 2.8912

            # {'ticker': 'sol1-aud', 'changepoint_prior_scale': 0.04} #mape 0.111159 365d -30.7% - 31/12/21 111.9578
            # {'ticker': 'sol1-aud', 'changepoint_prior_scale': 0.47} #mape 0.185178 365d 43.06% - 31/12/21 183.7448
            # {'ticker': 'sol1-aud', 'changepoint_prior_scale': 0.5} #mape 0.338449 365d 350.33% - 31/12/21 362.8368
            # {'ticker': 'sol1-aud', 'changepoint_prior_scale': 0.48} #mape 0.470881 365d 390.06% - 31/12/21 401.5069
            # {'ticker': 'sol1-aud', 'changepoint_prior_scale': 0.34} #mape 0.484830 365d 290.11% - 31/12/21 404.8892
            # {'ticker': 'sol1-aud', 'changepoint_prior_scale': 0.07} #mape 0.598107 365d 248.62% - 31/12/21 409.7637
            # {'ticker': 'sol1-aud', 'changepoint_prior_scale': 0.07} #mape 0.616238 365d 239.4% - 31/12/21 412.9456
            # {'ticker': 'sol1-aud', 'changepoint_prior_scale': 0.03} #mape 0.685370 365d % - 31/12/21
            # {'ticker': 'sol1-aud', 'changepoint_prior_scale': 0.02} #mape 0.680720 365d 343.04% - 31/12/21 400.3602
            # {'ticker': 'sol1-aud', 'changepoint_prior_scale': 0.03} #mape 0.681349 365d % - 31/12/21
            {'ticker': 'sol1-aud', 'changepoint_prior_scale': 0.02} #mape 0.672825 365d 360.05% - 31/12/21 371.166
        ]

        for index in range(len(self.cache_obj)):
            self.cache.set(self.cache_obj[index]['ticker'] + '_best_changepoint_prior_scale',
                self.cache_obj[index]['changepoint_prior_scale'], expire=self.cache_expire)

    def restrict_max_periods(self, periods):
        # Make sure the new number of periods to use is not bigger than 36% of the historical periods
        periods = int(periods)
        historical_periods_count = len(self.stock_info['historical_data'])
        # Estimate the number of maximum periods allowed. This was derived by trial and error
        max_periods = int(historical_periods_count * 0.36) + 1

        return max_periods if periods > max_periods else periods

    def forecaster(self):
        """
        Forecast the given ticker/quote a number of days into the future from today

        Inputs:
        ticker - is the ticker/quote of the stock as defined by Yahoo Finance
        periods - is the number of days into the future to forecast
        """

        cache_changepoint_prior_scale = self.ticker + '_best_changepoint_prior_scale'
        if not cache_changepoint_prior_scale in self.cache:
            print('No optimal changepoint_prior_scale was found in cache')
            optimal_forecast = self.make_forecast_finding_best_changepoint_prior_scale2()
            print('Results were for ticker', self.ticker)
            self.cache.set(
                cache_changepoint_prior_scale,
                optimal_forecast['changepoint_prior_scale'],
                expire = self.cache_expire
            )

            # Calculate deltas
            delta = optimal_forecast['forecast_info']['forecast']['yhat'].pct_change()
            optimal_forecast['forecast_info']['forecast'] = optimal_forecast['forecast_info']['forecast'].assign(delta = delta.values)

            fig_paths = self.make_graphs(optimal_forecast['forecast_info'])
            result = {
                'params_info': optimal_forecast['forecast_info']['params_info'],
                'forecast': optimal_forecast['forecast_info']['forecast'],
                'performance': optimal_forecast['diagnostics']['df_performance']
            }
        else:
            print('Using old changepoint_prior_scale found in cache')
            changepoint_prior_scale = self.cache.get(cache_changepoint_prior_scale)

            # Test the model using 25% of historical data as the horizon
            horizon_days = int(len(self.stock_info['historical_data']) * 0.25)

            forecast_info = self.make_forecast(changepoint_prior_scale)
            forecast_info['change_point_prior_scale'] = changepoint_prior_scale
            forecast_info['params_info']['horizon_days'] = horizon_days

            diagnostics = self.diagnose_model(horizon_days, forecast_info['model'])
            forecast_info['df_cross_validation'] = diagnostics['df_cross_validation']

            # Calculate deltas
            delta = forecast_info['forecast']['yhat'].pct_change()
            forecast_info['forecast'] = forecast_info['forecast'].assign(delta = delta.values)

            fig_paths = self.make_graphs(forecast_info)
            result = {
                'params_info': forecast_info['params_info'],
                'forecast': forecast_info['forecast'],
                'performance': diagnostics['df_performance']
            }
        result['stock_info'] = self.stock_info
        result['fig_paths'] = fig_paths
        result['returns'] = {}
        result['returns']['requested_period'] = (result['forecast'].tail(1)['yhat'].values[0] / result['stock_info']['info']['currentPrice']) -1
        result['returns']['annualised'] = result['returns']['requested_period'] / result['params_info']['periods'] * 365

        self.result = result

    #@cache.memoize(typed=True, expire=43200)  # cache for 12 hours
    def get_stock_info(self):
        """
        Retrieves stock's information from Yahoo Finance

        Inputs:
        ticker - is the ticker/quote of the stock as defined by Yahoo Finance
        """

        print('Retrieving data from Yahoo Finance')

        # Get historical data from Yahoo Finance
        stock_data = yf.Ticker(self.ticker)

        info = stock_data.info
        info['currentPrice'] = stock_data.history('1d')['Close'][0]
        # info['longBusinessSummary'] = info['longBusinessSummary'].value.decode('utf-8','ignore').encode("utf-8")

        dividends = stock_data.dividends

        # Yahoo Finance allows to retrieve historical data for:
        # 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
        historical_data = stock_data.history('max', auto_adjust=True)

        # Remove outliers. That is any close price value that is greater than 8 standard deviations
        outliers = historical_data[np.abs(historical_data.Close-historical_data.Close.mean()) > (8*historical_data.Close.std())].Close
        print('Close price values removed as outliers from historical data:', outliers)
        historical_data = historical_data[np.abs(historical_data.Close-historical_data.Close.mean()) <= (8*historical_data.Close.std())]

        return {
            'info': info,
            'dividends': dividends,
            'historical_data': historical_data
        }

    def make_forecast_finding_best_changepoint_prior_scale1(self):
        """
        Find the best changepoint prior scale to use, returning the forecast.
        According to the fphropet manual, the changepoint prior scale is probably the most
        impactful parameter: "It determines the flexibility of the trend, and in particular
        how much the trend changes at the trend changepoints. If it is too small, the trend
        will be underfit and variance that should have been modeled with trend changes will
        instead end up being handled with the noise term. If it is too large, the trend will
        overfit and in the most extreme case you can end up with the trend capturing yearly
        seasonality. The default of 0.05 works for many time series, but this could be tuned;
        a range of [0.001, 0.5] would likely be about right. Parameters like this
        (regularization penalties; this is effectively a lasso penalty) are often tuned on a
        log scale."

        This method starts with a change point prior scale of 0.01, evaluating the Mean
        Absolute Percent Error (MAPE) and continuing with the next change point until the MAPE
        starts to increase. This method choose the first minimum value and not necesarily the
        absolute minimum value.

        Inputs:
        historical_data - is the historical stock data
        periods - is the number of days to forecast
        """

        min_mape = 100
        changepoint_prior_scale = 0
        continue_loop = True
        # Test the model using 25% of historical data as the horizon
        horizon_days = int(len(self.stock_info['historical_data']) * 0.25)

        while continue_loop:
            changepoint_prior_scale += 0.01
            forecast_info = self.make_forecast(changepoint_prior_scale)
            forecast_info['params_info']['horizon_days'] = horizon_days

            diagnostics = self.diagnose_model(horizon_days, forecast_info['model'])
            forecast_info['df_cross_validation'] = diagnostics['df_cross_validation']
            mape = diagnostics['df_performance'].tail(1).mape.values

            print('mape=', mape)
            print('temp changepoint_prior_scale=', changepoint_prior_scale)
            if mape < min_mape:
                min_mape = mape
                result = {
                    'forecast_info': forecast_info,
                    'diagnostics': diagnostics,
                    'changepoint_prior_scale': changepoint_prior_scale
                }
            else:
                continue_loop = False

        print('min_mape=', min_mape)
        print('best changepoint_prior_scale=', result['changepoint_prior_scale'])
        return result

    def make_forecast_finding_best_changepoint_prior_scale2(self):
        """
        Find the best changepoint prior scale to use, returning the forecast.
        According to the fphropet manual, the changepoint prior scale is probably the most
        impactful parameter: "It determines the flexibility of the trend, and in particular
        how much the trend changes at the trend changepoints. If it is too small, the trend
        will be underfit and variance that should have been modeled with trend changes will
        instead end up being handled with the noise term. If it is too large, the trend will
        overfit and in the most extreme case you can end up with the trend capturing yearly
        seasonality. The default of 0.05 works for many time series, but this could be tuned;
        a range of [0.001, 0.5] would likely be about right. Parameters like this
        (regularization penalties; this is effectively a lasso penalty) are often tuned on a
        log scale."

        This method evaluates all point prior scale from 0.01 until 0.5 with a 0.01 step,
        choosing the value producing the minimum Mean Absolute Percent Error (MAPE). As it
        evaluates every sinlge point it finds the absolute minimum in the range evaluated
        at the cost of speed.

        Inputs:
        historical_data - is the historical stock data
        periods - is the number of days to forecast
        """

        start = time.time()
        # Test the model using 25% of historical data as the horizon
        horizon_days = int(len(self.stock_info['historical_data']) * 0.25)

        stats = []
        result_min_mape = {'mape': 100}

        # Loop from 0.01 to 0.5. n.arange doesn't include the stop, but the element before.
        for changepoint_prior_scale in np.arange(0.01, 0.51, 0.01):
            # for changepoint_prior_scale in np.arange(0.01, 0.02, 0.01):
            forecast_info = self.make_forecast(changepoint_prior_scale)
            forecast_info['params_info']['horizon_days'] = horizon_days

            diagnostics = self.diagnose_model(horizon_days, forecast_info['model'])
            forecast_info['df_cross_validation'] = diagnostics['df_cross_validation']
            mape = diagnostics['df_performance'].tail(1).mape.values[0]
            print('Evaluating changepoint_prior_scale=', changepoint_prior_scale)

            stat = {
                'changepoint_prior_scale': changepoint_prior_scale,
                'mape': mape
            }
            stats.append(stat)
            print(pd.DataFrame(stats).reindex(
                columns=['changepoint_prior_scale', 'mape']))

            if mape < result_min_mape['mape']:
                result_min_mape = {
                    'forecast_info': forecast_info,
                    'diagnostics': diagnostics,
                    'changepoint_prior_scale': changepoint_prior_scale,
                    'mape': mape
                }

            print('min mape so far =', result_min_mape['mape'])
            print('with changepoint_prior_scale=',
                result_min_mape['changepoint_prior_scale'])
            print('time=', time.time() - start)

        print('best changepoint_prior_scale=',
            result_min_mape['changepoint_prior_scale'])
        print('min mape=', result_min_mape['mape'])
        return result_min_mape

    def make_forecast_finding_best_changepoint_prior_scale3(self):
        """
        Find the best changepoint prior scale to use, returning the forecast.
        According to the fphropet manual, the changepoint prior scale is probably the most
        impactful parameter: "It determines the flexibility of the trend, and in particular
        how much the trend changes at the trend changepoints. If it is too small, the trend
        will be underfit and variance that should have been modeled with trend changes will
        instead end up being handled with the noise term. If it is too large, the trend will
        overfit and in the most extreme case you can end up with the trend capturing yearly
        seasonality. The default of 0.05 works for many time series, but this could be tuned;
        a range of [0.001, 0.5] would likely be about right. Parameters like this
        (regularization penalties; this is effectively a lasso penalty) are often tuned on a
        log scale."

        This method use a ternary search to find the a change point prior scale value that
        produces a minimum Mean Absolute Percent Error (MAPE). As the evaluated function is
        not necessarily and unimodal function, the minimum value found might not be necesarily
        be the minimum value.

        With some modifications this method can also use the golden section search algorithm

        Inputs:
        historical_data - is the historical stock data
        periods - is the number of days to forecast

        @todo refactor the method to remove double code
        """
        start = time.time()
        # Test the model using 25% of historical data as the horizon
        horizon_days = int(len(self.stock_info['historical_data']) * 0.25)

        stats = []

        left_cps = 0.001 # left changepoint prior scale
        right_cps = 0.5 # right changepoint prior scale
        precision = 0.01

        golden_ratio = (math.sqrt(5) +1) / 2

        while abs(right_cps - left_cps) >= precision:
            # left_cps_third = left_cps + (right_cps - left_cps) / 3
            # right_cps_third = right_cps - (right_cps - left_cps) / 3
            left_cps_third = right_cps - (right_cps - left_cps) / golden_ratio
            right_cps_third = left_cps + (right_cps - left_cps) / golden_ratio

            print('evaluating left ', left_cps_third)

            forecast_info = self.make_forecast(left_cps_third)
            forecast_info['params_info']['horizon_days'] = horizon_days

            diagnostics = self.diagnose_model(horizon_days, forecast_info['model'])
            forecast_info['df_cross_validation'] = diagnostics['df_cross_validation']
            left_mape = diagnostics['df_performance'].tail(1).mape.values[0]

            stat = {
                'changepoint_prior_scale': left_cps_third,
                'mape': left_mape
            }
            stats.append(stat)
            print(pd.DataFrame(stats).reindex(
                columns=['changepoint_prior_scale', 'mape']))

            print('evaluating right ', right_cps_third)
            forecast_info = self.make_forecast(right_cps_third)
            forecast_info['params_info']['horizon_days'] = horizon_days

            diagnostics = self.diagnose_model(horizon_days, forecast_info['model'])
            forecast_info['df_cross_validation'] = diagnostics['df_cross_validation']
            right_mape = diagnostics['df_performance'].tail(1).mape.values[0]

            stat = {
                'changepoint_prior_scale': right_cps_third,
                'mape': right_mape
            }
            stats.append(stat)
            print(pd.DataFrame(stats).reindex(
                columns=['changepoint_prior_scale', 'mape']))

            if left_mape > right_mape:
                left_cps = left_cps_third
            else:
                right_cps = right_cps_third

            print('time=', time.time() - start)

        best_changepoint_prior_scale = (left_cps + right_cps) / 2
        print('evaluating best ', best_changepoint_prior_scale)
        forecast_info = self.make_forecast(best_changepoint_prior_scale)
        forecast_info['params_info']['horizon_days'] = horizon_days

        diagnostics = self.diagnose_model(horizon_days, forecast_info['model'])
        forecast_info['df_cross_validation'] = diagnostics['df_cross_validation']
        mape = diagnostics['df_performance'].tail(1).mape.values[0]

        stat = {
            'changepoint_prior_scale': best_changepoint_prior_scale,
            'mape': mape
        }
        stats.append(stat)
        print(pd.DataFrame(stats).reindex(
            columns=['changepoint_prior_scale', 'mape']))
        print('time=', time.time() - start)

        result_min_mape = {
            'forecast_info': forecast_info,
            'diagnostics': diagnostics,
            'changepoint_prior_scale': best_changepoint_prior_scale,
            'mape': mape
        }

        return result_min_mape

    def make_forecast_finding_best_changepoint_prior_scale4(self):
        """
        Find the best changepoint prior scale to use, returning the forecast.
        According to the fphorpet manual, the changepoint prior scale is probably the most
        impactful parameter: "It determines the flexibility of the trend, and in particular
        how much the trend changes at the trend changepoints. If it is too small, the trend
        will be underfit and variance that should have been modeled with trend changes will
        instead end up being handled with the noise term. If it is too large, the trend will
        overfit and in the most extreme case you can end up with the trend capturing yearly
        seasonality. The default of 0.05 works for many time series, but this could be tuned;
        a range of [0.001, 0.5] would likely be about right. Parameters like this
        (regularization penalties; this is effectively a lasso penalty) are often tuned on a
        log scale."

        Similarly to the third method above, this method uses the a Golden section search
        algorithm to find the a change point prior scale value that produces a minimum Mean
        Absolute Percent Error (MAPE). As the evaluated function is not necessarily and unimodal
        function, the minimum value found might not be necesarily the minimum value. The
        difference with the third method above is that this method reuses function evaluations,
        saving evaluations per iteration, saving time in the process.

        Inputs:
        historical_data - is the historical stock data
        periods - is the number of days to forecast

        @todo refactor the method to remove double code
        """

        start = time.time()
        # Test the model using 25% of historical data as the horizon
        horizon_days = int(len(self.stock_info['historical_data']) * 0.25)

        stats = []
        #result_min_mape = {'mape': 100}

        left_cps = 0.001 # left changepoint prior scale
        right_cps = 0.5 # right changepoint prior scale
        precision = 0.01

        # based on golden ratio (math.sqrt(5) +1) / 2
        inv_phi = (math.sqrt(5)-1) / 2 # 1 / phi
        inv_phi2 = (3 - math.sqrt(5)) / 2 # 1 / phi^2

        distance = right_cps - left_cps
        max_steps = int(math.ceil(math.log(precision/distance) / math.log(inv_phi)))

        left_cps_tmp = left_cps + inv_phi2 * distance
        right_cps_tmp = left_cps + inv_phi * distance

        print('evaluating left ', left_cps_tmp)
        forecast_info = self.make_forecast(left_cps_tmp)
        forecast_info['params_info']['horizon_days'] = horizon_days

        diagnostics = self.diagnose_model(horizon_days, forecast_info['model'])
        forecast_info['df_cross_validation'] = diagnostics['df_cross_validation']
        left_mape = diagnostics['df_performance'].tail(1).mape.values[0]

        stat = {
            'changepoint_prior_scale': left_cps_tmp,
            'mape': left_mape
        }
        stats.append(stat)
        print(pd.DataFrame(stats).reindex(
            columns=['changepoint_prior_scale', 'mape']))

        print('evaluating right ', right_cps_tmp)
        forecast_info = self.make_forecast(right_cps_tmp)
        forecast_info['params_info']['horizon_days'] = horizon_days

        diagnostics = self.diagnose_model(horizon_days, forecast_info['model'])
        forecast_info['df_cross_validation'] = diagnostics['df_cross_validation']
        right_mape = diagnostics['df_performance'].tail(1).mape.values[0]

        stat = {
            'changepoint_prior_scale': right_cps_tmp,
            'mape': right_mape
        }
        stats.append(stat)
        print(pd.DataFrame(stats).reindex(
            columns=['changepoint_prior_scale', 'mape']))

        for i in range(max_steps):
            if left_mape < right_mape:
                right_cps = right_cps_tmp
                right_cps_tmp = left_cps_tmp
                right_mape = left_mape
                distance = inv_phi * distance
                left_cps_tmp = left_cps + inv_phi2 * distance

                print('evaluating left ', left_cps_tmp)
                forecast_info = self.make_forecast(left_cps_tmp)
                forecast_info['params_info']['horizon_days'] = horizon_days

                diagnostics = self.diagnose_model(horizon_days, forecast_info['model'])
                forecast_info['df_cross_validation'] = diagnostics['df_cross_validation']
                left_mape = diagnostics['df_performance'].tail(1).mape.values[0]

                stat = {
                    'changepoint_prior_scale': left_cps_tmp,
                    'mape': left_mape
                }
                stats.append(stat)
                print(pd.DataFrame(stats).reindex(
                    columns=['changepoint_prior_scale', 'mape']))
                print('time=', time.time() - start)
            else:
                left_cps = left_cps_tmp
                left_cps_tmp = right_cps_tmp
                left_mape = right_mape
                distance = inv_phi * distance
                right_cps_tmp = left_cps + inv_phi * distance

                print('evaluating right ', right_cps_tmp)
                forecast_info = self.make_forecast(right_cps_tmp)
                forecast_info['params_info']['horizon_days'] = horizon_days

                diagnostics = self.diagnose_model(horizon_days, forecast_info['model'])
                forecast_info['df_cross_validation'] = diagnostics['df_cross_validation']
                right_mape = diagnostics['df_performance'].tail(1).mape.values[0]

                stat = {
                    'changepoint_prior_scale': right_cps_tmp,
                    'mape': right_mape
                }
                stats.append(stat)
                print(pd.DataFrame(stats).reindex(
                    columns=['changepoint_prior_scale', 'mape']))
                print('time=', time.time() - start)

        if left_mape < right_mape:
            print('range ', left_cps, right_cps_tmp)
            print('best ', (left_cps + right_cps_tmp) / 2)
            best_changepoint_prior_scale = (left_cps + right_cps_tmp) / 2
        else:
            print('range ', left_cps_tmp, right_cps)
            print('best ', (left_cps_tmp + right_cps) / 2)
            best_changepoint_prior_scale = (left_cps_tmp + right_cps) / 2

        print('evaluating best ', best_changepoint_prior_scale)
        forecast_info = self.make_forecast(best_changepoint_prior_scale)
        forecast_info['params_info']['horizon_days'] = horizon_days

        diagnostics = self.diagnose_model(horizon_days, forecast_info['model'])
        forecast_info['df_cross_validation'] = diagnostics['df_cross_validation']
        mape = diagnostics['df_performance'].tail(1).mape.values[0]

        stat = {
            'changepoint_prior_scale': best_changepoint_prior_scale,
            'mape': mape
        }
        stats.append(stat)
        print(pd.DataFrame(stats).reindex(
            columns=['changepoint_prior_scale', 'mape']))
        print('time=', time.time() - start)

        result_min_mape = {
            'forecast_info': forecast_info,
            'diagnostics': diagnostics,
            'changepoint_prior_scale': best_changepoint_prior_scale,
            'mape': mape
        }

        return result_min_mape

    def make_forecast(self, changepoint_prior_scale=0.05):
        """
        Forecast the price of the stock on a future number of days
        Inputs:
        historical_data - is the historical stock data
        periods - is the number of days to forecast
        """

        # Prophet requires the dates (ds) and adjusted closing prices (y)
        # Create new data frame with the required data
        df_historical_data = pd.DataFrame()
        df_historical_data['ds'] = self.stock_info['historical_data'].index.values
        df_historical_data['y'] = self.stock_info['historical_data']['Close'].values
        # Set minimum posible value
        # df_historical_data['floor'] = 0

        # Create a Prophet model
        # As there is one single closing price daily, disable the daily seasonality
        model = Prophet(
            daily_seasonality=False,
            changepoint_prior_scale=changepoint_prior_scale
        )

        #model = Prophet(
        #    growth='logistic',
        #    #seasonality_mode='multiplicative',
        #    daily_seasonality=False,
        #    changepoint_prior_scale=changepoint_prior_scale
        #)
        #df_historical_data['floor'] = 0
        #df_historical_data['cap'] = 1.2 * df_historical_data['y'].max()

        #df_historical_data['y'] = np.log(1 + df_historical_data['y'])
        #model = Prophet(
        #    daily_seasonality=False,
        #    changepoint_prior_scale=changepoint_prior_scale
        #)

        # model.add_country_holidays(country_name='AU')
        model.fit(df_historical_data)

        total_future = model.make_future_dataframe(self.periods, freq='D')

        #total_future['floor'] = 0
        #total_future['cap'] = 1.2 * df_historical_data['y'].max()

        # As the stock exchange is closed on weekends, remove weekends in the future
        future_weekdays = total_future[total_future['ds'].dt.dayofweek < 5]
        # Enable for crypto
        future_weekdays = total_future

        # As some days were removed, recalculate number of available periods to display
        future_weekdays_count = self.periods - \
            (len(total_future) - len(future_weekdays))

        full_forecast = model.predict(future_weekdays)

        #model.history['y'] = np.exp(model.history['y']) -1
        #df_historical_data['y'] = np.exp(df_historical_data['y']) -1
        #for col in ['yhat', 'yhat_lower', 'yhat_upper', 'trend']:
        #    full_forecast[col] = np.exp(full_forecast[col]) -1

        # Return requested period
        # forecast = full_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(available_periods+1)
        forecast = full_forecast.tail(future_weekdays_count+1)

        result = {
            'historical_data': df_historical_data,
            'full_forecast': full_forecast,
            'forecast': forecast,
            'model': model,
            'params_info': {
                'periods': self.periods,
                'historical_periods': len(self.stock_info['historical_data']),
                'weekday_periods': future_weekdays_count,
                'changepoint_prior_scale': changepoint_prior_scale,
            }
        }

        return result

    def diagnose_model(self, horizon_days, model):
        """
        Diagnose the model

        Inputs:
        horizon_days - is the number of days to use when testing the model
        model - is the Phropet model
        """

        horizon = str(horizon_days) + ' days'

        df_cross_validation = cross_validation(model, horizon=horizon, parallel='processes')

        df_performance = performance_metrics(df_cross_validation)
        # print(df_performance)

        return {'df_cross_validation': df_cross_validation, 'df_performance': df_performance}

    def make_graphs(self, stock_data):
        """
        Create and save graphs on a directory for the browser to pickup

        Inputs:
        ticker - Is the ticker/quote of the stock as defined by Yahoo Finance
        stock_data - Dictionary containing information about the stock
        """
        # Save graphs
        fig_location = '/static/img/figures/'
        fig_paths = {
            'price': fig_location + 'price_' + self.ticker + '.png',
            'components': fig_location + 'components_' + self.ticker + '.png',
            # 'forecast': fig_location + 'forecast_' + ticker + '.png',
            'mape': fig_location + 'mape_' + self.ticker + '.png'
        }

        date_now = datetime.now()

        # Price & Forecast
        fig_price = plt.figure(facecolor='w', figsize=(10, 6))

        plt.title(self.ticker + ' - close price & forecast', fontsize=10, pad=1)
        plt.xlabel('Day (ds)', fontsize=10)
        plt.ylabel('Price (y)', fontsize=10)

        # plot changes in price and significate changes in price
        for changepoint in stock_data['model'].changepoints:
            plt.axvline(changepoint, color="lightsalmon", linestyle=":")

        signif_changepoint_threshold = 0.01
        signif_changepoints = stock_data['model'].changepoints[np.abs(np.nanmean(
            stock_data['model'].params['delta'], axis=0)) >= signif_changepoint_threshold] if len(stock_data['model'].changepoints) > 0 else []
        for signif_changepoint in signif_changepoints:
            plt.axvline(signif_changepoint, color='r', linestyle=':')

        # plot trend
        plt.plot(stock_data['full_forecast']['ds'],
                stock_data['full_forecast']['trend'], color='r')

        # plot historical data
        plt.plot(stock_data['historical_data']['ds'],
                stock_data['historical_data']['y'], color='k', linewidth=1)

        # plot forecast
        #@todo find a way to limit lower forecasted value to zero
        # in the meantime limit the y axis to zero
        if stock_data['full_forecast']['yhat_lower'].min() < 0:
            plt.ylim(0, stock_data['full_forecast']['yhat_upper'].max())

        plt.plot(
            stock_data['full_forecast']['ds'],
            stock_data['full_forecast']['yhat']
        )
        plt.fill_between(
            stock_data['full_forecast']['ds'], stock_data['full_forecast']['yhat_lower'],
            stock_data['full_forecast']['yhat_upper'], color='#0072B2', alpha=0.2
        )

        # plot today line
        plt.axvline(date_now, color='silver', linestyle=':')

        # plot grid
        plt.grid(True, which='major', color='gray', linestyle='-', linewidth=1, alpha=0.2)

        fig_price.savefig('../app' + fig_paths['price'])

        # Forecast
        # fig_forecast = stock_data['model'].plot(stock_data['full_forecast'])
        # add_changepoints_to_plot(
        #     fig_forecast.gca(), stock_data['model'], stock_data['full_forecast'])
        # plt.margins(x=0)
        # plt.title(ticker + ' price forecast', fontsize=10, pad=1)
        # plt.xlabel('Day (ds)', fontsize=10)
        # plt.ylabel('Price (y)', fontsize=10)
        # plt.axvline(date_now, color='k', linestyle=':')
        # fig_forecast.savefig('../app' + fig_paths['forecast'])

        # Components
        stock_data['model'].plot_components(stock_data['full_forecast']).savefig(
            '../app' + fig_paths['components'])

        # Performance - Cross validation of the percentage error (MAPE)
        plot_cross_validation_metric(stock_data['df_cross_validation'], metric='mape').savefig(
            '../app' + fig_paths['mape'])

        plt.close('all')

        return fig_paths
