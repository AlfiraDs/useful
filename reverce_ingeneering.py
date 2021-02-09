import itertools
import collections

import pandas as pd
import numpy as np
from tqdm import tqdm

cats = [
    # 'ContractCategorisation',
    # 'Status',
    'PaymentStatus',
    # 'RemedyMethod',
    # 'RemainInTranche'
]
nums = collections.OrderedDict([
    # ('HandsetReceivableFaceAmount', [-1, 0, 1]),
    # ('MonthlyHandsetReceivablePayment', [-1, 0, 1]),
    ('OpeningAmount', [-1, 0, 1]),
    ('ClosingAmount', [-1, 0, 1]),
    # ('Payment', [0, 1]),
    # ('BuybackOutstandingFaceAmount', [-1, 0, 1]),
    # ('RepurchasePriceOfBuybacks', [-1, 0, 1]),
    # ('TerminationCompensation', [-1, 0, 1]),
    ('BadDebtBBOutstandingFaceAmount', [-1, 0, 1]),
    # ('RepurchasePriceOfBadDebtBB', [-1, 0, 1]),
    # ('TotalDelinquency', [-1, 0, 1])
])
df = pd.read_csv("")
df = df[cats + list(nums.keys())]
df['filters'] = df[cats].apply(lambda x: ' - '.join(x), axis=1)
filters = df[cats].drop_duplicates().apply(lambda x: ' - '.join(x), axis=1)

filters_combs = itertools.product(*[[0, 1]] * filters.shape[0])
for filter in tqdm(list(filters_combs)):
    mapping = collections.OrderedDict(zip(filters.values, filter))
    major_mult = df['filters'].map(mapping).to_numpy()
    if major_mult.sum() == 0:
        continue


    rep_filter = df[cats].drop_duplicates()
    rep_filter['status'] = filter
    rep_filter = rep_filter.loc[rep_filter['status'] == 1]
    # print(rep)

    # print(pd.DataFrame({' - '.join(cats): filters, 'filter': filter}))

    m = list(itertools.product(*nums.values()))

    rep = pd.DataFrame(columns=m)

    a = np.array(m, dtype='int32')
    vals = (df.loc[major_mult != 0, nums.keys()].to_numpy() * major_mult[major_mult !=0].reshape(-1, 1)).astype('float32')
    b = vals.dot(a.T)

    rep.loc[0, :] = b.sum(axis=0)
    # print(rep)
    found_idx = np.argwhere(abs(b.sum(axis=0) - 5715509.98999913) <= 10)

    if len(found_idx) != 0:
        print('rows taken:', major_mult.sum())
        print(rep_filter)
        print('Found at:', found_idx)
        # print('---', b.sum(axis=0)[abs(b.sum(axis=0) - 4675324.82) <= 10])
        print('Multipliers:', np.array(m)[found_idx.flatten()])
        print('Values:', np.round(b.sum(axis=0), 2)[found_idx.flatten()])

    # for mults in itertools.product(*nums.values()):
    #     vals = df[nums.keys()].to_numpy() * major_mult.to_numpy().reshape(-1, 1)
    #
    #     val = df[nums.keys()].values * mults * major_mult.values.reshape(-1, 1)
