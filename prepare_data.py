import numpy as np
import os
import glob
import json
import pandas as pd


def timeit(func):
    def wrap(*args, **kwargs):
        start = pd.Timestamp.now()
        res = func(*args, **kwargs)
        elapsed = pd.Timestamp.now() - start
        print(f'elapsed {elapsed}')
        return res
    return wrap


@timeit
def main():
    idir = 'data/opendatascience Slack export Mar 12 2015 - Nov 3 2019 2'
    ofile_path = 'data/ods_slack_all2.csv'
    lines = []
    for channel_dir in glob.glob(f'{idir}/*[!.json]'):
        channel_files = glob.glob(f'{channel_dir}/*.json')
        channel = os.path.basename(channel_dir)
        print(channel, len(channel_files))
        for ifile_path in channel_files:
            with open(ifile_path, encoding='utf-8') as ifile:
                data_json_list = json.load(ifile)
                data_json_new_list = []
                for x in data_json_list:
                    x['channel'] = channel
                    data_json_new_list.append(x)
                lines.extend(data_json_new_list)
    df = pd.DataFrame(lines)
    df.to_csv(ofile_path, encoding='utf-8')
    # print(df.to_string())


if __name__ == '__main__':
    main()
