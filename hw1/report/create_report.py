import numpy as np
from scipy import stats
from scipy.stats import iqr
import os
import csv
import matplotlib.pyplot as plt
import seaborn as sns

def sanitize_data(data):
    print('len', len(data))
    print('mode', stats.mode(data))
    print('mean', np.mean(data))
    print('std', data.std())

    def remove_outside_quartile(data):
        z1 = -0.67
        z3 = 0.67
        mean = data.mean()
        std = data.std()
        q1 = z1 * std + mean
        q3 = z3 * std + mean

        current_iqr = iqr(data)
        top_outlier_fence = q3 + 10.0 * current_iqr
        lower_outlier_fence = q1  - 10.0 * current_iqr

        print('current_iqr', current_iqr)
        print('top_outlier_fence', top_outlier_fence)
        print('lower_outlier_fence', lower_outlier_fence)

        new_data = [datum for datum in data if datum <= top_outlier_fence and datum >= lower_outlier_fence]
        return np.asarray(new_data)

    print('cleaned')
    cleaned_data = remove_outside_quartile(data)
    print('len', len(cleaned_data))

    print('mode', stats.mode(cleaned_data))
    print('mean', cleaned_data.mean())
    print('std', cleaned_data.std())

    set_data = set(data)
    set_cleaned_data = set(cleaned_data)
    print('diff', set_data - set_cleaned_data)
    return cleaned_data

def generate_report(data):
    print('about to generate report')
    fig = plt.figure()

    #xdata = np.array([0,1,2,3,4,5,6])/5
    #sns.tsplot(time=xdata, data=data, color='r', linestyle='−')
    plt.plot(data)
    fig.savefig("foo.pdf", bbox_inches='tight')


def main():
    files = ['hopper.csv', 'ant.csv']

    for csv_file_name in files:
        print('======================')
        print('processing filename', csv_file_name)
        data = []
        with open(os.path.dirname(os.path.realpath(__file__)) + '/' + csv_file_name) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                data.append(float(row[0]))
        data = np.asarray(data, dtype=np.float32)
        sanitized_data = sanitize_data(data)

        generate_report(sanitized_data)

if __name__ == '__main__':
    main()
