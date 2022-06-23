from sklearn.model_selection import train_test_split
import os
import csv
import config as cf

data_lists = [str(i) for i in range(2,cf.DATA_NUM)]

train_datas, test_datas = train_test_split(data_lists, test_size=cf.TEST_RATE_FOR_ALL)
train_datas, val_datas = train_test_split(train_datas, test_size=cf.VAL_RATE_FOR_TRAIN)

label_dic = {}
csv_file = os.path.join(cf.ROOT_DIR, "label.csv")
with open (csv_file) as f:
    reader = csv.reader(f)
    for r in reader:
        label_dic[r[0]] = [int(r[1]), int(r[2]), int(r[3])]

# for i in data_lists:
#     label_dic[str(i)] = [85, 50, 50]

with open(cf.TRAIN_DATA_CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    for data in train_datas:
        writer.writerow([data, label_dic[data][0], label_dic[data][1], label_dic[data][2]])

with open(cf.VAL_DATA_CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    for data in val_datas:
        writer.writerow([data, label_dic[data][0], label_dic[data][1], label_dic[data][2]])

with open(cf.TEST_DATA_CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    for data in test_datas:
        writer.writerow([data, label_dic[data][0], label_dic[data][1], label_dic[data][2]])