from sklearn.model_selection import train_test_split
import os
import csv
import config as cf
import glob
import random
import shutil

def all():
    root_dir = "C:/Users/masuda/Desktop/crop_32"
    output_path = "./csv_data"
    data_dic = {}
    dir_paths = glob.glob(os.path.join(root_dir, "*"))
    for dir_path in dir_paths:
        csv_path = os.path.join(dir_path, "label.csv")
        with open(csv_path) as f2:
            reader = csv.reader(f2)
            for row in reader:
                # writer.writerow([os.path.join(dir_path, row[0]), row[1], row[2], row[3]])
                data_dic[os.path.join(dir_path, row[0])] = [row[1], row[2], row[3]]
    
    data_list = list(data_dic.keys())
    print(data_list[0])
    random.shuffle(data_list)
    print(data_list[0])

    num = int(len(data_list) / 5) 
    dl1 = data_list[0:num]
    dl2 = data_list[num:2*num]
    dl3 = data_list[2*num:3*num]
    dl4 = data_list[3*num:4*num]
    dl5 = data_list[4*num:]

    dls = [dl1, dl2, dl3, dl4, dl5]

    for i in range(5):
        data_list = []
        for idx, dl in enumerate(dls):
            if idx == i:
                continue
            data_list += dl
        

        train_datas, val_datas = train_test_split(data_list, test_size=cf.VAL_RATE_FOR_TRAIN)

        with open(os.path.join(output_path, f"train_{i}.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            count=0
            for data in train_datas:
                writer.writerow([data, data_dic[data][0], data_dic[data][1], data_dic[data][2]])
                count+=1
            print(count)

        with open(os.path.join(output_path, f"val_{i}.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            count=0
            for data in val_datas:
                writer.writerow([data, data_dic[data][0], data_dic[data][1], data_dic[data][2]])
                count+=1
            print(count)

        with open(os.path.join(output_path, f"test_{i}.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            count=0
            for data in dls[i]:
                writer.writerow([data, data_dic[data][0], data_dic[data][1], data_dic[data][2]])
                count+=1
            print(count)

def divide():
    root_dir = "C:/Users/masuda/Documents/data/train_data/2205_data_small_3"
    output_path = "./csv_data"
    
    dir_paths = glob.glob(os.path.join(root_dir, "*"))

    for idx in range(len(dir_paths)):
        data_dic = {}
        test_csv_path = os.path.join(dir_paths[idx], "label.csv")
        # shutil.copy(test_csv_path, os.path.join(output_path, f"test_{idx}.csv"))


        for j, dir_path in enumerate(dir_paths):
            if j == idx:
                with open(os.path.join(output_path, f"test_{idx}.csv"), "w", newline="") as fff1:
                    writer = csv.writer(fff1)
                    with open(test_csv_path) as fff2:
                        reader = csv.reader(fff2)
                        for row in reader:
                            writer.writerow([os.path.join(dir_path, row[0]), row[1], row[2], row[3]])
                continue
            csv_path = os.path.join(dir_path, "label.csv")
            with open(csv_path) as f2:
                reader = csv.reader(f2)
                for row in reader:
                    # writer.writerow([os.path.join(dir_path, row[0]), row[1], row[2], row[3]])
                    data_dic[os.path.join(dir_path, row[0])] = [row[1], row[2], row[3]]
    
            data_list = list(data_dic.keys())
            print(data_list[0])

            train_datas, val_datas = train_test_split(data_list, test_size=cf.VAL_RATE_FOR_TRAIN)

            with open(os.path.join(output_path, f"train_{idx}.csv"), "w", newline="") as f:
                writer = csv.writer(f)
                count=0
                for data in train_datas:
                    writer.writerow([data, data_dic[data][0], data_dic[data][1], data_dic[data][2]])
                    count+=1
                print(count)

            with open(os.path.join(output_path, f"val_{idx}.csv"), "w", newline="") as f:
                writer = csv.writer(f)
                count=0
                for data in val_datas:
                    writer.writerow([data, data_dic[data][0], data_dic[data][1], data_dic[data][2]])
                    count+=1
                print(count)

            
if __name__ == "__main__":
    all()
    # divide()