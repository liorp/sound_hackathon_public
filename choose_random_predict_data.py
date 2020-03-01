# Python3 code to rename multiple
# files in a directory or folder

# importing os module
import os
import csv
import uuid
import random

CSV_FILE_NAME = "train.csv"


def rename(folder_name, prefix, number_of_files=300):
    csv_columns = ['file', 'digit']
    dict_data = []
    try:
        with open(CSV_FILE_NAME, 'a') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
            file_list = os.listdir(folder_name)
            random.shuffle(file_list)
            file_list = file_list[:number_of_files]
            for file_name in file_list:
                dict_data.append({"file": folder_name + "/" + file_name, "digit": prefix})
            for data in dict_data:
                writer.writerow(data)
    except IOError:
        print("I/O error")


def main():
    rename("/Users/liorpollak/Downloads/data-speech_commands_v0.02/one", "one")
    rename("/Users/liorpollak/Downloads/data-speech_commands_v0.02/two", "two")
    rename("/Users/liorpollak/Downloads/data-speech_commands_v0.02/three", "three")
    rename("/Users/liorpollak/Downloads/data-speech_commands_v0.02/four", "four")
    rename("/Users/liorpollak/Downloads/data-speech_commands_v0.02/five", "five")
    rename("/Users/liorpollak/Downloads/data-speech_commands_v0.02/six", "six")
    rename("/Users/liorpollak/Downloads/data-speech_commands_v0.02/seven", "seven")
    rename("/Users/liorpollak/Downloads/data-speech_commands_v0.02/eight", "eight")
    rename("/Users/liorpollak/Downloads/data-speech_commands_v0.02/nine", "nine")


if __name__ == '__main__':
    main()
