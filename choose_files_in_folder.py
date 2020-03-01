# Python3 code to rename multiple files in a directory or folder

import os
import csv
import uuid


def pick_files(folder_name, prefix, number_of_files=300):
    csv_columns = ['file', 'digit']
    dict_data = []
    csv_file_name = "train.csv"
    try:
        with open(csv_file_name, 'a') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
            for i, file_name in enumerate(os.listdir(folder_name)[:number_of_files]):
                #new_file_name = "{}_{}_{}.wav".format(prefix, i, str(uuid.uuid4()))
                #os.rename(folder_name + "/" + file_name, folder_name + "/" + new_file_name)
                dict_data.append({"file": folder_name + "/" + file_name, "digit": prefix})
            for data in dict_data:
                writer.writerow(data)
    except IOError:
        print("I/O error")


def main():
    pick_files("/Users/liorpollak/Downloads/data-speech_commands_v0.02/one", "one")
    pick_files("/Users/liorpollak/Downloads/data-speech_commands_v0.02/two", "two")
    pick_files("/Users/liorpollak/Downloads/data-speech_commands_v0.02/three", "three")
    pick_files("/Users/liorpollak/Downloads/data-speech_commands_v0.02/four", "four")
    pick_files("/Users/liorpollak/Downloads/data-speech_commands_v0.02/five", "five")
    pick_files("/Users/liorpollak/Downloads/data-speech_commands_v0.02/six", "six")
    pick_files("/Users/liorpollak/Downloads/data-speech_commands_v0.02/seven", "seven")
    pick_files("/Users/liorpollak/Downloads/data-speech_commands_v0.02/eight", "eight")
    pick_files("/Users/liorpollak/Downloads/data-speech_commands_v0.02/nine", "nine")


if __name__ == '__main__':
    main()
