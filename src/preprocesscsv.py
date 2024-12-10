import os
import csv
import argparse
import pandas as pd
from dvclive import Live


def preprocess_csv(input_csv, output_csv):
    """
    Функция preprocess_csv принимает два аргумента: путь к входному CSV-файлу и путь к выходному CSV-файлу. Она читает данные из входного файла, добавляет новую колонку с диагнозом и записывает обновленные данные в выходной файл. Если ни одно из условий для определения диагноза не выполняется, в колонку записывается значение 'Unknown'
    """
    # Open the input CSV file
    with open(input_csv, 'r') as input_file:
        reader = csv.reader(input_file)
        rows = list(reader)

    # Open the output CSV file
    with open(output_csv, 'w', newline='') as output_file:

        writer = csv.writer(output_file)

        # Write the header row with the new column
        header = rows[0]
        header.append('diagnosis')
        writer.writerow(header)

        # Iterate over the remaining rows
        for row in rows[1:]:
            # Determine the diagnosis based on the existing columns
            if row[1] == '1.0':
                diagnosis = 'MEL'
            elif row[2] == '1.0':
                diagnosis = 'NV'
            elif row[3] == '1.0':
                diagnosis = 'BCC'
            elif row[4] == '1.0':
                diagnosis = 'AKIEC'
            elif row[5] == '1.0':
                diagnosis = 'BKL'
            elif row[6] == '1.0':
                diagnosis = 'DF'
            elif row[7] == '1.0':
                diagnosis = 'VASC'
            else:
                diagnosis = 'Unknown'

            # Append the diagnosis to the row
            row.append(diagnosis)

            # Write the modified row to the output file
            writer.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', type=str, required=True)
    parser.add_argument('--output_csv', type=str, required=True)
    args = parser.parse_args()

    preprocess_csv(args.input_csv, args.output_csv)
