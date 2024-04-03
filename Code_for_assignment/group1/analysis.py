import os
import json
import csv
from pathlib import Path
import matplotlib.pyplot as plt

# Define the folder containing JSON files
folder_path = Path('C:/Users/alexo/negmas/anl2024/tournaments/testGroup120240403H002657941815sXeqYpRJ/results')
plot_path = 'C:/Users/alexo/Desktop/plots'

results = {'Group1': {'agreements': 0,
                      'timeouts': 0,
                      'advantage': 0,
                      'advantage_opponent': 0,
                      'nash optimality': 0,
                      'pareto optimality': 0},
            'Group1_not_final_bid': {'agreements': 0,
                      'timeouts': 0,
                      'advantage': 0,
                      'advantage_opponent': 0,
                      'nash optimality': 0,
                      'pareto optimality': 0},
            'Group1_final_bid': {'agreements': 0,
                      'timeouts': 0,
                      'advantage': 0,
                      'advantage_opponent': 0,
                      'nash optimality': 0,
                      'pareto optimality': 0},
           'Boulware': {'agreements': 0,
                      'timeouts': 0,
                      'advantage': 0,
                      'advantage_opponent': 0,
                      'nash optimality': 0,
                      'pareto optimality': 0},
           'Conceder': {'agreements': 0,
                      'timeouts': 0,
                      'advantage': 0,
                      'advantage_opponent': 0,
                      'nash optimality': 0,
                      'pareto optimality': 0},
           'Linear': {'agreements': 0,
                      'timeouts': 0,
                      'advantage': 0,
                      'advantage_opponent': 0,
                      'nash optimality': 0,
                      'pareto optimality': 0},
           'RVFitter': {'agreements': 0,
                      'timeouts': 0,
                      'advantage': 0,
                      'advantage_opponent': 0,
                      'nash optimality': 0,
                      'pareto optimality': 0},
           'MiCRO': {'agreements': 0,
                      'timeouts': 0,
                      'advantage': 0,
                      'advantage_opponent': 0,
                      'nash optimality': 0,
                      'pareto optimality': 0},
           'NashSeeker': {'agreements': 0,
                      'timeouts': 0,
                      'advantage': 0,
                      'advantage_opponent': 0,
                      'nash optimality': 0,
                      'pareto optimality': 0}}

group_count = 0
group_final_count = 0
group_not_final_count = 0

total_timeout_count = 0
first_timeout_count = 0
second_timeout_count = 0
total_agreement_count = 0
first_agreement_count = 0
second_agreement_count = 0

for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        file_path = os.path.join(folder_path, filename)

        with open(file_path, 'r') as file:
            data = json.load(file)

        match data['negotiator_names'][0][0]:
            case 'G':
                group_count += 1
                if data['negotiator_names'][0][-1] == '0':
                    group_not_final_count += 1
                    if data['agreement'] is not None:
                        results['Group1_not_final_bid']['agreements'] += 1
                    else:
                        results['Group1_not_final_bid']['timeouts'] += 1
                    results['Group1_not_final_bid']['advantage'] += ((data['utilities'][0] - data['reserved_values'][0]) /
                                                       (data['max_utils'][0] - data['reserved_values'][0]))
                    results['Group1_not_final_bid']['advantage_opponent'] += ((data['utilities'][1] - data['reserved_values'][1]) /
                                                       (data['max_utils'][1] - data['reserved_values'][1]))
                    results['Group1_not_final_bid']['nash optimality'] += data['nash_optimality']
                    results['Group1_not_final_bid']['pareto optimality'] += data['pareto_optimality']
                if data['agreement'] is not None:
                    results['Group1']['agreements'] += 1
                else:
                    results['Group1']['timeouts'] += 1
                results['Group1']['advantage'] += ((data['utilities'][0] - data['reserved_values'][0]) /
                                                   (data['max_utils'][0] - data['reserved_values'][0]))
                results['Group1']['advantage_opponent'] += (
                        (data['utilities'][1] - data['reserved_values'][1]) /
                        (data['max_utils'][1] - data['reserved_values'][1]))
                results['Group1']['nash optimality'] += data['nash_optimality']
                results['Group1']['pareto optimality'] += data['pareto_optimality']
            case 'B':
                if data['agreement'] is not None:
                    total_agreement_count += 1
                    first_agreement_count += 1
                    results['Boulware']['agreements'] += 1
                else:
                    total_timeout_count += 1
                    first_timeout_count += 1
                    results['Boulware']['timeouts'] += 1
                results['Boulware']['advantage'] += ((data['utilities'][0] - data['reserved_values'][0]) /
                                                     (data['max_utils'][0] - data['reserved_values'][0]))
                results['Boulware']['advantage_opponent'] += ((data['utilities'][1] - data['reserved_values'][1]) /
                                                              (data['max_utils'][1] - data['reserved_values'][1]))
                results['Boulware']['nash optimality'] += data['nash_optimality']
                results['Boulware']['pareto optimality'] += data['pareto_optimality']
            case 'C':
                if data['agreement'] is not None:
                    results['Conceder']['agreements'] += 1
                else:
                    results['Conceder']['timeouts'] += 1
                results['Conceder']['advantage'] += ((data['utilities'][0] - data['reserved_values'][0]) /
                                                     (data['max_utils'][0] - data['reserved_values'][0]))
                results['Conceder']['advantage_opponent'] += ((data['utilities'][1] - data['reserved_values'][1]) /
                                                              (data['max_utils'][1] - data['reserved_values'][1]))
                results['Conceder']['nash optimality'] += data['nash_optimality']
                results['Conceder']['pareto optimality'] += data['pareto_optimality']
            case 'L':
                if data['agreement'] is not None:
                    results['Linear']['agreements'] += 1
                else:
                    results['Linear']['timeouts'] += 1
                results['Linear']['advantage'] += ((data['utilities'][0] - data['reserved_values'][0]) /
                                                     (data['max_utils'][0] - data['reserved_values'][0]))
                results['Linear']['advantage_opponent'] += ((data['utilities'][1] - data['reserved_values'][1]) /
                                                              (data['max_utils'][1] - data['reserved_values'][1]))
                results['Linear']['nash optimality'] += data['nash_optimality']
                results['Linear']['pareto optimality'] += data['pareto_optimality']
            case 'R':
                if data['agreement'] is not None:
                    results['RVFitter']['agreements'] += 1
                else:
                    results['RVFitter']['timeouts'] += 1
                results['RVFitter']['advantage'] += ((data['utilities'][0] - data['reserved_values'][0]) /
                                                   (data['max_utils'][0] - data['reserved_values'][0]))
                results['RVFitter']['advantage_opponent'] += ((data['utilities'][1] - data['reserved_values'][1]) /
                                                            (data['max_utils'][1] - data['reserved_values'][1]))
                results['RVFitter']['nash optimality'] += data['nash_optimality']
                results['RVFitter']['pareto optimality'] += data['pareto_optimality']
            case 'M':
                if data['agreement'] is not None:
                    results['MiCRO']['agreements'] += 1
                else:
                    results['MiCRO']['timeouts'] += 1
                results['MiCRO']['advantage'] += ((data['utilities'][0] - data['reserved_values'][0]) /
                                                   (data['max_utils'][0] - data['reserved_values'][0]))
                results['MiCRO']['advantage_opponent'] += ((data['utilities'][1] - data['reserved_values'][1]) /
                                                            (data['max_utils'][1] - data['reserved_values'][1]))
                results['MiCRO']['nash optimality'] += data['nash_optimality']
                results['MiCRO']['pareto optimality'] += data['pareto_optimality']
            case 'N':
                if data['agreement'] is not None:
                    results['NashSeeker']['agreements'] += 1
                else:
                    results['NashSeeker']['timeouts'] += 1
                results['NashSeeker']['advantage'] += ((data['utilities'][0] - data['reserved_values'][0]) /
                                                   (data['max_utils'][0] - data['reserved_values'][0]))
                results['NashSeeker']['advantage_opponent'] += ((data['utilities'][1] - data['reserved_values'][1]) /
                                                            (data['max_utils'][1] - data['reserved_values'][1]))
                results['NashSeeker']['nash optimality'] += data['nash_optimality']
                results['NashSeeker']['pareto optimality'] += data['pareto_optimality']

        match data['negotiator_names'][1][0]:
            case 'G':
                group_count += 1
                if data['negotiator_names'][1][-1] == '1':
                    group_final_count += 1
                    if data['agreement'] is not None:
                        results['Group1_final_bid']['agreements'] += 1
                    else:
                        results['Group1_final_bid']['timeouts'] += 1
                    results['Group1_final_bid']['advantage'] += ((data['utilities'][0] - data['reserved_values'][0]) /
                                                                 (data['max_utils'][0] - data['reserved_values'][0]))
                    results['Group1_final_bid']['advantage_opponent'] += (
                            (data['utilities'][1] - data['reserved_values'][1]) /
                            (data['max_utils'][1] - data['reserved_values'][1]))
                    results['Group1_final_bid']['nash optimality'] += data['nash_optimality']
                    results['Group1_final_bid']['pareto optimality'] += data['pareto_optimality']
                if data['agreement'] is not None:
                    results['Group1']['agreements'] += 1
                else:
                    results['Group1']['timeouts'] += 1
                results['Group1']['advantage'] += ((data['utilities'][0] - data['reserved_values'][0]) /
                                                   (data['max_utils'][0] - data['reserved_values'][0]))
                results['Group1']['advantage_opponent'] += (
                        (data['utilities'][1] - data['reserved_values'][1]) /
                        (data['max_utils'][1] - data['reserved_values'][1]))
                results['Group1']['nash optimality'] += data['nash_optimality']
                results['Group1']['pareto optimality'] += data['pareto_optimality']
            case 'B':
                if data['agreement'] is not None:
                    total_agreement_count += 1
                    second_agreement_count += 1
                    results['Boulware']['agreements'] += 1
                else:
                    total_timeout_count += 1
                    second_timeout_count += 1
                    results['Boulware']['timeouts'] += 1
                results['Boulware']['advantage'] += ((data['utilities'][0] - data['reserved_values'][0]) /
                                                     (data['max_utils'][0] - data['reserved_values'][0]))
                results['Boulware']['advantage_opponent'] += ((data['utilities'][1] - data['reserved_values'][1]) /
                                                              (data['max_utils'][1] - data['reserved_values'][1]))
                results['Boulware']['nash optimality'] += data['nash_optimality']
                results['Boulware']['pareto optimality'] += data['pareto_optimality']
            case 'C':
                if data['agreement'] is not None:
                    results['Conceder']['agreements'] += 1
                else:
                    results['Conceder']['timeouts'] += 1
                results['Conceder']['advantage'] += ((data['utilities'][0] - data['reserved_values'][0]) /
                                                     (data['max_utils'][0] - data['reserved_values'][0]))
                results['Conceder']['advantage_opponent'] += ((data['utilities'][1] - data['reserved_values'][1]) /
                                                              (data['max_utils'][1] - data['reserved_values'][1]))
                results['Conceder']['nash optimality'] += data['nash_optimality']
                results['Conceder']['pareto optimality'] += data['pareto_optimality']
            case 'L':
                if data['agreement'] is not None:
                    results['Linear']['agreements'] += 1
                else:
                    results['Linear']['timeouts'] += 1
                results['Linear']['advantage'] += ((data['utilities'][0] - data['reserved_values'][0]) /
                                                   (data['max_utils'][0] - data['reserved_values'][0]))
                results['Linear']['advantage_opponent'] += ((data['utilities'][1] - data['reserved_values'][1]) /
                                                            (data['max_utils'][1] - data['reserved_values'][1]))
                results['Linear']['nash optimality'] += data['nash_optimality']
                results['Linear']['pareto optimality'] += data['pareto_optimality']
            case 'R':
                if data['agreement'] is not None:
                    results['RVFitter']['agreements'] += 1
                else:
                    results['RVFitter']['timeouts'] += 1
                results['RVFitter']['advantage'] += ((data['utilities'][0] - data['reserved_values'][0]) /
                                                     (data['max_utils'][0] - data['reserved_values'][0]))
                results['RVFitter']['advantage_opponent'] += ((data['utilities'][1] - data['reserved_values'][1]) /
                                                              (data['max_utils'][1] - data['reserved_values'][1]))
                results['RVFitter']['nash optimality'] += data['nash_optimality']
                results['RVFitter']['pareto optimality'] += data['pareto_optimality']
            case 'M':
                if data['agreement'] is not None:
                    results['MiCRO']['agreements'] += 1
                else:
                    results['MiCRO']['timeouts'] += 1
                results['MiCRO']['advantage'] += ((data['utilities'][0] - data['reserved_values'][0]) /
                                                  (data['max_utils'][0] - data['reserved_values'][0]))
                results['MiCRO']['advantage_opponent'] += ((data['utilities'][1] - data['reserved_values'][1]) /
                                                           (data['max_utils'][1] - data['reserved_values'][1]))
                results['MiCRO']['nash optimality'] += data['nash_optimality']
                results['MiCRO']['pareto optimality'] += data['pareto_optimality']
            case 'N':
                if data['agreement'] is not None:
                    results['NashSeeker']['agreements'] += 1
                else:
                    results['NashSeeker']['timeouts'] += 1
                results['NashSeeker']['advantage'] += ((data['utilities'][0] - data['reserved_values'][0]) /
                                                       (data['max_utils'][0] - data['reserved_values'][0]))
                results['NashSeeker']['advantage_opponent'] += ((data['utilities'][1] - data['reserved_values'][1]) /
                                                                (data['max_utils'][1] - data['reserved_values'][1]))
                results['NashSeeker']['nash optimality'] += data['nash_optimality']
                results['NashSeeker']['pareto optimality'] += data['pareto_optimality']

for group_key in results:
    for key in results[group_key]:
        if group_key == 'Group1_final_bid' or group_key == 'Group1_not_final_bid':
            results[group_key][key] /= group_final_count
        else:
            results[group_key][key] /= group_count

def plot_nash_optimality(dictionary):
    plt.figure()
    x_values = ['Group1', 'Group1 (not final bid)', 'Group1 (final bid)' , 'Boulware', 'Conceder',
                'Linear', 'RVFitter', 'MiCRO', 'NashSeeker']
    y_values = []
    for group in dictionary:
        for metric, value in dictionary[group].items():
            if metric == 'nash optimality':
                y_values.append(value)

    colors = ['tomato' if x == 'Group1' else 'skyblue' for x in x_values]

    plt.bar(x_values, y_values, color=colors)
    plt.xlabel('Agent Name')
    plt.ylabel('Nash Optimality')
    plt.ylim(0, 1.0)
    plt.xticks(rotation=15)
    plt.title('Average Nash Optimality for different Negotiating Agents')
    plt.savefig(plot_path + '/nash_optimality.png')

def plot_advantage(dictionary):
    plt.figure()
    x_values = ['Group1', 'Group1 (not final bid)', 'Group1 (final bid)' , 'Boulware', 'Conceder',
                'Linear', 'RVFitter', 'MiCRO', 'NashSeeker']
    y_values = []
    for group in dictionary:
        for metric, value in dictionary[group].items():
            if metric == 'advantage':
                y_values.append(value)

    colors = ['tomato' if x == 'Group1' else 'skyblue' for x in x_values]

    plt.bar(x_values, y_values, color=colors)
    plt.xlabel('Agent Name')
    plt.ylabel('Normalized Advantage')
    plt.ylim(0, 1.0)
    plt.xticks(rotation=15)
    plt.title('Average Advantage for different Negotiating Agents')
    plt.savefig(plot_path + '/advantage.png')

def plot_agreement(dictionary):
    plt.figure()
    x_values = ['Group1', 'Group1 (not final bid)', 'Group1 (final bid)' , 'Boulware', 'Conceder',
                'Linear', 'RVFitter', 'MiCRO', 'NashSeeker']
    y_values = []
    for group in dictionary:
        for metric, value in dictionary[group].items():
            if metric == 'agreements':
                y_values.append(value)

    colors = ['tomato' if x == 'Group1' else 'skyblue' for x in x_values]

    plt.bar(x_values, y_values, color=colors)
    plt.xlabel('Agent Name')
    plt.ylabel('Agreements (%)')
    plt.ylim(0, 1.0)
    plt.xticks(rotation=15)
    plt.title('Average Agreements for different Negotiating Agents')
    plt.savefig(plot_path + '/agreements.png')
    plt.plot()

def plot_timeouts(dictionary):
    plt.figure()
    x_values = ['Group1', 'Group1 (not final bid)', 'Group1 (final bid)' , 'Boulware', 'Conceder',
                'Linear', 'RVFitter', 'MiCRO', 'NashSeeker']
    y_values = []
    for group in dictionary:
        for metric, value in dictionary[group].items():
            if metric == 'timeouts':
                y_values.append(value)

    colors = ['tomato' if x == 'Group1' else 'skyblue' for x in x_values]

    plt.bar(x_values, y_values, color=colors)
    plt.xlabel('Agent Name')
    plt.ylabel('Timeouts (%)')
    plt.ylim(0, 1.0)
    plt.xticks(rotation=15)
    plt.title('Average Timeouts for different Negotiating Agents')
    plt.savefig(plot_path + '/timeouts.png')
    plt.plot()

def plot_advantage_opp(dictionary):
    plt.figure()
    x_values = ['Group1', 'Group1 (not final bid)', 'Group1 (final bid)' , 'Boulware', 'Conceder',
                'Linear', 'RVFitter', 'MiCRO', 'NashSeeker']
    y_values = []
    for group in dictionary:
        for metric, value in dictionary[group].items():
            if metric == 'advantage_opponent':
                y_values.append(value)

    colors = ['tomato' if x == 'Group1' else 'skyblue' for x in x_values]

    plt.bar(x_values, y_values, color=colors)
    plt.xlabel('Agent Name')
    plt.ylabel("Opponent's Advantage")
    plt.ylim(0, 1.0)
    plt.xticks(rotation=15)
    plt.title("Average Opponent's Advantage for different Negotiating Agents")
    plt.savefig(plot_path + '/advantage_opponent.png')

def plot_pareto(dictionary):
    plt.figure()
    x_values = ['Group1', 'Group1 (not final bid)', 'Group1 (final bid)' , 'Boulware', 'Conceder',
                'Linear', 'RVFitter', 'MiCRO', 'NashSeeker']
    y_values = []
    for group in dictionary:
        for metric, value in dictionary[group].items():
            if metric == 'pareto optimality':
                y_values.append(value)

    colors = ['tomato' if x == 'Group1' else 'skyblue' for x in x_values]

    plt.bar(x_values, y_values, color=colors)
    plt.xlabel('Agent Name')
    plt.ylabel('Pareto Optimality')
    plt.ylim(0, 1.0)
    plt.xticks(rotation=15)
    plt.title('Average Pareto Optimality for different Negotiating Agents')
    plt.savefig(plot_path + '/pareto_optimality.png')


def plot_all(dictionary):
    plot_agreement(dictionary)
    plot_timeouts(dictionary)
    plot_advantage(dictionary)
    plot_advantage_opp(dictionary)
    plot_pareto(dictionary)
    plot_nash_optimality(dictionary)

def print_results(dictionary):
    for group in dictionary:
        print(group, ':')
        for metric in dictionary[group]:
            print(metric, ': ', dictionary[group][metric])
        print()

print_results(results)
plot_all(results)

# Write extracted values to a CSV file
# csv_file_path = 'extracted_values.csv'
# with open(csv_file_path, 'w', newline='') as csvfile:
#     writer = csv.DictWriter(csvfile, fieldnames=keys_to_extract)
#
#     # Write header row
#     writer.writeheader()
#
#     # Write data rows
#     for i in range(len(extracted_values[keys_to_extract[0]])):
#         row = {key: extracted_values[key][i] for key in keys_to_extract}
#         writer.writerow(row)
#
# print(f"Extracted values have been written to {csv_file_path}")
