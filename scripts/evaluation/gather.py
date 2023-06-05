from NPHM import env_paths
from NPHM.data.manager import DataManager
import os
import json
import tyro


def main(result_dir : str):
    manager = DataManager()
    all_metrics = {}
    all_metrics_face = {}
    total_number_of_scans = 0
    for subject in env_paths.subjects_test:
        try:

            expressions = manager.get_expressions(subject, testing=True)
            all_metrics[subject] =  {}
            all_metrics_face[subject] =  {}
            expressions.sort()
            for expression in expressions:
                with open(result_dir + '/evaluation/{}/expression_{}/metrics.json'.format(subject, expression), 'r') as fp:
                    metrics = json.load(fp)
                    #print(json.dumps(metrics, indent=4))
                    for k in metrics.keys():
                        if k not in all_metrics[subject]:
                            all_metrics[subject][k] = [metrics[k]]
                        else:
                            all_metrics[subject][k].append(metrics[k])
                    total_number_of_scans += 1

                with open(result_dir + '/evaluation/{}/expression_{}/metrics_face.json'.format(subject, expression), 'r') as fp:
                    metrics_face = json.load(fp)
                    #print(json.dumps(metrics_face, indent=4))
                    for k in metrics_face.keys():
                        if k not in all_metrics_face[subject]:
                            all_metrics_face[subject][k] = [metrics_face[k]]
                        else:
                            all_metrics_face[subject][k].append(metrics_face[k])
        except Exception as e:
            pass
    avg_metrics = {}
    avg_metrics_face = {}
    num_expr_per_subject = {}
    total_avg_metrics = {}
    total_avg_metrics_face = {}
    for subject in env_paths.subjects_test:
        num_expr_per_subject[subject] = 0
        if subject not in avg_metrics:
            avg_metrics[subject] = {}
            avg_metrics_face[subject] = {}
        if subject in all_metrics:
            for k in all_metrics[subject].keys():
                avg_metrics[subject][k] = sum(all_metrics[subject][k]) / len(all_metrics[subject][k])
                avg_metrics_face[subject][k] = sum(all_metrics_face[subject][k]) / len(all_metrics_face[subject][k])
                num_expr_per_subject[subject] += 1
                if k not in total_avg_metrics:
                    total_avg_metrics[k] = 0
                    total_avg_metrics_face[k] = 0
                total_avg_metrics[k] += sum(all_metrics[subject][k])
                total_avg_metrics_face[k] += sum(all_metrics_face[subject][k])

    print(num_expr_per_subject)
    print(json.dumps(avg_metrics, indent=4))
    print(json.dumps(avg_metrics_face, indent=4))

    for k in total_avg_metrics:
        total_avg_metrics[k] /= total_number_of_scans
        total_avg_metrics_face[k] /= total_number_of_scans

    print(json.dumps(total_avg_metrics, indent=4))
    print(json.dumps(total_avg_metrics_face, indent=4))

    print(total_number_of_scans)

    #with open('test.json', 'w') as f:
    #    json.dump(total_avg_metrics, f, ensure_ascii=False)
    #with open('test_face.json', 'w') as f:
    #    json.dump(total_avg_metrics_face, f, ensure_ascii=False)
    import csv


    with open(result_dir + '/evaluation/total_merics.csv', 'w') as f:  # You will need 'wb' mode in Python 2.x
        w = csv.DictWriter(f, total_avg_metrics.keys())
        w.writeheader()
        w.writerow(total_avg_metrics)

    with open(result_dir + '/evaluation/total_metrics_face.csv', 'w') as f:  # You will need 'wb' mode in Python 2.x
        w = csv.DictWriter(f, total_avg_metrics_face.keys())
        w.writeheader()
        w.writerow(total_avg_metrics_face)
    return total_avg_metrics, total_avg_metrics_face


if __name__ == '__main__':
    tyro.cli(main)