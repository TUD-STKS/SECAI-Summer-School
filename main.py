import argparse
import os
from pathlib import Path
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from sklearn.linear_model import RidgeClassifier
from sklearn.utils.fixes import loguniform
import joblib
import medmnist
import numpy as np
from medmnist import INFO, Evaluator
from medmnist.info import DEFAULT_ROOT


def main(data_flag, input_root, output_root, run, model_path):

    info = INFO[data_flag]
    task = info['task']
    _ = getattr(medmnist, INFO[data_flag]['python_class'])(
        split="train", root=input_root, download=True)

    output_root = Path(output_root) / data_flag
    output_root.mkdir(parents=True, exist_ok=True)

    npz_file = np.load(str(Path(input_root) / f"{data_flag}.npz"))

    x_train = npz_file['train_images']
    y_train = npz_file['train_labels']
    x_val = npz_file['val_images']
    y_val = npz_file['val_labels']
    x_test = npz_file['test_images']
    y_test = npz_file['test_labels']

    size = x_train[0].size
    X_train = x_train.reshape(x_train.shape[0], size, )
    X_val = x_val.reshape(x_val.shape[0], size, )
    X_test = x_test.reshape(x_test.shape[0], size, )

    if task != 'multi-label, binary-class':
        y_train = y_train.ravel()
        y_val = y_val.ravel()
        y_test = y_test.ravel()

    if model_path is not None:
        model = joblib.load(model_path)
        test(model, data_flag, X_train, 'train', output_root, run)
        test(model, data_flag, X_val, 'val', output_root, run)
        test(model, data_flag, X_test, 'test', output_root, run)

    model = train(data_flag, X_train, y_train, X_val, y_val, run)

    test(model, data_flag, X_train, 'train', output_root, run)
    test(model, data_flag, X_val, 'val', output_root, run)
    test(model, data_flag, X_test, 'test', output_root, run)


def train(data_flag, X_train, y_train, X_val, y_val, run):
    test_fold = np.zeros(len(X_train)+len(X_val))
    test_fold[:len(X_train)] = -1
    clf = RandomizedSearchCV(
        estimator=RidgeClassifier(),
        param_distributions={'alpha': loguniform(1e-5, 10 - 1e-5)}, n_iter=100,
        n_jobs=4, random_state=42, cv=PredefinedSplit(test_fold=test_fold),
        return_train_score=True, verbose=2)

    clf.fit(np.vstack((X_train, X_val)), np.hstack((y_train, y_val)))

    joblib.dump(clf, Path(output_root) / f'{data_flag}_ridge_clf_{run}.m')

    return clf


def test(model, data_flag, x, split, output_root, run):

    evaluator = medmnist.Evaluator(data_flag, split)
    y_pred = model.predict(x)
    auc, acc = evaluator.evaluate(y_pred, output_root, run)
    print('%s  auc: %.5f  acc: %.5f ' % (split, auc, acc))

    return auc, acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_flag', default='pathmnist', type=str)
    parser.add_argument('--input_root', default=DEFAULT_ROOT, type=str)
    parser.add_argument('--output_root', default='./results/sklearn', type=str)
    parser.add_argument('--run', default='model1', help='to name a standard '
                                                        'evaluation csv file, '
                                                        'named as {flag}_'
                                                        '{split}_[AUC]'
                                                        '{auc:.3f}_[ACC]'
                                                        '{acc:.3f}@{run}.csv',
                        type=str)
    parser.add_argument('--model_path', default=Path(
        r"./results/sklearn/pathmnist_ridge_clf_model1.m"),
                        help='root of the pretrained model to test', type=str)

    args = parser.parse_args()
    data_flag = args.data_flag
    input_root = args.input_root
    output_root = args.output_root
    run = args.run
    model_path = args.model_path

    main(data_flag, input_root, output_root, run, model_path)
