import numpy as np
import pandas
from matplotlib import pyplot
from argparse import ArgumentParser
from glob import glob
import os

parser = ArgumentParser()
parser.add_argument("input", help="Directory that contains .npz files of different classes")
parser.add_argument("output")

FIGSIZE = [6.4*1.2, 4.8*1.2]

def main_heatmaps(args):
    files = glob(os.path.join(args.input, "*"))

    fig = pyplot.figure(figsize=FIGSIZE, dpi=200)

    # Only handling maximized scores here,
    # no sense to plot all noise scores for
    # each class separately
    idx_and_score_and_name = []
    for filename in files:
        data = np.load(filename)
        class_name = os.path.basename(filename).split("_")[-1].split(".")[0]
        class_idx = data["class_index"].item()

        max_scores_2 = data["max_score_2"].squeeze()
        max_scores_2 = max_scores_2.mean(axis=0)

        idx_and_score_and_name.append((class_idx, max_scores_2, class_name))
    
    # Sort by class index
    idx_and_score_and_name = sorted(idx_and_score_and_name, key=lambda x: x[0], reverse=True)

    # plot N1 x N2 heatmap, where
    # each pixel represents average score
    # of N2 when noise was maximized to N1
    heatmap = np.array([x[1] for x in idx_and_score_and_name])

    pyplot.imshow(1 - np.flipud(heatmap), cmap="gray", vmin=0, vmax=1)

    ticks = list(range(len(heatmap)))
    ticknames = [x[2] for x in idx_and_score_and_name]

    pyplot.yticks(ticks, ticknames)
    pyplot.xticks(ticks, ticknames, rotation="vertical")

    pyplot.tight_layout()
    pyplot.savefig(args.output, bbox_inches="tight")
    pyplot.show()

def classification_results(args):
    files = glob(os.path.join(args.input, "*"))

    classifier_1_corrects = []
    classifier_2_corrects = []
    for filename in files:
        data = np.load(filename)

        class_idx = data["class_index"].item()
        scores_1 = data["max_score_1"].squeeze()
        scores_2 = data["max_score_2"].squeeze()

        classifier_1_corrects.append(class_idx == np.argmax(scores_1, axis=1))
        classifier_2_corrects.append(class_idx == np.argmax(scores_2, axis=1))
    
    classifier_1_corrects = np.array(classifier_1_corrects).ravel()
    classifier_2_corrects = np.array(classifier_2_corrects).ravel()

    print("Orig. class accuracy: %.4f" % classifier_1_corrects.mean())
    print("Seco. class accuracy: %.4f" % classifier_2_corrects.mean())

if __name__ == "__main__":
    args = parser.parse_args()
    main_heatmaps(args)
    classification_results(args)
