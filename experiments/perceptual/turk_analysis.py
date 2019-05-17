import numpy as np
import pandas
from matplotlib import pyplot
from argparse import ArgumentParser
import seaborn as sns

parser = ArgumentParser()
parser.add_argument("input")
parser.add_argument("operation", choices=["classes", "overall"])
parser.add_argument("output")

BAR_WIDTH = 0.5

def to_int(x):
    if "Bad -" in x:
        return 0
    elif "Poor -" in x:
        return 1
    elif "Fair -" in x:
        return 2
    elif "Good -" in x:
        return 3
    elif "Excellent -" in x:
        return 4
    else:
        raise RuntimeError()

def get_bad_worker_idxs(answer_labels, workers, experiment_id):
    # Return indexes of bad workers
    # (those who answered 3 or 4 in complete noise scenarios)
    bad_idxs = (((answer_labels == 3) | (answer_labels == 4)) & (experiment_id == 1))
    bad_workers = set(workers[bad_idxs])
    bad_idxs = np.array([worker in bad_workers for worker in workers])
    return bad_idxs

def original_or_not(audio_urls):
    return np.array(["original" in x for x in audio_urls])

def per_class_perceptual_qualities(args):
    input_file = args.input
    data = pandas.read_csv(input_file)

    answer_labels = np.array(list(map(to_int, data["Answer.audio-naturalness.label"])))
    experiment_id = np.array(data["Input.experiment"])
    workers = data["WorkerId"]
    audio_classes = data["Input.class_label"]
    original = original_or_not(data["Input.audio_url"])

    bad_idxs = get_bad_worker_idxs(answer_labels, workers, experiment_id)

    experiment_id = experiment_id[~bad_idxs]
    answer_labels = answer_labels[~bad_idxs]
    original = original[~bad_idxs]
    audio_classes = audio_classes[~bad_idxs]

    unique_classes = sorted(list(set(audio_classes)), reverse=False)

    fig = pyplot.figure(figsize=[6.4*1.5, 4.8], dpi=200)
    bars_ticklabels = []
    for i,audio_class in enumerate(unique_classes):
        class_idxs_orig = ((audio_classes == audio_class) & (original))
        class_idxs_max_class = ((audio_classes == audio_class) & (~original) & (experiment_id == 1))
        class_idxs_max_decod = ((audio_classes == audio_class) & (~original) & (experiment_id == 3))

        original_quality = answer_labels[class_idxs_orig].mean()
        classifier_quality = answer_labels[class_idxs_max_class].mean()
        decoder_quality = answer_labels[class_idxs_max_decod].mean()

        original_quality_std = answer_labels[class_idxs_orig].std()/2
        classifier_quality_std = answer_labels[class_idxs_max_class].std()/2
        decoder_quality_std = answer_labels[class_idxs_max_decod].std()/2

        #pyplot.bar([(i*2)-BAR_WIDTH,i*2,(i*2)+BAR_WIDTH], 
        #           [original_quality, classifier_quality, decoder_quality],
        #           #yerr=[original_quality_std, classifier_quality_std, decoder_quality_std],
        #           width=BAR_WIDTH, 
        #           edgecolor="black",   
        #           color=["g", "r", "b"])
        
        bars_ticklabels.append((original_quality, classifier_quality, decoder_quality, audio_class))
        #ticklabels.append(audio_class)

    bars_ticklabels = sorted(bars_ticklabels, key=lambda x: x[0], reverse=True)

    ticks = []
    ticklabels = []
    for i, bars_ticklabel in enumerate(bars_ticklabels):
        
        #pyplot.bar([(i*2)-BAR_WIDTH,i*2,(i*2)+BAR_WIDTH],
        #           [bars_ticklabel[0], bars_ticklabel[1], bars_ticklabel[2]],
        #           width=BAR_WIDTH, 
        #           edgecolor="black",
        #           linewidth=1,
        #           color=["g", "r", "b"])
        
        pyplot.bar([(i*2)-BAR_WIDTH],
                   [bars_ticklabel[0]],
                   width=BAR_WIDTH, 
                   #edgecolor="black",
                   linewidth=1,
                   color=["g"],
                   label="Original" if i == 0 else None)
        pyplot.bar([(i*2)],
                   [bars_ticklabel[1]],
                   width=BAR_WIDTH, 
                   #edgecolor="black",
                   linewidth=1,
                   color=["r"],
                   label="Classifier" if i == 0 else None)
        pyplot.bar([(i*2)+BAR_WIDTH],
                   [bars_ticklabel[2]],
                   width=BAR_WIDTH, 
                   #edgecolor="black",
                   linewidth=1,
                   color=["b"],
                   label="Decoder" if i == 0 else None)

        ticks.append(i*2)
        ticklabels.append(bars_ticklabel[3])

    pyplot.ylabel("Mean quality", fontsize=18)
    pyplot.xticks(ticks, ticklabels, rotation="vertical")
    pyplot.gca().tick_params(axis='x', which='major', labelsize=18)
    pyplot.gca().tick_params(axis='y', which='major', labelsize=16)
    pyplot.tight_layout()
    pyplot.legend(fontsize=15)
    pyplot.savefig(args.output, bbox_inches="tight")
    pyplot.show()

def main_exp_histograms(args):
    # Plot histograms of experiment 1 vs 3 and 2 vs 4
    input_file = args.input
    data = pandas.read_csv(input_file)

    answer_labels = np.array(list(map(to_int, data["Answer.audio-naturalness.label"])))
    experiment_id = np.array(data["Input.experiment"])
    workers = data["WorkerId"]
    original = original_or_not(data["Input.audio_url"])

    # Remove bad workers and original samples
    bad_idxs = get_bad_worker_idxs(answer_labels, workers, experiment_id)
    bad_idxs = bad_idxs | original

    experiment_id = experiment_id[~bad_idxs]
    answer_labels = answer_labels[~bad_idxs]

    # Plot 1 vs 3
    split_by_exp = dict([(exp_id, answer_labels[experiment_id == exp_id]) for exp_id in [1,2,3,4]])

    unique_1, counts_1 = np.unique(split_by_exp[1], return_counts=True)
    unique_3, counts_3 = np.unique(split_by_exp[3], return_counts=True)
    unique_2, counts_2 = np.unique(split_by_exp[2], return_counts=True)
    unique_4, counts_4 = np.unique(split_by_exp[4], return_counts=True)

    pyplot.bar(unique_1-BAR_WIDTH/2, counts_1, width=BAR_WIDTH)
    pyplot.bar(unique_3+BAR_WIDTH/2, counts_3, width=BAR_WIDTH)
    pyplot.savefig(args.output.replace(".", "_noise2class."))
    pyplot.show()
    
    pyplot.bar(unique_2-BAR_WIDTH/2, counts_2, width=BAR_WIDTH)
    pyplot.bar(unique_4+BAR_WIDTH/2, counts_4, width=BAR_WIDTH)
    pyplot.savefig(args.output.replace(".", "_class2class."))
    pyplot.show()


if __name__ == "__main__":
    args = parser.parse_args()
    if args.operation == "classes":
        per_class_perceptual_qualities(args)
    elif args.operation == "overall":
        main_exp_histograms(args)