import sys
from os import listdir
import json
import numpy as np
import h5py
from generate_summary import generate_summary
from evaluation_metrics import evaluate_summary

# with args (example usage: python compute_fscores.py <path_to_results> TVSum avg)
path = sys.argv[1];
dataset = sys.argv[2];
eval_method = sys.argv[3];

#without args
'''path = <path_to_results>
dataset = 'TVSum'
eval_method = 'avg'''

results = listdir(path)
results.sort(key=lambda video: int(video[6:-5]))
HOME_PATH = 'C:\JG\CODE\MainProject\AC-SUM-GAN\data/'
DATASET_PATH= HOME_PATH + dataset + '/eccv16_dataset_' + dataset.lower() + '_google_pool5.h5'

# for each epoch, read the results' file and compute the f_score
f_score_epochs = []
precision_epochs = []
recall_epochs = []
accuracy_epochs = []
for epoch in results:
    print(epoch)
    all_scores = []
    with open(path+'/'+epoch) as f:
        data = json.loads(f.read())
        keys = list(data.keys())

        for video_name in keys:
            scores = np.asarray(data[video_name])
            all_scores.append(scores)

    all_user_summary, all_shot_bound, all_nframes, all_positions = [], [], [], []
    with h5py.File(DATASET_PATH, 'r') as hdf:        
        for video_name in keys:
            video_index = video_name[6:]
            
            user_summary = np.array( hdf.get('video_'+video_index+'/user_summary') )
            sb = np.array( hdf.get('video_'+video_index+'/change_points') )
            n_frames = np.array( hdf.get('video_'+video_index+'/n_frames') )
            positions = np.array( hdf.get('video_'+video_index+'/picks') )

            all_user_summary.append(user_summary)
            all_shot_bound.append(sb)
            all_nframes.append(n_frames)
            all_positions.append(positions)

    all_summaries = generate_summary(all_shot_bound, all_scores, all_nframes, all_positions)

    all_f_scores = []
    all_precision = []
    all_recall = []
    all_accuracy = []
	# compare the resulting summary with the ground truth one, for each video
    for video_index in range(len(all_summaries)):
        summary = all_summaries[video_index]
        user_summary = all_user_summary[video_index]
        f_score,precision,recall,accuracy = evaluate_summary(summary, user_summary, eval_method)	
        all_f_scores.append(f_score)
        all_precision.append(precision)
        all_recall.append(recall)
        all_accuracy.append(accuracy)

    f_score_epochs.append(np.mean(all_f_scores))
    precision_epochs.append(np.mean(all_precision))
    recall_epochs.append(np.mean(all_recall))
    accuracy_epochs.append(np.mean(all_accuracy))
    print("f_score: ",np.mean(all_f_scores))
    print("precision: ",np.mean(all_precision))
    print("recall: ",np.mean(all_recall))
    print("accuracy: ",np.mean(all_accuracy))

with open(path+'/f_scores.txt', 'w') as outfile:  
    json.dump(f_score_epochs, outfile)

with open(path+'/precision.txt', 'w') as outfile:  
    json.dump(precision_epochs, outfile)

with open(path+'/recall.txt', 'w') as outfile:  
    json.dump(recall_epochs, outfile)

with open(path+'/accuracy.txt', 'w') as outfile:  
    json.dump(accuracy_epochs, outfile)