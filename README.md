Name|Description
:---|:---
num_frames|Total number of frames.
num_matches|Total number matches.
num_switches|Total number of track switches.
num_false_positives|Total number of false positives (false-alarms).
num_misses|Total number of misses.
num_detections|Total number of detected objects including matches and switches.
num_objects|Total number of unique object appearances over all frames.
num_predictions|Total number of unique prediction appearances over all frames.
num_unique_objects|Total number of unique object ids encountered.
mostly_tracked|Number of objects tracked for at least 80 percent of lifespan.
partially_tracked|Number of objects tracked between 20 and 80 percent of lifespan.
mostly_lost|Number of objects tracked less than 20 percent of lifespan.
num_fragmentations|Total number of switches from tracked to not tracked.
motp|Multiple object tracker precision.
mota|Multiple object tracker accuracy.
precision|Number of detected objects over sum of detected and false positives.
recall|Number of detections over number of objects.
idfp|ID measures: Number of false positive matches after global min-cost matching.
idfn|ID measures: Number of false negatives matches after global min-cost matching.
idtp|ID measures: Number of true positives matches after global min-cost matching.
idp|ID measures: global min-cost precision.
idr|ID measures: global min-cost recall.
idf1|ID measures: global min-cost F1 score.
obj_frequencies|`pd.Series` Total number of occurrences of individual objects over all frames.
pred_frequencies|`pd.Series` Total number of occurrences of individual predictions over all frames.
track_ratios|`pd.Series` Ratio of assigned to total appearance count per unique object id.
id_global_assignment| `dict` ID measures: Global min-cost assignment for ID measures.

