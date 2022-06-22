# Libraries
import csv
import os
import pickle

import matplotlib.pyplot as plt
import mne
import numpy as np
import seaborn as sns
import pyxdf

fname = "./raw/sub-P0102_ses-S001_task-Odd1_run-001_eeg.xdf"
out_dir = "./doc/img/pilot/"
streams, header = pyxdf.load_xdf(fname)
s_names = [s["info"]["name"] for s in streams]
n_streams = len(streams)
ms = ds = None # Marker and Data Stream numbers
for sn in range(n_streams):
    if streams[sn]["info"]["name"][0] == 'PsychoPy':
        ms = sn
    if streams[sn]["info"]["type"][0] == 'EEG':
        ds = sn
    else:
        continue
assert None not in [ms, ds], "Streams missing"
# raw data
data = streams[ds]["time_series"].T
data.shape
# metadata
t = np.array(streams[ds]["time_stamps"])
sfreq = float(streams[ds]["info"]["nominal_srate"][0])
orig_time = float(streams[ds]["info"]["created_at"][0])

ch_names, ch_types, ch_units = [], [], []
for i in streams[ds]["info"]["desc"][0]["channels"][0]["channel"]:
    ch_names.append(i["label"][0])
    ch_types.append(i["type"][0]) # TODO: transform to mne types
    ch_units.append(i["unit"][0])
    # Proper units of measure: * V: eeg, eog, seeg, dbs, emg, ecg, bio, ecog * T: mag * T/m: grad * M: hbo, hbr * Am: dipole * AU: misc
# ch_types = [s.lower() for s in ch_types]
assert data.shape[0] == len(ch_names)

names = [
    "Fz", "Cz", "Pz", "O1", "O2", "ECG1", "ECG2", "None",
]
types = [
    "eeg", "eeg", "eeg", "eeg", "eeg", "ecg", "ecg", "misc"
]

info = mne.create_info(names, sfreq, ch_types=types)

# Markers
markers_lb = streams[ms]["time_series"]
markers_ts = np.array(streams[ms]["time_stamps"])
# find the closest sample point to each marker. Error +/- 2* fs
markers_ts = [min(t, key=lambda x:abs(x-i)) for i in markers_ts]
markers_idx = [np.where(t == i)[0][0] for i in markers_ts]
markers_t = t[markers_idx] - min(t)
assert len(markers_idx) == len(markers_lb)

# info on trigger errors
labels = [i[0] for i in markers_lb]
timest = markers_t
assert np.abs(np.diff(timest)).min() > 0.0, f"Annotations share onset {fname}"
tr = np.vstack([labels, timest]).T


# # Plot that marker error
# rel = sns.displot(errors, rug=True, kde=True)
# rel.fig.suptitle(f"File {fname[4:-4]}, max {max(errors):.2f} sd {np.std(errors):.2f}")
# rel.set(xlabel="timing error (ms)")
# plt.show()

# Set annotations
annots = mne.Annotations(markers_t,
                        np.zeros_like(markers_t),
                        labels,
                        orig_time=None)
# Montage
info.set_montage(mne.channels.make_standard_montage("standard_1020"))
raw1 = mne.io.RawArray(data[0:8,:], info)
raw1._filenames = [fname]
# raw.set_meas_date(orig_time=orig_time)
raw1.set_annotations(annots)

raw2 = mne.io.RawArray(data[8:16,:], info)
raw2._filenames = [fname]
# raw.set_meas_date(orig_time=orig_time)
raw2.set_annotations(annots)



subjs = {
    "S1":raw1,
    "S2":raw2
}
for subj, raw in subjs.items():
    # Setup visualization scalings
    viz_scalings = dict(eeg=1e2, eog=1e-4, ecg=1e3, bio=1e-7, misc=1)

    raw.plot(scalings=viz_scalings)
    # plt.show()

    ## Filtering
    raw.filter(l_freq=0.1, h_freq=30)
    # raw.filter(l_freq=0.1, h_freq=None)

    fig = raw.plot_psd(fmax=60, average=True, show=False)
    fig.savefig(out_dir+f"/{subj}_psd.png")
    # plt.show()
    # fig.savefig(out_dir+f"/00_avg_psd.png")
    # raw.plot(scalings=viz_scalings)

    # scalings = dict(eeg=1e-5, eog=1e-4, ecg=1e-4, bio=1)
    scalings = dict(eeg=1e2, ecg=1e3, bio=1e-7, misc=1)


    ## Re-reference
    # Reference Electrode Standardization Technique infinity reference
    # mne.set_eeg_reference(raw, 'average', copy=False)  # in-place
    # raw.plot(scalings=viz_scalings)
    # plt.plot()

    ## Epoching
    events, event_dict = mne.events_from_annotations(raw)
    mne.viz.plot_events(events,
                        sfreq=raw.info['sfreq'],
                        first_samp=raw.first_samp,
                        event_id=event_dict,
                        )

    assert len(events) == len(set(i["onset"] for i in raw.annotations)), f"Annotations share onset {fname}"
    assert np.abs(np.diff([i["onset"] for i in raw.annotations])).min() > 0.0, f"Annotations share onset {fname}"

    # Automatic rejection criteria:
    # Reject epochs based on maximum peak-to-peak signal amplitude (PTP)
    reject_criteria = dict(eeg=120)  # 120 µV
    # Reject epochs based on minimum peak-to-peak signal amplitude (PTP).
    flat_criteria = dict(eeg=1e-1)   # .1 µV

    # Epoching
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_dict,
        baseline=(-0.1, 0),
        tmin=-0.1,
        tmax=0.5,
        reject=reject_criteria,
        flat=flat_criteria,
        preload=True,
    )
    fig = epochs.plot_drop_log(subject=fname, show=False)
    # plt.show()
    fig.savefig(out_dir+f"/{subj}_droplog.png")
    # fig.savefig(out_dir+f"/00_dropped_epochs.png")

    # control = epochs["landscape"].average()
    # target  = epochs["face"].average()
    # conditions = {"NT": control, "T": target}

    # for c in conditions:
    #     conditions[c].plot_joint(times=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])

    #     conditions[c].plot_topomap(times=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5], average=0.05)
        # plt.show()

    ## Visualize ERPs
    control = epochs["landscape"]
    target  = epochs["face"]
    conditions = {"NT": control, "T": target}
    for c in conditions:
        l = conditions[c].plot_image(
                                sigma=1,
                                combine="mean",
                                title=c,
                                # show=False,
                                )
                
        l[0].savefig(out_dir+f"/{subj}_img_{c}.png")

    conditions = {"NT": list(epochs["landscape"].iter_evoked()),
                    "T": list(epochs["face"].iter_evoked())}

    # Regions of Interest
    chs = names
    rois = dict(
        Fz=np.array([chs.index(i) for i in ["Fz"]]),
        Cz=np.array([chs.index(i) for i in ["Cz"]]),
        Pz=np.array([chs.index(i) for i in ["Pz"]]),
        O=np.array([chs.index(i) for i in ["O1", "O2"]]),
        # all=np.array([i for i,ch in enumerate(chs)]),
    )

    for roi in rois:
        l = mne.viz.plot_compare_evokeds(conditions,
                                        title=roi,
                                        picks=rois[roi],
                                        # show=False,
                                        combine='mean')
        l[0].savefig(out_dir+f"/{subj}_comp_{roi}.png")



## Grand Average
with open("epochsGA.pkl", "wb") as f:
    pickle.dump(epochsGA, f, protocol=pickle.HIGHEST_PROTOCOL)

# with open("epochsGA.pkl", 'rb') as f:
#     epochsGA = pickle.load(f)

# grand_average = mne.grand_average([evoked1, evoked2, ...])

# Epoching
epochs1 = mne.Epochs(
    raw1,
    events,
    event_id=event_dict,
    baseline=(-0.1, 0),
    tmin=-0.1,
    tmax=0.5,
    reject=reject_criteria,
    flat=flat_criteria,
    preload=True,
)
epochs2 = mne.Epochs(
    raw2,
    events,
    event_id=event_dict,
    baseline=(-0.1, 0),
    tmin=-0.1,
    tmax=0.5,
    reject=reject_criteria,
    flat=flat_criteria,
    preload=True,
)

epochs = [epochs1, epochs2]
ga = mne.grand_average([e.average() for e in epochs])

## 3 Plot GA over conditions and ROIs
data = {
    "landscape":[],
    "face":[],
}
conditions = [c for c in data]
for condition in conditions:
    s_trials = []
    for s, subj_ep in enumerate(epochs):
        e = subj_ep[condition].get_data() # Extract subject data
        s_trials.append(e)
    eps = np.concatenate(s_trials) # Concatenate subjcet data: eps can be stored if subj is irrelevant for analysis
    data[condition] = eps, eps.mean(axis=0), eps.std(axis=0)

for roi in rois:
    fig, ax = plt.subplots()
    samps = []
    for condition in conditions:
        eps, mu, sigma = data[condition]

        # Visualize means
        y = mu[rois[roi],:].mean(axis=0)
        t = np.linspace(-0.1,0.5,y.shape[0])
        ax.plot(t, y, "-", label=condition.upper())

        # Viz deviation
        # y_std = sigma[rois[roi],:].mean(axis=0) * 0.02 # ATTENTION: 2% of std for visualization
        # y_stdp = y + y_std
        # y_stdn = y - y_std
        # ax.fill_between(t, y_stdn, y_stdp, alpha=0.2)

        # Extract samples' mean for cotdition comparison
        samps.append(eps[:,rois[roi],:].mean(axis=1))

    ax.legend()
    # ax.axvspan(.300, .600, alpha=0.2, color="C3")
    # ax.axvspan(.160, .200, alpha=0.2, color="C3")
    ax.axhline(0, color="black", lw=1,)
    ax.axvline(0, color="black", lw=1, ls="--")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("$\mu$V")
    fig.suptitle(f"GA {roi}")
    plt.show()
    fig.savefig(out_dir+f"/GA_comp_{roi}.png")
    # fig.savefig(out_dir+f"/03_comp_{roi}_N170.png")
    # fig.savefig(out_dir+f"/03_comp_{roi}_N170.svg", format="svg")
    # # fig.savefig(out_dir+f"/03_comp_{roi}.png")
    # # fig.savefig(out_dir+f"/03_comp_{roi}.svg", format="svg")

    # plt.close("all")

data = streams[ds]["time_series"].T

c = np.corrcoef(data)
fig, ax = plt.subplots()
im = ax.imshow(c)
# plt.show()

# Show all ticks and label them with the respective list entries
lbs = [f"S1{n}" for n in names] + [f"S2{n}" for n in names]
ax.set_xticks(np.arange(len(lbs)), labels=lbs)
ax.set_yticks(np.arange(len(lbs)), labels=lbs)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(lbs)):
    for j in range(len(lbs)):
        text = ax.text(j, i, np.round(c[i, j], 1),
                       ha="center", va="center", color="w")
plt.legend()
ax.set_title("Channel Correlation")
fig.tight_layout()
plt.show()
fig.savefig(out_dir+f"/Corr.png")

# Coh
from scipy.signal import coherence as coh
n = data.shape[0]
c = np.empty([n,n])
for i in range(n):
    for j in range(n):
        c[i,j] = np.mean(coh(data[i,:], data[j,:], fs=sfreq)[1])

fig, ax = plt.subplots()
im = ax.imshow(c)

# Show all ticks and label them with the respective list entries
lbs = [f"S1{n}" for n in names] + [f"S2{n}" for n in names]
ax.set_xticks(np.arange(len(lbs)), labels=lbs)
ax.set_yticks(np.arange(len(lbs)), labels=lbs)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(lbs)):
    for j in range(len(lbs)):
        text = ax.text(j, i, np.round(c[i, j], 1),
                       ha="center", va="center", color="w")
plt.legend()
ax.set_title("Channel Coherence")
fig.tight_layout()
plt.show()
fig.savefig(out_dir+f"/Coh.png")