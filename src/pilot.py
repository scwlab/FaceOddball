# Libraries
import csv
import os
import pickle

import matplotlib.pyplot as plt
import mne
import numpy as np
import seaborn as sns
import pyxdf

fname = "/home/beto/Documents/db/sub-P0102_ses-S001_task-Odd1_run-001_eeg.xdf"
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

# Sample drop plot
cum_dropped = t - min(t)
abs_dropped = np.diff(cum_dropped)

sns.displot(1/abs_dropped, kde=True)
plt.xlim(0,sfreq*2)
plt.axvline(x=sfreq, c="red", label="Expected $F_s$")
plt.axvline(x=np.mean(1/abs_dropped), c="g", linestyle="--", label="Mean $F_s$")
plt.legend()
plt.title(f"Sampling frequency distribution {fname[4:-4]}")
plt.show()

plt.figure()
plt.plot(cum_dropped[:-1], 1/abs_dropped, label="Empirical $F_s$")
plt.axhline(y=512, label="Theoretical sfreq", linestyle="--", c="r")
plt.title(f"Sample frequency through time {fname[4:-4]}")
plt.legend()
plt.show()

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

info1 = mne.create_info(names, sfreq, ch_types=types)
info2 = mne.create_info(names, sfreq, ch_types=types)

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


x = np.diff(np.float32(tr[:,1]))
errors = x[x<=2]

# Plot that marker error
rel = sns.displot(errors, rug=True, kde=True)
rel.fig.suptitle(f"File {fname[4:-4]}, max {max(errors):.2f} sd {np.std(errors):.2f}")
rel.set(xlabel="timing error (ms)")
plt.show()

# Set annotations
annots = mne.Annotations(markers_t,
                        np.zeros_like(markers_t),
                        labels,
                        orig_time=None)
# Montage
info1.set_montage(mne.channels.make_standard_montage("standard_1020"))
raw1 = mne.io.RawArray(data[0:8,:], info1)
raw1._filenames = [fname]
# raw.set_meas_date(orig_time=orig_time)
raw1.set_annotations(annots)

raw = raw1



# Setup visualization scalings
viz_scalings = dict(eeg=1e2, eog=1e-4, ecg=1e3, bio=1e-7, misc=1)

# Setup evoked array for GA
epochsGA = {"landscape": [], "face": []}


    # Create directory for individual results
    out_dir = f"doc/subj/{fname}"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    else:
        continue


    print(f"Processing {fname}.xdf ...")
    # Open file
    raw = mne_read_xdf(fname=f"out/{fname}.xdf",
                       types=types,
                       )
    orig = raw.copy()
    ## ============================================
    raw = orig
    %matplotlib qt
    raw.plot(scalings=viz_scalings)
    plt.show()

    ## Filtering
    raw.filter(l_freq=0.1, h_freq=30)
    # raw.filter(l_freq=0.1, h_freq=None)

    fig = raw.plot_psd(fmax=60, average=True, show=False)
    plt.show()
    # fig.savefig(out_dir+f"/00_avg_psd.png")
    # raw.plot(scalings=viz_scalings)

    # scalings = dict(eeg=1e-5, eog=1e-4, ecg=1e-4, bio=1)
    scalings = dict(eeg=1e2, ecg=1e3, bio=1e-7, misc=1)

    ## Bad channel rejection
    if raw.info["bads"]:
        raw.interpolate_bads(exclude=("EOGU", "EOGD", "EOGL", "EOGR"))
    # raw.plot(scalings=viz_scalings)

    ## Re-reference
    # Reference Electrode Standardization Technique infinity reference
    # mne.set_eeg_reference(raw, 'average', copy=False)  # in-place
    raw.plot(scalings=viz_scalings)
    plt.plot()

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
        tmin=-0.1,
        tmax=0.5,
        reject=reject_criteria,
        flat=flat_criteria,
        preload=True,
    )
    fig = epochs.plot_drop_log(subject=fname, show=False)
    plt.show()
    # fig.savefig(out_dir+f"/00_dropped_epochs.png")

    ## Epoch rejection
    # TODO: This requires visual inspection and marking bad epochs.

    control = epochs["landscape"].average()
    target  = epochs["face"].average()
    conditions = {"NT": control, "T": target}

    for c in conditions:
        conditions[c].plot_joint(times=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])

        conditions[c].plot_topomap(times=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5], average=0.05)
        plt.show()

    # EOG
    # Reject epochs based on maximum peak-to-peak signal amplitude (PTP)
    reject_criteria = dict(eog=1e-2)  # 10 mV
    # Reject epochs based on minimum peak-to-peak signal amplitude (PTP).
    flat_criteria = dict(eog=1e-7)   # .1 µV
    eog_epochs = mne.preprocessing.create_eog_epochs(raw,
                                                     reject=reject_criteria,
                                                     flat=flat_criteria,
                                                     baseline=(None, -0.2),
                                                     )
    average_eog = eog_epochs.average()
    l = eog_epochs.plot_image(combine='mean', show=False)
    for i, fig in enumerate(l):
        # fig.savefig(out_dir+f"/02_eog_mean_{i}.png")
    fig = average_eog.plot_joint(show=False)
    # fig.savefig(out_dir+f"/02_eog_average.png")
    plt.close("all")

    # ECG
    # # Reject epochs based on maximum peak-to-peak signal amplitude (PTP)
    # reject_criteria = dict(eog=1e-3)  # 1 mV
    # # Reject epochs based on minimum peak-to-peak signal amplitude (PTP).
    # flat_criteria = dict(eog=1e-7)   # .1 µV
    ecg_epochs = mne.preprocessing.create_ecg_epochs(raw)
    average_ecg = ecg_epochs.average()
    # FIX: plot still shows
    fig = average_ecg.plot_joint(show=False)
    # fig.savefig(out_dir+f"/02_ecg_average.png")
    plt.close("all")
    #
    # ## ICA
    # method = 'fastica'
    # # method = 'picard'
    # # TODO: remove the selected bad channels from n_components
    # n_components = len([i for i in raw.get_channel_types() if i == "eeg"])  # if float, select n_components by explained variance of PCA
    # # if n_components > 24:
    # #     n_components = 24
    # # else:
    # #     raise "Less than 24 channels! check your input file."
    # decim = 4
    # random_state = 42
    # # Reject epochs based on maximum peak-to-peak signal amplitude (PTP)
    # reject_criteria = dict(eog=1e-3)  # 1 mV
    # # Reject epochs based on minimum peak-to-peak signal amplitude (PTP).
    # flat_criteria = dict(eog=1e-7)   # .1 µV
    # ica = mne.preprocessing.ICA(n_components=n_components,
    #                             max_iter="auto",
    #                             method=method,
    #                             random_state=random_state)
    # ica.fit(raw, decim=decim, reject=reject_criteria)
    #
    # # ICA plots
    #
    # # Create directory for individual component properties
    # ica_dir = f"doc/subj/{fname}/03_ica_components/"
    # if not os.path.exists(ica_dir):
    #     os.makedirs(ica_dir)
    #
    # l = ica.plot_properties(epochs, show=False)
    # for i, fig in enumerate(l):
    #     fig.savefig(ica_dir+f"/03_ica_comp{i}.png")
    #     plt.close("all")
    #
    # l = ica.plot_components(show=False)
    # for i, fig in enumerate(l):
    #     fig.savefig(out_dir+f"/03_ica_components_{i}.png")
    #     plt.close("all")
    #
    # fig = ica.plot_sources(epochs.average(), show=False)  # plot ICs applied to the averaged epochs
    # fig.savefig(out_dir+f"/03_ica_epochs_sources.png")
    # plt.close("all")
    #
    #
    #
    # # ICA eog removal
    # eog_inds, scores = ica.find_bads_eog(epochs,
    #                                      # threshold=0.99,
    #                                      # measure="correlation",
    #                                      verbose=True,
    #                                      )
    #
    # n_eog_comp = len(eog_inds)
    # print(f"EOG ICA components found: {n_eog_comp}")
    #
    # if n_eog_comp:  # only try EOG artefact removal if EOG artefacts were found
    #     try:
    #         p = ica.plot_properties(epochs, picks=eog_inds, show=False)  # Properties of EOG component
    #         for i, fig in enumerate(p):
    #             fig.savefig(out_dir+f"/04_ica_c{i}_properties.png")
    #         plt.close("all")
    #     except (IndexError, ValueError):
    #         pass
    #
    #     fig = ica.plot_scores(scores, show=False)  # look at r scores of components
    #     fig.savefig(out_dir+f"/04_ica_scores.png")
    #     plt.close("all")
    #
    #     # fig = ica.plot_sources(epochs, show_scrollbars=False)  # EOG matches highlighted
    #     # fig.savefig(out_dir+f"/01_ica_eog_sources.png")
    #     # plt.close("all")
    #     fig = ica.plot_sources(average_eog, show=False)  # plot ICs applied to the averaged EOG epochs, with EOG matches highlighted
    #     fig.savefig(out_dir+f"/04_ica_eog_average.png")
    #     plt.close("all")
    #     # fig = ica.plot_overlay(average_eog, exclude=eog_inds, show=False)
    #     # fig = ica.plot_overlay(average_eog, show=False)
    #     # fig.savefig(out_dir+f"/01_ica_eog_overlay.png")
    #     # plt.close("all")
    #
    #     # Exclude trials with high EOG correlation
    #     ica.exclude = []
    #     ica.exclude.extend(eog_inds)
    #
    #     # Return to ch space
    #     epochs_ica = epochs.copy()
    #     ica.apply(epochs_ica)
    #
    #     control   = epochs_ica["img"].average()
    #     scrambled = epochs_ica["sf"].average()
    #     toon      = epochs_ica["toon"].average()
    #     conditions = {"control": control, "scrambled": scrambled, "toon": toon}
    #
    #     for c in conditions:
    #         conditions[c].plot_joint(show=False)
    #         plt.savefig(out_dir+f"/05_ica_joint_{c}.png")
    #         plt.close("all")
    #
    #         # conditions[c].plot_topomap(times=[-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], average=0.05, show=False)
    #         # plt.savefig(out_dir+f"/05_ica_topomap_{c}.png")
    #         # plt.close("all")
    #
    #     epochs = epochs_ica

    ## Visualize ERPs
    control = epochs["landscape"]
    target  = epochs["face"]
    conditions = {"NT": control, "T": target}
    for c in conditions:
        conditions[c].plot_image(
                                sigma=1,
                                combine="mean",
                                title=c,
                                # show=False,
                                )

    conditions = {"control": list(epochs["landscape"].iter_evoked()),
                  "target": list(epochs["face"].iter_evoked())}

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
        fig = mne.viz.plot_compare_evokeds(conditions,
                                        title=roi,
                                        picks=rois[roi],
                                        # show=False,
                                        combine='mean')



    ## Storing epochs
    control   = epochs["img"]
    scrambled = epochs["sf"]
    toon      = epochs["toon"]
    conditions = {"control": control, "scrambled": scrambled, "toon": toon}

    for c in conditions:
        epochsGA[c].append(conditions[c])

## Grand Average
with open("epochsGA.pkl", "wb") as f:
    pickle.dump(epochsGA, f, protocol=pickle.HIGHEST_PROTOCOL)

# with open("epochsGA.pkl", 'rb') as f:
#     epochsGA = pickle.load(f)

# grand_average = mne.grand_average([evoked1, evoked2, ...])
