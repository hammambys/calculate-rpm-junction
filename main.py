import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time

from evio.core.recording import open_dat
from evio.source.dat_file import DatFileSource

##############################################
# PARAMETERS
##############################################

FILE = "fan_const_rpm.dat"  # <-- PUT YOUR DATASET FILE HERE
WINDOW_US = 5000  # 5 ms windows
ROI = (0, 240, 0, 240)  # x1, x2, y1, y2
MIN_PEAK_DIST = 3  # windows apart
HISTORY = 200  # number of windows to keep

##############################################
# EVENT DECODING HELPERS
##############################################


def decode_events(words_u32, timestamps, order, start, stop):
    """Return arrays x, y, pol, ts for events in [start:stop]."""
    idx = order[start:stop]

    w = words_u32[idx]
    ts = timestamps[idx]

    x = (w & 0x7FF).astype(np.int32)
    y = ((w >> 11) & 0x7FF).astype(np.int32)
    pol = ((w >> 22) & 1).astype(np.int8)

    return x, y, pol, ts


def count_roi(x, y, roi):
    x1, x2, y1, y2 = roi
    return np.sum((x >= x1) & (x < x2) & (y >= y1) & (y < y2))


##############################################
# RPM ESTIMATION
##############################################


def estimate_rpm(peak_times):
    if len(peak_times) < 2:
        return 0.0
    diffs = np.diff(peak_times)
    period = np.mean(diffs)
    freq = 1.0 / period
    return freq * 60.0


##############################################
# MAIN
##############################################


def main():
    # Load full recording
    rec = open_dat(FILE, width=1280, height=720)

    # Build time windows
    src = DatFileSource(
        FILE, window_length_us=WINDOW_US, width=rec.width, height=rec.height
    )

    words = src.event_words
    order = src.order
    timestamps = rec.timestamps

    counts = deque(maxlen=HISTORY)
    peaks = deque(maxlen=HISTORY)
    last_peak = -999

    plt.ion()
    fig, ax = plt.subplots()

    start_time = time.time()

    print("Starting event-based fan RPM estimation...")

    for idx, batch in enumerate(src.ranges()):

        # decode events in this window
        x, y, pol, ts = decode_events(words, timestamps, order, batch.start, batch.stop)

        # count events in ROI
        c = count_roi(x, y, ROI)
        counts.append(c)

        # detect peak
        if len(counts) > 5:
            if counts[-2] == max(list(counts)[-5:]):  # simple 5-window local max
                if idx - last_peak >= MIN_PEAK_DIST:
                    peaks.append(time.time())
                    last_peak = idx

        rpm = estimate_rpm(list(peaks))

        print(f"Window {idx:04d} | Events={c:5d} | RPM={rpm:6.2f}", end="\r")

        # live plot
        ax.clear()
        ax.plot(counts)
        ax.set_title(f"Event Count — RPM: {rpm:.1f}")
        ax.set_xlabel("Window index")
        ax.set_ylabel("Event count")
        plt.pause(0.001)

    print("\nDone.")


if __name__ == "__main__":
    main()
