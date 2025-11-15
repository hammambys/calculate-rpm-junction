import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time

from evio.core.recording import open_dat
from evio.source.dat_file import DatFileSource

##############################################
# PARAMETERS
##############################################

FILE = "fan_varying_rpm.dat"  # <-- PUT YOUR DATASET FILE HERE
WINDOW_US = 5000  # 5 ms windows
ROI = (0, 240, 0, 240)  # x1, x2, y1, y2
MIN_PEAK_DIST = 3  # windows apart
HISTORY = 200  # number of windows to keep
NUM_BLADES = 3  # <-- ESTIMATE: typically 3-5 blades for a fan
PLAYBACK_SPEED = 1.0  # 1.0 = realtime, >1 faster, <1 slower
PLOT_INTERVAL_S = 0.05  # update plot at most every 50 ms


##############################################
# EVENT DECODING HELPERS
##############################################


def decode_events(words_u32, timestamps, order, start, stop):
    """Return arrays x, y, pol, ts for events in [start:stop]."""
    idx = order[start:stop]

    w = words_u32[idx]
    ts = timestamps[idx]

    # DAT CD8 layout: [31:28]=polarity (4 bits), [27:14]=y (14 bits), [13:0]=x (14 bits)
    x = (w & 0x3FFF).astype(np.int32)
    y = ((w >> 14) & 0x3FFF).astype(np.int32)
    pol = (((w >> 28) & 0xF) > 0).astype(np.int8)

    return x, y, pol, ts


def count_roi(x, y, roi):
    x1, x2, y1, y2 = roi
    return np.sum((x >= x1) & (x < x2) & (y >= y1) & (y < y2))


##############################################
# RPM ESTIMATION
##############################################


def estimate_rpm(peak_times_us, num_blades=1):
    """peak_times_us: list of timestamps in microseconds"""
    if len(peak_times_us) < 2:
        return 0.0
    diffs_us = np.diff(peak_times_us)
    period_us = (
        np.mean(diffs_us) * num_blades
    )  # Each peak is 1/num_blades of a rotation
    period_s = period_us / 1e6
    freq = 1.0 / period_s
    return freq * 60.0


def estimate_rpm_fft(counts, window_us, num_blades=1):
    """Estimate RPM by finding dominant frequency in the counts time-series.

    - `counts` : 1D sequence of event counts per window
    - `window_us` : window length in microseconds
    - returns RPM (float)
    """
    if len(counts) < 8:
        return 0.0
    arr = np.asarray(counts, dtype=float)
    arr = arr - arr.mean()
    n = arr.size
    # sampling rate in Hz (windows per second)
    fs = 1.0 / (window_us / 1e6)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    spec = np.abs(np.fft.rfft(arr))
    # ignore DC
    spec[0] = 0.0
    idx = np.argmax(spec)
    f_spike = freqs[idx]
    if f_spike <= 0:
        return 0.0
    # convert spike frequency to rotations per minute
    rpm = (f_spike * 60.0) / float(num_blades)
    return rpm


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
    # two-panel layout: left = event image (ROI), right = counts/time plot
    fig, (ax_img, ax_plot) = plt.subplots(1, 2, figsize=(10, 5))

    start_time = time.time()
    # playback pacing: record real start and recording start timestamp
    try:
        rec_start_ts_us = int(src._ranges[0].start_ts_us)
    except Exception:
        rec_start_ts_us = int(rec.timestamps[0])
    playback_start_real = time.time()
    last_plot_time = 0.0

    print("Starting event-based fan RPM estimation...")

    # prepare ROI image buffer and plotting objects
    x1, x2, y1, y2 = ROI
    roi_w = int(x2 - x1)
    roi_h = int(y2 - y1)
    img_buf = np.zeros((roi_h, roi_w), dtype=np.uint16)

    im = ax_img.imshow(img_buf, cmap="gray", vmin=0, vmax=1, origin="lower")
    ax_img.set_title("Event Image (ROI)")
    ax_img.axis("off")

    (line_plot,) = ax_plot.plot([], [])
    ax_plot.set_xlim(0, HISTORY)
    ax_plot.set_ylim(0, 1)
    ax_plot.set_title("Event Count")
    ax_plot.set_xlabel("Window index")
    ax_plot.set_ylabel("Event count")

    rpm_text = ax_img.text(
        0.02, 0.98, "", transform=ax_img.transAxes, color="cyan", fontsize=12, va="top"
    )

    for idx, batch in enumerate(src.ranges()):

        # decode events in this window
        x, y, pol, ts = decode_events(words, timestamps, order, batch.start, batch.stop)

        # count events in ROI and build ROI image
        x1, x2, y1, y2 = ROI
        mask = (x >= x1) & (x < x2) & (y >= y1) & (y < y2)
        c = int(np.count_nonzero(mask))
        counts.append(c)

        # update image buffer (reuse allocation)
        img_buf.fill(0)
        if c > 0:
            xs = (x[mask] - x1).astype(np.intp)
            ys = (y[mask] - y1).astype(np.intp)
            # accumulate events; use np.add.at to handle repeated indices
            np.add.at(img_buf, (ys, xs), 1)

        # detect peak
        if len(counts) > 5:
            if counts[-2] == max(list(counts)[-5:]):  # simple 5-window local max
                if idx - last_peak >= MIN_PEAK_DIST:
                    # Use window center timestamp for stability (microseconds)
                    center_ts = int((batch.start_ts_us + batch.end_ts_us) // 2)
                    peaks.append(center_ts)
                    last_peak = idx

        rpm = estimate_rpm(list(peaks), num_blades=NUM_BLADES)
        rpm_fft = estimate_rpm_fft(list(counts), WINDOW_US, num_blades=NUM_BLADES)

        # occasional diagnostics to help spot bias (print every 100 windows)
        if idx % 100 == 0:
            if len(peaks) >= 2:
                diffs = np.diff(np.asarray(list(peaks), dtype=np.int64))
                mean_diff = float(np.mean(diffs))
                print(
                    f"Window {idx:04d} | Events={c:5d} | RPM(peaks)={rpm:6.2f} | RPM(fft)={rpm_fft:6.2f} | mean_diff_us={mean_diff:7.1f}"
                )
            else:
                print(f"Window {idx:04d} | Events={c:5d} | RPM(fft)={rpm_fft:6.2f}")

        # Pace playback so overall runtime matches recording duration
        # target corresponds to the end timestamp of this batch
        target_elapsed_s = (
            (batch.end_ts_us - rec_start_ts_us) / 1e6 / float(PLAYBACK_SPEED)
        )
        target_real = playback_start_real + target_elapsed_s
        now = time.time()
        sleep_time = target_real - now
        if sleep_time > 0:
            time.sleep(sleep_time)

        # live plot (throttle updates to avoid slow redraws)
        if time.time() - last_plot_time >= PLOT_INTERVAL_S:
            # update image
            im.set_data(img_buf)
            im.set_clim(0, max(1, int(img_buf.max())))
            rpm_text.set_text(f"RPM: {rpm:.1f}")

            # update counts plot
            ydata = list(counts)
            xdata = list(range(max(0, len(ydata) - HISTORY), len(ydata)))
            line_plot.set_data(range(len(ydata)), ydata)
            ax_plot.set_xlim(0, max(HISTORY, len(ydata)))
            ax_plot.set_ylim(0, max(10, max(ydata) if ydata else 1))

            fig.canvas.draw_idle()
            plt.pause(0.001)
            last_plot_time = time.time()

    print("\nDone.")


if __name__ == "__main__":
    main()
