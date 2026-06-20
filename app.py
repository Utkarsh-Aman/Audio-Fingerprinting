import streamlit as st
import pickle, os, io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display

from fingerprint import (
    load_audio, compute_spectrogram, find_peaks, generate_hashes, identify,
    SAMPLE_RATE, N_FFT, HOP_LENGTH, FREQ_LIMIT,
)

# ── page config ─────────────────────────────────────────
st.set_page_config(page_title="EE200: Audio Fingerprinting",
                   page_icon="🎵", layout="wide")

# ── load pre-built database ─────────────────────────────
@st.cache_resource
def load_db():
    with open("song_database.pkl", "rb") as f:
        db = pickle.load(f)
    with open("song_metadata.pkl", "rb") as f:
        meta = pickle.load(f)
    return db, meta

hash_db, metadata = load_db()

# ── confidence thresholds ───────────────────────────────
MIN_SCORE = 10
MIN_RATIO = 2.0

# ── header ──────────────────────────────────────────────
st.title("🎵 EE200: Audio Fingerprinting")
st.caption("SIGNALS, SYSTEMS & NETWORKS · PROJECT")
st.markdown("""
## Group Members: 
**Utkarsh Aman** 241114  
**Manish Kajla** 240622
""")

# ── tabs ────────────────────────────────────────────────
tab_lib, tab_id, tab_batch = st.tabs(["📚 LIBRARY", "🎯 IDENTIFY", "📦 BATCH"])


# ── generate thumbnails ─────────────────────────────────
# The demo video has thumbnails for each song, so I generated thumbnails for each song.
# thumbnail genertion was taking minutes to load all song thumbnails
# The thumbnails are now pre-generated as tiny PNG files  inside 
# the `thumbnails/` folder by `build_db.py`. This means the UI loads instantly from disk!
#
# @st.cache_data(show_spinner=True)
# def get_thumbnail(peaks, color="cyan"):
#     import random
#     # Plot only a sparse sample to render instantly 
#     # i had tried to generate full fut that was denser 
#     # and was taking minutes to load all song thumbnails
#     plot_peaks = random.sample(peaks, min(len(peaks), 50))
#     
#     fig, ax = plt.subplots(figsize=(3, 2))
#     ax.scatter([p[0] for p in plot_peaks], [p[1] for p in plot_peaks], s=0.5, c=color, alpha=0.8)
#     ax.set_facecolor("#0e1117")
#     ax.set_xticks([])
#     ax.set_yticks([])
#     for spine in ax.spines.values():
#         spine.set_visible(False)
#     fig.patch.set_facecolor("#0e1117")
#     plt.tight_layout(pad=0)
#     
#     buf = io.BytesIO()
#     fig.savefig(buf, format="png", bbox_inches='tight', pad_inches=0, facecolor="#0e1117")
#     plt.close(fig)
#     return buf.getvalue()

# ═════════════════════════════════════════════════════════
#  TAB 1 — Library
# ═════════════════════════════════════════════════════════
with tab_lib:
    st.header("Indexed Audio Library")
    st.write(f"The database currently contains fingerprints for **{len(metadata)}** reference tracks.")

    songs = sorted(metadata.keys())
    colors = ["#00d4ff", "#ffb84d", "#ff7373", "#ccff00", "#b266ff"]
    for i in range(0, len(songs), 4):
        cols = st.columns(4)
        for j in range(4):
            idx = i + j
            if idx < len(songs):
                name = songs[idx]
                m = metadata[name]
                with cols[j]:
                    with st.container(border=True):
                        # Load pre-generated thumbnail from disk instead of calculating 
                        thumb_path = os.path.join("thumbnails", f"{name}.png")
                        if os.path.exists(thumb_path):
                            st.image(thumb_path, use_container_width=True)
                        else:
                            st.markdown("🎵") # Fallback icon just in case
                        st.markdown(f"**{name}**")
                        st.caption(f"{m['num_hashes']:,} hashes")


# ═════════════════════════════════════════════════════════
#  TAB 2 — Identify
# ═════════════════════════════════════════════════════════
with tab_id:
    st.header("Identify a Clip")
    st.write("Upload an audio clip or pick one of the sample files below.")

    uploaded = st.file_uploader("Upload a clip (.mp3, .wav, .ogg)",
                                type=["mp3", "wav", "ogg", "flac"],
                                key="id_upload") #supporting other formats too for resume :)

    # ──clip buttons ─────────────────────────────
    sample_dir = "samples"
    samples = sorted([f for f in os.listdir(sample_dir)
                    if f.endswith((".wav", ".mp3"))])
    if samples:
        st.write("**Or try a sample:**")
        for i, s in enumerate(samples):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.audio(os.path.join(sample_dir, s))
            with col2:
                if st.button(f"Try '{s}'", key=f"smp_{i}", use_container_width=True):
                    st.session_state["sel_sample"] = os.path.join(sample_dir, s)
                    st.session_state["sel_sample_name"] = s

    # ── resolve audio source ────────────────────────────
    query_y = None
    query_label = None

    if uploaded is not None:
        query_y, _ = librosa.load(io.BytesIO(uploaded.getvalue()),
                                  sr=SAMPLE_RATE, mono=True)
        query_label = uploaded.name
        st.session_state.pop("sel_sample", None)
    elif st.session_state.get("sel_sample"):
        query_y = load_audio(st.session_state["sel_sample"])
        query_label = st.session_state["sel_sample_name"]

    # ── run identification ──────────────────────────────
    if query_y is not None:
        with st.spinner("Identifying …"):
            result = identify(query_y, hash_db, metadata)

        # ── pipeline stats ──────────────────────────────
        st.subheader("Pipeline")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        t = result["timings"]
        sp_shape = result["spectrogram"].shape
        c1.metric("Spectrogram",  f'{t["spectrogram"]:.0f} ms',
                  f'{sp_shape[0]}×{sp_shape[1]}')
        c2.metric("Constellation", f'{t["constellation"]:.0f} ms',
                  f'{len(result["peaks"])} peaks')
        c3.metric("Hashing",      f'{t["hashing"]:.0f} ms',
                  f'{len(result["hashes"]):,} hashes')
        c4.metric("DB Lookup",    f'{t["db_lookup"]:.0f} ms',
                  f'{len(metadata)} tracks')
        c5.metric("Scoring",      f'{t["scoring"]:.0f} ms',
                  f'offset {result["best_offsets"].get(result["prediction"], {}).get("offset", "—")}')
        c6.metric("Total",        f'{t["total"]:.0f} ms')

        # ── match result ────────────────────────────────
        pred = result["prediction"]
        confident = (result["score"] >= MIN_SCORE and
                     result["ratio"] >= MIN_RATIO)

        if pred and confident:
            st.success("**MATCH FOUND**")
            st.markdown(f"## {pred}")
            ratio_str = (f'{result["ratio"]:.0f}×'
                         if result["ratio"] != float("inf")
                         else "∞×")
            st.write(f'cluster score **{result["score"]:,}** · '
                     f'**{ratio_str}** the runner-up')
        else:
            st.warning("No confident match found.")

        # ── candidate scores (horizontal bar chart) ─────
        if result["ranked"]:
            st.subheader("Candidate Scores")
            top = result["ranked"][:5]
            songs  = [r[0] for r in top]
            scores = [r[1] for r in top]

            fig_c, ax_c = plt.subplots(figsize=(10, max(2, len(top) * 0.6)))
            bars = ax_c.barh(range(len(songs)), scores, color="teal")
            ax_c.set_yticks(range(len(songs)))
            ax_c.set_yticklabels(songs)
            ax_c.invert_yaxis()
            ax_c.set_xlabel("Score")
            for i, v in enumerate(scores):
                ax_c.text(v + max(scores) * 0.01, i, str(v), va="center")
            plt.tight_layout()
            st.pyplot(fig_c)
            plt.close(fig_c)

        # ── STEP 1 — spectrogram → constellation ───────
        st.markdown("---")
        st.subheader("Step 1 · Feature Extraction")
        st.markdown("**From spectrogram to constellation**")
        st.write(
            "First, the audio is transformed into a spectrogram (left) to visualize frequency content over time. "
            "To create a noise-resistant fingerprint, we extract only the local energy maxima. "
            f"This results in a constellation map (right) containing the **{len(result['peaks']):,} strongest peaks**, "
            "discarding amplitude data to remain robust against volume and EQ variations."
        )

        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # spectrogram
        librosa.display.specshow(result["spectrogram"],
                                 x_axis="time", y_axis="hz",
                                 sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
                                 ax=ax1, cmap="magma")
        ax1.set_ylim(0, 5000)
        ax1.set_title("Spectrogram")

        # constellation scatter
        peak_t = [p[0] * HOP_LENGTH / SAMPLE_RATE for p in result["peaks"]]
        peak_f = [p[1] * SAMPLE_RATE / N_FFT      for p in result["peaks"]]
        ax2.scatter(peak_t, peak_f, s=1, c="cyan", alpha=0.6)
        ax2.set_xlim(0, result["times"][-1] if len(result["times"]) else 1)
        ax2.set_ylim(0, 5000)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Frequency (Hz)")
        ax2.set_title(f"Constellation ({len(result['peaks']):,} peaks)")
        ax2.set_facecolor("#111")

        plt.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

        # ── STEP 2 — full fingerprint of matched song ──
        if pred and pred in metadata:
            st.markdown("---")
            st.subheader("Step 2 · Database Search")
            st.markdown("**Where in the song?**")
            st.write(
                f"We generated **{len(result['hashes']):,} hash pairs** from the query constellation "
                "and matched them against the full database. Below is the complete constellation for "
                f"*{pred}* from our storage. The shaded region indicates "
                "the predicted time window where the clip originated."
            )

            full_peaks = metadata[pred]["peaks"]
            fig2, ax = plt.subplots(figsize=(14, 4))
            ax.scatter([p[0] for p in full_peaks],
                       [p[1] for p in full_peaks],
                       s=1, c="cyan", alpha=0.4)

            # highlight query window
            off = result["best_offsets"][pred]["offset"]
            q_len = result["spectrogram"].shape[1]
            ax.axvspan(off, off + q_len, color="white", alpha=0.15,
                       label="query window")
            ax.set_xlabel("frame")
            ax.set_ylabel("freq bin")
            ax.set_title(f"Full fingerprint — {pred}")
            ax.set_facecolor("#111")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

        # ── STEP 3 — offset histogram ──────────────────
        if pred and pred in result["best_offsets"]:
            st.markdown("---")
            st.subheader("Step 3 · The Proof")
            st.markdown("**The alignment spike**")
            st.write(
                "To avoid false positives, each matching hash calculates a time difference "
                "(reference time minus query time). Incorrect matches distribute randomly, "
                "but true matches perfectly align at the exact same offset. "
                f"Here, **{result['score']:,} independent hashes voted for the identical offset**, "
                "confirming the match."
            )

            hist = result["best_offsets"][pred]["histogram"]
            best_offset = result["best_offsets"][pred]["offset"]
            offs = sorted(hist.keys())
            cnts = [hist[o] for o in offs]

            fig3, ax3 = plt.subplots(figsize=(14, 4))
            colors = ["orange" if o == best_offset else "teal" for o in offs]
            ax3.bar(offs, cnts, width=1, color=colors)
            ax3.set_xlabel("time offset  (database frame − query frame)")
            ax3.set_ylabel("# hashes")
            ax3.set_title("Offset Histogram")

            # annotate the spike
            if cnts:
                ax3.annotate(
                    f"{result['score']:,} hashes\nalign here",
                    xy=(best_offset, result["score"]),
                    xytext=(best_offset + max(1, len(offs) * 0.1),
                            result["score"] * 0.8),
                    arrowprops=dict(arrowstyle="->", color="orange"),
                    fontsize=10, color="orange",
                )
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close(fig3)


# ═════════════════════════════════════════════════════════
#  TAB 3 — Batch
# ═════════════════════════════════════════════════════════
with tab_batch:
    st.header("Batch")
    st.subheader("Identify many clips at once")
    st.write(
        "Upload multiple audio query clips. to test the system in bulk. Each file is compared against "
        "our **indexed database**, and the final output is saved to a "
        "`results.csv` containing the `filename` and `prediction`. "
        "If the system cannot confidently identify a match, the prediction will be logged as `none`."
    )

    batch_files = st.file_uploader(
        "Upload clips",
        type=["mp3", "wav", "ogg", "flac"],
        accept_multiple_files=True,
        key="batch_upload",
    )

    if batch_files:
        if st.button("🚀 Run batch", type="primary"):
            results = []
            bar = st.progress(0, text="Processing …")

            for i, f in enumerate(batch_files):
                y, _ = librosa.load(io.BytesIO(f.getvalue()),
                                    sr=SAMPLE_RATE, mono=True)
                r = identify(y, hash_db, metadata)

                confident = (r["score"] >= MIN_SCORE and
                             r["ratio"] >= MIN_RATIO)
                pred = r["prediction"] if (r["prediction"] and confident) else "none"
                results.append(dict(filename=f.name, prediction=pred))
                bar.progress((i + 1) / len(batch_files),
                             text=f"Processing {i+1}/{len(batch_files)} …")

            bar.empty()

            # show results
            st.subheader("Results")
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True, hide_index=True)

            matched = sum(1 for r in results if r["prediction"] != "none")
            none_ct = len(results) - matched
            st.write(
                f"**{matched} / {len(results)}** clips matched to a track "
                f"({none_ct} returned `none`)."
            )

            csv = df.to_csv(index=False)
            st.download_button("⬇ Download results.csv", csv,
                               file_name="results.csv", mime="text/csv")
