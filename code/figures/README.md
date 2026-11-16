# Figures

`paper_style.py` holds the shared palette and font (the phase colors of Fig. 2). Keep it next to the scripts.

| Script | Figure | Input |
| --- | --- | --- |
| `make_fig3.py` | Fig. 3, room camera and microscope at matched timestamps | raw capture folder (frames, microscope video, frame times); uses `bifocal_viewer.py` |
| `make_fig4.py` | Fig. 4, ambient waveform, audited speakers, pyannote clusters | ambient WAV + release annotations |
| `make_fig5_fig6.py` | Fig. 5, cam03 confusion matrix; Fig. 6, phase timelines | per-frame test predictions (`code/phase_recognition`) |
| `bifocal_viewer.py` | not a paper figure: all streams of a run at one moment, plus clap and synchronization check sheets | raw capture folder |

Commands used for the paper:

```bash
python make_fig3.py --root <raw capture> --labels <release metadata> --run 3 --camera camera05 \
    --times 1100 1140 1170 1200 --tags "Adjusting microscope" "Lights off" "Tool handover" "Placement of second trocar"
python make_fig4.py --root <raw capture> --labels <release metadata> --run 7 --start 220
python make_fig5_fig6.py --repo ../phase_recognition --run 4
```

`make_fig3.py` and `bifocal_viewer.py` read the raw capture layout (`rec_*_Run_<n>/export_full/...`). To use them with the Hugging Face release, extract the per-camera archives into that layout.
