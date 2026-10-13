# Eye Fatigue Scanner (live demo)

**Try it:** https://grufter6.github.io/eye-fatigue-scanner/

This is a webcam eye scanner that runs entirely in your browser using onnxruntime-web. Camera images never leave your device.

- **Eye open/closed:** a small CNN trained on 84,898 MRL eye images. It scored 97.7% on 5 people it never saw.
- **Alert vs fatigued:** DINOv2 ViT-S/14 plus a logistic regression trained on 120 labelled phone photos and 2,632 reviewed web photos. It scored 72.5% vs 53.3% guessing (nested cross-validation by day).
- **3-way alert / strained / fatigued:** 51.5% vs 38.8% guessing.
- **Blink & PERCLOS meter:** blink rate, blink length and the share of time the eye is closed over 60 s.

This is a science-fair project. It is a weak signal, not a medical diagnosis. The models were exported with `web/export.py` from the training repo.
