# augsig

**augsig** is a lightweight and modular Python package for augmenting physiological signals such as PPG and ECG. It supports test-time and training-time augmentation using smooth time and amplitude warping strategies (Bezier and PCHIP), along with synthetic signal generation.

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/augsig.git
cd augsig
```

Install dependencies:
```bash
pip install numpy
pip install scipy
```

---

## Usage 
### Example: See test.py file.
---

## Package Structure

```
augsig/
├── augmenter.py      # Augment class, augment_np, augment_torch
├── warper.py         # Bezier and PCHIP warping logic
├── utils.py          # Normalization and helpers
├── synthesizer.py    # Synthetic signal generation (extendable)
```

---

## 🔧 TODO

- Add CLI support for batch augmentation
- Export augmentation configs and results
- Add waveform visualizations for debug and publication

---

## 📄 License

MIT License

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.
