# NIRS-Vis: Decoding Visual Experiences from fNIRS Brain Signals

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Welcome to NIRS-Vis, a cutting-edge project that aims to unlock the mysteries of the human visual system by decoding visual experiences directly from fNIRS (functional near-infrared spectroscopy) brain recordings.

## 💡 Inspiration

Inspired by the groundbreaking **MinD-Vis** framework, NIRS-Vis leverages the strengths of fNIRS, a non-invasive neuroimaging technique, to reconstruct visual stimuli from brain activity patterns.

## 🚀 Our Goals

1. **Develop a robust model:** Train advanced deep learning models (transformers, autoencoders) on diverse fNIRS datasets to accurately decode visual stimuli from brain signals.
2. **Generate visual representations:** Utilize the Stable Diffusion model (and potentially other innovative approaches) to create visual representations of the decoded experiences.
3. **Advance fNIRS research:** Push the boundaries of fNIRS-based brain decoding and visualization, contributing to the broader field of neuroscience.

## 📊 Data Sources

* **In-House fNIRS Data:** Proprietary dataset collected using the Oxysoft device on two volunteers during visual stimulation with ImageNet images. (This dataset is intended for fine-tuning our deep learning models.)

* **Open Source fNIRS Datasets:**
    * **Mental Workload (fNIRS2MW):** 70-subject dataset publicly available on Box ([link](https://tufts.app.box.com/s/1e0831syu1evlmk9zx2pukpl3i32md6r/folder/144902920345)).
    * **Passive Auditory fNIRS Responses:** Dataset associated with a published paper (OSF: [link](https://osf.io/f6tdk/)).
    * **fNIRS Audio & Visual Speech:** Dataset accompanying a published paper (OSF: [link](https://osf.io/u23rb/)).

## 📚 Related Work

* **MinD-Vis:** The foundational project upon which NIRS-Vis is built. ([link](https://github.com/zjc062/mind-vis))
* **fNIRS Transformer:** A transformer model for fNIRS classification ([code](https://github.com/wzhlearning/fNIRS-Transformer), paper: [link](https://ieeexplore.ieee.org/document/9670659)).
* **Decoding Semantic Representations from fNIRS:** A study exploring semantic decoding from fNIRS. ([link](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5568915/))

## 📆 Timeline

Targeted completion by the first week of July 2024.

## 🙌 Contributing

We welcome contributions in the form of data analysis, model development, and general feedback. Please contact us for collaboration opportunities.

## 📄 License

This project is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for details.
