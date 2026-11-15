<div align="center">

# 🔭 SPHEREx Explorer: Multi-Epoch Infrared Transient & Planet X Detection
**NASA Space Apps Challenge 2026 — Official Project Repository**

[![NASA Data](https://img.shields.io/badge/NASA_Data-IRSA_SPHEREx_DR1%2FQR3-0B3D91?style=for-the-badge&logo=nasa&logoColor=white)](https://irsa.ipac.caltech.edu/)
[![Live App](https://img.shields.io/badge/Live_App-SPHEREx_Explorer-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://spherex-explorer-2026.streamlit.app/)
[![Demo Video](https://img.shields.io/badge/Demo-Watch_Walkthrough-4285F4?style=for-the-badge&logo=googledrive&logoColor=white)](https://drive.google.com/file/d/1yeZU9GpD0aRBDXzyLciH6QACtieWdK2d/view?usp=drivesdk)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Astropy](https://img.shields.io/badge/Astronomy-Astropy_%7C_Reproject-orange?style=for-the-badge)](https://www.astropy.org/)

*An automated astronomical data retrieval, World Coordinate System (WCS) sky-alignment, and interactive multi-epoch flux differencing system designed to isolate moving celestial objects and Planet X candidates.*

[🚀 Launch Live App](https://spherex-explorer-2026.streamlit.app/) · [🎥 Watch Demo Video](https://drive.google.com/file/d/1yeZU9GpD0aRBDXzyLciH6QACtieWdK2d/view?usp=drivesdk) · [📓 Data Pipeline Notebook](./SPHEREx_Data_Collection.ipynb) · [📊 Metadata CSV](./spherex_metadata.csv)

</div>

---

## 🎯 Executive Summary

Detecting distant, faint, and slow-moving solar system bodies (such as trans-Neptunian objects, asteroids, comets, and hypothetical **Planet X** candidates) requires comparing deep-sky observations separated by meaningful time baselines.

**SPHEREx Explorer** automates this workflow end-to-end:
1. Queries **NASA/Caltech's Infrared Science Archive (IRSA)** to retrieve real calibrated Level-2 observations from the **SPHEREx** mission.
2. Aligns multi-epoch infrared observations separated by **~407 days** (`2025-04-29` vs. `2026-06-10` in the `SPHEREx-D1` spectral band) onto a common World Coordinate System (WCS) pixel grid.
3. Computes multi-epoch residual flux maps so stationary background stars cancel out while moving celestial bodies stand out, presented through an interactive web interface and demo walkthrough.

---

## 🎬 Live App & Demo Walkthrough

* **Interactive Web Application:** [https://spherex-explorer-2026.streamlit.app/](https://spherex-explorer-2026.streamlit.app/)
* **Project Demo Video:** [https://drive.google.com/file/d/1yeZU9GpD0aRBDXzyLciH6QACtieWdK2d/view?usp=drivesdk](https://drive.google.com/file/d/1yeZU9GpD0aRBDXzyLciH6QACtieWdK2d/view?usp=drivesdk)

---

## 🏗️ End-to-End System Architecture

```text
┌──────────────────────────────────────────────────────────────────────────┐
│ 1. DATA INGESTION (NASA / Caltech IRSA SIA2 TAP Service)                 │
│    • Queries SPHEREx DR1 / QR3 catalog by celestial coordinates (RA/Dec) │
│    • Filters Level-2 calibrated FITS exposures in SPHEREx-D1 band        │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ 2. ASTROPHYSICAL PREPROCESSING & WCS SKY ALIGNMENT                       │
│    • Parses primary ImageHDU arrays (2040 × 2040 pixels, float32)        │
│    • Imputes detector NaN artifacts & applies ZScale intensity scaling   │
│    • Reprojects Epoch 2 onto Epoch 1 WCS grid (astropy.wcs + reproject)  │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ 3. MULTI-EPOCH DIFFERENCING & INTERACTIVE EXPLORATION (Streamlit App)    │
│    • Computes residual flux map (Epoch 2 − Epoch 1) to isolate motion    │
│    • Exports standardized .npy arrays, metadata audit CSV & visual maps  │
│    • Renders live interactive dashboard for inspection & evaluation      │
└──────────────────────────────────────────────────────────────────────────┘


