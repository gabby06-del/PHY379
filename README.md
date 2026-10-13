# PHY379-Self-supervised-learning-research-project-
This is my GitHub repo for all the work done for my PHY379Y1 research project:
 - **Project Title**: Solving Inverse Problems in MRI with Self-Supervised Learning

Throughout the year I will be updating this repo with self written code, written pieces concerning relevant research, and any experimentation done with cloned repos. 

This README will contain all relevant citations and credits to those repos used as well as a progress report table of my research thus far. 

| Date | Work done | Output and next steps |
| :---:         |     :----:      |          :---: |
| Oct 6   | created git repository and did my first commits   | I debugged and utilized code from the ICLR 2025 paper, “UNSURE: Unknown Noise Level Stein's Unbiased Risk Estimator” by Julian Tachella, Mike Davies, and Laurent Jacques. Through this work, I was able to generate images using SURE for MRI images. This allowed me to become more familiar with the structure of the code, the expected inputs and outputs, and how SURE can be applied to image reconstruction. As a next step, I will develop a small project to better understand how SURE and GSURE work when applied to images outside of MRI. This will allow me to develop a stronger understanding of the underlying methods before applying them more directly to medical imaging problems. <br> **Output Images from Debugged Code** : <img width="594" height="200" alt="Screenshot 2026-10-06 at 12 13 45 PM" src="https://github.com/user-attachments/assets/78c32a04-f64f-48c4-84f2-dd0c5584b1d3" /> |
|Oct 13 |   Created a project relating SURE, MSE, and soft threshold function to better understand the math and logic behind SURE **(paper paired with project in progress)**|  This project demonstrates **unsupervised signal denoising** using **Stein's Unbiased Risk Estimate (SURE)** paired with soft thresholding. It generates a synthetic noisy waveform with sharp step changes, evaluates risk across a range of thresholds (τ), and reconstructs the signal without needing prior knowledge of the ground-truth data which is what is needed for my research project. `reconstruction_plot.png`:<img width="1337" height="496" alt="Screenshot 2026-10-13 at 3 29 40 PM" src="https://github.com/user-attachments/assets/1ede60c1-a699-43b2-8e78-e22083d6eca0" />  **Left Panel**: Shows how closely the unsupervised SURE curve tracks the actual, hidden True MSE curve.  **Right Panel**: Displays the raw, noisy input overlaid with the final filtered/denoised output. `groundtruth_plot.png`:<img width="1139" height="580" alt="Screenshot 2026-10-13 at 3 28 58 PM" src="https://github.com/user-attachments/assets/6e023e0e-31d3-4101-bcfb-9815cc5f98ba" />    A clean visual mapping of the targeted underlying pattern before noise was applied.|

## Citations 
@inproceedings{
tachella2025unsure,
title={{UNSURE}: self-supervised learning with Unknown Noise level  and Stein's Unbiased Risk Estimate},
author={Juli{\'a}n Tachella and Mike Davies and Laurent Jacques},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=ScVnYBaSEw}
}
