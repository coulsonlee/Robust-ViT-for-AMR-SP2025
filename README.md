# Robust-ViT-for-AMC-S&P-2025
[S&amp;P2025] Adversarial Robust ViT-based Automatic Modulation Recognition in Practical Deep Learning-based Wireless Systems

## Introduction
We present a robust AMR system based on Vision Transformer to defend against adversarial attacks in non-cooperative wireless environments. The proposed
design leverages a feature extraction module and tailored feature and positional embeddings within the Transformer encoder to enhance model robustness. We further introduce a noise-adaptive AT to mitigate the practical impact of noise in generating effective adversarial perturbation. Through comprehensive real-world experiments based on a self-collected dataset, we validate the efficiency, accuracy, and effectiveness of the proposed framework.

<img src="./images/intro.png">

The overview of our proposed ViT-based adversarial robustness AMR design. As the first step, using the novel ViT-based models is expected to significantly boost the AMR classification accuracy (on clean samples) by exploiting the long-range dependencies in each signal. To minimize the impact of adversarial attacks on clean signals, our second step is to perform adversarial training using adversarial samples crafted to fool the trained ViT-based model. Instead of generating a wireless signal as a malicious perturbation from an ideal environment, our key innovation is to introduce a noise-adaptive perturbation crafting strategy considering real noises in clean samples. By doing so, the generated practical perturbations included in the adversarial training can further enhance the model robustness in AMR.

## Our self-collected dataset **(dataset can be downloaded at [This link](https://blogs.clemson.edu/rcd/))**
[Dataset](https://app.globus.org/file-manager?origin_id=33fdf1d5-7b1c-4a21-8d3b-dd4b48d60dfa&origin_path=%2Fproject%2Flinkeg%2Fguolab%2F)
The following table shows the details of our collected dataset.

<img src="./images/Dataset Collection Parameters.png">

* Transmitter. We modulate wireless signals according to the official standard in Wi-Fi, Bluetooth, and ZigBee protocol.
* Receiver. The sampling rate is set for 20 MHz for all protocols, and the central frequency is 2,360MHz as in the standard.

## Usage guideline
* Download the dataset.
* Load the dataset with our sample code.

We will release the sample code later.
