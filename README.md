# GSP x NetworkTheory

Final Project for course of "Signal Processing of Big Data", held by Prof. Sergio Barbarossa and Prof. Paolo di Lorenzo.

### Description

The goal of the project is to replicate the results of the work "Integration of network topological features and graph Fourier transform for fMRI data analysis", by Wang et al. (ISBI 2018).

In this work the authors combine two different approaches for classifying fMRI data: Graph Signal Processing and Network Theory. Obtaining features from both the approaches, they train an SVM model that returns good results in classifying individuals according their age.

We personally replicated this result, taking into account also the scenario where signals are sampled and then reconstructed. We applied these methodologies to the ABIDE dataset, obtaining coherent results, with both approaches.

### Contents

`features_scraper.py`: Takes data in input, and extract the features necessary to the SVM model.

`classification.py`: Perform the classification.

`utils.py`: Contains all the functions involved in the process.

`results.pdf`: Presentation of the results obtained.
