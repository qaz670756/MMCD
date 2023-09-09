# Change Detection beyond 2D with Multimodal Data

[arXiv Paper](https://google.com)

This repository contains the code and resources for multimodal change detection, a technique for detecting changes between different data modalities, such as satellite images and digital surface models (DSMs). 

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Testing](#testing)
- [Citation](#citation)
- [License](#license)

<a name="introduction"></a>
## Introduction

![compare](/utils/src/fig1_compare.jpg "compare")




![singlevsdouble](/utils/src/singlevsdouble.jpg "singlevsdouble")


<a name="dataset"></a>
## Dataset
We provide a DSM-to-image multimodal dataset, which detecting multi-category building change from height data and aerial images, called Hi-BCD. You can download the dataset via: [BaiduNetdisk](https://google.com) or [GoogleNetdisk](https://google.com).

![Data](/utils/src/data1.jpg "Data")

It is constructed for detecting 2D and 3D changes simultaneously from cross-dimensional modalities. Some samples are as follow:

 ![Data_sample](/utils/src/data_sample.jpg "Data_sample")

It includes 1500 pairs of high-resolution tiles emcompassing three cities in the Netherlands. The details of Hi-BCD dataset are as follow:
| Attribute                 | Category       | Amsterdam | Rotterdam | Utrecht |
| ------------------------- | ---------- | ---------- | ------- |------- |
|   changed objects         | newly-built    | 389        | 510        | 458     |
|                           | demolished     | 251        | 229        | 187     |
|   changed pixels          | amount         | 6.625M     | 5.139M     | 7.73M   |
|                           | $prop_{/total}$| 1.3%       | 1.0%       | 1.5%    |
|    samples                | total          | 500        | 500        | 500     |
|                           | with change    | 40.8%      | 34.2%      | 43%     |











<a name="repository-structure"></a>
## Repository Structure

- `src/`: Source code for multimodal change detection.
- `data/`: Example datasets or links to datasets.
- `models/`: Pre-trained models (if applicable).
- `results/`: Folder to store results.
- `docs/`: Documentation and tutorials.
- 
<a name="installation"></a>
## Installation

1. Clone this repository:

```bash
git clone https://github.com/your-username/multimodal-change-detection.git
```
Install the required dependencies:
pip install -r requirements.txt

## Usage
### Rreproduce results in the paper

'''bash
bash reproduce.sh 
'''

<a name="training"></a>
### Training
To train your own multimodal change detection model, follow the instructions in the Training documentation.

<a name="testing"></a>
### Testing
To perform change detection on your own data, check out the Testing tutorial.


<a name="citation"></a>
## Citation

If you find this code or dataset useful in your research, please consider citing our paper:

<a name="license"></a>
## License

Feel free to customize this template according to your specific project's details and needs. Replace placeholders like `your-username`, `your_paper_reference`, and the sections' content with your actual information. Additionally, create the relevant documentation files (e.g., `training.md` and `testing.md`) in the `docs/` directory to provide more detailed tutorials and usage instructions.


