# RadioGalaxies-BNNs
Approx Bayesian Inference for Radio Galaxy Classification

## Installation
- git clone this repo ```<repo>```
- cd ```<repo>```
- Create a python virtual environment 
- Activate virtual env
- Install packages using the requirements.txt file: ``` pip install -r requirements.txt ```

## Directory Structure

## Training
 
1. HMC 
    - To run HMC chains run_hmc.py. This will save HMC chains for every checkpoint (every 1000 steps by default)
    - Run combine.py to thin the HMC chains for a specified thinning factor (default thinning factor = 1000)
    - Calculate the Gelman-Rubin diagnostic to examine the convergence of specific parameters by running convergence.py (calculated for last layer weights by default)
    - Each HMC chain was run on an A100 GPU for 170 hrs. Thinned chains for the model are available at <insert link>

2. Variational Inference
    - See <https://github.com/devinamhn/RadioGalaxies-BBB> 

3. (nonBayesian) CNN / Deep Ensembles
    - Train 10 nonBayesian CNN models for different random seeds and random shuffling between training and validation datasets by running cnn_train.py

4. Last-Layer Laplace Approximation (LLA)
    - Using the MAP values learned by the (non-Bayesian CNNs), fit Laplace approximations for last layer weights of the model by running laplace_approx.py

5. MC Dropout
    - Train 10 dropout models by running dropout_train.py
    
## Evaluation protocols


## Results 

## Data 
Our BNNs are trained on the MiraBest dataset. To examine the sensitivity of our BNNs to different types of distribution shifts we use radio galaxies from MIGHTEE and optical galaxies from GalaxyMNIST.

### MiraBest
### MIGHTEE
Download image and source catalogue fits for the COSMOS and XMMLSS fields from the [Early Science Data](https://archive-gw-1.kat.ac.za/public/repository/10.48479/emmd-kf31/index.html):

Expert classifications for 117 objects are available at [MIGHTEE FR Dataset](https://zenodo.org/records/8188867) <sup> citation </sup>
### GalaxyMNIST


<!-- ![Galaxy Images](/radiogalaxies_bnns/postage/galaxy_image.png) -->