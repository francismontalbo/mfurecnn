# mfurecnn
<p> Author: Francis Jesmar P. Montalbo </p>
<p> Affiliation: Batangas State University </p>
<p> Email: francismontalbo@ieee.org </p>
<p><a href="https://francismontalbo.github.io">Personal Webpage</a></p>
***PLEASE CONTANCT ME IF YOU ARE HAVING TROUBLE. I CAN OFFER ASSITANCE***

# Dataset: 
<p><a href="https://datasets.simula.no/kvasir/"></a>KVASIR Dataset</p>
<h3>Paper to cite:</h3>
<p>KVASIR: A Multi-Class Image Dataset for Computer Aided Gastrointestinal Disease Detection</p>

***:heavy_exclamation_mark:For a faster method, you may download the already prepared dataset used in the given link below.*** 

<a href="">CLICK ME FOR THE PREPARED DATASET USED IN THIS STUDY</a>

# How to use:
# Disclaimer
***:heavy_exclamation_mark:If training the model, the dependencies included a `tensorflow-gpu`. You may change the `tensorflow-gpu` to `tensorflow` if no GPU is to be used. However, the results from the paper were produced using a GPU (RTX 3060 12gb) and may have slight differences***

Dependencies included in the `requirements.txt`: 
- jupyter==1.0.0
- keras==2.4.3
- matplotlib==3.4.1
- numpy==1.19.5
- opencv-python==3.4.11.41
- pandas==1.2.4
- Pillow==8.2.0
- scikit-learn==0.24.1
- scikit-image==0.18.1
- scikit-plot==0.3.7
- scipy==1.2.0
- tf-nightly-gpu==2.6.0 (Note: This is optional and can train even with just a CPU or tensorflow non-gpu variant. Nightly is used to compensate the new RTX 3060 card)

========================================================================================
## General Instruction:
You may clone using git or download the repository and extract the files manually:
- Once cloned, CD into the folder and enter `pip install -r requirements.txt`. 
- After installation of the dependecies, there are two options, either evaluate from the given weights (EASY and FAST) or train the model again (DIFFICULT and LENGTHY).

========================================================================================


***:heavy_exclamation_mark: HOW TO USE:*** 
============================================================================================================
**First (easier):**
- Clone this repository or download as zip.

- Install the requirements on a newly created environment to prevent issues with other existing ones.

- Directly open the `006-Evaluator.ipynb` go to the `Kernel` tab above then proceed with `Restart & Run All` to evaluate. 

- In cell #5 of the `006-Evaluator.ipynb` the `architecture` and `condition` variables have values that can be changed to evaluate the other models. As `architecture` has a list of `['MFuRe', 'MFNR']` and the condition has `['alpha', 'standard', 'no']`. Values can be interchanged as needed depending on the model to be evaluated. 

- For testing, open the `007-Tester_with_gradcam.ipynb` go to the `Kernel` tab above then proceed with `Restart & Run All` to test.

- In cell #9 of the `007-Tester_with_gradcam.ipynb` the `case` has values that can be changed to evaluate the various cases in isolation `a` for normal, `b` for ulcer, `c` for poylp, `d` for esophagitis.

**Second (difficult):**
- Make sure to download the <a href="https://drive.google.com/drive/folders/1WVKRTS5Wg8FdL7GCKwebIUz9jXnKx9nH?usp=sharing">PREPARED dataset</a> and extract it to a folder within the `mfure_cnn/` like `mfure_cnn/data/`

- Clone this repository or download as zip.

- Install the requirements on a newly created environment to prevent issues with other existing ones.

- Open one of the trainers like `000-MFuReCNN_alpha_do.ipynb` then go to the `Kernel` tab above then proceed with `Restart & Run All` to train. You may train the other models if needed that has notebook numbers from `000` to `005`. If there are trained models that exist in the `mfurecnn/models/` they will be overwritten depending on the model re-trained. BE CAREFUL. 

- Once trained, you may now again use the `006-Evaluator.ipynb` and `007-Tester_with_gradcam.ipynb` and you are done. Make sure that the respective weights are present.

**REMEMBER THIS IS A LONGER PROCESS (Second process) WHEN TESTING AND SIMULATING THE MODEL.**
