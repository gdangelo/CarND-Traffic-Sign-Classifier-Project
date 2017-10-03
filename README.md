# CarND-Traffic-Sign-Classifier-Project
> Traffic Sign Classification Project for Self-Driving Car Nanodegree.

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The purpose of this project is to use deep neural networks and convolutional neural networks to classify traffic signs. I built, trained and validated a model so it can classify traffic sign images using the German Traffic Sign Dataset. After the model was trained, I then tried out my model on images of German traffic signs found on the web. More info in the [writeup](https://github.com/gdangelo/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup.md).

## Overview
Starting to work on this project consists of the following steps:

1. Install [`miniconda`](http://conda.pydata.org/miniconda.html) on your computer
2. Create a new `conda` [environment](http://conda.pydata.org/docs/using/envs.html) using this project
3. Each time you wish to work, activate your `conda` environment
4. Run the Jupyter notebook and visit [http://localhost:8000](http://localhost:8000)

---

## Installation

**Download** the latest version of `miniconda` that matches your system.

**NOTE**: There have been reports of issues creating an environment using miniconda `v4.3.13`. If it gives you issues try versions `4.3.11` or `4.2.12` from [here](https://repo.continuum.io/miniconda/).

|        | Linux | Mac | Windows | 
|--------|-------|-----|---------|
| 64-bit | [64-bit (bash installer)][lin64] | [64-bit (bash installer)][mac64] | [64-bit (exe installer)][win64]
| 32-bit | [32-bit (bash installer)][lin32] |  | [32-bit (exe installer)][win32]

[win64]: https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86_64.exe
[win32]: https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86.exe
[mac64]: https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
[lin64]: https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
[lin32]: https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86.sh

**Install** [miniconda](http://conda.pydata.org/miniconda.html) on your machine. Detailed instructions:

- **Linux:** http://conda.pydata.org/docs/install/quick.html#linux-miniconda-install
- **Mac:** http://conda.pydata.org/docs/install/quick.html#os-x-miniconda-install
- **Windows:** http://conda.pydata.org/docs/install/quick.html#windows-miniconda-install

**Clone** the project and **download** the dataset

```sh
git clone https://github.com/gdangelo/CarND-Traffic-Sign-Classifier-Project.git
cd CarND-Traffic-Sign-Classifier-Project
```

Then download the dataset by following this [link](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip). This is a pickled dataset in which the images are already resized to 32x32.

Create a *data/* folder at the root of the project and unzip the dataset files inside. You should now have 3 pickle files:

```sh
CarND-Traffic-Sign-Classifier-Project/
 └── data/ 
      ├── test.p
      ├── train.p
      └── valid.p
```

**Setup** your `carnd` environment. 

If you are on Windows, **rename**   
`meta_windows_patch.yml` to   
`meta.yml`

**Create** carnd.  Running this command will create a new `conda` environment that is provisioned with all libraries you need to be successful in this program.
```
conda env create -f environment.yml
```

*Note*: Some Mac users have reported issues installing TensorFlow using this method. The cause is unknown but seems to be related to `pip`. For the time being, we recommend opening environment.yml in a text editor and swapping
```yaml
    - tensorflow==0.12.1
```
with
```yaml
    - https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.12.1-py3-none-any.whl
```

**Verify** that the carnd environment was created in your environments:

```sh
conda info --envs
```

**Cleanup** downloaded libraries (remove tarballs, zip files, etc):

```sh
conda clean -tp
```

### Uninstalling 

To uninstall the environment:

```sh
conda env remove -n carnd
```

---

## Usage

Now that you have created an environment, in order to use it, you will need to activate the environment. This must be done **each** time you begin a new working session i.e. open a new terminal window. 

**Activate** the `carnd` environment:

### OS X and Linux
```sh
$ source activate carnd
```
### Windows
Depending on shell either:
```sh
$ source activate carnd
```
or

```sh
$ activate carnd
```

Now all of the `carnd` libraries are available to you. 

**Open** the code in a Jupyter Notebook: 

```sh
$ jupyter notebook Traffic_Sign_Classifier.ipynb
```

That's it. To exit the environment when you have completed your work session, simply close the terminal window.

---

## Questions or Feedback

> Contact me anytime for anything about my projects or machine learning in general. I'd be happy to help you :wink:

* Twitter: [@gdangel0](https://twitter.com/gdangel0)
* Linkedin: [Grégory D'Angelo](https://www.linkedin.com/in/gregorydangelo)
* Email: [gregory@gdangelo.fr](mailto:gregory@gdangelo.fr)
