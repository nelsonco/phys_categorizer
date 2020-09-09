# Phys_Categorizer: Categorize Physics Articles as Technical or Non-Technical
This tool categorizes physics articles supplied by the ArXiv as technical or non-technical, where technical and non-technical as categories that contain the following:

**Non-Technical Audience:**  
•General Physics  
•History and Philosophy of Physics  
•Physics and Society  
•Physics Education  
•Popular Physics  
  
**Technical Audience:**  
•Accelerator Physics  
•Atmospheric and Oceanic Physics  
•Atomic and Molecular Clusters  
•Atomic Physics  
•Biological Physics  
•Chemical Physics  
•Classical Physics  
•Computational Physics  
•Data Analysis  
•Statistics and Probability  
•Fluid Dynamics  
•Geophysics  
•Instrumentation and Detectors  
•Medical Physics  
•Optics  
•Plasma Physics  
•Space Physics  

## Dependency Installation

Use the package manager pip to install the necessary dependencies.

```bash
pip install pandas
```

```bash
pip install pdfminer
```

```bash
pip install nltk
```

```bash
pip install -U scikit-learn
```

```bash
pip install os
```

```bash
pip install plotly
```


```bash
conda install -c plotly plotly-orca
```

## How to use:  
### To use the preprocessed and catagorized article's associated text file:
Run main.py if using python 3  
Run main.ipynb if using jupyter notebook  
Note: main_directory = must be set to the full directory where your physics file is located  

```bash
main_directory = 'directory repositroy was cloned to' + '\physics'
```

### To download the documents mine them for text and categorize the files: 
Download the documents from Kaggle, in this experiment only the physics folders documents were downloaded.
https://www.kaggle.com/Cornell-University/arxiv

```bash
pip install kaggle
```
List files available in the ArXiv buckent  
```bash
# List files available in the ArXiv buckent  
gsutil ls gs://arxiv-dataset/arxiv/  

# Download all of the files in the physics folder into a local directory
gsutil cp gs://arxiv-dataset/arxiv/physics/ ./a_local_directory/  
```



**If using a regular python 3 environment**  
First run convertpdf.py  
Note:  
MODIFY: Modify the directory to where you downloaded the files  
MODIFY: Lines 72 and 77 with the directory you would like to saves  
the files to this should be the same directory as that of main.py.  
Next run main.py  

**If using the Jupyter Notebook environment**  
First run convertpdf.ipynb  
Note:  
MODIFY: Modify the directory to where you downloaded the files  
MODIFY: Lines 72 and 77 with the directory you would like to saves  
the files to this should be the same directory as that of main.ipynb.  
Next run main.ipynb    

