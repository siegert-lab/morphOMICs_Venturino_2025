## Installation

You can install kxa_analysis using either Conda or pip. Follow the steps below based on your preferred package manager.

### Using Conda
1. **Clone the Repository:**  
   ```bash
   git clone git@github.com:kxa_analysis.git  
   cd kxa_analysis
   ```

2. **Create and Activate the Conda Environment:**  
   ```bash
   conda create --name kxa-ana python=3.9
   conda activate kxa-ana
   ```

3. **Install the Package:**  
   ```bash
   pip install -e .
   ```

4. **Install Morphomics (Temporary Step)**  
   The `morphomics 2.0` package is not yet released, so you need to install it manually:  

   - Navigate to the directory where you want to clone the morphomics repository:  
     ```bash
     cd path/to/morphomics
     ```  

   - Clone and install:  
     ```bash
     git clone git@github.com:ThomasNgl/morphOMICs.git  
     cd morphOMICs  
     pip install -e .
     ```  

---

### Using pip
1. **Clone the Repository:**  
   ```bash
   git clone git@github.com:kxa_analysis.git  
   cd kxa_analysis
   ```

2. **Create and Activate the Virtual Environment:**  
   ```bash
   python -m venv kxa-ana  
   ```

   - On macOS/Linux:  
     ```bash
     source kxa-ana/bin/activate  
     ```  
   - On Windows:  
     ```bash
     kxa-ana\Scripts\activate  
     ```  

3. **Install the Required Packages:**  
   ```bash
   pip install -e .
   ```

4. **Install Morphomics (Temporary Step)**  
   The `morphomics 2.0` package is not yet released, so you need to install it manually:  

   - Navigate to the directory where you want to clone the morphomics repository:  
     ```bash
     cd path/to/morphomics
     ```  

   - Clone and install:  
     ```bash
     git clone git@github.com:ThomasNgl/morphOMICs.git  
     cd morphOMICs  
     pip install -e .
     ```  
