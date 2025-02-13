The instructions are for students using the teaching cluster.
### **1. Install Required Packages**
On the head node: 

First and foremost, install Conda. We recommend using [Miniconda](https://docs.anaconda.com/miniconda/install/). The teaching cluster is using Ubuntu Linux. Please follow the **Linux installation guidelines**.

After conda is initialized, create a `~/.bash_profile` file and write the contents below so that your `~/.bashrc` will run when you ssh log in:
```bash
if [ -f ~/.bashrc ]; then
  . ~/.bashrc
fi
```
Then run:
```bash
git clone https://github.com/ed-aisys/edin-mls-25-spring.git
cd edin-mls-25-spring/resources/1-pytorch-demo
conda create -n mlsys python=3.10 -y
conda activate mlsys
pip install -r requirements.txt
```
If you encounter a `No disk space` error, try logging into Interactive mode first and installing the environment on the node.

---

### **2. Run the Demo in Interactive Mode**
To start the demo interactively, use:
```bash
srun --gres gpu:1 --pty bash
```

This command will allocate and enter a compute node with one GPU. Once inside the compute node, run the following commands to execute the demo:

```bash
conda activate mlsys
python demo.py
```

If the script runs without any errors, congratulations!

---

### **3. Run the Demo in Batch Mode**
To run the demo in batch mode, first create a Bash script (e.g., `test.sh`) with the following content:

```bash
#!/bin/bash
source ~/miniconda3/bin/activate
conda activate mlsys
python demo.py
```

Then submit the job using:
```bash
sbatch --gres gpu:1 test.sh
```

If the output file does not show any errors, congratulations! The demo has successfully executed.
