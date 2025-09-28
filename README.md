## What is TusoAI?

**TusoAI** is an open-source, agentic system for **scientific method optimization**. Given a task description and template (Python) script, TusoAI will autonomously implement and test optimizations to maximize a user-defined score. It mimics the process by which a researcher may optimize their own method: by considering their own knowledge of the task, extracting insights from the literature, diagnosing intermediate steps of their method/data, and considering feedback from previous attempts. This process can be initialized as _cold start_, without any existing method, or _warm start_, building off of an existing method. TusoAI outputs a final optimized method which scientists may use in downstream applications. See details below and in our paper.

![TusoAI overview](method_overview.png)

### Who should use TusoAI?

TusoAI is intended for scientists in the process of building their method to perform well on some benchmark, e.g., simulations, train/test splits, real data benchmarks, etc. It can also be applied to general ML tasks.

---

## How to use TusoAI

### Installation

Download and unzip this directory. run_tusoai.ipynb is the main starting point of running TusoAI, see details in further sections below.

TusoAI requires only 4 base packages, listed in requirements.txt, which can be installed with the following.

```bash
pip install -r requirements.txt
```

TusoAI does not install further packages while running. However, it may propose using further packages while optimizing. It is usually beneficial to have an environment set up with many useful packages beforehand, so if TusoAI proposes an optimization that requires a package, it is able to implement this. This can vary by domain, e.g., a single-cell researcher will have many single-cell packages installed, or an ML practicioner will have many ML packages installed. A general purpose environment which has many useful ML packages installed beforehand is the aideml package (https://github.com/WecoAI/aideml/tree/main). This can be installed as below:

```bash
pip install -U aideml
```

### Setup

### Running

### Extracting history

---

## Citation
