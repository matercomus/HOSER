## <div align="center"> Knowledge Distillation for Trajectory Generation </div>
## <div align="center"> Compressing LM-TAD into HOSER </div>

<div align="center">

[![Work in Progress](https://img.shields.io/badge/Status-Work%20in%20Progress-yellow)]()
[![Research](https://img.shields.io/badge/Research-Knowledge%20Distillation-blue)](docs/LMTAD-Distillation.md)
[![Built on HOSER](https://img.shields.io/badge/Built%20on-HOSER-orange)](https://github.com/caoji2001/HOSER)
[![Teacher: LM-TAD](https://img.shields.io/badge/Teacher-LM--TAD-purple)](https://github.com/jonathankabala/LMTAD)

</div>

---

### ⚠️ Work in Progress

This repository is under active development. Implementation details, APIs, and documentation are subject to change.

---

### Overview

This research investigates **knowledge distillation** from LM-TAD (85M parameters) into HOSER (6.7M parameters) for navigational trajectory generation, achieving **12.7× model compression** with **6× inference speedup**.

**Research Goal**: Compress a large language model teacher into a deployment-ready student while retaining trajectory generation quality.

⚠️ **Note**: This is a research fork. For original implementations:
- **HOSER**: [caoji2001/HOSER](https://github.com/caoji2001/HOSER)
- **LM-TAD**: [jonathankabala/LMTAD](https://github.com/jonathankabala/LMTAD)

---

### Key Results

| Metric | Result |
|--------|--------|
| **Model Size** | 85M → 6.7M params (12.7× compression) |
| **Inference Speed** | 6× faster (distilled vs vanilla) |
| **Teacher F1** | 83.89% (Beijing), 91.10% (Porto) |
| **Student Throughput** | 0.30 vs 0.05 traj/s (distilled vs vanilla) |

---

### Documentation

- 📘 **[Distillation Methodology](docs/LMTAD-Distillation.md)** - Complete technical guide
- 📊 **[Teacher Baseline](docs/results/TEACHER_BASELINE_COMPARISON.md)** - Performance metrics
- ✅ **[Vocabulary Mapping](docs/VOCABULARY_MAPPING_VALIDATION.md)** - Mapping validation
- 🔬 **[Evaluation Results](docs/EVALUATION_COMPARISON.md)** - Beijing & Porto analysis

---

### Reproducibility

**Hardware**: NVIDIA RTX 4090 (24GB), AMD Ryzen 9 7950X, 64GB DDR5  
**Software**: Python 3.12, PyTorch 2.5.1, CUDA 12.4

**Performance** (A* search):
- Distilled: 3.3s/trajectory (0.30 traj/s)
- Vanilla: 20s/trajectory (0.05 traj/s)
- **6× speedup** with distillation

---

### Citation & Acknowledgments

This work builds upon two excellent implementations:

#### HOSER (Student Architecture)

```bibtex
@inproceedings{cao2025hoser,
  title={Holistic Semantic Representation for Navigational Trajectory Generation},
  author={Cao, Ji and Zheng, Tongya and Guo, Qinghong and Wang, Yu and Dai, Junshu and Liu, Shunyu and Yang, Jie and Song, Jie and Song, Mingli},
  booktitle={AAAI},
  year={2025}
}
```

**Repository**: [caoji2001/HOSER](https://github.com/caoji2001/HOSER)  
**Paper**: [AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/31978) | [arXiv:2501.02737](https://arxiv.org/abs/2501.02737)

#### LM-TAD (Teacher Model)

```bibtex
@inproceedings{li2024lmtad,
  title={Trajectory Anomaly Detection with Language Models},
  author={Li, Boyang and others},
  booktitle={SIGSPATIAL},
  year={2024}
}
```

**Repository**: [jonathankabala/LMTAD](https://github.com/jonathankabala/LMTAD)

#### Knowledge Distillation References

- Hinton et al. (2015). Distilling the knowledge in a neural network. *arXiv:1503.02531*
- Sanh et al. (2019). DistilBERT, a distilled version of BERT. *arXiv:1910.01108*

---

### Acknowledgments

**Original Works**:
- HOSER team (Zhejiang University) for the student architecture and dataset
- LM-TAD team for the teacher model implementation

**Data**: Beijing taxi trajectory data from Bayou Tech (Hong Kong) Limited via HOSER dataset

---

### Contact

**Issues**: [GitHub Issues](https://github.com/matercomus/HOSER/issues)

For questions about original works:
- **HOSER**: [caoji2001/HOSER](https://github.com/caoji2001/HOSER)
- **LM-TAD**: [jonathankabala/LMTAD](https://github.com/jonathankabala/LMTAD)

---

### License

This research fork maintains compatibility with original HOSER and LM-TAD licenses. See respective repositories for details.
