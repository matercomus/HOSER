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

This research investigates **knowledge distillation** from LM-TAD into HOSER for navigational trajectory generation.

**Research Goal**: Compress a large language model teacher into a deployment-ready student while retaining trajectory generation quality.

---

### Documentation

- 📘 **[Distillation Methodology](docs/LMTAD-Distillation.md)** - Complete technical guide
- 🏗️ **[Architecture Specification](docs/ARCHITECTURE_SPECIFICATION.md)** - Complete model architecture details
- 💾 **[Checkpoint Strategy](docs/CHECKPOINT_STRATEGY.md)** - Model saving and loading guide
- 📊 **[Teacher Baseline](docs/results/TEACHER_BASELINE_COMPARISON.md)** - Performance metrics
- ✅ **[Vocabulary Mapping](docs/VOCABULARY_MAPPING_VALIDATION.md)** - Mapping validation
- 🔍 **[Search Method Selection](docs/SEARCH_METHOD_GUIDANCE.md)** - A* vs Beam Search guidance
- 📈 **[Evaluation Comparison](docs/EVALUATION_COMPARISON.md)** - Cross-dataset analysis
- 📊 **[Paired Statistical Tests](docs/PAIRED_STATISTICAL_TESTS_GUIDE.md)** - Model comparison methodology

---

### Acknowledgments

This work builds upon open source implementations:
- **HOSER** (student architecture): [caoji2001/HOSER](https://github.com/caoji2001/HOSER)
- **LM-TAD** (teacher model): [jonathankabala/LMTAD](https://github.com/jonathankabala/LMTAD)

---

### Contact

**Issues**: [GitHub Issues](https://github.com/matercomus/HOSER/issues)

For questions about original works:
- **HOSER**: [caoji2001/HOSER](https://github.com/caoji2001/HOSER)
- **LM-TAD**: [jonathankabala/LMTAD](https://github.com/jonathankabala/LMTAD)

---

### License

This research fork maintains compatibility with original HOSER and LM-TAD licenses. See respective repositories for details.
