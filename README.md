# Mitigating Data Leakage in a WiFi CSI Benchmark for Human Action Recognition
**Abstract**: Human action recognition using WiFi channel state information (CSI) has gained attention due to its non-intrusive nature and potential applications in healthcare, smart environments, and security. However, the reliability of methods developed for CSI-based action recognition is often contingent on the quality of the datasets and evaluation protocols used. In this paper, we uncovered a critical data leakage issue, which arises from improper data partitioning, in a widely used WiFi CSI benchmark dataset. Specifically, the benchmark fails to separate individuals between the training and test sets, leading to inflated performance metrics as models inadvertently learn individual-specific features rather than generalizable action patterns. We analyzed this issue in depth, retrained several benchmarked models using corrected data partitioning methods, and demonstrated a significant drop in accuracy when individuals were properly separated across training and testing. Our findings highlight the importance of rigorous data partitioning in CSI-based action recognition and provide recommendations for mitigating data leakage in future research. This work contributes to the development of more robust and reliable human action recognition systems using WiFi CSI.<br/><br/>

## Cite
If you use this code, please cite the following paper:
```
@article{varga2024mitigating,
  title={Mitigating Data Leakage in a WiFi CSI Benchmark for Human Action Recognition},
  author={Varga, Domonkos},
  journal={Sensors},
  volume={24},
  number={24},
  pages={8201},
  year={2024},
  publisher={MDPI}
}
```
