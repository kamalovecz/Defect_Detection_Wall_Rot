# Environment Notes

The final remote validation environment is named `harpnet_acceptance`.

Original baseline environment observed on 2026-07-14:

- Python: 3.10.19
- torch: 2.1.0+cu121
- torchvision: 0.16.0+cu121
- numpy: 1.26.4
- PyYAML: 6.0.3
- OpenCV: 4.11.0
- onnx: 1.20.1
- onnxruntime: 1.23.2

`environment.yml` is the acceptance environment specification. It pins the
Python, PyTorch, CUDA, ONNX, and ONNX Runtime compatibility line and installs
the bundled Ultralytics runtime from this repository. It is not a byte-for-byte
package lockfile.

Create and validate a clean environment from the repository root:

```powershell
conda env create -n harpnet_acceptance -f environment.yml
conda run -n harpnet_acceptance python training_project/verify_all.py
```

The clean acceptance environment created on 2026-07-18 resolved to Python
3.10.20, PyTorch 2.1.2, CUDA 12.1, Ultralytics 8.2.50, ONNX 1.20.1, and ONNX
Runtime 1.23.2. CUDA was available on the RTX 4090 host.
