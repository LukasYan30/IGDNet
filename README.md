<h1 align="center">[IEEE TAI] IGDNet: Zero-Shot Robust Underexposed Image Enhancement via Illumination-Guided and Denoising</h1>

<div align="center">
  <hr>
  Hailong Yan<sup>1</sup>&nbsp;
  Junjian Huang<sup>2,†</sup>&nbsp;
  Tingwen Huang, Fellow, IEEE<sup>3</sup>&nbsp;
  <br>
  <sup>1</sup> UESTC&nbsp;&nbsp; <sup>2</sup> SWU&nbsp;&nbsp; <sup>3</sup> SUAT-SZ<br>
  <sup>†</sup> Corresponding authors.<br>

  <h4>
    <a href="https://www.arxiv.org/pdf/2507.02445">📄 arXiv Paper</a> &nbsp; 
  </h4>
</div>

<blockquote>
<b>Abstract:</b> <i>Current methods for restoring underexposed images typically rely on supervised learning with paired underexposed and well-illuminated images. However, collecting such datasets is often impractical in real-world scenarios. Moreover, these methods can lead to over-enhancement, distorting well-illuminated regions. To address these issues, we propose IGDNet, a Zero-Shot enhancement method that operates solely on a single test image, without requiring guiding priors or training data. IGDNet exhibits strong generalization ability and effectively suppresses noise while restoring illumination. The framework comprises a decomposition module and a denoising module. The former separates the image into illumination and reflection components via a dense connection network, while the latter enhances non-uniformly illuminated regions using an illumination-guided pixel adaptive correction method. A noise pair is generated through downsampling and refined iteratively to produce the final result. Extensive experiments on four public datasets demonstrate that IGDNet significantly improves visual quality under complex lighting conditions. Quantitative results on metrics like PSNR (20.41dB) and SSIM (0.860dB) show that it outperforms 14 state-of-the-art unsupervised methods.</i>
</blockquote>

---

### Demo

```bash
python main.py
```

### Citation

If you find the code helpful in your research or work, please cite the following paper:

```
@article{yan2025igdnet,
    title={IGDNet: Zero-Shot Robust Underexposed Image Enhancement via Illumination-Guided and Denoising},
    author={Yan, Hailong and Huang, Junjian, Huang, Tingwen},
    journal={IEEE Transactions on Artificial Intelligence},
    year={2025},
    publisher={IEEE}
}
```
