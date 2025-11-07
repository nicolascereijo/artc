# artc.core
## Metrics and their comparisons
### Comparisons

- <span style="color:#7B68EE">**Pearson correlation**</span>

![\color{white}PC_{X,Y}=\frac{{\sum_{i=1}^{n}(X_i-\bar{X})(Y_i-\bar{Y})}}{{\sqrt{\sum_{i=1}^{n}(X_i-\bar{X})^2}\cdot&space;\sqrt{\sum_{i=1}^{n}(Y_i-\bar{Y})^2}}}](https://latex.codecogs.com/svg.image?\color{white}PC_{X,Y}=\frac{{\sum_{i=1}^{n}(X_i-\bar{X})(Y_i-\bar{Y})}}{{\sqrt{\sum_{i=1}^{n}(X_i-\bar{X})^2}\cdot&space;\sqrt{\sum_{i=1}^{n}(Y_i-\bar{Y})^2}}})

- <span style="color:#559B2F">**Cosine Similarity**</span>

![\color{white}CS_{X,Y}=\frac{X\cdot&space;Y}{\left\|X\right\|\cdot&space;\left\|Y\right\|}](https://latex.codecogs.com/svg.image?\color{white}CS_{X,Y}=\frac{X\cdot&space;Y}{\left\|X\right\|\cdot&space;\left\|Y\right\|})

- <span style="color:#FF0082">**Normalized relative difference**</span>

![\color{white}NRD_{X,Y}=1-\frac{{|X-Y|}}{{\max(X,Y)}}](https://latex.codecogs.com/svg.image?\color{white}NRD_{X,Y}=1-\frac{{|X-Y|}}{{\max(X,Y)}})

- <span style="color:#1682B4">**Normalized Euclidean distance**</span>

![\color{white}NED_{X,Y}=\frac{\sqrt{\sum_{i=1}^{n}(X_{i}-Y_{i})^2}}{\sqrt[]{n}}](https://latex.codecogs.com/svg.image?\color{white}NED_{X,Y}=\frac{\sqrt{\sum_{i=1}^{n}(X_{i}-Y_{i})^2}}{\sqrt[]{n}})

### Metrics
- beat_alignment ⟶ <span style="color:#FF0082">Normalized relative difference</span>
- chroma ⟶ <span style="color:#7B68EE">Pearson correlation</span>
- dynamic_time_warping ⟶ <span style="color:#7B68EE">Pearson correlation</span>
- energy_envelope ⟶ <span style="color:#7B68EE">Pearson correlation</span>
- harmonic_noise_ratio ⟶ <span style="color:#FF0082">Normalized relative difference</span>
- loudness ⟶ <span style="color:#1682B4">Normalized euclidean distance</span>
- mfcc (mel-frequency cepstral coefficients) ⟶ <span style="color:#559B2F">Cosine Similarity</span>
- onset_detection ⟶ <span style="color:#FF0082">Normalized relative difference</span>
- peak_matching ⟶ <span style="color:#FF0082">Normalized relative difference</span>
- pitch ⟶ <span style="color:#FF0082">Normalized relative difference</span>
- rhythm ⟶ <span style="color:#FF0082">Normalized relative difference</span>
- spectral_bandwidth ⟶ <span style="color:#1682B4">Normalized Euclidean distance</span>
- spectral_centroid ⟶ <span style="color:#1682B4">Normalized euclidean distance</span>
- spectral_contrast ⟶ <span style="color:#7B68EE">Pearson correlation</span>
- spectral_flatness ⟶ <span style="color:#FF0082">Normalized relative difference</span>
- spectral_roll_off ⟶ <span style="color:#FF0082">Normalized relative difference</span>
- spectrogram ⟶ <span style="color:#559B2F">Cosine Similarity</span>
- temporal_centroid ⟶ <span style="color:#FF0082">Normalized relative difference</span>
- temporal_flux ⟶ <span style="color:#FF0082">Normalized relative difference</span>
- zero_crossing_rate ⟶ <span style="color:#FF0082">Normalized relative difference</span>
