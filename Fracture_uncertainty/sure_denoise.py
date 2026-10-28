import sys
import numpy as np
from PIL import Image
from scipy.fft import dctn, idctn
 
min_sig = 1.0 / 255.0 / np.sqrt(12.0)
 
 
def softthresh(c, tau):
    return np.sign(c) * np.maximum(np.abs(c) - tau, 0)
 
def sure_curve(c, sigma, taus):
    """Per-pixel SURE for soft thresholding at every tau in taus (c = noisy DCT coefficients)."""
    a = np.sort(np.asarray(c, dtype=np.float64).ravel() ** 2)
    d = a.size
    cs = np.concatenate([[0.0], np.cumsum(a)])
    t2 = np.asarray(taus, dtype=np.float64) ** 2
    k = np.searchsorted(a, t2, side='right')       # coefficients with c^2 <= tau^2
    mse_to_observed = (cs[k] + t2 * (d - k)) / d   # mean((f(c) - c)^2)
    divergence = d - k                             # coefficients that survive
    return mse_to_observed - sigma ** 2 + 2.0 * sigma ** 2 * divergence / d
 
 
def estimate_sigma(c):
    """Noise level from the highest-frequency quadrant of the DCT ."""
    h, w = c.shape
    band = c[h // 2:, w // 2:]
    return float(np.median(np.abs(band)) / 0.6745)
 
 
def sure_denoise_array(arr, min_sigma=min_sig, n_taus=200):
    """arr: 2-D float array in [0, 1]. returns (denoised array, tau, sigma_hat)."""
    c = dctn(arr, norm='ortho')
    sigma = max(estimate_sigma(c), min_sigma)
    taus = np.linspace(0.0, sigma * np.sqrt(2.0 * np.log(c.size)), n_taus)
    tau = float(taus[int(np.argmin(sure_curve(c, sigma, taus)))])
    return idctn(softthresh(c, tau), norm='ortho'), tau, sigma
 
 
class SUREDenoise:
    """torchvision-style transform: PIL image in, SURE-denoised PIL image (mode 'L') out."""
 
    def __call__(self, img):
        arr = np.asarray(img.convert('L'), dtype=np.float64) / 255.0
        x_hat, _, _ = sure_denoise_array(arr)
        return Image.fromarray(np.rint(np.clip(x_hat, 0, 1) * 255).astype(np.uint8))
 
 
 
if __name__ == '__main__':
    for path in sys.argv[1:]:
        arr = np.asarray(Image.open(path).convert('L'), dtype=np.float64) / 255.0
        x_hat, tau, sigma = sure_denoise_array(arr)
        kept = np.mean(np.abs(dctn(arr, norm='ortho')) > tau)
        print(f"{path}\n  size={arr.shape}  sigma_hat={sigma:.5f}  tau={tau:.5f}  "
              f"kept {100 * kept:.1f}% of coefficients  "
              f"mean change={255 * np.mean(np.abs(x_hat - arr)):.3f} gray levels", mode='L')
 