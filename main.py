import numpy as np
import matplotlib.pyplot as plt
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Data.pixel_grid import PixelGrid
from lenstronomy.Data.psf import PSF  # <-- 1. ADD THIS IMPORT
from lenstronomy.ImSim.image_model import ImageModel


def simulate_first_lens():
    print("🌌 Starting Strong Lens Simulation...")

    # 1. Define the Coordinate Grid (Our Camera Sensor)
    num_pix = 100
    delta_pix = 0.05

    kwargs_grid = {
        'nx': num_pix, 'ny': num_pix,
        'transform_pix2angle': np.array([[delta_pix, 0], [0, delta_pix]]),
        'ra_at_xy_0': -(num_pix * delta_pix) / 2.0,
        'dec_at_xy_0': -(num_pix * delta_pix) / 2.0
    }
    pixel_grid = PixelGrid(**kwargs_grid)

    # 2. Define an Empty PSF object (Fixes the AttributeError)
    psf_class = PSF(psf_type='NONE')  # <-- 2. INITIALIZE CORECTLY HERE

    # 3. Define the Lens Model (The Foreground Mass / Galaxy)
    lens_model_list = ['SIS']
    lens_model = LensModel(lens_model_list=lens_model_list)
    kwargs_lens = [{'theta_E': 1.2, 'center_x': 0.0, 'center_y': 0.0}]

    # 4. Define the Source Light Model (The Background Galaxy)
    source_model_list = ['SERSIC']
    source_light_model = LightModel(light_model_list=source_model_list)
    kwargs_source = [{'amp': 100, 'R_sersic': 0.2, 'n_sersic': 1, 'center_x': 0.1, 'center_y': 0.1}]

    # 5. Combine them into an Image Model simulator
    # We pass our new psf_class instance here instead of None
    image_model = ImageModel(data_class=pixel_grid, psf_class=psf_class,
                             lens_model_class=lens_model, source_model_class=source_light_model)

    # 6. Generate the simulated lens image
    image = image_model.image(kwargs_lens=kwargs_lens, kwargs_source=kwargs_source)

    # 7. Add some realistic cosmic background noise
    noise = np.random.normal(loc=0, scale=1.0, size=image.shape)
    image_with_noise = image + noise

    # 8. Plot the Result!
    plt.figure(figsize=(6, 6))
    plt.imshow(image_with_noise, origin='lower', cmap='magma')
    plt.title("Our First Simulated Gravitational Lens")
    plt.colorbar(label="Intensity")

    plt.savefig('figures/my_first_lens.png')
    print("✅ Success! Image saved to figures/my_first_lens.png")
    plt.show()


if __name__ == "__main__":
    simulate_first_lens()