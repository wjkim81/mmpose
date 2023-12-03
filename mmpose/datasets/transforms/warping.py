import cv2
import numpy as np
import mmcv
from mmcv.transforms import BaseTransform, TRANSFORMS

@TRANSFORMS.register_module()
class Warping(BaseTransform):
    def __init__(self, direction: str, n_beams: int, scale: float):
        super().__init__()
        self.direction = direction
        self.n_beams = n_beams
        self.scale = scale

    def transform(self, results: dict) -> dict:
        img = results['img']

        if self.direction == 'cart2polar':
            cart = img
            ws = cart.shape[0]
            cart = cv2.flip(cart, 0)
            
            dsize = (ws, self.n_beams)
            center = (ws // 2.0, ws // 2.0)
            max_radius = ws // 2.0
            flags = cv2.WARP_POLAR_LINEAR + cv2.WARP_FILL_OUTLIERS + cv2.INTER_CUBIC

            polar = cv2.warpPolar(cart,
                                dsize=dsize,
                                center=center,
                                maxRadius=max_radius, 
                                flags=flags
                                )
            
            img = cv2.rotate(cart, cv2.ROTATE_90_CLOCKWISE)
        else:
            polar_dtype = polar.dtype
            if polar_dtype == np.bool_:
                polar = polar.astype(np.uint8) * 255
        
            ws = int(2 * self.scale * polar.shape[1])
            dsize = (ws, ws)
            center = (ws // 2.0, ws // 2.0)
            max_radius = ws // 2
            flags = cv2.WARP_POLAR_LINEAR | cv2.WARP_INVERSE_MAP | cv2.WARP_FILL_OUTLIERS | cv2.INTER_CUBIC
            polar = cv2.rotate(polar, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cart = cv2.warpPolar(polar,
                                dsize=dsize,
                                center=center,
                                maxRadius=max_radius,
                                flags=flags
                                )
            cart = cv2.flip(cart, 0)

            if polar_dtype == np.bool_:
                cart = cart.astype(np.bool_)

            img = cart
                
        results['img'] = img
        return results