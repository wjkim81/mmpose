import math
import numpy as np
import cv2

def polar_to_cart(polar, scale=0.5, rotate=True):
    polar_dtype = polar.dtype

    # Dealing with mask
    if polar_dtype == np.bool_:
        polar = polar.astype(np.uint8) * 255
        
    ws = int(2 * scale * polar.shape[1])
    dsize = (ws, ws)
    center = (ws // 2.0, ws // 2.0)
    max_radius = ws // 2
    flags = cv2.WARP_POLAR_LINEAR | cv2.WARP_INVERSE_MAP | cv2.WARP_FILL_OUTLIERS | cv2.INTER_CUBIC
    cart = cv2.rotate(polar, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cart = cv2.warpPolar(polar,
                          dsize=dsize,
                          center=center,
                          maxRadius=max_radius,
                          flags=flags
                          )
    cart = cv2.flip(cart, 0)

    if polar_dtype == np.bool_:
        cart = cart.astype(np.bool_)
    return cart

def point_polar_to_cart(point, img_sz, scale=0.5):
    # Don't be confused
    # point = (x, y) = (width, height)
    # img_sz = (height, width) = (n_beams, resolution)

    # The polar coordinate is rotated, n_beams != ims_sz[1]
    n_beams = img_sz[1]
    
    ws = int(2 * scale * img_sz[0])
    # dsize = (ws, ws)
    center = (ws // 2.0, ws // 2.0)
    # max_radius = ws // 2
    
    angle_step = 2 * math.pi / n_beams
    angle = point[0] * angle_step
    # print('angle:', angle)
    mx = math.cos(angle) * point[1] // 2
    my = math.sin(angle) * point[1] // 2

    tx = int(center[0] + mx)
    ty = int(center[1] + my)
    
    return (tx, ty)

def cart_to_polar(cart, n_beams):
    ws = cart.shape[0]
    cart = cv2.flip(cart, 0)
    
    dsize = (ws, n_beams)
    center = (ws // 2.0, ws // 2.0)
    max_radius = ws // 2.0
    flags = cv2.WARP_POLAR_LINEAR + cv2.WARP_FILL_OUTLIERS + cv2.INTER_CUBIC

    polar = cv2.warpPolar(cart,
                         dsize=dsize,
                         center=center,
                         maxRadius=max_radius, 
                         flags=flags
                         )
    
    polar = cv2.rotate(polar, cv2.ROTATE_90_CLOCKWISE)
    return polar

def point_cart_to_polar(point, cart_img_sz, n_beams):
    # Don't be confused
    # point = (x, y) = (width, height)
    # img_sz = (height, width) = (n_beams, resolution)

    # The cartesian coordinate is rotated, n_beams != ims_sz[1]
    center_x = cart_img_sz[1] // 2
    center_y = cart_img_sz[0] // 2

    scale = n_beams / center_x

    x = point[0] - center_x
    y = point[1] - center_y
    r = round(scale * math.sqrt(x**2 + y**2))
    # r = int(math.dist(x, y))

    angle_step = 2 * math.pi / n_beams

    degrees = math.atan2(y, x)
    # degrees = radians * 180 / math.pi
    angle = round(degrees / angle_step)
    if angle < 0:
        angle += n_beams

    return (angle, r)