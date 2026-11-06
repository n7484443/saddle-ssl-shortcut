from PIL import Image
import numpy as np


def crop_and_resize(source_img, target_img):
    """
    Make source_img exactly the same as target_img by expanding/shrinking and
    cropping appropriately.

    If source_img's dimensions are strictly greater than or equal to the
    corresponding target img dimensions, we crop left/right or top/bottom
    depending on aspect ratio, then shrink down.

    If any of source img's dimensions are smaller than target img's dimensions,
    we expand the source img and then crop accordingly

    Modified from
    https://stackoverflow.com/questions/4744372/reducing-the-width-height-of-an-image-to-fit-a-given-aspect-ratio-how-python
    """
    source_width = source_img.size[0]
    source_height = source_img.size[1]

    target_width = target_img.size[0]
    target_height = target_img.size[1]

    # Check if source does not completely cover target
    if (source_width < target_width) or (source_height < target_height):
        # Try matching width
        width_resize = (target_width, int((target_width / source_width) * source_height))
        if (width_resize[0] >= target_width) and (width_resize[1] >= target_height):
            source_resized = source_img.resize(width_resize, Image.LANCZOS)
        else:
            height_resize = (int((target_height / source_height) * source_width), target_height)
            assert (height_resize[0] >= target_width) and (height_resize[1] >= target_height)
            source_resized = source_img.resize(height_resize, Image.LANCZOS)
        # Rerun the cropping
        return crop_and_resize(source_resized, target_img)

    source_aspect = source_width / source_height
    target_aspect = target_width / target_height

    if source_aspect > target_aspect:
        # Crop left/right
        new_source_width = int(target_aspect * source_height)
        offset = (source_width - new_source_width) // 2
        resize = (offset, 0, source_width - offset, source_height)
    else:
        # Crop top/bottom
        new_source_height = int(source_width / target_aspect)
        offset = (source_height - new_source_height) // 2
        resize = (0, offset, source_width, source_height - offset)

    source_resized = source_img.crop(resize).resize((target_width, target_height), Image.LANCZOS)
    return source_resized


def combine_and_mask(background, bird, bird_mask, original_ratio, ratio):
    """
    Combine a bird image with a background image, resizing the bird to meet the target ratio.
    Uses a pre-created bird mask to determine which pixels to overlay.

    Parameters:
    - background: numpy array of the background image (RGB)
    - bird: numpy array of the bird image (RGB)
    - bird_mask: numpy array boolean mask indicating bird pixels (True for bird, False for background)
    - original_ratio: float, original ratio of background_pixels / bird_pixels
    - ratio: float, target ratio of background_pixels / bird_pixels

    Returns:
    - combined_image: numpy array of the combined image with the bird centered
    - real_ratio: float, the actual ratio achieved after processing
    """
    import numpy as np
    import cv2

    # Get the dimensions
    bg_height, bg_width = background.shape[:2]
    bird_height, bird_width = bird.shape[:2]

    # Calculate background pixels
    bg_pixels = bg_height * bg_width

    # Calculate scaling factor
    pixel_pref_bird = bg_pixels * ratio
    pixel_current = bird_height * bird_width * original_ratio
    scaling_factor = np.sqrt(pixel_pref_bird / pixel_current)

    # Calculate new dimensions (maintaining aspect ratio)
    new_bird_height = int(bird_height * scaling_factor)
    new_bird_width = int(bird_width * scaling_factor)

    # Ensure new dimensions are at least 1 pixel
    new_bird_height = max(1, new_bird_height)
    new_bird_width = max(1, new_bird_width)

    # Resize the bird image and mask

    resized_bird = cv2.resize(bird, (new_bird_width, new_bird_height), interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(bird_mask.astype(np.uint8), (new_bird_width, new_bird_height), interpolation=cv2.INTER_NEAREST).astype(bool)

    # Calculate the position to place the bird in the center
    x_offset = max(0, (bg_width - new_bird_width) // 2)
    y_offset = max(0, (bg_height - new_bird_height) // 2)

    # Create a copy of the background
    result = background.copy()

    # Calculate the actual height and width to copy (to avoid going out of bounds)
    h = min(new_bird_height, bg_height - y_offset)
    w = min(new_bird_width, bg_width - x_offset)

    # Get the portion of the mask and bird we'll use
    mask_section = resized_mask[:h, :w]
    bird_section = resized_bird[:h, :w]

    # Get the region of interest in the background
    roi = result[y_offset:y_offset + h, x_offset:x_offset + w]

    # Only replace background pixels where the mask is True
    for c in range(3):
        roi[:, :, c] = np.where(mask_section, bird_section[:, :, c], roi[:, :, c])

    # Put the ROI back into the result
    result[y_offset:y_offset + h, x_offset:x_offset + w] = roi

    # Calculate the real ratio achieved (using mask for bird pixel count)
    bird_pixels = np.sum(resized_mask)
    real_ratio = max(1, bird_pixels) / bg_pixels   # Avoid division by zero

    return result, real_ratio