from pathlib import Path

image_id = "V0002882.jpg"
image_url = (
    f"https://iiif.wellcomecollection.org/image/{image_id}/"
    "full/224,224/0/default.jpg"
)
iiif_url = f"https://iiif.wellcomecollection.org/image/{image_id}/info.json"
local_image_path = "file://" + str(
    (Path(__file__).parent / image_id).absolute()
)

invalid_url = "https://example.com/pictures/123/skateboard.png"
