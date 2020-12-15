from pathlib import Path

miro_id = "V0002882.jpg"
image_url = (
    f"https://iiif.wellcomecollection.org/image/{miro_id}/"
    "full/224,224/0/default.jpg"
)
iiif_url = f"https://iiif.wellcomecollection.org/image/{miro_id}/info.json"
local_image_path = "file://" + str(
    (Path(__file__).parent / miro_id).absolute()
)

invalid_url = "https://example.com/pictures/123/skateboard.png"
