from skinburnsegmentation.data_preprocessing.transforms import (
    get_image_transform,
    get_label_transform
)
from skinburnsegmentation.constants import CLASS_NAME_ID_MAP


def test_get_image_transform_gets_train_transform():
    input_size = (256, 256)
    is_train = True

    image_transform = get_image_transform(input_size, is_train)

    assert image_transform.transforms[0].__class__.__name__ == "Resize"
    assert image_transform.transforms[1].__class__.__name__ == "RandomHorizontalFlip"
    assert image_transform.transforms[2].__class__.__name__ == "RandomRotation"
    assert image_transform.transforms[3].__class__.__name__ == "ToTensor"
    assert image_transform.transforms[4].__class__.__name__ == "Normalize"

    assert image_transform.transforms[0].size == input_size

def test_get_image_transform_gets_not_train_transform():
    input_size = (256, 256)
    is_train = False

    image_transform = get_image_transform(input_size, is_train)

    assert image_transform.transforms[0].__class__.__name__ == "Resize"
    assert image_transform.transforms[1].__class__.__name__ == "ToTensor"
    assert image_transform.transforms[2].__class__.__name__ == "Normalize"

    assert image_transform.transforms[0].size == input_size

def test_label_transform():
    label_transform = get_label_transform()
    for key, value in CLASS_NAME_ID_MAP.items():
        assert label_transform(key) == value
