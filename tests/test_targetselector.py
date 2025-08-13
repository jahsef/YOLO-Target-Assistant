import pytest
import numpy as np
import math
from src.aimbot.data_parsing.targetselector import TargetSelector

# Mock configuration for testing
MOCK_CFG = {
    "targeting_settings": {
        "gravity": 196.2,
        "target_real_height": 5,
        "target_real_width": 3.5,
        "distance_calibration_factor": 0.45,
        "target_cls_id": 0,
        "crosshair_cls_id": 2,
        "head_toggle": True,
        "predict_drop": True,
        "predict_crosshair": True,
        "zoom": 1.5,
        "projectile_velocity": 2300,
        "base_head_offset": 0.35,
        "fov": 80
    },
    "sensitivity_settings": {
        "max_deltas": 160
    }
}

@pytest.fixture
def default_target_selector():
    return TargetSelector(
        cfg=MOCK_CFG,
        detection_window_dim=(640, 480),
        screen_hw=(1080, 1920),
        head_toggle=MOCK_CFG["targeting_settings"]["head_toggle"],
        base_head_offset=MOCK_CFG["targeting_settings"]["base_head_offset"],
        target_cls_id=MOCK_CFG["targeting_settings"]["target_cls_id"],
        crosshair_cls_id=MOCK_CFG["targeting_settings"]["crosshair_cls_id"],
        max_deltas=MOCK_CFG["sensitivity_settings"]["max_deltas"],
        predict_drop=MOCK_CFG["targeting_settings"]["predict_drop"],
        predict_crosshair=MOCK_CFG["targeting_settings"]["predict_crosshair"],
        zoom=MOCK_CFG["targeting_settings"]["zoom"],
        projectile_velocity=MOCK_CFG["targeting_settings"]["projectile_velocity"],
        hFOV_degrees=MOCK_CFG["targeting_settings"]["fov"]
    )

# Test cases for _calculate_distance
def test_calculate_distance_height_only(default_target_selector):
    # Example values, adjust based on expected behavior
    ts = default_target_selector
    distance = ts._calculate_distance(target_height_pixels=100, target_real_height=5)
    assert isinstance(distance, float)
    assert distance > 0

def test_calculate_distance_width_only(default_target_selector):
    ts = default_target_selector
    distance = ts._calculate_distance(target_width_pixels=100, target_real_width=3.5)
    assert isinstance(distance, float)
    assert distance > 0

def test_calculate_distance_both(default_target_selector):
    ts = default_target_selector
    distance = ts._calculate_distance(target_height_pixels=100, target_real_height=5, target_width_pixels=70, target_real_width=3.5)
    assert isinstance(distance, float)
    assert distance > 0

# Test cases for _calculate_bullet_drop
def test_calculate_bullet_drop(default_target_selector):
    ts = default_target_selector
    time_of_flight = 0.5  # seconds
    drop = ts._calculate_bullet_drop(time_of_flight)
    expected_drop = 0.5 * ts.GRAVITY * (time_of_flight ** 2)
    assert drop == pytest.approx(expected_drop)

# Test cases for _convert_to_screen_drop
def test_convert_to_screen_drop(default_target_selector):
    ts = default_target_selector
    real_drop = 1.0  # meters
    distance = 10.0  # meters
    screen_drop = ts._convert_to_screen_drop(real_drop, distance)
    
    pixels_per_radian = ts.screen_height / ts.vfov_rad
    angular_drop_rad = real_drop / distance
    expected_screen_drop = angular_drop_rad * pixels_per_radian
    assert screen_drop == pytest.approx(expected_screen_drop)

# Test cases for _get_closest_detection
def test_get_closest_detection(default_target_selector):
    ts = default_target_selector
    # Detections: [x1, y1, x2, y2, track_id, confidence, class_id, strack_idx]
    detections = np.array([
        [100, 100, 120, 120, 1, 0.9, 0, 1],
        [300, 300, 320, 320, 2, 0.8, 0, 2],
        [10, 10, 30, 30, 3, 0.7, 0, 3]
    ])
    reference_point = (15, 15) # Closest to the third detection
    closest = ts._get_closest_detection(detections, reference_point)
    assert np.array_equal(closest, detections[2])

    reference_point = (110, 110) # Closest to the first detection
    closest = ts._get_closest_detection(detections, reference_point)
    assert np.array_equal(closest, detections[0])

# Test cases for _get_deltas
def test_get_deltas_within_max(default_target_selector):
    ts = default_target_selector
    detection_xy = (330, 250)
    crosshair_xy = (320, 240)
    deltas = ts._get_deltas(detection_xy, crosshair_xy)
    assert deltas == (10, 10)

def test_get_deltas_exceeds_max(default_target_selector):
    ts = default_target_selector
    detection_xy = (500, 500)
    crosshair_xy = (320, 240)
    deltas = ts._get_deltas(detection_xy, crosshair_xy)
    assert deltas == (0, 0) # Should return (0,0) if exceeds max_deltas

# Test cases for _get_crosshair
def test_get_crosshair_no_prediction(default_target_selector):
    ts = default_target_selector
    ts.predict_crosshair = False # Temporarily disable prediction
    detections = np.array([
        [100, 100, 120, 120, 1, 0.9, 0, 1],
        [300, 300, 320, 320, 2, 0.8, 2, 2] # Crosshair detection
    ])
    crosshair = ts._get_crosshair(detections)
    assert crosshair == ts.detection_window_center # Should be center if prediction is off

def test_get_crosshair_with_prediction(default_target_selector):
    ts = default_target_selector
    ts.predict_crosshair = True # Ensure prediction is on
    detections = np.array([
        [100, 100, 120, 120, 1, 0.9, 0, 1],
        [300, 300, 320, 320, 2, 0.8, 2, 2] # Crosshair detection
    ])
    # Expected crosshair is center of the crosshair detection
    expected_crosshair_x = (300 + 320) // 2
    expected_crosshair_y = (300 + 320) // 2
    crosshair = ts._get_crosshair(detections)
    assert crosshair == (expected_crosshair_x, expected_crosshair_y)

# Test cases for get_deltas (main public method)
def test_get_deltas_no_enemy(default_target_selector):
    ts = default_target_selector
    detections = np.array([
        [100, 100, 120, 120, 1, 0.9, MOCK_CFG["targeting_settings"]["target_cls_id"] + 1, 1], # Friendly (assuming friendly is target_cls_id + 1)
        [300, 300, 320, 320, 2, 0.8, MOCK_CFG["targeting_settings"]["crosshair_cls_id"], 2]  # Crosshair
    ])
    deltas = ts.get_deltas(detections)
    assert deltas == (0, 0)

def test_get_deltas_with_enemy_no_prediction(default_target_selector):
    ts = default_target_selector
    ts.predict_drop = False
    ts.head_toggle = False
    ts.predict_crosshair = False # Explicitly disable crosshair prediction for this test
    detections = np.array([
        [330, 250, 350, 270, 1, 0.9, MOCK_CFG["targeting_settings"]["target_cls_id"], 1], # Enemy
        [300, 300, 320, 320, 2, 0.8, MOCK_CFG["targeting_settings"]["crosshair_cls_id"], 2]  # Crosshair
    ])
    # Detection window center is (320, 240) for a (640, 480) window
    # Enemy center is (340, 260)
    # If no prediction, crosshair is (320, 240)
    # Target is (340, 260)
    # Deltas should be (340-320, 260-240) = (20, 20)
    deltas = ts.get_deltas(detections)
    assert deltas == (20, 20)

def test_get_deltas_with_enemy_and_head_toggle(default_target_selector):
    ts = default_target_selector
    ts.predict_drop = False
    ts.head_toggle = True
    ts.predict_crosshair = False # Explicitly disable crosshair prediction for this test
    detections = np.array([
        [330, 250, 350, 270, 1, 0.9, MOCK_CFG["targeting_settings"]["target_cls_id"], 1], # Enemy (w=20, h=20)
        [300, 300, 320, 320, 2, 0.8, MOCK_CFG["targeting_settings"]["crosshair_cls_id"], 2]  # Crosshair
    ])
    # Enemy center (340, 260)
    # Head offset: 20 * 0.35 = 7
    # Aim Y: 260 - 7 = 253
    # Crosshair (320, 240)
    # Deltas: (340-320, 253-240) = (20, 13)
    deltas = ts.get_deltas(detections)
    assert deltas == (20, 13)

def test_get_deltas_with_enemy_and_drop_prediction(default_target_selector):
    ts = default_target_selector
    ts.predict_drop = True
    ts.head_toggle = False
    ts.predict_crosshair = False # Explicitly disable crosshair prediction for this test
    # Create a detection that will result in a calculable drop
    # This test is more about ensuring the drop calculation is *called* and affects the y-delta
    # rather than asserting a precise numerical value without knowing the exact distance calculation.
    detections = np.array([
        [330, 250, 350, 270, 1, 0.9, MOCK_CFG["targeting_settings"]["target_cls_id"], 1], # Enemy (w=20, h=20)
        [300, 300, 320, 320, 2, 0.8, MOCK_CFG["targeting_settings"]["crosshair_cls_id"], 2]  # Crosshair
    ])
    # Expected: y-delta should be less than 20 due to drop prediction
    deltas = ts.get_deltas(detections)
    assert deltas[0] == 20 # X-delta should be unchanged
    assert deltas[1] < 20 # Y-delta should be reduced due to drop
    assert deltas[1] > 0 # Should still be positive if target is below crosshair
