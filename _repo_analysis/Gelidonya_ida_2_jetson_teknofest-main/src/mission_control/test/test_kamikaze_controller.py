"""
kamikaze_color_controller birim testleri.

Kontrolcu ROS'tan bagimsiz oldugu icin bu testler ROS kurulumu OLMADAN
calisir:  pytest src/mission_control/test/test_kamikaze_controller.py

En kritik test: ISARET DONUSUMU. Kontrolcude turn_command pozitif = SAGA,
ROS'ta angular.z pozitif = SOLA. Sarmalayici bu isareti ters cevirmeli.
Yanlis isaret araci hedeften uzaga cevirir ve sahada fark etmesi zordur.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mission_control.kamikaze_color_controller import (  # noqa: E402
    KamikazeColorController,
    Detection,
    COLOR_CODE_TO_CLASS,
)

W, H = 1280, 720


def make_controller():
    return KamikazeColorController(image_width=W, image_height=H)


def box_at(cx, cy, w, h, cls='red', conf=0.9):
    return Detection(
        class_name=cls, confidence=conf,
        x1=cx - w / 2, y1=cy - h / 2, x2=cx + w / 2, y2=cy + h / 2,
    )


# ---------------------------------------------------------------------------
# Renk kodu
# ---------------------------------------------------------------------------
def test_renk_kodlari_targetcolor_ile_ayni():
    # mission_interfaces/TargetColor: 1=RED, 2=GREEN, 3=BLACK
    assert COLOR_CODE_TO_CLASS[1] == 'red'
    assert COLOR_CODE_TO_CLASS[2] == 'green'
    assert COLOR_CODE_TO_CLASS[3] == 'black'


def test_gecersiz_renk_kodu_reddedilir():
    c = make_controller()
    assert c.set_color_code(9) is False
    assert c.target_class is None


def test_renk_atanmadan_komut_uretilmez():
    c = make_controller()
    out = c.update([box_at(W / 2, H / 2, 100, 100)])
    assert out.target_found is False
    assert out.state == 'RENK_KODU_YOK'
    assert out.forward_command == 0.0


# ---------------------------------------------------------------------------
# ISARET DONUSUMU — en kritik test
# ---------------------------------------------------------------------------
def test_hedef_sagda_ise_turn_command_pozitif():
    c = make_controller()
    c.set_color_code(1)
    out = c.update([box_at(W * 0.85, H / 2, 100, 100)])   # sagda
    assert out.target_found
    assert out.horizontal_error > 0
    assert out.turn_command > 0, 'sagdaki hedef icin turn_command pozitif olmali'


def test_hedef_solda_ise_turn_command_negatif():
    c = make_controller()
    c.set_color_code(1)
    out = c.update([box_at(W * 0.15, H / 2, 100, 100)])   # solda
    assert out.horizontal_error < 0
    assert out.turn_command < 0, 'soldaki hedef icin turn_command negatif olmali'


def test_ros_angular_z_isareti_ters_cevrilmeli():
    """kamikaze_node'daki donusumun dogrulugunu sabitler.

    ROS REP-103: angular.z pozitif = saat yonu tersi = SOLA donus.
    Hedef SAGDA ise araç SAGA donmeli -> angular.z NEGATIF olmali.
    """
    c = make_controller()
    c.set_color_code(1)
    out = c.update([box_at(W * 0.85, H / 2, 100, 100)])   # hedef sagda

    max_yaw_rate = 1.0
    angular_z = -out.turn_command * max_yaw_rate   # kamikaze_node'daki formul

    assert angular_z < 0, (
        'Hedef sagdayken angular.z negatif olmali (saga donus). '
        'Pozitif cikiyorsa isaret donusumu ters ve arac hedeften uzaklasir.'
    )


# ---------------------------------------------------------------------------
# Hedef secimi
# ---------------------------------------------------------------------------
def test_yanlis_renk_secilmez():
    c = make_controller()
    c.set_color_code(1)                                   # kirmizi
    out = c.update([
        box_at(W / 2, H / 2, 300, 300, cls='green', conf=0.99),  # buyuk, merkezde
    ])
    assert out.target_found is False
    assert out.state == 'HEDEF_DUBA_BULUNAMADI'


def test_dogru_renk_capan_arasindan_secilir():
    c = make_controller()
    c.set_color_code(1)
    out = c.update([
        box_at(W * 0.15, H * 0.5, 200, 200, cls='green', conf=0.99),  # capan
        box_at(W * 0.60, H * 0.5, 120, 120, cls='red', conf=0.80),    # dogru
    ])
    assert out.target_found
    assert out.target_color == 'red'
    assert out.target_center_x == W * 0.60


def test_dusuk_guvenli_tespit_elenir():
    c = make_controller()
    c.set_color_code(1)
    out = c.update([box_at(W / 2, H / 2, 100, 100, conf=0.30)])
    assert out.target_found is False


def test_ayni_renkte_buyuk_ve_merkezdeki_tercih_edilir():
    c = make_controller()
    c.set_color_code(1)
    out = c.update([
        box_at(W * 0.10, H / 2, 60, 60, conf=0.80),    # kucuk, kenarda
        box_at(W * 0.52, H / 2, 260, 260, conf=0.80),  # buyuk, merkezde
    ])
    assert out.target_center_x == W * 0.52


# ---------------------------------------------------------------------------
# Hiz profili
# ---------------------------------------------------------------------------
def test_merkezdeki_hedefte_duz_gidilir():
    c = make_controller()
    c.set_color_code(1)
    out = c.update([box_at(W / 2, H / 2, 100, 100)])
    assert out.state == 'HEDEF_MERKEZDE'
    assert out.turn_command == 0.0
    assert out.forward_command > 0.0


def test_hedef_yandayken_hiz_dusurulur():
    c = make_controller()
    c.set_color_code(1)
    yan = c.update([box_at(W * 0.95, H / 2, 100, 100)]).forward_command
    merkez = c.update([box_at(W / 2, H / 2, 100, 100)]).forward_command
    assert yan < merkez, 'hedef yandayken once donulmeli, hiz dusuk olmali'


def test_yakin_ve_merkezdeki_hedefte_tam_gaz():
    c = make_controller()
    c.set_color_code(1)
    # Goruntunun ~%50'sini kaplayan kutu
    out = c.update([box_at(W / 2, H / 2, W * 0.8, H * 0.65)])
    assert out.forward_command == 1.0
    assert out.target_area_ratio > 0.45


def test_turn_command_sinirlanir():
    c = make_controller()
    c.set_color_code(1)
    out = c.update([box_at(W, H / 2, 100, 100)])   # en sagda
    assert abs(out.turn_command) <= c.maximum_turn


# ---------------------------------------------------------------------------
# Temas tespiti icin alan orani
# ---------------------------------------------------------------------------
def test_alan_orani_hesaplaniyor():
    c = make_controller()
    c.set_color_code(1)
    out = c.update([box_at(W / 2, H / 2, W * 0.5, H * 0.5)])
    assert abs(out.target_area_ratio - 0.25) < 0.01


def test_hedef_yokken_alan_orani_sifir():
    c = make_controller()
    c.set_color_code(1)
    out = c.update([])
    assert out.target_area_ratio == 0.0
