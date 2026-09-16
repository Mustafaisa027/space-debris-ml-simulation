# Parkur-3 icin angaje olunacak hedef dubasinin renk bilgisi.
# Sartname 5.5.2.1: 3 hedef duba rengi RAL 9005 (siyah), RAL 3026 (kirmizi),
# RAL 6037 (yesil). Renk bilgisi IHA tarafindan tespit edilip veya gorev
# yuklemede elle girilip /iha/target_color uzerinden aktarilir.
# ONEMLI (sartname 5.5.3.1): hedef bilgisi arac HAREKETE BASLADIKTAN sonra
# aktarilamaz -> mission_manager sadece STATE_READY iken bunu kabul eder.

uint8 UNKNOWN=0
uint8 RED=1      # RAL 3026
uint8 GREEN=2    # RAL 6037
uint8 BLACK=3    # RAL 9005

std_msgs/Header header

uint8 color            # yukaridaki sabitlerden biri
string color_name      # "red" / "green" / "black"
string ral_code        # "RAL 3026" / "RAL 6037" / "RAL 9005"
float32 confidence     # 0.0 - 1.0 (IHA tespit guveni; elle girildiyse 1.0)
string source          # "iha" / "manual"
