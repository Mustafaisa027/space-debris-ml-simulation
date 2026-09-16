# Gorevin anlik durumu. mission_manager_node tarafindan /mission/state
# uzerinden TRANSIENT_LOCAL QoS ile yayinlanir; tum dugumler bunu dinleyip
# kendini o parkura gore aktif/pasif eder (gecis operator girisi olmadan olur).

# Durum makinesi (state machine) degerleri
uint8 STATE_BOOT=0        # MAVROS/GPS/sensor bekleniyor
uint8 STATE_READY=1       # gorev + hedef renk yuklu, START bekleniyor
uint8 STATE_PARKUR1=2     # nokta takip (WP1 -> WP4), kacis KAPALI
uint8 STATE_PARKUR2=3     # engelli nokta takip (WP4 -> WP5), kacis ACIK
uint8 STATE_PARKUR3=4     # kamikaze angajman (WP5 sonrasi)
uint8 STATE_DONE=5        # gorev tamam
uint8 STATE_EMERGENCY=6   # acil durdurma / kill

std_msgs/Header header

uint8 state               # yukaridaki STATE_* sabitlerinden biri
uint8 parkur              # 0=hazir, 1, 2, 3
int32 current_wp          # su an hedeflenen waypoint sirasi (0-index)
int32 last_reached_wp     # MAVROS'tan ulasildigi bildirilen son WP (0-index)

bool avoidance_active     # engelden kacinma besleme aktif mi
bool kamikaze_active      # kamikaze mantigi aktif mi
bool armed                # arac arm durumda mi
string flight_mode        # MAVROS ucus modu (AUTO, GUIDED, HOLD...)

string info               # insan-okur durum aciklamasi
