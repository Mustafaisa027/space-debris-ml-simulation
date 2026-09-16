# kamikaze_node tarafindan Parkur-3 sirasinda /kamikaze/status uzerinde yayinlanir.
# mission_manager bunu izleyip angajman tamamlaninca STATE_DONE'a gecer.

std_msgs/Header header

bool target_locked         # dogru renkte hedef goruntude kilitlendi mi
uint8 target_color         # TargetColor sabitleri (angaje olunan renk)

float32 bearing_deg        # hedefe goreli kerteriz (+sol, -sag), derece
float32 image_center_error # hedef merkezinin goruntu merkezinden pixel sapmasi
float32 distance_m         # kaba mesafe tahmini (-1 = bilinmiyor)

uint8 PHASE_SEARCH=0       # hedef araniyor
uint8 PHASE_APPROACH=1     # hedefe yaklasiliyor / guidance veriliyor
uint8 PHASE_TERMINAL=2     # son hucum (tam gaz)
uint8 PHASE_IMPACT=3       # temas / angajman tamam
uint8 phase

bool engaged               # fiziksel temas / angajman gerceklesti
string info
